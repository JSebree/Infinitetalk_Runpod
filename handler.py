
import runpod
from runpod.serverless.utils import rp_upload  # kept for potential future upload usage
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii  # Base64 error handling
import subprocess
import time
import librosa
import traceback
import boto3
import string
import tempfile
import math

# ---------------------------
# Attention backend safety toggles (to avoid SM90 SageAttention crashes)
# ---------------------------
# Default to disabling SageAttention in WanVideoWrapper unless explicitly overridden.
# This prevents "SM90 kernel is not available" assertion crashes on H100 when the
# installed sageattention wheel doesn't include sm90 kernels for the current CUDA/Torch.
os.environ.setdefault("WAN_DISABLE_SAGEATTN", os.getenv("WAN_DISABLE_SAGEATTN", "1"))
# Prefer a safe SDPA backend; if FlashAttention2 is present in the image it will be used,
# otherwise fall back to math which is universal (slower but stable).
os.environ.setdefault("PYTORCH_SDP_BACKEND", os.getenv("PYTORCH_SDP_BACKEND", "math"))
# Avoid some inductor fused SDPA edge cases on mixed stacks.
os.environ.setdefault("TORCHINDUCTOR_DISABLE_FUSED_SDP", os.getenv("TORCHINDUCTOR_DISABLE_FUSED_SDP", "1"))

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Global config
# ---------------------------
server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())

# ---------------------------
# DigitalOcean Spaces client
# ---------------------------
_SPACES_ENDPOINT = os.environ["SPACES_ENDPOINT_URL"]
_REGION          = os.environ.get("SPACES_REGION", "")
_BUCKET          = os.environ["SPACES_BUCKET_NAME"]
_S3 = boto3.client(
    "s3",
    region_name=_REGION,
    endpoint_url=_SPACES_ENDPOINT,
    aws_access_key_id=os.environ["SPACES_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["SPACES_SECRET_ACCESS_KEY"]
)

# ---------------------------
# Output naming helpers
# ---------------------------
_SAFE_CHARS = set(string.ascii_letters + string.digits + "-_.")

def _sanitize_filename(name: str, default: str) -> str:
    """Keep basename and restrict to safe chars."""
    if not name:
        return default
    name = os.path.basename(name.strip())
    if not name:
        return default
    safe = "".join(c if c in _SAFE_CHARS else "-" for c in name)
    return safe

def _derive_output_name(job_input: dict, task_id: str, src_path: str) -> str:
    """
    Choose output file name. If user provided a name, keep it; otherwise use task_id.
    Preserve the extension from src_path (gif/mp4).
    """
    user_nm = (job_input.get("output_name")
               or job_input.get("output_basename")
               or job_input.get("output_filename"))
    base = user_nm or f"{task_id}"
    _, ext = os.path.splitext(src_path)
    if not ext:
        ext = ".gif"
    name = f"{base}{ext}"
    return _sanitize_filename(name, f"{task_id}{ext}")



def _run_ffmpeg(cmd_args):
    """Run ffmpeg with the provided list of args; raise on failure."""
    try:
        result = subprocess.run(cmd_args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {' '.join(cmd_args)}\nSTDERR:\n{result.stderr}")
        return result
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found in PATH; please install it in the container/image.")


# ---------------------------
# Face/audio branch helpers
# ---------------------------
def _normalize_branch(label: str) -> str:
    """
    Normalize various face labels into canonical branch IDs: A, B, C, D.
    Accepts aliases like left/right/top/bottom or face_a..face_d; falls back to 'A'.
    """
    if not label:
        return "A"
    s = str(label).strip().lower()
    if s in ("left", "l", "a", "face_a"):
        return "A"
    if s in ("right", "r", "b", "face_b"):
        return "B"
    if s in ("up", "top", "t", "c", "face_c"):
        return "C"
    if s in ("down", "bottom", "d", "face_d"):
        return "D"
    if s in ("a", "b", "c", "d"):
        return s.upper()
    return "A"

def _get_audio_node_map(input_type: str, prompt: dict) -> dict:
    """
    Return a mapping of branch IDs (A-D) to ComfyUI audio node IDs present in the loaded workflow.
    Defaults:
      - Primary branch A -> node '125'
      - Secondary branch B -> node '307' (I2V) or '313' (V2V) when present
    Can be extended via env ITALK_AUDIO_NODE_MAP_IMAGE / ITALK_AUDIO_NODE_MAP_VIDEO (JSON dict).
    Only returns entries for node IDs that actually exist in `prompt`.
    """
    node_map = {"A": "125"}
    if input_type == "image":
        if "307" in prompt:
            node_map["B"] = "307"
    else:
        if "313" in prompt:
            node_map["B"] = "313"
    env_key = "ITALK_AUDIO_NODE_MAP_IMAGE" if input_type == "image" else "ITALK_AUDIO_NODE_MAP_VIDEO"
    try:
        extra = json.loads(os.environ.get(env_key, "{}"))
        if isinstance(extra, dict):
            for k, v in extra.items():
                kN = _normalize_branch(k)
                if str(v) in prompt:
                    node_map[kN] = str(v)
    except Exception:
        pass
    return node_map


# ---------------------------
# Helpers: downloads / IO
def download_file_from_url(url, output_path):
    """Download a file from a URL to output_path using wget."""
    try:
        result = subprocess.run(
            ['wget', '-O', output_path, '--no-verbose', '--timeout=30', url],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            logger.info(f"✅ Downloaded: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget failed: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ Download timed out")
        raise Exception("Download timed out")
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        raise Exception(f"Download error: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Decode base64 data and save to file. Returns absolute path."""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"✅ Saved base64 input to '{file_path}'")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 decode failed: {e}")
        raise Exception(f"Base64 decode failed: {e}")



def process_input(input_data, temp_dir, output_filename, input_type):
    """
    Normalize input into a file path.
    input_type in {'path','url','base64'}.
    """
    if input_type == "path":
        logger.info(f"📁 Using path input: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"🌐 Using URL input: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info(f"🔢 Using base64 input.")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input_type: {input_type}")


# ---------------------------
# Segment/stem utilities
# ---------------------------
def _segments_total_duration(segments: list) -> float:
    """Return total timeline duration derived from the largest segment end time."""
    total = 0.0
    for seg in segments or []:
        try:
            total = max(total, float(seg.get("end", 0.0)))
        except Exception:
            continue
    return total

def build_stems_from_segments(mix_wav_path: str, segments: list, speakers: list, out_dir: str, sample_rate: int = 24000) -> dict:
    """
    Build full-length stems for each speaker from a single multi-speaker WAV using FFmpeg.
    Each stem has the same total duration as the mix; outside that speaker's segments it is silence.
    Returns { speaker_id: stem_path }.
    """
    os.makedirs(out_dir, exist_ok=True)
    total_dur = _segments_total_duration(segments)
    if total_dur <= 0:
        raise RuntimeError("Segments missing or invalid; total duration is 0.")

    stems = {}
    # For each speaker, collect segments and synthesize a stem
    for spk in speakers:
        spk_segs = [s for s in segments if str(s.get("voice_id")) == str(spk)]
        out_path = os.path.join(out_dir, f"stem_{spk}.wav")
        if not spk_segs:
            # Pure silence when speaker has no segments
            _run_ffmpeg([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-t", f"{total_dur:.6f}",
                "-i", f"anullsrc=r={sample_rate}:cl=mono",
                "-ac", "1", "-ar", str(sample_rate),
                out_path
            ])
            stems[spk] = out_path
            continue

        # Build filter_complex from all segments for this speaker:
        #   [0:a]atrim=start=st:end=en,asetpts=PTS-STARTPTS,adelay=st_ms|st_ms[s0]; ... ; [s0][s1]...amix=inputs=N[out]
        fc_parts = []
        out_labels = []
        for idx, seg in enumerate(spk_segs):
            st = float(seg.get("start", 0.0))
            en = float(seg.get("end", 0.0))
            if en <= st:
                continue
            lbl = f"s{idx}"
            delay_ms = int(round(st * 1000.0))
            fc_parts.append(f"[0:a]atrim=start={st:.6f}:end={en:.6f},asetpts=PTS-STARTPTS,adelay={delay_ms}|{delay_ms}[{lbl}]")
            out_labels.append(f"[{lbl}]")
        if not out_labels:
            _run_ffmpeg([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-t", f"{total_dur:.6f}",
                "-i", f"anullsrc=r={sample_rate}:cl=mono",
                "-ac", "1", "-ar", str(sample_rate),
                out_path
            ])
            stems[spk] = out_path
            continue

        amix_in = "".join(out_labels)
        inputs_n = len(out_labels)
        fc = ";".join(fc_parts) + f";{amix_in}amix=inputs={inputs_n}:dropout_transition=0:normalize=0,aformat=sample_fmts=s16:channel_layouts=mono:sample_rates={sample_rate}[out]"
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", mix_wav_path,
            "-filter_complex", fc,
            "-map", "[out]",
            "-ac", "1", "-ar", str(sample_rate),
            out_path
        ])
        stems[spk] = out_path

    return stems


# ---------------------------
# ComfyUI interaction
# ---------------------------
def queue_prompt(prompt, input_type="image", person_count="single"):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')

    # Debugging hints about key nodes your workflows expect
    logger.info(f"Workflow node count: {len(prompt)}")
    if input_type == "image":
        logger.info(f"Image node(284): {prompt.get('284', {}).get('inputs', {}).get('image', 'NOT_SET')}")
    else:
        logger.info(f"Video node(228): {prompt.get('228', {}).get('inputs', {}).get('video', 'NOT_SET')}")
    logger.info(f"Audio node(125): {prompt.get('125', {}).get('inputs', {}).get('audio', 'NOT_SET')}")
    logger.info(f"Text node(241): {prompt.get('241', {}).get('inputs', {}).get('positive_prompt', 'NOT_SET')}")
    if person_count == "multi":
        if "307" in prompt:
            logger.info(f"Second audio node(307): {prompt.get('307', {}).get('inputs', {}).get('audio', 'NOT_SET')}")
        elif "313" in prompt:
            logger.info(f"Second audio node(313): {prompt.get('313', {}).get('inputs', {}).get('audio', 'NOT_SET')}")
        # Log optional extra audio nodes if present (e.g., C/D branches)
        for extra_node in ("401", "402", "403", "404"):
            if extra_node in prompt:
                logger.info(
                    f"Extra audio node({extra_node}): "
                    f"{prompt.get(extra_node, {}).get('inputs', {}).get('audio', 'NOT_SET')}"
                )

    req = urllib.request.Request(url, data=data)
    req.add_header('Content-Type', 'application/json')

    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        logger.info(f"Prompt queued successfully: {result}")
        return result
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode('utf-8')
        except Exception:
            pass
        logger.error(f"HTTP error: {e.code} - {e.reason} | body={body}")
        raise
    except Exception as e:
        logger.error(f"Error queueing prompt: {e}")
        raise


def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def get_videos(ws, prompt, input_type="image", person_count="single"):
    """
    Runs the workflow, waits for completion, and returns (videos_by_node_b64, files_by_node, prompt_id).
    Each entry in files_by_node is a list of absolute file paths written by ComfyUI.
    """
    queued = queue_prompt(prompt, input_type, person_count)
    prompt_id = queued.get('prompt_id')
    videos_b64 = {}
    files_by_node = {}

    # Wait for execution to complete
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message.get('type') == 'executing':
                data = message.get('data', {})
                if data.get('node') is None and data.get('prompt_id') == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id, node_output in history.get('outputs', {}).items():
        b64_list, file_list = [], []
        if 'gifs' in node_output:
            for video in node_output['gifs']:
                # capture file path
                fp = video.get('fullpath')
                if fp and os.path.exists(fp):
                    file_list.append(fp)
                    with open(fp, 'rb') as f:
                        b64_list.append(base64.b64encode(f.read()).decode('utf-8'))
        if 'videos' in node_output:
            for video in node_output['videos']:
                fp = video.get('fullpath')
                if fp and os.path.exists(fp):
                    file_list.append(fp)
                    with open(fp, 'rb') as f:
                        b64_list.append(base64.b64encode(f.read()).decode('utf-8'))

        if b64_list:
            videos_b64[node_id] = b64_list
        if file_list:
            files_by_node[node_id] = file_list

    return videos_b64, files_by_node, prompt_id


def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)


def get_workflow_path(input_type, person_count):
    """Return workflow path based on input_type and person_count."""
    if input_type == "image":
        return "/I2V_single.json" if person_count == "single" else "/I2V_multi.json"
    else:
        return "/V2V_single.json" if person_count == "single" else "/V2V_multi.json"


# ---------------------------
# Audio duration → max_frames
# ---------------------------
def get_audio_duration(audio_path):
    """Return audio duration (seconds) or None on failure."""
    try:
        return librosa.get_duration(path=audio_path)
    except Exception as e:
        logger.warning(f"Failed to get audio duration ({audio_path}): {e}")
        return None


def calculate_max_frames_from_audio(wav_path, wav_path_2=None, fps=25):
    """
    Calculate max_frames from one or two audio tracks. Adds a tail of 81 frames.
    """
    durations = []
    d1 = get_audio_duration(wav_path)
    if d1 is not None:
        durations.append(d1)
        logger.info(f"Primary audio duration: {d1:.2f}s")
    if wav_path_2:
        d2 = get_audio_duration(wav_path_2)
        if d2 is not None:
            durations.append(d2)
            logger.info(f"Secondary audio duration: {d2:.2f}s")

    if not durations:
        logger.warning("Could not determine audio duration(s). Using default 81 frames.")
        return 81

    max_duration = max(durations)
    max_frames = int(max_duration * fps) + 81
    logger.info(f"Longest audio: {max_duration:.2f}s → max_frames={max_frames}")
    return max_frames


# ---------------------------
# Response envelope builders
# ---------------------------
def success_envelope(task_id, model, latency_ms, gpu_name, workflow_path, prompt_id,
                     video_base64=None, video_format="gif", video_url=None, warnings=None, extra_debug=None):
    return {
        "id": task_id,
        "status": "succeeded",
        "model": model,
        "metrics": {
            "latency_ms": int(latency_ms),
            "gpu": gpu_name or "unknown"
        },
        "artifacts": {
            "video_base64": video_base64,
            "format": video_format,
            "video_url": video_url
        },
        "warnings": warnings or [],
        "debug": {
            "workflow": workflow_path,
            "prompt_id": prompt_id,
            **(extra_debug or {})
        }
    }


def error_envelope(task_id, code, message, details=None, trace=None):
    out = {
        "id": task_id,
        "status": "failed",
        "error": {
            "code": code,
            "message": message
        }
    }
    if details is not None:
        out["error"]["details"] = details
    if trace is not None:
        out["error"]["trace"] = trace
    return out


# ---------------------------
# Main handler
# ---------------------------
def handler(job):
    """
    Expected input (examples):
    {
      "input_type": "image" | "video",
      "person_count": "single" | "multi",
      "image_path" | "image_url" | "image_base64": "...",
      "video_path" | "video_url" | "video_base64": "...",
      "wav_path" | "wav_url" | "wav_base64": "...",
      "wav_path_2" | "wav_url_2" | "wav_base64_2": "...",  # only for multi
      "prompt": "A person talking naturally",
      "width": 512,
      "height": 512,
      "max_frame": 300
    }
    """
    t0 = time.time()
    job_input = job.get("input", {}) or {}
    task_id = f"task_{uuid.uuid4()}"
    gpu_name = os.environ.get("RUNPOD_GPU", "unknown")
    model_name = os.environ.get("MODEL_NAME", "InfiniteTalk/ComfyUI")

    logger.info(f"Received job input: {job_input}")
    logger.info(f"Task ID: {task_id}")

    try:
        # Input type and speaker count
        input_type = job_input.get("input_type", "image")          # "image" or "video"
        person_count = job_input.get("person_count", "single")     # "single" or "multi"
        logger.info(f"Workflow type: {input_type}, persons: {person_count}")

        # Option B: single multi-speaker WAV + segments → stems (state holders)
        use_mix = False
        mix_wav_path = None
        segments = job_input.get("segments")
        # Accept either 'face_map' or legacy 'speaker_map' from caller
        face_map = job_input.get("face_map") or job_input.get("speaker_map")
        if isinstance(face_map, dict):
            face_map = { str(k): _normalize_branch(v) for k, v in face_map.items() }
        else:
            face_map = None

        # Workflow path
        workflow_path = get_workflow_path(input_type, person_count)
        logger.info(f"Using workflow: {workflow_path}")

        # Optional per-request overrides for attention backends
        disable_sage = str(job_input.get("disable_sageattention", os.environ.get("WAN_DISABLE_SAGEATTN", "1"))).lower() in ("1","true","yes")
        sdp_backend = job_input.get("sdp_backend", os.environ.get("PYTORCH_SDP_BACKEND", "math"))

        # Apply overrides for this process (ComfyUI reads env at import; this still documents intent
        # and helps when ComfyUI is launched from the same process/image with these envs)
        os.environ["WAN_DISABLE_SAGEATTN"] = "1" if disable_sage else "0"
        os.environ["PYTORCH_SDP_BACKEND"] = sdp_backend

        logger.info(f"Attention config → WAN_DISABLE_SAGEATTN={os.environ['WAN_DISABLE_SAGEATTN']}, PYTORCH_SDP_BACKEND={os.environ['PYTORCH_SDP_BACKEND']}")

        # Detect single multi-speaker WAV input (Option B)
        mix_keys = [k for k in ("multi_speaker_wav_path","multi_speaker_wav_url","multi_speaker_wav_base64") if k in job_input]
        if mix_keys:
            k = mix_keys[0]
            if k.endswith("_path"):
                mix_wav_path = process_input(job_input[k], task_id, "mix_audio.wav", "path")
            elif k.endswith("_url"):
                mix_wav_path = process_input(job_input[k], task_id, "mix_audio.wav", "url")
            else:
                mix_wav_path = process_input(job_input[k], task_id, "mix_audio.wav", "base64")
            use_mix = True
            logger.info(f"Detected multi-speaker WAV via '{k}' → {mix_wav_path}")

        # Media input
        if input_type == "image":
            if "image_path" in job_input:
                media_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
            elif "image_url" in job_input:
                media_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
            elif "image_base64" in job_input:
                media_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
            else:
                media_path = "/examples/image.jpg"
                logger.info("Using default image: /examples/image.jpg")
        else:
            if "video_path" in job_input:
                media_path = process_input(job_input["video_path"], task_id, "input_video.mp4", "path")
            elif "video_url" in job_input:
                media_path = process_input(job_input["video_url"], task_id, "input_video.mp4", "url")
            elif "video_base64" in job_input:
                media_path = process_input(job_input["video_base64"], task_id, "input_video.mp4", "base64")
            else:
                media_path = "/examples/video.mp4"
                logger.info("Using default video as fallback: /examples/video.mp4")

        # Audio input handling (Option B: mix+segments → stems; else legacy explicit wavs)
        stems_by_branch = {}
        if use_mix:
            if not os.path.exists(mix_wav_path):
                return error_envelope(task_id, "missing_audio", f"Multi-speaker audio file not found: {mix_wav_path}")
            if not isinstance(segments, list) or not segments:
                return error_envelope(task_id, "missing_segments", "When using multi_speaker_wav_*, you must supply 'segments': [{start,end,voice_id}, ...].")

            # Identify unique speakers (voice_ids) in timeline order of first appearance
            speakers = []
            seen = set()
            for s in segments:
                vid = str(s.get("voice_id"))
                if vid and vid not in seen:
                    speakers.append(vid); seen.add(vid)

            # Build stems per speaker
            temp_audio_dir = os.path.abspath(os.path.join(task_id, "stems"))
            stems = build_stems_from_segments(
                mix_wav_path,
                segments,
                speakers,
                temp_audio_dir,
                sample_rate=int(job_input.get("sample_rate", 24000))
            )

            # Probe workflow to understand which audio branches exist
            prompt_probe = load_workflow(get_workflow_path(input_type, "multi" if len(speakers) > 1 else "single"))
            node_map = _get_audio_node_map(input_type, prompt_probe)  # e.g., {"A":"125","B":"307", "C":"401", "D":"402"}
            available_branches = [b for b in ("A","B","C","D") if b in node_map]

            # Map stems to branches using provided face_map, or deterministic default
            if face_map:
                # Validate branch labels
                for sid, br in face_map.items():
                    if br not in available_branches:
                        return error_envelope(task_id, "invalid_face_map", f"Branch '{br}' not available in this workflow. Available: {available_branches}")
                # Assign
                for sid, br in face_map.items():
                    if sid in stems:
                        stems_by_branch[br] = stems[sid]
            else:
                for idx, sid in enumerate(speakers):
                    if idx < len(available_branches):
                        stems_by_branch[available_branches[idx]] = stems.get(sid)

            # Fill any remaining branches with pure silence of total duration
            total_dur = _segments_total_duration(segments)
            sample_rate = int(job_input.get("sample_rate", 24000))
            for br in available_branches:
                if br not in stems_by_branch or not stems_by_branch[br]:
                    silent_out = os.path.abspath(os.path.join(temp_audio_dir, f"stem_silence_{br}.wav"))
                    _run_ffmpeg([
                        "ffmpeg","-y",
                        "-f","lavfi",
                        "-t",f"{total_dur:.6f}",
                        "-i",f"anullsrc=r={sample_rate}:cl=mono",
                        "-ac","1","-ar",str(sample_rate),
                        silent_out
                    ])
                    stems_by_branch[br] = silent_out

            # Choose A/B for legacy downstream variables; C/D kept in stems_by_branch for later assignment
            wav_path = stems_by_branch.get("A") or list(stems_by_branch.values())[0]
            wav_path_2 = stems_by_branch.get("B") or wav_path
        else:
            # Legacy explicit WAVs
            if "wav_path" in job_input:
                wav_path = process_input(job_input["wav_path"], task_id, "input_audio.wav", "path")
            elif "wav_url" in job_input:
                wav_path = process_input(job_input["wav_url"], task_id, "input_audio.wav", "url")
            elif "wav_base64" in job_input:
                wav_path = process_input(job_input["wav_base64"], task_id, "input_audio.wav", "base64")
            else:
                wav_path = "/examples/audio.mp3"
                logger.info("Using default audio: /examples/audio.mp3")

            wav_path_2 = None
            if person_count == "multi":
                if "wav_path_2" in job_input:
                    wav_path_2 = process_input(job_input["wav_path_2"], task_id, "input_audio_2.wav", "path")
                elif "wav_url_2" in job_input:
                    wav_path_2 = process_input(job_input["wav_url_2"], task_id, "input_audio_2.wav", "url")
                elif "wav_base64_2" in job_input:
                    wav_path_2 = process_input(job_input["wav_base64_2"], task_id, "input_audio_2.wav", "base64")
                else:
                    wav_path_2 = wav_path
                    logger.info("Second audio not provided → reusing first audio")

        # Basic fields
        prompt_text = job_input.get("prompt", "A person talking naturally")
        width = int(job_input.get("width", 512))
        height = int(job_input.get("height", 512))

        # max_frame: derive from audio or segments if not provided
        max_frame = job_input.get("max_frame")
        if max_frame is None:
            if use_mix and isinstance(segments, list) and segments:
                longest = _segments_total_duration(segments)
                max_frame = int(longest * 25) + 81
                logger.info(f"No max_frame provided → using segments duration {longest:.2f}s → max_frames={max_frame}")
            else:
                logger.info("No max_frame provided → deriving from audio duration")
                max_frame = calculate_max_frames_from_audio(wav_path, wav_path_2 if person_count == "multi" else None)
        else:
            logger.info(f"Using user-provided max_frame: {max_frame}")

        # Existence checks
        if not os.path.exists(media_path):
            return error_envelope(task_id, "missing_media", f"Media file not found: {media_path}")
        if use_mix:
            if not os.path.exists(wav_path):
                return error_envelope(task_id, "missing_audio", f"Stem A not found: {wav_path}")
            if person_count == "multi" and wav_path_2 and not os.path.exists(wav_path_2):
                return error_envelope(task_id, "missing_audio_2", f"Stem B not found: {wav_path_2}")
        else:
            if not os.path.exists(wav_path):
                return error_envelope(task_id, "missing_audio", f"Audio file not found: {wav_path}")
            if person_count == "multi" and wav_path_2 and not os.path.exists(wav_path_2):
                return error_envelope(task_id, "missing_audio_2", f"Second audio file not found: {wav_path_2}")

        logger.info(f"Media size: {os.path.getsize(media_path)} bytes")
        logger.info(f"Audio size: {os.path.getsize(wav_path)} bytes")
        if person_count == "multi" and wav_path_2:
            logger.info(f"Second audio size: {os.path.getsize(wav_path_2)} bytes")

        # ---------------------------
        # Configure workflow nodes
        # ---------------------------
        prompt = load_workflow(workflow_path)

        if input_type == "image":
            if "284" in prompt:
                prompt["284"]["inputs"]["image"] = media_path
            else:
                logger.warning("Workflow missing expected image node (284).")
        else:
            if "228" in prompt:
                prompt["228"]["inputs"]["video"] = media_path
            else:
                logger.warning("Workflow missing expected video node (228).")

        # Common params (guarded)
        if "125" in prompt:
            prompt["125"]["inputs"]["audio"] = wav_path
        else:
            logger.warning("Workflow missing primary audio node (125).")
        if "241" in prompt:
            prompt["241"]["inputs"]["positive_prompt"] = prompt_text
        else:
            logger.warning("Workflow missing text prompt node (241).")
        if "245" in prompt:
            prompt["245"]["inputs"]["value"] = width
        else:
            logger.warning("Workflow missing width node (245).")
        if "246" in prompt:
            prompt["246"]["inputs"]["value"] = height
        else:
            logger.warning("Workflow missing height node (246).")
        if "270" in prompt:
            prompt["270"]["inputs"]["value"] = max_frame
        else:
            logger.warning("Workflow missing max_frame node (270).")

        # Multi-person: assign audio stems to available nodes dynamically (A-D)
        if person_count == "multi":
            node_map = _get_audio_node_map(input_type, prompt)  # {"A":"125","B":"307"/"313", ...}
            # A and B are set using wav_path / wav_path_2
            if "A" in node_map and node_map["A"] in prompt:
                prompt[node_map["A"]]["inputs"]["audio"] = wav_path
            if "B" in node_map and node_map["B"] in prompt:
                prompt[node_map["B"]]["inputs"]["audio"] = wav_path_2 or wav_path
            # If we are in mix mode and produced stems_by_branch, also assign C/D if present
            if use_mix and 'stems_by_branch' in locals():
                for br in ("C","D"):
                    node_id = node_map.get(br)
                    if node_id and node_id in prompt:
                        stem_path = stems_by_branch.get(br)
                        if stem_path:
                            prompt[node_id]["inputs"]["audio"] = stem_path

        # ---------------------------
        # Connect to ComfyUI (HTTP health + WS)
        # ---------------------------
        ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
        http_url = f"http://{server_address}:8188/"
        logger.info(f"Checking HTTP connection to: {http_url}")

        max_http_attempts = 180  # up to 3 minutes
        for attempt in range(max_http_attempts):
            try:
                urllib.request.urlopen(http_url, timeout=5)
                logger.info(f"HTTP OK (attempt {attempt+1})")
                break
            except Exception as e:
                logger.warning(f"HTTP failed (attempt {attempt+1}/{max_http_attempts}): {e}")
                if attempt == max_http_attempts - 1:
                    return error_envelope(task_id, "comfyui_unreachable",
                                          "Cannot connect to ComfyUI HTTP endpoint. Is the server running?")
                time.sleep(1)

        ws = websocket.WebSocket()
        max_ws_attempts = int(180/5)  # 3 minutes
        for attempt in range(max_ws_attempts):
            try:
                ws.connect(ws_url)
                logger.info(f"WebSocket connected (attempt {attempt+1})")
                break
            except Exception as e:
                logger.warning(f"WebSocket connect failed (attempt {attempt+1}/{max_ws_attempts}): {e}")
                if attempt == max_ws_attempts - 1:
                    return error_envelope(task_id, "websocket_timeout", "WebSocket connect timed out (3 minutes)")
                time.sleep(5)

        try:
            videos_by_node, files_by_node, prompt_id = get_videos(ws, prompt, input_type, person_count)
        finally:
            try:
                ws.close()
            except Exception:
                pass

        # Prefer file path for upload; fall back to base64 if needed
        first_file_path = None
        first_video_b64 = None
        for node_id, paths in files_by_node.items():
            if paths:
                first_file_path = paths[0]
                break
        if not first_file_path:
            for node_id, vids in videos_by_node.items():
                if vids:
                    first_video_b64 = vids[0]
                    break

        if first_file_path and os.path.exists(first_file_path):
            # Derive output name and folder; upload to DO Spaces as public-read
            out_name = _derive_output_name(job_input, task_id, first_file_path)
            out_key  = f"infinitetalk_out/{out_name}"
            _S3.upload_file(
                Filename=first_file_path,
                Bucket=_BUCKET,
                Key=out_key,
                ExtraArgs={"ACL": "public-read"}
            )
            public_url = f"{_SPACES_ENDPOINT}/{_BUCKET}/{out_key}"

            latency_ms = (time.time() - t0) * 1000.0
            return success_envelope(
                task_id=task_id,
                model=model_name,
                latency_ms=latency_ms,
                gpu_name=gpu_name,
                workflow_path=workflow_path,
                prompt_id=prompt_id,
                video_base64=None,
                video_format=os.path.splitext(first_file_path)[1].lstrip(".").lower() or "gif",
                video_url=public_url,
                warnings=[],
                extra_debug={
                    "input_type": input_type,
                    "person_count": person_count,
                    "width": width,
                    "height": height,
                    "max_frame": max_frame
                }
            )

        if first_video_b64:
            # Fallback path: still return base64 if file path not captured
            latency_ms = (time.time() - t0) * 1000.0
            return success_envelope(
                task_id=task_id,
                model=model_name,
                latency_ms=latency_ms,
                gpu_name=gpu_name,
                workflow_path=workflow_path,
                prompt_id=prompt_id,
                video_base64=first_video_b64,
                video_format="gif",
                video_url=None,
                warnings=["returned_base64_fallback"],
                extra_debug={
                    "input_type": input_type,
                    "person_count": person_count,
                    "width": width,
                    "height": height,
                    "max_frame": max_frame
                }
            )

        # No video found
        return error_envelope(task_id, "no_video_found", "No video was produced by the workflow.")

    except Exception as e:
        # Structured error envelope with traceback and special-casing SageAttention SM90 assertion
        tb = traceback.format_exc()
        msg = str(e)
        if "SM90 kernel is not available" in msg or "sageattn_qk_int8_pv_fp8_cuda_sm90" in tb:
            hint = {
                "hint": "SageAttention failed to load SM90 kernels on H100. The fastest workaround is to disable SageAttention.",
                "what_to_do": {
                    "set_env": {
                        "WAN_DISABLE_SAGEATTN": "1",
                        "PYTORCH_SDP_BACKEND": os.environ.get("PYTORCH_SDP_BACKEND", "math"),
                        "TORCHINDUCTOR_DISABLE_FUSED_SDP": os.environ.get("TORCHINDUCTOR_DISABLE_FUSED_SDP", "1")
                    },
                    "or_per_request": {
                        "disable_sageattention": True,
                        "sdp_backend": os.environ.get("PYTORCH_SDP_BACKEND", "math")
                    },
                    "long_term": "Install a sageattention wheel built for your exact Torch/CUDA with sm90 support."
                }
            }
            return error_envelope(
                task_id=task_id,
                code="attention_backend_incompatible",
                message="SageAttention SM90 kernels are unavailable for this image (H100).",
                details=hint,
                trace=tb
            )
        return error_envelope(
            task_id=task_id,
            code="internal_error",
            message=msg,
            trace=tb
        )


# Wire the serverless handler
runpod.serverless.start({"handler": handler})
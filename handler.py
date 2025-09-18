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


# ---------------------------
# Helpers: downloads / IO
# ---------------------------
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

        # Workflow path
        workflow_path = get_workflow_path(input_type, person_count)
        logger.info(f"Using workflow: {workflow_path}")

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
                media_path = "/examples/image.jpg"
                logger.info("Using default image as fallback: /examples/image.jpg")

        # Audio input (required)
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

        # max_frame: derive from audio if not provided
        max_frame = job_input.get("max_frame")
        if max_frame is None:
            logger.info("No max_frame provided → deriving from audio duration")
            max_frame = calculate_max_frames_from_audio(wav_path, wav_path_2 if person_count == "multi" else None)
        else:
            logger.info(f"Using user-provided max_frame: {max_frame}")

        # Existence checks
        if not os.path.exists(media_path):
            return error_envelope(task_id, "missing_media", f"Media file not found: {media_path}")
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
            # I2V: set image input
            prompt["284"]["inputs"]["image"] = media_path
        else:
            # V2V: set video input
            prompt["228"]["inputs"]["video"] = media_path

        # Common params
        prompt["125"]["inputs"]["audio"] = wav_path
        prompt["241"]["inputs"]["positive_prompt"] = prompt_text
        prompt["245"]["inputs"]["value"] = width
        prompt["246"]["inputs"]["value"] = height
        prompt["270"]["inputs"]["value"] = max_frame

        # Multi-person: set second audio if present in workflow
        if person_count == "multi":
            if input_type == "image":
                if "307" in prompt:
                    prompt["307"]["inputs"]["audio"] = wav_path_2
            else:
                if "313" in prompt:
                    prompt["313"]["inputs"]["audio"] = wav_path_2

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
        # Structured error envelope with traceback
        return error_envelope(
            task_id=task_id,
            code="internal_error",
            message=str(e),
            trace=traceback.format_exc()
        )


# Wire the serverless handler
runpod.serverless.start({"handler": handler})
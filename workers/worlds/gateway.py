"""Anonymous, authenticated single-GPU gateway with cancellable chunk inference.

CPU contract tests use an explicit fixture adapter. Production never synthesizes
frames, reports readiness without checkpoints, or silently substitutes a model.
"""
from __future__ import annotations
import base64
import binascii
from dataclasses import dataclass, field
import hmac
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit
import uuid
import warnings

from media import validate_raw_frames
from streaming import StreamingError, StreamingService

MAX_BODY = 16 * 1024 * 1024
MAX_IMAGE = 10 * 1024 * 1024
MAX_PROMPT = 8000
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,80}$")


class ApiError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, *args, **kwargs):
        self._slots = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)
    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise
    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def image_bytes(value):
    if not isinstance(value, str):
        raise ApiError(400, "Images must be PNG, JPEG or WebP data URLs")
    match = re.fullmatch(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=\r\n]+)", value)
    if not match:
        raise ApiError(400, "Remote URLs and local paths are not accepted; supply an image data URL")
    try:
        data = base64.b64decode(match[2], validate=True)
    except (ValueError, binascii.Error):
        raise ApiError(400, "Invalid image encoding") from None
    if not data or len(data) > MAX_IMAGE:
        raise ApiError(413, "The image must be smaller than 10 MiB")
    signatures = {
        "png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        "jpeg": data.startswith(b"\xff\xd8\xff"),
        "webp": data.startswith(b"RIFF") and data[8:12] == b"WEBP",
    }
    if not signatures[match[1]]:
        raise ApiError(400, "Image content does not match its declared format")
    from PIL import Image
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > 16_000_000 or max(image.size) > 8192:
                    raise ApiError(413, "Reference images are limited to 16 megapixels and 8192 pixels per side")
                image.verify()
    except ApiError:
        raise
    except Exception:
        raise ApiError(400, "Reference image is corrupt or exceeds decoder safety limits") from None
    return data


def child_environment():
    # Inference sees cache/configuration, never control-plane/cloud credentials.
    allowed = ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PYTHONPATH", "HF_HOME", "TORCH_HOME", "TMPDIR")
    result = {key: os.environ[key] for key in allowed if key in os.environ}
    result.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", WANDB_MODE="disabled")
    return result


@dataclass
class Session:
    id: str
    prompt: str
    seed: int
    quality: str
    directory: Path
    image: Path | None = None
    action: str = "stop"
    action_expires_at: float = 0
    status: str = "loading"
    error: str | None = None
    frame: bytes | None = None
    raw_frame: bytes | None = None
    raw_size: tuple | None = None
    jpeg_index: int = -1
    model_id: str = "astronex-world"
    resident: bool = False
    audio: bytes | None = None
    audio_index: int = 0
    audio_started_at: float = 0
    audio_paused_at: float | None = None
    frame_index: int = 0
    chunks: int = 0
    generation_seconds: float | None = None
    generated_fps: float | None = None
    total_generated_frames: int = 0
    total_generation_seconds: float = 0
    model_load_seconds: float | None = None
    model_load_count: int = 0
    transformer_load_count: int | None = None
    prompt_encoding_count: int | None = None
    continuity: str | None = None
    residency: str | None = None
    rendered_window_count: int | None = None
    sampling_seconds: float | None = None
    prompt_truncated: bool = False
    peak_vram_bytes: int | None = None
    base_prompt: str | None = None
    prompt_amendments: list = field(default_factory=list)
    queued_revision: int = 0
    applied_revision: int = 0
    generating_revision: int | None = None
    buffered_chunks: int = 0
    last_used: float = field(default_factory=time.monotonic)
    stop: threading.Event = field(default_factory=threading.Event)
    resumed: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    process: subprocess.Popen | None = None
    thread: threading.Thread | None = None

    def current_action(self, now=None):
        with self.lock:
            return self.action if (time.monotonic() if now is None else now) < self.action_expires_at else "stop"

    def latest_jpeg(self):
        with self.lock:
            data, size, index = self.raw_frame, self.raw_size, self.frame_index
            if data is None or self.jpeg_index == index:
                return self.frame, index
        from PIL import Image
        output = io.BytesIO()
        Image.frombytes("RGB", size, data).save(output, "JPEG", quality=90)
        encoded = output.getvalue()
        with self.lock:
            if self.frame_index == index and not self.stop.is_set():
                self.frame, self.jpeg_index = encoded, index
        return encoded, index

    def public(self, capabilities):
        with self.lock:
            return {"id": self.id, "status": self.status, "error": self.error, "modelId": self.model_id,
                    "capabilities": capabilities, "seed": self.seed, "frameIndex": self.frame_index, "chunks": self.chunks,
                    "generationSeconds": self.generation_seconds, "generatedFPS": self.generated_fps,
                    "totalGeneratedFrames": self.total_generated_frames, "totalGenerationSeconds": self.total_generation_seconds,
                    "modelLoadSeconds": self.model_load_seconds, "modelLoadCount": self.model_load_count,
                    "transformerLoadCount": self.transformer_load_count, "promptEncodingCount": self.prompt_encoding_count,
                    "continuity": self.continuity, "residency": self.residency,
                    "renderedWindowCount": self.rendered_window_count, "samplingSeconds": self.sampling_seconds,
                    "promptTruncated": self.prompt_truncated,
                    "peakVRAMBytes": self.peak_vram_bytes,
                    "queuedRevision": self.queued_revision, "appliedRevision": self.applied_revision,
                    "generatingRevision": self.generating_revision, "bufferedChunks": self.buffered_chunks,
                    "interactionMode": capabilities.get("runtime", {}).get("interactionMode", "next-clip"),
                    "audioAvailable": self.audio is not None, "resumeKind": "visual-checkpoint"}


class Gateway:
    def __init__(self, adapter=None, root=None, idle_seconds=300):
        if adapter is None:
            from adapters import selected_adapter
            adapter = selected_adapter()
        self.adapter = adapter
        self.root = Path(root or tempfile.mkdtemp(prefix="world-worker-"))
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.idle_seconds = idle_seconds
        self.sessions = {}
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.streaming = StreamingService(self)
        self.reaper = threading.Thread(target=self._reap, daemon=True)
        self.reaper.start()

    def create(self, body):
        if body.get("modelId") != self.adapter.model_id:
            raise ApiError(422, "This worker does not implement the selected model")
        ready, reason = self.adapter.ready()
        if not ready:
            raise ApiError(503, reason)
        identifier = body.get("id") or str(uuid.uuid4())
        if not isinstance(identifier, str) or not SESSION_ID.fullmatch(identifier):
            raise ApiError(400, "Invalid anonymous session identifier")
        inputs = body.get("input", {})
        if not isinstance(inputs, dict):
            raise ApiError(400, "input must be an object")
        prompt = inputs.get("prompt", "")
        input_caps = self.adapter.capabilities.get("input", {})
        if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT or input_caps.get("text", True) and not prompt.strip():
            raise ApiError(400, "A supported prompt of at most 8000 characters is required")
        if not input_caps.get("text", True) and prompt.strip():
            raise ApiError(422, "This adapter is image conditioned and does not support text prompts")
        if any(inputs.get(key) for key in ("video", "audio", "characters", "multiImage", "characterDescription")):
            raise ApiError(422, "This adapter accepts text and one image only")
        images = inputs.get("images", [])
        if not isinstance(images, list) or len(images) > 1:
            raise ApiError(422, "This adapter accepts at most one reference image")
        if images and not input_caps.get("image", False):
            raise ApiError(422, "This adapter does not accept reference images")
        if input_caps.get("requiredImage") and not images:
            raise ApiError(422, "This adapter requires a reference image")
        image = image_bytes(images[0]) if images else None
        seed = body.get("seed")
        if seed is None:
            seed = secrets.randbits(32)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ApiError(400, "seed must be an unsigned 32-bit integer")
        resolutions = self.adapter.capabilities.get("runtime", {}).get("resolutionOptions", [])
        if body.get("resolution") is not None and body["resolution"] not in resolutions:
            raise ApiError(422, "Unsupported adapter resolution")
        quality = body.get("quality", "balanced")
        quality_options = self.adapter.capabilities.get("runtime", {}).get("qualityOptions", ("quality", "balanced", "low-latency", "speed"))
        if quality not in quality_options:
            raise ApiError(422, "Unsupported quality mode")
        with self.lock:
            if self.closed.is_set():
                raise ApiError(503, "Worker is shutting down")
            if identifier in self.sessions:
                raise ApiError(409, "Session already exists")
            if self.sessions:
                raise ApiError(409, "This single-GPU worker already has a session; close it first")
            directory = self.root / identifier
            directory.mkdir(mode=0o700)
            session = Session(identifier, prompt, seed, quality, directory, model_id=self.adapter.model_id)
            session.base_prompt = prompt
            if image:
                session.image = directory / "input-image"
                session.image.write_bytes(image)
            session.resumed.set()
            self.sessions[identifier] = session
            session.thread = threading.Thread(target=self._generate, args=(session,), daemon=True)
            session.thread.start()
        return session.public(self.adapter.capabilities)

    def get(self, identifier, active=False):
        with self.lock:
            session = self.sessions.get(identifier)
            if not session:
                raise ApiError(404, "Session not found")
            if active:
                session.last_used = time.monotonic()
            return session

    def action(self, identifier, body):
        session = self.get(identifier, active=True)
        with session.lock:
            if session.error:
                raise ApiError(409, "The session failed; close it and start a new session")
            kind = body.get("type")
            changed = True
            if kind == "native":
                action = body.get("action")
                native_actions = getattr(self.adapter, "capabilities", {}).get("nativeActions", [])
                if not isinstance(action, str) or action not in native_actions:
                    raise ApiError(422, "Unsupported native action")
                values = body.get("values", {})
                if not isinstance(values, dict):
                    raise ApiError(400, "Action values must be an object")
                if "planned" in values:
                    if not getattr(self.adapter, "offline_clip", False):
                        raise ApiError(422, "Planned controls are supported only by offline-clip adapters")
                    if not isinstance(values["planned"], bool):
                        raise ApiError(400, "Native planned state must be a boolean")
                if "pressed" in values and not isinstance(values["pressed"], bool):
                    raise ApiError(400, "Native pressed state must be a boolean")
                if hasattr(self.adapter, "apply_native"):
                    try:
                        changed = self.adapter.apply_native(session, action, values) is not False
                    except ValueError as error:
                        raise ApiError(400, str(error)) from None
                if values.get("pressed") is False:
                    if session.action == action:
                        session.action = "stop"
                        session.action_expires_at = 0
                else:
                    session.action = action
                    # Clients refresh held controls; a lost release cannot leave movement stuck.
                    session.action_expires_at = time.monotonic() + 2
            elif kind == "prompt":
                controls = getattr(self.adapter, "capabilities", {}).get("control", {})
                if not controls.get("promptSwitching") and not controls.get("promptDuringRollout"):
                    raise ApiError(422, "This adapter does not support prompt switching")
                prompt = body.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT:
                    raise ApiError(400, "A prompt of 1–8000 characters is required")
                changed = session.prompt != prompt
                if getattr(self.adapter, "continuous_chunks", False):
                    if len(prompt) > 3000:
                        raise ApiError(400, "Live LTX prompt amendments are limited to 3000 characters")
                    if changed:
                        session.prompt_amendments.append(prompt)
                        while len(session.prompt_amendments) > 6 or sum(map(len, session.prompt_amendments)) > 3000:
                            session.prompt_amendments.pop(0)
                session.prompt = prompt
            elif kind == "pause":
                if session.resumed.is_set():
                    session.audio_paused_at = time.monotonic()
                session.resumed.clear()
                session.action = "stop"
                session.action_expires_at = 0
                if hasattr(self.adapter, "pause_controls"):
                    changed = self.adapter.pause_controls(session)
                elif hasattr(self.adapter, "clear_controls"):
                    self.adapter.clear_controls(session)
                session.status = "paused"
            elif kind == "resume":
                if hasattr(self.adapter, "resume_controls"):
                    changed = self.adapter.resume_controls(session)
                if session.audio_paused_at is not None:
                    session.audio_started_at += time.monotonic() - session.audio_paused_at
                    session.audio_paused_at = None
                session.resumed.set()
                session.status = "generating"
            else:
                raise ApiError(422, "This adapter supports native camera actions, next-clip prompts, pause and resume")
            continuous = getattr(self.adapter, "continuous_chunks", False)
            if continuous and changed and kind in ("native", "prompt", "pause", "resume"):
                session.queued_revision += 1
            revision = session.queued_revision
        return {"accepted": True, "appliesAt": ("next-chunk" if continuous else "next-clip") if kind in ("native", "prompt") else "playback-and-next-clip", "id": identifier, "revision": revision}

    def _run(self, session, command, cwd=None, timeout=1200):
        with session.lock:
            if session.stop.is_set():
                raise InterruptedError()
            session.process = subprocess.Popen(command, cwd=cwd, env=child_environment(), stdin=subprocess.DEVNULL,
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            process = session.process
        try:
            result = process.wait(timeout=timeout)
            if session.stop.is_set():
                raise InterruptedError()
            if result:
                raise RuntimeError("Inference or media decoding failed. Verify GPU memory, the pinned environment and complete checkpoint files. Prompt/media logs are disabled.")
        except subprocess.TimeoutExpired:
            self._terminate(process)
            raise RuntimeError("The worker operation timed out and was stopped") from None
        finally:
            with session.lock:
                if session.process is process:
                    session.process = None

    @staticmethod
    def _terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def _wait_running(self, session):
        while not session.stop.is_set():
            if session.resumed.wait(0.2):
                return True
        return False

    def _generate(self, session):
        resident = None
        try:
            # Decode and rewrite uploads to strip metadata and enforce bounded dimensions.
            if session.image:
                clean = session.directory / "reference.png"
                header = session.image.read_bytes()[:12]
                decoder = "png" if header.startswith(b"\x89PNG") else "mjpeg" if header.startswith(b"\xff\xd8") else "webp"
                resolution = self.adapter.capabilities.get("runtime", {}).get("resolutionOptions", ["832x480"])[0]
                if not re.fullmatch(r"[0-9]{2,4}x[0-9]{2,4}", resolution):
                    raise RuntimeError("Invalid adapter resolution")
                self._run(session, ["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe", "-f", "image2pipe", "-c:v", decoder, "-i", str(session.image), "-frames:v", "1", "-vf", "scale=" + resolution.replace("x", ":") + ":force_original_aspect_ratio=decrease", "-map_metadata", "-1", str(clean)], timeout=30)
                session.image.unlink()
                session.image = clean
            if hasattr(self.adapter, "open_session"):
                session.resident = True
                resident = self.adapter.open_session(child_environment(), stop_event=session.stop)
                with session.lock:
                    session.process = resident.process
            if getattr(self.adapter, "continuous_chunks", False):
                if resident is None:
                    raise RuntimeError("Continuous chunks require a resident inference process")
                self._generate_continuous(session, resident)
                return
            while self._wait_running(session):
                with session.lock:
                    if session.stop.is_set():
                        return
                    if not session.resumed.is_set():
                        continue
                    session.status = "generating"
                    prompt = session.prompt
                    action = self.adapter.control_state(session) if hasattr(self.adapter, "control_state") else session.current_action()
                directory = session.directory / "chunk"
                directory.mkdir(mode=0o700)
                started = time.monotonic()
                audio = None
                if resident:
                    def phase(value):
                        with session.lock:
                            if session.resumed.is_set() and not session.stop.is_set():
                                session.status = "loading" if value == "loading-model" else "generating"
                    result = resident.generate(directory, prompt, session.image, action, (session.seed + session.chunks) % (2**32), session.quality, session.stop, on_phase=phase)
                    raw = result.get("rawFrames")
                    raw_path = validate_raw_frames(raw, directory) if raw is not None else None
                    files = [Path(path) for path in result.get("framePaths", [])]
                    if len(files) > 240 or any(not path.resolve().is_relative_to(directory.resolve()) for path in files):
                        raise RuntimeError("The resident worker returned invalid output paths")
                    source_continuation = Path(result["continuationPath"])
                    if not source_continuation.resolve().is_relative_to(directory.resolve()):
                        raise RuntimeError("The resident worker returned invalid continuation data")
                    playback_fps = max(1, min(60, float(result["fps"])))
                    duration = max(0.001, float(result["generationSeconds"]))
                    with session.lock:
                        session.model_load_seconds = result.get("loadSeconds", result.get("modelLoadSeconds"))
                        session.model_load_count = result.get("loadCount", result.get("modelLoadCount", 0))
                        session.peak_vram_bytes = result.get("peakVRAMBytes")
                else:
                    raw = raw_path = None
                    command, cwd = self.adapter.command(directory, prompt, session.image, action,
                                                        (session.seed + session.chunks) % (2**32), session.quality)
                    self._run(session, command, cwd)
                    video = directory / "chunk.mp4"
                    if not video.is_file():
                        raise RuntimeError("The model completed without producing the expected video")
                    resolution = self.adapter.capabilities.get("runtime", {}).get("resolutionOptions", ["832x480"])[0]
                    playback_fps = getattr(self.adapter, "output_fps", 12)
                    if getattr(self.adapter, "offline_clip", False):
                        width, height = map(int, resolution.split("x"))
                        raw_path = directory / "frames.rgb"
                        self._run(session, ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", "0:v:0", "-vf", f"fps={playback_fps},scale={width}:{height}",
                                            "-frames:v", str(self.adapter.output_frames), "-an", "-pix_fmt", "rgb24", "-f", "rawvideo", "-fs", str(192 * 1024 * 1024), str(raw_path)], timeout=60)
                        frame_size = width * height * 3
                        raw = {"path": str(raw_path), "width": width, "height": height, "count": raw_path.stat().st_size // frame_size, "pixelFormat": "rgb24"}
                        validate_raw_frames(raw, directory)
                        source_continuation = directory / "last.png"
                        from PIL import Image
                        with raw_path.open("rb") as spool:
                            spool.seek(-frame_size, 2)
                            Image.frombytes("RGB", (width, height), spool.read(frame_size)).save(source_continuation)
                        files = []
                    else:
                        frames = directory / "frames"
                        frames.mkdir()
                        self._run(session, ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-vf", f"fps={playback_fps},scale=" + resolution.replace("x", ":"), "-frames:v", "240", "-q:v", "3", str(frames / "%05d.jpg")], timeout=60)
                        files = sorted(frames.glob("*.jpg"))
                        source_continuation = files[-1] if files else None
                    if self.adapter.capabilities.get("output", {}).get("audio"):
                        audio_path = directory / "audio.pcm"
                        self._run(session, ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", "-t", "30", "-f", "s16le", "-fs", "5760000", str(audio_path)], timeout=60)
                        audio = audio_path.read_bytes()
                        if not audio or len(audio) % 4 or len(audio) > 5760000:
                            raise RuntimeError("The model did not produce valid bounded stereo audio")
                    duration = max(0.001, time.monotonic() - started)
                count = raw["count"] if raw else len(files)
                if not count:
                    raise RuntimeError("The generated video contained no decodable frames")
                with session.lock:
                    session.generation_seconds = round(duration, 3)
                    # Resident duration excludes one-time model loading; playback is separate.
                    session.generated_fps = round(count / duration, 3)
                    session.total_generated_frames += count
                    session.total_generation_seconds += duration
                    session.audio = audio
                    session.audio_index += 1
                    session.audio_started_at = time.monotonic()
                    session.audio_paused_at = None if session.resumed.is_set() else session.audio_started_at
                def decoded_frames():
                    if raw_path:
                        size = raw["width"] * raw["height"] * 3
                        with raw_path.open("rb") as spool:
                            for _ in range(count):
                                data = spool.read(size)
                                if len(data) != size:
                                    raise RuntimeError("Raw frame spool was truncated")
                                yield data
                    else:
                        for path in files:
                            yield path.read_bytes()
                for data in decoded_frames():
                    if not self._wait_running(session):
                        return
                    with session.lock:
                        if raw:
                            session.raw_frame = data
                            session.raw_size = (raw["width"], raw["height"])
                            session.frame = None
                        else:
                            session.frame = data
                            session.raw_frame = session.raw_size = None
                        session.frame_index += 1
                        session.status = "playing"
                    if session.stop.wait(1 / playback_fps):
                        return
                continuation = session.directory / ("continuation.png" if resident or raw else "continuation.jpg")
                continuation.write_bytes(source_continuation.read_bytes())
                if session.image and session.image != continuation:
                    session.image.unlink(missing_ok=True)
                session.image = continuation
                session.chunks += 1
                shutil.rmtree(directory)
                if getattr(self.adapter, "offline_clip", False):
                    with session.lock:
                        session.resumed.clear()
                        session.audio_paused_at = time.monotonic()
                        session.status = "paused"
        except InterruptedError:
            pass
        except Exception as error:
            with session.lock:
                session.status = "error"
                # Only our known messages cross the boundary. Exceptions may contain prompts/paths.
                session.error = str(error) if isinstance(error, RuntimeError) else "The worker failed. Verify its installed runtime and GPU availability."
        finally:
            if resident:
                if hasattr(self.adapter, "release_session"):
                    self.adapter.release_session(resident, clean=session.error is None)
                else:
                    resident.close()
                with session.lock:
                    session.process = None
            if session.stop.is_set():
                shutil.rmtree(session.directory, ignore_errors=True)

    def _generate_continuous(self, session, resident):
        """Overlap playback with one future chunk, counting in-flight work in the bound.

        Pause lets an already started GPU operation finish and retains its carry;
        it never starts another operation until resumed. A private numbered spool
        survives only until its packet has played, or until cancellation cleanup.
        """
        from ltx25.adapter import validate_audio
        packets = queue.Queue(maxsize=1)
        capacity = threading.Semaphore(1)
        producer_stop = threading.Event()
        producer_done = threading.Event()
        failures = []

        def produce():
            number = 0
            try:
                while not session.stop.is_set() and not producer_stop.is_set():
                    if not capacity.acquire(timeout=0.1):
                        continue
                    while not session.resumed.wait(0.1):
                        if session.stop.is_set() or producer_stop.is_set():
                            return
                    if session.stop.is_set() or producer_stop.is_set():
                        return
                    with session.lock:
                        # The pause action and this scheduling boundary share a lock.
                        if not session.resumed.is_set():
                            capacity.release()
                            continue
                        action = self.adapter.control_state(session)
                        prompt = session.prompt
                        session.generating_revision = action['revision']
                        if session.frame_index == 0:
                            session.status = 'generating'
                    directory = session.directory / f'chunk-{number:08d}'
                    directory.mkdir(mode=0o700)
                    def phase(value):
                        with session.lock:
                            if session.resumed.is_set() and not session.stop.is_set() and not session.frame_index:
                                session.status = 'loading' if value == 'loading-model' else 'generating'
                    result = resident.generate(directory, prompt, session.image, action,
                                               session.seed, session.quality,
                                               session.stop, on_phase=phase)
                    raw = result.get('rawFrames')
                    path = validate_raw_frames(raw, directory)
                    audio_path = validate_audio(result.get('audio'), directory)
                    continuation = Path(result['continuationPath'])
                    if not continuation.is_file() or not continuation.resolve().is_relative_to(directory.resolve()):
                        raise RuntimeError('Invalid continuous chunk continuation')
                    duration = result.get('generationSeconds')
                    import math
                    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                        raise RuntimeError('Invalid continuous generation timing')
                    revision = result.get('appliedRevision')
                    if isinstance(revision, bool) or not isinstance(revision, int) or not 0 <= revision <= action['revision']:
                        raise RuntimeError('Invalid continuous generation revision')
                    if result.get('fps') != 24:
                        raise RuntimeError('Unexpected continuous generation frame rate')
                    # Audio duration must agree with this packet, including seam tails.
                    if abs(result['audio']['samples'] / 48000 - raw['count'] / 24) > 0.1:
                        raise RuntimeError('Generated audio and video durations disagree')
                    with session.lock:
                        session.generation_seconds = round(duration, 3)
                        session.generated_fps = round(raw['count'] / duration, 3)
                        session.total_generated_frames += raw['count']
                        session.total_generation_seconds += duration
                        session.model_load_seconds = result.get('loadSeconds', result.get('modelLoadSeconds'))
                        session.model_load_count = result.get('loadCount', result.get('modelLoadCount', 0))
                        session.peak_vram_bytes = result.get('peakVRAMBytes')
                        session.transformer_load_count = result.get('transformerLoadCount')
                        session.prompt_encoding_count = result.get('promptEncodingCount')
                        session.continuity = result.get('continuity')
                        session.residency = result.get('residency')
                        session.rendered_window_count = result.get('renderedWindowCount')
                        session.sampling_seconds = result.get('samplingSeconds')
                        session.prompt_truncated = result.get('promptTruncated') is True
                        session.generating_revision = None
                        session.buffered_chunks = 1
                    packets.put_nowait((directory, path, audio_path, result))
                    number += 1
            except InterruptedError:
                pass
            except BaseException as error:
                failures.append(error)
            finally:
                producer_done.set()

        def wait_playing():
            while not session.stop.is_set():
                if failures:
                    raise failures[0]
                if session.resumed.wait(0.1):
                    return True
            return False

        producer = threading.Thread(target=produce, name='world-continuous-producer', daemon=True)
        producer.start()
        try:
            while wait_playing():
                try:
                    directory, path, audio_path, result = packets.get(timeout=0.1)
                except queue.Empty:
                    if producer_done.is_set():
                        if failures:
                            raise failures[0]
                        return
                    continue
                # Taking the packet releases exactly one future inference slot.
                capacity.release()
                raw = result['rawFrames']
                with session.lock:
                    session.buffered_chunks = 0
                    session.audio = audio_path.read_bytes()
                    session.audio_index += 1
                    session.audio_started_at = time.monotonic()
                    session.audio_paused_at = None if session.resumed.is_set() else session.audio_started_at
                    session.applied_revision = result['appliedRevision']
                size = raw['width'] * raw['height'] * 3
                with path.open('rb') as spool:
                    for _ in range(raw['count']):
                        if not wait_playing():
                            return
                        data = spool.read(size)
                        if len(data) != size:
                            raise RuntimeError('Raw frame spool was truncated')
                        with session.lock:
                            session.raw_frame = data
                            session.raw_size = (raw['width'], raw['height'])
                            session.frame = None
                            session.frame_index += 1
                            session.status = 'playing'
                        if session.stop.wait(1 / 24):
                            return
                with session.lock:
                    session.chunks += 1
                shutil.rmtree(directory)
        finally:
            producer_stop.set()
            # Closing interrupts blocked protocol reads and terminates the GPU group.
            # LTX reuse is deliberately disabled until upstream reset isolation is proven.
            resident.close()
            producer.join(timeout=12)
            if producer.is_alive():
                raise RuntimeError('Continuous producer did not stop; worker requires cleanup')
            for path in session.directory.glob('chunk-*'):
                if path.is_dir() and path.resolve().parent == session.directory.resolve():
                    shutil.rmtree(path)

    def delete(self, identifier):
        with self.lock:
            session = self.sessions.get(identifier)
            if not session:
                raise ApiError(404, "Session not found")
            session.stop.set()
            session.resumed.set()
        self.streaming.close_session(identifier)
        with session.lock:
            process = session.process
        if process and not session.resident:
            self._terminate(process)
        if session.thread:
            session.thread.join(timeout=12)
        if session.thread and session.thread.is_alive():
            raise ApiError(503, "Worker is still stopping; retry deletion")
        shutil.rmtree(session.directory, ignore_errors=True)
        with self.lock:
            self.sessions.pop(identifier, None)
        session.frame = session.raw_frame = session.audio = None
        session.prompt = ""
        session.base_prompt = None
        session.prompt_amendments.clear()
        if hasattr(self.adapter, "clear_controls"):
            self.adapter.clear_controls(session)

    def _reap(self):
        while not self.closed.wait(min(10, self.idle_seconds)):
            with self.lock:
                expired = [key for key, value in self.sessions.items() if time.monotonic() - value.last_used > self.idle_seconds]
            for identifier in expired:
                try:
                    self.delete(identifier)
                except ApiError:
                    pass

    def close(self):
        with self.lock:
            self.closed.set()
            identifiers = list(self.sessions)
        failure = None
        try:
            for identifier in identifiers:
                try:
                    self.delete(identifier)
                except ApiError as error:
                    # The reaper or an in-flight DELETE may already have removed it.
                    if error.status != 404:
                        failure = failure or error
                except Exception as error:
                    failure = failure or error
        finally:
            self.streaming.close()
            if hasattr(self.adapter, "close"):
                self.adapter.close()
            # A process that has not stopped may still use its private directory.
            # Keep that failure explicit, rather than claiming complete deletion.
            if not self.sessions:
                shutil.rmtree(self.root, ignore_errors=True)
        if failure is not None:
            raise failure


def handler_for(gateway, token):
    class Handler(BaseHTTPRequestHandler):
        server_version = "WorldWorker/1"
        def setup(self):
            self.request.settimeout(15)
            super().setup()
        def log_message(self, *_):
            pass  # URLs, prompts, payloads and bearer credentials are never logged.

        def _reply(self, status, value=None, content_type="application/json", extra=None):
            payload = b"" if value is None else value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, item in (extra or {}).items():
                self.send_header(key, str(item))
            self.end_headers()
            if payload:
                self.wfile.write(payload)

        def _body(self):
            if self.headers.get("Transfer-Encoding"):
                raise ApiError(400, "Transfer-Encoding is not supported")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ApiError(400, "Invalid content length") from None
            if length < 0 or length > MAX_BODY:
                raise ApiError(413, "Request is too large")
            if self.headers.get_content_type() != "application/json":
                raise ApiError(415, "Content-Type must be application/json")
            try:
                data = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeError):
                raise ApiError(400, "Invalid JSON") from None
            if not isinstance(data, dict):
                raise ApiError(400, "Expected a JSON object")
            return data

        def _dispatch(self):
            self.connection.settimeout(15)
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()):
                raise ApiError(401, "Worker authentication required")
            path = urlsplit(self.path).path
            if self.command == "GET" and path == "/health":
                metadata = gateway.adapter.metadata()
                self._reply(200, {"status": metadata["status"], "protocolVersion": 1, "models": [metadata]})
                return
            if self.command == "POST" and path == "/sessions":
                self._reply(201, gateway.create(self._body()))
                return
            match = re.fullmatch(r"/sessions/([A-Za-z0-9_-]{8,80})(?:/(actions|frame|snapshot|offer|transport-config|heartbeat))?", path)
            if not match:
                raise ApiError(404, "Endpoint not found")
            identifier, endpoint = match.groups()
            session = gateway.get(identifier)
            if not endpoint and self.command == "DELETE":
                gateway.delete(identifier)
                self._reply(204)
            elif not endpoint and self.command == "GET":
                self._reply(200, session.public(gateway.adapter.capabilities))
            elif endpoint == "actions" and self.command == "POST":
                self._reply(200, gateway.action(identifier, self._body()))
            elif endpoint == "frame" and self.command == "GET":
                frame, index = session.latest_jpeg()
                self._reply(200 if frame else 204, frame, "image/jpeg", {"X-Frame-Index": index})
            elif endpoint == "snapshot" and self.command == "POST":
                frame, _index = session.latest_jpeg()
                if not frame:
                    raise ApiError(409, "No generated frame is available yet")
                self._reply(200, {"resumeKind": "visual-checkpoint", "frame": "data:image/jpeg;base64," + base64.b64encode(frame).decode(), "state": None})
            elif endpoint == "offer" and self.command == "POST":
                self._reply(200, gateway.streaming.offer(identifier, self._body()))
            elif endpoint == "transport-config" and self.command == "GET":
                self._reply(200, gateway.streaming.config(identifier))
            elif endpoint == "heartbeat" and self.command == "POST":
                body = self._body()
                if not isinstance(body.get("active"), bool):
                    raise ApiError(400, "Heartbeat requires an active boolean")
                gateway.get(identifier, active=body["active"])
                self._reply(200, {"id": identifier, "status": session.status})
            else:
                raise ApiError(405, "Method not supported")

        def _handle(self):
            try:
                self._dispatch()
            except (ApiError, StreamingError) as error:
                self._reply(error.status, {"error": error.message})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except Exception:
                self._reply(500, {"error": "Worker request failed"})

        do_GET = do_POST = do_DELETE = _handle
    return Handler


def main():
    token = os.environ.get("WORLD_GATEWAY_TOKEN", "")
    if len(token) < 32 or not token.isascii() or any(not 33 <= ord(char) <= 126 for char in token):
        raise SystemExit("Set WORLD_GATEWAY_TOKEN to a random printable ASCII token of at least 32 characters, without spaces")
    try:
        idle_seconds = int(os.environ.get("WORLD_IDLE_SECONDS", "300"))
        if not 60 <= idle_seconds <= 14400:
            raise ValueError
    except ValueError:
        raise SystemExit("WORLD_IDLE_SECONDS must be an integer between 60 and 14400") from None
    gateway = Gateway(idle_seconds=idle_seconds)
    server = BoundedHTTPServer((os.environ.get("WORLD_BIND", "127.0.0.1"), int(os.environ.get("PORT", "8789"))), handler_for(gateway, token))
    def shutdown_signal(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown_signal)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        gateway.close()


if __name__ == "__main__":
    main()

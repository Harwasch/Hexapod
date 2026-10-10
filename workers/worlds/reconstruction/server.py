"""Private, bounded GPU reconstruction service; provisions no compute itself."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
MAX_MEDIA = 64 * 1024 * 1024
MAX_WIRE = MAX_MEDIA + 1024 * 1024
FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
}
OUTPUTS = {
    "splat": ("scene.ply", "ply", "application/octet-stream"),
    "points": ("points.ply", "point-cloud", "application/octet-stream"),
    "mesh": ("mesh.glb", "glb", "model/gltf-binary"),
    "diagnostics": ("diagnostics.json", None, "application/json"),
    "cameras": ("cameras.json", None, "application/json"),
}


class BoundedRoute(APIRoute):
    def get_route_handler(self):
        handle = super().get_route_handler()

        async def bounded(request):
            if request.method != "POST":
                return await handle(request)
            try:
                size = int(request.headers.get("content-length", "0"))
            except ValueError:
                raise HTTPException(400, "Invalid content length") from None
            if not 0 <= size <= MAX_WIRE:
                raise HTTPException(413, "Media upload exceeds 64 MiB")
            total = 0

            async def receive():
                nonlocal total
                message = await request.receive()
                if message["type"] == "http.request":
                    total += len(message.get("body", b""))
                    if total > MAX_WIRE:
                        raise HTTPException(413, "Media upload exceeds 64 MiB")
                return message

            return await handle(Request(request.scope, receive=receive))

        return bounded


def url_ok(value):
    parsed = urlsplit(value)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Public URL must not contain credentials, queries or fragments"
        )
    if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
        raise ValueError("Hosted reconstruction requires HTTPS")
    return value.rstrip("/")


def child_environment():
    keep = (
        "PATH",
        "HOME",
        "LANG",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "TORCH_HOME",
        "HF_HOME",
        "TMPDIR",
    )
    return {
        **{key: os.environ[key] for key in keep if key in os.environ},
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_MODE": "disabled",
        "DO_NOT_TRACK": "1",
    }


@dataclass
class Job:
    id: str
    directory: Path
    created: float
    status: str = "queued"
    stage: str = "Queued"
    progress: float = 0
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    process: subprocess.Popen | None = None


class Engine:
    def __init__(
        self,
        root,
        token,
        public_url,
        source,
        weights,
        *,
        max_frames=12,
        target_size=518,
        retention=3600,
        timeout=1800,
        command=None,
        ready_check=None,
    ):
        if (
            len(token) < 32
            or not token.isascii()
            or any(not 33 <= ord(c) <= 126 for c in token)
        ):
            raise ValueError(
                "Set a random reconstruction gateway token of at least 32 characters"
            )
        if not 60 <= retention <= 86400 or not 60 <= timeout <= 3600:
            raise ValueError("Retention/runtime settings exceed supported bounds")
        if not 2 <= max_frames <= 24 or target_size not in (280, 392, 518):
            raise ValueError("Unsupported frame or resolution limit")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.token, self.public_url = token, url_ok(public_url)
        self.source, self.weights = Path(source).resolve(), Path(weights).resolve()
        self.max_frames, self.target_size = max_frames, target_size
        self.retention, self.timeout = retention, timeout
        self.command, self.ready_check = command, ready_check
        self.jobs = {}
        self.lock = threading.RLock()
        self.gpu = threading.Semaphore(1)
        self.stop = threading.Event()
        self._recover()
        self.reaper = threading.Thread(target=self._reap, daemon=True)
        self.reaper.start()

    def ready(self):
        if self.ready_check is not None:
            return self.ready_check()
        try:
            revision = (
                subprocess.check_output(
                    ["git", "-C", str(self.source), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                .decode()
                .strip()
            )
            if revision != MANIFEST["sourceRevision"]:
                return False
            marker = (self.weights / ".worlds-mirror-checkpoint").read_text().strip()
            return marker == MANIFEST["checkpointRevision"] and all(
                (self.weights / file).is_file()
                for file in ("model.safetensors", "config.json")
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return False

    def _persist(self, job):
        data = {
            "id": job.id,
            "created": job.created,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "error": job.error,
        }
        temporary = job.directory / "job.tmp"
        temporary.write_text(json.dumps(data))
        temporary.replace(job.directory / "job.json")

    def _recover(self):
        for directory in self.root.iterdir():
            if not directory.is_dir() or not re.fullmatch(
                r"[a-f0-9]{32}", directory.name
            ):
                continue
            try:
                data = json.loads((directory / "job.json").read_text())
                if time.time() - float(data["created"]) > self.retention:
                    shutil.rmtree(directory)
                    continue
                job = Job(
                    directory.name,
                    directory,
                    float(data["created"]),
                    data["status"],
                    data["stage"],
                    float(data["progress"]),
                    data.get("error"),
                )
                if job.status not in {"completed", "failed"}:
                    job.status, job.stage, job.error = (
                        "failed",
                        "Failed",
                        "Worker restarted before reconstruction completed",
                    )
                    shutil.rmtree(directory / "output", ignore_errors=True)
                for name in ("inputs", "frames"):
                    shutil.rmtree(directory / name, ignore_errors=True)
                self.jobs[job.id] = job
                self._persist(job)
            except (OSError, ValueError, KeyError, TypeError):
                shutil.rmtree(directory, ignore_errors=True)

    def reserve(self):
        if not self.ready():
            raise HTTPException(
                503,
                "Prepare the pinned Mirror source and checkpoint before reconstruction",
            )
        with self.lock:
            if (
                self.stop.is_set()
                or sum(j.status in {"queued", "running"} for j in self.jobs.values())
                >= 2
            ):
                raise HTTPException(
                    429,
                    "The reconstruction queue is full; try again after the current job",
                )
            if len(self.jobs) >= 32:
                raise HTTPException(
                    429, "Delete old reconstructions or wait for their retention period"
                )
            identifier = uuid.uuid4().hex
            directory = self.root / identifier
            directory.mkdir(mode=0o700)
            (directory / "inputs").mkdir(mode=0o700)
            job = Job(identifier, directory, time.time())
            self.jobs[identifier] = job
            self._persist(job)
            return job

    def start(self, job):
        with self.lock:
            if self.stop.is_set() or job.cancel.is_set():
                raise HTTPException(503, "The worker is stopping")
            job.thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            job.thread.start()

    @staticmethod
    def _terminate(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def _run(self, job):
        acquired = False
        try:
            while not job.cancel.is_set() and not self.stop.is_set():
                if self.gpu.acquire(timeout=0.2):
                    acquired = True
                    break
            if not acquired or job.cancel.is_set():
                return
            with self.lock:
                job.status, job.stage = "running", "Preparing sources"
                self._persist(job)
                command = (
                    self.command(job)
                    if self.command
                    else [
                        sys.executable,
                        str(ROOT / "runner.py"),
                        "--job",
                        str(job.directory),
                        "--source",
                        str(self.source),
                        "--weights",
                        str(self.weights),
                        "--max-frames",
                        str(self.max_frames),
                        "--target-size",
                        str(self.target_size),
                    ]
                )
                job.process = subprocess.Popen(
                    command,
                    env=child_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            started = time.monotonic()
            while job.process.poll() is None:
                if (
                    job.cancel.wait(0.2)
                    or self.stop.is_set()
                    or time.monotonic() - started > self.timeout
                ):
                    self._terminate(job.process)
                    raise RuntimeError(
                        "Reconstruction stopped or exceeded its runtime limit"
                    )
                self._progress(job)
            if job.process.returncode != 0:
                raise RuntimeError(
                    "Reconstruction failed; verify CUDA memory, checkpoint files and overlapping views"
                )
            if (
                not (job.directory / "output" / "scene.ply").is_file()
                or not (job.directory / "output" / "points.ply").is_file()
            ):
                raise RuntimeError("The engine did not produce the required geometry")
            with self.lock:
                job.status, job.stage, job.progress = "completed", "Completed", 1
                self._persist(job)
        except Exception as error:  # noqa: BLE001 -- contain upstream/subprocess failures per job
            with self.lock:
                job.status, job.stage = "failed", "Failed"
                job.error = (
                    str(error)
                    if isinstance(error, RuntimeError)
                    else "The reconstruction worker failed"
                )
                shutil.rmtree(job.directory / "output", ignore_errors=True)
                self._persist(job)
        finally:
            for name in ("inputs", "frames"):
                shutil.rmtree(job.directory / name, ignore_errors=True)
            if acquired:
                self.gpu.release()

    def _progress(self, job):
        try:
            data = json.loads((job.directory / "progress.json").read_text())
            stage = data.get("stage")
            progress = float(data.get("progress", 0))
            if (
                stage
                in {
                    "Preparing sources",
                    "Reconstructing scene",
                    "Exporting geometry",
                    "Completed",
                }
                and 0 <= progress <= 1
            ):
                with self.lock:
                    job.stage, job.progress = stage, progress
                    self._persist(job)
        except (OSError, ValueError, TypeError):
            pass

    def get(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise HTTPException(404, "Reconstruction not found or expired")
            return job

    def signature(self, identifier, kind, expires):
        return hmac.new(
            self.token.encode(),
            f"{identifier}/{kind}/{expires}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def url(self, identifier, kind):
        expires = int(time.time()) + 300
        return f"{self.public_url}/artifacts/{identifier}/{kind}?" + urlencode(
            {"expires": expires, "signature": self.signature(identifier, kind, expires)}
        )

    def public(self, job):
        with self.lock:
            result = {
                "id": job.id,
                "status": job.status,
                "stage": job.stage,
                "progress": job.progress,
                "error": job.error,
                "artifacts": [],
            }
            if job.status == "completed":
                for kind, (name, format_name, _mime) in OUTPUTS.items():
                    if format_name and (job.directory / "output" / name).is_file():
                        result["artifacts"].append(
                            {"format": format_name, "url": self.url(job.id, kind)}
                        )
                result["diagnosticsUrl"] = self.url(job.id, "diagnostics")
                result["camerasUrl"] = self.url(job.id, "cameras")
            return result

    def delete(self, identifier):
        job = self.get(identifier)
        with self.lock:
            job.cancel.set()
            process = job.process
        if process is not None:
            self._terminate(process)
        if job.thread:
            job.thread.join(timeout=12)
            if job.thread.is_alive():
                raise HTTPException(503, "The job is still stopping; retry deletion")
        with self.lock:
            shutil.rmtree(job.directory)
            self.jobs.pop(identifier, None)
        return {"deleted": True}

    def _reap(self):
        while not self.stop.wait(5):
            with self.lock:
                expired = [
                    j.id
                    for j in self.jobs.values()
                    if time.time() - j.created > self.retention
                ]
            for identifier in expired:
                try:
                    self.delete(identifier)
                except (HTTPException, OSError):
                    pass

    def close(self):
        self.stop.set()
        with self.lock:
            active = [
                j.id for j in self.jobs.values() if j.status in {"queued", "running"}
            ]
        for identifier in active:
            try:
                self.delete(identifier)
            except (HTTPException, OSError):
                # Stop every active process even if another job's cleanup failed.
                continue
        self.reaper.join(timeout=6)


def create_app(engine: Engine, origins=()):
    @asynccontextmanager
    async def lifespan(_app):
        yield
        await asyncio.to_thread(engine.close)

    app = FastAPI(
        title="Worlds Mirror reconstruction worker",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.router.route_class = BoundedRoute
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(origins),
            allow_methods=["GET"],
            allow_headers=[],
        )

    def authenticate(request: Request):
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(
            supplied.encode(), ("Bearer " + engine.token).encode()
        ):
            raise HTTPException(401, "Worker authentication required")

    @app.get("/health", dependencies=[Depends(authenticate)])
    def health():
        ready = engine.ready()
        return {
            "status": "ready" if ready else "unavailable",
            "protocolVersion": 1,
            "models": [
                {
                    "id": "hunyuanworld-mirror",
                    "status": "ready" if ready else "unavailable",
                    "version": MANIFEST["sourceRevision"],
                    "checkpointRevision": MANIFEST["checkpointRevision"],
                    "readiness": "artifacts-verified-gpu-unverified",
                }
            ],
            "retentionSeconds": engine.retention,
            "maxFrames": engine.max_frames,
        }

    @app.post("/reconstructions", status_code=202, dependencies=[Depends(authenticate)])
    async def submit(request: Request):
        job = engine.reserve()
        try:
            # Authenticate before multipart parsing. No prompts or original filenames
            # are saved. The API boundary already strips image EXIF metadata.
            async with request.form(
                max_files=120, max_fields=1, max_part_size=64 * 1024
            ) as form:
                files = form.getlist("files")
                if not 1 <= len(files) <= 120:
                    raise HTTPException(422, "Choose between 1 and 120 source files")
                total = 0
                for index, file in enumerate(files):
                    mime = (getattr(file, "content_type", "") or "").split(";")[0]
                    if mime not in FORMATS:
                        raise HTTPException(422, "Unsupported source media")
                    data = await file.read(MAX_MEDIA - total + 1)
                    total += len(data)
                    if not data or total > MAX_MEDIA:
                        raise HTTPException(413, "Media is empty or exceeds 64 MiB")
                    (
                        job.directory / "inputs" / f"source-{index:03d}.{FORMATS[mime]}"
                    ).write_bytes(data)
            engine.start(job)
            return engine.public(job)
        except BaseException:
            await asyncio.to_thread(engine.delete, job.id)
            raise

    @app.get("/reconstructions/{identifier}", dependencies=[Depends(authenticate)])
    def status(identifier: str):
        return engine.public(engine.get(identifier))

    @app.delete("/reconstructions/{identifier}", dependencies=[Depends(authenticate)])
    async def delete(identifier: str):
        return await asyncio.to_thread(engine.delete, identifier)

    @app.get("/artifacts/{identifier}/{kind}")
    def artifact(identifier: str, kind: str, expires: int, signature: str):
        if (
            not 0 < expires - time.time() <= 300
            or not re.fullmatch(r"[0-9a-f]{64}", signature)
            or not hmac.compare_digest(
                signature, engine.signature(identifier, kind, expires)
            )
        ):
            raise HTTPException(403, "Download link expired or invalid")
        job = engine.get(identifier)
        if job.status != "completed" or kind not in OUTPUTS:
            raise HTTPException(404, "Artifact unavailable")
        name, _format, mime = OUTPUTS[kind]
        path = job.directory / "output" / name
        if not path.is_file():
            raise HTTPException(404, "Artifact unavailable")
        return FileResponse(
            path,
            media_type=mime,
            filename=name,
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return app


def main():
    import uvicorn

    os.umask(0o077)
    port = int(os.environ.get("PORT", "8790"))
    engine = Engine(
        os.environ.get("WORLD_RECONSTRUCTION_DATA_DIR", "data/reconstruction"),
        os.environ.get("WORLD_RECONSTRUCTION_GATEWAY_TOKEN", ""),
        os.environ.get("WORLD_RECONSTRUCTION_PUBLIC_URL", f"http://127.0.0.1:{port}"),
        os.environ.get("WORLD_RECONSTRUCTION_SOURCE", "/opt/hunyuanworld-mirror"),
        os.environ.get("WORLD_RECONSTRUCTION_WEIGHTS", "/models/hunyuanworld-mirror"),
        max_frames=int(os.environ.get("WORLD_RECONSTRUCTION_MAX_FRAMES", "12")),
        target_size=int(os.environ.get("WORLD_RECONSTRUCTION_TARGET_SIZE", "518")),
        retention=int(os.environ.get("WORLD_RECONSTRUCTION_RETENTION_SECONDS", "3600")),
        timeout=int(os.environ.get("WORLD_RECONSTRUCTION_TIMEOUT_SECONDS", "1800")),
    )
    origins = [
        value.strip()
        for value in os.environ.get("WORLD_RECONSTRUCTION_BROWSER_ORIGINS", "").split(
            ","
        )
        if value.strip()
    ]
    uvicorn.run(
        create_app(engine, origins),
        host=os.environ.get("WORLD_BIND", "127.0.0.1"),
        port=port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

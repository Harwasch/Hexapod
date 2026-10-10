"""Optional Worlds-only reconstruction gateway; never registers an Earth site.

The operator supplies a worker, normally on RunPod. Nothing provisions, deploys, or
falls back to the legacy Modal/capture pipeline. The worker accepts multipart POST
/reconstructions and GET /reconstructions/{id}. Credentials stay server-side.
"""

from __future__ import annotations

import io
import json
import os
import re
from collections.abc import Callable, Coroutine
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.routing import APIRoute
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.types import Message

from app.api.deps import RequireWriteToken, require_write_token, write_token_scheme
from app.config import REPO_ROOT
from app.worlds.config import validate_url
from app.worlds.providers import MAX_RESPONSE_BYTES, request, response_json

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_MULTIPART_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024


class BoundedUploadRoute(APIRoute):
    """Enforce a wire limit before FastAPI spools multipart files, including chunked bodies."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handle = super().get_route_handler()

        async def bounded(request: Request) -> Response:
            if request.method != "POST":
                return await handle(request)
            # FastAPI resolves File/Form bodies before ordinary dependencies. Use
            # this app's configured auth here too, before accepting any upload bytes.
            settings = getattr(request.app.state, "settings", None)
            if settings is not None:
                require_write_token(settings, await write_token_scheme(request))
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    size = int(length)
                except ValueError:
                    raise HTTPException(400, "Invalid upload length.") from None
                if size < 0 or size > MAX_MULTIPART_BYTES:
                    raise HTTPException(
                        413, "Reconstruction upload exceeds the 64 MiB media limit."
                    )
            received = 0

            async def receive() -> Message:
                nonlocal received
                message = await request.receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > MAX_MULTIPART_BYTES:
                        # Starlette closes temporary files when stream parsing raises.
                        raise HTTPException(
                            413, "Reconstruction upload exceeds the 64 MiB media limit."
                        )
                return message

            return await handle(Request(request.scope, receive=receive))

        return bounded


router = APIRouter(
    prefix="/worlds/reconstructions",
    tags=["worlds"],
    dependencies=[RequireWriteToken],
    route_class=BoundedUploadRoute,
)
MIME_TYPES = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
}


def image_without_metadata(content: bytes, mime: str) -> bytes:
    """Forward pixels, not EXIF/GPS/XMP; normalize orientation into a fresh PNG."""
    try:
        with Image.open(io.BytesIO(content)) as source:
            expected = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[mime]
            if source.format != expected or getattr(source, "n_frames", 1) != 1:
                raise HTTPException(
                    422, "Choose a still image whose file type matches its content."
                )
            if source.width * source.height > 16_000_000:
                raise HTTPException(
                    422, "Resize source images below 16 megapixels before reconstruction."
                )
            oriented = ImageOps.exif_transpose(source)
            pixels = oriented.convert("RGBA" if "A" in oriented.getbands() else "RGB")
            clean = Image.new(pixels.mode, pixels.size)
            clean.paste(pixels)
            output = io.BytesIO()
            clean.save(output, format="PNG")
            return output.getvalue()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(422, "A source image could not be decoded safely.") from None


class ReconstructionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WORLD_RECONSTRUCTION_",
        extra="ignore",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
    )
    gateway_url: str | None = None
    gateway_token: SecretStr | None = None

    @classmethod
    def load(cls) -> ReconstructionSettings:
        path = os.getenv("WORLD_ENV_FILE")
        return cls(_env_file=path) if path else cls()


class ReconstructionArtifact(BaseModel):
    format: Literal["ply", "spz", "glb", "gltf", "point-cloud"]
    url: str

    @field_validator("url")
    @classmethod
    def safe_download(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
            or (
                parsed.scheme != "https"
                and not (
                    parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                )
            )
        ):
            raise ValueError(
                "Artifact URLs must be HTTPS, or HTTP on localhost, without credentials"
            )
        # Signed short-lived object URLs may carry signatures. These are output download
        # credentials, never the worker/provider token. No server fetch uses this URL.
        return value


class ReconstructionJob(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    status: Literal["queued", "running", "completed", "failed"]
    progress: float | None = Field(default=None, ge=0, le=1)
    stage: str | None = Field(default=None, max_length=100)
    artifacts: list[ReconstructionArtifact] = Field(default_factory=list, max_length=10)
    error: str | None = None
    diagnostics_url: str | None = Field(default=None, alias="diagnosticsUrl")
    cameras_url: str | None = Field(default=None, alias="camerasUrl")

    @field_validator("diagnostics_url", "cameras_url")
    @classmethod
    def safe_metadata_download(cls, value: str | None) -> str | None:
        return ReconstructionArtifact.safe_download(value) if value is not None else None


def connection() -> tuple[str, str | None]:
    settings = ReconstructionSettings.load()
    if not settings.gateway_url:
        raise HTTPException(
            503,
            "3D reconstruction is not connected. Export a local source bundle, "
            "or configure WORLD_RECONSTRUCTION_GATEWAY_URL on the API server.",
        )
    try:
        validate_url(settings.gateway_url, local=True)
    except ValueError:
        raise HTTPException(503, "The reconstruction gateway configuration is invalid.") from None
    token = settings.gateway_token.get_secret_value() if settings.gateway_token else None
    loopback = urlsplit(settings.gateway_url).hostname in {"localhost", "127.0.0.1", "::1"}
    if not loopback and not (token and token.strip()):
        raise HTTPException(
            503,
            "Remote reconstruction workers require WORLD_RECONSTRUCTION_GATEWAY_TOKEN on the API server.",
        )
    return settings.gateway_url.rstrip("/"), token


def normalized_job(payload: dict[str, Any], token: str | None) -> ReconstructionJob:
    # Do not surface raw worker exception traces, internal URLs or credential material.
    stage = payload.get("stage")
    allowed_stages = {
        "Queued",
        "Preparing sources",
        "Extracting frames",
        "Estimating camera poses",
        "Estimating depth",
        "Reconstructing scene",
        "Training splat",
        "Exporting geometry",
        "Completed",
        "Failed",
    }
    payload = {
        **payload,
        "stage": stage if isinstance(stage, str) and stage in allowed_stages else None,
        "error": "The reconstruction worker failed. Try another static segment."
        if payload.get("status") == "failed"
        else None,
    }
    if token and token in json.dumps(payload):
        raise HTTPException(502, "Reconstruction worker returned an unsafe response.")
    try:
        return ReconstructionJob.model_validate(payload)
    except ValidationError:
        raise HTTPException(
            502, "Reconstruction worker returned an invalid job response."
        ) from None


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    try:
        connection()
        configured = True
    except HTTPException:
        configured = False
    return {
        "configured": configured,
        "maxUploadBytes": MAX_UPLOAD_BYTES,
        "formats": ["ply", "spz", "glb", "gltf", "point-cloud"],
        "message": "A separately configured worker processes selected media. No Earth site is published."
        if configured
        else "Reconstruction worker is not configured. Local source export is available.",
    }


@router.post("", response_model=ReconstructionJob, status_code=202)
def create_reconstruction(
    files: Annotated[list[UploadFile], File()],
    metadata: Annotated[str, Form()] = "{}",
) -> ReconstructionJob:
    url, token = connection()
    if not 1 <= len(files) <= 120:
        raise HTTPException(422, "Choose between 1 and 120 source files.")
    if len(metadata) > 64 * 1024:
        raise HTTPException(413, "Reconstruction metadata is too large.")
    try:
        supplied = json.loads(metadata)
        if not isinstance(supplied, dict):
            raise ValueError("Expected an object")
    except (ValueError, TypeError):
        raise HTTPException(422, "Reconstruction metadata must be a JSON object.") from None
    # Only source-relative timestamps/poses are required by a reconstruction worker.
    # Prompts, account information, user filenames and local project IDs are omitted.
    frames = supplied.get("frames", [])
    if not isinstance(frames, list) or len(frames) != len(files):
        raise HTTPException(422, "Metadata must describe each selected source file.")
    clean_frames = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise HTTPException(422, "Invalid source metadata.")
        clean: dict[str, Any] = {}
        timestamp = frame.get("timestampMs")
        if (
            isinstance(timestamp, int | float)
            and not isinstance(timestamp, bool)
            and 0 <= timestamp <= 86_400_000
        ):
            clean["timestampMs"] = timestamp
        pose = frame.get("cameraPose")
        if (
            isinstance(pose, list)
            and len(pose) in {12, 16}
            and all(
                isinstance(value, int | float) and not isinstance(value, bool) and abs(value) < 1e10
                for value in pose
            )
        ):
            clean["cameraPose"] = pose
        clean_frames.append(clean)
    payload = {
        "schema": "worlds-reconstruction/v1",
        "frames": clean_frames,
        "source": "generated-world",
        "geometry": "unverified",
    }
    upload_parts = []
    total = 0
    normalized_total = 0
    for index, file in enumerate(files):
        mime = (file.content_type or "").split(";")[0]
        if mime not in MIME_TYPES:
            raise HTTPException(422, "Choose JPEG, PNG, WebP, MP4, WebM or MOV source media.")
        content = file.file.read(MAX_UPLOAD_BYTES - total + 1)
        total += len(content)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, "Select a smaller segment; the reconstruction upload limit is 64 MiB."
            )
        if not content:
            raise HTTPException(422, "Source files must not be empty.")
        if mime.startswith("image/"):
            content = image_without_metadata(content, mime)
            mime = "image/png"
        normalized_total += len(content)
        if normalized_total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, "Normalized media exceeds 64 MiB. Select fewer or smaller images."
            )
        upload_parts.append(("files", (f"source-{index:05d}.{MIME_TYPES[mime]}", content, mime)))
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with (
            httpx.Client(timeout=httpx.Timeout(120, connect=10), follow_redirects=False) as client,
            client.stream(
                "POST",
                url + "/reconstructions",
                headers=headers,
                files=upload_parts,
                data={"metadata": json.dumps(payload)},
            ) as response,
        ):
            if response.status_code >= 300:
                code = (
                    response.status_code
                    if response.status_code in {409, 413, 422, 429, 501, 503}
                    else 502
                )
                raise HTTPException(code, "Reconstruction worker declined the request.")
            chunks = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise HTTPException(502, "Reconstruction response was too large.")
                chunks.append(chunk)
            result = response_json(httpx.Response(response.status_code, content=b"".join(chunks)))
            return normalized_job(result, token)
    except httpx.TimeoutException:
        raise HTTPException(
            504, "Reconstruction submission timed out; check the worker before retrying."
        ) from None
    except httpx.HTTPError:
        raise HTTPException(502, "Reconstruction worker is unreachable.") from None


@router.get("/{job_id}", response_model=ReconstructionJob)
def get_reconstruction(job_id: str) -> ReconstructionJob:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job_id):
        raise HTTPException(422, "Invalid reconstruction job ID.")
    url, token = connection()
    result = normalized_job(
        response_json(request("GET", url + f"/reconstructions/{job_id}", token=token)), token
    )
    if result.id != job_id:
        raise HTTPException(502, "Reconstruction worker returned a different job.")
    return result


@router.delete("/{job_id}")
def delete_reconstruction(job_id: str) -> dict[str, bool]:
    """Explicit worker cleanup; success requires the gateway to confirm deletion.

    The external worker must delete the job's input, partial and output media. A
    gateway without this contract returns an error; local state is not cleared.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job_id):
        raise HTTPException(422, "Invalid reconstruction job ID.")
    url, token = connection()
    response = request("DELETE", url + f"/reconstructions/{job_id}", token=token)
    if response.status_code != 204 and response_json(response).get("deleted") is not True:
        raise HTTPException(502, "The worker did not confirm deletion. Its data may still exist.")
    return {"deleted": True}

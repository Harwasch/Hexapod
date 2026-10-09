from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import morecantile
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analysis.image_registration import fit
from app.models.land_archive_image import LandArchiveImage
from app.models.land_image_registration import LandImageRegistration, LandImageRegistrationBlob
from app.models.workspace import Workspace
from app.schemas.land_image_registrations import (
    ImageRegistrationCreate,
    ImageRegistrationRead,
    ImageRegistrationResult,
)
from app.services import land_archive_images as images
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.land_rasters import TMS, transparent_tile


def scoped(db: Session, workspace_id: uuid.UUID, identifier: uuid.UUID) -> LandImageRegistration:
    row = db.get(LandImageRegistration, identifier)
    if row is None:
        raise NotFoundError("image alignment", identifier)
    images.scoped_evidence(db, workspace_id, row.evidence_id)
    return row


def read(row: LandImageRegistration) -> ImageRegistrationRead:
    return ImageRegistrationRead.model_validate(row)


def preview(
    db: Session, workspace_id: uuid.UUID, evidence_id: uuid.UUID, request: ImageRegistrationCreate
) -> ImageRegistrationResult:
    evidence = images.scoped_evidence(db, workspace_id, evidence_id)
    if images.media(evidence).kind != "historical-map":
        raise InvalidInputError("Use a map-sheet image for this affine alignment workflow.")
    image = db.get(LandArchiveImage, evidence_id)
    if image is None:
        raise InvalidInputError("Save the archive image before aligning it.")
    metadata = images.read(image)
    if metadata.sha256 != request.image_sha256:
        raise ConflictError(
            "The control points refer to a different saved image. Reload its snapshot."
        )
    try:
        return fit(request, metadata.width, metadata.height)
    except ValueError as error:
        raise InvalidInputError(str(error)) from error


def existing(
    db: Session, evidence_id: uuid.UUID, request: ImageRegistrationCreate
) -> LandImageRegistration | None:
    identifier = uuid.uuid5(evidence_id, f"image-registration/{request.request_key}")
    row = db.get(LandImageRegistration, identifier)
    if row and row.request != request.model_dump(mode="json"):
        raise ConflictError(
            "This alignment request was already used with different control points."
        )
    return row


def render_image(source: bytes, result: ImageRegistrationResult) -> tuple[bytes, int, int]:
    with tempfile.TemporaryDirectory(prefix="land-registration-") as temporary:
        directory = Path(temporary)
        (directory / "source.png").write_bytes(source)
        (directory / "fit.json").write_text(result.model_dump_json())
        try:
            process = subprocess.run(  # noqa: S603 -- fixed processor with validated data in a generated directory
                [sys.executable, "-m", "app.analysis.image_registration_job", str(directory)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "GDAL_NUM_THREADS": "1",
                },
            )
        except subprocess.TimeoutExpired as error:
            raise InvalidInputError(
                "The image alignment reached its processing time limit."
            ) from error
        metadata_file, output = directory / "result.json", directory / "output.tif"
        if process.returncode or not metadata_file.is_file() or metadata_file.stat().st_size > 4096:
            raise InvalidInputError(
                "The image alignment could not be rendered within its processing limits."
            )
        metadata = json.loads(metadata_file.read_text())
        if "error" in metadata or not output.is_file() or output.stat().st_size > 8 * 1024 * 1024:
            raise InvalidInputError(
                "The image alignment could not be rendered within its processing limits."
            )
        return output.read_bytes(), int(metadata["width"]), int(metadata["height"])


def save(
    db: Session,
    workspace_id: uuid.UUID,
    evidence_id: uuid.UUID,
    request: ImageRegistrationCreate,
    result: ImageRegistrationResult,
    rendered: tuple[bytes, int, int],
    quota_bytes: int,
) -> LandImageRegistration:
    images.scoped_evidence(db, workspace_id, evidence_id)
    db.execute(select(Workspace.id).where(Workspace.id == workspace_id).with_for_update())
    found = existing(db, evidence_id, request)
    if found is not None:
        return found
    data, width, height = rendered
    if images.workspace_used_bytes(db, workspace_id) + len(data) > quota_bytes:
        raise InvalidInputError("This workspace's archive image storage allowance is full.")
    row = LandImageRegistration(
        id=uuid.uuid5(evidence_id, f"image-registration/{request.request_key}"),
        evidence_id=evidence_id,
        request=request.model_dump(mode="json"),
        result=result.model_dump(mode="json"),
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        display_width=width,
        display_height=height,
    )
    db.add(row)
    db.flush()
    db.add(LandImageRegistrationBlob(registration_id=row.id, data=data))
    db.flush()
    return row


def blob(db: Session, row: LandImageRegistration) -> bytes:
    value = db.get(LandImageRegistrationBlob, row.id)
    if value is None:
        raise NotFoundError("aligned image bytes", row.id)
    return value.data


def tile(db: Session, row: LandImageRegistration, z: int, x: int, y: int) -> bytes:
    if not 0 <= z <= 22 or not 0 <= x < 2 ** (z + 1) or not 0 <= y < 2**z:
        raise NotFoundError("aligned image tile", f"{z}/{x}/{y}")
    bounds = TMS.bounds(morecantile.Tile(x, y, z))
    west, south, east, north = row.result["bounds"]
    if bounds.right <= west or bounds.left >= east or bounds.top <= south or bounds.bottom >= north:
        return transparent_tile()
    destination = np.zeros((4, 256, 256), dtype=np.uint8)
    with (
        rasterio.Env(GDAL_CACHEMAX=16 * 1024 * 1024),
        MemoryFile(blob(db, row)) as memory,
        memory.open(driver="GTiff") as dataset,
    ):
        reproject(
            dataset.read(),
            destination,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            src_alpha=4,
            dst_alpha=4,
            dst_transform=from_bounds(*bounds, 256, 256),
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            num_threads=1,
            warp_mem_limit=16,
        )
    output = io.BytesIO()
    Image.fromarray(np.moveaxis(destination, 0, 2)).save(output, format="PNG")
    return output.getvalue()

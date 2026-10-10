"""Fixed bounded raster writer; source image and control-point fit are data only."""

from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.shutil import copy as raster_copy

from app.schemas.land_image_registrations import ImageRegistrationResult

MAX_BYTES = 8 * 1024 * 1024


def build(source: Path, result: ImageRegistrationResult, output: Path) -> tuple[int, int]:
    with Image.open(source) as original:
        if (
            original.format != "PNG"
            or original.size != (result.image_width, result.image_height)
            or original.width * original.height > 8_000_000
        ):
            raise ValueError("The saved image does not match the alignment.")
        original.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        pixels = np.array(original.convert("RGBA"))
    height, width = pixels.shape[:2]
    transform = Affine(*result.transform) * Affine.scale(
        result.image_width / width, result.image_height / height
    )
    with rasterio.Env(GDAL_CACHEMAX=16 * 1024 * 1024), MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=width,
            height=height,
            count=4,
            dtype="uint8",
            crs=result.crs,
            transform=transform,
        ) as dataset:
            dataset.write(np.moveaxis(pixels, 2, 0))
            dataset.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )
            dataset.update_tags(
                algorithm=result.algorithm,
                rms_error_m=result.rms_error_m,
                interpretation="User-matched affine registration; fit residuals do not establish surveyed accuracy.",
            )
        raster_copy(
            memory.name,
            str(output),
            driver="COG",
            compress="DEFLATE",
            blocksize=256,
            overview_resampling="average",
        )
    if output.stat().st_size > MAX_BYTES:
        raise ValueError("The georeferenced image exceeds its output size limit.")
    return width, height


def main() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    directory = Path(sys.argv[1])
    try:
        result = ImageRegistrationResult.model_validate_json((directory / "fit.json").read_text())
        width, height = build(directory / "source.png", result, directory / "output.tif")
        metadata: dict[str, str | int] = {"width": width, "height": height}
    except Exception:
        metadata = {
            "error": "The aligned image could not be prepared within its processing limits."
        }
    (directory / "result.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()

"""Decode only bounded archive previews in a fixed, resource-limited subprocess."""

from __future__ import annotations

import io
import json
import resource
import sys
import warnings
from pathlib import Path

from PIL import Image, ImageOps

MAX_SOURCE_BYTES = 5 * 1024 * 1024
MAX_PREVIEW_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 8_000_000
FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


def normalize(data: bytes) -> tuple[bytes, dict[str, str | int]]:
    if not 0 < len(data) <= MAX_SOURCE_BYTES:
        raise ValueError("Archive previews must be at most 5 MiB.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in FORMATS or getattr(source, "n_frames", 1) != 1:
                raise ValueError("Only single-frame JPEG, PNG and WebP previews are supported.")
            if source.width * source.height > MAX_PIXELS or max(source.size) > 8192:
                raise ValueError(
                    "The archive preview exceeds the 8-megapixel or 8192-pixel side limit."
                )
            media_type = FORMATS[source.format]
            source.load()
            # A canonical orientation and metadata-free RGBA PNG gives later control points
            # an unambiguous pixel coordinate system without discarding the original bytes.
            image = ImageOps.exif_transpose(source).convert("RGBA")
            image.info.clear()
            output = io.BytesIO()
            image.save(output, format="PNG")
            data = output.getvalue()
            if len(data) > MAX_PREVIEW_BYTES:
                raise ValueError("The normalized preview exceeds the 16 MiB limit.")
            return data, {
                "width": image.width,
                "height": image.height,
                "sourceMediaType": media_type,
            }


def main() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_PREVIEW_BYTES, MAX_PREVIEW_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    directory = Path(sys.argv[1])
    try:
        preview, metadata = normalize((directory / "source").read_bytes())
        (directory / "preview.png").write_bytes(preview)
        (directory / "result.json").write_text(json.dumps(metadata))
    except Exception:
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "error": "The archive preview could not be safely decoded within its image limits."
                }
            )
        )


if __name__ == "__main__":
    main()

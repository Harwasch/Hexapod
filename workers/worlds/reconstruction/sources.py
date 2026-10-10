"""Bounded local media sampling and diagnostics; no network or GPU work."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


class SourceError(ValueError):
    pass


def image_pixels(path):
    with Image.open(path) as image:
        if (
            image.width * image.height > 16_000_000
            or getattr(image, "n_frames", 1) != 1
        ):
            raise SourceError("Choose still images below 16 megapixels")
        return ImageOps.exif_transpose(image).convert("RGB")


def prepare_sources(directory: Path, output: Path, max_frames=12, target_size=518):
    if not 2 <= max_frames <= 24 or target_size not in (280, 392, 518):
        raise SourceError("Unsupported bounded reconstruction settings")
    paths = sorted(p for p in directory.iterdir() if p.is_file())
    images = [p for p in paths if p.suffix in (".png", ".jpg", ".jpeg", ".webp")]
    videos = [p for p in paths if p.suffix in (".mp4", ".webm", ".mov")]
    if len(videos) > 1 or (videos and images):
        raise SourceError("Choose images or one replay video, not mixed sources")
    output.mkdir(parents=True, exist_ok=True)
    sampled = []
    if videos:
        import cv2

        video = cv2.VideoCapture(str(videos[0]))
        try:
            count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(video.get(cv2.CAP_PROP_FPS))
            if not video.isOpened() or count < 2 or not np.isfinite(fps) or fps <= 0:
                raise SourceError(
                    "Video has no usable frame count or frame rate; extract frames in the browser"
                )
            duration = count / fps
            if duration > 3600:
                raise SourceError("Select a replay segment shorter than one hour")
            if (
                video.get(cv2.CAP_PROP_FRAME_WIDTH)
                * video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                > 16_000_000
            ):
                raise SourceError("Resize video frames below 16 megapixels")
            for index in np.linspace(0, count - 1, min(max_frames, count), dtype=int):
                video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = video.read()
                if not ok:
                    raise SourceError("A selected replay frame could not be decoded")
                sampled.append(
                    (
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                        float(index / fps * 1000),
                    )
                )
        finally:
            video.release()
    else:
        if len(images) < 2:
            raise SourceError("Reconstruction needs at least two overlapping views")
        for index in np.linspace(
            0, len(images) - 1, min(max_frames, len(images)), dtype=int
        ):
            sampled.append((image_pixels(images[index]), None))
    prepared, sharpness, changes, timestamps = [], [], [], []
    previous = None
    for index, (image, timestamp) in enumerate(sampled):
        # Mirror's documented pad preprocessing keeps the longest edge bounded.
        # This also avoids huge intermediate allocations for extreme aspect ratios.
        width, height = image.size
        factor = target_size / max(width, height)
        scaled_width = max(14, round(width * factor / 14) * 14)
        scaled_height = max(14, round(height * factor / 14) * 14)
        image = image.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)
        padded = Image.new("RGB", (target_size, target_size), "white")
        padded.paste(
            image,
            ((target_size - scaled_width) // 2, (target_size - scaled_height) // 2),
        )
        image = padded
        image.save(output / f"frame-{index:03d}.png")
        array = np.asarray(image).astype(np.float32) / 255
        prepared.append(array)
        gray = np.asarray(image.resize((64, 64)).convert("L"), dtype=np.float32) / 255
        laplacian = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        sharpness.append(float(laplacian.var()))
        if previous is not None:
            changes.append(float(np.abs(gray - previous).mean()))
        previous = gray
        timestamps.append(timestamp)
    diagnostics = {
        "sourceFileCount": len(paths),
        "selectedFrameCount": len(prepared),
        "selection": "uniform-source-order",
        "frameTimestampsMs": timestamps,
        "medianLaplacianVariance": float(np.median(sharpness)),
        "meanAdjacentImageChange": float(np.mean(changes)) if changes else 0,
        "inputAssessment": "heuristics-not-reconstruction-accuracy",
        "warnings": [],
    }
    if changes and max(changes) < 0.01:
        diagnostics["warnings"].append(
            "Views are nearly identical; camera translation may be insufficient."
        )
    if changes and max(changes) > 0.35:
        diagnostics["warnings"].append(
            "Large visual changes may indicate cuts or scene drift."
        )
    (output / "source-diagnostics.json").write_text(json.dumps(diagnostics))
    return np.stack(prepared), diagnostics

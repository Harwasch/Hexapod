"""Render both plain PLY exports with one camera per view in the trainer interpreter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.bench.evaluate import lookup_frames
from experiments.bench.matched_maths import closeup_camera
from experiments.bench.topdown import render

VIEWS = ("ground", "topdown", "closeup")
PNG_NAMES = tuple(
    [f"{label}-{view}.png" for view in VIEWS for label in ("A", "B", "comparison")]
    + ["GT-ground.png", "GT-closeup.png"]
)


def panels(directory: Path, *, synthetic: bool) -> None:
    """Label panes, retaining native pixels (including the native-resolution GT crop)."""
    for view in VIEWS:
        labels = ["A", "B"] + ([] if view == "topdown" else ["GT"])
        images = [Image.open(directory / f"{label}-{view}.png").convert("RGB") for label in labels]
        width = max(im.width for im in images)
        height = max(im.height for im in images)
        canvas = Image.new("RGB", (width * len(images), height + 28), "black")
        draw = ImageDraw.Draw(canvas)
        for index, (label, im) in enumerate(zip(labels, images, strict=True)):
            canvas.paste(im, (index * width, 28))
            suffix = " SYNTHETIC REHEARSAL" if synthetic else ""
            draw.text((index * width + 4, 6), f"{label} / {view}{suffix}", fill="white")
        canvas.save(directory / f"comparison-{view}.png")


def render_pair(settings: dict[str, Any], output: Path) -> None:
    """Dataset's undistorted frozen test camera is shared, never optimized poses."""
    sys.path.insert(0, str(Path(settings["trainer"]).parent))
    from datasets.colmap import Dataset, Parser

    parser = Parser(data_dir=settings["data"], factor=1, normalize=False, test_every=1)
    dataset = Dataset(parser, split="val")
    lookup = lookup_frames(parser.image_names, dataset.indices, [settings["frame"]])
    data = dataset[lookup[settings["frame"]]]
    pixels = data["image"].numpy().astype(np.uint8)
    height, width = pixels.shape[:2]
    ground = {
        "viewmat": np.linalg.inv(data["camtoworld"].numpy()).tolist(),
        "K": data["K"].numpy().tolist(),
        "width": width,
        "height": height,
        "near": 0.01,
        "far": 1e10,
    }
    closeup, crop = closeup_camera(ground)
    cameras = {"ground": ground, "topdown": settings["topdown"], "closeup": closeup}
    gt = Image.fromarray(pixels)
    gt.save(output / "GT-ground.png")
    gt.crop(crop).save(output / "GT-closeup.png")
    for label, source in zip(("A", "B"), settings["sources"], strict=True):
        for view, camera in cameras.items():
            target = output / f"{label}-{view}.png"
            # Both exports use the same plain-3DGS loader, black background, classic
            # renderer and degree. No PPISP controller or per-run pose correction.
            render(Path(source["ply"]), camera, target, settings["degree"], False)
            if "mask" in data and view != "topdown":
                mask = data["mask"].numpy()
                if view == "closeup":
                    left, top, right, bottom = crop
                    mask = mask[top:bottom, left:right]
                rgb = np.array(Image.open(target))
                rgb[~mask.astype(bool)] = 0
                Image.fromarray(rgb).save(target)
    (output / "cameras.json").write_text(
        json.dumps(
            {
                "frame": settings["frame"],
                "crop": list(crop),
                "cameras": cameras,
                "renderer": "classic",
                "shDegree": settings["degree"],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    panels(output, synthetic=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    render_pair(json.loads(args.settings.read_text()), args.out)


if __name__ == "__main__":
    main()

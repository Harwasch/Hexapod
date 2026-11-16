"""Deterministic synthetic-only backends using the committed geometry and depth.

This is a geometry oracle for plumbing tests, not inference. It deliberately needs
sibling ground_truth.json with synthetic=true and never assigns a real species.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from hex_analysis.lifting import lift_depth
from hex_analysis.registry import Detection, Frame


def load_fixture(directory: Path, index: int = 2) -> tuple[Frame, dict[str, Any]]:
    truth = json.loads((directory / "ground_truth.json").read_text())
    if truth.get("synthetic") is not True:
        raise ValueError("fixture backend requires explicitly synthetic ground truth")
    records = [
        json.loads(line) for line in (directory / "bundle/frames.jsonl").read_text().splitlines()
    ]
    if not 0 <= index < len(records):
        raise ValueError("frame index outside bundle")
    record = records[index]
    depth = record["depth"]
    depth_path = (directory / "bundle" / depth["file"]).resolve()
    if depth_path.parent != (directory / "bundle").resolve():
        raise ValueError("depth file must be flat within bundle")
    return Frame(
        index=index,
        image=b"",  # Geometry fixture does not decode RGB or pretend to supply image context.
        depth=np.fromfile(depth_path, dtype="<f2")
        .astype(np.float64)
        .reshape(depth["height"], depth["width"]),
        intrinsics=np.asarray(record["intrinsics"], dtype=np.float64).reshape(3, 3),
        transform=np.asarray(record["transform"], dtype=np.float64).reshape(4, 4),
        image_size=(record["imageWidth"], record["imageHeight"]),
        pixel_centres="scaled-intrinsics",
    ), truth


class FixtureSegmenter:
    def __init__(self, truth: dict[str, Any]) -> None:
        if truth.get("synthetic") is not True:
            raise ValueError("fixture backend requires synthetic geometry")
        self.objects = {obj["label"]: obj for obj in truth["objects"]}

    def segment(self, frame: Frame, prompt: str) -> list[Detection]:
        if frame.depth is None:
            return []
        valid = np.isfinite(frame.depth) & (frame.depth > 0)
        points = lift_depth(
            frame.depth,
            valid,
            frame.intrinsics,
            frame.transform,
            frame.image_size,
            pixel_centres=frame.pixel_centres,
        )
        masks = {}
        # <8 m float16 depth has <=0.002 m axial rounding error. 0.01 m
        # covers propagation through the rays/rotation for these bounded objects.
        epsilon = 0.01
        for label, obj in self.objects.items():
            offset = points - np.asarray(obj["centre"])
            half = np.asarray(obj["extents"]) / 2
            if label == "crown":
                selected = np.abs(np.linalg.norm(offset, axis=1) - half[0]) <= epsilon
            elif label == "trunk":
                radial = np.linalg.norm(offset[:, [0, 2]], axis=1)
                selected = (
                    (radial <= half[0] + epsilon)
                    & (np.abs(offset[:, 1]) <= half[1] + epsilon)
                    & (
                        (np.abs(radial - half[0]) <= epsilon)
                        | (np.abs(np.abs(offset[:, 1]) - half[1]) <= epsilon)
                    )
                )
            else:
                selected = (np.abs(offset) <= half + epsilon).all(axis=1)
            mask = np.zeros_like(valid)
            mask[valid] = selected
            masks[label] = mask
        groups = {"tree": masks["trunk"] | masks["crown"], "shed": masks["shed"]}
        text = prompt.casefold().strip()
        labels = {"plant": ["tree"], "all": ["tree", "shed"]}.get(text, [text])
        return [
            Detection(label, groups[label], 1.0, "fixture", {"synthetic": True})
            for label in labels
            if label in groups and groups[label].any()
        ]


class FixturePlantIdentifier:
    def identify(self, frame: Frame, detection: Detection) -> dict[str, Any]:
        return {"taxon": None, "identifier": "fixture", "identity": "synthetic tree"}

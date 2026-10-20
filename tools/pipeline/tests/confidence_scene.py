"""A field with a tree on it, captured from part of the way round or all of it.

The confidence stage's tests and its demo (`experiments/confidence_demo.py`) build their
epochs here, the way `synthetic_scene.py` builds the quality stage's orbit: a COLMAP model
on disk and a trained splat whose answers are known. The ground is a square of gaussians
at z = 0 with a pond in it (a disc with no gaussians at all: what a capture *should* have
seen and has no evidence for), and the "tree" is a box standing on it, its four sides and
top covered. Positions are already east/north/up metres: `georef.json` places them by the
identity, so the COLMAP frame, `canonical.ply`'s and the tileset's are the same and a cell
of the grid is a cell of this field.

Every capture here is synthetic. Nothing in it is a measurement of a real one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import gaussians
from synthetic_scene import look_at, write_model

#: Where every camera here looks: the tree's middle.
TARGET = np.array([0.0, 0.0, 1.5])
#: The pond: centre (east, north) and radius, metres.
POND = (2.5, -2.0, 0.9)


def field(
    count: int,
    *,
    ground_m: float = 16.0,
    tree_m: tuple[float, float] = (2.0, 3.0),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """A trained splat's columns: ground (with the pond cut out) and the tree box."""
    rng = np.random.default_rng(seed)
    width, height = tree_m
    faces = 4 * width * height + width * width
    area = ground_m * ground_m
    trees = int(count * faces / (faces + area))
    grounds = count - trees
    xy = rng.uniform(-ground_m / 2, ground_m / 2, (grounds * 2, 2))
    px, py, pr = POND
    dry = np.hypot(xy[:, 0] - px, xy[:, 1] - py) > pr
    # Off the tree's footprint too: the box stands on the ground, it is not inside it.
    clear = (np.abs(xy[:, 0]) > width / 2) | (np.abs(xy[:, 1]) > width / 2)
    xy = xy[dry & clear][:grounds]
    ground = np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)
    # The tree: each face sampled in proportion to its area.
    u = rng.uniform(-width / 2, width / 2, trees)
    v = rng.uniform(0, height, trees)
    face = rng.choice(5, size=trees, p=[width * height / faces] * 4 + [width * width / faces])
    half = width / 2
    tree = np.stack(
        [
            np.select([face == 0, face == 1, face == 2, face == 3], [half, -half, u, u], u),
            np.select(
                [face == 0, face == 1, face == 2, face == 3],
                [u, u, half, -half],
                rng.uniform(-half, half, trees),
            ),
            np.where(face == 4, height, v),
        ],
        axis=1,
    )
    xyz = np.concatenate([ground, tree]).astype(np.float32)
    n = xyz.shape[0]
    spacing = math.sqrt(area / max(1, grounds))
    log_scale = np.float32(math.log(0.6 * spacing))
    return {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "f_dc_0": rng.normal(0, 0.5, size=(n,)).astype(np.float32),
        "f_dc_1": rng.normal(0, 0.5, size=(n,)).astype(np.float32),
        "f_dc_2": rng.normal(0, 0.5, size=(n,)).astype(np.float32),
        # alpha ~0.88: opaque enough to occlude, so the tree hides what is behind it.
        "opacity": np.full(n, 2.0, np.float32),
        "scale_0": np.full(n, log_scale, np.float32),
        "scale_1": np.full(n, log_scale, np.float32),
        "scale_2": np.full(n, np.float32(log_scale - 2.0), np.float32),
        "rot_0": np.ones(n, np.float32),
        "rot_1": np.zeros(n, np.float32),
        "rot_2": np.zeros(n, np.float32),
        "rot_3": np.zeros(n, np.float32),
    }


def arc(
    count: int,
    *,
    degrees: float,
    radius: float = 7.0,
    height: float = 4.0,
    facing_deg: float = -90.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`count` cameras over `degrees` of a circle centred on azimuth `facing_deg` (-90:
    the south side), each looking at the tree: (R, centre)."""
    out = []
    for index in range(count):
        t = 0.5 if count == 1 else index / (count - 1)
        azimuth = math.radians(facing_deg + degrees * (t - 0.5))
        centre = np.array([radius * math.cos(azimuth), radius * math.sin(azimuth), height])
        out.append((look_at(centre, TARGET), centre))
    return out


def orbit(
    count: int, *, radius: float = 7.0, height: float = 4.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A full circle of `count` cameras round the tree."""
    out = []
    for index in range(count):
        azimuth = 2 * math.pi * index / count
        centre = np.array([radius * math.cos(azimuth), radius * math.sin(azimuth), height])
        out.append((look_at(centre, TARGET), centre))
    return out


def obliques(
    count: int, *, radius: float = 4.0, height: float = 9.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`count` steep views from above, offset by half a step from `orbit`'s."""
    out = []
    for index in range(count):
        azimuth = 2 * math.pi * (index + 0.5) / count
        centre = np.array([radius * math.cos(azimuth), radius * math.sin(azimuth), height])
        out.append((look_at(centre, TARGET), centre))
    return out


def write_epoch(
    root: Path,
    columns: dict[str, np.ndarray],
    cameras: list[tuple[np.ndarray, np.ndarray]],
    *,
    prior_epochs: int | None = None,
    alignment: dict[str, Any] | None = None,
    uncertainty_m: float = 5.0,
) -> list[str]:
    """An epoch's recipe inputs under `root` (a workdir's `inputs/`): `trained.ply`,
    `poses`, `georef.json` (the identity, metric), `site_context/` (with `priorEpochs`
    when given) and, when given, `epoch_alignment.json`. Returns the input names."""
    root.mkdir(parents=True, exist_ok=True)
    gaussians.write_ply(root / "trained.ply", columns)
    write_model(root / "poses", cameras)
    frame = {
        "source": "exif-gps-similarity",
        "scale": 1.0,
        "rotation": np.eye(3).tolist(),
        "translationM": [0.0, 0.0, 0.0],
        "recentre": False,
    }
    georef = {
        "lat": 46.84,
        "lon": -91.99,
        "height": 180.0,
        "frame": frame,
        "scaleSource": "exif-gps",
        "uncertaintyM": uncertainty_m,
    }
    (root / "georef.json").write_text(json.dumps(georef), encoding="utf-8")
    context: dict[str, Any] = {"version": 1, "siteId": None, "reference": None}
    if prior_epochs is not None:
        context["priorEpochs"] = prior_epochs
    (root / "site_context").mkdir(exist_ok=True)
    (root / "site_context" / "site_context.json").write_text(json.dumps(context), encoding="utf-8")
    names = ["trained.ply", "poses", "georef.json", "site_context"]
    if alignment is not None:
        (root / "epoch_alignment.json").write_text(json.dumps(alignment), encoding="utf-8")
        names.append("epoch_alignment.json")
    return names


def alignment_of(uncertainty_m: float, verdict: str = "coarse") -> dict[str, Any]:
    """An `epoch_alignment.json` as `align` writes it for an epoch it kept in its
    georeference's frame (`frameSource: georef`, so `place` is unchanged by it), with the
    uncertainty asked for."""
    return {
        "version": 1,
        "method": "visual" if verdict != "unaligned" else "none",
        "verdict": verdict,
        "uncertaintyM": uncertainty_m,
        "label": f"synthetic, ±{uncertainty_m:.2g} m",
        "frameSource": "georef",
        "frame": None,
    }

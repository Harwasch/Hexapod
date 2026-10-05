"""The backfill's estimate is the georeference stage's, on the same pose model.

`tools/captures/estimate_scale.py` sizes a scan registered before the pipeline estimated
scales, from the pose model its run left, and the asset is then drawn at that size. It is
only worth anything if it is the number `exif_gps` would have written had the run happened
today: the same reader, the same levelling, the same estimate. This holds it to that, on a
phone orbit with a known truth, through `stages._frame_from_poses` itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

import captures_bridge  # noqa: F401 - puts tools/captures on the path
import sfm
import stages
import synthetic_scene
from contracts import StageContext

METRES_PER_UNIT = 0.4
HELD_AT_M = 1.5


class _Defaults:
    """What `_frame_from_poses` asks a stage context: its params, none of them given."""

    def param(self, name: str, default: object = None) -> object:
        return default


def _orbit(root: Path) -> Path:
    """Thirty-six cameras round a table-top subject at `HELD_AT_M`, over a floor out to
    4 m, written as COLMAP would: y down, in units of `METRES_PER_UNIT`."""
    rng = np.random.default_rng(5)
    cameras = synthetic_scene.ring(36, radius=2.0, height=HELD_AT_M, target=np.array([0, 0, 0.6]))
    radius, angle = 4.0 * np.sqrt(rng.random(20_000)), 2 * np.pi * rng.random(20_000)
    ground = np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(20_000)], 1)
    subject = np.stack(
        [rng.normal(0, 0.2, 8_000), rng.normal(0, 0.2, 8_000), rng.uniform(0.7, 1.0, 8_000)], 1
    )
    to_model = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    rig = [
        (rotation @ to_model.T, to_model @ centre / METRES_PER_UNIT) for rotation, centre in cameras
    ]
    points = np.concatenate([ground, subject]) @ to_model.T / METRES_PER_UNIT
    poses = synthetic_scene.write_model(root / "poses", rig, width=32, height=24, focal=30.0)
    count = points.shape[0]
    sfm.write_points3d(
        poses / "points3D.bin",
        sfm.Points3D(
            ids=np.arange(1, count + 1, dtype=np.uint64),
            xyz=points,
            rgb=np.full((count, 3), 128, dtype=np.uint8),
            error=np.full(count, 0.5),
            track=np.zeros((0, 2), dtype=np.uint32),
            track_offsets=np.zeros(count + 1, dtype=np.int64),
        ),
    )
    return poses


def test_the_backfill_estimates_what_the_georeference_stage_estimates(tmp_path: Path) -> None:
    # tools/captures, importable once `captures_bridge` has put it on the path -- which an
    # import sorted above that one would not see.
    import estimate_scale

    poses = _orbit(tmp_path)

    frame = stages._frame_from_poses(cast(StageContext, _Defaults()), poses, estimate_scale=True)
    found = estimate_scale.estimate(poses)

    assert frame["scaleSource"] == "camera-height-estimate"
    assert found.estimate is not None
    assert found.estimate.scale == pytest.approx(float(cast(float, frame["scale"])), rel=1e-12)
    assert found.estimate.scale == pytest.approx(METRES_PER_UNIT, rel=0.03)
    assert found.estimate.to_dict() == frame["scaleEstimate"]
    assert found.up == pytest.approx(cast(dict[str, object], frame["up"])["up"], abs=1e-6)

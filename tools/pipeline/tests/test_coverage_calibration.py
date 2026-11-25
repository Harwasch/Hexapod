"""Numpy-only end-to-end calibration rehearsal, with deliberately planted image error."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments.coverage_quality.calibration import (
    COLUMNS,
    fit_credit,
    patch_table,
    world_points,
    write_outputs,
)
from experiments.coverage_quality.geometry import credit, unit
from experiments.coverage_quality.gpu import make_dataset, parser, run
from experiments.coverage_quality.renderers import Camera, NumpyBackend, Scene


def camera(name: str, x: float) -> Camera:
    centre = np.array([x, 0.0, 0.0])
    forward = unit(np.array([0.0, 0.0, 3.0]) - centre)
    right = unit(np.cross([0, 1, 0], forward))
    up = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack([right, up, forward])
    c2w[:3, 3] = centre
    return Camera(name, c2w, np.array([[65.0, 0, 32], [0, 65.0, 24], [0, 0, 1]]), 64, 48)


def fixture() -> Scene:
    x, y = np.meshgrid(np.linspace(-0.7, 0.7, 20), np.linspace(-0.5, 0.5, 15))
    xyz = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 3.0)])
    scales = np.tile([0.055, 0.045, 0.008], (len(xyz), 1))
    quats = np.tile([1.0, 0, 0, 0], (len(xyz), 1))
    colors = np.tile([0.3, 0.35, 0.4], (len(xyz), 1))
    return Scene(xyz, scales, quats, np.full(len(xyz), 0.95), colors)


def test_cpu_end_to_end_recovers_planted_view_angle_relation(tmp_path: Path) -> None:
    scene, backend = fixture(), NumpyBackend()
    training = [camera(f"train{i}", x) for i, x in enumerate([-0.5, -0.25, 0, 0.25, 0.5])]
    tests = []
    for name, x in (("test0", 0.9), ("test1", 1.5)):
        cam = camera(name, x)
        rendered = backend.render(scene, cam)
        points = world_points(cam, rendered.depth)
        directions = unit(np.array([c.centre for c in training])[:, None, None, :] - points)
        test_direction = unit(cam.centre - points)
        nearest = np.degrees(np.arccos(np.clip((directions * test_direction).sum(-1), -1, 1))).min(
            0
        )
        # Image-space ground truth encodes a known angular error law. No patch errors
        # are injected into the output table; both L1 and SSIM must run through images.
        truth = rendered.rgb + (0.01 + 0.002 * nearest)[..., None]
        tests.append((cam, truth, rendered.alpha > 0.7))
    rows = patch_table(scene, training, tests, backend, max_patches_per_view=512)
    assert len(rows) >= 40
    assert all(set(row) == set(COLUMNS) for row in rows)
    assert all(row["lpips"] is None and row["blur"] is None for row in rows)
    assert all(0 < row["observing_views"] <= 5 for row in rows)
    assert all(row["best_mm_px"] > 0 and row["parallax_deg"] > 0 for row in rows)
    fit = fit_credit(rows, "l1")
    assert fit["angle_baseline"]["slope_per_deg"] == pytest.approx(0.002, abs=0.0003)
    assert fit["angle_baseline"]["intercept"] == pytest.approx(0.01, abs=0.004)
    assert fit["angle_baseline"]["r2"] > 0.93
    assert fit["r2"] > 0.9
    write_outputs(rows, tmp_path)
    assert (tmp_path / "patches.csv").read_text().splitlines()[0].split(",") == list(COLUMNS)
    assert json.loads((tmp_path / "fit.json").read_text())["l1"]["rows"] == len(rows)


def test_alpha_compositing_depth_and_anisotropic_footprint() -> None:
    cam = camera("centre", 0)
    scene = Scene(
        np.array([[0.0, 0, 2], [0.0, 0, 3]]),
        np.tile([0.12, 0.025, 0.01], (2, 1)),
        np.tile([1.0, 0, 0, 0], (2, 1)),
        np.array([0.5, 0.5]),
        np.array([[1.0, 0, 0], [0.0, 0, 1]]),
    )
    result = NumpyBackend().render(scene, cam)
    assert result.rgb[24, 32] == pytest.approx([0.5, 0, 0.25])
    assert result.alpha[24, 32] == pytest.approx(0.75)
    assert result.depth[24, 32] == pytest.approx((0.5 * 2 + 0.25 * 3) / 0.75)
    assert result.alpha[24, 35] > result.alpha[27, 32]


def test_credit_and_all_frame_model_guard(tmp_path: Path) -> None:
    assert credit(np.array([1.0, np.cos(np.radians(25)), 0.0]), 25, 2) == pytest.approx([1, 0, 0])
    args = parser().parse_args(
        [
            "--dataset",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--trainer",
            str(tmp_path / "trainer.py"),
            "--trained-test-every",
            "0",
            "--metres-per-unit",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="all-frame model"):
        run(args)  # Must reject BEFORE importing torch/gsplat or constructing a Modal app.
    with pytest.raises(ValueError, match="eight"):
        fit_credit([])


def test_all_frame_dataset_preserves_every_camera_without_modulo_zero() -> None:
    data = SimpleNamespace(test_every=0, image_names=["a", "b", "c"])

    def upstream(parser: Any, split: str) -> Any:
        assert parser.test_every > 0
        indices = np.arange(len(parser.image_names))
        return SimpleNamespace(indices=indices[indices % parser.test_every != 0])

    assert make_dataset(data, upstream, "train").indices.tolist() == [0, 1, 2]
    assert make_dataset(data, upstream, "val").indices.tolist() == []
    assert data.test_every == 0

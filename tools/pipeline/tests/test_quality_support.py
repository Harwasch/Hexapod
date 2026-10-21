"""`support/`: the quality stage keeps every trained gaussian's evidence for `confidence`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import gaussians
import holdout_maths
import quality
import sfm
import splat_io
from captures_bridge import sigmoid
from conftest import make_recipe
from executor import execute
from runners import RunnerSet
from synthetic_scene import ring, scene, write_model
from workdir import Workdir

N = 12_000


def _seed(workdir: Workdir, *, held: bool = False, metric: bool = False) -> dict[str, Any]:
    columns, _ = scene(N, seed=3)
    trained = workdir.input_path("trained.ply")
    trained.parent.mkdir(parents=True, exist_ok=True)
    gaussians.write_ply(trained, columns)
    write_model(workdir.input_path("poses"), ring(20, radius=1.5, height=1.0))
    frame = {
        "source": "exif-gps-similarity" if metric else "camera-up",
        "scale": 2.0 if metric else 1.0,
        "rotation": np.eye(3).tolist(),
        "translationM": [0.0, 0.0, 0.0] if metric else None,
        "recentre": not metric,
    }
    georef: dict[str, Any] = {"lat": 1.0, "lon": 2.0, "frame": frame}
    if metric:
        georef["scaleSource"] = "exif-gps"
    workdir.input_path("georef.json").write_text(json.dumps(georef))
    inputs = ["trained.ply", "poses", "georef.json"]
    if held:
        rng = np.random.default_rng(5)
        error = np.full(N, 0.01, dtype=np.float32)
        error[rng.random(N) < 0.1] = 0.9  # well past twice the median: demoted from keep
        weight = np.full(N, 10.0, dtype=np.float32)
        holdout_maths.write(
            workdir.input_path("holdout"), error, weight, None, {"status": "ok", "views": 2}
        )
        inputs.append("holdout")
    return {"columns": columns, "inputs": inputs, "georef": georef}


def _run(workdir: Workdir, inputs: list[str], params: dict[str, Any]) -> Path:
    execute(
        make_recipe([{"id": "quality", "impl": "support_gate", "params": params}], inputs=inputs),
        workdir,
        RunnerSet.local(),
    )
    return workdir.out_dir("quality")


def _read(folder: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    document = json.loads((folder / quality.SUPPORT_FILE).read_text())
    columns = {
        name: splat_io.Column(folder / spec["file"], spec["dtype"]).read(0, document["rows"])
        for name, spec in document["columns"].items()
    }
    return document, columns


def test_support_has_a_row_for_every_trained_gaussian_with_its_final_tier(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seeded = _seed(workdir)

    out = _run(workdir, seeded["inputs"], {"bar": "strict", "chunk_gaussians": 4099})

    document, columns = _read(out / "support")
    assert document["rows"] == N and document["frame"] == "colmap"
    assert document["source"] == "trained.ply" and document["status"] == "measured"
    assert sorted(name for name, _ in quality.SUPPORT_COLUMNS) == sorted(document["columns"])
    for name, values in columns.items():
        assert values.shape == (N,), name
        assert (out / "support" / f"{name}.bin").stat().st_size == N * values.dtype.itemsize
    trained = seeded["columns"]
    for axis in "xyz":
        np.testing.assert_array_equal(columns[axis], trained[axis])
    # The tiers are the stage's final ones: strict publishes exactly the keep rows, in order.
    gated = gaussians.read_splat(out / "gated.ply")
    keep = columns["tier"] == quality.TIER_KEEP
    np.testing.assert_array_equal(gated.xyz, np.stack([columns[a] for a in "xyz"], 1)[keep])
    counts = json.loads((out / "quality.json").read_text())["gaussians"]
    for tier, name in ((0, "drop"), (1, "context"), (2, "keep")):
        assert int((columns["tier"] == tier).sum()) == counts[name], name
    # Keep rows carry no reasons; every visible row that is not keep carries one.
    assert not columns["reasons"][keep].any()
    visible = columns["alpha"] >= 13
    assert (columns["reasons"][~keep & visible] > 0).all()
    assert np.all(columns["alpha"] == np.rint(sigmoid(trained["opacity"]) * 255))
    # No metric scale: no millimetres.
    assert np.isnan(columns["gsdMm"].astype(np.float64)).all()
    assert document["metresPerUnit"] is None


def test_the_reason_bits_are_the_tier_rules_failed_by_brute_force(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seeded = _seed(workdir)
    out = _run(workdir, seeded["inputs"], {"bar": "balanced"})
    document, columns = _read(out / "support")

    # Measure again, whole and independently of the stage's columns...
    trained = seeded["columns"]
    xyz = np.stack([trained[a] for a in "xyz"], axis=1)
    alpha = sigmoid(trained["opacity"]).astype(np.float32)
    cameras = quality.Cameras.from_model(sfm.read_model(workdir.input_path("poses")))
    measured = quality.measure_support(xyz, alpha, cameras, footprint=quality._footprint(trained))
    np.testing.assert_array_equal(columns["views"], np.clip(measured.views, 0, 65535))
    np.testing.assert_array_equal(columns["spreadDeg"], np.rint(measured.spread_deg))
    th = quality.Thresholds.from_params(document["thresholds"])
    median = np.float32(document["medianGsd"])
    ratio = measured.gsd / median
    roi = document["roi"]
    radii = np.linalg.norm(xyz.astype(np.float64) - roi["center"], axis=1) / roi["radius"]
    tiers = quality.assign_tiers(
        measured.views, measured.spread_deg, ratio, alpha, th, None, radii.astype(np.float32)
    )
    np.testing.assert_array_equal(columns["tier"], tiers)

    # ...and judge every row by the rules as written, one at a time.
    def bits(row: int) -> int:
        if tiers[row] == quality.TIER_KEEP:
            return 0
        out = 0
        if not measured.views[row] >= th.keep_min_views:
            out |= 1
        if not measured.spread_deg[row] >= th.keep_min_spread_deg:
            out |= 2
        if not ratio[row] <= th.keep_max_gsd_ratio:
            out |= 4
        return out

    expected = np.array([bits(row) for row in range(N)], dtype=np.uint8)
    np.testing.assert_array_equal(columns["reasons"], expected)
    # The scene really exercises the bits: the fringe falls short on more than one rule.
    for bit in (1, 2, 4):
        assert int(((columns["reasons"] & bit) > 0).sum()) > 0, bit


def test_a_gaussian_demoted_by_its_held_out_error_says_accuracy(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seeded = _seed(workdir, held=True)
    out = _run(workdir, seeded["inputs"], {"bar": "balanced"})
    document, columns = _read(out / "support")

    assert document["heldOut"] == "used"
    demoted = (columns["reasons"] & 8) > 0
    assert int(demoted.sum()) > 100
    # Demoted from keep, so context, and it failed nothing else.
    assert (columns["tier"][demoted] == quality.TIER_CONTEXT).all()
    assert (columns["reasons"][demoted] == 8).all()
    held = json.loads((out / "quality.json").read_text())["heldOut"]
    assert int(demoted.sum()) == held["demotedFromKeep"]


def test_a_metric_capture_keeps_its_gsd_in_millimetres(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seeded = _seed(workdir, metric=True)
    out = _run(workdir, seeded["inputs"], {"bar": "balanced"})
    document, columns = _read(out / "support")

    assert document["metresPerUnit"] == 2.0
    trained = seeded["columns"]
    xyz = np.stack([trained[a] for a in "xyz"], axis=1)
    cameras = quality.Cameras.from_model(sfm.read_model(workdir.input_path("poses")))
    measured = quality.measure_support(
        xyz, sigmoid(trained["opacity"]), cameras, footprint=quality._footprint(trained)
    )
    seen = measured.views > 0
    expected = (measured.gsd * np.float32(2000.0)).astype(np.float16)
    np.testing.assert_array_equal(columns["gsdMm"][seen], expected[seen])
    assert np.isinf(columns["gsdMm"][~seen].astype(np.float64)).all()


@pytest.mark.parametrize("bar", ["strict", "everything"])
def test_support_does_not_depend_on_the_bar(tmp_path: Path, bar: str) -> None:
    """The grid is built from every row whatever the bar published: `strict` keeps only
    keep in gated.ply, and a confidence map of that would call every published cell strong."""
    workdir = Workdir.create(tmp_path / bar)
    seeded = _seed(workdir)
    out = _run(workdir, seeded["inputs"], {"bar": bar})
    reference = Workdir.create(tmp_path / f"{bar}-balanced")
    _seed(reference)
    other = _run(reference, seeded["inputs"], {"bar": "balanced"})
    for name, _ in quality.SUPPORT_COLUMNS:
        assert (out / "support" / f"{name}.bin").read_bytes() == (
            other / "support" / f"{name}.bin"
        ).read_bytes(), name

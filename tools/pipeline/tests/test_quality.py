"""The quality bar: its tier rules, what it measures, and the stage around them."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import gaussians
import quality
import sfm
from conftest import make_recipe
from executor import execute
from runners import RunnerSet
from synthetic_scene import TARGET, look_at, ring, scene, write_model
from workdir import Workdir

TH = quality.Thresholds()


# --- the tier rules, which are pure ----------------------------------------------------


def _tiers(
    views: list[int], spread: list[float], ratio: list[float], alpha: list[float]
) -> list[int]:
    tiers = quality.assign_tiers(
        np.asarray(views, dtype=np.int32),
        np.asarray(spread, dtype=np.float32),
        np.asarray(ratio, dtype=np.float32),
        np.asarray(alpha, dtype=np.float32),
        TH,
    )
    return [int(tier) for tier in tiers]


def test_keep_needs_every_criterion_at_its_threshold() -> None:
    keep, context, drop = quality.TIER_KEEP, quality.TIER_CONTEXT, quality.TIER_DROP
    assert _tiers([8], [45.0], [2.0], [0.5]) == [keep]
    # One short on each criterion in turn is context, not keep.
    assert _tiers([7, 8, 8], [45.0, 44.9, 45.0], [2.0, 2.0, 2.01], [0.5] * 3) == [context] * 3
    # And short of the context rule too is drop.
    assert _tiers([2, 3, 3], [90.0, 14.9, 90.0], [1.0, 1.0, 6.01], [0.5] * 3) == [drop] * 3


def test_a_near_transparent_gaussian_is_dropped_however_well_it_was_seen() -> None:
    assert _tiers([90], [180.0], [1.0], [0.01]) == [quality.TIER_DROP]


def test_unseen_gaussians_are_dropped() -> None:
    assert _tiers([0], [0.0], [float("inf")], [1.0]) == [quality.TIER_DROP]


def test_an_absolute_gsd_ceiling_applies_only_when_given_a_metric_gsd() -> None:
    strict = quality.Thresholds(keep_max_gsd_mm=1.0)
    args = (
        np.asarray([10], dtype=np.int32),
        np.asarray([90.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        strict,
    )
    assert quality.assign_tiers(*args).tolist() == [quality.TIER_KEEP]
    coarse = np.asarray([2.5], dtype=np.float32)
    assert quality.assign_tiers(*args, gsd_mm=coarse).tolist() == [quality.TIER_CONTEXT]


def test_context_is_only_kept_near_the_subject() -> None:
    args = (
        np.asarray([4, 4, 20], dtype=np.int32),
        np.asarray([20.0, 20.0, 90.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
        TH,
    )
    radii = np.asarray([1.5, 3.0, 9.0], dtype=np.float32)
    tiers = quality.assign_tiers(*args, roi_radii=radii).tolist()
    # Near context stays; far context goes; far *keep* is kept, since it earned it.
    assert tiers == [quality.TIER_CONTEXT, quality.TIER_DROP, quality.TIER_KEEP]


def test_thresholds_come_from_stage_params_by_name() -> None:
    parsed = quality.Thresholds.from_params({"keep_min_views": "12", "keep_min_spread_deg": 30})
    assert parsed.keep_min_views == 12
    assert parsed.keep_min_spread_deg == 30.0
    assert parsed.keep_max_gsd_mm is None
    assert parsed.context_min_views == TH.context_min_views


# --- the gate --------------------------------------------------------------------------


def _columns(n: int) -> dict[str, np.ndarray]:
    columns, _ = scene(n)
    return columns


def test_strict_keeps_only_keep_and_everything_keeps_everything() -> None:
    columns = _columns(6)
    tiers = np.asarray([2, 1, 0, 2, 1, 0], dtype=np.uint8)

    strict = quality.gate(columns, tiers, "strict")
    everything = quality.gate(columns, tiers, "everything")

    assert strict["x"].tolist() == columns["x"][[0, 3]].tolist()
    assert set(strict) == set(gaussians.CANONICAL_PROPERTIES)
    for name in gaussians.CANONICAL_PROPERTIES:
        np.testing.assert_array_equal(everything[name], columns[name])


def test_balanced_keeps_context_at_a_fraction_of_its_opacity() -> None:
    columns = _columns(3)
    tiers = np.asarray([2, 1, 0], dtype=np.uint8)

    balanced = quality.gate(columns, tiers, "balanced", context_fade=0.5)

    alpha = 1.0 / (1.0 + np.exp(-balanced["opacity"].astype(np.float64)))
    original = 1.0 / (1.0 + np.exp(-2.0))
    assert balanced["x"].shape[0] == 2
    assert alpha[0] == pytest.approx(original, rel=1e-6)
    assert alpha[1] == pytest.approx(original * 0.5, rel=1e-5)


def test_an_unknown_bar_is_refused_with_the_known_ones() -> None:
    with pytest.raises(ValueError, match="strict, balanced, everything"):
        quality.gate(_columns(1), np.zeros(1, dtype=np.uint8), "lenient")


# --- the region of interest ------------------------------------------------------------


def test_the_roi_is_where_an_orbit_points() -> None:
    target = np.array([0.3, -0.2, 0.5])
    cams = ring(36, radius=2.0, height=1.2, target=target)
    centres = np.stack([c for _, c in cams])
    axes = np.stack([r[2] for r, _ in cams])

    roi = quality.estimate_roi(centres, axes, radius_factor=0.5)

    assert roi.method == "optical-axes"
    np.testing.assert_allclose(roi.centre, target, atol=1e-6)
    distance = float(np.median(np.linalg.norm(centres - target, axis=1)))
    assert roi.radius == pytest.approx(0.5 * distance)
    assert roi.to_dict()["frame"] == "colmap"


def test_a_walk_with_parallel_axes_falls_back_to_the_median_depth() -> None:
    centres = np.stack([np.array([x, 0.0, 1.0]) for x in np.linspace(0, 5, 20)])
    axes = np.tile(np.array([0.0, 1.0, 0.0]), (20, 1))

    roi = quality.estimate_roi(centres, axes, np.full(20, 3.0))

    assert roi.method == "median-depth"
    np.testing.assert_allclose(roi.centre, [2.5, 3.0, 1.0], atol=1e-9)


# --- what is measured ------------------------------------------------------------------


def _cameras(tmp_path: Path, cams: list[tuple[np.ndarray, np.ndarray]]) -> quality.Cameras:
    return quality.Cameras.from_model(sfm.read_model(write_model(tmp_path / "poses", cams)))


def test_views_count_only_the_cameras_that_are_not_blocked(tmp_path: Path) -> None:
    """An opaque wall between a point and half the ring hides it from that half."""
    cams = ring(16, radius=3.0, height=0.0, target=np.zeros(3))
    cameras = _cameras(tmp_path, cams)
    # The point at the origin, and a wall of opaque gaussians at x = 0.5 facing +x.
    ys, zs = np.meshgrid(np.linspace(-1.5, 1.5, 60), np.linspace(-1.5, 1.5, 60))
    wall = np.stack([np.full(ys.size, 0.5), ys.ravel(), zs.ravel()], axis=1)
    xyz = np.concatenate([[[0.0, 0.0, 0.0]], wall]).astype(np.float32)
    alpha = np.full(xyz.shape[0], 0.9, dtype=np.float32)
    alpha[0] = 0.1  # the point itself is not an occluder

    # Gaussians 5 cm across on a 5 cm grid: a surface, not a sieve of centres.
    footprint = np.full(xyz.shape[0], 0.05, dtype=np.float32)
    support = quality.measure_support(xyz, alpha, cameras, footprint=footprint)
    without = quality.measure_support(xyz, alpha, cameras)

    centres = cameras.centres
    # Centres alone leave holes a point behind the wall is seen through.
    assert without.views[0] > support.views[0]
    blocked = centres[:, 0] > 0.6  # cameras whose line of sight crosses the wall
    assert support.views[0] == int((~blocked).sum())
    assert 0 < support.views[0] < 16


def test_spread_is_the_widest_pair_of_viewing_directions(tmp_path: Path) -> None:
    """Checked against the brute-force maximum over every pair of seeing cameras."""
    cams = ring(24, radius=2.0, height=1.0)
    cameras = _cameras(tmp_path, cams)
    columns, _ = scene(3000, fringe_share=0.3, seed=3)
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    alpha = np.full(xyz.shape[0], 0.1, dtype=np.float32)  # nothing occludes: all see all

    support = quality.measure_support(xyz, alpha, cameras)

    rng = np.random.default_rng(0)
    for i in rng.choice(xyz.shape[0], 40, replace=False):
        seen = []
        for j in range(cameras.count):
            local = cameras.rotations[j] @ xyz[i] + cameras.translations[j]
            if local[2] <= 0:
                continue
            u = cameras.fx[j] * local[0] / local[2] + cameras.cx[j]
            v = cameras.fy[j] * local[1] / local[2] + cameras.cy[j]
            if 0 <= u < cameras.width[j] and 0 <= v < cameras.height[j]:
                d = cameras.centres[j] - xyz[i]
                seen.append(d / np.linalg.norm(d))
        assert support.views[i] == len(seen)
        exact = 0.0
        for a in seen:
            for b in seen:
                exact = max(exact, float(np.degrees(np.arccos(np.clip(a @ b, -1, 1)))))
        # The farthest-point estimate: never above the truth, never below half of it,
        # and on an orbit within a few degrees.
        assert exact / 2 - 1e-3 <= support.spread_deg[i] <= exact + 1e-3
        assert support.spread_deg[i] == pytest.approx(exact, abs=5.0)


def test_gsd_is_depth_over_focal_in_the_closest_view(tmp_path: Path) -> None:
    near = look_at(np.array([0.0, -1.0, 0.0]), np.zeros(3))
    far = look_at(np.array([0.0, 4.0, 0.0]), np.zeros(3))
    cameras = _cameras(
        tmp_path, [(near, np.array([0.0, -1.0, 0.0])), (far, np.array([0.0, 4.0, 0.0]))]
    )
    xyz = np.zeros((1, 3), dtype=np.float32)

    support = quality.measure_support(xyz, np.full(1, 0.1, np.float32), cameras)

    assert support.views[0] == 2
    assert support.gsd[0] == pytest.approx(1.0 / 1300.0, rel=1e-5)
    assert support.spread_deg[0] == pytest.approx(180.0, abs=0.1)


def test_non_finite_positions_are_never_seen(tmp_path: Path) -> None:
    cameras = _cameras(tmp_path, ring(8, radius=2.0, height=0.5))
    xyz = np.asarray([[np.nan, 0, 0], [0.0, 0.0, 0.4]], dtype=np.float32)

    support = quality.measure_support(xyz, np.full(2, 0.1, np.float32), cameras)

    assert support.views[0] == 0
    assert np.isinf(support.gsd[0])
    assert support.views[1] == 8


# --- coverage.ply ----------------------------------------------------------------------


def test_coverage_round_trips_and_is_coloured_by_tier(tmp_path: Path) -> None:
    xyz = np.asarray([[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    tiers = np.asarray([2, 1, 0, 3], dtype=np.uint8)
    path = tmp_path / "coverage.ply"

    quality.write_coverage(path, xyz, tiers)
    back, back_tiers = quality.read_coverage(path)

    np.testing.assert_array_equal(back, xyz)
    np.testing.assert_array_equal(back_tiers, tiers)
    # Any PLY reader sees x/y/z and an sRGB colour per point, the tier's.
    raw = path.read_bytes()
    assert b"property uchar red\nproperty uchar green\nproperty uchar blue\n" in raw
    body = np.frombuffer(raw[raw.index(b"end_header\n") + 11 :], dtype=np.uint8).reshape(4, 16)
    assert [tuple(row[12:15]) for row in body] == [quality.TIER_COLOURS[t] for t in tiers.tolist()]


# --- the stage, and `place` after it ---------------------------------------------------


def _seed(workdir: Workdir, *, n: int, cameras: int, height: float = 1.0) -> np.ndarray:
    columns, fringe = scene(n)
    trained = workdir.input_path("trained.ply")
    trained.parent.mkdir(parents=True, exist_ok=True)
    gaussians.write_ply(trained, columns)
    write_model(workdir.input_path("poses"), ring(cameras, height=height))
    workdir.input_path("train_metrics.json").write_text(json.dumps({"psnr": 23.04}))
    frame = {
        "source": "camera-up",
        "scale": 1.0,
        "rotation": np.eye(3).tolist(),
        "translationM": None,
        "recentre": True,
    }
    workdir.input_path("georef.json").write_text(
        json.dumps({"lat": 1.0, "lon": 2.0, "frame": frame})
    )
    return fringe


def _run(workdir: Workdir, params: dict[str, object] | None = None) -> None:
    execute(
        make_recipe(
            [
                {"id": "quality", "impl": "support_gate", "params": params or {}},
                {"id": "place", "impl": "place_splat"},
            ],
            inputs=["trained.ply", "poses", "georef.json", "train_metrics.json"],
        ),
        workdir,
        RunnerSet.local(),
    )


def test_the_stage_gates_the_fringe_and_place_publishes_what_it_kept(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    fringe = _seed(workdir, n=20_000, cameras=24)

    _run(workdir, {"bar": "strict", "mode": "preview"})

    document = json.loads((workdir.out_dir("quality") / "quality.json").read_text())
    assert document["mode"] == "preview"
    assert document["barApplied"] == "strict"
    np.testing.assert_allclose(document["roi"]["center"], TARGET, atol=1e-6)
    assert document["roi"]["frame"] == "colmap"
    assert document["heldOutPsnr"] == 23.04
    assert document["gsd"]["metric"] is False
    counts = document["gaussians"]
    assert counts["keep"] + counts["context"] + counts["drop"] == counts["in"] == 20_000
    # The object is kept, the ground past the ring is not.
    assert counts["keep"] >= 0.9 * int((~fringe).sum())
    assert counts["out"] == counts["keep"]
    assert document["keepPct"] > 90

    gated = gaussians.read_splat(workdir.out_dir("quality") / "gated.ply")
    placed = gaussians.read_splat(workdir.out_dir("place") / "canonical.ply")
    assert gated.count == placed.count == counts["keep"]
    # Almost none of the ground past the ring got through (a sliver at its inner edge,
    # seen across the table by most of the ring, is honestly well observed).
    outside = np.hypot(gated.xyz[:, 0], gated.xyz[:, 1]) > 1.8
    assert int(outside.sum()) < 0.05 * int(fringe.sum())

    # coverage_enu.ply is coverage.ply moved exactly as the splat was.
    coverage, tiers = quality.read_coverage(workdir.out_dir("quality") / "coverage.ply")
    placed_cov, placed_tiers = quality.read_coverage(workdir.out_dir("place") / "coverage_enu.ply")
    np.testing.assert_array_equal(tiers, placed_tiers)
    assert int((tiers == quality.TIER_CAMERA).sum()) == 24
    kept_cov = placed_cov[placed_tiers == quality.TIER_KEEP]
    shift = placed.xyz.mean(axis=0) - gated.xyz.mean(axis=0)
    source_kept = coverage[tiers == quality.TIER_KEEP]
    np.testing.assert_allclose(kept_cov, source_kept + shift, atol=1e-4)


def test_everything_passes_the_splat_through_untouched(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    _seed(workdir, n=5_000, cameras=16)

    _run(workdir, {"bar": "everything"})

    trained = gaussians.read_splat(workdir.input_path("trained.ply"))
    gated = gaussians.read_splat(workdir.out_dir("quality") / "gated.ply")
    for name in gaussians.CANONICAL_PROPERTIES:
        np.testing.assert_array_equal(gated.columns[name], trained.columns[name])


def test_a_bar_that_would_leave_nothing_falls_back_and_says_so(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    _seed(workdir, n=5_000, cameras=16)

    _run(workdir, {"bar": "strict", "keep_min_views": 999, "context_min_views": 999})

    document = json.loads((workdir.out_dir("quality") / "quality.json").read_text())
    assert document["bar"] == "strict"
    assert document["barApplied"] == "everything"
    assert document["gaussians"]["out"] == 5_000


def test_tips_come_from_the_capture_itself(tmp_path: Path) -> None:
    """A single loop at the subject's own height, three quarters of the way round."""
    workdir = Workdir.create(tmp_path / "run")
    columns, _ = scene(8_000)
    trained = workdir.input_path("trained.ply")
    trained.parent.mkdir(parents=True, exist_ok=True)
    gaussians.write_ply(trained, columns)
    arc = [c for c in ring(32, radius=1.5, height=0.45) if np.arctan2(c[1][1], c[1][0]) < np.pi / 2]
    write_model(workdir.input_path("poses"), arc)
    workdir.input_path("georef.json").write_text(json.dumps({"lat": 0, "lon": 0}))
    execute(
        make_recipe(
            [{"id": "quality", "impl": "support_gate"}],
            inputs=["trained.ply", "poses", "georef.json"],
        ),
        workdir,
        RunnerSet.local(),
    )

    document = json.loads((workdir.out_dir("quality") / "quality.json").read_text())
    ids = [tip["id"] for tip in document["tips"]]
    assert "from-above" in ids
    assert "all-around" in ids
    # Three quarters of the way round leaves at least a quarter, measured as the widest
    # gap about the ROI's centre (which the partial ring pulls off the ring's own centre),
    # plus one camera spacing -- not as a count of occupied 45-degree sectors.
    gap = document["capture"]["azimuthGapDeg"]
    assert 90 <= gap <= 135
    around = next(tip for tip in document["tips"] if tip["id"] == "all-around")
    assert f"about {int(round(gap / 15.0) * 15)}°" in around["text"]
    assert document["heldOutPsnr"] is None
    assert 1 <= len(ids) <= 4


def test_an_exif_aligned_capture_reports_its_gsd_in_millimetres(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    _seed(workdir, n=5_000, cameras=16)
    frame = {
        "source": "exif-gps-similarity",
        "scale": 2.0,
        "rotation": np.eye(3).tolist(),
        "translationM": [0.0, 0.0, 0.0],
        "recentre": False,
    }
    workdir.input_path("georef.json").write_text(
        json.dumps({"lat": 1.0, "lon": 2.0, "scaleSource": "exif-gps", "frame": frame})
    )

    _run(workdir)

    gsd = json.loads((workdir.out_dir("quality") / "quality.json").read_text())["gsd"]
    assert gsd["metric"] is True
    assert gsd["medianRoiMm"] == pytest.approx(gsd["medianRoiModelUnits"] * 2000.0, rel=1e-3)


def test_an_unknown_mode_is_refused(tmp_path: Path) -> None:
    from errors import StageFailedError

    workdir = Workdir.create(tmp_path / "run")
    _seed(workdir, n=500, cameras=8)
    with pytest.raises(StageFailedError, match="preview, refine"):
        _run(workdir, {"mode": "draft"})


@pytest.mark.skipif(
    os.environ.get("PIPELINE_BENCH") != "1",
    reason="the 500k x 90 benchmark takes a while; PIPELINE_BENCH=1 runs it",
)
def test_benchmark_500k_gaussians_by_90_cameras(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    _seed(workdir, n=500_000, cameras=90)

    started = time.perf_counter()
    execute(
        make_recipe(
            [{"id": "quality", "impl": "support_gate"}],
            inputs=["trained.ply", "poses", "georef.json", "train_metrics.json"],
        ),
        workdir,
        RunnerSet.local(),
    )
    seconds = time.perf_counter() - started

    document = json.loads((workdir.out_dir("quality") / "quality.json").read_text())
    sys.stdout.write(f"\nquality stage, 500k x 90: {seconds:.1f} s; {document['gaussians']}\n")
    assert seconds < 120

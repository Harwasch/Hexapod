"""Coverage, budget and production routing for backbone + registration (CPU only)."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_live import write_model

import colmap4
import keyframes
import pose_backbone
import sfm
import tree_frames
from conftest import FIXTURE_PLY, make_recipe
from contracts import StageContext
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir


def trajectory(count: int, shift: float = 0.0, parallax: float = 0.0) -> keyframes.Analysis:
    homography = np.eye(3)
    homography[0, 2] = shift * 480
    residuals = np.zeros((4, 4, 2))
    residuals[:, :, 0] = parallax * 480
    return keyframes.Analysis(
        (1.0,) * count,
        (None, *(keyframes.PairMotion(homography, residuals, 16, True) for _ in range(count - 1))),
        (480, 270),
    )


def test_backbone_scales_with_viewpoint_coverage_instead_of_frame_count() -> None:
    orbit = keyframes.coverage_backbone(trajectory(2037, parallax=0.001))
    walk = keyframes.coverage_backbone(trajectory(2037, shift=0.02, parallax=0.003))
    assert len(walk) > 2 * len(orbit)
    # More native samples of the same path do not demand a larger backbone.
    dense = keyframes.coverage_backbone(trajectory(4073, parallax=0.0005))
    assert abs(len(dense) - len(orbit)) <= 2
    assert orbit[0] == 0 and orbit[-1] == 2036


def test_beach_plan_fits_the_pose_budget_without_a_fixed_backbone_cap() -> None:
    # Synthetic proxy for ~6 native views per ~1.2 degree backbone step. The real
    # beach trajectory is not available in this worktree; 300 is the solved bench.
    backbone = keyframes.coverage_backbone(trajectory(2037, parallax=0.0015))
    assert 200 <= len(backbone) <= 450
    plan = pose_backbone.estimate_seconds(2037, len(backbone))
    assert plan["backboneS"] < 1.5 * 3600
    assert plan["incrementalS"] > 6 * 3600
    assert len(keyframes.coverage_backbone(trajectory(8000, parallax=0.003))) > 900


def test_unknown_motion_keeps_coverage_and_static_capture_keeps_anchors() -> None:
    unknown = keyframes.Analysis((1.0,) * 20, (None,) * 20, (480, 270))
    assert keyframes.coverage_backbone(unknown) == tuple(range(20))
    assert len(keyframes.coverage_backbone(trajectory(1000))) >= 2


@pytest.mark.parametrize("source", ["video", "images", None])
def test_auto_preserves_small_captures_and_off_preserves_all_captures(source: str | None) -> None:
    assert not pose_backbone.enabled("auto", 40, source)
    assert not pose_backbone.enabled("auto", 199, source)
    assert pose_backbone.enabled("auto", 200, source)
    assert pose_backbone.enabled("auto", 2037, source)
    assert not pose_backbone.enabled("off", 2037, source)
    with pytest.raises(ValueError, match="backbone"):
        pose_backbone.enabled("300", 2037, source)


def test_sparse_registration_pairs_touch_only_solved_anchors() -> None:
    names = [f"frame_{i:04}.jpg" for i in range(100)]
    anchors = {names[i] for i in (0, 20, 40, 60, 80, 99)}
    pairs = pose_backbone.registration_pairs(names, anchors, ordered=True)
    assert len(pairs) <= 8 * (100 - len(anchors))
    assert all(a not in anchors and b in anchors for a, b in pairs)
    assert {a for a, _ in pairs} == set(names) - anchors
    unordered = pose_backbone.registration_pairs(names, anchors, ordered=False)
    assert len(unordered) == len(anchors) * (100 - len(anchors))


@pytest.mark.parametrize("mode, target", [("auto", 0.8), ("off", 0.8), ("auto", 1.0)])
def test_cli_stage_routes_only_backbone_through_mapper_and_records_drops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, target: float
) -> None:
    from PIL import Image

    workdir = Workdir.create(tmp_path / "run")
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True)
    poses = tree_frames.orbit(12)
    for pose in poses:
        Image.new("RGB", (640, 480)).save(frames / pose.name)
    workdir.input_path("source_meta.json").write_text('{"format":"video"}')
    monkeypatch.setattr(pose_backbone, "MIN_FRAMES", 1)
    monkeypatch.setattr(keyframes, "analyse", lambda paths: trajectory(len(paths)))
    monkeypatch.setattr(keyframes, "coverage_backbone", lambda analysis: (0, 3, 6, 9, 11))
    monkeypatch.setattr(sfm, "colmap_exe", lambda: "colmap-stand-in")
    monkeypatch.setattr(sfm, "colmap_version", lambda: "3.9.1-stand-in")
    monkeypatch.setattr(sfm, "count_pairs", lambda db: (10, 10))
    commands: list[list[str]] = []

    def run(ctx: StageContext, argv: list[str]) -> None:
        commands.append(argv)

        def flag(name: str) -> Path:
            return Path(argv[argv.index(name) + 1])

        if argv[1] == "mapper":
            names = {p.name for p in flag("--image_path").iterdir()}
            write_model(flag("--output_path") / "0", tuple(p for p in poses if p.name in names))
        elif argv[1] == "image_registrator":
            # One native frame fails PnP; this is a model stand-in, not evidence
            # about COLMAP's registration accuracy.
            write_model(flag("--output_path"), poses[:-2] + poses[-1:])
        elif argv[1] in ("point_triangulator", "bundle_adjuster"):
            shutil.copytree(flag("--input_path"), flag("--output_path"), dirs_exist_ok=True)

    monkeypatch.setattr(StageContext, "run", run)
    recipe = make_recipe(
        [
            {
                "id": "pose",
                "impl": "colmap",
                "params": {
                    "backbone": mode,
                    "matcher": "exhaustive",
                    "live": False,
                    "rematches": 0,
                    "threads": 2,
                    "min_registered_fraction": target,
                },
            }
        ],
        inputs=["frames", "source_meta.json"],
    )
    if target == 1.0:
        with pytest.raises(Exception, match="backbone_coverage"):
            execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    else:
        execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    summary = json.loads((workdir.artifact_path("pose", "poses") / "poses.json").read_text())
    if mode == "off":
        assert summary["registered"] == 12
        assert not any(c[1] == "image_registrator" for c in commands)
    else:
        assert summary["backbone"]["size"] == 5
        assert summary["mapperAttempts"][0]["registered"] == 5
        assert summary["registered"] == 11
        registration = summary["backbone"]["registration"]
        assert registration["attempted"] == 7 and registration["registered"] == 6
        assert registration["failed"] == 1
        assert registration["failures"][0]["name"] == poses[-2].name
        assert summary["gates"][-1]["passed"] == (target < 1.0)
        for command in commands:
            if command[1] == "bundle_adjuster":
                assert command[command.index("--BundleAdjustment.max_num_iterations") + 1] == "10"
            if command[1] == "feature_extractor" and "--ImageReader.existing_camera_id" in command:
                assert command[command.index("--ImageReader.existing_camera_id") + 1] == "1"


@pytest.mark.skipif(colmap4.version(colmap4.python()) is None, reason="needs cached pycolmap 4.2")
def test_real_42_registers_held_out_views_into_the_production_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    workdir = Workdir.create(tmp_path / "run")
    frames = workdir.input_path("frames")
    # The 40 uniformly spaced backbone views are the existing pose suite's known
    # orbit. Interleave 40 new views; this checks registration without widening
    # that fixture's proven angular gaps or changing its accuracy thresholds.
    truth = tree_frames.render_orbit(FIXTURE_PLY, frames, count=80)
    Image.new("RGB", (640, 480)).save(frames / "unrelated.jpg")
    # A plausible EXIF seed (~4% above the known 520px camera), free to refine.
    # Calibration preservation is checked on the production EXIF-seeded path.
    workdir.input_path("frames_meta.json").write_text(
        json.dumps(
            {
                "focal": {"px": 540.0, "from": "exif-35mm"},
                "frames": [{"frameSize": [640, 480]}],
            }
        )
    )
    monkeypatch.setattr(pose_backbone, "MIN_FRAMES", 1)
    monkeypatch.setattr(keyframes, "analyse", lambda paths: trajectory(len(paths)))
    monkeypatch.setattr(keyframes, "coverage_backbone", lambda analysis: tuple(range(0, 80, 2)))
    recipe = make_recipe(
        [
            {
                "id": "pose",
                "impl": "colmap",
                "params": {
                    "colmap": "4.2",
                    "matcher": "exhaustive",
                    "live": False,
                    "threads": 4,
                    "max_features": 4096,
                    "first_octave": 0,
                    "focal_from": "exif",
                },
            }
        ],
        inputs=["frames", "frames_meta.json"],
    )
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    output = workdir.artifact_path("pose", "poses")
    summary: Any = json.loads((output / "poses.json").read_text())
    model = sfm.read_model(output)
    assert model.registered == 80
    assert len(model.cameras) == 1
    assert summary["backbone"]["size"] == 40
    registration = summary["backbone"]["registration"]
    assert registration["attempted"] == 41 and registration["registered"] == 40
    assert registration["failures"][0]["name"] == "unrelated.jpg"
    assert model.mean_reprojection_error < 1.0
    known = truth.by_name()
    fit = sfm.umeyama(model.centres(), np.stack([known[i.name].centre for i in model.images]))
    errors = [
        sfm.rotation_angle_deg(fit.rotation @ i.rotation.T, known[i.name].rotation.T)
        for i in model.images
    ]
    # Same accuracy bar as the existing synthetic-orbit pose tests.
    assert float(np.median(errors)) < 1.0
    originals = [sfm.read_model(p) for p in (workdir.work_dir("pose") / "sparse").glob("*/*")]
    original = max(originals, key=lambda candidate: candidate.registered)
    # Production normally refines the focal while solving the backbone. Extending
    # the database must preserve that calibration, rather than its original seed.
    assert model.cameras[0].params == original.cameras[0].params
    assert summary["focal"]["initialPx"] == 540.0
    assert model.cameras[0].focal_px != 540.0
    final = model.by_name()
    for anchor in original.images:
        assert np.linalg.norm(final[anchor.name].centre - anchor.centre) < 1e-9


def test_all_anchor_photo_fallback_needs_no_registration_commands(tmp_path: Path) -> None:
    from types import SimpleNamespace

    poses = tree_frames.orbit(12)
    frames = tmp_path / "frames"
    frames.mkdir()
    for image in poses:
        (frames / image.name).touch()
    model = tmp_path / "model"
    write_model(model, poses)

    def unexpected_run(argv: list[str]) -> None:
        pytest.fail(f"all-anchor fallback must not run registration: {argv}")

    context: Any = SimpleNamespace(work_dir=tmp_path, run=unexpected_run)
    report = pose_backbone.register(
        context,
        tmp_path / "database.db",
        frames,
        model,
        tmp_path / "output",
        four=True,
        interpreter="unused",
        threads=4,
        ordered=False,
    )
    assert report["attempted"] == 0 and report["registered"] == 0 and report["failed"] == 0
    assert sfm.read_model(tmp_path / "output").registered == 12

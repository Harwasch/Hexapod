from __future__ import annotations

import shutil
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import quality
import sfm
import train_split
import training
import video
from experiments.bench import __main__ as cli
from experiments.bench import full_data, full_register, ledger, rehearsal, scene


def test_exclusion_is_inclusive_and_never_changes_test(tmp_path: Path) -> None:
    scene.write_json(
        tmp_path / "frames_meta.json",
        {
            "frames": [
                {"name": n, "offsetS": t}
                for n, t in [("test", 1), ("a", 0.5), ("b", 1.5), ("c", 1.50001), ("d", 0)]
            ]
        },
    )
    split = {"test": ["test"], "pool": ["a", "b", "c", "d"]}
    filtered, report = full_data.exclude_pool(tmp_path, split, 0.5)
    assert filtered["pool"] == ["c", "d"]
    assert filtered["test"] == split["test"] == ["test"]
    assert split["pool"] == ["a", "b", "c", "d"]
    assert report["excludedCount"] == 2
    with pytest.raises(ValueError, match="timestamps"):
        full_data.exclude_pool(tmp_path, {"test": ["missing"], "pool": ["a"]}, 0.5)


def test_selection_keeps_motion_without_a_count_cap() -> None:
    rows = [{"name": str(i), "sharpness": 100 if i != 4 else 10} for i in range(20)]
    thumbs = [np.full((3, 3), i * 4, dtype=np.uint8) for i in range(20)]
    thumbs[8] = thumbs[7].copy()
    selected = full_data.select_frames(rows, thumbs)
    assert selected[4]["selectionReason"] == "blur"
    assert selected[8]["selectionReason"] == "near-duplicate"
    assert sum(r["kept"] for r in selected) == 18
    # Compare to last KEPT, not previous decoded frame: slow motion accumulates.
    slow = full_data.select_frames(
        rows[:4], [np.full((2, 2), i, dtype=np.uint8) for i in range(4)], blur_ratio=0
    )
    assert [r["kept"] for r in slow] == [True, False, True, False]


def test_native_decode_neither_resamples_nor_shrinks() -> None:
    argv = video.extract_frames_argv(
        Path("source.mov"),
        Path("full_%06d.jpg"),
        fps=None,
        tonemap=video.TONEMAP_FILTER,
        timestamps=True,
    )
    chain = argv[argv.index("-vf") + 1]
    assert chain == video.TONEMAP_FILTER + ",showinfo"
    assert argv[argv.index("-fps_mode") + 1] == "passthrough"
    assert "fps=" not in chain


def test_pairs_are_temporal_and_bounded_per_native_frame() -> None:
    anchors = {f"a{i}": float(i) for i in range(20)}
    pairs = full_register.temporal_pairs(anchors, {"new": 10.2}, 2)
    assert {a for a, b in pairs if b == "native/new"} == {f"backbone/a{i}" for i in (9, 10, 11, 12)}
    assert ("backbone/a0", "backbone/a19") not in pairs


def test_inherited_scene_keeps_exact_test_identity(tmp_path: Path) -> None:
    original = tmp_path / "original"
    rehearsal.synthetic_scene(original, "beach", "test")
    manifest, split = scene.verify(original)
    identity = full_data.evaluation_identity(original, split)
    target = tmp_path / "new"
    shutil.copytree(original, target)
    body = {
        **split,
        "scene": "beach-full",
        "rule": full_data.RULE,
        "parentManifestDigest": manifest["digest"],
        "pool": split["pool"][::2],
    }
    body.pop("digest")
    inherited = {**body, "digest": scene.digest(body)}
    scene.seal(
        target,
        "beach-full",
        "test",
        "full",
        "test",
        inherited_split=inherited,
        evaluation_identity=identity,
    )
    _, verified = scene.verify(target)
    assert verified["test"] == split["test"]
    assert full_data.evaluation_identity(target, verified) == identity
    (target / "frames" / split["test"][0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest changed"):
        scene.verify(target)


def test_fixed_pose_guard_rejects_motion_and_preserves_storage(tmp_path: Path) -> None:
    rehearsal.synthetic_scene(tmp_path / "s", "beach", "test")
    poses = tmp_path / "s" / "poses"
    original = sfm.read_model(poses)
    before = (poses / "images.bin").read_bytes()
    full_register.restore_exact_backbone(poses, original)
    assert (poses / "images.bin").read_bytes() == before
    data = bytearray(before)
    struct.pack_into("d", data, 8 + 4 + 32, 10.0)
    (poses / "images.bin").write_bytes(data)
    with pytest.raises(ValueError, match="moved"):
        full_register.restore_exact_backbone(poses, original)


def test_no_holdout_dataset_really_trains_all_and_never_evaluates() -> None:
    class Dataset:
        def __init__(self, parser: Any, split: str, **kwargs: Any) -> None:
            indices = np.arange(len(parser.image_names))
            self.indices = indices[indices % parser.test_every != 0]

    class Runner:
        def eval(self) -> None:
            raise AssertionError("validation should be disabled")

    module: dict[str, Any] = {"Dataset": Dataset}
    cfg = SimpleNamespace(test_every=0)
    train_split.install(module, Runner, cfg)
    parser = SimpleNamespace(test_every=0, image_names=list(range(19)))
    assert module["Dataset"](parser, split="train").indices.tolist() == list(range(19))
    assert module["Dataset"](parser, split="val").indices.size == 0
    assert parser.test_every == 0
    Runner().eval()
    assert training.held_out_split(19, 0) == (19, 0)
    with pytest.raises(ValueError):
        training.held_out_split(19, 1)


def test_opt_in_holdout_keeps_original_dataset() -> None:
    dataset = object()
    module = {"Dataset": dataset}
    train_split.install(module, None, SimpleNamespace(test_every=8))
    assert module["Dataset"] is dataset


def test_schedule_grows_above_sixty_without_internal_validation() -> None:
    assert training.schedule_scale(240, full_at=60, floor=0.25, growth="sqrt") == 2
    assert training.schedule_scale(960, full_at=60, floor=0.25, growth="sqrt") == 4
    assert training.schedule_scale(15, full_at=60, floor=0.25, growth="sqrt") == 0.25


def test_quality_mark_keeps_geometry_and_opacity_with_tiers() -> None:
    columns = {
        "x": np.array([1, 2, 100], dtype=np.float32),
        "opacity": np.array([0, 1, 2], dtype=np.float32),
    }
    tiers = np.array([quality.TIER_KEEP, quality.TIER_CONTEXT, quality.TIER_DROP], dtype=np.uint8)
    marked = quality.gate(columns, tiers, "mark")
    for key, values in columns.items():
        np.testing.assert_array_equal(marked[key], values)
    np.testing.assert_array_equal(marked["quality_tier"], tiers)


def test_full_freeze_is_cpu_and_spend_cap_covers_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    args = cli.parser().parse_args(["freeze-full", "--tag", "beach-full-data", "--total-cap", "8"])
    freeze = cli.plans_for(args)[0]
    assert freeze["tier"] == "cpu4" and freeze["excludeTestSeconds"] == 0.1
    assert freeze["schedule"] == "freeze-full"
    worst = ledger.hourly_rate("cpu4") * (7200 + 240) / 3600
    worst += ledger.hourly_rate("l4") * ((5400 + 240) + (16200 + 240) + (5400 + 240)) / 3600
    assert worst < 8


def test_full_variants_have_no_quality_features_and_keep_outputs() -> None:
    variants = Path(__file__).parents[1] / "experiments/variants/beach-full.yaml"
    for name, which_scene in (("upgrade-excl", "beach"), ("full-data", "beach-full")):
        args = cli.parser().parse_args(
            [
                "run",
                "--scene",
                which_scene,
                "--variants",
                str(variants),
                "--variant",
                name,
                "--schedule",
                "full",
                "--live",
            ]
        )
        plan = cli.plans_for(args)[0]
        assert plan["live"] and plan["keepSplat"]
        assert plan["testExclusionSeconds"] == 0.1
        assert plan["trainParams"]["test_every"] == 0
        assert plan["trainParams"]["cap_max"] == "auto"
        assert "_bench_quality" not in plan["trainParams"]


def test_default_no_holdout_and_refine_does_not_crop_preview(tmp_path: Path) -> None:
    import json

    from executor import execute
    from runners import LocalRunner, RunnerSet
    from test_train_gsplat import seed_inputs, stand_in_params, train_recipe
    from workdir import Workdir

    work = Workdir.create(tmp_path / "run")
    seed_inputs(work)
    params = stand_in_params()
    params.pop("test_every")
    # A malformed preview mask is ignored by default, rather than restricting recovery.
    params["support_mask"] = {"not": "a region"}
    execute(train_recipe(params), work, RunnerSet(cpu=LocalRunner()))
    document = json.loads((work.out_dir("train") / "train_metrics.json").read_text())
    assert document["heldOut"]["testEvery"] == 0
    assert document["heldOut"]["valFrames"] == 0
    assert document["psnr"] is None
    assert document["convergence"]["enabled"] is False
    assert "--test_every 0" in work.log_path("train").read_text()
    assert document["roi"] is None


def test_quality_tier_survives_canonical_transform(tmp_path: Path) -> None:
    import gaussians
    import splat_io
    import splat_stream

    columns = {name: np.zeros(3, np.float32) for name in gaussians.ply_properties(0)}
    columns["rot_0"][:] = 1
    columns["quality_tier"] = np.array([0, 1, 2], dtype=np.float32)
    path = tmp_path / "marked.ply"
    with splat_io.PlyWriter(path, tuple(columns), count=3) as writer:
        writer.append(columns)
    source = splat_stream.PlySource(path)
    assert "quality_tier" in source.properties
    transformed = gaussians.transform(source.read(0, 3), np.eye(3), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(transformed["quality_tier"], columns["quality_tier"])


def test_block_training_can_use_every_frame_and_records_limits(tmp_path: Path) -> None:
    from test_blocks import block_params, run_blocks, seed_prior, seed_walk

    from workdir import Workdir

    work = Workdir.create(tmp_path / "run")
    seed_walk(work)
    seed_prior(work, tmp_path / "scratch")
    document = run_blocks(work, block_params(test_every=0))
    assert document["heldOut"]["testEvery"] == 0
    assert document["heldOut"]["valFrames"] == 0
    assert document["heldOut"]["trainFrames"] == 48
    assert document["psnr"] is None
    assert all(b["cameras"]["val"] == 0 for b in document["blocks"]["blocks"])
    assert {r["name"] for r in document["limits"]} == {
        "blockGaussians",
        "trainingSteps",
        "internalHoldout",
    }
    assert all(r["reason"] for r in document["limits"])


def test_original_lookup_handles_worker_normalize_without_guessing() -> None:
    class Store:
        def list(self, prefix: str) -> list[dict[str, Any]]:
            if prefix == "captures/":
                return [{"Key": "captures/a/source/b/IMG_3450.mov"}]
            return []

    assert full_data.find_original(Store(), "runs/job/")["Key"].startswith("captures/")

    class Ambiguous:
        def list(self, prefix: str) -> list[dict[str, Any]]:
            return [{"Key": "a/IMG_3450.mov"}, {"Key": "b/IMG_3450.mov"}]

    with pytest.raises(ValueError, match="2 originals"):
        full_data.find_original(Ambiguous(), "runs/job/")


def test_cross_scene_rescore_requires_exact_evaluation_identity(tmp_path: Path) -> None:
    from test_bench_matched import A, B, pair

    from experiments.bench import rescore

    manifest, split = pair(tmp_path)
    identity = full_data.evaluation_identity(tmp_path / "scenes/beach", split)
    manifest["evaluationIdentity"] = identity
    path = tmp_path / "runs" / B / "result.json"
    result = scene.read_json(path)
    result.update(
        scene="beach-full", manifestDigest="new", splitDigest="new", evaluationIdentity=identity
    )
    scene.write_json(path, result)
    plan = {"scene": "beach", "runs": [A, B], "ply": ["trained"], "rehearsal": True}
    selected, _ = rescore.inventory(tmp_path / "runs", plan, manifest)
    assert len(selected) == 2
    result["evaluationIdentity"] = "changed camera or image"
    scene.write_json(path, result)
    with pytest.raises(ValueError, match="identical frozen"):
        rescore.inventory(tmp_path / "runs", plan, manifest)


def test_paid_forecast_does_not_reuse_a_different_recipe() -> None:
    entries: list[dict[str, Any]] = [
        {
            "event": "planned",
            "runId": "old",
            "scene": "beach",
            "schedule": "full",
            "tier": "l4",
            "variant": {"name": "crashfix"},
        },
        {"event": "complete", "runId": "old", "costUsd": 1.96},
    ]
    assert ledger.estimate(entries, "beach", "full", "l4", variant="upgrade-excl") == 0.9
    assert ledger.estimate(entries, "beach", "full", "l4", variant="crashfix") == 1.96


def test_native_decode_reports_actual_rate_count_and_exposure(tmp_path: Path) -> None:
    import subprocess

    source = tmp_path / "clip.mkv"
    subprocess.run(
        [
            video.ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=7",
            "-frames:v",
            "7",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
    )
    report = full_data.extract(source, tmp_path / "frames", blur_ratio=0, duplicate_mae=0)
    assert report["decodedFrames"] == 7
    assert report["meanDecodedFps"] == pytest.approx(
        7, abs=0.01
    )  # container's millisecond timebase
    assert len(report["frames"]) == 7
    assert report["frames"][0]["offsetS"] == 0
    assert all(r["frameSize"] == [96, 64] for r in report["frames"])
    assert all(0 <= r["exposure"]["lumaMean"] <= 255 for r in report["frames"])


def test_pycolmap_fixed_backbone_at_two_image_scales(tmp_path: Path) -> None:
    pytest.importorskip("pycolmap", minversion="4.2.0")
    from PIL import Image

    import tree_frames
    from conftest import FIXTURE_PLY
    from synthetic_scene import write_model

    base, target = tmp_path / "base", tmp_path / "target"
    truth = tree_frames.render_orbit(
        FIXTURE_PLY, target / "frames", count=40, width=640, height=480, focal_px=520
    )
    write_model(
        base / "poses",
        [(p.rotation, p.centre) for p in truth.poses],
        width=640,
        height=480,
        focal=520,
    )
    backbone = {p.name: float(i) for i, p in enumerate(truth.poses)}
    native = {}
    for i in (3, 7, 11):
        name = f"full_{i:06d}.jpg"
        with Image.open(target / "frames" / truth.poses[i].name) as photo:
            photo.resize((800, 600)).save(target / "frames" / name, quality=95)
        native[name] = float(i)
    original = sfm.read_model(base / "poses")
    report = full_register.register(base, target, tmp_path / "work", backbone, native)
    assert report["registered"] == 3 and report["failures"] == []
    after = {i.name: i for i in sfm.read_model(target / "poses").images}
    for image in original.images:
        assert after[image.name].qvec == image.qvec
        assert after[image.name].tvec == image.tvec


def test_normalization_recovery_is_defined_for_a_line_of_cameras() -> None:
    original = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([4.0, -2.0, 8.0])
    normalized = (original - translation) @ rotation / 2
    scale, restored, offset = full_register.gauge_transform(
        original, np.eye(3), normalized, rotation
    )
    assert scale == 2
    np.testing.assert_array_equal(restored, rotation)
    np.testing.assert_array_equal(scale * normalized @ restored.T + offset, original)

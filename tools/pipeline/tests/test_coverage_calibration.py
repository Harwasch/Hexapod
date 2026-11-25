"""Numpy-only end-to-end calibration rehearsal, with deliberately planted image error."""

import argparse
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


def test_campaign_budget_pinned_training_and_evaluation_commands(tmp_path: Path) -> None:
    from experiments.bench.scene import SCENES, evaluation_subset, frame_selection, split_frames
    from experiments.coverage_quality.campaign import (
        ACTIONS,
        CAP_USD,
        budget,
        evaluation_argv,
        reserve_local,
        training_plan,
    )

    plan = budget()
    assert plan["totalWorstCaseUsd"] == pytest.approx(4.6641696)
    assert plan["totalWorstCaseUsd"] < CAP_USD == 5
    assert "freeze" not in ACTIONS  # Successful freeze must never be dispatched again.
    assert ACTIONS["main-c935"][1] <= 6300
    assert ACTIONS["main-beach"][1] <= 900
    assert all(ACTIONS[a][1] <= 480 for a in ACTIONS if a.startswith("rehearse"))
    assert SCENES["c935-coverage"].startswith("c9357a92-")
    names = [f"frame_{i:04}.jpg" for i in range(724)]
    split = split_frames("c935-coverage", list(reversed(names)))
    assert split["test"] == names[::8]
    assert len(split["pool"]) == 633 and len(split["test"]) == 91
    assert evaluation_subset(split, split["test"][:4])["test"] == names[::8][:4]
    for invalid in ([], [names[1]], [names[0], names[0]]):
        with pytest.raises(ValueError, match="subset"):
            evaluation_subset(split, invalid)
    roles = frame_selection(split, 0)
    assert roles["train"] == split["pool"] and roles["val"] == []
    roles = frame_selection(split, 8)
    assert roles["val"] == sorted(split["pool"])[::8]
    assert not set(roles["val"]) & set(roles["train"])
    with pytest.raises(ValueError, match="test_every"):
        frame_selection(split, 1)
    train = training_plan("main-c935", "test")["trainParams"]
    import training

    assert training.scaled_steps(train["iterations"], training.frame_schedule(633, train)) == 60000
    assert train["test_every"] == 0  # Frozen test frames already removed, no second split.
    assert train["strategy"] == "mcmc" and train["cap_max"] == 5626152
    assert train["init_from"] == "sfm"
    assert train["ship_sh_degree"] == 3 and train["opacity_reg"] == 0.002
    assert train["pose_opt"] is False and train["train_max_side"] is None
    short = training_plan("rehearse-c935", "test")["trainParams"]
    assert short["iterations"] * short["schedule_scale"] == 300
    for action in ("reeval-c935-rehearsal",):
        argv = evaluation_argv(action, {})
        assert argv[argv.index("--max-test-frames") + 1] == "4"
        assert argv[argv.index("--max-patches") + 1] == "64"
    argv = evaluation_argv("main-beach", {})
    assert "--max-test-frames" not in argv
    with pytest.raises(ValueError, match="not complete"):
        reserve_local(tmp_path, "main-c935")
    (tmp_path / "freeze").mkdir()
    (tmp_path / "freeze/status.json").write_text('{"status":"complete"}')
    # The original failed rehearsal record is retained; the repair has one fixed ID.
    (tmp_path / "campaign.jsonl").write_text('{"action":"rehearse-c935"}\n')
    (tmp_path / "rehearse-beach-r2").mkdir()
    (tmp_path / "rehearse-beach-r2/status.json").write_text('{"status":"complete"}')
    reserve_local(tmp_path, "reeval-c935-rehearsal")
    with pytest.raises(ValueError, match="already attempted"):
        reserve_local(tmp_path, "reeval-c935-rehearsal")


def test_named_split_prevents_test_and_validation_camera_leakage() -> None:
    from experiments.coverage_quality.selection import choose_test_views, select_frames

    all_names = [f"f{i}" for i in range(12)]
    test_names = [f"t{i}" for i in range(91)]
    assert choose_test_views(test_names, 4) == ["t0", "t30", "t60", "t90"]
    assert choose_test_views(test_names, 0) == test_names
    assert choose_test_views(test_names, 1) == ["t0"]
    with pytest.raises(ValueError, match="nonnegative"):
        choose_test_views(test_names, -1)
    selection = {"test": ["f0", "f8"], "train": ["f1", "f2"], "val": ["f3"]}
    train, test, source = select_frames(
        all_names, selection, {}, {}, {"perFrame": [{"name": "f0"}]}
    )
    assert train == ["f1", "f2"] and test == ["f0", "f8"]
    assert "explicit" in source
    pool = sorted(set(all_names) - {"f0", "f8"})
    train, _, _ = select_frames(
        all_names, {}, {"test": ["f0", "f8"], "pool": pool}, {"trainParams": {"test_every": 8}}, {}
    )
    assert train == [n for i, n in enumerate(pool) if i % 8 != 0]
    with pytest.raises(ValueError, match="overlap"):
        select_frames(all_names, {"test": ["f0"], "train": ["f0"]}, {}, {}, {})
    with pytest.raises(ValueError, match="no actual train"):
        select_frames(all_names, {}, {"test": ["f0"], "pool": ["f1"]}, {}, {})
    with pytest.raises(ValueError, match="validation"):
        select_frames(all_names, {**selection, "val": ["f1"]}, {}, {}, {})
    with pytest.raises(ValueError, match="optimized"):
        select_frames(all_names, selection, {}, {"trainParams": {"pose_opt": True}}, {})


def test_metric_evidence_and_metadata_are_explicit() -> None:
    from experiments.coverage_quality.selection import frame_metadata, metric_scale

    assert metric_scale({"manifest": {"metresPerUnit": 1, "scaleSource": "ARKit"}})[0] == 1
    with pytest.raises(ValueError, match="unknown"):
        metric_scale({"georef": {"scale": 1, "scaleSource": "unresolved"}})
    with pytest.raises(ValueError, match="conflicting"):
        metric_scale(
            {"metresPerUnit": 2, "source": "survey"},
            attested_scale=1,
            attested_source="operator verified metre coordinates",
        )
    rows = frame_metadata({"frames": [{"name": "a", "blur": 0.0, "exposure": {"ev100": 0.0}}]})
    assert rows == {"a": {"blur": 0.0, "exposure": 0.0}}


def test_pooled_fit_and_scene_transfer_recover_planted_kernel(tmp_path: Path) -> None:
    from experiments.coverage_quality.fitting import compact_rows, pooled_fit, predict, read_rows

    rows = []
    rng = np.random.default_rng(901)
    for scene, intercept in (("c935", 0.03), ("beach", 0.05)):
        raw = []
        for index in range(96):
            angles = np.empty(0) if index % 12 == 0 else rng.uniform(0, 65, int(rng.integers(1, 9)))
            support = min(1, float(credit(np.cos(np.radians(angles)), 25, 2).sum()) / 4)
            error = intercept + 0.4 * (1 - support)
            raw.append(
                {
                    "scene": scene,
                    "view": f"v{index % 4}",
                    "patch_i": index,
                    "patch_j": 0,
                    "patch_k": 0,
                    "observing_views": len(angles),
                    "angles_deg": angles.tolist(),
                    "nearest_deg": float(angles.min()) if len(angles) else 180.0,
                    "best_mm_px": 1.0 if len(angles) else None,
                    "parallax_deg": 45.0 if len(angles) else 0.0,
                    "error": error,
                    "l1": error,
                    "lpips": error,
                }
            )
        original = fit_credit(raw)
        compact = compact_rows(raw, scene)
        assert fit_credit(compact)["r2"] == pytest.approx(original["r2"], abs=1e-12)
        rows += compact
    fit = pooled_fit(rows)
    model = fit["pooled"]
    assert model["scene_intercepts"] is True
    assert model["rows"] == len(rows)  # Zero-training-view test patches are not selected away.
    assert model["r2"] > 0.999
    assert model["leave_view_out_r2"] > 0.999
    assert (model["cone_deg"], model["sharpness"], model["views_to_fill"]) == (25, 2, 4)
    assert model["leave_one_scene_out_r2"] > 0.9
    assert set(model["leave_one_scene_out"]) == {"c935", "beach"}
    for held in ("c935", "beach"):
        transfer = model["leave_one_scene_out"][held]["fit"]
        assert transfer["scene_intercepts"] is False
        assert transfer["scene_names"] == ["beach" if held == "c935" else "c935"]
    y = np.array([r["error"] for r in rows])
    assert predict(rows, model) == pytest.approx(y, abs=1e-8)
    # Compact derived CSV is sufficient for the LOCAL pooled fit, without camera lists.
    complete = [
        {
            **dict.fromkeys(COLUMNS),
            **r,
            "pixels": 4,
            "ssim": 1,
            "within10": 0,
            "within20": 0,
            "within30": 0,
        }
        for r in rows
    ]
    write_outputs(complete, tmp_path)
    restored = read_rows(tmp_path / "patches.csv")
    assert predict(restored, model) == pytest.approx(y, abs=1e-8)
    assert len((tmp_path / "patches.csv").read_bytes()) < 250000


def test_modal_campaign_definitions_bound_cost_and_download_only_derivatives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import sys

    from experiments.coverage_quality import campaign

    definitions = []
    mounts = []

    class FakeImage:
        def add_local_dir(self, *args: Any, **kwargs: Any) -> Any:
            mounts.append(kwargs)
            return self

        def add_local_file(self, *args: Any, **kwargs: Any) -> Any:
            mounts.append(kwargs)
            return self

    class FakeApp:
        def __init__(self, name: str) -> None:
            assert name == campaign.CAMPAIGN

        def function(self, **options: Any) -> Any:
            definitions.append(options)
            return lambda body: body

    fake = SimpleNamespace(
        App=FakeApp,
        Function=SimpleNamespace(
            from_name=lambda *a: SimpleNamespace(
                info=lambda: SimpleNamespace(image_info=SimpleNamespace(image_id="im-cached"))
            )
        ),
        Image=SimpleNamespace(from_id=lambda image: FakeImage()),
        Volume=SimpleNamespace(from_name=lambda *a, **kw: "bench-volume"),
        Secret=SimpleNamespace(from_name=lambda name: name),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    _, _, functions = campaign.build_app()
    assert set(functions) == set(campaign.ACTIONS)
    assert all(m["copy"] is False for m in mounts)  # No paid image-build path.
    for definition in definitions:
        action = definition["name"]
        assert definition["timeout"] == campaign.ACTIONS[action][1]
        assert definition["retries"] == 0 and definition["startup_timeout"] == 60
        assert definition["single_use_containers"] is True
        assert definition["serialized"] is True
        assert definition["max_containers"] == 1
        assert definition["secrets"] == (["twin-object-storage"] if action == "freeze" else [])
        assert definition["memory"][0] == definition["memory"][1]
        assert definition["cpu"][0] == definition["cpu"][1]

    requested = []

    class FakeVolume:
        def read_file(self, path: str) -> Any:
            requested.append(path)
            if path.endswith("status.json"):
                yield b'{"status":"complete"}'
            else:
                raise FileNotFoundError(path)

    campaign.download(FakeVolume(), "rehearse-beach", tmp_path)
    assert {Path(p).name for p in requested} == set(campaign.DOWNLOADS)
    assert not any(p.endswith((".ply", ".jpg", ".bin")) for p in requested)
    assert (
        json.loads((tmp_path / "rehearse-beach-r2/status.json").read_text())["status"] == "complete"
    )


def test_pooled_upload_whitelist_contains_derived_reports_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import sys

    from experiments.coverage_quality.campaign import upload_pooled

    uploaded = []

    class Batch:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def put_file(self, source: Path, remote: str) -> None:
            uploaded.append((source.relative_to(tmp_path / "pooled").as_posix(), remote))

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Volume=SimpleNamespace(
                from_name=lambda *a, **kw: SimpleNamespace(batch_upload=lambda **opts: Batch())
            )
        ),
    )
    expected = {"fit.json", "provenance.json", "c935/fit_error.png", "beach/fit_error.png"}
    for name in expected | {"trained.ply"}:
        target = tmp_path / "pooled" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
    upload_pooled(tmp_path)
    assert {local for local, _ in uploaded} == expected
    assert all(remote.startswith("research/coverage-quality-") for _, remote in uploaded)


def test_campaign_exact_training_plans_use_trainer_validation() -> None:
    import training
    from experiments.coverage_quality.campaign import ACTIONS, training_plan, validate_training_plan

    for action in ACTIONS:
        if not action.endswith("c935"):
            continue
        plan = training_plan(action, "test")
        params = plan["trainParams"]
        scale = training.frame_schedule(633, params)
        assert validate_training_plan(plan, 633) == training.scaled_steps(
            params["iterations"], scale
        )
        if "schedule_scale" in params:
            assert training.check_schedule_scale(params["schedule_scale"]) == scale
    for invalid in (0.01, 2.0):
        plan = training_plan("main-c935", "test")
        plan["trainParams"]["schedule_scale"] = invalid
        with pytest.raises(ValueError, match="outside"):
            validate_training_plan(plan, 633)


def test_resumed_campaign_budget_includes_previous_spend() -> None:
    from experiments.coverage_quality.campaign import budget

    plan = budget(0.5844)
    assert plan["totalWorstCaseUsd"] < 5
    assert plan["remainingWorstCaseUsd"] <= 4.4156
    assert plan["spentUsd"] == 0.5844
    with pytest.raises(ValueError, match="exceeds"):
        budget(0.93)
    for invalid in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            budget(invalid)


def test_vector_visibility_matches_scalar_and_skips_irrelevant_cameras() -> None:
    from experiments.coverage_quality.calibration import frustum, observing, visible_points
    from experiments.coverage_quality.renderers import Camera, Render

    k = np.array([[8.0, 0, 4], [0, 8, 4], [0, 0, 1]])
    front = Camera("front", np.eye(4), k, 8, 8)
    away_pose = np.diag([-1.0, 1, -1, 1])
    away = Camera("away", away_pose, k, 8, 8)
    points = np.array([[0.0, 0, 2], [0.1, 0, 2], [3.0, 0, 2], [0.0, 0, -2]])
    render = Render(np.zeros((8, 8, 3)), np.ones((8, 8)), np.full((8, 8), 2.0))
    expected = np.stack([observing(p, [front, away], [render, render], 0.05) for p in points])
    actual = visible_points(points, [front, away], {0: render, 1: render}, 0.05)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(frustum(points[:2], [front, away]).any(0), [True, False])


def test_progress_fails_fast_and_lpips_requires_cached_weights(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.coverage_quality.bench_eval import require_lpips_weights
    from experiments.coverage_quality.progress import Progress

    with pytest.raises(FileNotFoundError, match="offline"):
        require_lpips_weights(tmp_path)
    weights = tmp_path / "checkpoints/alexnet-owt-7be5be79.pth"
    weights.parent.mkdir()
    weights.write_bytes(b"baked fixture")
    assert require_lpips_weights(tmp_path) == weights
    progress = Progress(120)
    with progress.phase("setup", "fixture"):
        progress.log("camera metadata loaded")
    output = capsys.readouterr().out
    assert "+0." in output and "setup start" in output and "setup end" in output
    monkeypatch.setattr(
        "experiments.coverage_quality.progress.time.monotonic", lambda: progress.started + 10
    )
    with pytest.raises(TimeoutError, match="projection"):
        progress.estimate(4, 1818, progress.started)


def test_patch_visibility_batches_cache_once_and_skip_backwards_camera() -> None:
    class Counting(NumpyBackend):
        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def render_many(self, scene: Scene, cameras: list[Camera]) -> list[Any]:
            self.batches.append([c.name for c in cameras])
            return [self.render(scene, c) for c in cameras]

    backend = Counting()
    scene = fixture()
    training = [camera(f"t{i}", x) for i, x in enumerate([-0.5, -0.25, 0, 0.25, 0.5])]
    away = camera("away", 0)
    away.c2w[:3, :3] *= [-1, 1, -1]
    training.append(away)
    tests = []
    for index, x in enumerate([0.9, 1.5]):
        cam = camera(f"test{index}", x)
        rendered = backend.render(scene, cam)
        tests.append((cam, rendered.rgb + 0.02, None))
    rows = patch_table(scene, training, tests, backend, max_patches_per_view=64)
    assert len(rows) >= 8
    rendered_names = [name for batch in backend.batches for name in batch]
    assert len(rendered_names) == len(set(rendered_names)) == 5
    assert "away" not in rendered_names
    assert all(len(batch) <= 4 for batch in backend.batches)


def test_gsplat_batch_preserves_settings_and_uploads_scene_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from contextlib import nullcontext

    from experiments.coverage_quality.renderers import GsplatBackend

    uploads = []
    calls = []

    class Tensor:
        def __init__(self, value: Any) -> None:
            self.value = np.asarray(value)

        def detach(self) -> Any:
            return self

        def cpu(self) -> Any:
            return self

        def numpy(self) -> Any:
            return self.value

    def tensor(value: Any, **kwargs: Any) -> Any:
        uploads.append(np.asarray(value).shape)
        return Tensor(value)

    def rasterization(**kwargs: Any) -> Any:
        calls.append(kwargs)
        n = len(kwargs["viewmats"].value)
        h, w = kwargs["height"], kwargs["width"]
        return Tensor(np.zeros((n, h, w, 4))), Tensor(np.ones((n, h, w, 1))), {}

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(as_tensor=tensor, float32=np.float32, no_grad=nullcontext),
    )
    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(rasterization=rasterization))
    backend, scene = GsplatBackend(antialiased=False, packed=True), fixture()
    scene.sh_degree = 3
    cams = [camera(f"v{i}", i * 0.1) for i in range(4)]
    assert len(backend.render_many(scene, cams)) == 4
    assert backend.render(scene, cams[0]).depth.shape == (48, 64)
    assert len(uploads) == 5 + 2 * 2  # Five model arrays once; view matrices/K each call.
    assert calls[0]["means"] is calls[1]["means"]
    for call in calls:
        assert call["packed"] is True
        assert call["sh_degree"] == 3
        assert call["rasterize_mode"] == "classic"
        assert call["render_mode"] == "RGB+ED"


def test_timeout_writes_partial_derived_rows_without_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.coverage_quality import bench_eval

    def fail(args: Any, progress: Any, rows: Any) -> Any:
        rows.append(
            {
                **dict.fromkeys(COLUMNS),
                "view": "heldout",
                "angles_deg": [],
                "patch_i": 0,
                "patch_j": 0,
                "patch_k": 0,
                "observing_views": 0,
            }
        )
        raise TimeoutError("projected time")

    monkeypatch.setattr(bench_eval, "_evaluate", fail)
    args = argparse.Namespace(time_budget=120, label="beach", out=tmp_path)
    with pytest.raises(TimeoutError, match="projected time"):
        bench_eval.evaluate(args)
    report = json.loads((tmp_path / "provenance.json").read_text())
    assert report["status"] == "partial" and report["rows"] == 1
    assert (tmp_path / "patches.csv").is_file()
    assert not (tmp_path / "fit.json").exists()


def test_compacted_partial_table_preserves_kernel_counts(tmp_path: Path) -> None:
    from experiments.coverage_quality.fitting import compact_rows, read_rows

    raw = [
        {
            **dict.fromkeys(COLUMNS),
            "angles_deg": [0.0, 12.0, 27.0],
            "view": "heldout",
            "patch_i": 0,
            "patch_j": 0,
            "patch_k": 0,
        }
    ]
    compact = compact_rows(raw, "beach")
    before = json.dumps(compact)
    compact_rows(compact, "beach")
    assert json.dumps(compact) == before
    compact[0]["views_in_cone"] = 3  # Diagnostic plotting field is not a CSV column.
    write_outputs(compact, tmp_path, fit=False)
    restored = read_rows(tmp_path / "patches.csv")
    assert restored[0]["cone_counts"] == compact[0]["cone_counts"]


def test_over_budget_dispatch_refused_before_modal_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from experiments.coverage_quality import campaign

    def forbid() -> Any:
        pytest.fail("over-budget plan must not contact Modal")

    monkeypatch.setattr(campaign, "build_app", forbid)
    for command in ("plan", "reeval-c935-rehearsal", "main-c935", "main-beach"):
        monkeypatch.setattr(
            sys, "argv", ["campaign", command, "--spent-usd", "0.93", "--local-out", str(tmp_path)]
        )
        with pytest.raises(ValueError, match="exceeds"):
            campaign.main()


def test_reeval_campaign_uses_retained_model_without_training() -> None:
    from experiments.coverage_quality.campaign import ACTIONS, budget, evaluation_argv

    assert set(ACTIONS) == {"reeval-c935-rehearsal", "main-c935", "main-beach"}
    assert ACTIONS["reeval-c935-rehearsal"] == ("l4", 360)
    assert ACTIONS["main-beach"] == ("l4", 900)
    plan = budget(0.5844)
    assert plan["remainingWorstCaseUsd"] == pytest.approx(4.0797696)
    assert plan["totalWorstCaseUsd"] == pytest.approx(4.6641696)
    argv = evaluation_argv("reeval-c935-rehearsal", {})
    assert argv[argv.index("--run") + 1].endswith("coverage-quality-20261125-v2-rehearse-c935-r2")
    assert argv[argv.index("--max-test-frames") + 1] == "4"
    assert argv[argv.index("--label") + 1] == "c935"
    assert "ARKit metric poses (job c9357a92 georeference scaleSource=arkit, scale 1.0)" in argv
    beach = evaluation_argv("main-beach", {})
    assert beach[beach.index("--metres-per-unit") + 1] == "2.232937"
    assert (
        "camera-height estimate from production job 4dd02171 georeference (unverified, not ARKit)"
        in beach
    )


def test_spread_order_covers_time_range_in_every_prefix() -> None:
    from experiments.coverage_quality.selection import spread_test_views

    ordered = spread_test_views([f"f{i:03}" for i in range(91)])
    assert ordered[:4] == ["f000", "f090", "f045", "f022"]
    assert len(ordered) == len(set(ordered)) == 91
    assert spread_test_views(["b", "a"]) == ["a", "b"]


def write_colmap_metadata(root: Path) -> None:
    import struct

    root.mkdir(parents=True)
    with (root / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 3))
        for i in range(3):
            stream.write(struct.pack("<IiQQ4d", i + 1, 1, 64, 48, 65.0, 65.0, 32.0, 24.0))
    with (root / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 3))
        for i in range(3):
            stream.write(
                struct.pack("<IdddddddI", i + 1, 1.0, 0.0, 0.0, 0.0, -float(i), 0.0, 0.0, i + 1)
            )
            stream.write(f"f{i}.jpg".encode() + b"\x00")
            stream.write(struct.pack("<Qddq", 1, 1.0, 2.0, -1))
    # No points3D.bin: metadata reader must not need it.


def test_metadata_reader_skips_points_tracks_and_single_view_has_no_training_frames(
    tmp_path: Path,
) -> None:
    import struct

    from experiments.coverage_quality.views import metadata_cameras, read_views, single_view_source

    frozen = tmp_path / "frozen"
    write_colmap_metadata(frozen / "poses")
    (frozen / "frames").mkdir()
    for i in range(3):
        (frozen / f"frames/f{i}.jpg").write_bytes(b"not RGB; must not decode during setup")
    views = read_views(frozen / "poses")
    cameras = metadata_cameras(views, ["f0.jpg", "f2.jpg"], (1.0, 1.0), {}, 2.0)
    assert len(views.images) == 3 and len(views.cameras) == 3
    np.testing.assert_allclose(cameras[1].centre, [4.0, 0, 0])
    assert (cameras[0].width, cameras[0].height) == (64, 48)
    np.testing.assert_allclose(cameras[0].k, [[65.0, 0, 32.0], [0, 65.0, 24.0], [0, 0, 1.0]])
    single = tmp_path / "single"
    single_view_source(views, "f1.jpg", frozen, single)
    assert [p.name for p in (single / "frames").iterdir()] == ["f1.jpg"]
    selected = read_views(single / "poses")
    assert list(selected.images) == ["f1.jpg"]
    assert list(selected.cameras) == [2]
    assert (single / "poses/points3D.bin").read_bytes() == struct.pack("<Q", 0)
    assert selected.images["f1.jpg"].points2d == 0


def test_training_distortion_metadata_never_allocates_dense_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from dataclasses import replace

    from experiments.coverage_quality.views import metadata_cameras, read_views

    write_colmap_metadata(tmp_path / "poses")
    views = read_views(tmp_path / "poses")
    source = views.cameras[1]
    views.cameras[1] = replace(source, model="OPENCV", params=(*source.params, 0.0, 0.01, 0.0, 0.0))
    calls = []

    def optimal(k: Any, distortion: Any, size: Any, alpha: Any) -> Any:
        calls.append(size)
        return k, (1, 2, size[0] - 2, size[1] - 4)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("metadata features must never build per-pixel maps")

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            setNumThreads=lambda n: None,
            getOptimalNewCameraMatrix=optimal,
            initUndistortRectifyMap=forbidden,
        ),
    )
    cameras = metadata_cameras(views, ["f0.jpg"], (1.0, 1.0), {}, 1.0)
    assert calls == [(64, 48)]
    assert (cameras[0].width, cameras[0].height) == (62, 44)
    assert cameras[0].k[0, 2] == 32.0  # Preserve upstream Parser's perspective crop convention.


def test_per_view_checkpoints_preserve_completed_rows_before_next_decode_failure(
    tmp_path: Path,
) -> None:
    backend, scene = NumpyBackend(), fixture()
    training = [camera("train", 0)]
    saved = []

    def tests() -> Any:
        test = camera("first", 0.9)
        yield test, backend.render(scene, test).rgb + 0.02, None
        raise TimeoutError("next view too slow")

    def checkpoint(rows: Any, name: str) -> None:
        saved.append((name, len(rows)))
        write_outputs(rows, tmp_path)

    with pytest.raises(TimeoutError, match="next view"):
        patch_table(scene, training, tests(), backend, on_view=checkpoint, max_patches_per_view=64)
    assert saved[0][0] == "first" and saved[0][1] >= 8
    fits = json.loads((tmp_path / "fit.json").read_text())
    assert fits["error"]["rows"] == saved[0][1]
    assert (tmp_path / "patches.csv").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_partial_evaluation_refits_only_completed_views_and_records_scale_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from experiments.coverage_quality import bench_eval

    run, frozen, out = tmp_path / "run", tmp_path / "frozen", tmp_path / "out"
    run.mkdir()
    (run / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "trainParams": {"test_every": 0, "ship_sh_degree": 3, "packed": True},
            }
        )
    )
    (run / "frame_selection.json").write_text(
        json.dumps({"train": ["f0.jpg"], "test": ["f1.jpg", "f2.jpg"]})
    )
    (run / "trained.ply").write_bytes(b"unit fixture, PLY reader injected")
    write_colmap_metadata(frozen / "poses")
    trainer = tmp_path / "trainer.py"
    trainer.write_text("# fixture\n")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets/colmap.py").write_text("# pixel adapter fixture\n")
    weights = tmp_path / "checkpoints/alexnet-owt-7be5be79.pth"
    weights.parent.mkdir()
    weights.write_bytes(b"offline weights fixture")
    monkeypatch.setenv("GSPLAT_TRAINER", str(trainer))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda n: None,
            hub=SimpleNamespace(
                get_dir=lambda: str(tmp_path), download_url_to_file=lambda *a: None
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "datasets.colmap", SimpleNamespace(Dataset=object, Parser=object)
    )
    monkeypatch.setitem(
        sys.modules,
        "torchmetrics.image.lpip",
        SimpleNamespace(
            LearnedPerceptualImagePatchSimilarity=lambda **kw: SimpleNamespace(
                cuda=lambda: object()
            )
        ),
    )
    model = fixture()
    model.sh_degree = 3
    monkeypatch.setattr(bench_eval, "scene_from_ply", lambda *a: model)
    monkeypatch.setattr(bench_eval, "image_ratio", lambda *a, **kw: (1.0, 1.0))
    monkeypatch.setattr(bench_eval, "GsplatBackend", lambda **kw: object())
    monkeypatch.setattr(
        bench_eval,
        "plot_fit",
        lambda rows, fits, path: (path / "fit_error.png").write_bytes(b"plot fixture"),
    )

    def partial(*args: Any, **kwargs: Any) -> Any:
        rows = kwargs["rows"]
        for index in range(32):
            angles = [float(index)]
            error = 0.02 + 0.2 * (1 - float(credit(np.cos(np.radians(angles)), 25, 2).sum()))
            rows.append(
                {
                    **dict.fromkeys(COLUMNS),
                    "view": "f1.jpg",
                    "patch_i": index,
                    "patch_j": 0,
                    "patch_k": 0,
                    "pixels": 4,
                    "angles_deg": angles,
                    "nearest_deg": index,
                    "error": error,
                    "l1": error,
                    "lpips": error,
                    "best_mm_px": 1.0,
                    "parallax_deg": 20.0,
                    "observing_views": 1,
                }
            )
        kwargs["on_view"](rows, "f1.jpg")
        rows.append({**rows[0], "view": "f2.jpg", "error": 0.9})
        raise TimeoutError("reserve finalization time")

    monkeypatch.setattr(bench_eval, "patch_table", partial)
    args = argparse.Namespace(
        time_budget=180,
        run=run,
        scene=frozen,
        out=out,
        label="beach",
        max_test_frames=0,
        max_patches=64,
        metres_per_unit=2.232937,
        scale_source="camera-height estimate (unverified, not ARKit)",
    )
    result = bench_eval.evaluate(args)
    assert result["status"] == "partial"
    assert result["completedTestNames"] == ["f1.jpg"] and result["rows"] == 32
    assert result["scaleUncertain"] is True
    assert result["metresPerUnit"] == 2.232937
    fits = json.loads((out / "fit.json").read_text())
    assert fits["error"]["rows"] == 32
    assert "f2.jpg" not in (out / "patches.csv").read_text()


def test_reeval_execute_never_invokes_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from experiments.bench import worker
    from experiments.coverage_quality import campaign

    root = tmp_path / "research"
    for dependency in ("freeze", "rehearse-beach-r2"):
        (root / dependency).mkdir(parents=True)
        (root / dependency / "status.json").write_text('{"status":"complete"}')
    monkeypatch.setattr(campaign, "ROOT", root)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("reevaluation must never train")

    monkeypatch.setattr(worker, "run_training", forbidden)
    calls = []

    def evaluated(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv)
        assert argv[argv.index("--run") + 1].endswith("-rehearse-c935-r2")
        out = Path(argv[argv.index("--out") + 1])
        (out / "provenance.json").write_text('{"status":"complete","rows":16}')
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", evaluated)
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    result = campaign.execute(
        "reeval-c935-rehearsal", {"spentUsd": 0.5844, "commit": "test"}, volume
    )
    assert result["status"] == "complete" and len(calls) == 1
    assert result["worstCaseUsd"] == pytest.approx(0.170232)


def test_image_ratio_reads_only_first_training_header_and_mirrors_resize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from contextlib import nullcontext

    from experiments.coverage_quality.views import image_ratio, read_views

    write_colmap_metadata(tmp_path / "poses")
    views = read_views(tmp_path / "poses")
    opened = []

    def header(path: Path) -> Any:
        opened.append(path.name)
        return nullcontext(SimpleNamespace(size=(64, 48)))

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=SimpleNamespace(open=header)))
    assert image_ratio(views, tmp_path / "frames", 32, reference="f2.jpg") == (0.5, 0.5)
    assert opened == ["f2.jpg"]


def test_bench_calibration_retains_model_and_defers_duplicate_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil
    import subprocess
    import sys

    from experiments.bench import worker
    from experiments.bench.rehearsal import stand_in_params, synthetic_scene

    scene, run = tmp_path / "scene", tmp_path / "run"
    synthetic_scene(scene, "spool", "test")
    params = stand_in_params()
    monkeypatch.setenv("GSPLAT_TRAINER", params["trainer"])
    monkeypatch.setenv("GSPLAT_PYTHON", sys.executable)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(
        worker,
        "GpuSampler",
        lambda unused: SimpleNamespace(samples=[], start=lambda: None, stop=lambda: None),
    )
    real_run = subprocess.run

    def command(argv: list[str], **kwargs: Any) -> Any:
        if len(argv) > 1 and Path(argv[1]).name == "evaluate.py":
            pytest.fail("calibration plan must not invoke the all-camera benchmark evaluator")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", command)
    result = worker.run_training(
        scene,
        run,
        {
            "runId": "calibration",
            "scene": "spool",
            "seed": 42,
            "schedule": "full",
            "variant": {"name": "baseline", "overrides": {}},
            "tier": "l4",
            "benchCommit": "test",
            "timeoutSeconds": 1800,
            "trainParams": params,
            "renders": 0,
            "keepSplat": True,
            "calibrationEvaluationOnly": True,
        },
    )
    assert result["status"] == "complete", result["error"]
    assert (run / "trained.ply").is_file()
    report = result["artifacts"]["test.json"]
    assert report["status"] == "deferred" and report["frames"] == 0
    assert "psnr" not in report and report["perFrame"] == []


def test_calibration_diagnostics_use_view_bootstrap_and_disjoint_bins() -> None:
    from experiments.coverage_quality.analyze_calibration import binned_curve, rank

    x = np.tile([0.0, 1.0], 100)
    y = np.repeat([0.0, 1.0], 100)
    views = np.repeat(["a", "b"], 100)
    curves = binned_curve(x, y, views, np.array([0.0, 1.0, 2.0]), draws=200)
    assert sum(r["rows"] for r in curves) == 200
    for row in curves:
        assert row["views"] == 2 and row["mean"] == 0.5
        assert row["ci_low"] == 0.0 and row["ci_high"] == 1.0
    np.testing.assert_allclose(rank(np.array([2.0, 1.0, 2.0, 4.0])), [2.5, 1.0, 2.5, 4.0])


def test_calibration_variance_repeat_consistency_is_not_noise_ceiling() -> None:
    from experiments.coverage_quality.analyze_calibration import variance

    views = np.tile(["a", "b"], 12)
    patches = np.column_stack([np.repeat(np.arange(12), 2), np.zeros((24, 2))])
    y = np.repeat(np.arange(12, dtype=float), 2) + np.tile([0.0, 1.0], 12)
    result = variance(y, views, patches)
    assert result["between_view_fraction"] + result["within_view_fraction"] == pytest.approx(1.0)
    assert result["repeated_patches"] == 12
    assert result["noise_ceiling"] is None
    assert result["view_demeaned"]["split_half_pearson_median"] == pytest.approx(1.0)
    unique = variance(y, views, np.column_stack([np.arange(24), np.zeros((24, 2))]))
    assert unique["repeat_status"].startswith("too few")


def test_calibration_extension_refuses_missing_angles_and_recovers_narrow_raw_kernel() -> None:
    from experiments.coverage_quality.analyze_calibration import extend_grid

    missing = [{"observing_views": 4, "angles_deg": [], "credit_sums": [1.0] * 24}]
    assert extend_grid(missing)["status"] == "not_identifiable_from_saved_statistics"
    rng = np.random.default_rng(902)
    rows = []
    for index in range(80):
        angles = rng.uniform(0, 10, int(rng.integers(1, 9)))
        support = min(1.0, float(credit(np.cos(np.radians(angles)), 5, 8).sum()) / 2)
        rows.append(
            {
                "view": f"v{index % 4}",
                "patch_i": index,
                "patch_j": 0,
                "patch_k": 0,
                "angles_deg": angles.tolist(),
                "observing_views": len(angles),
                "nearest_deg": float(angles.min()),
                "best_mm_px": 1.0,
                "parallax_deg": 30.0,
                "error": 0.02 + 0.2 * (1 - support),
            }
        )
    extended = extend_grid(rows)
    assert extended["status"] == "exact_raw_angle_refit"
    fit = extended["fit"]
    assert (fit["cone_deg"], fit["sharpness"], fit["views_to_fill"]) == (5, 8, 2)
    assert fit["r2"] > 0.999
    # A local analysis refit must not change the calibration tool's production grid.
    from experiments.coverage_quality import calibration

    module: Any = calibration
    assert module.CONES == (10, 20, 25, 30, 45, 60)
    assert module.SHARPNESS == (0.5, 1, 2, 4)


def test_calibration_partial_ranks_remove_view_confounding() -> None:
    from experiments.coverage_quality.analyze_calibration import correlations

    rng = np.random.default_rng(903)
    views = np.repeat(np.arange(20), 20)
    x = rng.normal(size=(400, 7))
    x[:, 0] += views * 10
    y = views * 10 + x[:, 1] * 0.3 + rng.normal(0, 0.1, 400)
    result = correlations(x, y, views, np.full(400, 10.0))
    assert result[0]["spearman"] > 0.95
    assert abs(result[0]["partial_rank"]) < 0.15
    assert result[1]["partial_rank"] > 0.6


def test_calibration_table_archive_obeys_one_mb_limit(tmp_path: Path) -> None:
    from experiments.coverage_quality.analyze_calibration import copy_table

    source, out = tmp_path / "source.csv", tmp_path / "out"
    out.mkdir()
    source.write_bytes(b"a,b\n1,2\n")
    result = copy_table(source, out, "small")
    assert result["copy"] == "small-patches.csv"
    source.write_bytes(b"a" * 1_000_001)
    result = copy_table(source, out, "compressible")
    assert result["encoding"] == "gzip" and result["gzip_bytes"] < 1_000_000
    raw = np.random.default_rng(904).integers(0, 256, 1_100_000, dtype=np.uint8).tobytes()
    source.write_bytes(raw)
    result = copy_table(source, out, "large")
    assert result["copy"] is None and result["gzip_bytes"] > 1_000_000
    assert source.read_bytes() == raw
    assert not (out / "large-patches.csv").exists()

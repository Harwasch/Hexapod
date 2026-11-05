"""Opt-in stack boundaries and NumPy references; no claim of GPU quality."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import far_field
import gaussian_budget
import live
import prior_maths
import quality
import quality_stack
import quality_trainer
import splat_io
import training


def test_metrics_are_a_versioned_public_vocabulary() -> None:
    assert quality_stack.METRIC_NAMES == (
        "splatCount",
        "farFieldShare",
        "elongationShare",
        "lossTerms",
        "valPsnrCurve",
        "denseInitSeconds",
        "denseInitPoints",
        "denseInitAddedPoints",
        "densePeakMemoryGb",
        "densePeakReservedGb",
        "exportMismatchDb",
        "trainMemoryCurve",
    )
    payload = {"splatCount": 10, "farFieldShare": 0.1, "lossTerms": {"depth": 0.2}}
    import json

    assert live.parse_line("live-metrics: " + json.dumps(payload)) == ("metrics", payload)


def test_defaults_do_not_change_trainer_argv() -> None:
    args = ("python", Path("trainer.py"), Path("dataset"), Path("result"))
    assert training.gsplat_argv(*args) == training.gsplat_argv(
        *args, with_ut=False, with_eval3d=False, rolling_shutter=False, post_processing=None
    )
    assert not quality_stack.options({})["enabled"]


@pytest.mark.parametrize(
    "params",
    [
        {"init": "url"},
        {"init": "sfm+dense"},
        {"strategy": "mcmc", "priors": {"depth_weight": -1}},
        {"strategy": "mcmc", "aniso_cap": {"weight": 0.1}, "batch_size": 2},
    ],
)
def test_settings_fail_before_gpu(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        quality_stack.options(params)


def test_scale_shift_alignment_and_view_facing_normals() -> None:
    x = np.arange(1.0, 101.0).reshape(10, 10)
    scale, shift = prior_maths.scale_shift(x, 2 * x + 3, np.ones_like(x, dtype=bool))
    np.testing.assert_allclose([scale, shift], [2, 3], atol=1e-12)
    z = np.full((7, 9), 5.0)
    normal = prior_maths.depth_normals(z, [[50, 0, 4], [0, 50, 3], [0, 0, 1]])
    np.testing.assert_allclose(normal, np.broadcast_to([0, 0, -1], normal.shape), atol=1e-12)
    assert prior_maths.aniso_penalty(np.log([[1, 2, 20], [1, 2, 3]])) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        prior_maths.scale_shift(np.ones(5), np.ones(5), np.ones(5, dtype=bool))


def test_dense_alignment_rejects_inconsistent_geometry_and_deduplicates() -> None:
    points = np.random.default_rng(42).normal(size=(20, 3))
    mapped, report = prior_maths.aligned_chunk(points, points, points * 1.1 + 2, 5)
    np.testing.assert_allclose(mapped, points * 1.1 + 2, atol=1e-12)
    assert report["inliers"] == 20
    with pytest.raises(ValueError, match="consensus"):
        prior_maths.aligned_chunk(points, points, points * [10, 1, 1], 5)
    assert prior_maths.voxel_select(
        np.array([[0.0, 0, 0], [0.01, 0, 0], [1, 0, 0]]), 0.1, 2
    ).tolist() == [0, 2]


def test_cache_identity_includes_frame_revision_and_geometry(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"a")
    first = prior_maths.cache_key(frame, "r1", [1, 2])
    assert first != prior_maths.cache_key(frame, "r2", [1, 2])
    assert first != prior_maths.cache_key(frame, "r1", [1, 3])
    frame.write_bytes(b"b")
    assert first != prior_maths.cache_key(frame, "r1", [1, 2])


def test_far_budget_caps_geometric_outliers_and_strips_bounds(tmp_path: Path) -> None:
    assert gaussian_budget.far_allocation(100, 0.1, 256) == (90, 10)
    with pytest.raises(ValueError):
        gaussian_budget.far_allocation(None, 0.1, 256)
    points = np.random.default_rng(42).normal(size=(90, 3))
    points = np.concatenate([points, np.full((50, 3), 100)])
    columns = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2], "opacity": np.ones(140)}
    bounded, kept = far_field.export_rows(columns, [0, 0, 0], 2.0, 0.1)
    assert kept.sum() == 100
    assert bounded["far_field"].mean() == pytest.approx(0.1)
    source = tmp_path / "all.ply"
    with splat_io.PlyWriter(source, tuple(bounded)) as writer:
        writer.append(bounded)
    target = far_field.strip(source, tmp_path / "near.ply")
    reader = splat_io.SplatReader(target)
    assert reader.count == 90
    assert "far_field" not in reader.properties
    assert max(float(c["x"].max()) for _, c in reader.chunks()) < 10


def test_fade_retains_rows_and_far_field_but_strict_still_deletes() -> None:
    columns = {
        "x": np.arange(4, dtype=np.float32),
        "opacity": np.zeros(4, dtype=np.float32),
        "far_field": np.array([0, 0, 0, 1], dtype=np.float32),
    }
    tiers = np.array(
        [quality.TIER_KEEP, quality.TIER_CONTEXT, quality.TIER_DROP, quality.TIER_DROP],
        dtype=np.uint8,
    )
    faded = quality.gate(columns, tiers, "fade", context_fade=0.5)
    assert faded["x"].tolist() == [0, 1, 2, 3]
    expected_alpha = np.array([0.5, 0.25, 0.125, 0.5], dtype=np.float64)
    np.testing.assert_array_equal(
        faded["opacity"], np.log(expected_alpha / (1 - expected_alpha)).astype(np.float32)
    )
    assert quality.gate(columns, tiers, "strict")["x"].tolist() == [0, 3]


def test_source_patch_refuses_moved_seams_and_preserves_plain_export(tmp_path: Path) -> None:
    source = "\n".join(quality_trainer.ANCHORS)
    patched = quality_trainer.patch_source(
        source, tmp_path / "settings.json", tmp_path / "trainer.py"
    )
    assert "_hex_quality.loss" in patched
    with pytest.raises(ValueError, match="seam moved"):
        quality_trainer.patch_source(
            source.replace("loss.backward()", "loss.backward(retain_graph=True)"),
            tmp_path / "settings.json",
            tmp_path / "trainer.py",
        )


@pytest.mark.parametrize(
    "flags",
    [
        {"with_ut": True},
        {"strategy": "mcmc", "with_eval3d": True},
        {"strategy": "mcmc", "with_ut": True, "antialiased": True},
        {"rolling_shutter": True},
        {"post_processing": "ppisp", "batch_size": 2},
    ],
)
def test_incompatible_upstream_flags_refused(flags: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        training.gsplat_argv("python", Path("trainer"), Path("dataset"), Path("result"), **flags)


def test_new_post_processing_cli_and_controller_switches() -> None:
    argv = training.gsplat_argv(
        "python",
        Path("trainer"),
        Path("dataset"),
        Path("result"),
        strategy="mcmc",
        with_ut=True,
        with_eval3d=True,
        post_processing="ppisp",
        ppisp_use_controller=False,
    )
    assert argv[-1] != "--use_bilateral_grid"
    assert "--post_processing" in argv and "ppisp" in argv
    assert "--no-ppisp-use-controller" in argv
    assert "--with_ut" in argv and "--with_eval3d" in argv


@pytest.mark.parametrize(
    "variants_file, count", [("quality-stack.yaml", 2), ("quality-stack-diagnostic.yaml", 3)]
)
def test_every_variant_rehearses_without_modal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variants_file: str, count: int
) -> None:
    from experiments.bench import __main__ as cli
    from experiments.bench.recipe import PIPELINE
    from experiments.bench.scene import read_json

    monkeypatch.setattr(cli, "REPO", tmp_path / "not-repo")
    monkeypatch.setattr(cli, "git_commit", lambda: "synthetic-quality-stack")

    def forbidden() -> None:
        raise AssertionError("no Modal in rehearsal")

    monkeypatch.setattr(cli, "load_modal", forbidden)
    assert (
        cli.main(
            [
                "run",
                "--scene",
                "beach",
                "--rehearse",
                "--variants",
                str(PIPELINE / "experiments/variants" / variants_file),
                "--state-dir",
                str(tmp_path),
                "--keep-splat",
                "--cap",
                "10",
            ]
        )
        == 0
    )
    results = list((tmp_path / "rehearsal/runs").glob("*/result.json"))
    assert len(results) == count
    for path in results:
        result = read_json(path)
        assert result["status"] == "complete", result["error"]
        assert result["rehearsal"] and result["costUsd"] == 0
        metrics = read_json(path.parent / "train_metrics.json")
        assert set(quality_stack.METRIC_NAMES) <= metrics.keys()
        assert metrics["farFieldShare"] <= 0.1
        if result["variant"]["name"].startswith("all-on"):
            quality_report = read_json(path.parent / "quality.json")
            assert quality_report
            assert result["qualityGate"] == "fade"
            assert result["headlinePly"] == "faded.ply"
            assert result["testFaded"] == result["testTrained"]  # synthetic, not GPU quality
            assert (path.parent / "faded.ply").is_file()
            assert (path.parent / "trained.ply").is_file()
        else:
            assert result["qualityGate"] is None and result["testFaded"] is None


def test_dense_param_cannot_redirect_bench_inputs() -> None:
    from experiments.bench.recipe import train_params

    assert train_params("screen", {"train": {"init": "sfm+dense"}})["init"] == "sfm+dense"
    with pytest.raises(ValueError):
        train_params("screen", {"train": {"init": "/test/frames"}})
    with pytest.raises(ValueError):
        train_params("screen", {"train": {"init_from": "preview"}})


def test_source_anchors_against_available_pinned_clone() -> None:
    import block_trainer
    import converge_trainer

    clone = Path.home() / ".cache/hexapod-upstream/gsplat/examples/simple_trainer.py"
    if not clone.exists():
        pytest.skip("optional read-only pinned upstream clone is not installed")
    source = clone.read_text()
    assert all(
        anchor in source
        for anchor in (*block_trainer.REQUIRED_SOURCE, *converge_trainer.REQUIRED_SOURCE)
    )
    patched = quality_trainer.patch_source(source, Path("settings.json"), clone)
    compile(patched, str(clone), "exec")
    assert patched.index("sys.path.insert") < patched.index("from datasets.colmap")


def test_prior_cache_hits_never_open_validation_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sys
    from types import SimpleNamespace

    import prior_cache
    import quality_prepass
    import sfm

    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "sparse/0").mkdir(parents=True)
    sfm.write_points3d(
        dataset / "sparse/0/points3D.bin",
        sfm.Points3D(
            ids=np.arange(1, 4, dtype=np.uint64),
            xyz=np.eye(3),
            rgb=np.full((3, 3), 100, dtype=np.uint8),
            error=np.zeros(3),
            track=np.empty((0, 2), dtype=np.uint32),
            track_offsets=np.zeros(4, dtype=np.int64),
        ),
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    # Validation image deliberately does not exist: opening it would fail the test.
    names = ["validation.png", "train1.png", "train2.png"]
    k = np.eye(3)
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    for name in names[1:]:
        path = dataset / "images" / name
        path.write_bytes(name.encode())
        key = prior_maths.cache_key(
            path, quality_stack.MOGE_REVISION, prior_maths.cache_geometry(image, k)
        )
        prior_cache.write(
            cache,
            key,
            {
                "depth": np.ones((4, 5)),
                "normal": np.zeros((4, 5, 3)),
                "mask": np.ones((4, 5), dtype=bool),
            },
        )

    class Dataset:
        indices = np.array([1, 2])

        def __init__(self, parser: Any, split: str) -> None:
            assert split == "train"

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {"K": k, "image": image}

    monkeypatch.setitem(
        sys.modules,
        "datasets.colmap",
        SimpleNamespace(Parser=lambda **kw: SimpleNamespace(image_names=names), Dataset=Dataset),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    settings = {
        **quality_stack.options({"strategy": "mcmc", "priors": {"depth_weight": 0.05}}),
        "dataset": str(dataset),
        "cache": str(cache),
        "trainer": str(tmp_path / "trainer.py"),
        "data_factor": 1,
    }
    report = quality_prepass.run(settings)
    assert report["priorFrames"] == 2
    assert set(json.loads((cache / "manifest.json").read_text())["frames"]) == set(names[1:])


def test_ppisp_controller_must_activate_during_short_schedule() -> None:
    with pytest.raises(ValueError, match="scaled final step"):
        training.gsplat_argv(
            "python",
            Path("trainer"),
            Path("dataset"),
            Path("result"),
            post_processing="ppisp",
            steps_scaler=0.1,
        )
    argv = training.gsplat_argv(
        "python",
        Path("trainer"),
        Path("dataset"),
        Path("result"),
        post_processing="ppisp",
        steps_scaler=0.1,
        ppisp_controller_activation_num_steps=1000,
    )
    assert "1000" in argv


def test_fade_does_not_roundtrip_supported_or_background_opacity() -> None:
    columns = {
        "x": np.arange(3, dtype=np.float32),
        "opacity": np.array([20, 20, -20], dtype=np.float32),
        "far_field": np.array([0, 1, 0], dtype=np.float32),
    }
    tiers = np.array([quality.TIER_KEEP, quality.TIER_DROP, quality.TIER_CONTEXT], dtype=np.uint8)
    result = quality.gate(columns, tiers, "fade")
    np.testing.assert_array_equal(result["opacity"][:2], columns["opacity"][:2])


def test_cache_geometry_changes_with_undistortion_even_if_frame_and_k_match() -> None:
    original = np.zeros((4, 5, 3), dtype=np.uint8)
    changed = original.copy()
    changed[0, 0, 0] = 1
    assert prior_maths.cache_geometry(original, np.eye(3)) != prior_maths.cache_geometry(
        changed, np.eye(3)
    )


def test_placed_bounds_and_origin_ignore_background(tmp_path: Path) -> None:
    import json

    from test_placement import _splat_columns

    from conftest import make_recipe
    from executor import execute
    from runners import LocalRunner, RunnerSet
    from workdir import Workdir

    columns = _splat_columns(100)
    outputs = []
    for include in (False, True):
        work = Workdir.create(tmp_path / str(include))
        path = work.input_path("trained.ply")
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            {k: np.concatenate([v, v[:1]]) for k, v in columns.items()}
            if include
            else dict(columns)
        )
        if include:
            for axis in "xyz":
                rows[axis][-1] = 100_000
            rows["far_field"] = np.r_[np.zeros(100), 1].astype(np.float32)
        with splat_io.PlyWriter(path, tuple(rows)) as writer:
            writer.append(rows)
        work.input_path("georef.json").write_text(
            json.dumps(
                {
                    "lat": 1.0,
                    "lon": 2.0,
                    "height": 0.0,
                    "frame": {
                        "source": "test",
                        "scale": 1.0,
                        "rotation": np.eye(3).tolist(),
                        "translationM": [0.0, 0.0, 0.0],
                        "recentre": True,
                    },
                }
            )
        )
        recipe = make_recipe(
            [{"id": "place", "impl": "place_splat"}], inputs=["trained.ply", "georef.json"]
        )
        execute(recipe, work, RunnerSet(cpu=LocalRunner()))
        outputs.append(
            (
                work.artifact_path("place", "canonical.ply").read_bytes(),
                work.artifact_path("place", "placement.json").read_bytes(),
            )
        )
    assert outputs[0] == outputs[1]


def test_interrupted_shared_cache_publish_keeps_previous_complete_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.bench import worker

    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "key.npz").write_bytes(b"new-complete")
    (target / "key.npz").write_bytes(b"previous-complete")

    def interrupted(src: Path, dst: Path) -> None:
        dst.write_bytes(b"partial")
        raise OSError("interrupted copy")

    monkeypatch.setattr("experiments.bench.worker.shutil.copyfile", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        worker.publish_cache(source, target)
    assert (target / "key.npz").read_bytes() == b"previous-complete"
    assert sorted(p.name for p in target.iterdir()) == ["key.npz"]

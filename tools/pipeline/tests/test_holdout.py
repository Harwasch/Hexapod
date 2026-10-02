"""Measured accuracy: the numpy half of `holdout_error.py`, and the stages around it.

**Nothing here renders a splat or runs on a GPU.** gsplat's rasteriser needs CUDA and no
machine this was written on has one. What is tested:

* the arithmetic `holdout_error.py` runs on the GPU box -- error map, PSNR, per-frame
  diagnostics, accumulation, mean error (`holdout_maths.py`, the same code there);
* the claim the attribution rests on -- that the gradient of a linear colour probe is
  each gaussian's blending weight times the pixel's error -- against a front-to-back
  alpha compositor written here in numpy, by finite differences. That is the compositing
  model gsplat's 3DGS rasteriser implements; that gsplat's backward pass computes it is
  gsplat's to be right about, and is not checked here;
* the `train` stage writing `holdout/` with a stand-in script (`holdout_stand_in.py`,
  named in every test that uses it), aligned with `trained.ply` through its crop, and
  surviving a script that fails;
* the `quality` stage's verdict from given arrays.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import gaussians
import holdout
import holdout_maths
import quality
import sfm
import support_mask
from conftest import make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from synthetic_scene import ring, scene, write_model
from test_train_gsplat import ring_points, seed_inputs, stand_in_params, train_recipe
from workdir import Workdir

HOLDOUT_STAND_IN = Path(__file__).resolve().parent / "holdout_stand_in.py"


# --- the error map ----------------------------------------------------------------------


def _image(seed: int, size: tuple[int, int] = (40, 48)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 0.9, size=(*size, 3)).astype(np.float32)


def test_a_perfect_render_has_no_error_and_ssim_one() -> None:
    image = _image(0)

    np.testing.assert_allclose(holdout_maths.ssim_map(image, image), 1.0, atol=1e-5)
    np.testing.assert_allclose(holdout_maths.error_map(image, image), 0.0, atol=1e-5)
    assert holdout_maths.psnr(image, image) == 99.0


def test_the_error_map_is_the_trainers_loss_per_pixel() -> None:
    """0.8 L1 + 0.2 (1 - SSIM): with the SSIM weight at zero it is exactly the L1."""
    real = _image(1)
    render = np.clip(real + 0.1, 0.0, 1.0)

    l1_only = holdout_maths.error_map(render, real, ssim_weight=0.0)
    np.testing.assert_allclose(l1_only, np.abs(render - real).mean(axis=2), atol=1e-6)
    blended = holdout_maths.error_map(render, real)
    dssim = 1.0 - holdout_maths.ssim_map(render, real)
    np.testing.assert_allclose(blended, 0.8 * l1_only + 0.2 * dssim, atol=1e-5)
    # Noise on a flat wall: SSIM sees structure that is not there; with the ~0.91 that
    # theory gives for the same noise on busy texture (2 cov / (var + var + noise)).
    flat = np.full((40, 48, 3), 0.5, dtype=np.float32)
    noise = np.random.default_rng(2).normal(0, 0.1, flat.shape)
    assert holdout_maths.ssim_map(np.clip(flat + noise, 0, 1), flat).mean() < 0.2
    busy = holdout_maths.ssim_map(np.clip(real + noise, 0, 1), real).mean()
    assert 0.85 < busy < 0.95


def test_psnr_is_ten_log_of_the_inverse_mse() -> None:
    real = np.full((8, 8, 3), 0.5, dtype=np.float32)
    assert holdout_maths.psnr(real + 0.1, real) == pytest.approx(20.0, abs=1e-4)
    mask = np.zeros((8, 8), dtype=bool)
    mask[:2] = True
    render = real.copy()
    render[2:] = 0.0  # wrong only outside the mask
    assert holdout_maths.psnr(render, real, mask) == 99.0


def test_a_frame_says_whether_it_was_brighter_or_blurred() -> None:
    real = _image(3, (64, 64))
    alpha = np.ones((64, 64), dtype=np.float32)
    brighter = holdout_maths.view_stats(
        np.clip(real + 0.1, 0, 1), real, np.zeros((64, 64), np.float32), alpha
    )
    assert brighter["bias"] == pytest.approx(0.1, abs=0.02)
    blurred_frame = holdout_maths.blur(real)
    stats = holdout_maths.view_stats(real, blurred_frame, np.zeros((64, 64), np.float32), alpha)
    assert stats["sharpness"] is not None and stats["sharpness"] < 0.7
    empty = holdout_maths.view_stats(real, real, np.zeros((64, 64), np.float32), alpha * 0)
    assert empty["psnr"] is None and empty["coverage"] == 0.0


# --- attribution ------------------------------------------------------------------------


def _composite(colours: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Front-to-back alpha compositing, as 3DGS rasterises: per pixel, gaussians sorted
    near to far, `C = sum_i c_i a_i T_i`, `T_i = prod_{j<i} (1 - a_j)`.

    `colours` (N, D); `alphas` (P, N), gaussian i's opacity at pixel p, in depth order.
    Returns the image (P, D) and the blending weights `a_i T_i` (P, N).
    """
    transmittance = np.cumprod(
        np.concatenate([np.ones((alphas.shape[0], 1)), 1.0 - alphas[:, :-1]], axis=1), axis=1
    )
    weights = alphas * transmittance
    return weights @ colours, weights


def test_a_linear_colour_probes_gradient_is_error_times_blending_weight() -> None:
    """The identity `holdout_error.attribute` reads off gsplat's backward pass, checked by
    finite differences on the compositing model it differentiates."""
    rng = np.random.default_rng(4)
    pixels, count = 30, 6
    alphas = rng.uniform(0.0, 0.8, size=(pixels, count))
    alphas[:, 5] = 0.0  # a gaussian that paints nothing
    error = rng.uniform(0.0, 0.5, size=pixels)
    probe = np.zeros((count, 2))

    def objective(colours: np.ndarray) -> float:
        image, _ = _composite(colours, alphas)
        return float((image[:, 0] * error).sum() + image[:, 1].sum())

    step = 1e-6
    gradient = np.zeros_like(probe)
    for i in range(count):
        for channel in range(2):
            bumped = probe.copy()
            bumped[i, channel] += step
            gradient[i, channel] = (objective(bumped) - objective(probe)) / step
    _, weights = _composite(probe, alphas)
    np.testing.assert_allclose(gradient[:, 0], (weights * error[:, None]).sum(axis=0), atol=1e-6)
    np.testing.assert_allclose(gradient[:, 1], weights.sum(axis=0), atol=1e-6)

    accumulator = holdout_maths.Accumulator(count)
    accumulator.add(gradient[:, 0], gradient[:, 1])
    mean = accumulator.mean_error()
    expected = (weights * error[:, None]).sum(axis=0)[:5] / weights.sum(axis=0)[:5]
    np.testing.assert_allclose(mean[:5], expected, rtol=1e-4)
    # No weight, no error: NaN, not zero.
    assert np.isnan(mean[5])


def test_a_hidden_gaussian_carries_none_of_the_error_it_is_behind() -> None:
    """Occlusion is in the weights: behind an opaque gaussian, T is zero."""
    alphas = np.array([[0.99, 0.9], [0.99, 0.9]])
    _, weights = _composite(np.zeros((2, 1)), alphas)
    assert weights[:, 1].sum() < 0.02 * weights[:, 0].sum()


def test_frames_accumulate_and_views_count_frames_that_saw_a_gaussian() -> None:
    accumulator = holdout_maths.Accumulator(3)
    accumulator.add([0.2, 0.0, 0.1], [2.0, 0.0, 0.2])
    accumulator.add([0.6, 0.0, -1e-9], [2.0, 0.0, -1e-9])  # float noise is clipped

    np.testing.assert_allclose(accumulator.mean_error(), [0.2, np.nan, 0.5], rtol=1e-6)
    assert accumulator.views.tolist() == [2, 0, 0]
    with pytest.raises(ValueError, match="gaussians"):
        accumulator.add([1.0], [1.0])


def test_the_script_reads_the_plys_spherical_harmonics_back_in_gsplats_order(
    tmp_path: Path,
) -> None:
    """`export_splats` writes shN (N, K, 3) channel-major; `sh_rest` undoes it."""
    from holdout_error import read_ply, sh_rest

    rest = np.arange(2 * 15 * 3, dtype=np.float32).reshape(2, 15, 3)
    flat = rest.transpose(0, 2, 1).reshape(2, -1)
    names = ["x", "y", "z", *[f"f_rest_{i}" for i in range(45)]]
    rows = np.zeros(2, dtype=np.dtype([(name, "<f4") for name in names]))
    for i in range(45):
        rows[f"f_rest_{i}"] = flat[:, i]
    rows["x"] = [1.0, 2.0]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    path = tmp_path / "point_cloud_99.ply"
    path.write_bytes(header.encode() + rows.tobytes())

    columns = read_ply(path)
    assert columns["x"].tolist() == [1.0, 2.0]
    restored = sh_rest(columns)
    assert restored is not None
    np.testing.assert_array_equal(restored, rest)


def test_the_script_renders_the_degree_trained_ply_ships() -> None:
    """`shipped_sh` is what the held-out frames are rendered with: DC, then the first
    bands of the trainer's SH -- the same coefficients the `train` stage's channel-major
    truncation keeps (`harmonics.sources`), so the error measured is of what ships."""
    import harmonics
    from holdout_error import SH_DIMS, shipped_sh

    assert tuple(SH_DIMS) == harmonics.SH_DIMS
    n = 6
    sh0 = np.arange(n * 3, dtype=np.float32).reshape(n, 1, 3)
    rest = np.arange(n * 45, dtype=np.float32).reshape(n, 15, 3) + 1000
    for degree, width in ((0, 1), (1, 4), (2, 9), (3, 16)):
        colours, used = shipped_sh(sh0, rest, degree)
        assert used == degree and colours.shape == (n, width, 3)
        np.testing.assert_array_equal(colours[:, 0], sh0[:, 0])
        np.testing.assert_array_equal(colours[:, 1:], rest[:, : width - 1])
    # Against the stage: the PLY's channel-major f_rest truncated as trained.ply is.
    flat = rest.transpose(0, 2, 1).reshape(n, -1)
    columns = {f"f_rest_{i}": flat[:, i] for i in range(45)}
    shipped = harmonics.truncate(columns, 1)
    as_trained = np.stack(
        [np.stack([shipped[f"f_rest_{c * 3 + j}"] for c in range(3)], axis=1) for j in range(3)],
        axis=1,
    )
    np.testing.assert_array_equal(shipped_sh(sh0, rest, 1)[0][:, 1:], as_trained)
    # Fewer bands than asked: what there is. None at all: DC.
    assert shipped_sh(sh0, rest[:, :3], 3)[1] == 1
    assert shipped_sh(sh0, None, 3)[1] == 0


# --- the rule ---------------------------------------------------------------------------


def _held(
    error: list[float], weight: list[float], views: list[dict[str, object]] | None = None
) -> holdout.HeldOut:
    return holdout.HeldOut(
        error=np.asarray(error, dtype=np.float32),
        weight=np.asarray(weight, dtype=np.float32),
        views=None,
        summary={"status": "ok", "views": 2, "perView": views or []},
    )


KEEP, CONTEXT, DROP = quality.TIER_KEEP, quality.TIER_CONTEXT, quality.TIER_DROP


def test_keep_needs_its_held_out_error_under_the_limit_where_it_was_measured() -> None:
    tiers = np.array([KEEP, KEEP, KEEP, KEEP, KEEP, CONTEXT], dtype=np.uint8)
    held = _held(
        [0.05, 0.06, 0.20, np.nan, 0.50, 0.9],
        [10.0, 10.0, 10.0, 0.0, 1.0, 10.0],
    )

    verdict = holdout.judge(
        tiers, held, max_ratio=2.0, max_abs=None, min_weight=2.0, keep=KEEP, demote_to=CONTEXT
    )

    # The median of measured coverage-keep: 0.05, 0.06, 0.20 -> 0.06; limit 0.12.
    assert verdict.reference == pytest.approx(0.06)
    assert verdict.limit == pytest.approx(0.12)
    assert verdict.tiers.tolist() == [KEEP, KEEP, CONTEXT, KEEP, KEEP, CONTEXT]
    # Unseen (NaN) and barely seen (1 px < 2 px) keep stays keep, on coverage alone.
    assert verdict.verified.tolist() == [True, True, False, False, False, False]
    assert verdict.demoted.tolist() == [False, False, True, False, False, False]
    # Context is not judged by accuracy.
    assert verdict.tiers[5] == CONTEXT


def test_an_absolute_ceiling_catches_a_capture_that_is_bad_all_over() -> None:
    tiers = np.full(4, KEEP, dtype=np.uint8)
    held = _held([0.30, 0.31, 0.32, 0.33], [5.0] * 4)

    relative = holdout.judge(
        tiers, held, max_ratio=2.0, max_abs=None, min_weight=1.0, keep=KEEP, demote_to=CONTEXT
    )
    both = holdout.judge(
        tiers, held, max_ratio=2.0, max_abs=0.25, min_weight=1.0, keep=KEEP, demote_to=CONTEXT
    )
    assert int(relative.demoted.sum()) == 0
    assert both.limit == 0.25 and int(both.demoted.sum()) == 4


def test_the_tip_says_what_failed_when_accuracy_did() -> None:
    bright = [{"name": "a", "bias": 0.09, "sharpness": 1.0}, {"name": "b", "bias": 0.0}]
    blurred = [{"name": "a", "bias": 0.0, "sharpness": 0.5}]
    fine = [{"name": "a", "bias": 0.01, "sharpness": 1.0}]

    assert holdout.accuracy_tip({"perView": bright}, 0.10) is None
    exposure = holdout.accuracy_tip({"perView": bright}, 0.4)
    assert exposure is not None and exposure["id"] == "accuracy-exposure"
    assert "About 40%" in exposure["text"]
    blur = holdout.accuracy_tip({"perView": blurred}, 0.4)
    assert blur is not None and blur["id"] == "accuracy-blur"
    generic = holdout.accuracy_tip({"perView": fine}, 0.4)
    assert generic is not None and generic["id"] == "accuracy"


def test_an_artifact_that_cannot_be_trusted_is_not_used(tmp_path: Path) -> None:
    holdout_maths.write_summary(tmp_path / "failed", {"status": "failed", "reason": "no GPU"})
    held, status = holdout.load(tmp_path / "failed", 3)
    assert held is None and status == {"status": "failed", "reason": "no GPU"}

    holdout_maths.write(tmp_path / "short", [0.1, 0.2], [1.0, 1.0], None, {"status": "ok"})
    held, status = holdout.load(tmp_path / "short", 3)
    assert held is None and status["status"] == "mismatch"

    (tmp_path / "stub").mkdir()
    (tmp_path / "stub" / "holdout.json").write_bytes(b"\x00\x01 stub bytes")
    held, status = holdout.load(tmp_path / "stub", 3)
    assert held is None and status["status"] == "unreadable"

    holdout_maths.write(tmp_path / "ok", [0.1, 0.2, 0.3], [1.0, 1.0, 1.0], None, {"status": "ok"})
    held, status = holdout.load(tmp_path / "ok", 3)
    assert held is not None and status == {"status": "ok"}


# --- train: holdout/ -----------------------------------------------------------------------


def _holdout_params(**overrides: object) -> dict[str, object]:
    return stand_in_params(holdout_error=True, holdout_script=str(HOLDOUT_STAND_IN), **overrides)


def test_train_writes_held_out_arrays_in_trained_plys_order_through_its_crop(
    tmp_path: Path,
) -> None:
    """With stand-ins for the trainer and the held-out script -- not a measurement.

    The stand-in's "error" is a known function of each gaussian's position, so every row
    of `holdout_error.npy` can be checked against the same row of `trained.ply` after the
    stage's ROI crop has removed some of them.
    """
    from holdout_stand_in import made_up_error, seen

    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    sfm.write_points3d(
        workdir.input_path("poses") / "points3D.bin", ring_points(100, images=4, radius=3.0)
    )
    params = _holdout_params(roi={"center": [0.0, 0.0, 0.0], "radius": 1.0})

    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    out = workdir.out_dir("train")
    trained = gaussians.read_splat(out / "trained.ply")
    assert 0 < trained.count < 64  # the crop removed some
    error = np.load(out / "holdout" / "holdout_error.npy")
    weight = np.load(out / "holdout" / "holdout_weight.npy")
    assert error.shape == weight.shape == (trained.count,)
    x, y = trained.xyz[:, 0], trained.xyz[:, 1]
    visible = seen(x)
    np.testing.assert_allclose(error[visible], made_up_error(x, y)[visible], rtol=1e-6)
    assert np.isnan(error[~visible]).all()
    summary = json.loads((out / "holdout" / "holdout.json").read_text())
    assert summary["status"] == "ok"
    assert summary["gaussiansExported"] == 64 and summary["gaussians"] == trained.count
    assert summary["cropped"] is True
    metrics = json.loads((out / "train_metrics.json").read_text())
    assert metrics["holdout"]["status"] == "ok" and metrics["holdout"]["views"] == 2


def test_a_failing_held_out_script_costs_nothing_but_its_own_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOLDOUT_STAND_IN", "fail")
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(train_recipe(_holdout_params()), workdir, RunnerSet(cpu=LocalRunner()))

    out = workdir.out_dir("train")
    assert gaussians.read_splat(out / "trained.ply").count == 64
    summary = json.loads((out / "holdout" / "holdout.json").read_text())
    assert summary["status"] == "failed" and "CalledProcessError" in summary["reason"]
    assert not (out / "holdout" / "holdout_error.npy").exists()
    assert "WARNING: held-out error was not measured" in workdir.log_path("train").read_text()


def test_arrays_of_the_wrong_length_are_refused_not_misaligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOLDOUT_STAND_IN", "short")
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(train_recipe(_holdout_params()), workdir, RunnerSet(cpu=LocalRunner()))

    summary = json.loads((workdir.out_dir("train") / "holdout" / "holdout.json").read_text())
    assert summary["status"] == "failed" and "64 gaussians" in summary["reason"]


def test_the_real_script_without_torch_fails_cleanly_and_training_survives(
    tmp_path: Path,
) -> None:
    """`holdout_error.py` itself, under this interpreter: no torch, so it cannot run --
    the case of a trainer venv that lost a dependency. The stage carries on."""
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(
        train_recipe(stand_in_params(holdout_error=True)),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )

    out = workdir.out_dir("train")
    assert (out / "trained.ply").is_file()
    summary = json.loads((out / "holdout" / "holdout.json").read_text())
    assert summary["status"] == "failed"
    assert str(holdout.SCRIPT) in workdir.log_path("train").read_text()


def test_held_out_error_is_off_unless_asked_for(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(train_recipe(stand_in_params()), workdir, RunnerSet(cpu=LocalRunner()))

    summary = json.loads((workdir.out_dir("train") / "holdout" / "holdout.json").read_text())
    assert summary["status"] == "off"


def test_the_script_is_run_with_the_trainers_interpreter_and_dataset(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(
        train_recipe(_holdout_params(antialiased=True)),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )

    log = workdir.log_path("train").read_text()
    line = next(row for row in log.splitlines() if str(HOLDOUT_STAND_IN) in row)
    assert line.startswith(f"$ {sys.executable} ")
    assert f"--data_dir {workdir.work_dir('train') / 'dataset'}" in line
    assert "--test_every 8" in line and "--antialiased" in line


# --- quality: the verdict --------------------------------------------------------------------


def _seed_quality(workdir: Workdir, *, n: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    columns, fringe = scene(n)
    trained = workdir.input_path("trained.ply")
    trained.parent.mkdir(parents=True, exist_ok=True)
    gaussians.write_ply(trained, columns)
    write_model(workdir.input_path("poses"), ring(24))
    workdir.input_path("georef.json").write_text(json.dumps({"lat": 1.0, "lon": 2.0}))
    return columns, fringe


def _run_quality(workdir: Workdir, params: dict[str, object] | None = None) -> dict[str, object]:
    inputs = ["trained.ply", "poses", "georef.json"]
    if workdir.input_path("holdout").exists():
        inputs.append("holdout")
    execute(
        make_recipe(
            [{"id": "quality", "impl": "support_gate", "params": params or {}}], inputs=inputs
        ),
        workdir,
        RunnerSet.local(),
    )
    document: dict[str, object] = json.loads(
        (workdir.out_dir("quality") / "quality.json").read_text()
    )
    return document


def test_quality_keeps_only_what_held_out_frames_confirm_where_they_saw_it(
    tmp_path: Path,
) -> None:
    """One side of the object is wrong in the held-out frames; a band is unseen by them."""
    workdir = Workdir.create(tmp_path / "run")
    columns, _ = _seed_quality(workdir, n=20_000)
    x, y = columns["x"], columns["y"]
    error = np.full(x.shape, 0.05, dtype=np.float32)
    wrong = x > 0.1
    error[wrong] = 0.4
    weight = np.full(x.shape, 10.0, dtype=np.float32)
    unseen = (y < -0.1) & ~wrong
    error[unseen] = np.nan
    weight[unseen] = 0.0
    bright = [{"name": "frame_0000.jpg", "psnr": 20.0, "bias": 0.12, "sharpness": 1.0}]
    holdout_maths.write(
        workdir.input_path("holdout"),
        error,
        weight,
        None,
        {"status": "ok", "views": 1, "perView": bright, "meanPsnr": 20.0},
    )

    document = _run_quality(workdir, {"bar": "strict"})
    baseline_dir = Workdir.create(tmp_path / "baseline")
    _seed_quality(baseline_dir, n=20_000)
    baseline = _run_quality(baseline_dir, {"bar": "strict"})

    held = document["heldOut"]
    assert isinstance(held, dict) and held["status"] == "ok"
    coverage_keep = baseline["gaussians"]["keep"]  # type: ignore[index]
    counts = document["gaussians"]
    assert isinstance(counts, dict)
    # Everything coverage kept on the wrong side left keep; nothing else did.
    assert held["demotedFromKeep"] > 0
    assert counts["keep"] == coverage_keep - held["demotedFromKeep"]
    assert held["keepVerified"] + held["keepGeometryOnly"] == counts["keep"]
    assert held["keepGeometryOnly"] > 0  # the band no held-out frame saw
    assert held["limit"] == pytest.approx(0.1, rel=1e-3)
    assert held["tiers"]["keep"]["medianError"] == pytest.approx(0.05)
    gated = gaussians.read_splat(workdir.out_dir("quality") / "gated.ply")
    assert float(gated.xyz[:, 0].max()) <= 0.1
    # Verified keep is a smaller share of the scene than keep.
    assert document["keepVerifiedPct"] is not None
    assert document["keepVerifiedPct"] < document["keepPct"]  # type: ignore[operator]
    # The support mask was built from the verified keep: the unseen band is outside it.
    assert held["supportMaskFrom"] == "verified-keep"
    mask = support_mask.SupportMask.parse(document["supportMask"])
    assert mask is not None
    band = np.stack([x, y, columns["z"]], axis=1)[unseen & (y < -0.3) & (columns["z"] < 0.45)]
    assert band.size and not mask.contains(band).all()
    # Accuracy failed, and the held-out frame was brighter: the tip says exposure, first.
    tips = document["tips"]
    assert isinstance(tips, list) and tips[0]["id"] == "accuracy-exposure"


def test_quality_without_held_out_error_is_coverage_alone_and_says_so(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    _seed_quality(workdir, n=5_000)

    document = _run_quality(workdir)

    assert document["keepVerifiedPct"] is None
    held = document["heldOut"]
    assert isinstance(held, dict) and held["status"] == "missing"
    assert held["supportMaskFrom"] == "keep"


def test_quality_ignores_held_out_arrays_for_another_splat(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    _seed_quality(workdir, n=5_000)
    holdout_maths.write(
        workdir.input_path("holdout"), np.zeros(10), np.ones(10), None, {"status": "ok"}
    )

    document = _run_quality(workdir)

    held = document["heldOut"]
    assert isinstance(held, dict) and held["status"] == "mismatch"
    assert document["keepVerifiedPct"] is None

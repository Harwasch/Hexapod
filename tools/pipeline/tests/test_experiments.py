"""The experiment runner's and the benchmark's pure parts: no network, no API, no GPU."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest

from experiments import benchmark, run_variants
from experiments.run_variants import PlanError

VARIANTS = Path(__file__).resolve().parents[1] / "experiments" / "variants"
REPO = Path(__file__).resolve().parents[3]


# --- variant files --------------------------------------------------------------------------


def test_variants_are_merged_over_the_base_stage_by_stage() -> None:
    plan = run_variants.load_plan(
        """
capture: abc
base:
  normalize: {max_side: 1600, fps: 4}
  train: {schedule_scale: 1.0, antialiased: true}
variants:
  - name: baseline
  - name: more-frames
    params: {normalize: {fps: 8}, train: {antialiased: null}}
"""
    )
    assert plan.capture == "abc"
    assert plan.recipe == "photo-reconstruct"
    assert plan.baseline == "baseline"
    baseline, more = plan.variants
    assert baseline.params == {
        "normalize": {"max_side": 1600, "fps": 4},
        "train": {"schedule_scale": 1.0, "antialiased": True},
    }
    # fps replaced, max_side kept; `null` takes a base setting away.
    assert more.params == {
        "normalize": {"max_side": 1600, "fps": 8},
        "train": {"schedule_scale": 1.0},
    }


def test_repeat_makes_numbered_copies_and_a_bare_list_is_a_plan() -> None:
    plan = run_variants.load_plan('[{"name": "baseline", "repeat": 3}]')
    assert [v.name for v in plan.variants] == ["baseline", "baseline#2", "baseline#3"]
    assert plan.capture is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("variants: []", "non-empty"),
        ("variants: [{name: 'bad name'}]", "must be 1-64"),
        ("variants: [{name: a}, {name: a}]", "unique"),
        ("variants: [{name: a, params: {train: 3}}]", "mapping of params"),
        ("variants: [{name: a, repeat: 9}]", "repeat"),
        ("variants: [{name: a, oops: 1}]", "unknown keys"),
        ("baseline: b\nvariants: [{name: a}]", "not one of the variants"),
        ("surprise: 1\nvariants: [{name: a}]", "unknown keys"),
    ],
)
def test_a_variant_file_that_cannot_run_is_refused_by_name(text: str, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        run_variants.load_plan(text)


def test_the_spool_table_file_parses_and_stays_on_the_phones_whitelist() -> None:
    """Every stage and param it sends is one the phone key may set -- read from the API's
    own PHONE_OPTIONS source, so a variant refused at 2 a.m. is caught here instead."""
    plan = run_variants.load_plan((VARIANTS / "spool-table.yaml").read_text())
    assert plan.capture == "87d772f0-92c9-49be-8570-3c830425d3a3"
    names = [v.name for v in plan.variants]
    assert names[:2] == ["baseline", "baseline#2"]
    for wanted in ("antialiased", "opacity-reg-0.001", "depth-loss", "pose-opt"):
        assert wanted in names
    assert "bilateral-grid" in names and "fps-8" in names and "all-of-the-recipe" in names

    source = (REPO / "apps" / "api" / "app" / "api" / "v1" / "phone.py").read_text()
    block = source[source.index('"photo-reconstruct": {') : source.index('"splat-ingest": {')]
    for variant in plan.variants:
        for stage, params in variant.params.items():
            assert f'"{stage}": {{' in block, (variant.name, stage)
            for name in params:
                assert f'"{name}"' in block, (variant.name, stage, name)
    combined = next(v for v in plan.variants if v.name == "all-of-the-recipe").params
    assert combined["train"] == {
        "schedule_scale": 1.0,
        "antialiased": True,
        "opacity_reg": 0.001,
        "depth_loss": True,
        "pose_opt": True,
        "bilateral_grid": True,
    }


# --- reading jobs back and the table --------------------------------------------------------


def job_document(**overrides: Any) -> dict[str, Any]:
    """A `JobRead` as the API serialises it (camelCase, Decimal cost as a string)."""
    document: dict[str, Any] = {
        "id": "11111111-1111-1111-1111-111111111111",
        "status": "complete",
        "error": None,
        "durationS": 2712.4,
        "costUsd": "0.4512",
        "params": {"train": {"schedule_scale": 1.0}},
        "steps": [
            {"stageId": "normalize", "status": "complete", "metrics": {"durationS": 31.0}},
            {
                "stageId": "pose",
                "status": "complete",
                "metrics": {"durationS": 612.5, "stageCostUsd": 0.043, "tier": "cpu4"},
            },
            {
                "stageId": "train",
                "status": "complete",
                "metrics": {
                    "durationS": 1830.0,
                    "stageCostUsd": 0.4001,
                    "tier": "l4",
                    "psnr": 23.1,
                    "ssim": 0.771,
                    "lpips": 0.184,
                    "gaussians": 500000,
                    "iterations": 30000,
                },
            },
            {
                "stageId": "quality",
                "status": "complete",
                "metrics": {"durationS": 88.0, "stageCostUsd": 0.0061, "keepPct": 41.5},
            },
        ],
    }
    document.update(overrides)
    return document


def test_a_job_becomes_a_row_of_what_it_recorded() -> None:
    row = run_variants.summarise("baseline", job_document(), wall_s=2800.04)

    assert row["status"] == "complete"
    assert row["wallS"] == 2800.0
    assert row["jobDurationS"] == 2712.4
    assert row["jobCostUsd"] == 0.4512
    assert row["stageCostUsd"] == round(0.043 + 0.4001 + 0.0061, 4)
    assert row["stageSeconds"] == {
        "normalize": 31.0,
        "pose": 612.5,
        "train": 1830.0,
        "quality": 88.0,
    }
    assert (row["psnr"], row["ssim"], row["lpips"]) == (23.1, 0.771, 0.184)
    assert row["gaussians"] == 500000 and row["keepPct"] == 41.5
    assert row["ccPsnr"] is None and row["heldOutPsnr"] is None
    assert row["trainTier"] == "l4"


def test_a_job_that_failed_before_training_has_no_quality_rather_than_zero() -> None:
    failed = job_document(
        status="error",
        error="pose registered 3 of 87",
        costUsd=None,
        steps=[{"stageId": "pose", "status": "error", "metrics": {"durationS": 90.0}}],
    )
    row = run_variants.summarise("pose-opt", failed)

    assert row["psnr"] is None and row["gaussians"] is None and row["stageCostUsd"] is None
    assert row["jobCostUsd"] is None


def test_the_table_carries_deltas_against_the_baseline_and_says_what_failed() -> None:
    rows = [
        run_variants.summarise("baseline", job_document(), wall_s=2800),
        run_variants.summarise(
            "pose-opt",
            job_document(
                steps=[
                    {
                        "stageId": "train",
                        "status": "complete",
                        "metrics": {"psnr": 23.6, "ssim": 0.781, "lpips": 0.179},
                    }
                ]
            ),
            wall_s=3000,
        ),
        run_variants.summarise("depth-loss", job_document(status="error", error="boom")),
    ]
    text = run_variants.markdown(rows, baseline="baseline")

    assert "| `baseline` | complete | 23.10 | 0.771 | 0.184 |" in text
    assert "23.60 (+0.50)" in text
    assert "0.781 (+0.010)" in text
    assert "0.179 (-0.005)" in text
    assert "| normalize | pose | train | quality |" in text
    assert "`depth-loss` (error): boom" in text
    assert "`pose-opt`: `" in text


# --- the runner against a fake API -----------------------------------------------------------


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_the_runner_waits_out_a_busy_capture_then_polls_to_the_end() -> None:
    calls: list[tuple[str, str, Any]] = []
    polls = iter(["in-progress", "in-progress", "complete"])
    busy = [True]

    def opener(request: urllib.request.Request, timeout: float) -> FakeResponse:
        body = json.loads(request.data) if isinstance(request.data, bytes) else None
        calls.append((request.get_method(), request.full_url, body))
        assert request.get_header("Authorization") == "Bearer k"
        if request.get_method() == "POST":
            if busy[0]:
                busy[0] = False
                raise urllib.error.HTTPError(
                    request.full_url,
                    409,
                    "Conflict",
                    {},  # type: ignore[arg-type]
                    io.BytesIO(b'{"detail": "capture c already has job j queued or running"}'),
                )
            return FakeResponse(json.dumps({"id": "j1", "status": "not-started"}).encode())
        return FakeResponse(json.dumps(job_document(id="j1", status=next(polls))).encode())

    api = run_variants.Api("https://api.example", "k", opener=opener)
    clock = iter(range(0, 10_000, 10))
    row = run_variants._run_one(
        api,
        "c",
        "photo-reconstruct",
        run_variants.Variant("baseline", {"train": {"pose_opt": True}}),
        poll_s=0,
        job_timeout_s=1e9,
        busy_timeout_s=1e9,
        sleep=lambda _s: None,
        clock=lambda: float(next(clock)),
    )

    posts = [call for call in calls if call[0] == "POST"]
    assert len(posts) == 2  # refused once as busy, then accepted
    assert posts[-1][1] == "https://api.example/api/v1/phone/captures/c/process"
    assert posts[-1][2] == {"recipe": "photo-reconstruct", "params": {"train": {"pose_opt": True}}}
    assert row["status"] == "complete" and row["psnr"] == 23.1
    assert row["params"] == {"train": {"pose_opt": True}}


def test_a_refused_variant_is_recorded_not_retried() -> None:
    def opener(request: urllib.request.Request, timeout: float) -> FakeResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            409,
            "Conflict",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"detail": "A phone cannot set train.trainer."}'),
        )

    api = run_variants.Api("https://api.example", "k", opener=opener)
    row = run_variants._run_one(
        api,
        "c",
        "photo-reconstruct",
        run_variants.Variant("x", {"train": {"trainer": "/bin/sh"}}),
        poll_s=0,
        job_timeout_s=1,
        busy_timeout_s=1,
        sleep=lambda _s: None,
    )
    assert row["status"] == "refused"
    assert "cannot set" in row["error"]


def test_the_key_is_required_and_the_url_must_be_http() -> None:
    with pytest.raises(ValueError, match="TWIN_PHONE_KEY"):
        run_variants.Api("https://api.example", "")
    with pytest.raises(ValueError, match="http"):
        run_variants.Api("file:///etc/passwd", "k")


def test_a_dry_run_prints_the_merged_params_and_touches_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_variants.main(["--variants", str(VARIANTS / "spool-table.yaml"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 11
    assert out.startswith("baseline: ")


# --- the benchmark ---------------------------------------------------------------------------


def test_the_published_numbers_are_the_papers_and_cite_it() -> None:
    truck = benchmark.SCENES["truck"]
    assert (truck.published.psnr, truck.published.ssim, truck.published.lpips) == (
        25.187,
        0.879,
        0.148,
    )
    assert truck.published.url == "https://arxiv.org/abs/2308.04079"
    assert truck.published.lpips_net == "vgg"
    assert truck.params["extra_args"] == ["--lpips_net", "vgg"]


def test_a_spec_is_a_config_and_switches_and_nothing_else() -> None:
    assert benchmark.parse_spec("") == ("gsplat-default", {})
    assert benchmark.parse_spec("recipe+pose_opt+bilateral_grid") == (
        "recipe",
        {"pose_opt": True, "bilateral_grid": True},
    )
    with pytest.raises(ValueError, match="config"):
        benchmark.parse_spec("fast")
    with pytest.raises(ValueError, match="switch"):
        benchmark.parse_spec("gsplat-default+trainer")


def test_train_params_keep_the_scenes_lpips_net_under_any_switches() -> None:
    truck = benchmark.SCENES["truck"]
    params = benchmark.train_params(truck, "recipe", {"pose_opt": True})
    assert params == {
        "data_factor": 1,
        "extra_args": ["--lpips_net", "vgg"],
        "iterations": 30_000,
        "strategy": "mcmc",
        "cap_max": 500_000,
        "pose_opt": True,
    }
    assert "schedule_full_at" not in params  # the full schedule, never scaled down


@pytest.mark.parametrize(
    ("psnr", "config", "switches", "verdict"),
    [
        (25.3, "gsplat-default", {}, "match"),
        (24.8, "gsplat-default", {}, "match"),
        (24.4, "gsplat-default", {}, "below"),
        (23.9, "gsplat-default", {}, "fail"),
        (None, "gsplat-default", {}, "fail"),
        (22.0, "recipe", {}, "reference"),
        (22.0, "gsplat-default", {"pose_opt": True}, "reference"),
    ],
)
def test_the_verdict_judges_only_the_run_the_paper_describes(
    psnr: float | None, config: str, switches: dict[str, Any], verdict: str
) -> None:
    metrics = {"psnr": psnr, "ssim": 0.87, "lpips": 0.15, "heldOut": {"valFrames": 32}}
    result = benchmark.compare(benchmark.SCENES["truck"], config, metrics, switches)
    assert result["verdict"] == verdict
    text = benchmark.markdown(result, cost={"tier": "l4", "billedSeconds": 1500, "usd": 0.333})
    assert "https://arxiv.org/abs/2308.04079" in text
    assert "| PSNR |" in text and "25.187" in text
    assert "$0.333" in text


def fake_archive(path: Path, *, images: int = 3) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for index in range(images):
            bundle.writestr(f"tandt/truck/images/{index + 1:06d}.jpg", b"jpeg")
        for name in ("cameras.bin", "images.bin", "points3D.bin", "project.ini"):
            bundle.writestr(f"tandt/truck/sparse/0/{name}", name.encode())
        bundle.writestr("tandt/train/images/000001.jpg", b"other scene")
        bundle.writestr("db/playroom/sparse/0/cameras.bin", b"other scene")
    return path


def test_extraction_takes_the_scene_and_only_the_colmap_model(tmp_path: Path) -> None:
    archive = fake_archive(tmp_path / "tandt_db.zip")
    frames, poses = benchmark.extract(
        archive, benchmark.SCENES["truck"], tmp_path / "inputs", verify=False
    )
    assert sorted(p.name for p in frames.iterdir()) == ["000001.jpg", "000002.jpg", "000003.jpg"]
    assert sorted(p.name for p in poses.iterdir()) == ["cameras.bin", "images.bin", "points3D.bin"]


def test_an_archive_that_is_not_the_pinned_one_is_refused(tmp_path: Path) -> None:
    archive = fake_archive(tmp_path / "tandt_db.zip")
    with pytest.raises(ValueError, match="not the"):
        benchmark.extract(archive, benchmark.SCENES["truck"], tmp_path / "inputs")


def test_the_benchmark_rehearses_end_to_end_with_the_stand_in(tmp_path: Path) -> None:
    """The Modal driver, minus Modal: extraction, the one-stage recipe, the cloud runner
    over a local bucket, the stand-in trainer, and the report. Proves the plumbing only."""
    archive = fake_archive(tmp_path / "tandt_db.zip", images=9)
    work = tmp_path / "work"
    done = subprocess.run(
        [
            sys.executable,
            str(REPO / "infra" / "modal" / "benchmark.py"),
            "--rehearse",
            "--archive",
            str(archive),
            "--work",
            str(work),
            "--spec",
            "gsplat-default+bilateral_grid",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    (run,) = work.glob("bench-truck-*")
    result = json.loads((run / "benchmark.json").read_text())
    assert result["verdict"] == "reference"  # a switch is on: not the paper's run
    assert result["switches"] == {"bilateral_grid": True}
    assert result["settings"]["bilateralGrid"] is True
    assert result["valFrames"] == 2  # 9 frames: indices 0 and 8 held out
    log = (run / "stages" / "train" / "log.txt").read_text()
    assert "--lpips_net vgg" in log and "--use_bilateral_grid" in log
    assert "Rehearsal" in (run / "benchmark.md").read_text()

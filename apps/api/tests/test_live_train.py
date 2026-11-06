"""The heartbeat's bounded merge and the API's derived events, without Postgres."""

from __future__ import annotations

import copy
import json
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pytest
from sqlalchemy.orm import Session

from app.models import Capture, Job, JobStep
from app.models.enums import RunStatus
from app.schemas.live import LiveTrain
from app.services.live import _state, _train
from app.services.live_train import derive_events, end_reason, final_metrics, merge_live
from app.storage import NullStorage


@pytest.mark.parametrize(
    ("errors", "preempted", "expected"),
    [
        (
            [
                "InternalFailure: Runner killed (SIGKILL), exit code: 137. Most often this "
                "happens when the process runs out of memory."
            ],
            True,
            "oom",
        ),
        (["torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"], False, "oom"),
        (["CUDA error: out of memory"], True, "oom"),
        (["exit code: 137"], False, "oom"),
        (["exit code 137"], True, "oom"),
        (["SIGKILL: exceeded host memory limit"], True, "oom"),
        (["OOMKilled"], False, "oom"),
        (["OUT OF MEMORY"], False, "oom"),
        (["OutOfMemoryError"], False, "oom"),
        (
            ["the process running the recipe exited with code 137 without reporting a result"],
            False,
            "oom",
        ),
        (
            [
                "recipe 'photo-reconstruct', stage 'train' (impl 'gsplat') failed on "
                "'modal': part b0 preempted on 'modal': InternalFailure: exit code: 137"
            ],
            True,
            "oom",
        ),
        ([None, "GPU reclaimed"], True, "preempted"),
        (["ValueError: invalid training input"], False, "failed"),
        (["SIGKILL"], False, "failed"),
        (
            ["the process running the recipe exited with code -9 without reporting a result"],
            False,
            "failed",
        ),
        (["memory usage: 32 GB"], False, "failed"),
        (["exit code 1370"], False, "failed"),
        ([None, ""], False, "failed"),
        ([], True, "preempted"),
        (["plain job error", "OutOfMemoryError"], True, "oom"),
        (["exit code 137", "plain step error"], False, "oom"),
    ],
)
def test_end_reason_reads_oom_before_preemption(
    errors: list[str | None], preempted: bool, expected: str
) -> None:
    assert end_reason(errors, preempted) == expected


@pytest.mark.parametrize(
    ("job_error", "step_error", "preempted", "reason"),
    [
        ("InternalFailure: exit code: 137", None, True, "oom"),
        ("plain job error", "torch.cuda.OutOfMemoryError", True, "oom"),
        ("GPU reclaimed", None, True, "preempted"),
        ("plain error", None, False, "failed"),
        ("GPU reclaimed", None, False, "failed"),
    ],
)
def test_live_state_uses_recorded_errors_and_preemption(
    job_error: str, step_error: str | None, preempted: bool, reason: str
) -> None:
    capture = Capture(id=uuid.uuid4(), name="Bench tree")
    job = Job(id=uuid.uuid4(), status=RunStatus.ERROR, recipe="photo-reconstruct", error=job_error)
    step = JobStep(
        stage_id="train",
        impl="gsplat",
        ordinal=1,
        status=RunStatus.ERROR,
        preempted_at=datetime.now(UTC) if preempted else None,
        metrics={
            "error": step_error,
            "live": merge_live({}, [sample(15400), ("train", {"k": "e", "step": 15600})]),
        },
    )
    db = Mock(spec=Session)
    db.scalars.return_value.all.return_value = [step]

    state = _state(db, NullStorage(), job, capture)

    assert state.train is not None and state.train.ended is not None
    assert state.train.ended.reason == reason
    assert state.train.ended.step == 15600
    assert state.error == job_error


def sample(step: int, run: str | None = "a", **fields: Any) -> tuple[str, dict[str, Any]]:
    return (
        "train",
        {
            "v": 1,
            "k": "s",
            "run": run,
            "step": step,
            "t": step / 10,
            "gs": 100 + step,
            "lr": 0.00016,
            "sh": min(3, step // 1000),
            "its": 10,
            "mem": 2.3,
            "loss": {"total": 0.1, "future": 0.01234567},
            **fields,
        },
    )


def snapshot(step: int) -> tuple[str, dict[str, Any]]:
    return (
        "splat",
        {
            "v": 1,
            "step": step,
            "key": f"runs/r/train/checkpoint/live/s{step}.spz",
            "count": 100,
            "of": 200,
            "bytes": 123,
        },
    )


def test_overlapping_tails_dedupe_and_restart_replaces_snapshots_and_stop() -> None:
    records = [
        ("train", {"k": "p", "run": "a", "maxSteps": 60000}),
        sample(0),
        snapshot(100),
        sample(200),
    ]
    merged = merge_live({}, records)
    original = copy.deepcopy(merged)
    assert merge_live(merged, records) == merged
    assert merged == original
    newer = merge_live(
        merged, [*records, ("train", {"k": "p", "run": "b", "maxSteps": 30000}), sample(0, "b")]
    )
    assert newer["train"]["series"]["step"] == [0]
    assert newer["snapshots"] == [] and "splat" not in newer
    assert newer["train"]["plan"]["stepsMax"] == 30000
    assert (
        merge_live(
            newer, [*records, ("train", {"k": "p", "run": "b", "maxSteps": 30000}), sample(0, "b")]
        )
        == newer
    )


def test_restart_inside_a_tail_and_block_switch_replace_series() -> None:
    first = merge_live({}, [sample(0), sample(200)])
    other = merge_live(first, [sample(100, "b", block="b1")])
    assert other["train"]["series"]["step"] == [100]
    assert other["train"]["block"] == "b1"
    legacy = merge_live({}, [sample(1000, None)])
    assert merge_live(legacy, [sample(0, None)])["train"]["series"]["step"] == [0]


def test_long_run_is_bounded_and_keeps_endpoints_evals_and_components() -> None:
    live = merge_live({}, [("plan", {"stepsMax": 60000})])
    for end in range(1000, 60001, 1000):
        records = [sample(i) for i in range(max(0, end - 1200), end + 1, 20)]
        records.append(
            (
                "train",
                {
                    "k": "e",
                    "run": "a",
                    "step": end,
                    "psnr": 24.1234567,
                    "gain": 0.012345,
                    "decision": "improving",
                },
            )
        )
        live = merge_live(live, records)
    series = live["train"]["series"]
    assert len(series["step"]) <= 300
    assert series["step"][0] == 0 and series["step"][-1] == 60000
    assert len(live["train"]["evals"]) == 60
    assert series["loss"]["future"][-1] == 0.0123
    assert len(json.dumps(live).encode()) < 48000
    assert max(b - a for a, b in zip(series["step"], series["step"][1:], strict=False)) < 600
    LiveTrain.model_validate(live["train"])


def test_event_values_and_savings_use_last_speed_and_known_rate() -> None:
    live = merge_live(
        {},
        [
            (
                "plan",
                {
                    "stepsMax": 30000,
                    "cap": 1100,
                    "refineStop": 1000,
                    "usdPerHour": 1.2,
                    "frames": 40,
                    "sparse": 100,
                },
            ),
            sample(0),
            sample(1000),
            sample(2000),
            ("train", {"k": "e", "run": "a", "step": 2000, "psnr": 24, "gain": 0.02}),
            ("train", {"k": "x", "run": "a", "step": 2001, "maxSteps": 30000, "gain": 0.02}),
        ],
    )
    events = derive_events(live["train"])
    assert {e["kind"] for e in events} == {"setup", "sh", "cap", "densify", "eval", "stop"}
    assert events[-1]["stepsSaved"] == 27999
    assert events[-1]["usdSaved"] == pytest.approx(27999 / 10 / 3600 * 1.2)
    del live["train"]["plan"]["usdPerHour"]
    assert derive_events(live["train"])[-1]["usdSaved"] is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        42,
        {"series": {"step": [1], "gs": []}},
        {"evals": [{"step": 1, "psnr": float("nan")}]},
        {"plan": {"metresPerUnit": -1}},
    ],
)
def test_bad_stored_data_is_none(value: Any) -> None:
    assert _train({"train": value}, NullStorage()) is None


def test_optional_quality_and_unsigned_snapshots_survive_without_storage() -> None:
    live = merge_live({}, [sample(200), snapshot(100)])
    live["snapshots"][0].update(farFieldShare=0.2, elongationShare=0.05)
    parsed = _train(live, NullStorage())
    assert parsed is not None
    assert parsed.snapshots[0].url is None
    assert parsed.snapshots[0].far_field_share == 0.2
    assert parsed.snapshots[0].elongation_share == 0.05
    live["snapshots"][0]["key"] = "runs/r/checkpoint/live/../secret.spz"
    assert _train(live, NullStorage()) is None


def test_legacy_restart_with_both_attempts_in_one_tail_discards_earlier_points() -> None:
    tail = [sample(0, None), snapshot(100), sample(400, None), sample(0, None), sample(200, None)]
    merged = merge_live({}, tail)
    assert merged["train"]["series"]["step"] == [0, 200]
    assert merged["snapshots"] == []
    assert merge_live(merged, tail) == merged


def test_unwrapped_preview_restart_replaces_snapshots_too() -> None:
    first = [("plan", {"id": "a", "stepsMax": 3000}), snapshot(1000)]
    live = merge_live({}, first)
    repeated = merge_live(live, first)
    assert repeated == live
    tail = [*first, ("plan", {"id": "b", "stepsMax": 3000}), snapshot(300)]
    restarted = merge_live(live, tail)
    assert [row["step"] for row in restarted["snapshots"]] == [300]
    assert merge_live(restarted, tail) == restarted


def test_missing_writer_loss_stays_empty_columns() -> None:
    merged = merge_live({}, [sample(200, loss=None)])
    assert merged["train"]["series"]["loss"] == {}
    assert merged["train"]["series"]["step"] == [200]


def test_block_switch_keeps_splats_but_replaces_chart_history() -> None:
    first = merge_live(
        {},
        [
            ("plan", {"id": "plan-a", "stepsMax": 30000}),
            sample(200, block="b0"),
            snapshot(200),
            ("train", {"k": "e", "run": "a", "block": "b0", "step": 200, "psnr": 20}),
            ("train", {"k": "x", "run": "a", "block": "b0", "step": 200}),
            ("train", {"k": "r", "run": "a", "block": "b0", "step": 200, "count": 10}),
            ("metrics", {"step": 200, "farFieldShare": 0.1}),
        ],
    )
    tail = [("train", {"k": "p", "run": "b", "block": "b1"}), sample(0, "b", block="b1")]
    switched = merge_live(first, tail)
    assert switched["snapshots"] == first["snapshots"]
    assert switched["splat"] == first["splat"]
    assert switched["train"]["series"]["step"] == [0]
    assert switched["train"]["evals"] == switched["train"]["relocations"] == []
    assert "stop" not in switched["train"] and "quality" not in switched["train"]
    assert switched["train"]["planId"] == "plan-a"
    assert merge_live(switched, tail) == switched
    parsed = _train(switched, NullStorage())
    assert parsed is not None and parsed.plan_id == "plan-a"
    # A switch can arrive in the same heartbeat as the preceding block's last snapshot.
    combined = merge_live(first, [snapshot(400), *tail])
    assert [row["step"] for row in combined["snapshots"]] == [200, 400]
    assert combined["splat"]["step"] == 400
    retry = merge_live(switched, [sample(0, "c", block="b1")])
    assert retry["snapshots"] == [] and "splat" not in retry
    new_plan = merge_live(switched, [("plan", {"id": "plan-b"}), sample(0, "c", block="b2")])
    assert new_plan["snapshots"] == [] and "splat" not in new_plan


def test_quality_metrics_keep_only_numeric_readouts_and_reset() -> None:
    merged = merge_live(
        {},
        [
            (
                "metrics",
                {
                    "step": 200,
                    "farFieldShare": 0.2,
                    "elongationShare": 0.03,
                    "lossTerms": {"depth": 1},
                    "valPsnrCurve": [20],
                },
            )
        ],
    )
    assert {k: v for k, v in merged["train"]["quality"].items() if k != "series"} == {
        "step": 200,
        "farFieldShare": 0.2,
        "elongationShare": 0.03,
    }
    parsed = _train(merged, NullStorage())
    assert parsed is not None and parsed.quality is not None
    assert parsed.quality.elongation_share == 0.03
    invalid = merge_live(
        merged,
        [
            (
                "metrics",
                {
                    "step": "bad",
                    "farFieldShare": True,
                    "elongationShare": float("nan"),
                },
            )
        ],
    )
    assert invalid["train"]["quality"] == merged["train"]["quality"]
    restarted = merge_live(merged, [("plan", {"id": "new"})])
    assert "quality" not in restarted["train"]


@pytest.mark.parametrize(
    "stored",
    [
        {"train": {"series": [1]}},
        {"train": {"evals": [1]}},
        {"snapshots": [{"step": 1}]},
    ],
)
def test_reporting_malformed_history_never_fails_training(
    stored: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    from sqlalchemy.orm import Session

    from app.models import JobStep
    from app.models.enums import RunStatus
    from app.worker import steps
    from app.worker.steps import report_live

    # A mock, not caplog: Alembic's fileConfig disables existing loggers when a migration
    # test ran earlier in the session.
    logger = Mock()
    monkeypatch.setattr(steps, "logger", logger)
    db = Mock(spec=Session)
    step = JobStep(status=RunStatus.IN_PROGRESS, metrics={"live": stored})
    before = copy.deepcopy(step.metrics)
    assert report_live(db, step, {}, records=[]) is False
    assert step.metrics == before
    db.commit.assert_not_called()
    logger.warning.assert_called_once()


def test_real_bench_metrics_and_both_caps_survive() -> None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "tools/pipeline/tests/fixtures/run6fb-train.log"
    records = [
        (line.split(": ", 1)[0][5:], json.loads(line.split(": ", 1)[1]))
        for line in path.read_text().splitlines()
    ]
    live = merge_live({}, records)
    assert path.stat().st_size < 40000
    train = live["train"]
    assert train["plan"]["cap"] == 3456214
    assert train["plan"]["capMax"] == 3455958
    assert train["series"]["gs"][-1] == 3456214
    assert train["quality"]["series"]["peakAllocatedGb"][-1] == 5.9109
    assert train["quality"]["series"]["loss"]["depth"][-1] == 0.0537
    assert len(live["snapshots"]) == 2
    assert merge_live(live, records) == live
    assert _train(live, NullStorage()) is not None


def test_quality_series_and_twenty_snapshots_stay_under_the_json_budget() -> None:
    live: dict[str, Any] = {}
    for step in range(0, 60001, 200):
        records = [
            sample(step),
            (
                "metrics",
                {
                    "step": step,
                    "farFieldShare": 0.14,
                    "elongationShare": 0.06,
                    "lossTerms": {
                        k: 0.1234567 for k in ("depth", "normal", "gradient", "anisotropy")
                    },
                    "trainMemoryCurve": [{"step": step, "peakAllocatedGb": 5.912345}],
                    "valPsnrCurve": [{"step": step, "psnr": 25.123456}],
                    "denseInitPoints": 800000,
                    "denseInitAddedPoints": 160000,
                },
            ),
        ]
        if step and step % 3000 == 0:
            records.append(snapshot(step))
        live = merge_live(live, records)
    assert len(live["snapshots"]) == 20
    assert len(json.dumps(live)) < 48000
    for series in (live["train"]["series"], live["train"]["quality"]["series"]):
        assert len(series["step"]) <= 300
        assert series["step"][0] == 0 and series["step"][-1] == 60000
    assert live["train"]["quality"]["denseInitAddedPoints"] == 160000
    assert len(live["train"]["evals"]) == 256
    assert _train(live, NullStorage()) is not None


def test_seeds_before_trainer_entry_are_signed_and_reset_only_on_retry() -> None:
    from unittest.mock import Mock

    from app.storage import ObjectStorage

    seeds = {
        "v": 1,
        "key": "runs/r/train/checkpoint/live/seeds.bin",
        "count": 2,
        "of": 10,
        "sources": {"sfm": 7, "dense": 0, "seed": 3},
        "origin": [0, 0, 0],
        "scale": 1,
        "bytes": 20,
    }
    live = merge_live(
        {}, [("plan", {"id": "a"}), ("seeds", seeds), ("train", {"k": "p", "run": "a"}), sample(0)]
    )
    assert live["train"]["seeds"] == seeds
    storage = Mock(spec=ObjectStorage)
    storage.head_object.return_value = object()
    storage.presign_get.return_value = "https://bucket/seeds.bin"
    parsed = _train(live, storage)
    assert (
        parsed is not None
        and parsed.seeds is not None
        and parsed.seeds.url == "https://bucket/seeds.bin"
    )
    storage.head_object.assert_called_once_with(seeds["key"])
    storage.head_object.return_value = None
    pending = _train(live, storage)
    assert pending is not None and pending.seeds is not None and pending.seeds.url is None
    assert "seeds" not in merge_live(live, [sample(0, "b")])["train"]
    assert merge_live(live, [sample(0, "b", block="b1")])["train"]["seeds"] == seeds
    live["train"]["seeds"]["key"] = "captures/private.bin"
    parsed = _train(live, storage)
    assert parsed is not None and parsed.seeds is None


def test_final_scores_come_from_the_train_step_metrics() -> None:
    final = final_metrics(
        {
            "psnr": 26.5,
            "ssim": 0.9,
            "lpips": 0.2,
            "ccPsnr": 27,
            "ccSsim": 0.92,
            "ccLpips": 0.18,
            "durationS": 100,
        }
    )
    parsed = _train({}, NullStorage(), final=final)
    assert parsed is not None and parsed.final is not None
    assert parsed.final.cc_psnr == 27 and parsed.final.ssim == 0.9
    assert final_metrics({"psnr": float("nan"), "ssim": True}) is None


def test_cap_event_uses_the_lower_resolved_cap_and_includes_equality() -> None:
    train = merge_live(
        {},
        [("plan", {"cap": 1200}), ("train", {"k": "p", "run": "a", "capMax": 1100}), sample(1000)],
    )["train"]
    event = next(event for event in derive_events(train) if event["kind"] == "cap")
    assert event["step"] == 1000 and event["text"] == "Cap reached · 1,100"


def test_job_training_never_claims_a_bench_replacement() -> None:
    parsed = _train({"train": {"replacedBy": "fa4f62cf183e45e89358cae2b4627e41"}}, NullStorage())
    assert parsed is not None and parsed.replaced_by is None

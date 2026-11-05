"""The heartbeat's bounded merge and the API's derived events, without Postgres."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from app.schemas.live import LiveTrain
from app.services.live import _train
from app.services.live_train import derive_events, merge_live
from app.storage import NullStorage


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
    assert merged["train"]["quality"] == {
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
    assert invalid["train"]["quality"] == {}
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


def test_partial_memory_metrics_do_not_clear_quality_readouts() -> None:
    quality = {"step": 200, "farFieldShare": 0.1, "elongationShare": 0.2}
    merged = merge_live({}, [("metrics", quality)])
    updated = merge_live(merged, [("metrics", {"trainMemoryCurve": [{"step": 1000}]})])
    assert updated["train"]["quality"] == quality

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.bench import ledger


def plan(run: str = "one", **kwargs: Any) -> dict[str, Any]:
    return {"runId": run, "scene": "spool", "schedule": "screen", "tier": "l4", **kwargs}


def test_cap_refuses_entire_batch_without_appending(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger.reserve(path, [plan()], 2)
    ledger.finish(path, "one", cost=1.7, failed=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="nothing launched"):
        ledger.reserve(path, [plan("two"), plan("three")], 2)
    assert path.read_bytes() == before


def test_unfinished_plans_and_unknown_failures_keep_reservation(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger.reserve(path, [plan()], 2)
    entries = ledger.read(path)
    reserve = entries[0]["reservedUsd"]
    assert ledger.summary(entries)["totalUsd"] == reserve
    ledger.finish(path, "one", cost=None, failed=True)
    assert ledger.summary(ledger.read(path))["totalUsd"] == reserve
    # Older/minimal records with just an estimate retain that estimate.
    assert ledger.accounted([{**plan(), "event": "planned", "estimateUsd": 0.2}]) == {"one": 0.2}


def test_estimates_prefer_max_actual_for_matching_scene_schedule_tier(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    for i, cost in enumerate((0.12, 0.15, 0.13)):
        ledger.reserve(path, [plan(str(i))], 2)
        ledger.finish(path, str(i), cost=cost, failed=False)
    entries = ledger.read(path)
    assert ledger.estimate(entries, "spool", "screen", "l4") == 0.15
    assert ledger.estimate(entries, "pumpkin", "screen", "l4") == 0.2
    assert ledger.estimate(entries, "spool", "full", "l4") == 0.9
    assert ledger.estimate(entries, "spool", "screen", "a100") > 0.2
    assert ledger.estimate(entries, "spool", "freeze", "cpu4") == 0.05


@pytest.mark.parametrize(
    "line",
    [
        "garbage",
        "",
        "{}",
        '{"runId":"x","event":"failed"}',
        '{"runId":"x","event":"planned","estimateUsd":NaN}',
    ],
)
def test_corrupt_line_fails_closed(tmp_path: Path, line: str) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(line + "\n")
    with pytest.raises(ValueError, match="corrupt ledger line 1"):
        ledger.reserve(path, [plan()], 2)
    assert path.read_text() == line + "\n"


def test_timeout_reservation_catches_cheap_forecast_and_includes_cpu_memory() -> None:
    plans = [plan("a", schedule="full"), plan("b", schedule="full")]
    # The forecasts ($1.80) fit a $1.81 cap; the two 55-min timeouts do not.
    with pytest.raises(ValueError, match="timeout reservations"):
        ledger.price_plan([], plans, 1.81)
    priced = ledger.price_plan([], plans, 2)
    assert priced["estimateUsd"] == 1.8
    assert plans[0]["timeoutSeconds"] == 3300
    assert plans[0]["worstCaseUsd"] == pytest.approx(ledger.hourly_rate("l4") * 3540 / 3600)
    assert ledger.hourly_rate("l4") == pytest.approx(0.7992 + 2 * 0.04716 + 8 * 0.007992)


def test_failed_actual_is_charged_and_duplicate_outcome_refused(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger.reserve(path, [plan()], 2)
    ledger.finish(path, "one", cost=0.1, failed=True)
    assert ledger.summary(ledger.read(path))["totalUsd"] == 0.1
    with pytest.raises(ValueError, match="duplicate outcome"):
        ledger.finish(path, "one", cost=0, failed=True)
    assert len([json.loads(line) for line in path.read_text().splitlines()]) == 2

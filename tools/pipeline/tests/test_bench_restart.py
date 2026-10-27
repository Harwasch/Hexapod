"""Modal re-runs a preempted GPU input (retries=0 does not stop it): one paid attempt per run."""

from pathlib import Path

import pytest

from experiments.bench import ledger
from experiments.bench.evaluate import lookup_frames, summarise
from experiments.bench.worker import RestartedError, billed_from, claim, restart_result


def test_claim_writes_a_marker_and_refuses_a_second_attempt(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "one"
    claim(run, attempt_started=100.0)
    assert (run / "started.json").is_file()
    with pytest.raises(RestartedError, match="already started"):
        claim(run, attempt_started=200.0)


def test_a_restarted_attempt_is_recorded_as_preempted_with_unknown_cost() -> None:
    plan = {"runId": "one", "scene": "spool", "seed": 42, "schedule": "full"}
    result = restart_result(plan, RestartedError("already started"))
    assert result["status"] == "preempted"
    # Unknown cost keeps the ledger's whole worst-case reservation for this run.
    assert result["costUsd"] is None
    assert "already started" in result["error"]


def test_reservation_covers_one_attempt_plus_a_refused_restart() -> None:
    plans = [{"runId": r, "scene": "spool", "schedule": "full", "tier": "l4"} for r in ("a", "b")]
    ledger.price_plan([], plans, 2)
    seconds = 3300 + ledger.OVERHEAD_SECONDS + ledger.RESTART_SECONDS
    assert plans[0]["worstCaseUsd"] == pytest.approx(ledger.hourly_rate("l4") * seconds / 3600)
    assert ledger.RESTART_SECONDS >= ledger.STARTUP_SECONDS


def test_cold_start_is_billed_even_when_the_container_start_is_unknown() -> None:
    assert billed_from(started_at=90.0, booted_at=100.0, first_call=True, now=500.0) == (
        90.0,
        "container-uptime",
    )
    since, source = billed_from(started_at=None, booted_at=100.0, first_call=True, now=500.0)
    assert since == 100.0 - ledger.STARTUP_SECONDS and source == "boot-minus-startup-limit"
    assert billed_from(started_at=90.0, booted_at=100.0, first_call=False, now=500.0) == (
        500.0,
        "warm-call",
    )


def test_summarise_ignores_missing_values_and_reports_none_when_all_missing() -> None:
    rows = [
        {"psnr": 20.0, "psnrFullSh": None},
        {"psnr": 22.0, "psnrFullSh": None},
    ]
    assert summarise(rows, ("psnr", "psnrFullSh")) == {"psnr": 21.0, "psnrFullSh": None}


def test_lookup_frames_maps_test_names_and_refuses_missing_or_duplicates() -> None:
    names = ["a.jpg", "b.jpg", "c.jpg"]
    assert lookup_frames(names, [0, 1, 2], ["c.jpg", "a.jpg"]) == {"a.jpg": 0, "c.jpg": 2}
    with pytest.raises(ValueError, match="missing"):
        lookup_frames(names, [0, 1, 2], ["z.jpg"])
    with pytest.raises(ValueError, match="duplicated"):
        lookup_frames(names, [0, 1, 2], ["a.jpg", "a.jpg"])
    with pytest.raises(ValueError, match="no test"):
        lookup_frames(names, [0, 1, 2], [])

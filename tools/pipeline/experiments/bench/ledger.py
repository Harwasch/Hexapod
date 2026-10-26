"""Append-only spending records, with locked all-or-nothing batch reservations.

Estimates are forecasts, not limits. Reserve at least the function timeout's price as
well, so four concurrent stragglers cannot spend four estimates beyond the cap.
Unknown outcomes retain that reservation. Corrupt ledgers fail closed.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from modal_adapter import GPU_NAMES, reservation_rate
from providers import provider

# Full: Spool's production train stage took 2,197 s on an L4 (job ec644db5); 55 min leaves
# room for evaluation while two full seeds still reserve under the default $2 cap.
TIMEOUTS = {"screen": 1800, "full": 3300, "freeze": 600}
# Startup/teardown allowance in the modelled reservation, not a measured invoice.
STARTUP_SECONDS = 60
OVERHEAD_SECONDS = STARTUP_SECONDS + 60


def hourly_rate(tier: str) -> float:
    modal = provider("modal")
    rate = None if modal is None else modal.rate(tier)
    if rate is None or tier not in {*GPU_NAMES, "cpu4"}:
        raise ValueError(f"unpriced/unsupported Modal tier: {tier}")
    reservation = reservation_rate(tier)
    return rate.usd_per_hour + (0 if reservation is None else reservation.usd_per_hour)


def money(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("cost must be a finite nonnegative number")
    return float(value)


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text()
    if text and not text.endswith("\n"):
        raise ValueError("corrupt ledger: incomplete final line; refusing paid work")
    entries = []
    planned: set[str] = set()
    ended: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            entry = json.loads(line)
            run = entry["runId"]
            state = entry["event"]
            if not isinstance(run, str) or not run:
                raise ValueError("missing run id")
            if state == "planned":
                if run in planned:
                    raise ValueError("duplicate plan")
                planned.add(run)
                money(entry["estimateUsd"])
                money(entry.get("reservedUsd", entry["estimateUsd"]))
                for key in ("scene", "schedule", "tier"):
                    if not isinstance(entry[key], str):
                        raise ValueError(f"invalid {key}")
            elif state in {"finished", "failed"}:
                if run not in planned or run in ended:
                    raise ValueError("orphan or duplicate outcome")
                ended.add(run)
                if entry.get("costUsd") is not None:
                    money(entry["costUsd"])
            else:
                raise ValueError("unknown event")
            entries.append(entry)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"corrupt ledger line {line_number}; refusing paid work") from error
    return entries


def accounted(entries: list[dict[str, Any]]) -> dict[str, float]:
    charges = {}
    for entry in entries:
        if entry["event"] == "planned":
            charges[entry["runId"]] = money(entry.get("reservedUsd", entry["estimateUsd"]))
        elif entry.get("costUsd") is not None:
            charges[entry["runId"]] = money(entry["costUsd"])
    return charges


def estimate(entries: list[dict[str, Any]], scene: str, schedule: str, tier: str) -> float:
    plans = {e["runId"]: e for e in entries if e["event"] == "planned"}
    actuals = [
        money(e["costUsd"])
        for e in entries
        if e["event"] != "planned"
        and e.get("costUsd") is not None
        and not plans[e["runId"]].get("rehearsal")
        and all(
            plans[e["runId"]][k] == v
            for k, v in (("scene", scene), ("schedule", schedule), ("tier", tier))
        )
    ]
    if actuals:
        return max(actuals)
    if schedule == "freeze":
        return 0.05
    return {"screen": 0.20, "full": 0.90}[schedule] * hourly_rate(tier) / hourly_rate("l4")


def price_plan(
    entries: list[dict[str, Any]], plans: list[dict[str, Any]], cap: float
) -> dict[str, Any]:
    money(cap)
    for plan in plans:
        forecast = estimate(entries, plan["scene"], plan["schedule"], plan["tier"])
        timeout = TIMEOUTS[plan["schedule"]]
        maximum = hourly_rate(plan["tier"]) * (timeout + OVERHEAD_SECONDS) / 3600
        plan.update(
            estimateUsd=forecast,
            timeoutSeconds=timeout,
            worstCaseUsd=maximum,
            reservedUsd=max(forecast, maximum),
        )
        if plan.get("rehearsal"):
            plan.update(estimateUsd=0.0, worstCaseUsd=0.0, reservedUsd=0.0)
    spent = sum(accounted(entries).values())
    forecast_total = sum(p["estimateUsd"] for p in plans)
    reserved = sum(p["reservedUsd"] for p in plans)
    if spent + forecast_total > cap or spent + reserved > cap:
        raise ValueError(
            f"cap ${cap:.4f} exceeded: accounted ${spent:.4f} + "
            f"estimates ${forecast_total:.4f}, timeout reservations ${reserved:.4f}; "
            "nothing launched"
        )
    return {
        "accountedUsd": spent,
        "estimateUsd": forecast_total,
        "reservedUsd": reserved,
        "capUsd": cap,
        "plans": plans,
    }


@contextmanager
def locked(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append(handle: TextIO, entry: dict[str, Any]) -> None:
    handle.write(json.dumps(entry, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def reserve(path: Path, plans: list[dict[str, Any]], cap: float) -> None:
    with locked(path) as handle:
        entries = read(path)
        price_plan(entries, plans, cap)
        known = {e["runId"] for e in entries}
        if any(p["runId"] in known for p in plans) or len({p["runId"] for p in plans}) != len(
            plans
        ):
            raise ValueError("duplicate run IDs")
        for plan in plans:
            _append(handle, {**plan, "event": "planned"})


def finish(
    path: Path, run: str, *, cost: float | None, failed: bool, error: str | None = None
) -> None:
    if cost is not None:
        money(cost)
    with locked(path) as handle:
        entries = read(path)
        if not any(e["runId"] == run and e["event"] == "planned" for e in entries):
            raise ValueError("outcome without a plan")
        if any(e["runId"] == run and e["event"] != "planned" for e in entries):
            raise ValueError("duplicate outcome")
        _append(
            handle,
            {
                "runId": run,
                "event": "failed" if failed else "finished",
                "costUsd": cost,
                "error": error,
            },
        )


def summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    charges = accounted(entries)
    groups: dict[str, float] = defaultdict(float)
    for e in entries:
        if e["event"] == "planned":
            groups[f"{e['scene']}/{e['schedule']}"] += charges[e["runId"]]
    return {
        "bySceneSchedule": dict(groups),
        "totalUsd": sum(charges.values()),
        "note": "includes reservations for missing or unpriced outcomes",
    }

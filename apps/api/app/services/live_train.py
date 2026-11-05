"""Bounded training history and derived events; no database or storage dependencies.

The heartbeat rereads an overlapping tail. A wrapper UUID distinguishes attempts, so a
retry's step zero replaces the prior series without treating an old tail as a restart.
Columns keep JSON compact; time buckets retain an even spread, plus both endpoints.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

MAX_POINTS = 300
MAX_EVALS = 256
MAX_BYTES = 48_000
COLUMNS = ("step", "t", "gs", "lr", "sh", "its", "mem")


def _number(value: Any, digits: int = 4) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return value if isinstance(value, int) else round(value, digits)


def _spread(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    return [rows[round(i * (len(rows) - 1) / (limit - 1))] for i in range(limit)]


def _rows(series: dict[str, Any]) -> list[dict[str, Any]]:
    steps = series.get("step", [])
    loss = series.get("loss", {})
    return [
        {
            **{key: series.get(key, [None] * len(steps))[i] for key in COLUMNS},
            "loss": {key: values[i] for key, values in loss.items()},
        }
        for i in range(len(steps))
    ]


def _columns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in (row.get("loss") or {})})
    return {
        **{
            key: [_number(row.get(key), 10 if key == "lr" else 4) for row in rows]
            for key in COLUMNS
        },
        "loss": {key: [_number((row.get("loss") or {}).get(key)) for row in rows] for key in keys},
    }


def merge_live(
    current: dict[str, Any], records: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    """Reset charts on a block switch, but discard retained splats only on an attempt retry."""
    result = copy.deepcopy(current)
    train = result.get("train") or {}
    if not isinstance(train, dict):
        train = {}
    plans = [(index, value) for index, (kind, value) in enumerate(records) if kind == "plan"]
    plan_id = plans[-1][1].get("id") if plans else None
    plan_reset = plan_id is not None and plan_id != train.get("planId")
    if plan_id is not None:
        # A preview may have no trainer hook. Its stage-plan UUID still identifies a
        # fresh attempt, including a provider retry below the stage attempt counter.
        records = records[plans[-1][0] :]
    incoming = [value for kind, value in records if kind == "train"]
    latest = incoming[-1] if incoming else None
    identity = None if latest is None else (latest.get("run"), latest.get("block"))
    old_identity = (train.get("run"), train.get("block"))
    block_change = latest is not None and latest.get("block") != train.get("block")
    retry = plan_reset or bool(
        latest and not block_change and identity != old_identity and any(identity or ())
    )
    boundary = 0
    previous_sample: dict[str, Any] | None = None
    for index, (kind, value) in enumerate(records):
        if (
            kind == "train"
            and value.get("k") == "s"
            and (value.get("run"), value.get("block")) == identity
        ):
            if previous_sample is not None and value["step"] < previous_sample["step"]:
                boundary = index
            previous_sample = value
    retry = retry or boundary > 0
    # Compatibility with producers predating UUIDs: time and step must both go back.
    samples = [value for value in incoming if value.get("k") == "s"]
    old_steps = train.get("series", {}).get("step", [])
    old_times = train.get("series", {}).get("t", [])
    if samples and old_steps and old_times and not any(identity or ()):
        end = samples[-1]
        retry = retry or (
            end["step"] < old_steps[-1] and (end.get("t") or 0) < (old_times[-1] or 0)
        )
    reset = retry or block_change
    if reset:
        train = {
            "plan": {} if plan_reset else train.get("plan", {}),
            "planId": train.get("planId"),
        }
    if retry:
        result.pop("splat", None)
        result["snapshots"] = []
    if plan_id is not None:
        train["planId"] = plan_id
    if identity is not None:
        train["run"], train["block"] = identity
    # Ignore the earlier attempt's snapshot records still present in this tail.
    for i, (kind, value) in enumerate(records):
        if (
            kind == "train"
            and value.get("k") == "p"
            and (value.get("run"), value.get("block")) == identity
        ):
            boundary = max(boundary, i)
    try:
        rows = _rows(train.get("series", {}))
    except (TypeError, IndexError, AttributeError):
        rows = []
    by_step = {row["step"]: row for row in rows}
    evals = {row["step"]: row for row in train.get("evals", [])}
    relocations = {row["step"]: row for row in train.get("relocations", [])}
    snapshots = {row["key"]: row for row in result.get("snapshots", [])}
    plan = train.get("plan") or {}
    for index, (kind, value) in enumerate(records):
        if kind == "cameras":
            result["cameras"] = value
        elif kind == "plan":
            # Resolved trainer fields win over repeated stage-plan lines.
            plan = {**value, **plan} if not reset else {**plan, **value}
        elif kind == "splat" and (index >= boundary or (block_change and not retry)):
            snapshots[value["key"]] = value
            result["splat"] = value
        elif kind == "metrics" and index >= boundary:
            train["quality"] = {
                **{key: val for key, val in train.get("quality", {}).items() if key not in value},
                **{
                    key: numeric
                    for key in ("step", "farFieldShare", "elongationShare")
                    if (numeric := _number(value.get(key))) is not None
                    and (
                        isinstance(numeric, int) and numeric >= 0
                        if key == "step"
                        else 0 <= numeric <= 1
                    )
                },
            }
        elif kind == "train":
            if index < boundary or (value.get("run"), value.get("block")) != identity:
                continue
            event = value.get("k")
            if event == "p":
                plan.update(
                    {
                        key: val
                        for key, val in value.items()
                        if key not in ("k", "v", "run", "block")
                    }
                )
                if value.get("maxSteps") is not None:
                    plan["stepsMax"] = value["maxSteps"]
                if value.get("capMax") is not None:
                    plan["cap"] = value["capMax"]
            elif event == "s":
                by_step[value["step"]] = value
            elif event == "e":
                evals[value["step"]] = {
                    key: (_number(value.get(key)) if key in ("psnr", "gain") else value.get(key))
                    for key in ("step", "psnr", "gain", "decision")
                }
            elif event == "x":
                train["stop"] = {key: value.get(key) for key in ("step", "maxSteps", "gain")}
            elif event == "r":
                relocations[value["step"]] = {key: value.get(key) for key in ("step", "count")}
    rows = sorted(by_step.values(), key=lambda row: row["step"])
    if len(rows) > MAX_POINTS:
        width = max(1, math.ceil((plan.get("stepsMax") or rows[-1]["step"]) / (MAX_POINTS - 2)))
        buckets = {row["step"] // width: row for row in rows[1:-1]}
        rows = _spread([rows[0], *buckets.values(), rows[-1]], MAX_POINTS)
    train.update(
        plan=plan,
        series=_columns(rows),
        evals=_spread(sorted(evals.values(), key=lambda row: row["step"]), MAX_EVALS),
    )
    train["relocations"] = _spread(sorted(relocations.values(), key=lambda row: row["step"]), 64)
    result["snapshots"] = list(snapshots.values())[-64:]
    if incoming or plan or current.get("train") or "quality" in train:
        result["train"] = train
        # Optional loss components can consume more space than the ordinary seven columns.
        while (
            len(json.dumps({"train": train, "snapshots": result["snapshots"]})) > MAX_BYTES
            and len(rows) > 2
        ):
            rows = _spread(rows, max(2, len(rows) * 3 // 4))
            train["series"] = _columns(rows)
    return result


def derive_events(train: dict[str, Any]) -> list[dict[str, Any]]:
    """Events are reconstructed on read, rather than repeated in every heartbeat's JSON."""
    plan = train.get("plan") or {}
    series = train.get("series") or {}
    steps = series.get("step", [])
    stop = train.get("stop")
    last = max([*steps, *[e["step"] for e in train.get("evals", [])], (stop or {}).get("step", 0)])
    events: list[dict[str, Any]] = []

    def add(step: int, kind: str, text: str, **fields: Any) -> None:
        events.append({"step": step, "kind": kind, "text": text, **fields})

    if plan:
        counts = [
            f"{plan[key]:,} {label}"
            for key, label in (("frames", "frames"), ("sparse", "seed points"))
            if isinstance(plan.get(key), int)
        ]
        add(0, "setup", " · ".join(counts) or "Setup")
    previous_sh = None
    cap_seen = False
    for i, step in enumerate(steps):
        sh = series.get("sh", [None] * len(steps))[i]
        if sh is not None and sh != previous_sh:
            add(step, "sh", f"SH {sh}")
            previous_sh = sh
        gs = series.get("gs", [None] * len(steps))[i]
        if not cap_seen and gs is not None and plan.get("cap") and gs >= plan["cap"]:
            add(step, "cap", f"Cap reached · {plan['cap']:,}")
            cap_seen = True
    refine = plan.get("refineStop")
    if isinstance(refine, int) and last >= refine:
        add(refine, "densify", "Densify done")
    for row in train.get("relocations", []):
        if isinstance(row.get("count"), int):
            add(row["step"], "relocate", f"Relocated {row['count']:,}")
    for row in train.get("evals", []):
        if row.get("psnr") is not None:
            gain = row.get("gain")
            suffix = "" if gain is None else f" · gain {gain:.3f} dB"
            add(row["step"], "eval", f"PSNR {row['psnr']:.2f} dB{suffix}")
    if stop:
        saved = max(0, (stop.get("maxSteps") or plan.get("stepsMax") or last) - stop["step"])
        speed = next((v for v in reversed(series.get("its", [])) if v is not None and v > 0), None)
        rate = plan.get("usdPerHour")
        usd = saved / speed / 3600 * rate if speed and rate is not None else None
        add(
            stop["step"],
            "stop",
            f"Stopped · {saved:,} steps saved"
            + (f" · ${usd:.2f} saved" if usd is not None else ""),
            stepsSaved=saved,
            usdSaved=usd,
        )
    return sorted(events, key=lambda event: event["step"])

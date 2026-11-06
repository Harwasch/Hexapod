"""The stage's resolved training plan, before the trainer resolves its own config."""

from __future__ import annotations

import contextlib
import json
import math
import struct
import uuid
from typing import Any

import convergence
import gaussian_budget
from contracts import StageContext


def emit(
    ctx: StageContext,
    *,
    budget: gaussian_budget.Budget | None,
    cap: int | None,
    steps: int,
    factor: float,
    scaler: float,
    strategy: str,
    rule: convergence.Rule | None,
    blocks: int = 1,
    frames: int | None = None,
    sparse: int | None = None,
) -> None:
    """Only resolved scale is metric; a placement with unresolved scale is not evidence."""
    with contextlib.suppress(Exception):
        metres = None
        if ctx.has_input("georef.json"):
            georef = json.loads(ctx.input("georef.json").read_text(encoding="utf-8"))
            frame = georef.get("frame", {})
            source = georef.get("scaleSource", frame.get("scaleSource"))
            value = frame.get("scale")
            if (
                source
                and source not in ("unresolved", "none")
                and isinstance(value, int | float)
                and math.isfinite(value)
                and value > 0
            ):
                metres = value
        if sparse is None:
            # The COLMAP binary header is the count; avoid reading a second point cloud.
            with (
                contextlib.suppress(OSError, ValueError, AttributeError, struct.error),
                (ctx.work_dir / "dataset" / "sparse" / "0" / "points3D.bin").open("rb") as handle,
            ):
                sparse = struct.unpack("<Q", handle.read(8))[0]
        stop = convergence.REFINE_STOP_ITER.get(strategy)
        payload: dict[str, Any] = {
            "v": 1,
            "id": uuid.uuid4().hex[:12],
            "attempt": ctx.attempt,
            "cap": cap,
            "mode": None if budget is None else budget.mode,
            "clamp": None if budget is None else budget.clamp,
            "stepsMax": steps,
            "scheduleFactor": factor,
            "refineStop": None if stop is None else int(stop * scaler),
            "rule": None
            if rule is None
            else {
                "every": max(1, int(rule.every * scaler)),
                "window": max(1, int(rule.window * scaler)),
                "minGainDb": rule.min_gain_db,
            },
            "blocks": blocks,
            "gpu": ctx.gpu_tier,
            "cardGb": (
                ctx.param("gpu_memory_gb")
                if ctx.param("gpu_memory_gb") is not None
                else gaussian_budget.GPU_MEMORY_GB.get(ctx.gpu_tier or "")
            ),
            "usdPerHour": (
                ctx.param("_live_usd_per_hour")
                if ctx.param("_live_rate_tier") == ctx.gpu_tier
                else None
            ),
            "metresPerUnit": metres,
            "frames": frames,
            "sparse": sparse,
            "antialiased": ctx.param("antialiased"),
        }
        ctx.log("live-plan: " + json.dumps(payload, separators=(",", ":")))

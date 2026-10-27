"""Pure viewer frame statistics and relative costs. No browser, GPU or Modal imports."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from itertools import pairwise
from typing import Any


def frame_stats(timestamps: Sequence[float]) -> dict[str, Any]:
    """All rAF intervals, including hitches; p95 is nearest rank, like FrameMeter."""
    if len(timestamps) < 2 or any(not math.isfinite(t) for t in timestamps):
        raise ValueError("Need at least two finite rAF timestamps")
    intervals = [b - a for a, b in pairwise(timestamps)]
    if any(dt <= 0 for dt in intervals):
        raise ValueError("rAF timestamps must increase strictly")
    ordered = sorted(intervals)
    return {
        "frames": len(intervals),
        "frame_ms": intervals,
        "median_ms": statistics.median(intervals),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "mean_ms": statistics.fmean(intervals),
    }


def gpu_stats(samples: Sequence[dict[str, float]]) -> dict[str, float | int | None]:
    """Whole-device memory, not per-process allocation; software smoke has no samples."""
    for sample in samples:
        for key in ("memory_mib", "utilization_pct"):
            if not math.isfinite(sample[key]) or sample[key] < 0:
                raise ValueError(f"Invalid GPU sample: {key}")
        if sample["utilization_pct"] > 100:
            raise ValueError("GPU utilization exceeds 100%")
    return {
        "gpu_samples": len(samples),
        "gpu_mem_max_mib": max(s["memory_mib"] for s in samples) if samples else None,
        "gpu_util_mean_pct": statistics.fmean(s["utilization_pct"] for s in samples)
        if samples
        else None,
    }


def require_renderer(renderer: str, *, software: bool) -> None:
    """Check the actual drawing canvas, not a separate probe context or adapter."""
    lower = renderer.lower()
    fallback = any(name in lower for name in ("swiftshader", "llvmpipe", "softpipe"))
    if software:
        if not fallback:
            raise ValueError(f"Local smoke requires a software renderer, got {renderer}")
    elif "nvidia" not in lower or fallback:
        raise ValueError(f"Expected NVIDIA hardware; refusing results from {renderer}")


def relative_costs(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Refuse incomparable paths/viewports/counts; always divide by SH 0, never row 1."""
    by_degree = {row["degree"]: row for row in results}
    if len(by_degree) != len(results) or 0 not in by_degree:
        raise ValueError("Unique degrees and an SH 0 baseline are required")
    baseline = by_degree[0]
    for row in results:
        if row["degree"] not in range(4):
            raise ValueError("SH degree must be 0..3")
        for key in ("viewport", "renderer", "splats", "path", "software", "flags"):
            if row[key] != baseline[key]:
                raise ValueError(f"Cannot compare different {key}")
        for key in ("median_ms", "p95_ms"):
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError(f"Invalid {key}")
    return [
        {
            **row,
            "median_relative_sh0": row["median_ms"] / baseline["median_ms"],
            "p95_relative_sh0": row["p95_ms"] / baseline["p95_ms"],
        }
        for row in sorted(results, key=lambda row: row["degree"])
    ]


def markdown(results: Sequence[dict[str, Any]]) -> str:
    rows = relative_costs(results)
    label = "software renderer, relative only" if rows[0]["software"] else "NVIDIA hardware"
    lines = [
        f"Viewer frame cost — {label}\n",
        "| SH | Frames | Median ms | p95 ms | Mean ms | Median / SH0 | p95 / SH0 | "
        "GPU max MiB | GPU util % | Splats | Tileset bytes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        mem = row["gpu_mem_max_mib"]
        util = row["gpu_util_mean_pct"]
        lines.append(
            f"| {row['degree']} | {row['frames']} | {row['median_ms']:.3f} | "
            f"{row['p95_ms']:.3f} | {row['mean_ms']:.3f} | "
            f"{row['median_relative_sh0']:.3f} | {row['p95_relative_sh0']:.3f} | "
            f"{mem if mem is not None else 'n/a'} | "
            f"{f'{util:.1f}' if util is not None else 'n/a'} | "
            f"{row['splats']} | {row['tileset_bytes']} |"
        )
    lines += [
        "",
        "rAF wall intervals, not GPU execution time; all hitches retained. Fixed full-detail "
        "tiles and pixel ratio 1; no adaptive LOD/resolution. GPU memory is whole-device usage "
        "sampled every 0.5 s during measured frames (not loading peaks). Fresh browser per degree.",
        "",
        "Uncapped Chromium flags are recorded in JSON. Inspect pacing warnings before choosing "
        "a degree: a display-rate plateau can hide relative cost. No absolute FPS claim.",
        "",
    ]
    return "\n".join(lines)


def pacing_warning(timestamps: Sequence[float]) -> str | None:
    """Diagnostic only: a 60/120 Hz plateau is suspicious, not proof of throttling."""
    intervals = frame_stats(timestamps)["frame_ms"]
    for period in (1000 / 60, 1000 / 120):
        share = sum(abs(dt - period) < 0.5 for dt in intervals) / len(intervals)
        if share >= 0.8:
            return "At least 80% of intervals lie within 0.5 ms of 60/120 Hz; check frame pacing."
    return None

"""Tolerant optional stage reports and comparisons that do not mix frozen scenes."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from holdout_error import read_ply

from .gpu import summarise_samples
from .ledger import hourly_rate
from .scene import read_json

METRICS = ("psnr", "ssim", "lpips", "psnrFullSh", "psnrFullShGap")


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def optional(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def parse_result(
    plan: dict[str, Any],
    directory: Path,
    *,
    params: dict[str, Any],
    split_digest: str | None,
    manifest_digest: str | None,
    wall: float,
    train_seconds: float | None,
    gpu_name: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    metrics = optional(directory / "train_metrics.json")
    holdout = optional(directory / "holdout.json") or metrics.get("holdout", {})
    test = optional(directory / "test.json")
    ply = directory / "trained.ply"
    count = len(read_ply(ply)["x"]) if ply.is_file() else None
    iterations = number(metrics.get("iterations"))
    return {
        "schemaVersion": 1,
        "runId": plan["runId"],
        "scene": plan["scene"],
        "variant": plan["variant"],
        "seed": plan["seed"],
        "schedule": plan["schedule"],
        "trainParams": params,
        "splitDigest": split_digest,
        "manifestDigest": manifest_digest,
        "test": {
            **{k: number(test.get(k)) for k in METRICS},
            "frames": test.get("frames", 0),
            "perFrame": test.get("perFrame", []),
            "regions": test.get("regions"),
            "regionsReason": test.get("regionsReason", "evaluation did not run"),
            "fullShReason": test.get("fullShReason"),
            "shDegree": test.get("shDegree"),
            "byDegree": test.get("byDegree", {}),
        },
        "val": {
            **{k: number(metrics.get(k)) for k in ("psnr", "ssim", "lpips")},
            "meanPsnrFullSh": number(holdout.get("meanPsnrFullSh")),
        },
        "splats": count,
        "outputBytes": ply.stat().st_size if ply.is_file() else None,
        "trainSeconds": train_seconds,
        "iterations": iterations,
        "trainSecondsPer1kSteps": (
            train_seconds * 1000 / iterations
            if train_seconds is not None and iterations is not None and iterations > 0
            else None
        ),
        **summarise_samples(optional(directory / "gpu_samples.json").get("samples", [])),
        "wallSeconds": wall,
        "gpu": {"name": gpu_name, "tier": plan["tier"]},
        "costUsd": 0.0 if plan.get("rehearsal") else wall * hourly_rate(plan["tier"]) / 3600,
        "benchCommit": plan["benchCommit"],
        "status": "failed" if error else "complete",
        "error": error,
        "rehearsal": bool(plan.get("rehearsal")),
        "timeoutSeconds": plan["timeoutSeconds"],
    }


def markdown(results: list[dict[str, Any]]) -> str:
    def cell(value: Any) -> str:
        return "-" if number(value) is None else f"{value:.4f}"

    lines = ["# Frozen-input bench", ""]
    if any(row.get("rehearsal") for row in results):
        lines += ["Rehearsal rows are synthetic; they are not quality measurements.", ""]
    # Test PSNR/SSIM/LPIPS are of the shipped SH degree; full-SH PSNR is what dropping the
    # higher bands costs on the same frames.
    lines += [
        "| run | scene | variant | seed | schedule | status | test PSNR | SSIM | LPIPS "
        "| full-SH PSNR | splats | MB | train s | train s/1k steps "
        "| GPU util mean % | GPU util p50 % | GPU mem max MiB | wall s | $ |",
        "|" + "---|" * 19,
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        test = row["test"]
        label = row["variant"]["name"] + (" (rehearsal)" if row.get("rehearsal") else "")
        lines.append(
            "| "
            + " | ".join(
                [
                    row["runId"],
                    row["scene"],
                    label,
                    str(row["seed"]),
                    row["schedule"],
                    row["status"],
                    *(cell(test[k]) for k in ("psnr", "ssim", "lpips", "psnrFullSh")),
                    cell(row["splats"]),
                    cell(None if row.get("outputBytes") is None else row["outputBytes"] / 1e6),
                    cell(row["trainSeconds"]),
                    *(
                        cell(row.get(k))
                        for k in (
                            "trainSecondsPer1kSteps",
                            "gpuUtilMean",
                            "gpuUtilP50",
                            "gpuMemMaxMiB",
                        )
                    ),
                    cell(row["wallSeconds"]),
                    cell(row["costUsd"]),
                ]
            )
            + " |"
        )
        # A refreeze, changed overrides, GPU, or schedule is a different experiment.
        key = json.dumps(
            [
                row["scene"],
                row["schedule"],
                row["variant"],
                row["splitDigest"],
                row.get("manifestDigest"),
                row["gpu"]["tier"],
                row.get("rehearsal"),
                row["benchCommit"],
            ],
            sort_keys=True,
        )
        groups[key].append(row)
    lines += [
        "",
        "Test colour by degree (same frames and geometry):",
        "",
        "| run | SH degree | frames | PSNR | SSIM | LPIPS |",
        "|---|---|---|---|---|---|",
    ]
    for row in results:
        for degree, scores in sorted(row["test"].get("byDegree", {}).items()):
            lines.append(
                f"| {row['runId']} | {degree} | {scores.get('frames', '-')} | "
                + " | ".join(cell(scores.get(k)) for k in ("psnr", "ssim", "lpips"))
                + " |"
            )
    lines += [
        "",
        "Across completed seeds (mean / min / max / spread):",
        "",
        "| experiment | metric | n | mean | min | max | spread |",
        "|---|---|---|---|---|---|---|",
    ]
    for rows in groups.values():
        first = rows[0]
        label = f"{first['scene']}/{first['schedule']}/{first['variant']['name']}"
        label += f" ({str(first.get('manifestDigest'))[:8]}, {first['gpu']['tier']})"
        if first.get("rehearsal"):
            label += " REHEARSAL"
        for key in (
            "psnr",
            "ssim",
            "lpips",
            "psnrFullSh",
            "psnrFullShGap",
            "splats",
            "trainSeconds",
            "trainSecondsPer1kSteps",
            "gpuUtilMean",
            "gpuUtilP50",
            "gpuMemMaxMiB",
            "wallSeconds",
            "costUsd",
        ):
            values = [
                value
                for r in rows
                if r["status"] == "complete"
                if (value := number((r["test"] if key in METRICS else r).get(key))) is not None
            ]
            if values:
                stats = [
                    sum(values) / len(values),
                    min(values),
                    max(values),
                    max(values) - min(values),
                ]
                lines.append(
                    f"| {label} | {key} | {len(values)} | " + " | ".join(map(cell, stats)) + " |"
                )
        for degree in range(4):
            for metric in ("psnr", "ssim", "lpips"):
                values = [
                    value
                    for r in rows
                    if r["status"] == "complete"
                    if (
                        value := number(
                            r["test"].get("byDegree", {}).get(str(degree), {}).get(metric)
                        )
                    )
                    is not None
                ]
                if values:
                    stats = [
                        sum(values) / len(values),
                        min(values),
                        max(values),
                        max(values) - min(values),
                    ]
                    lines.append(
                        f"| {label} | sh{degree}.{metric} | {len(values)} | "
                        + " | ".join(map(cell, stats))
                        + " |"
                    )
    for row in results:
        if row.get("error"):
            lines += ["", f"Failed `{row['runId']}`: {row['error']}"]
    return "\n".join(lines) + "\n"

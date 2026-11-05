"""Tolerant optional stage reports and comparisons that do not mix frozen scenes."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import quality_stack
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
    gate = optional(directory / "quality.json").get("bar")
    headline = "trained.ply" if gate is None else "faded.ply" if gate == "fade" else "gated.ply"
    ply = directory / headline
    count = len(read_ply(ply)["x"]) if ply.is_file() else None
    iterations = number(metrics.get("iterations"))
    return {
        "schemaVersion": 1,
        "runId": plan["runId"],
        "scene": plan["scene"],
        "tag": plan.get("tag"),
        "variant": plan["variant"],
        "seed": plan["seed"],
        "schedule": plan["schedule"],
        "trainParams": params,
        "splitDigest": split_digest,
        "manifestDigest": manifest_digest,
        "qualityGate": gate,
        "headlinePly": headline,
        "testTrained": optional(directory / "test-trained.json") or (test if gate is None else {}),
        "testFaded": test if gate == "fade" else None,
        "testRenderer": "antialiased" if params.get("antialiased") else "classic",
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
        **quality_stack.export_check(metrics.get("psnr"), holdout),
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
    for row in results:
        if row.get("exportWarning"):
            lines += ["", f"**{row['runId']}: {row['exportWarning']}**"]
    return "\n".join(lines) + "\n"


def bakeoff_markdown(results: list[dict[str, Any]]) -> str:
    """One row per run: keep provenance visible instead of silently pooling smoke/full."""
    lines = [
        "| scene | variant | test PSNR | SSIM | LPIPS | splats | GPU min | $ "
        "| run / schedule / seed / status | top-down |",
        "|" + "---|" * 10,
    ]
    for row in results:
        test = row.get("test", {})
        wall = number(row.get("wallSeconds"))
        values = [test.get(k) for k in ("psnr", "ssim", "lpips")]
        values += [
            row.get("splats"),
            None if wall is None or row.get("rehearsal") else wall / 60,
            row.get("costUsd"),
        ]
        cells = ["-" if number(v) is None else f"{v:.4f}" for v in values]
        preview = row.get("topdown", {})
        link = preview.get("localPath")
        image = f"[PNG]({Path(link).as_uri()})" if link else preview.get("status", "off")
        label = row.get("variant", {}).get("name", "unknown")
        if row.get("rehearsal"):
            label += " (synthetic)"
        lines.append(
            f"| {row['scene']} | {label} | "
            + " | ".join(cells)
            + f" | {row['runId']} / {row.get('schedule')} / {row.get('seed')} "
            + f"/ {row['status']} | {image} |"
        )
    lines += [
        "",
        "GPU minutes are billed container wall time, including startup and evaluation; "
        "costs use the ledger's rate model.",
        "",
        "Headline scores use the gate output when enabled. Baseline-main has no bench gate.",
        "",
        "| run | PLY scored | PSNR | SSIM | LPIPS |",
        "|---|---|---|---|---|",
    ]
    for row in results:
        for name, scores in (("trained", row.get("testTrained")), ("faded", row.get("testFaded"))):
            # Old baseline results predate the explicit gate fields; its test score
            # was of trained.ply. Do not invent a pre-gate score for older candidates.
            if (
                scores is None
                and name == "trained"
                and row.get("variant", {}).get("name") == "baseline-main"
            ):
                scores = row.get("test")
            if scores:
                cells = [
                    "-" if number(scores.get(k)) is None else f"{scores[k]:.4f}"
                    for k in ("psnr", "ssim", "lpips")
                ]
                lines.append(f"| {row['runId']} | {name} | " + " | ".join(cells) + " |")
    lines.append("")
    for row in results:
        if row.get("exportWarning"):
            lines += ["", f"**{row['runId']}: {row['exportWarning']}**"]
    return "\n".join(lines)


def flip_rule(
    results: list[dict[str, Any]],
    baseline: str,
    candidate: str,
    scenes: tuple[str, ...] = ("spool", "truck", "beach"),
) -> dict[str, Any]:
    """Full L4 only; pair identical frozen inputs/seeds; ambiguous/missing runs abstain.

    Commits may differ (the old image is the baseline), but each arm must use one
    commit and override set. Screen and synthetic results can never flip a default.
    Caller can select explicit run IDs to disambiguate reruns, or compare adjacent arms
    by changing baseline. The verdict is PSNR-only, as specified by the bake-off rule.
    """
    rows = [
        r
        for r in results
        if not r.get("rehearsal")
        and r.get("schedule") == "full"
        and r.get("gpu", {}).get("tier") == "l4"
        and r.get("variant", {}).get("name") in {baseline, candidate}
    ]
    deltas: dict[str, float | None] = dict.fromkeys(scenes)
    reasons = []
    for arm in (baseline, candidate):
        configs = {
            json.dumps([r.get("benchCommit"), r["variant"].get("overrides")], sort_keys=True)
            for r in rows
            if r["variant"]["name"] == arm
        }
        if len(configs) > 1:
            reasons.append(f"{arm}: mixed commits/overrides; select explicit --runs")
    for scene in scenes:
        pairs = []
        for arm in (baseline, candidate):
            selected = [r for r in rows if r["scene"] == scene and r["variant"]["name"] == arm]
            by_seed = {r.get("seed"): r for r in selected}
            if not selected or len(selected) != len(by_seed):
                reasons.append(f"{scene}/{arm}: missing or duplicate seeds")
            pairs.append(by_seed)
        left, right = pairs
        if not left or left.keys() != right.keys():
            reasons.append(f"{scene}: unmatched seeds")
            continue
        differences = []
        for seed in left:
            a, b = left[seed], right[seed]
            valid = all(
                a.get(k) is not None and a.get(k) == b.get(k)
                for k in ("manifestDigest", "splitDigest")
            )
            av, bv = number(a.get("test", {}).get("psnr")), number(b.get("test", {}).get("psnr"))
            if (
                not valid
                or a["status"] != "complete"
                or b["status"] != "complete"
                or av is None
                or bv is None
            ):
                reasons.append(f"{scene}/{seed}: failed, unscored, or different frozen inputs")
                continue
            differences.append(bv - av)
        if len(differences) == len(left):
            deltas[scene] = sum(differences) / len(differences)
    values = [v for v in deltas.values() if v is not None]
    eligible = not reasons and len(values) == len(scenes)
    passes = eligible and sum(v > 0 for v in values) >= 2 and min(values) >= -0.3
    return {
        "baseline": baseline,
        "candidate": candidate,
        "deltaPsnr": deltas,
        "verdict": "eligible" if passes else "keep-off" if eligible else "incomplete",
        "reasons": reasons,
    }


def flip_markdown(results: list[dict[str, Any]], baseline: str, candidate: str | None) -> str:
    candidates = (
        [candidate]
        if candidate
        else sorted({r.get("variant", {}).get("name", "unknown") for r in results} - {baseline})
    )
    lines = [
        "",
        f"PSNR flip rule against `{baseline}` (full L4, matched seeds; "
        "eligible is not an automatic default change):",
        "",
        "| candidate | spool Δ dB | truck Δ dB | beach Δ dB | verdict | reason |",
        "|---|---|---|---|---|---|",
    ]
    for name in candidates:
        rule = flip_rule(results, baseline, name)
        cells = ["-" if v is None else f"{v:+.4f}" for v in rule["deltaPsnr"].values()]
        lines.append(
            f"| {name} | "
            + " | ".join(cells)
            + f" | {rule['verdict']} | {'; '.join(rule['reasons'])} |"
        )
    return "\n".join(lines) + "\n"

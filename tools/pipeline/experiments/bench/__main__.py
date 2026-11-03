"""Offline plans/rehearsals, or an ephemeral Modal app with at most four concurrent calls."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

from . import ledger, rehearsal
from .recipe import PIPELINE, train_params
from .results import markdown, parse_result
from .scene import SCENES, read_json, verify, write_json
from .worker import run_training

REPO = PIPELINE.parents[1]
STATE = Path.home() / ".local/share/hexapod-bench"


def git_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        return subprocess.check_output([git, "rev-parse", "HEAD"], cwd=REPO, text=True).strip()  # noqa: S603
    except subprocess.CalledProcessError:
        return "unknown"


def load_modal() -> Any:
    path = REPO / "infra/modal/bench.py"
    spec = importlib.util.spec_from_file_location("hexapod_bench_modal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--scene", choices=sorted(SCENES), required=True)
    freeze.add_argument("--run", help="override the registry UUID or 8-hex run prefix")
    freeze.add_argument("--generation", help="root or p<16 hex>; required for ambiguity")
    freeze.add_argument("--refreeze", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--scene", choices=sorted(SCENES), required=True)
    run.add_argument("--variant", default="baseline")
    run.add_argument("--variants", type=Path, help="YAML mapping name -> {train: overrides}")
    run.add_argument("--set", default="{}", dest="overrides")
    run.add_argument("--seeds", default="42")
    run.add_argument("--schedule", choices=("screen", "full"), default="screen")
    run.add_argument("--tier", choices=("l4", "a10", "l40s", "a100"), default="l4")
    run.add_argument("--keep-splat", action="store_true")
    run.add_argument("--renders", type=int, default=0, help="save the first N test frames as PNGs")
    report = commands.add_parser("report")
    report.add_argument("--runs", nargs="*", help="local run IDs or paths to result.json")
    spend = commands.add_parser("ledger")
    for command in (freeze, run, report, spend):
        command.add_argument("--ledger", type=Path)
        command.add_argument("--state-dir", type=Path, default=STATE)
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--rehearse", action="store_true")
    for command in (freeze, run):
        command.add_argument("--cap", type=float, default=2.0)
    return root


def plans_for(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = {"scene": args.scene, "benchCommit": git_commit(), "rehearsal": args.rehearse}
    if args.command == "freeze":
        return [
            {
                **base,
                "runId": "freeze-" + uuid.uuid4().hex,
                "schedule": "freeze",
                "tier": "cpu4",
                "sourceRun": args.run or SCENES[args.scene],
                "generation": args.generation,
                "refreeze": args.refreeze,
            }
        ]
    if args.renders < 0:
        raise ValueError("--renders must be nonnegative")
    common = json.loads(args.overrides)
    variants = yaml.safe_load(args.variants.read_text()) if args.variants else {args.variant: {}}
    if not isinstance(variants, dict) or not variants:
        raise ValueError("variants must map names to {train: overrides}")
    # Validate common overrides independently, then merge params with --set last.
    train_params(args.schedule, common)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must be distinct unsigned 32-bit integers")
    plans = []
    for name, overrides in variants.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_.+-]{1,64}", name):
            raise ValueError("variant names must be 1-64 letters, digits, _, ., + or -")
        if not isinstance(overrides, dict):
            raise ValueError("variant overrides must be a mapping")
        train_params(args.schedule, overrides)
        merged = {"train": {**overrides.get("train", {}), **common.get("train", {})}}
        params = train_params(args.schedule, merged)
        for seed in seeds:
            plans.append(
                {
                    **base,
                    "runId": uuid.uuid4().hex,
                    "variant": {"name": name, "overrides": merged},
                    "seed": seed,
                    "schedule": args.schedule,
                    "tier": args.tier,
                    "trainParams": params,
                    "keepSplat": args.keep_splat,
                    "renders": args.renders,
                }
            )
    return plans


def save_outcome(state: Path, book: Path, plan: dict[str, Any], result: dict[str, Any]) -> None:
    directory = state / "runs" / plan["runId"]
    write_json(directory / "result.json", result)
    for name, value in result.get("artifacts", {}).items():
        # Only reports known to this version, never a remote-supplied path.
        from .worker import ARTIFACTS

        if name in ARTIFACTS:
            write_json(directory / name, value)
    ledger.finish(
        book,
        plan["runId"],
        cost=result.get("costUsd"),
        failed=result["status"] != "complete",
        error=result.get("error"),
    )


def lost_result(state: Path, plan: dict[str, Any], error: Exception) -> dict[str, Any]:
    message = f"Remote outcome unknown: {type(error).__name__}: {error}"
    if plan["schedule"] == "freeze":
        return {
            "runId": plan["runId"],
            "scene": plan["scene"],
            "status": "failed",
            "error": message,
            "costUsd": None,
        }
    result = parse_result(
        plan,
        state / "runs" / plan["runId"],
        params=plan["trainParams"],
        split_digest=plan.get("splitDigest"),
        manifest_digest=plan.get("manifestDigest"),
        wall=0,
        train_seconds=None,
        gpu_name=None,
        error=message,
    )
    result.update(costUsd=None, wallSeconds=None)
    return result


def remote_batch(
    plans: list[dict[str, Any]], state: Path, book: Path, cap: float
) -> list[dict[str, Any]]:
    # No imports of Modal (nor app/image creation) before the local cap preflight.
    ledger.price_plan(ledger.read(book), plans, cap)
    module = load_modal()
    results = []
    with module.app.run():
        if plans[0]["schedule"] != "freeze":
            # Volume API read: no CPU/GPU preflight function, no R2 credentials.
            manifest = json.loads(
                b"".join(module.volume.read_file(f"scenes/{plans[0]['scene']}/manifest.json"))
            )
            for plan in plans:
                plan.update(manifestDigest=manifest["digest"], splitDigest=manifest["splitDigest"])
        ledger.reserve(book, plans, cap)
        for offset in range(0, len(plans), 4):
            pending = []
            for plan in plans[offset : offset + 4]:
                function = (
                    module.freeze
                    if plan["schedule"] == "freeze"
                    else module.FUNCTIONS[(plan["tier"], plan["schedule"])]
                )
                try:
                    pending.append((plan, function.spawn(plan)))
                except Exception as error:
                    result = lost_result(state, plan, error)
                    save_outcome(state, book, plan, result)
                    results.append(result)
            for plan, call in pending:
                try:
                    result = call.get()
                except Exception as error:
                    result = lost_result(state, plan, error)
                save_outcome(state, book, plan, result)
                results.append(result)
    return results


def local_batch(
    plans: list[dict[str, Any]], state: Path, book: Path, cap: float
) -> list[dict[str, Any]]:
    scene = state / "scenes" / plans[0]["scene"]
    if plans[0]["schedule"] == "freeze" and scene.exists() and not plans[0]["refreeze"]:
        raise ValueError("synthetic scene exists; use --refreeze")
    if plans[0]["schedule"] == "freeze" and scene.exists():
        shutil.rmtree(scene)
    if not scene.exists():
        rehearsal.synthetic_scene(scene, plans[0]["scene"], plans[0]["benchCommit"])
    manifest, split = verify(scene)
    if manifest["sourcePrefix"] != "synthetic://rehearsal":
        raise ValueError("rehearsal requires synthetic inputs")
    for plan in plans:
        plan.update(manifestDigest=manifest["digest"], splitDigest=split["digest"])
    ledger.reserve(book, plans, cap)
    results = []
    for plan in plans:
        if plan["schedule"] == "freeze":
            result = {
                "runId": plan["runId"],
                "scene": plan["scene"],
                "status": "complete",
                "manifest": manifest,
                "costUsd": 0.0,
                "rehearsal": True,
            }
        else:
            result = run_training(scene, state / "runs" / plan["runId"], plan)
        save_outcome(state, book, plan, result)
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    state = args.state_dir.expanduser().resolve()
    if args.rehearse:
        state /= "rehearsal"
    book = args.ledger.expanduser().resolve() if args.ledger else state / "ledger.jsonl"
    try:
        if book.is_relative_to(REPO):
            raise ValueError("ledger must live outside the repository")
        entries = ledger.read(book)
        if args.command == "ledger":
            sys.stdout.write(json.dumps(ledger.summary(entries), indent=2) + "\n")
            return 0
        if args.command == "report":
            paths = (
                [
                    Path(run) if Path(run).is_file() else state / "runs" / run / "result.json"
                    for run in args.runs
                ]
                if args.runs
                else sorted((state / "runs").glob("*/result.json"))
            )
            results = [read_json(path) for path in paths]
            sys.stdout.write(markdown([r for r in results if r.get("schemaVersion") == 1]))
            return 0
        plans = plans_for(args)
        priced = ledger.price_plan(entries, plans, args.cap)
        sys.stdout.write(json.dumps(priced, indent=2) + "\n")
        if args.dry_run:
            return 0
        results = (local_batch if args.rehearse else remote_batch)(plans, state, book, args.cap)
        if args.command == "run":
            sys.stdout.write(markdown(results))
        else:
            sys.stdout.write(json.dumps(results, indent=2) + "\n")
        return 0 if all(r["status"] == "complete" for r in results) else 1
    except (ValueError, OSError, ImportError) as error:
        sys.stderr.write(f"bench: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

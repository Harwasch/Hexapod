"""Offline plans/rehearsals, or an ephemeral Modal app with at most four concurrent calls."""

from __future__ import annotations

import argparse
import base64
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
from .recipe import PIPELINE, train_params, validate_renderer
from .results import bakeoff_markdown, flip_markdown, markdown, parse_result
from .scene import PUBLIC_SCENES, SCENES, read_json, verify, write_json
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
    distance = commands.add_parser("distance", help="local CPU splat distance to camera path")
    distance.add_argument("--scene-dir", type=Path, required=True)
    distance.add_argument("--ply", type=Path, action="append", required=True)
    distance.add_argument("--camera-height-m", type=float, default=1.4)
    distance.add_argument("--output", type=Path, help="directory for distance.json and distance.md")
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--scene", choices=sorted(SCENES), required=True)
    freeze.add_argument("--run", help="override the registry UUID or 8-hex run prefix")
    freeze.add_argument("--generation", help="root or p<16 hex>; required for ambiguity")
    freeze.add_argument("--refreeze", action="store_true")
    full = commands.add_parser(
        "freeze-full", help="native frames registered into frozen beach backbone (CPU only)"
    )
    full.add_argument("--scene", choices=("beach-full",), default="beach-full")
    full.add_argument(
        "--source-key", help="original upload key if absent from source run artifacts"
    )
    full.add_argument("--exclude-test-seconds", type=float, default=0.1)
    full.add_argument("--blur-ratio", type=float, default=0.5)
    full.add_argument("--duplicate-mae", type=float, default=1.0)
    run = commands.add_parser("run")
    run.add_argument("--scene", choices=sorted(SCENES), required=True)
    run.add_argument("--variant", help="one named variant; with --variants, select only this arm")
    run.add_argument("--variants", type=Path, help="YAML mapping name -> {train: overrides}")
    run.add_argument("--set", default="{}", dest="overrides")
    run.add_argument("--seeds", default="42")
    run.add_argument("--schedule", choices=("screen", "full"), default="screen")
    run.add_argument("--tier", choices=("l4", "a10", "l40s", "a100"), default="l4")
    run.add_argument("--keep-splat", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--topdown", action="store_true", help="best-effort GPU overview PNG")
    run.add_argument("--live", action="store_true", help="publish live console telemetry")
    run.add_argument("--renders", type=int, default=0, help="save the first N test frames as PNGs")
    matched = commands.add_parser(
        "matched-renders", help="same cameras for two kept PLYs, one L4 call"
    )
    matched.add_argument("--scene", choices=sorted(SCENES), required=True)
    matched.add_argument("--runs", nargs=2, required=True, metavar=("A", "B"))
    matched.add_argument("--frame", help="exact frozen test filename; default: median altitude")
    rescore = commands.add_parser(
        "rescore", help="one evaluation-only L4 call; common full/half yardstick"
    )
    rescore.add_argument("--scene", choices=sorted(SCENES), required=True)
    rescore.add_argument("--runs", nargs="+", required=True)
    rescore.add_argument(
        "--ply", choices=("trained", "faded"), help="default: each available trained/faded PLY"
    )
    rescore_table = commands.add_parser("rescore-table")
    rescore_table.add_argument("--runs", nargs="+", required=True)
    rescore_table.add_argument(
        "--output", type=Path, help="JSON table destination; default: state-dir/rescore-table.json"
    )
    report = commands.add_parser("report")
    report.add_argument("--runs", nargs="*", help="local run IDs or paths to result.json")
    results = commands.add_parser("results", help="compact bake-off table and optional flip rule")
    results.add_argument("--runs", nargs="*")
    results.add_argument("--flip-rule", action="store_true")
    results.add_argument("--baseline", default="baseline-main")
    results.add_argument("--candidate", help="one candidate for the flip rule; default: all")
    spend = commands.add_parser("ledger")
    for command in (freeze, full, run, matched, rescore, rescore_table, report, results, spend):
        command.add_argument("--ledger", type=Path)
        command.add_argument("--state-dir", type=Path, default=STATE)
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--rehearse", action="store_true")
        command.add_argument("--tag", help="spend/report group in the shared ledger")
    for command in (freeze, full, run, matched, rescore):
        command.add_argument("--cap", type=float, default=2.0)
        command.add_argument(
            "--total-cap", type=float, help="cumulative cap for --tag; --cap bounds each batch"
        )
    rescore.set_defaults(tag="rescore", total_cap=5.0, cap=1.0)
    return root


def plans_for(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.tag is not None and not re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", args.tag):
        raise ValueError("tag must be 1-64 letters, digits, _, . or -")
    base = {
        "scene": args.scene,
        "benchCommit": git_commit(),
        "rehearsal": args.rehearse,
        "tag": args.tag,
    }
    if args.command == "freeze-full":
        if not (
            0 <= args.exclude_test_seconds < 68
            and 0 <= args.blur_ratio < 1
            and 0 <= args.duplicate_mae <= 255
        ):
            raise ValueError("invalid full-data selection thresholds")
        return [
            {
                **base,
                "runId": "freeze-full-" + uuid.uuid4().hex,
                "schedule": "freeze-full",
                "tier": "cpu4",
                "sourceRun": "4dd02171-341d-4970-bad2-8e410fb8d233",
                "sourceKey": args.source_key,
                "excludeTestSeconds": args.exclude_test_seconds,
                "blurRatio": args.blur_ratio,
                "duplicateMae": args.duplicate_mae,
            }
        ]
    if args.command == "rescore":
        if not 1 <= len(args.runs) <= 4 or len(set(args.runs)) != len(args.runs):
            raise ValueError("rescore needs one to four distinct run IDs")
        if any(not re.fullmatch(r"[a-f0-9]{8,32}", run) for run in args.runs):
            raise ValueError("--runs needs bench run IDs or 8+ hex prefixes")
        return [
            {
                **base,
                "runId": "rescore-" + uuid.uuid4().hex,
                "schedule": "rescore",
                "tier": "l4",
                "runs": args.runs,
                "ply": [args.ply] if args.ply else None,
            }
        ]
    if args.command == "matched-renders":
        if any(not re.fullmatch(r"[a-f0-9]{8,32}", run) for run in args.runs):
            raise ValueError("--runs needs bench run IDs or 8+ hex prefixes")
        if args.runs[0] == args.runs[1]:
            raise ValueError("matched renders need two distinct runs")
        return [
            {
                **base,
                "runId": "matched-" + uuid.uuid4().hex,
                "schedule": "matched",
                "tier": "l4",
                "runs": args.runs,
                "frame": args.frame,
            }
        ]
    if args.command == "freeze":
        if args.scene == "beach-full":
            raise ValueError("use freeze-full to preserve the beach test set")
        public = args.scene in PUBLIC_SCENES
        if public and (args.run or args.generation):
            raise ValueError("public scenes use their pinned archive, not --run/--generation")
        return [
            {
                **base,
                "runId": "freeze-" + uuid.uuid4().hex,
                "schedule": "freeze",
                "tier": "cpu4",
                "sourceRun": args.run or SCENES[args.scene],
                "sourceType": "public" if public else "r2",
                "generation": args.generation,
                "refreeze": args.refreeze,
            }
        ]
    if args.renders < 0:
        raise ValueError("--renders must be nonnegative")
    common = json.loads(args.overrides)
    variants = (
        yaml.safe_load(args.variants.read_text())
        if args.variants
        else {args.variant or "baseline": {}}
    )
    if not isinstance(variants, dict) or not variants:
        raise ValueError("variants must map names to {train: overrides}")
    if args.variants and args.variant:
        if args.variant not in variants:
            raise ValueError(f"variant {args.variant!r} is absent from {args.variants}")
        variants = {args.variant: variants[args.variant]}
    # Validate common overrides independently, then merge params with --set last.
    train_params(args.schedule, common)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must be distinct unsigned 32-bit integers")
    plans = []
    for name, overrides in variants.items():
        if name == "baseline-main" and not args.rehearse:
            raise ValueError("baseline-main must run from c7fb90c8 (gsplat 1.5.3), not this image")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_.+-]{1,64}", name):
            raise ValueError("variant names must be 1-64 letters, digits, _, ., + or -")
        if not isinstance(overrides, dict):
            raise ValueError(
                f"variant {name!r} is pending implementation; overrides must be a mapping"
            )
        bench = overrides.get("bench", {})
        if (
            not isinstance(bench, dict)
            or set(bench) - {"keep_splat", "test_exclusion_seconds"}
            or ("keep_splat" in bench and not isinstance(bench["keep_splat"], bool))
        ):
            raise ValueError("variant bench settings only accept keep_splat: true/false")
        exclusion = bench.get("test_exclusion_seconds")
        if exclusion is not None and (
            not isinstance(exclusion, int | float) or not 0 <= exclusion < 68
        ):
            raise ValueError("test_exclusion_seconds must be in [0,68)")
        overrides = {k: v for k, v in overrides.items() if k != "bench"}
        train_params(args.schedule, overrides)
        merged = {"train": {**overrides.get("train", {}), **common.get("train", {})}}
        if "quality" in overrides or "quality" in common:
            merged["quality"] = {**overrides.get("quality", {}), **common.get("quality", {})}
        params = train_params(args.schedule, merged)
        validate_renderer(params)
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
                    "keepSplat": args.keep_splat
                    if args.keep_splat is not None
                    else bench.get("keep_splat", False),
                    "testExclusionSeconds": bench.get("test_exclusion_seconds"),
                    "renders": args.renders,
                    "topdown": args.topdown,
                    **({"live": True} if args.live else {}),
                }
            )
    return plans


def save_outcome(state: Path, book: Path, plan: dict[str, Any], result: dict[str, Any]) -> None:
    directory = state / "runs" / plan["runId"]
    if plan["schedule"] == "rescore":
        from .rescore import save_local

        save_local(state, result)
    if plan["schedule"] == "matched":
        from .matched_gpu import PNG_NAMES

        payload = result.pop("pngs", {})
        if result["status"] == "complete":
            runs = result["runs"]
            if len(runs) != 2 or any(not re.fullmatch(r"[a-f0-9]{32}", r) for r in runs):
                raise ValueError("invalid matched run IDs returned")
            target = state / "runs" / f"matched-{runs[0][:8]}-{runs[1][:8]}"
            target.mkdir(parents=True, exist_ok=True)
            if set(payload) != set(PNG_NAMES):
                raise ValueError("missing or unexpected matched PNGs")
            for name in PNG_NAMES:
                encoded = payload[name]
                if not isinstance(encoded, str) or len(encoded) > 43_000_000:
                    raise ValueError("oversized matched PNG")
                png = base64.b64decode(encoded, validate=True)
                if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("invalid matched PNG")
                (target / name).write_bytes(png)
            result["localPath"] = str(target)
            write_json(target / "cameras.json", result["cameras"])
            write_json(target / "result.json", result)
    # Transport only the fixed preview path; do not trust a remote-supplied filename.
    preview = dict(result.get("topdown", {}))
    encoded = preview.pop("pngBase64", None)
    if encoded is not None:
        try:
            if not isinstance(encoded, str) or len(encoded) > 1_400_000:
                raise ValueError("oversized top-down payload")
            png = base64.b64decode(encoded, validate=True)
            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("invalid top-down PNG")
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "topdown.png").write_bytes(png)
            preview["localPath"] = str(directory / "topdown.png")
        except (ValueError, OSError) as error:
            preview.update(status="failed", warning=f"local top-down save: {error}")
    if preview:
        result["topdown"] = preview
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
    if plan["schedule"] in {"freeze", "freeze-full", "matched", "rescore"}:
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
    plans: list[dict[str, Any]],
    state: Path,
    book: Path,
    cap: float,
    total_cap: float | None = None,
) -> list[dict[str, Any]]:
    # No imports of Modal (nor app/image creation) before the local cap preflight.
    ledger.price_plan(ledger.read(book), plans, cap, total_cap)
    module = load_modal()
    results = []
    with module.app.run():
        if plans[0]["schedule"] not in {"freeze", "freeze-full"}:
            # Volume API read: no CPU/GPU preflight function, no R2 credentials.
            manifest = json.loads(
                b"".join(module.volume.read_file(f"scenes/{plans[0]['scene']}/manifest.json"))
            )
            for plan in plans:
                plan.update(manifestDigest=manifest["digest"], splitDigest=manifest["splitDigest"])
        ledger.reserve(book, plans, cap, total_cap)
        for offset in range(0, len(plans), 4):
            pending = []
            for plan in plans[offset : offset + 4]:
                function = (
                    module.freeze_full
                    if plan["schedule"] == "freeze-full"
                    else (
                        module.freeze_public
                        if plan.get("sourceType") == "public"
                        else module.freeze
                    )
                    if plan["schedule"] == "freeze"
                    else module.matched_renders
                    if plan["schedule"] == "matched"
                    else module.rescore
                    if plan["schedule"] == "rescore"
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
    plans: list[dict[str, Any]],
    state: Path,
    book: Path,
    cap: float,
    total_cap: float | None = None,
) -> list[dict[str, Any]]:
    if plans[0]["schedule"] == "freeze-full":
        raise ValueError(
            "freeze-full requires the actual video and pycolmap; use --dry-run for its cost plan"
        )
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
    ledger.reserve(book, plans, cap, total_cap)
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
                "tag": plan.get("tag"),
            }
        elif plan["schedule"] == "matched":
            from .matched import run_matched

            result = run_matched(state, state / "runs" / plan["runId"], plan)
        elif plan["schedule"] == "rescore":
            from .rescore import run_rescore

            result = run_rescore(state, state / "runs" / plan["runId"], plan)
        else:
            result = run_training(scene, state / "runs" / plan["runId"], plan)
        save_outcome(state, book, plan, result)
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "distance":
        from experiments.distance import run as distance_run

        try:
            document = distance_run(args.scene_dir, args.ply, args.output, args.camera_height_m)
            sys.stdout.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
            return 0 if all(row["status"] == "complete" for row in document["rows"]) else 1
        except (ValueError, OSError) as error:
            sys.stderr.write(f"bench distance: {error}\n")
            return 2
    state = args.state_dir.expanduser().resolve()
    if args.rehearse:
        state /= "rehearsal"
    book = args.ledger.expanduser().resolve() if args.ledger else state / "ledger.jsonl"
    try:
        if book.is_relative_to(REPO) and not args.rehearse:
            raise ValueError("ledger must live outside the repository")
        entries = ledger.read(book)
        if args.command == "rescore-table":
            from .matched import source_run
            from .rescore import table

            paths = []
            for run in args.runs:
                path = Path(run)
                if not path.is_file():
                    folder = state / "runs" / run
                    if not folder.exists():
                        folder = source_run(state / "runs", run)
                    path = folder / "rescore.json"
                paths.append(path)
            text, document = table([read_json(path) for path in paths])
            write_json(args.output or state / "rescore-table.json", document)
            sys.stdout.write(text)
            return 0
        if args.command == "ledger":
            sys.stdout.write(json.dumps(ledger.summary(entries), indent=2) + "\n")
            return 0
        if args.command in {"report", "results"}:
            paths = (
                [
                    Path(run) if Path(run).is_file() else state / "runs" / run / "result.json"
                    for run in args.runs
                ]
                if args.runs
                else sorted((state / "runs").glob("*/result.json"))
            )
            results = [read_json(path) for path in paths]
            results = [
                r
                for r in results
                if r.get("schemaVersion") == 1
                and r.get("schedule") not in {"rescore", "matched"}
                and (not args.tag or r.get("tag") == args.tag)
            ]
            if args.command == "report":
                sys.stdout.write(markdown(results))
            else:
                sys.stdout.write(bakeoff_markdown(results))
                if args.flip_rule:
                    sys.stdout.write(flip_markdown(results, args.baseline, args.candidate))
            return 0
        plans = plans_for(args)
        priced = ledger.price_plan(entries, plans, args.cap, args.total_cap)
        sys.stdout.write(json.dumps(priced, indent=2) + "\n")
        if args.dry_run:
            return 0
        results = (local_batch if args.rehearse else remote_batch)(
            plans, state, book, args.cap, args.total_cap
        )
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

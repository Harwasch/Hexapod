"""Budgeted full-PLY versus console-SPZ diagnostic, sharing the frozen solve frame."""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image

import sfm
import training

from .console_gpu import panels
from .console_maths import PNG_NAMES, VIEWS, artifact_info, live_inputs, orbit_camera
from .full_data import evaluation_identity, timestamps
from .ledger import hourly_rate
from .matched import source_run
from .rescore import MAX_PNG_BASE64_BYTES, encode_pngs, trained_to_completion
from .scene import read_json, verify, write_json

BASELINE = "ce19d9f1a7a54ee6b98930e757d94d84"


def selected_frames(scene: Path, split: dict[str, Any], result: dict[str, Any]) -> list[str]:
    excluded = set((result.get("frameSelection") or {}).get("excluded", []))
    pool = sorted(set(split["pool"]) - excluded)
    every = result.get("trainParams", {}).get("test_every", 0)
    if every:
        pool = [name for i, name in enumerate(pool) if i % every != 0]
    if len(pool) < 3:
        raise ValueError("console-check requires at least three actual training cameras")
    if (scene / "frames_meta.json").is_file():
        offsets = timestamps(scene)
        if set(pool) <= offsets.keys():
            pool.sort(key=lambda name: (offsets[name], name))
    return [pool[int((len(pool) - 1) * fraction + 0.5)] for fraction in (0.1, 0.5, 0.9)]


def source(
    root: Path, run_id: str, rehearsal: bool
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    run = source_run(root / "runs", run_id)
    result = read_json(run / "result.json")
    if not trained_to_completion(result) or bool(result.get("rehearsal")) != rehearsal:
        raise ValueError(f"{run.name}: training must have completed in this rehearsal namespace")
    if not (run / "trained.ply").is_file():
        raise ValueError(f"{run.name}: missing trained.ply on volume (requires --keep-splat)")
    scene = root / "scenes" / result["scene"]
    manifest, split = verify(scene, result.get("manifestDigest"))
    if result.get("splitDigest") != manifest["splitDigest"]:
        raise ValueError("source run split no longer matches frozen scene")
    return run, scene, result, split


def synthetic(settings: dict[str, Any], output: Path) -> None:
    cameras = {key: settings["orbit"] for key in VIEWS}
    for label in ("trained", "snapshot", "baseline"):
        for key in ("orbit",) if label == "baseline" else VIEWS:
            Image.new("RGB", (800, 500), (40, 60, 80)).save(output / f"{label}-{key}.png")
    panels(output, settings)
    write_json(output / "cameras.json", {"cameras": cameras, "rehearsal": True})
    write_json(output / "scores.json", {"rehearsal": True, "trainingCameras": {}})


def run_console_check(
    root: Path, destination: Path, plan: dict[str, Any], *, billed_since: float | None = None
) -> dict[str, Any]:
    started = time.time() if billed_since is None else billed_since
    result = {**plan, "status": "failed", "error": None}
    destination.mkdir(parents=True, exist_ok=True)
    try:
        rehearsal = bool(plan.get("rehearsal"))
        run, scene, trained, split = source(root, plan["run"], rehearsal)
        other, other_scene, _, other_split = source(root, plan["compareRun"], rehearsal)
        if trained["scene"] != plan["scene"]:
            raise ValueError("--scene must match the primary run")
        if evaluation_identity(scene, split) != evaluation_identity(other_scene, other_split):
            raise ValueError("comparison requires the same frozen test pixels and world poses")
        payload, spz = live_inputs(run)
        settings: dict[str, Any] = {
            "sources": {
                label: {"runId": folder.name, **artifact_info(path)}
                for label, folder, path in (
                    ("trained", run, run / "trained.ply"),
                    ("snapshot", run, spz),
                    ("baseline", other, other / "trained.ply"),
                )
            },
            "frames": selected_frames(scene, split, trained),
            "liveCameras": payload,
            "orbit": orbit_camera(payload, plan["viewport"]),
            "rehearsal": rehearsal,
        }
        result["sources"] = settings["sources"]
        settings["sources"]["snapshot"]["step"] = int(spz.stem.split("_")[1])
        settings["sources"]["trained"]["iterations"] = trained.get("iterations")
        result["frameSelection"] = {
            "frames": settings["frames"],
            "rule": "nearest 10/50/90% time-ordered pool rank after exclusions; lexical fallback",
        }
        if rehearsal:
            synthetic(settings, destination)
        else:
            with tempfile.TemporaryDirectory(prefix="console-check-") as scratch:
                work = Path(scratch)
                frames, poses = work / "frames", work / "poses"
                frames.mkdir()
                for name in settings["frames"]:
                    shutil.copyfile(scene / "frames" / name, frames / name)
                shutil.copytree(scene / "poses", poses)
                sfm.keep_images(poses, set(settings["frames"]))
                data = training.build_dataset(
                    frames,
                    poses,
                    work / "data",
                    max_side=trained.get("trainParams", {}).get("train_max_side"),
                )
                settings.update(data=str(data), trainer=str(training.trainer_script(None)))
                write_json(destination / "settings.json", settings)
                remaining = plan["timeoutSeconds"] - (time.time() - started) - 15
                if remaining <= 0:
                    raise TimeoutError("no console-check render time left")
                subprocess.run(  # noqa: S603 - fixed script/interpreter, no shell
                    [
                        training.trainer_python(None),
                        str(Path(__file__).with_name("console_gpu.py")),
                        "--settings",
                        str(destination / "settings.json"),
                        "--out",
                        str(destination),
                    ],
                    check=True,
                    timeout=remaining,
                )
        result.update(
            status="complete",
            pngs=encode_pngs(destination),
            cameras=read_json(destination / "cameras.json"),
            scores=read_json(destination / "scores.json"),
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    elapsed = time.time() - started
    result.update(
        wallSeconds=elapsed,
        costUsd=0.0
        if plan.get("rehearsal")
        else elapsed * hourly_rate("l4", plan["hostMemoryGiB"]) / 3600,
    )
    write_json(destination / "result.json", {k: v for k, v in result.items() if k != "pngs"})
    return result


def save_local(state: Path, result: dict[str, Any]) -> None:
    if not re.fullmatch(r"console-check-[a-f0-9]{32}", result["runId"]):
        raise ValueError("invalid console-check result ID")
    directory = state / "runs" / result["runId"]
    images = result.pop("pngs", {})
    if result.get("status") == "complete" and set(images) != set(PNG_NAMES):
        raise ValueError("incomplete console-check PNG set")
    if set(images) - set(PNG_NAMES) or sum(len(v) for v in images.values()) > MAX_PNG_BASE64_BYTES:
        raise ValueError("invalid/oversized console-check PNG payload")
    directory.mkdir(parents=True, exist_ok=True)
    for name, encoded in images.items():
        content = base64.b64decode(encoded, validate=True)
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("invalid console-check PNG")
        (directory / name).write_bytes(content)
    for key in ("cameras", "scores"):
        if key in result:
            write_json(directory / f"{key}.json", result[key])
    result["localPath"] = str(directory)
    write_json(directory / "console-check.json", result)

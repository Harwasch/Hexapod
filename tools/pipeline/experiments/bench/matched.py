"""One budgeted comparison call; frozen inputs and kept exports never leave the volume."""

from __future__ import annotations

import base64
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import sfm
import training
from holdout_error import read_ply

from .ledger import hourly_rate
from .matched_gpu import PNG_NAMES, panels
from .matched_maths import choose_frame, closeup_camera
from .scene import read_json, verify, write_json
from .topdown_maths import frame_camera


def source_run(root: Path, prefix: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{8,32}", prefix):
        raise ValueError("--runs needs bench run IDs or unambiguous 8+ hex prefixes")
    matches = [
        p for p in root.glob(prefix + "*") if re.fullmatch(r"[a-f0-9]{32}", p.name) and p.is_dir()
    ]
    if len(matches) != 1:
        raise ValueError(f"run prefix {prefix!r} is missing or ambiguous on the volume")
    return matches[0]


def sources(root: Path, plan: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for prefix in plan["runs"]:
        run = source_run(root, prefix)
        result = read_json(run / "result.json")
        if result.get("status") != "complete" or result.get("rehearsal", False) != plan.get(
            "rehearsal", False
        ):
            raise ValueError(
                "matched renders require completed runs in the same rehearsal namespace"
            )
        if any(
            result.get(key) != value
            for key, value in (
                ("scene", plan["scene"]),
                ("manifestDigest", manifest["digest"]),
                ("splitDigest", manifest["splitDigest"]),
            )
        ):
            raise ValueError("matched renders require the same frozen frames, poses and split")
        name = result.get("headlinePly", "trained.ply")
        if name not in ("trained.ply", "faded.ply", "gated.ply"):
            raise ValueError("unknown headline PLY")
        ply = run / name
        if not ply.is_file():
            raise ValueError(f"{run.name}: missing {name}; training must use --keep-splat")
        selected.append({"runId": run.name, "ply": str(ply), "result": result})
    if selected[0]["runId"] == selected[1]["runId"]:
        raise ValueError("matched renders need two distinct runs")
    degrees = [s["result"].get("test", {}).get("shDegree") for s in selected]
    degrees = [
        s["result"].get("trainParams", {}).get("ship_sh_degree", 0) if d is None else d
        for s, d in zip(selected, degrees, strict=True)
    ]
    if degrees[0] != degrees[1]:
        raise ValueError("matched renders require the same shipped SH degree")
    return selected


def synthetic_renders(
    scene: Path, output: Path, settings: dict[str, Any], model: sfm.Model
) -> None:
    """Exercise cameras, crops, transport and labels; these are not GPU predictions."""
    pose = model.by_name()[settings["frame"]]
    camera = next(c for c in model.cameras if c.id == pose.camera_id)
    if camera.model != "PINHOLE":
        raise ValueError("synthetic matched renderer requires a PINHOLE camera")
    fx, fy, cx, cy = camera.params
    view = np.eye(4)
    view[:3, :3], view[:3, 3] = pose.rotation, pose.tvec
    ground = {
        "viewmat": view.tolist(),
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "width": camera.width,
        "height": camera.height,
        "near": 0.01,
        "far": 1e10,
    }
    closeup, crop = closeup_camera(ground)
    cameras = {"ground": ground, "topdown": settings["topdown"], "closeup": closeup}
    gt = Image.open(scene / "frames" / settings["frame"])
    gt.save(output / "GT-ground.png")
    gt.crop(crop).save(output / "GT-closeup.png")
    for label, color in (("A", (80, 40, 20)), ("B", (20, 80, 40))):
        for view_name, spec in cameras.items():
            Image.new("RGB", (spec["width"], spec["height"]), color).save(
                output / f"{label}-{view_name}.png"
            )
    write_json(
        output / "cameras.json",
        {
            "frame": settings["frame"],
            "crop": list(crop),
            "cameras": cameras,
            "renderer": "synthetic",
            "rehearsal": True,
        },
    )
    panels(output, synthetic=True)


def run_matched(
    root: Path, destination: Path, plan: dict[str, Any], *, billed_since: float | None = None
) -> dict[str, Any]:
    started = time.time() if billed_since is None else billed_since
    result: dict[str, Any] = {
        "runId": plan["runId"],
        "scene": plan["scene"],
        "schedule": "matched",
        "tag": plan.get("tag"),
        "rehearsal": bool(plan.get("rehearsal")),
        "benchCommit": plan["benchCommit"],
        "status": "failed",
        "error": None,
    }
    destination.mkdir(parents=True, exist_ok=True)
    try:
        scene = root / "scenes" / plan["scene"]
        manifest, split = verify(scene, plan.get("manifestDigest"))
        pair = sources(root / "runs", plan, manifest)
        result.update(
            runs=[s["runId"] for s in pair],
            manifestDigest=manifest["digest"],
            splitDigest=manifest["splitDigest"],
        )
        model = sfm.read_model(scene / "poses")
        up = sfm.camera_up(model)
        columns = read_ply(Path(pair[0]["ply"]))
        topdown = frame_camera(
            np.stack([columns[k] for k in ("x", "y", "z")], axis=1),
            model.centres(),
            None if up is None else up.up,
        )
        frame = choose_frame(
            split["test"],
            [model.by_name()[name].centre for name in split["test"]],
            topdown["up"],
            plan.get("frame"),
        )
        settings = {
            "frame": frame,
            "topdown": topdown,
            "sources": pair,
            "degree": pair[0]["result"].get("trainParams", {}).get("ship_sh_degree", 0),
        }
        if plan.get("rehearsal"):
            synthetic_renders(scene, destination, settings, model)
        else:
            with tempfile.TemporaryDirectory(prefix="bench-matched-") as temporary:
                data = training.build_dataset(
                    scene / "frames",
                    scene / "poses",
                    Path(temporary) / "data",
                    max_side=pair[0]["result"].get("trainParams", {}).get("train_max_side"),
                )
                # A refreeze during dataset construction cannot silently substitute
                # different pixels or poses. Verify again before starting the render.
                verify(scene, manifest["digest"])
                settings.update(data=str(data), trainer=str(training.trainer_script(None)))
                path = destination / "settings.json"
                write_json(path, settings)
                remaining = plan["timeoutSeconds"] - (time.time() - started) - 15
                if remaining <= 0:
                    raise TimeoutError("no matched render time left")
                subprocess.run(  # noqa: S603 - fixed interpreter/script; no shell
                    [
                        training.trainer_python(None),
                        str(Path(__file__).with_name("matched_gpu.py")),
                        "--settings",
                        str(path),
                        "--out",
                        str(destination),
                    ],
                    check=True,
                    timeout=remaining,
                )
        images = {}
        for name in PNG_NAMES:
            content = (destination / name).read_bytes()
            if len(content) > 32_000_000:
                raise ValueError("matched PNG exceeds 32 MB transport limit")
            images[name] = base64.b64encode(content).decode("ascii")
        result.update(
            status="complete",
            pngs=images,
            cameras=read_json(destination / "cameras.json"),
            sources=[
                {
                    "runId": s["runId"],
                    "ply": Path(s["ply"]).name,
                    "qualityGate": s["result"].get("qualityGate"),
                }
                for s in pair
            ],
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    elapsed = time.time() - started
    result.update(
        wallSeconds=elapsed,
        costUsd=0.0 if plan.get("rehearsal") else elapsed * hourly_rate("l4") / 3600,
    )
    write_json(destination / "result.json", {k: v for k, v in result.items() if k != "pngs"})
    return result

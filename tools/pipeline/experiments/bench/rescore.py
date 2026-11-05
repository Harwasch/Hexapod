"""Budgeted, evaluation-only rescore orchestration and local report transport."""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import sfm
import training
from holdout_error import read_ply

from .ledger import hourly_rate
from .matched import source_run
from .rescore_maths import MODES, RESOLUTIONS, footprint, needle_scores, shared_cameras
from .scene import read_json, verify, write_json


def inventory(
    root: Path, plan: dict[str, Any], manifest: dict[str, Any]
) -> tuple[list[Any], list[Any]]:
    selected, unavailable = [], []
    resolved: set[str] = set()
    for prefix in plan["runs"]:
        run = source_run(root, prefix)
        if run.name in resolved:
            raise ValueError("rescore needs distinct resolved run IDs")
        resolved.add(run.name)
        result = read_json(run / "result.json")
        if result.get("status") != "complete" or result.get("rehearsal", False) != plan.get(
            "rehearsal", False
        ):
            raise ValueError("rescore needs completed runs in the same rehearsal namespace")
        for key, expected in (
            ("scene", plan["scene"]),
            ("manifestDigest", manifest["digest"]),
            ("splitDigest", manifest["splitDigest"]),
        ):
            if result.get(key) != expected:
                raise ValueError("rescore requires identical frozen frames, poses and split")
        requested = plan.get("ply") or ["trained", "faded"]
        for kind in requested:
            ply = run / f"{kind}.ply"
            if not ply.is_file():
                if plan.get("ply") or kind == "trained":
                    unavailable.append(
                        {
                            "runId": run.name,
                            "ply": kind,
                            "reason": (
                                "no kept PLY on volume; cannot rescore "
                                "(training needed --keep-splat)"
                            ),
                        }
                    )
                continue
            selected.append(
                {
                    "runId": run.name,
                    "plyKind": kind,
                    "ply": str(ply),
                    "plyBytes": ply.stat().st_size,
                    "result": result,
                }
            )
    if len(selected) > 4:
        raise ValueError("at most four kept PLY rows per 40-minute rescore call")
    degrees = {s["result"].get("trainParams", {}).get("ship_sh_degree", 0) for s in selected}
    if len(degrees) > 1:
        raise ValueError("rescore requires the same shipped SH degree")
    return selected, unavailable


def trained_at(source: dict[str, Any], full: list[int]) -> dict[str, Any]:
    from prior_maths import resize_geometry

    params = source["result"].get("trainParams", {})
    h, w = full
    h, w, _, _ = resize_geometry(h, w, params.get("train_max_side") or 0)
    factor = int(params.get("data_factor", 1))
    h, w = h // factor, w // factor
    h, w, _, _ = resize_geometry(h, w, params.get("step_max_side") or 0)
    return {
        "height": h,
        "width": w,
        "source": "frozen dimensions + recorded train_max_side/data_factor/step_max_side",
    }


def row_base(source: dict[str, Any], full: list[int]) -> dict[str, Any]:
    return {
        "runId": source["runId"],
        "ply": source["plyKind"],
        "plyBytes": source["plyBytes"],
        "variant": source["result"].get("variant", {}).get("name", "unknown"),
        "trainedAt": trained_at(source, full),
        "trainingLoss": "historical PLY; not modified or retrained by rescore",
    }


def comparisons(output: Path, rows: list[Any], cameras: dict[str, Any]) -> None:
    for camera in cameras["cameras"]:
        name = camera["name"]
        images = [
            Image.open(output / f"row-{i}-{name}.png").convert("RGB") for i in range(len(rows))
        ]
        panel = Image.new("RGB", (camera["width"] * len(rows), camera["height"] + 45))
        draw = ImageDraw.Draw(panel)
        for i, (row, image) in enumerate(zip(rows, images, strict=True)):
            panel.paste(image, (i * camera["width"], 45))
            draw.text(
                (i * camera["width"] + 4, 4),
                f"{row['runId'][:8]} {row['ply']}\nforeground only / {name}",
                fill="white",
            )
        panel.save(output / f"comparison-{name}.png")


def synthetic(settings: dict[str, Any], output: Path) -> None:
    """Read stand-in PLYs; never label synthetic photo predictions as GPU measurements."""
    rows = []
    full = settings["fullSize"]
    for i, source in enumerate(settings["sources"]):
        row = row_base(source, full)
        row.update(
            rehearsal=True,
            needle=needle_scores(read_ply(Path(source["ply"])), settings["cameras"]["extent"]),
            topdown={"holeMean": None, "perCamera": []},
            resolutions={},
        )
        for name, size in (("full", full), ("half", [v // 2 for v in full])):
            row["resolutions"][name] = {
                "scoredAt": {"height": size[0], "width": size[1]},
                "metrics": {k: dict(psnr=None, ssim=None, lpips=None) for k in MODES},
                "perFrame": [],
                "poseMoveMedian": {"translationPct": None, "rotationDeg": None},
            }
        for camera in settings["cameras"]["cameras"]:
            image = Image.new("RGB", (camera["width"], camera["height"]), (30 + 20 * i, 60, 90))
            ImageDraw.Draw(image).text((8, 8), "SYNTHETIC REHEARSAL", fill="white")
            image.save(output / f"row-{i}-{camera['name']}.png")
        rows.append(row)
    write_json(output / "scores.json", {"rows": rows})


def run_rescore(
    root: Path, destination: Path, plan: dict[str, Any], *, billed_since: float | None = None
) -> dict[str, Any]:
    started = time.time() if billed_since is None else billed_since
    result: dict[str, Any] = {
        "schemaVersion": 1,
        "runId": plan["runId"],
        "scene": plan["scene"],
        "schedule": "rescore",
        "tag": plan.get("tag"),
        "benchCommit": plan["benchCommit"],
        "status": "failed",
        "error": None,
        "rehearsal": bool(plan.get("rehearsal")),
        "renderer": "classic 3DGS, black background, shipped SH; no ISP",
        "fitWarning": (
            "Aligned and colour-corrected metrics fit the test image itself; "
            "these flatter every run alike and are not untouched-test predictions."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    try:
        scene = root / "scenes" / plan["scene"]
        manifest, split = verify(scene, plan.get("manifestDigest"))
        sources, unavailable = inventory(root / "runs", plan, manifest)
        result.update(
            unavailable=unavailable, manifestDigest=manifest["digest"], splitDigest=split["digest"]
        )
        if not sources:
            raise ValueError("no kept PLYs available; no scores fabricated")
        with tempfile.TemporaryDirectory(prefix="bench-rescore-") as temporary:
            frozen = Path(temporary) / "frozen"
            shutil.copytree(scene, frozen)
            verify(frozen, manifest["digest"])
            model = sfm.read_model(frozen / "poses")
            up = sfm.camera_up(model)
            cameras = shared_cameras(
                sfm.read_points3d(frozen / "poses/points3D.bin").xyz,
                model.centres(),
                None if up is None else up.up,
            )
            cameras.update(manifestDigest=manifest["digest"], splitDigest=split["digest"])
            write_json(destination / "cameras.json", cameras)
            for camera in cameras["cameras"]:
                mask = footprint(camera, cameras["footprint"])
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    destination / f"footprint-{camera['name']}.png"
                )
            sizes = {Image.open(frozen / "frames" / name).size for name in split["test"]}
            if len(sizes) != 1:
                raise ValueError("rescore requires one explicit full size across the test set")
            width, height = sizes.pop()
            if width % 2 or height % 2:
                raise ValueError("full/half scoring requires even frozen dimensions")
            settings = {
                "sources": sources,
                "cameras": cameras,
                "names": split["test"],
                "fullSize": [height, width],
                "steps": [160, 40],
                "maxTranslationPct": 2.0,
                "maxRotationDeg": 3.0,
                "deadline": started + plan["timeoutSeconds"] - 30,
            }
            result["alignment"] = {
                key: settings[key] for key in ("steps", "maxTranslationPct", "maxRotationDeg")
            }
            if plan.get("rehearsal"):
                synthetic(settings, destination)
            else:
                data = training.build_dataset(
                    frozen / "frames", frozen / "poses", Path(temporary) / "data"
                )
                settings.update(data=str(data), trainer=str(training.trainer_script(None)))
                config = destination / "settings.json"
                write_json(config, settings)
                with (destination / "rescore.log").open("w") as log:
                    subprocess.run(  # noqa: S603 - fixed interpreter and script
                        [
                            training.trainer_python(None),
                            str(Path(__file__).with_name("rescore_gpu.py")),
                            "--settings",
                            str(config),
                            "--out",
                            str(destination),
                        ],
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=max(1, settings["deadline"] - time.time()),
                    )
            result.update(read_json(destination / "scores.json"))
            comparisons(destination, result["rows"], cameras)
            result.update(status="complete", cameras=cameras)
    except Exception as error:
        from progress import tail

        result["error"] = f"{type(error).__name__}: {error}\n" + "\n".join(
            tail(destination / "rescore.log").splitlines()[-40:]
        )
    elapsed = time.time() - started
    result.update(
        wallSeconds=elapsed,
        costUsd=0.0 if plan.get("rehearsal") else elapsed * hourly_rate("l4") / 3600,
    )
    write_json(destination / "rescore.json", result)
    result["pngs"] = {
        p.name: base64.b64encode(p.read_bytes()).decode("ascii") for p in destination.glob("*.png")
    }
    return result


def save_local(state: Path, result: dict[str, Any]) -> None:
    """Only generated flat PNG names and validated UUIDs can cross the transport."""
    if not re.fullmatch(r"rescore-[a-f0-9]{32}", result["runId"]):
        raise ValueError("invalid rescore batch ID")
    directory = state / "runs" / result["runId"]
    directory.mkdir(parents=True, exist_ok=True)
    pngs = result.pop("pngs", {})
    if sum(len(value) for value in pngs.values()) > 96_000_000:
        raise ValueError("rescore PNG payload exceeds 96 MB")
    for name, encoded in pngs.items():
        if not re.fullmatch(
            r"(?:row-\d+-[a-z0-9+-]+|comparison-overhead-\d+|footprint-overhead-\d+)\.png", name
        ):
            raise ValueError("invalid rescore PNG name")
        content = base64.b64decode(encoded, validate=True)
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("invalid rescore PNG")
        (directory / name).write_bytes(content)
    write_json(directory / "rescore.json", result)
    if "cameras" in result:
        write_json(directory / "cameras.json", result["cameras"])
    by_run: dict[str, list[Any]] = {}
    for row in result.get("rows", []):
        if not re.fullmatch(r"[a-f0-9]{32}", row["runId"]):
            raise ValueError("invalid source run ID")
        by_run.setdefault(row["runId"], []).append(row)
    for run, rows in by_run.items():
        target = state / "runs" / run
        write_json(target / "rescore.json", {**result, "rows": rows, "batchPath": str(directory)})
        for i, row in enumerate(result["rows"]):
            if row["runId"] == run:
                for path in directory.glob(f"row-{i}-*.png"):
                    folder = target / "rescore-renders"
                    folder.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, folder / path.name)


def table(documents: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Never silently mix different frozen cameras/protocols in an apples-to-apples table."""
    if (
        len(
            {
                (
                    d.get("manifestDigest"),
                    d.get("splitDigest"),
                    d.get("protocol", "rescore-v1"),
                    bool(d.get("rehearsal")),
                )
                for d in documents
            }
        )
        > 1
    ):
        raise ValueError("rescore table requires identical frozen scene and protocol")
    rows = [r for d in documents if d["status"] == "complete" for r in d.get("rows", [])]
    unique = {(r["runId"], r["ply"]): r for r in rows}
    lines = ["Classic 3DGS. Aligned/CC fit the test frame itself and flatter all runs alike.\n"]
    if any(d.get("rehearsal") for d in documents):
        lines.append(
            "SYNTHETIC REHEARSAL: photo/hole metrics are unavailable; not GPU measurements.\n"
        )
    tables: dict[str, Any] = {}

    def number(value: Any, scale: float = 1) -> str:
        return "—" if value is None else f"{value * scale:.3f}"

    def shape(size: dict[str, Any]) -> str:
        return f"{size['height']}x{size['width']}"

    for resolution in RESOLUTIONS:
        lines += [
            f"\n{resolution}\n",
            (
                "| run | variant | trainedAt | scoredAt | PSNR/SSIM/LPIPS raw | aligned | cc | "
                "aligned+cc | top-down hole % | elongation share | oversized 2% | "
                "pose move median (% extent / deg) |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        tables[resolution] = []
        for row in unique.values():
            scored = row["resolutions"][resolution]
            trained, size = shape(row["trainedAt"]), shape(scored["scoredAt"])
            values = [
                f"{row['runId'][:8]}:{row['ply']}",
                row["variant"],
                trained,
                size + (" (differs)" if size != trained else ""),
            ]
            values += [
                "/".join(number(scored["metrics"][mode][k]) for k in ("psnr", "ssim", "lpips"))
                for mode in MODES
            ]
            values += [
                number(row["topdown"]["holeMean"], 100),
                number(row["needle"]["elongation"]["share"]),
                number(row["needle"]["oversized"]["2"]["share"]),
                "/".join(
                    number(scored["poseMoveMedian"][k]) for k in ("translationPct", "rotationDeg")
                ),
            ]
            lines.append(
                "| "
                + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values)
                + " |"
            )
            tables[resolution].append(
                {
                    "runId": row["runId"],
                    "ply": row["ply"],
                    "variant": row["variant"],
                    "trainedAt": row["trainedAt"],
                    **scored,
                    "topdown": row["topdown"],
                    "needle": row["needle"],
                }
            )
    unavailable = list(
        {
            (r["runId"], r.get("ply")): r for d in documents for r in d.get("unavailable", [])
        }.values()
    )
    failures = [
        {"runId": d.get("runId"), "error": d.get("error")}
        for d in documents
        if d.get("status") != "complete"
    ]
    lines += [f"\nFailed batch {f['runId']}: {f['error']}" for f in failures]
    lines += [
        f"\nUnavailable: {r['runId'][:8]} {r.get('ply', '')}: {r['reason']}" for r in unavailable
    ]
    return "\n".join(lines) + "\n", {
        "renderer": "classic",
        "protocol": "rescore-v1",
        "manifestDigest": documents[0].get("manifestDigest") if documents else None,
        "splitDigest": documents[0].get("splitDigest") if documents else None,
        "rehearsal": any(d.get("rehearsal") for d in documents),
        "fitWarning": "Aligned/CC fit the test image itself; not untouched-test predictions.",
        "tables": tables,
        "failures": failures,
        "unavailable": unavailable,
    }

"""Native-rate frames and leakage-safe pools anchored to an immutable bench test set."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import exposure
import gaussian_budget
import sfm
import training
import video

from .scene import digest, read_json, safe_name, seal, verify, write_json

RULE = "inherited test cameras; native registered frames outside inclusive test time windows"


def timestamps(root: Path) -> dict[str, float]:
    records = read_json(root / "frames_meta.json")["frames"]
    result = {r["name"]: float(r["offsetS"]) for r in records if r.get("offsetS") is not None}
    if any(not math.isfinite(t) for t in result.values()):
        raise ValueError("nonfinite frame timestamp")
    return result


def exclude_pool(
    root: Path, split: dict[str, Any], seconds: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("test exclusion must be finite and nonnegative")
    offsets = timestamps(root)
    if set(split["pool"] + split["test"]) - offsets.keys():
        raise ValueError("test exclusion requires timestamps for every pool and test frame")
    test = np.array([offsets[n] for n in split["test"]])
    dropped = [n for n in split["pool"] if np.min(np.abs(test - offsets[n])) <= seconds]
    excluded = set(dropped)
    pool = [n for n in split["pool"] if n not in excluded]
    report = {
        "seconds": seconds,
        "inclusive": True,
        "poolBefore": len(split["pool"]),
        "poolAfter": len(pool),
        "excludedCount": len(dropped),
        "excluded": dropped,
        "reason": "prevent temporally adjacent test-frame leakage",
        "test": split["test"],
    }
    return {**split, "pool": pool}, report


def evaluation_identity(root: Path, split: dict[str, Any]) -> str:
    """Exact pixels, intrinsics and extrinsics; independent of database numeric IDs."""
    import hashlib

    model = sfm.read_model(root / "poses")
    images = {i.name: i for i in model.images}
    records = []
    for name in sorted(split["test"]):
        image = images[name]
        camera = next(c for c in model.cameras if c.id == image.camera_id)
        records.append(
            {
                "name": name,
                "pixels": hashlib.sha256(
                    (root / "frames" / safe_name(name)).read_bytes()
                ).hexdigest(),
                "qvec": image.qvec,
                "tvec": image.tvec,
                "model": camera.model,
                "width": camera.width,
                "height": camera.height,
                "params": camera.params,
            }
        )
    return digest(records)


def select_frames(
    records: list[dict[str, Any]],
    thumbs: list[np.ndarray],
    *,
    blur_ratio: float = 0.5,
    duplicate_mae: float = 1.0,
) -> list[dict[str, Any]]:
    """Relative blur outliers; near-identical frames compared to the last retained frame.

    Thresholds are explicit first-principles rules, not a capture-calibrated optimum.
    One 8-bit level MAE at 160x thumbnails is conservative; no top-K or count limit.
    """
    if len(records) != len(thumbs) or not 0 <= blur_ratio < 1 or duplicate_mae < 0:
        raise ValueError("invalid frame selection inputs")
    sharp = np.array([r["sharpness"] for r in records], dtype=float)
    last: np.ndarray | None = None
    output = []
    for index, record in enumerate(records):
        neighbourhood = sharp[max(0, index - 5) : index + 6]
        reference = float(np.median(neighbourhood))
        relative = float(sharp[index] / reference) if reference > 0 else 1.0
        difference = (
            None if last is None else float(np.mean(np.abs(thumbs[index].astype(float) - last)))
        )
        reason = (
            "blur"
            if relative < blur_ratio
            else "near-duplicate"
            if difference is not None and difference <= duplicate_mae
            else "kept"
        )
        output.append(
            {
                **record,
                "relativeSharpness": relative,
                "neighbourMedianSharpness": reference,
                "differenceFromLastKept": difference,
                "selectionReason": reason,
                "kept": reason == "kept",
            }
        )
        if reason == "kept":
            last = thumbs[index]
    return output


def extract(
    source: Path, frames: Path, *, blur_ratio: float, duplicate_mae: float
) -> dict[str, Any]:
    meta = video.probe(source)
    tonemap = video.TONEMAP_FILTER if meta.is_hdr else None
    if tonemap and not video.ffmpeg_filters() >= video.TONEMAP_NEEDS:
        raise ValueError("HDR source requires normalize's zscale/tonemap filters")
    frames.mkdir(parents=True)
    command = video.extract_frames_argv(
        source, frames / "full_%06d.jpg", fps=None, tonemap=tonemap, timestamps=True
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
    offsets = [
        float(t)
        for t in re.findall(r"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:([-+\d.e]+)", completed.stderr)
    ]
    paths = sorted(frames.glob("full_*.jpg"))
    if len(offsets) != len(paths) or len(paths) < 2 or any(b <= a for a, b in pairwise(offsets)):
        raise ValueError("decode timestamps must account for every source frame in order")
    records, thumbs = [], []
    for index, (path, offset) in enumerate(zip(paths, offsets, strict=True)):
        with Image.open(path) as image:
            grey = np.asarray(image.convert("L"))
            luma, high, low = exposure.luma_stats(grey)
            thumb = image.convert("L")
            thumb.thumbnail((160, 160))
            thumbs.append(np.asarray(thumb).copy())
            records.append(
                {
                    "name": path.name,
                    "source": source.name,
                    "candidateIndex": index,
                    "offsetS": offset - offsets[0],
                    "sharpness": video.sharpness(path),
                    "frameSize": list(image.size),
                    "exposure": {"lumaMean": luma, "clippedHighPct": high, "clippedLowPct": low},
                }
            )
    selected = select_frames(records, thumbs, blur_ratio=blur_ratio, duplicate_mae=duplicate_mae)
    for row in selected:
        if not row["kept"]:
            (frames / row["name"]).unlink()
    return {
        "source": meta.to_dict(),
        "reportedFps": meta.fps,
        "decodedFrames": len(paths),
        "meanDecodedFps": (len(paths) - 1) / (offsets[-1] - offsets[0]),
        "tonemap": tonemap,
        "selection": {
            "blurRatio": blur_ratio,
            "duplicateMae8bit": duplicate_mae,
            "neighbourRadiusFrames": 5,
            "thumbnailMaxSide": 160,
            "countCap": None,
            "basis": (
                "relative blur below half local median; MAE within one 8-bit code level; "
                "not calibrated on this capture"
            ),
        },
        "counts": dict(Counter(r["selectionReason"] for r in selected)),
        "frames": selected,
    }


def find_original(store: Any, prefix: str, source_key: str | None = None) -> dict[str, Any]:
    """Run transfer inputs first, then the capture uploads' unique original filename.

    Normalize can run on the worker, leaving its upload only under captures/, not runs/.
    Never guess between duplicate filenames from different captures.
    """

    def matches(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [obj for obj in objects if Path(obj["Key"]).name.lower() == "img_3450.mov"]

    candidates: list[dict[str, Any]]
    if source_key:
        candidates = [obj for obj in store.list(source_key) if obj["Key"] == source_key]
    else:
        candidates = matches(store.list(prefix))
        if not candidates:
            candidates = matches(store.list("captures/"))
    if len(candidates) != 1:
        raise ValueError(f"found {len(candidates)} originals; supply --source-key for IMG_3450.mov")
    return candidates[0]


def freeze_full(store: Any, scenes: Path, plan: dict[str, Any]) -> dict[str, Any]:
    from .freeze import resolve_run
    from .full_register import register

    began = time.monotonic()
    base = scenes / "beach"
    original, split = verify(base)
    destination = scenes / "beach-full"
    if destination.exists():
        raise ValueError("beach-full already frozen; keep it immutable (use a new scene revision)")
    work = scenes / (".full-" + plan["runId"])
    if work.exists():
        raise ValueError("freeze attempt already exists; restart refused")
    work.mkdir(parents=True)
    prefix = resolve_run(store, plan["sourceRun"])
    original_object = find_original(store, prefix, plan.get("sourceKey"))
    source = work / "IMG_3450.mov"
    store.get(original_object, source)
    root = work / "scene"
    report = extract(
        source, root / "frames", blur_ratio=plan["blurRatio"], duplicate_mae=plan["duplicateMae"]
    )
    offsets = timestamps(base)
    native_offsets = {r["name"]: r["offsetS"] for r in report["frames"] if r["kept"]}
    for name in offsets:
        shutil.copyfile(base / "frames" / name, root / "frames" / name)
    registration = register(base, root, work / "registration", offsets, native_offsets)
    model = sfm.read_model(root / "poses")
    registered = {i.name for i in model.images}
    combined = [*read_json(base / "frames_meta.json")["frames"], *report["frames"]]
    for row in combined:
        row["registered"] = row["name"] in registered
    write_json(root / "frames_meta.json", {**report, "frames": combined})
    inherited = {
        "scene": "beach-full",
        "rule": RULE,
        "test": split["test"],
        "pool": sorted(registered & native_offsets.keys()),
        "parentManifestDigest": original["digest"],
    }
    inherited, exclusion = exclude_pool(root, inherited, plan["excludeTestSeconds"])
    inherited["exclusionSeconds"] = plan["excludeTestSeconds"]
    inherited["digest"] = digest(inherited)
    if len(inherited["pool"]) < 2:
        raise ValueError("fewer than two eligible native training frames")
    pool = set(inherited["pool"])
    excluded = set(exclusion["excluded"])
    for row in combined:
        name = row["name"]
        row["trainingEligible"] = name in pool
        row["trainingReason"] = (
            "test"
            if name in split["test"]
            else "backbone replaced by native source frames"
            if name in offsets
            else row["selectionReason"]
            if not row.get("kept", True)
            else "registration failed"
            if name not in registered
            else "test time exclusion"
            if name in excluded
            else "training"
        )
    write_json(root / "frames_meta.json", {**report, "frames": combined})
    budget_model = work / "budget-model"
    shutil.copytree(root / "poses", budget_model)
    sfm.keep_images(budget_model, pool)
    with Image.open(root / "frames" / inherited["pool"][0]) as photo:
        size = photo.size
    budget = gaussian_budget.plan("auto", model=budget_model, train_size=size, pixel_scale=1.0)
    assert budget is not None
    steps = min(
        60000,
        training.scaled_steps(
            30000, training.schedule_scale(len(pool), full_at=60, floor=0.25, growth="sqrt")
        ),
    )
    # A planning proxy, not a measured throughput for the new solve. The timeout wins.
    estimated_step_s = (
        (4368.045 / 55800) * (size[0] * size[1] / (900 * 1600)) * (budget.cap / 3456214)
    )
    report["trainingForecast"] = {
        "budget": budget.to_dict(),
        "steps": steps,
        "rule": "30000 * sqrt(pool/60) above 60; floor .25, maximum 60000 steps",
        "secondsEstimate": steps * estimated_step_s + 600,
        "basis": (
            "upgrade measured 4368.045 s/55800 steps at 900x1600 and 3456214 splats; "
            "linear pixels*budget proxy plus 600 s overhead; not a guarantee"
        ),
        "hardTimeoutSeconds": 16200,
        "limits": [
            {
                "name": "steps",
                "value": 60000,
                "binding": steps == 60000,
                "reason": "$8 experiment cap",
            },
            {
                "name": "gpuMemoryGb",
                "value": 24,
                "binding": budget.clamp == "gpu-memory",
                "reason": "L4 device capacity",
            },
        ],
    }
    identity = evaluation_identity(base, split)
    if evaluation_identity(root, inherited) != identity:
        raise ValueError("frozen test pixels or camera parameters changed")
    report.update(
        registration=registration,
        exclusion=exclusion,
        evaluationIdentity=identity,
        backboneFrames=len(offsets),
        nativePoolFrames=len(inherited["pool"]),
        wallSeconds=time.monotonic() - began,
        sourceObject={k: original_object[k] for k in ("Key", "Size", "ETag")},
    )
    write_json(root / "full_data.json", report)
    manifest = seal(
        root,
        "beach-full",
        prefix,
        "full-data",
        plan["benchCommit"],
        inherited_split=inherited,
        evaluation_identity=identity,
    )
    root.rename(destination)
    shutil.rmtree(work)
    return manifest

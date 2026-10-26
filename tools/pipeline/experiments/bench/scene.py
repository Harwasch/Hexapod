"""Immutable scenes, deterministic splits, and the train-only COLMAP model."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sfm

SCENES = {
    "spool": "ec644db5-ceaf-4c01-bf89-b261623327c2",
    "pumpkin": "71556411",
    "nwm-d1": "50bc26d9",
}
RULE = "sha256(scene:frame_name), lowest max(1, floor(n/8 + 0.5)); lexical tie-break"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def split_frames(scene: str, names: list[str]) -> dict[str, Any]:
    if not names or len(set(names)) != len(names):
        raise ValueError("registered frame names must be nonempty and unique")
    ranked = sorted(
        names, key=lambda name: (hashlib.sha256(f"{scene}:{name}".encode()).digest(), name)
    )
    count = max(1, (len(names) + 4) // 8)
    test = sorted(ranked[:count])
    pool = sorted(set(names) - set(test))
    content = {"scene": scene, "rule": RULE, "test": test, "pool": pool}
    return {**content, "digest": digest(content)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def safe_name(name: str) -> str:
    # build_dataset supports flat frame directories only. Fail rather than silently omit.
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"expected a flat file name: {name!r}")
    return name


def inventory(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("frozen inputs cannot contain symlinks")
        if not path.is_file() or path.relative_to(root).as_posix() in {
            "manifest.json",
            "split.json",
        }:
            continue
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                checksum.update(block)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": checksum.hexdigest(),
            }
        )
    return files


def seal(root: Path, scene: str, source: str, generation: str, commit: str) -> dict[str, Any]:
    model = sfm.read_model(root / "poses")
    names = [safe_name(image.name) for image in model.images]
    for name in names:
        if not (root / "frames" / name).is_file():
            raise ValueError(f"registered frame missing: {name}")
    split = split_frames(scene, names)
    manifest = {
        "schemaVersion": 1,
        "scene": scene,
        "sourcePrefix": source,
        "generation": generation,
        "files": inventory(root),
        "frozenAt": datetime.now(UTC).isoformat(),
        "benchCommit": commit,
        "splitDigest": split["digest"],
    }
    manifest["digest"] = digest(manifest)
    write_json(root / "split.json", split)
    write_json(root / "manifest.json", manifest)
    return manifest


def verify(root: Path, expected: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, split = read_json(root / "manifest.json"), read_json(root / "split.json")
    body = {k: v for k, v in manifest.items() if k != "digest"}
    if digest(body) != manifest["digest"] or inventory(root) != manifest["files"]:
        raise ValueError("frozen scene digest changed; refreeze explicitly")
    if expected is not None and manifest["digest"] != expected:
        raise ValueError("frozen scene changed since this batch was planned")
    actual = split_frames(
        manifest["scene"], [i.name for i in sfm.read_model(root / "poses").images]
    )
    if split != actual or manifest["splitDigest"] != split["digest"]:
        raise ValueError("frozen split digest changed")
    return manifest, split


def prepare_pool(root: Path, inputs: Path, split: dict[str, Any]) -> None:
    if len(split["pool"]) < 2:
        raise ValueError("need at least two pool images for train and validation")
    frames, poses = inputs / "frames", inputs / "poses"
    frames.mkdir(parents=True)
    for name in split["pool"]:
        shutil.copyfile(root / "frames" / safe_name(name), frames / name)
    poses.mkdir()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copyfile(root / "poses" / name, poses / name)
    # The existing binary editor preserves extrinsics/observations and cuts tracks.
    # Do not copy poses.json: it contains the full model's stale registered count.
    sfm.keep_images(poses, set(split["pool"]))

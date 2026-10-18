"""Prepare pinned source and verify operator-supplied weights, without GPU imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text())
RECEIPT = ".worlds-ltx25-verified.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtimeNs": stat.st_mtime_ns}


def source_errors(source: Path, manifest: dict) -> list[str]:
    errors = []
    expected = manifest["sourceFiles"]
    for relative, checksum in expected.items():
        path = source / relative
        if not path.is_file() or digest(path) != checksum:
            errors.append("source:" + relative)
    # Refuse extra Python modules that could shadow pinned imports.
    for package in ("ltx-core", "ltx-pipelines"):
        for path in (source / "packages" / package / "src").rglob("*.py"):
            if path.relative_to(source).as_posix() not in expected:
                errors.append(
                    "unreviewed-source:" + path.relative_to(source).as_posix()
                )
    return errors


def check_artifacts(source: Path, weights: Path, manifest: dict | None = None) -> dict:
    """Cheap readiness: hash small source; require receipt from full weight verification.

    Mounted weights must remain immutable. Receipt fingerprints detect replacement;
    --verify-assets explicitly rehashes every byte before accepting a new mount.
    """
    manifest = manifest or MANIFEST
    source, weights = Path(source), Path(weights)
    missing = []
    try:
        missing.extend(source_errors(source, manifest))
        receipt = json.loads((weights / RECEIPT).read_text())
        if receipt.get("checkpointRevision") != manifest["checkpointRevision"]:
            missing.append("verified-checkpoint-revision")
        for item in manifest["files"]:
            path = weights / item["path"]
            record = receipt.get("files", {}).get(item["path"], {})
            if (
                not path.is_file()
                or path.stat().st_size != item["size"]
                or record.get("sha256") != item["sha256"]
                or record.get("signature") != signature(path)
            ):
                missing.append("weights:" + item["path"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        missing.append("verified-assets-receipt")
    return {
        "ready": not missing,
        "missing": missing,
        "sourceRevision": manifest["commit"],
        "weightsRevision": manifest["checkpointRevision"],
        "gpuInferenceVerified": False,
    }


def verify_packed_text_encoder(path: Path) -> None:
    # Safetensors stores a bounded JSON header before tensor bytes. No torch needed.
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Text encoder safetensors header is missing")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 64 * 1024 * 1024:
            raise ValueError(
                "Text encoder safetensors header is outside the supported bound"
            )
        header = json.loads(stream.read(length))
    metadata = header.get("__metadata__", {})
    if "gemma_config" not in metadata or "tokenizer_json" not in header:
        raise ValueError("Packed text encoder must contain Gemma config and tokenizer")
    for name in ("tokenizer_config.json", "processor_config.json"):
        if "hf_asset__" + name not in header and name not in metadata:
            raise ValueError("Packed text encoder is missing " + name)


def verify_assets(weights: Path, manifest: dict | None = None) -> None:
    manifest = manifest or MANIFEST
    records = {}
    for item in manifest["files"]:
        path = weights / item["path"]
        before = signature(path)
        if before["size"] != item["size"] or digest(path) != item["sha256"]:
            raise ValueError("Checkpoint size/hash mismatch: " + item["path"])
        if signature(path) != before:
            raise ValueError("Checkpoint changed while being verified")
        records[item["path"]] = {"signature": before, "sha256": item["sha256"]}
    verify_packed_text_encoder(weights / manifest["components"]["text_encoder"])
    receipt = {"checkpointRevision": manifest["checkpointRevision"], "files": records}
    temporary = weights / (RECEIPT + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, weights / RECEIPT)


def prepare_source(source: Path, manifest: dict | None = None) -> None:
    manifest = manifest or MANIFEST
    if not source.exists():
        source.mkdir(parents=True)
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "remote",
                "add",
                "origin",
                manifest["repository"],
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "fetch",
                "--depth",
                "1",
                "origin",
                manifest["commit"],
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True
        )
    errors = source_errors(source, manifest)
    if errors:
        raise ValueError("Pinned source contract mismatch: " + ", ".join(errors[:5]))
    (source / ".worlds-ltx25-source").write_text(manifest["commit"] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--verify-assets",
        action="store_true",
        help="Hash existing weights and write readiness receipt; never downloads checkpoints",
    )
    args = parser.parse_args()
    if args.verify_assets and not args.weights:
        parser.error("--verify-assets requires --weights")
    try:
        prepare_source(args.source.resolve())
        if args.verify_assets:
            verify_assets(args.weights.resolve())
        print(
            json.dumps(
                {
                    "sourceRevision": MANIFEST["commit"],
                    "weightsDownloaded": False,
                    "assetsVerified": args.verify_assets,
                    "gpuInferenceVerified": False,
                }
            )
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()

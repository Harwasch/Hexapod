"""Prepare pinned public source/checkpoint locally; never provisions or deploys a worker."""

import argparse
import json
import subprocess
from pathlib import Path

MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        help="Download the pinned checkpoint to this directory (several GB)",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    if source.exists():
        revision = (
            subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        if revision != MANIFEST["sourceRevision"]:
            raise SystemExit(
                "Source directory exists at a different revision; choose an empty directory"
            )
    else:
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "remote",
                "add",
                "origin",
                MANIFEST["repository"],
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
                MANIFEST["sourceRevision"],
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True
        )
    if args.weights:
        from huggingface_hub import snapshot_download

        weights = args.weights.resolve()
        snapshot_download(
            repo_id=MANIFEST["checkpoint"],
            revision=MANIFEST["checkpointRevision"],
            local_dir=weights,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "License.txt",
                "Notice.txt",
            ],
        )
        for name in ("config.json", "model.safetensors"):
            if not (weights / name).is_file():
                raise SystemExit("Checkpoint download is incomplete")
        (weights / ".worlds-mirror-checkpoint").write_text(
            MANIFEST["checkpointRevision"] + "\n"
        )
    print("Pinned reconstruction artifacts prepared. No inference was run.")


if __name__ == "__main__":
    main()

"""Fetch a reviewed source revision, patch unsafe output naming, optionally fetch weights.

Never invoked by the gateway. Run explicitly during image building/setup.
"""
import argparse
import json
from pathlib import Path
import subprocess

MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text())


def prepare_source(destination: Path):
    if destination.exists():
        raise SystemExit("Source destination exists; use a new directory to preserve local changes")
    subprocess.run(["git", "init", str(destination)], check=True)
    subprocess.run(["git", "remote", "add", "origin", MANIFEST["repository"]], cwd=destination, check=True)
    subprocess.run(["git", "fetch", "--depth", "1", "origin", MANIFEST["commit"]], cwd=destination, check=True)
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination, check=True)
    sample = destination / "inference/sample.py"
    source = sample.read_text()
    # Upstream derives filenames from raw prompts. Anonymous, fixed filenames prevent
    # prompt path traversal, filesystem disclosure and accidental prompt metadata.
    original = "f'{prompt[:100]}{traj_suffix}.mp4'"
    if source.count(original) != 1:
        raise SystemExit("Pinned upstream output naming changed; review required")
    sample.write_text(source.replace(original, "'chunk.mp4'"))
    (destination / ".hexapod-reviewed-source").write_text(MANIFEST["commit"] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if args.source:
        prepare_source(args.source)
    if args.weights:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MANIFEST["checkpoint"], revision=MANIFEST["checkpointRevision"], local_dir=str(args.weights))
        (args.weights / ".hexapod-checkpoint-revision").write_text(MANIFEST["checkpointRevision"] + "\n")
    if not (args.source or args.weights):
        parser.error("select --source and/or --weights")


if __name__ == "__main__":
    main()

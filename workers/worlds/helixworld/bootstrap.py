"""Prepare pinned source or verify existing assets; never download weights here."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--verify-assets", action="store_true", help="Read and hash already prepared weights; does not download assets")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.exists():
        source.mkdir(parents=True)
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", MANIFEST["repository"]], check=True)
        subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", MANIFEST["commit"]], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--", "."], text=True)
    if head != MANIFEST["commit"] or dirty:
        raise SystemExit("The source must match the reviewed commit without tracked modifications")
    (source / ".worlds-helix-source").write_text(head + "\n")
    if args.verify_assets:
        subprocess.run([sys.executable, str(source / "download_models.py"), "--verify-only"], cwd=source, check=True)
    print(json.dumps({"sourceRevision": head, "weightsDownloaded": False, "gpuInferenceVerified": False, "assetsVerified": args.verify_assets}))


if __name__ == "__main__":
    main()

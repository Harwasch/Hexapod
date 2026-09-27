"""A stand-in for `block_views.py`. **It renders nothing.**

The real script renders the prior with and without each block on the GPU and writes
`1 - SSIM` per (frame, block). This one reads the same arguments and writes a scripted
matrix in the same format -- 0.5 for every third registered frame (by name) and every
block, 0 otherwise -- so a test can tell which cameras the stage assigned by the camera
test and which by position.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sfm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args, _unknown = parser.parse_known_args(argv)
    if not args.prior.is_file():
        sys.stderr.write(f"stand-in: no prior at {args.prior}\n")
        return 2
    blocks = len(json.loads(args.partition.read_text(encoding="utf-8"))["blocks"])
    names = [image.name for image in sfm.read_model(args.data_dir / "sparse" / "0").images]
    loss = [[0.5 if index % 3 == 0 else 0.0] * blocks for index in range(len(names))]
    args.out.write_text(json.dumps({"names": names, "loss": loss}), encoding="utf-8")
    sys.stdout.write(f"stand-in: {len(names)} frames x {blocks} blocks, scripted\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

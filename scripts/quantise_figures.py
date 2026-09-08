#!/opt/hw-py/bin/python
"""Palette-quantise the review figures before they are embedded.

The review page carries every figure as a base64 data URI, so a page over the
16 MiB Artifact cap simply cannot be published. These are flat matplotlib and
z-buffer renders: 256 colours is visually identical and roughly halves them.

Idempotent — re-quantising an already-quantised file changes nothing. Run from
scripts/build-review.sh, before review-artifact reads the files.
"""
import glob
import os
import sys

from PIL import Image

LIMIT_KB = 220          # review-artifact refuses to embed a single file above this


def main():
    before = after = 0
    biggest = []
    for f in sorted(glob.glob("docs/design/**/*.png", recursive=True)):
        b0 = os.path.getsize(f)
        im = Image.open(f)
        if im.mode == "P" and b0 // 1024 < LIMIT_KB:
            before += b0; after += b0
            continue                                    # already a palette image and small enough
        im.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT).save(f, optimize=True)
        b1 = os.path.getsize(f)
        before += b0; after += b1
        if b1 // 1024 >= LIMIT_KB:
            biggest.append((b1 // 1024, f))
    print(f"figures: {before//1024} -> {after//1024} kB")
    for kb, f in sorted(biggest, reverse=True):
        print(f"  STILL TOO BIG: {kb} kB  {f}", file=sys.stderr)
    return 1 if biggest else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Full-detail CPU Gaussian renders and hand-checkable semantic audit inputs.

The tiny scalar C++ raster kernel is compiled into the output cache, not installed.
Perspective covariance, 3-sigma support, front-to-back alpha compositing, DC colour.
It also retains dominant-contribution splat IDs; a projected centre is not visibility.
Round-3 camera headings/pitches/ranges are preserved. This is a diagnostic renderer,
not a reproduction of the production renderer's SH/antialiasing/LOD behaviour.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

KERNEL = r"""
#include <cmath>
#include <algorithm>
extern "C" void raster(int n, int w, int h, const float* p, const float* colours,
    const int* order, float* out, float* trans, float* best, int* ids, float* contribution) {
  for(int i=0;i<w*h;i++){trans[i]=1;best[i]=0;ids[i]=-1;}
  for(int j=0;j<n;j++) {
    int id=order[j]; const float* q=p+id*6;
    float det=q[2]*q[4]-q[3]*q[3]; if(det<=0)continue;
    int rx=std::ceil(3*std::sqrt(q[2])), ry=std::ceil(3*std::sqrt(q[4]));
    int x0=std::max(0,int(std::floor(q[0]))-rx), x1=std::min(w-1,int(std::ceil(q[0]))+rx);
    int y0=std::max(0,int(std::floor(q[1]))-ry), y1=std::min(h-1,int(std::ceil(q[1]))+ry);
    float sum=0;
    for(int y=y0;y<=y1;y++)for(int x=x0;x<=x1;x++){
      int k=y*w+x; if(trans[k]<.001f)continue;
      float dx=x+.5f-q[0],dy=y+.5f-q[1];
      float d=(q[4]*dx*dx-2*q[3]*dx*dy+q[2]*dy*dy)/det;
      if(d>9)continue;
      float a=std::min(.99f,q[5]*std::exp(-.5f*d));if(a<1.f/255)continue;
      float weight=trans[k]*a;sum+=weight;
      for(int c=0;c<9;c++)out[k*9+c]+=weight*colours[id*9+c];
      if(weight>best[k]){best[k]=weight;ids[k]=id;}
      trans[k]*=1-a;
    }
    contribution[id]=sum;
  }
}
"""


def raster_python(n, w, h, p, colours, order, out, trans, best, ids, contribution):
    """Same scalar kernel; cloud Debian slim uses Numba when g++ is unavailable."""
    trans[:] = 1
    best[:] = 0
    ids[:] = -1
    for j in range(n):
        index = order[j]
        q = p[index]
        det = q[2] * q[4] - q[3] * q[3]
        if det <= 0:
            continue
        rx, ry = math.ceil(3 * math.sqrt(q[2])), math.ceil(3 * math.sqrt(q[4]))
        x0, x1 = max(0, math.floor(q[0]) - rx), min(w - 1, math.ceil(q[0]) + rx)
        y0, y1 = max(0, math.floor(q[1]) - ry), min(h - 1, math.ceil(q[1]) + ry)
        total = 0.0
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if trans[y, x] < 0.001:
                    continue
                dx, dy = x + 0.5 - q[0], y + 0.5 - q[1]
                d = (q[4] * dx * dx - 2 * q[3] * dx * dy + q[2] * dy * dy) / det
                if d > 9:
                    continue
                a = min(0.99, q[5] * math.exp(-0.5 * d))
                if a < 1 / 255:
                    continue
                weight = trans[y, x] * a
                total += weight
                for c in range(9):
                    out[y, x, c] += weight * colours[index, c]
                if weight > best[y, x]:
                    best[y, x] = weight
                    ids[y, x] = index
                trans[y, x] *= 1 - a
        contribution[index] = total


def camera(target, heading=35, pitch=-12, distance=7.8):
    heading, pitch = np.deg2rad([heading, pitch])
    forward = np.array(
        [np.sin(heading) * np.cos(pitch), np.cos(heading) * np.cos(pitch), np.sin(pitch)]
    )
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.asarray(target) - forward * distance, np.stack([right, -up, forward])


def project(data, target, width, height, heading=35, distance=7.8):
    eye, view = camera(target, heading, -12, distance)
    xyz = (data["points"] - eye) @ view.T
    # Cesium's 60 degree horizontal field of view at landscape aspect ratios.
    focal = width / (2 * np.tan(np.pi / 6))
    depth = np.maximum(xyz[:, 2], 0.01)
    uv = xyz[:, :2] * (focal / depth[:, None]) + [width / 2, height / 2]
    r = Rotation.from_quat(data["quat"]).as_matrix()
    r = view[None] @ r
    r *= data["scales"][:, None, :]
    j = np.zeros((len(xyz), 2, 3))
    j[:, 0, 0] = focal / depth
    j[:, 1, 1] = focal / depth
    j[:, :, 2] = -xyz[:, :2] * focal / depth[:, None] ** 2
    jr = j @ r
    cov = jr @ jr.transpose(0, 2, 1)
    cov[:, 0, 0] += 0.3
    cov[:, 1, 1] += 0.3
    packed = np.column_stack([uv, cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1], data["alpha"]]).astype(
        np.float32
    )
    rx, ry = 3 * np.sqrt(cov[:, 0, 0]), 3 * np.sqrt(cov[:, 1, 1])
    valid = (
        (xyz[:, 2] > 0.05)
        & (uv[:, 0] + rx >= 0)
        & (uv[:, 0] - rx < width)
        & (uv[:, 1] + ry >= 0)
        & (uv[:, 1] - ry < height)
    )
    order = np.flatnonzero(valid)
    return packed, order[np.argsort(xyz[order, 2])].astype(np.int32)


def palette(labels):
    rng = np.random.default_rng(481)
    colours = rng.uniform(0.1, 1, (int(labels.max()) + 1, 3))
    return colours[labels]


def render(
    data, labels, wood, out: Path, views=("tree", "crown", "crown-back"), width=1024, height=640
):
    out.mkdir(parents=True, exist_ok=True)
    cpp, lib = out / "raster.cpp", out / "raster.so"
    cpp.write_text(KERNEL)
    if shutil.which("g++"):
        subprocess.run(["g++", "-O3", "-shared", "-fPIC", str(cpp), "-o", str(lib)], check=True)
        kernel = ctypes.CDLL(str(lib)).raster
        fp = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        kernel.argtypes = [ctypes.c_int] * 3 + [fp, fp, ip, fp, fp, fp, ip, fp]
        kernel.restype = None
    else:
        from numba import njit  # type: ignore[import-not-found]  # Cloud bootstrap dependency.

        kernel = njit(raster_python)
    semantic = np.where(wood[:, None], [0.85, 0.25, 0.12], [0.15, 0.75, 0.2])
    colours = np.column_stack([data["rgb"], palette(labels), semantic]).astype(np.float32)
    stats = {}
    for name in views:
        target = (
            [0, 0, 4.6] if "crown" in name else (data["points"].min(0) + data["points"].max(0)) / 2
        )
        heading = float(name.split("-h")[1]) if "-h" in name else 215 if "back" in name else 35
        distance = 7.8 if "crown" in name else 15.4
        p, order = project(data, target, width, height, heading, distance)
        output = np.zeros((height, width, 9), dtype=np.float32)
        trans = np.ones((height, width), dtype=np.float32)
        best = np.zeros_like(trans)
        ids = np.full((height, width), -1, dtype=np.int32)
        contribution = np.zeros(len(labels), dtype=np.float32)
        kernel(len(order), width, height, p, colours, order, output, trans, best, ids, contribution)
        for i, arm in enumerate(("rgb", "pieces", "wood")):
            background = np.array([159, 179, 200]) / 255
            im = output[:, :, i * 3 : i * 3 + 3] + trans[:, :, None] * background
            Image.fromarray(np.rint(np.clip(im, 0, 1) * 255).astype(np.uint8)).save(
                out / f"{name}-{arm}.png"
            )
        visible = (ids >= 0) & (best >= 0.05)
        _, pixel_count = np.unique(labels[ids[visible]], return_counts=True)
        stats[name] = {
            "dominantVisiblePieces": len(pixel_count),
            "piecesAtLeast16Pixels": int(np.sum(pixel_count >= 16)),
            "piecesAtLeast64Pixels": int(np.sum(pixel_count >= 64)),
            "pixels": int(visible.sum()),
            "caveat": "Contribution visibility only, not human-confirmed biological separability",
        }
        np.savez(out / f"{name}-contributions.npz", ids=ids, best=best, contribution=contribution)
        print(name, json.dumps(stats[name]), flush=True)
    (out / "visibility.json").write_text(json.dumps(stats, indent=2) + "\n")


def audit_sheet(data, derived, out):
    """Seeded balanced sample of visible predicted classes, not cherry-picked successes."""
    rng = np.random.default_rng(81)
    records, patches = [], []
    for view in ("tree", "crown", "crown-back"):
        image = Image.open(out / f"{view}-rgb.png").convert("RGB")
        contrib = np.load(out / f"{view}-contributions.npz")
        ids, best = contrib["ids"], contrib["best"]
        visible = (ids >= 0) & (best >= 0.1)
        candidates = np.unique(ids[visible])
        for wood in (True, False):
            own = candidates[derived["wood"][candidates] == wood]
            chosen = rng.choice(own, min(12, len(own)), replace=False)
            for splat in chosen:
                yy, xx = np.where((ids == splat) & visible)
                k = np.argmax(best[yy, xx])
                x, y = int(xx[k]), int(yy[k])
                index = len(records)
                patch = image.crop((x - 40, y - 40, x + 40, y + 40)).resize((160, 160))
                draw = ImageDraw.Draw(patch)
                draw.line((72, 80, 78, 80), fill="magenta", width=1)
                draw.line((82, 80, 88, 80), fill="magenta", width=1)
                draw.line((80, 72, 80, 78), fill="magenta", width=1)
                draw.line((80, 82, 80, 88), fill="magenta", width=1)
                draw.text(
                    (3, 3), f"{index:02} {view}", fill="white", stroke_width=1, stroke_fill="black"
                )
                patches.append(patch)
                records.append(
                    {
                        "id": index,
                        "view": view,
                        "splat": int(splat),
                        "pixel": [x, y],
                        "predicted": "wood" if wood else "foliage",
                        "manual": "unlabelled",
                    }
                )
    for page in range((len(patches) + 23) // 24):
        sheet = Image.new("RGB", (960, 640))
        for i, patch in enumerate(patches[page * 24 : page * 24 + 24]):
            sheet.paste(patch, ((i % 6) * 160, (i // 6) * 160))
        sheet.save(out / f"audit-{page}.png")
    (out / "audit.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    data = dict(np.load(args.probe / "scan.npz"))
    derived = dict(np.load(args.probe / "derived.npz"))
    labels = np.zeros(len(data["points"]), dtype=np.int32)
    labels[derived["crown"]] = np.load(args.probe / "labels-r0.025-c0.12-n0.8-seed17.npy") + 1
    render(data, labels, derived["wood"], args.out)
    audit_sheet(data, derived, args.out)

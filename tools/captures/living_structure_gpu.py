"""Optional budget-bounded SAM 2 tiny probe; launched ONLY through ops cloud-run.

Entry --bootstrap installs a disposable environment under /tmp. The git snapshot
contains code only: all scan bytes and open model weights are fetched publicly.
This uses dominant-alpha-contribution voting, not FlashSplat's full optimization.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

SAM_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
BASE = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/sites/minnetonka-tree/"
CHECKPOINT = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
EXPECTED_SCAN_SHA256 = "a5416c3768ab7b88b6f6df2ff2d108633851055a0208a499d44a5ef3832352c0"


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        path.write_bytes(response.read())


def bootstrap():
    """Pinned torch/CUDA wheel and source revision; no credentials or local data."""
    import tarfile

    env = Path("/tmp/structure-env")
    subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    pip = [str(env / "bin/python"), "-m", "pip", "install", "--no-cache-dir"]
    subprocess.run(
        [
            *pip,
            "torch==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
    )
    subprocess.run(
        [
            *pip,
            "numpy==2.2.6",
            "scipy==1.15.3",
            "pillow==11.3.0",
            "numba==0.61.2",
            "setuptools",
            "wheel",
        ],
        check=True,
    )
    archive = Path("/tmp/sam2.tar.gz")
    fetch(f"https://github.com/facebookresearch/sam2/archive/{SAM_REVISION}.tar.gz", archive)
    with tarfile.open(archive) as source:
        source.extractall("/tmp", filter="data")
    install_env = dict(os.environ, SAM2_BUILD_CUDA="0")
    subprocess.run(
        [*pip, "--no-build-isolation", f"/tmp/sam2-{SAM_REVISION}"], env=install_env, check=True
    )
    os.execv(str(env / "bin/python"), [str(env / "bin/python"), __file__])


def lift_masks(masks, ids, weight, count):
    """Smallest confident mask owns each pixel; vote by dominant visible alpha.

    Keep a Gaussian only if >=70% of its dominant contribution votes agree and its
    accumulated vote exceeds half a pixel. Hidden splats remain unassigned.
    """
    import numpy as np

    pixel = np.full(ids.shape, -1, dtype=np.int32)
    for i in sorted(range(len(masks)), key=lambda i: masks[i]["area"], reverse=True):
        pixel[masks[i]["segmentation"]] = i
    visible = (ids >= 0) & (weight >= 0.02)
    total = np.bincount(ids[visible], weights=weight[visible], minlength=count)
    use = visible & (pixel >= 0)
    keys = ids[use].astype(np.int64) * (len(masks) + 1) + pixel[use]
    unique, inverse = np.unique(keys, return_inverse=True)
    votes = np.bincount(inverse, weights=weight[use])
    splat, mask = unique // (len(masks) + 1), unique % (len(masks) + 1)
    # Sort ascending weight; later assignments overwrite with each splat's maximum.
    order = np.argsort(votes)
    chosen = np.full(count, -1, dtype=np.int32)
    valid = (votes >= 0.5) & (votes >= 0.7 * total[splat])
    order = order[valid[order]]
    chosen[splat[order]] = mask[order]
    return chosen


def merge_views(labels, minimum_iou=0.3):
    """Overlap graph of masks; no position-based filling of unseen crown splats."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    shifted, offset = [], 0
    for labels_one in labels:
        own = labels_one.copy()
        own[own >= 0] += offset
        offset = max(offset, int(own.max()) + 1)
        shifted.append(own)
    rows, cols = [], []
    for i, a in enumerate(shifted):
        ca = np.bincount(a[a >= 0], minlength=offset)
        for b in shifted[i + 1 :]:
            cb = np.bincount(b[b >= 0], minlength=offset)
            use = (a >= 0) & (b >= 0)
            pairs, counts = np.unique(np.column_stack([a[use], b[use]]), axis=0, return_counts=True)
            if not len(pairs):
                continue
            iou = counts / (ca[pairs[:, 0]] + cb[pairs[:, 1]] - counts)
            accept = (counts >= 8) & (iou >= minimum_iou)
            rows.extend(pairs[accept, 0])
            cols.extend(pairs[accept, 1])
    if offset == 0:
        return np.full(len(labels[0]), -1, dtype=np.int32)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(offset, offset))
    components = connected_components(graph, directed=False)[1]
    result = np.full(len(labels[0]), -1, dtype=np.int32)
    conflict = np.zeros(len(result), dtype=bool)
    for own in shifted:
        use = own >= 0
        new = components[own[use]]
        conflict[use] |= (result[use] >= 0) & (result[use] != new)
        result[use] = new
    result[conflict] = -1
    return result


def run():
    import numpy as np
    import torch  # type: ignore[import-not-found]  # Installed by bootstrap, never needed locally.
    from PIL import Image
    from sam2.automatic_mask_generator import (  # type: ignore[import-not-found]
        SAM2AutomaticMaskGenerator,
    )
    from sam2.build_sam import build_sam2  # type: ignore[import-not-found]

    from living_structure import adjusted_rand, describe, leaf_uris, load_scan, nearest_segments
    from living_structure_render import render

    out = Path(os.environ["CLOUD_OUT"])
    scratch = Path("/tmp/structure-data")
    source = scratch / "source"
    fetch(BASE + "splat/tileset.json", source / "splat/tileset.json")
    doc = json.loads((source / "splat/tileset.json").read_text())
    for uri in leaf_uris(doc):
        fetch(BASE + "splat/" + uri, source / "splat" / uri)
    fetch(BASE + "source/rig.json", source / "source/rig.json")
    scratch.mkdir(parents=True, exist_ok=True)
    data = load_scan(source, scratch)
    (out / "source.json").write_bytes((scratch / "source.json").read_bytes())
    if (
        json.loads((out / "source.json").read_text())["sha256LeafUriAndBytes"]
        != EXPECTED_SCAN_SHA256
    ):
        raise RuntimeError("Public scan differs from the CPU probe; do not compare these runs")
    nodes = json.loads((source / "source/rig.json").read_text())["nodes"]
    distance, _ = nearest_segments(data["points"], nodes)
    crown = data["points"][:, 2] >= 2
    views = tuple(f"crown-h{heading}" for heading in (35, 95, 155, 215, 275, 335))
    render(
        data,
        np.zeros(len(crown), dtype=np.int32),
        np.zeros(len(crown), dtype=bool),
        out / "renders",
        views,
    )
    checkpoint = scratch / "sam2.1_hiera_tiny.pt"
    fetch(CHECKPOINT, checkpoint)
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(checkpoint), device="cuda")
    summary = {
        "method": "SAM 2.1 tiny + dominant-alpha votes, six crown azimuths",
        "samRevision": SAM_REVISION,
        "seeds": {},
        "scanSplats": len(crown),
        "crownSplats": int(crown.sum()),
    }
    combined = []
    for seed in (17, 29):
        rng = np.random.default_rng(seed)
        grid = (np.indices((32, 32)).reshape(2, -1).T + rng.uniform(0.2, 0.8, (1024, 2))) / 32
        generator = SAM2AutomaticMaskGenerator(
            model,
            point_grids=[grid],
            points_per_side=None,
            points_per_batch=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            crop_n_layers=0,
            min_mask_region_area=0,
        )
        labels, counts = [], []
        for view in views:
            image = np.asarray(Image.open(out / "renders" / f"{view}-rgb.png"))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                masks = generator.generate(image)
            # 16 to 5,000 px: leaf/cluster scale; whole crown masks are excluded.
            masks = [m for m in masks if 16 <= m["area"] <= 5000]
            weights = np.load(out / "renders" / f"{view}-contributions.npz")
            own = lift_masks(masks, weights["ids"], weights["best"], len(crown))
            labels.append(own)
            counts.append(len(masks))
            np.savez_compressed(
                out / f"masks-{view}-{seed}.npz",
                masks=np.array([m["segmentation"] for m in masks]),
                labels=own,
            )
        result = merge_views(labels)
        np.save(out / f"labels-{seed}.npy", result)
        combined.append(result)
        assigned = crown & (result >= 0)
        summary["seeds"][str(seed)] = {
            "maskCounts": counts,
            "assignedCrownShare": float(assigned.sum() / crown.sum()),
            "pieces": describe(
                data["points"][assigned],
                data["scales"][assigned],
                result[assigned],
                distance[assigned],
            )
            if assigned.any()
            else None,
        }
        (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["allCrownARI"] = adjusted_rand(combined[0][crown], combined[1][crown])
    union = crown & ((combined[0] >= 0) | (combined[1] >= 0))
    summary["assignedUnionARI"] = adjusted_rand(combined[0][union], combined[1][union])
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    if args.bootstrap:
        bootstrap()
    run()

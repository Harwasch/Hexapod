"""`splat_render.GsplatRenderer` against `splat_render.render` on the committed yard: the
two renderers must agree on what a view covers and how far away it is, so `teacher_fill`'s
masks and lifted depths mean the same whichever drew the views. Runs where gsplat and a CUDA
GPU are (`infra/modal/fill.py --parity-test`); skipped elsewhere.

Not pixel equality. The CPU renderer point-samples each gaussian: speckle between samples
(its coverage has gaps gsplat fills, so gsplat covers more), and a gaussian that is large on
screen gets at most `max_samples` samples spread thin -- all but transparent. gsplat splats
every gaussian whole (EWA), as a viewer does. So the views are the fill's (`ring`, from
outside), where nothing is large on screen; coverage is compared as recall (what the CPU
covers, gsplat covers) and IoU; depth where both cover; and the fill mask, the decision
that matters, pixel by pixel. From inside the yard's canopy (the `near` views, a few metres
off) the two differ by design: leaves at arm's length fill gsplat's frame and are sparse
samples on the CPU; those views are reported, not asserted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import teacher_fill as tf
import view_cones as vc
from splat_render import GsplatRenderer, load_tileset

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")
if not torch.cuda.is_available():
    pytest.skip("gsplat needs a CUDA GPU", allow_module_level=True)

TILESET = Path(
    os.environ.get(
        "HEXAPOD_YARD_TILESET",
        Path(__file__).resolve().parents[3] / "data/tiles/synthetic-yard/splat/tileset.json",
    )
)
#: Of the pixels the CPU covers (alpha >= 0.5), the share gsplat covers too.
MIN_COVERAGE_RECALL = 0.9
#: Of the pixels either covers, the share both do (the CPU's speckle keeps this lower).
MIN_COVERAGE_IOU = 0.6
#: Where both cover: the median, and the 90th percentile, of |depth difference| / depth.
MAX_MEDIAN_DEPTH_ERROR = 0.02
MAX_P90_DEPTH_ERROR = 0.15
#: The teacher's fill mask (seen < 0.5 where covered >= 0.5): pixels where the two agree.
MIN_MASK_AGREEMENT = 0.97


def _stats(cpu, gpu) -> dict[str, float]:
    a, b = cpu.alpha >= 0.5, gpu.alpha >= 0.5
    both = a & b
    rel = np.abs(cpu.depth[both] - gpu.depth[both]) / np.maximum(gpu.depth[both], 1e-6)
    return {
        "coverageRecall": float(both.sum() / max(a.sum(), 1)),
        "coverageIoU": float(both.sum() / max((a | b).sum(), 1)),
        "coverageCpu": float(a.mean()),
        "coverageGsplat": float(b.mean()),
        "depthMedianRel": float(np.median(rel)) if rel.size else 0.0,
        "depthP90Rel": float(np.percentile(rel, 90)) if rel.size else 0.0,
    }


def _compare(splats, grid, cameras, gpu) -> list[dict]:
    out = []
    for camera in cameras:
        cpu_cond = tf.condition(splats, camera, grid)
        gpu_cond = tf.condition(splats, camera, grid, renderer=gpu)
        out.append(
            {
                "full": _stats(cpu_cond.full, gpu_cond.full),
                "seen": _stats(cpu_cond.seen, gpu_cond.seen),
                "maskAgreement": float((cpu_cond.mask == gpu_cond.mask).mean()),
                "maskCpu": float(cpu_cond.mask.mean()),
                "maskGsplat": float(gpu_cond.mask.mean()),
            }
        )
    return out


def test_gsplat_agrees_with_the_cpu_renderer_on_coverage_depth_and_masks() -> None:
    splats = load_tileset(TILESET)
    grid = vc.cone_grid_from_tileset(TILESET)
    texels = vc.lookup(grid.texels, grid.origin, grid.cell, grid.dims, splats.positions)
    faded = splats.positions[texels[:, 2] != vc.OMNI]
    reach = float(np.percentile(np.linalg.norm(splats.positions - faded.mean(axis=0), axis=1), 95))
    # The fill's views (fill_scan's ring), at its distance and a little nearer.
    ring = [
        camera
        for scale in (1.5, 1.2)
        for camera in tf.plan_views(
            grid, faded, count=4, mode="ring", ring_radius_m=scale * reach, width=320, height=180
        )
    ]
    observer = grid.observers[int(np.argmax(grid.observer_weights))]
    near = np.argsort(np.linalg.norm(splats.positions - observer, axis=1))[:2000]
    inside = tf.plan_views(
        grid, splats.positions[near].mean(axis=0)[None], count=2, width=320, height=180,
        distance_m=4.0,
    )  # fmt: skip
    gpu = GsplatRenderer()
    report = {
        "ring": _compare(splats, grid, ring, gpu),
        "insideCanopy": _compare(splats, grid, inside, gpu),
    }
    print(json.dumps(report, indent=1))  # read by fill.py --parity-test
    for view in report["ring"]:
        for which in ("full", "seen"):
            stats = view[which]
            if stats["coverageCpu"] < 1e-3:
                continue
            assert stats["coverageRecall"] >= MIN_COVERAGE_RECALL, (which, stats)
            assert stats["coverageIoU"] >= MIN_COVERAGE_IOU, (which, stats)
            assert stats["depthMedianRel"] <= MAX_MEDIAN_DEPTH_ERROR, (which, stats)
            assert stats["depthP90Rel"] <= MAX_P90_DEPTH_ERROR, (which, stats)
        assert view["maskAgreement"] >= MIN_MASK_AGREEMENT, view

"""`splat_render.GsplatRenderer` against `splat_render.render` on the committed yard: the
two renderers must agree on what a view covers and how far away it is, so `teacher_fill`'s
masks and lifted depths mean the same whichever drew the views. Runs where gsplat and a CUDA
GPU are (`infra/modal/fill.py --parity-test`); skipped elsewhere.

Not pixel equality: the CPU renderer point-samples each gaussian (grainy edges, speckle on
thin structure), gsplat splats it (EWA, smooth edges), so coverage is compared where it is
decided (alpha >= 0.5) and depth where both cover.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import teacher_fill as tf
import view_cones as vc
from splat_render import GsplatRenderer, load_tileset, render

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
#: Of the pixels either renderer covers (alpha >= 0.5), the share both do.
MIN_COVERAGE_IOU = 0.85
#: Where both cover: the median, and the 90th percentile, of |depth difference| / depth.
MAX_MEDIAN_DEPTH_ERROR = 0.02
MAX_P90_DEPTH_ERROR = 0.10
#: The teacher's fill mask (seen < 0.5 where covered >= 0.5): the share of pixels where the
#: two renderers' masks agree.
MIN_MASK_AGREEMENT = 0.9


def _stats(cpu, gpu) -> dict[str, float]:
    a, b = cpu.alpha >= 0.5, gpu.alpha >= 0.5
    both = a & b
    rel = np.abs(cpu.depth[both] - gpu.depth[both]) / np.maximum(gpu.depth[both], 1e-6)
    return {
        "coverageIoU": float(both.sum() / max((a | b).sum(), 1)),
        "coverageCpu": float(a.mean()),
        "coverageGsplat": float(b.mean()),
        "depthMedianRel": float(np.median(rel)) if rel.size else 0.0,
        "depthP90Rel": float(np.percentile(rel, 90)) if rel.size else 0.0,
    }


def test_gsplat_agrees_with_the_cpu_renderer_on_coverage_depth_and_masks() -> None:
    splats = load_tileset(TILESET)
    grid = vc.cone_grid_from_tileset(TILESET)
    observer = grid.observers[int(np.argmax(grid.observer_weights))]
    near = np.argsort(np.linalg.norm(splats.positions - observer, axis=1))[:2000]
    centre = splats.positions[near].mean(axis=0)
    texels = vc.lookup(grid.texels, grid.origin, grid.cell, grid.dims, splats.positions)
    faded = splats.positions[texels[:, 2] != vc.OMNI]
    reach = float(np.percentile(np.linalg.norm(splats.positions - faded.mean(axis=0), axis=1), 95))
    cameras = tf.plan_views(grid, centre[None], count=2, width=320, height=180, distance_m=4.0)
    cameras += tf.plan_views(
        grid, faded, count=2, mode="ring", ring_radius_m=1.5 * reach, width=320, height=180
    )
    gpu = GsplatRenderer()
    report = []
    for camera in cameras:
        full = _stats(render(splats, camera), gpu(splats, camera))
        cpu_cond = tf.condition(splats, camera, grid)
        gpu_cond = tf.condition(splats, camera, grid, renderer=gpu)
        seen = _stats(cpu_cond.seen, gpu_cond.seen)
        mask_agree = float((cpu_cond.mask == gpu_cond.mask).mean())
        report.append({"full": full, "seen": seen, "maskAgreement": mask_agree})
    print(json.dumps(report, indent=1))  # read by fill.py --parity-test
    for view in report:
        for which in ("full", "seen"):
            stats = view[which]
            if stats["coverageCpu"] + stats["coverageGsplat"] < 1e-3:
                continue
            assert stats["coverageIoU"] >= MIN_COVERAGE_IOU, (which, stats)
            assert stats["depthMedianRel"] <= MAX_MEDIAN_DEPTH_ERROR, (which, stats)
            assert stats["depthP90Rel"] <= MAX_P90_DEPTH_ERROR, (which, stats)
        assert view["maskAgreement"] >= MIN_MASK_AGREEMENT, view

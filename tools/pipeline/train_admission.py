"""Shared worker/container admission: buy capacity before partitioning a scan.

The worker reads image headers and the sparse model, without assembling a second dataset.
The container repeats the calculation on its assembled dataset. Both use the parser's
reference intrinsics ratio and largest registered render view.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import gaussian_budget as gb
import sfm
import support_mask
import training

TIERS = ("l4", "l40s", "a100")
DEFAULT_MAX_TIER = "l40s"
PLAN_PARAM = "_train_admission"


def frame_geometry(
    frames: Path, poses: Path, data_factor: int = 1, max_side: int | None = None
) -> tuple[tuple[int, int] | None, float | None]:
    """Predict build_dataset's resize and the pinned gsplat parser's intrinsics ratio."""
    from PIL import Image

    try:
        model = sfm.read_model(poses)
        cameras = {camera.id: camera for camera in model.cameras}
        for view in model.images:
            path = frames / view.name
            camera = cameras.get(view.camera_id)
            if not path.is_file() or camera is None or camera.width <= 0:
                continue
            with Image.open(path) as image:
                width, height = image.size
            if max_side is not None and max(width, height) > max_side:
                scale = max_side / max(width, height)
                width, height = max(1, round(width * scale)), max(1, round(height * scale))
            factor = max(1, data_factor)
            size = (max(1, round(width / factor)), max(1, round(height / factor)))
            pixel_scale = size[0] / camera.width
            used = {image.camera_id for image in model.images}
            sizes = [size] + [
                (max(1, round(c.width * pixel_scale)), max(1, round(c.height * pixel_scale)))
                for c in model.cameras
                if c.id in used and c.width > 0 and c.height > 0
            ]
            return max(sizes, key=lambda value: value[0] * value[1]), pixel_scale
    except (OSError, ValueError, KeyError, IndexError, struct.error):
        pass
    return None, None


def allowed_tiers(base: str, maximum: str) -> tuple[str, ...]:
    if base not in TIERS and maximum == base and base in gb.GPU_MEMORY_GB:
        return (base,)
    if maximum not in TIERS:
        raise ValueError(f"max_gpu_tier must be one of {TIERS}, not {maximum!r}")
    if base not in TIERS:
        return (base,)
    if TIERS.index(maximum) < TIERS.index(base):
        raise ValueError(f"max_gpu_tier {maximum!r} is below the recipe tier {base!r}")
    return TIERS[TIERS.index(base) : TIERS.index(maximum) + 1]


def admit(
    budget: gb.Budget,
    base: str,
    maximum: str,
    *,
    images_per_step: int = 1,
    memory_override: object = None,
) -> gb.Budget:
    """Keep the density target; clamp only per-call capacity for blocks at the max tier."""
    tiers = allowed_tiers(base, maximum)
    pixels = 0 if budget.train_size is None else budget.train_size[0] * budget.train_size[1]
    target = budget.raw if budget.raw is not None else budget.cap
    target = max(budget.floor, target) if budget.mode == "auto" else target
    if budget.budget_max is not None and budget.mode == "auto":
        target = min(target, budget.budget_max)
    chosen = base
    reason = "training image size unknown; retaining recipe tier"
    ceiling = budget.memory_ceiling
    if pixels:
        for chosen in tiers:
            memory = gb.gpu_memory_for(chosen)
            if memory_override is not None:
                memory = min(memory, float(str(memory_override)))
            ceiling = gb.memory_ceiling(pixels * images_per_step, memory)
            if target <= ceiling:
                reason = f"smallest allowed tier holding {target} gaussians (ceiling {ceiling})"
                break
        else:
            reason = f"target {target} exceeds max tier {chosen} ceiling {ceiling}; blocks required"
    memory = gb.gpu_memory_for(chosen)
    if memory_override is not None:
        memory = min(memory, float(str(memory_override)))
    cap, clamp = budget.cap, budget.clamp
    if budget.mode == "auto":
        cap, clamp = target, "budget-max" if target < (budget.raw or 0) else None
        if target == budget.floor and target > (budget.raw or 0):
            clamp = "floor"
        if ceiling is not None and cap > ceiling:
            cap, clamp = max(1, ceiling), "gpu-memory"
    return replace(
        budget,
        cap=cap,
        clamp=clamp,
        memory_ceiling=ceiling,
        gpu_memory_gb=memory,
        gpu_tier=chosen,
        recipe_gpu_tier=base,
        max_gpu_tier=maximum,
        tier_reason=reason,
    )


def plan(
    params: Mapping[str, Any],
    base: str | None,
    poses: Path,
    size: tuple[int, int] | None,
    pixel_scale: float | None,
    *,
    fixed_tier: bool = False,
) -> gb.Budget | None:
    """One policy used before submission and before building the trainer's argv."""
    placed = base or "l4"
    original = params.get(PLAN_PARAM, {})
    base = str(original.get("recipeTier", placed))
    maximum = str(params.get("max_gpu_tier", DEFAULT_MAX_TIER if base in TIERS[:2] else base))
    tiers = allowed_tiers(base, maximum)
    # Only the worker can acquire a bigger card. A container or LocalRunner is
    # already placed (bench runs use LocalRunner inside a fixed-tier GPU function).
    effective_max = placed if fixed_tier else maximum
    if fixed_tier:
        tiers = allowed_tiers(base, effective_max)
    mode = str(params.get("budget_mode", "density"))
    if mode not in ("density", "gpu-max"):
        raise ValueError(f"budget_mode must be density or gpu-max, not {mode!r}")
    mask = support_mask.SupportMask.parse(params.get("support_mask"))
    region = mask if mask is not None else training.Roi.parse(params.get("roi"))
    batch = training.check_batch_size(params.get("batch_size", 1))
    memory = gb.gpu_memory_for(tiers[-1])
    override = params.get("gpu_memory_gb")
    if override is not None:
        memory = min(memory, float(str(override)))
    budget = gb.plan(
        gb.parse_cap(params.get("cap_max")),
        model=poses,
        train_size=size,
        pixel_scale=pixel_scale,
        region_contains=None if region is None else lambda xyz: training.in_region(region, xyz),
        outside_weight=1.0 / training.ROI_OUTSIDE_EVERY,
        density=float(params.get("gaussian_density", gb.DEFAULT_DENSITY)),
        density_scale=float(params.get("density_scale", 1.0)),
        floor=int(params.get("budget_floor", gb.DEFAULT_FLOOR)),
        budget_max=None if params.get("budget_max") is None else int(params["budget_max"]),
        gpu_memory_gb=memory,
        images_per_step=batch,
        budget_mode=mode,
    )
    if budget is None:
        return None
    admitted = admit(budget, base, effective_max, images_per_step=batch, memory_override=override)
    return replace(admitted, max_gpu_tier=maximum)

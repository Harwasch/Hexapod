"""Refine an inferred layer against the views it was filled from, the measured scan frozen.

`teacher_fill.lift` places one flat disc per masked pixel per view: right where each view
put it, but the views disagree and nothing reconciles them. This optimises the lifted
gaussians (position, shape, colour, opacity) so that the measured scan **plus** them
renders, in every view, the filled pixels inside the mask and the measured scan's own
render outside it -- an inferred gaussian may not spoil what was measured. The measured
gaussians never move (they are not parameters at all).

Self-contained on purpose (numpy and torch, no other module of this package): it runs in
the `Distill` function of `infra/modal/world_models.py` with gsplat as the rasteriser,
and on a CPU with `torch_rasterize`, a small reference rasteriser good for tests.

Arrays, not `Splats`: a scan is a dict of `positions` (n,3), `rotations` (n,4 wxyz),
`scales` (n,3 linear), `colours` (n,3 in 0..1), `opacities` (n,). Views are a list of
`{"rotation", "centre", "focal", "width", "height"}` (as `splat_render.Camera.to_json`),
with `images` (v,h,w,3) uint8 and `masks` (v,h,w) bool.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence

import numpy as np

KEYS = ("positions", "rotations", "scales", "colours", "opacities")

#: Outside the mask, a view's target is the measured render, weighted this much: enough
#: that an inferred gaussian drifting in front of measured content is pushed back.
OUTSIDE_WEIGHT = 0.5
#: Pull towards where `lift` put each gaussian, per square metre of drift, so a disc fixes
#: its colour before it wanders off to fix someone else's.
ANCHOR_WEIGHT = 1.0


def pack(scan: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{k}": np.asarray(scan[k], np.float32) for k in KEYS}


def pack_views(cameras: Sequence[dict], images: np.ndarray, masks: np.ndarray) -> bytes:
    """Views as one npz: the request body of `Distill.run`."""
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        cameras=np.array(json.dumps(list(cameras))),
        images=np.asarray(images, np.uint8),
        masks=np.asarray(masks, bool),
    )
    return buffer.getvalue()


def unpack_views(blob: bytes) -> tuple[list[dict], np.ndarray, np.ndarray]:
    with np.load(io.BytesIO(blob)) as z:
        return json.loads(str(z["cameras"])), z["images"], z["masks"]


def pack_scan(scan: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **pack(scan))
    return buffer.getvalue()


def unpack_scan(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob)) as z:
        return {k: z[k].astype(np.float64) for k in KEYS}


Rasterize = Callable[..., tuple[object, object]]
"""`(means, quats, scales, opacities, colours, viewmat, K, width, height) -> (rgb (h,w,3),
alpha (h,w))`, torch tensors, differentiable in the gaussians."""


def gsplat_rasterize(means, quats, scales, opacities, colours, viewmat, K, width, height):
    """gsplat's `rasterization` for one camera (CUDA)."""
    from gsplat import rasterization

    rgb, alpha, _ = rasterization(
        means, quats, scales, opacities, colours, viewmat[None], K[None], width, height
    )
    return rgb[0], alpha[0, ..., 0]


def gsplat_frame(means, quats, scales, opacities, colours, viewmat, K, width, height, *, near, far):
    """gsplat's `rasterization` for one camera with its expected depth: `(rgb (h,w,3), alpha
    (h,w), depth (h,w))`, depth along the view axis normalised by coverage (what
    `splat_render.render` reports). `teacher_fill`'s views rendered on a GPU
    (`splat_render.GsplatRenderer`)."""
    from gsplat import rasterization

    out, alpha, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colours,
        viewmat[None],
        K[None],
        width,
        height,
        near_plane=near,
        far_plane=far,
        render_mode="RGB+ED",
    )
    return out[0, ..., :3], alpha[0, ..., 0], out[0, ..., 3]


def torch_rasterize(means, quats, scales, opacities, colours, viewmat, K, width, height):
    """A reference rasteriser in plain torch: each gaussian an isotropic splat of its mean
    scale, composited front to back. Dense in (n, h, w): for tests on tiny images only."""
    import torch

    del quats
    cam = means @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = cam[:, 2].clamp(min=1e-3)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    sigma = (K[0, 0] * scales.mean(dim=1) / z).clamp(min=0.3)
    order = torch.argsort(z)
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=means.dtype) + 0.5,
        torch.arange(width, dtype=means.dtype) + 0.5,
        indexing="ij",
    )
    d2 = (xs[None] - u[order, None, None]) ** 2 + (ys[None] - v[order, None, None]) ** 2
    a = opacities[order, None, None] * torch.exp(-0.5 * d2 / sigma[order, None, None] ** 2)
    a = a * (cam[order, 2, None, None] > 1e-3)
    a = a.clamp(max=0.99)
    transmit = torch.cumprod(torch.cat([torch.ones_like(a[:1]), 1 - a[:-1]]), dim=0)
    weight = a * transmit
    rgb = (weight[..., None] * colours[order, None, None, :]).sum(dim=0)
    return rgb, weight.sum(dim=0)


def camera_tensors(view: dict, torch, device):
    r = torch.tensor(view["rotation"], dtype=torch.float32, device=device)
    c = torch.tensor(view["centre"], dtype=torch.float32, device=device)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = r
    viewmat[:3, 3] = -r @ c
    f, w, h = float(view["focal"]), int(view["width"]), int(view["height"])
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=torch.float32, device=device)
    return viewmat, K, w, h


def distill(
    measured: dict[str, np.ndarray],
    init: dict[str, np.ndarray],
    cameras: Sequence[dict],
    images: np.ndarray,
    masks: np.ndarray,
    *,
    iterations: int = 1500,
    rasterize: Rasterize | None = None,
    device: str | None = None,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """The inferred gaussians after `iterations` steps of Adam, and a report: the masked
    and outside L1 per view before and after."""
    import torch

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rasterize = rasterize or (gsplat_rasterize if device == "cuda" else torch_rasterize)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    fixed = {k: t(measured[k]) for k in KEYS}
    anchor = t(init["positions"])
    params = {
        "positions": anchor.clone().requires_grad_(True),
        "rotations": t(init["rotations"]).requires_grad_(True),
        "log_scales": torch.log(t(init["scales"]).clamp(min=1e-6)).requires_grad_(True),
        "colour_logits": torch.logit(t(init["colours"]).clamp(1e-3, 1 - 1e-3)).requires_grad_(True),
        "opacity_logits": torch.logit(t(init["opacities"]).clamp(1e-3, 1 - 1e-3)).requires_grad_(
            True
        ),
    }
    extent = float(np.ptp(np.asarray(init["positions"]), axis=0).max()) if len(anchor) else 1.0
    optimiser = torch.optim.Adam(
        [
            {"params": [params["positions"]], "lr": 1e-3 * max(extent, 1e-3)},
            {"params": [params["rotations"]], "lr": 1e-3},
            {"params": [params["log_scales"]], "lr": 5e-3},
            {"params": [params["colour_logits"]], "lr": 2.5e-2},
            {"params": [params["opacity_logits"]], "lr": 2.5e-2},
        ]
    )
    views = [camera_tensors(v, torch, device) for v in cameras]
    filled = [t(im / 255.0) for im in np.asarray(images)]
    inside = [torch.tensor(np.asarray(m), dtype=torch.bool, device=device) for m in masks]

    def render(view, extra):
        viewmat, K, w, h = view
        means = torch.cat([fixed["positions"], extra["positions"]])
        quats = torch.cat([fixed["rotations"], extra["rotations"]])
        quats = quats / quats.norm(dim=1, keepdim=True).clamp(min=1e-12)
        scales = torch.cat([fixed["scales"], extra["scales"]])
        opacities = torch.cat([fixed["opacities"], extra["opacities"]])
        colours = torch.cat([fixed["colours"], extra["colours"]])
        return rasterize(means, quats, scales, opacities, colours, viewmat, K, w, h)

    def current() -> dict:
        return {
            "positions": params["positions"],
            "rotations": params["rotations"],
            "scales": params["log_scales"].exp(),
            "colours": torch.sigmoid(params["colour_logits"]),
            "opacities": torch.sigmoid(params["opacity_logits"]),
        }

    empty = {k: v[:0] for k, v in fixed.items()}
    with torch.no_grad():
        # Outside the mask a view should show what the measured scan alone shows.
        targets = []
        for k, view in enumerate(views):
            measured_rgb, _ = render(view, empty)
            m = inside[k][..., None]
            targets.append(torch.where(m, filled[k], measured_rgb))

    def losses(extra) -> list[tuple[float, float]]:
        out = []
        with torch.no_grad():
            for k, view in enumerate(views):
                rgb, _ = render(view, extra)
                err = (rgb - targets[k]).abs().mean(dim=-1)
                m = inside[k]
                out.append(
                    (
                        float(err[m].mean()) if m.any() else 0.0,
                        float(err[~m].mean()) if (~m).any() else 0.0,
                    )
                )
        return out

    before = losses(current())
    for step in range(iterations):
        k = step % len(views)
        rgb, _ = render(views[k], current())
        err = (rgb - targets[k]).abs().mean(dim=-1)
        m = inside[k]
        loss = err[m].mean() if m.any() else err.sum() * 0
        if (~m).any():
            loss = loss + OUTSIDE_WEIGHT * err[~m].mean()
        loss = loss + ANCHOR_WEIGHT * ((params["positions"] - anchor) ** 2).sum(dim=1).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    final = current()
    after = losses(final)
    out = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in final.items()}
    out["rotations"] /= np.maximum(np.linalg.norm(out["rotations"], axis=1, keepdims=True), 1e-12)
    report = {
        "iterations": iterations,
        "device": device,
        "rasteriser": getattr(rasterize, "__name__", "custom"),
        "gaussians": len(anchor),
        "views": [
            {
                "maskedL1Before": round(b[0], 5),
                "maskedL1After": round(a[0], 5),
                "outsideL1Before": round(b[1], 5),
                "outsideL1After": round(a[1], 5),
            }
            for b, a in zip(before, after, strict=True)
        ],
        "meanDriftM": round(
            float(np.linalg.norm(out["positions"] - np.asarray(init["positions"]), axis=1).mean())
            if len(anchor)
            else 0.0,
            6,
        ),
    }
    return out, report


def run(request: dict) -> dict:
    """The body of `Distill.run`: `{"measured": npz, "init": npz, "views": npz,
    "iterations"?}` -> `{"inferred": npz, "report": {...}}`."""
    cameras, images, masks = unpack_views(request["views"])
    out, report = distill(
        unpack_scan(request["measured"]),
        unpack_scan(request["init"]),
        cameras,
        images,
        masks,
        iterations=int(request.get("iterations", 1500)),
    )
    return {"inferred": pack_scan(out), "report": report}

"""`distill_fill`: the inferred layer is pulled to the filled pixels, the measured scan is
left as it was. On the CPU with the reference rasteriser (skipped without torch; the GPU
image has it, with gsplat)."""

from __future__ import annotations

import numpy as np
import pytest

import distill_fill as df
from splat_render import Camera


def _plane(xs: np.ndarray, ys: np.ndarray, colour: list[float], size: float) -> dict:
    gx, gy = np.meshgrid(xs, ys)
    n = gx.size
    return {
        "positions": np.column_stack([gx.ravel(), np.zeros(n), gy.ravel()]),
        "rotations": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "scales": np.full((n, 3), size),
        "colours": np.tile(colour, (n, 1)),
        "opacities": np.full(n, 0.9),
    }


def test_views_and_scans_round_trip() -> None:
    camera = Camera.look_at([0.0, -3.0, 0.0], [0.0, 0.0, 0.0], width=8, height=6)
    images = np.random.default_rng(0).integers(0, 255, (1, 6, 8, 3), dtype=np.uint8)
    masks = images[..., 0] > 128
    cameras, back, back_masks = df.unpack_views(df.pack_views([camera.to_json()], images, masks))
    assert cameras == [camera.to_json()]
    assert np.array_equal(back, images) and np.array_equal(back_masks, masks)
    scan = _plane(np.linspace(-1, 1, 3), np.linspace(-1, 1, 2), [0.1, 0.2, 0.3], 0.1)
    again = df.unpack_scan(df.pack_scan(scan))
    assert all(np.allclose(again[k], scan[k], atol=1e-6) for k in df.KEYS)


def test_distill_fills_the_mask_and_spares_the_measured() -> None:
    pytest.importorskip("torch")
    w, h = 32, 24
    measured = _plane(np.linspace(-1.0, -0.1, 10), np.linspace(-0.7, 0.7, 8), [0.8, 0.1, 0.1], 0.08)
    init = _plane(np.linspace(0.1, 1.0, 10), np.linspace(-0.7, 0.7, 8), [0.5, 0.5, 0.5], 0.08)
    cameras = [
        Camera.look_at([dx, -3.0, 0.2], [0.0, 0.0, 0.0], width=w, height=h).to_json()
        for dx in (-0.5, 0.5)
    ]
    images = np.zeros((2, h, w, 3), np.uint8)
    masks = np.zeros((2, h, w), bool)
    masks[:, 3:-3, w // 2 + 2 : -3] = True
    images[masks] = (40, 60, 220)  # the filler painted the unseen half blue
    out, report = df.distill(measured, init, cameras, images, masks, iterations=150)
    for view in report["views"]:
        assert view["maskedL1After"] < 0.5 * view["maskedL1Before"]
        assert view["outsideL1After"] <= view["outsideL1Before"] + 0.01
    assert out["colours"][:, 2].mean() > 0.6  # blue now
    assert report["meanDriftM"] < 0.1
    assert report["rasteriser"] == "torch_rasterize"


def test_view_weights_round_trip() -> None:
    camera = Camera.look_at([0.0, -3.0, 0.0], [0.0, 0.0, 0.0], width=8, height=6)
    images = np.zeros((2, 6, 8, 3), np.uint8)
    masks = np.ones((2, 6, 8), bool)
    views = [camera.to_json()] * 2
    blob = df.pack_views(views, images, masks, weights=[1.0, 0.2], outside=[0.0, 0.5])
    weights, outside = df.unpack_view_weights(blob)
    assert np.allclose(weights, [1.0, 0.2]) and np.allclose(outside, [0.0, 0.5])
    plain = df.pack_views([camera.to_json()], images[:1], masks[:1])
    assert df.unpack_view_weights(plain) == (None, None)


def test_distill_weighs_views_of_several_sizes() -> None:
    pytest.importorskip("torch")
    measured = _plane(np.linspace(-1.0, -0.1, 6), np.linspace(-0.7, 0.7, 5), [0.8, 0.1, 0.1], 0.1)
    init = _plane(np.linspace(0.1, 1.0, 6), np.linspace(-0.7, 0.7, 5), [0.5, 0.5, 0.5], 0.1)
    big = Camera.look_at([0.0, -3.0, 0.2], [0.0, 0.0, 0.0], width=24, height=16)
    small = Camera.look_at([0.3, -3.0, 0.2], [0.0, 0.0, 0.0], width=16, height=12)
    # Views of two sizes, padded to the larger (as `pack_views` carries them).
    images = np.zeros((2, 16, 24, 3), np.uint8)
    masks = np.zeros((2, 16, 24), bool)
    masks[0, 2:-2, 14:-2] = True
    masks[1, 2:10, 9:14] = True
    images[0][masks[0]] = (40, 60, 220)
    images[1][masks[1]] = (220, 220, 40)  # a low-weight view that disagrees
    out, report = df.distill(
        measured,
        init,
        [big.to_json(), small.to_json()],
        images,
        masks,
        iterations=60,
        weights=[1.0, 0.05],
        outside=[0.5, 0.0],
    )
    assert len(report["views"]) == 2
    assert out["colours"][:, 2].mean() > out["colours"][:, 0].mean() - 0.05  # the heavy view wins

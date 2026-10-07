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


def test_a_rigid_gaussian_keeps_its_shape_and_its_opacity_floor() -> None:
    pytest.importorskip("torch")
    w, h = 32, 24
    measured = _plane(np.linspace(-1.0, -0.1, 6), np.linspace(-0.7, 0.7, 5), [0.8, 0.1, 0.1], 0.1)
    init = _plane(np.linspace(0.1, 1.0, 6), np.linspace(-0.7, 0.7, 5), [0.5, 0.5, 0.5], 0.1)
    init["opacities"] = np.full(len(init["positions"]), 0.95)
    camera = Camera.look_at([0.0, -3.0, 0.2], [0.0, 0.0, 0.0], width=w, height=h)
    images = np.zeros((1, h, w, 3), np.uint8)
    masks = np.zeros((1, h, w), bool)
    masks[:, 3:-3, w // 2 + 2 : -3] = True  # the filler says: nothing there (black)
    rigid = np.zeros(len(init["positions"]), bool)
    rigid[::2] = True
    floor = np.where(rigid, 0.85, 0.0)
    back_rigid, back_floor = df.unpack_constraints(df.pack_constraints(rigid, floor))
    assert (back_rigid == rigid).all() and np.allclose(back_floor, floor)
    out, _ = df.distill(
        measured,
        init,
        [camera.to_json()],
        images,
        masks,
        iterations=80,
        rigid=back_rigid,
        opacity_floor=back_floor,
    )
    assert np.allclose(out["positions"][rigid], init["positions"][rigid], atol=1e-6)
    assert np.allclose(out["scales"][rigid], init["scales"][rigid], rtol=1e-4)
    assert (out["opacities"][rigid] >= 0.85 - 1e-6).all()
    # The free ones may fade (the view shows nothing there); the rigid ones cannot.
    assert out["opacities"][~rigid].min() < out["opacities"][rigid].min()


def _thin_layer(n_side: int = 6) -> tuple[dict, dict]:
    """A see-through inferred patch in the plane y = 0 (sparse, faint, needle-shaped and
    tilted), nothing measured near it, and a view of it."""
    rng = np.random.default_rng(3)
    measured = _plane(np.linspace(-1.0, -0.8, 2), np.linspace(-0.7, -0.5, 2), [0.8, 0.1, 0.1], 0.05)
    init = _plane(
        np.linspace(-0.4, 0.4, n_side), np.linspace(-0.4, 0.4, n_side), [0.5, 0.5, 0.5], 0.04
    )
    n = len(init["positions"])
    init["scales"] = np.column_stack([np.full(n, 0.06), np.full(n, 0.004), np.full(n, 0.004)])
    q = rng.normal(size=(n, 4))
    init["rotations"] = q / np.linalg.norm(q, axis=1, keepdims=True)
    init["opacities"] = np.full(n, 0.3)
    return measured, init


def test_solid_views_round_trip_and_the_terms_make_a_thin_patch_solid() -> None:
    torch = pytest.importorskip("torch")
    w, h = 16, 12
    measured, init = _thin_layer(5)
    view = Camera.look_at([0.0, -2.5, 0.0], [0.0, 0.0, 0.0], width=w, height=h)
    # The colour views agree with whatever is drawn: no colour pressure either way.
    images = np.full((1, h, w, 3), 128, np.uint8)
    masks = np.zeros((1, h, w), bool)
    # The fitted surface: the square y = 0, |x|, |z| <= 0.4, seen from the front and a side.
    solid_cams = [view, Camera.look_at([1.2, -2.2, 0.3], [0.0, 0.0, 0.0], width=w, height=h)]
    smasks, sdepths = [], []
    for cam in solid_cams:
        g = np.linspace(-0.3, 0.3, 30)
        gx, gz = np.meshgrid(g, g)
        uv, z = cam.project(np.column_stack([gx.ravel(), np.zeros(gx.size), gz.ravel()]))
        m = np.zeros((h, w), bool)
        d = np.zeros((h, w))
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = True
        d[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = z[ok]
        smasks.append(m)
        sdepths.append(d)
    normals = np.tile([0.0, 1.0, 0.0], (len(init["positions"]), 1))
    blob = df.pack_solid(
        [c.to_json() for c in solid_cams], np.array(smasks), np.array(sdepths), normals
    )
    solid = df.unpack_solid(blob)
    assert solid is not None and len(solid["cameras"]) == 2
    assert np.array_equal(solid["masks"], np.array(smasks)) and np.allclose(
        solid["normals"], normals
    )

    def coverage(out: dict) -> float:
        rast = df.torch_rasterize
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        viewmat, K, ww, hh = df.camera_tensors(view.to_json(), torch, "cpu")
        _, alpha = rast(
            t(out["positions"]),
            t(out["rotations"]),
            t(out["scales"]),
            t(out["opacities"]),
            t(out["colours"]),
            viewmat,
            K,
            ww,
            hh,
        )
        return float(alpha.numpy()[smasks[0]].mean())

    base, _ = df.distill(measured, init, [view.to_json()], images, masks, iterations=60)
    solid_out, report = df.distill(
        measured,
        init,
        [view.to_json()],
        images,
        masks,
        iterations=60,
        solidity=df.SOLIDITY_PRESETS["full"],
        solid=solid,
    )
    # Opaque where the surface is, as the terms ask; the needles widened; lying along it.
    assert coverage(solid_out) > coverage(base) + 0.2
    s = report["solidity"]
    assert set(s["terms"]) == set(df.SOLIDITY_TERMS) and s["solidViews"] == 2
    for term in ("alpha", "normal", "erank"):
        assert s["last"][term] < s["first"][term], (term, s)
    assert s["last"]["depth"] < 0.05  # the patch stays on the surface
    er_before = df.effective_rank(torch.tensor(init["scales"])).mean()
    er_after = df.effective_rank(torch.tensor(solid_out["scales"])).mean()
    assert er_after > er_before
    axis = df.thinnest_axis(torch.tensor(solid_out["rotations"]), torch.tensor(solid_out["scales"]))
    axis0 = df.thinnest_axis(torch.tensor(init["rotations"]), torch.tensor(init["scales"]))
    assert axis[:, 1].abs().mean() > axis0[:, 1].abs().mean()
    # Off, nothing changes: the same as no solidity at all.
    off, report_off = df.distill(
        measured, init, [view.to_json()], images, masks, iterations=60, solidity={}, solid=solid
    )
    assert "solidity" not in report_off
    assert np.allclose(off["opacities"], base["opacities"])

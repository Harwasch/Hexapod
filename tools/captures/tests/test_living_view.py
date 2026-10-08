"""`living_view`: plants from instances, the feathered mask, and motion only.

The flow tests build clips whose motion is known exactly -- a textured frame warped by a
camera creep (a translation) plus a sway inside a disc (the "plant") -- and check that DIS
finds it, that the creep is fitted from outside the disc and taken out, and that the render
moved by what is left starts and ends on the render and leaves everything outside the plant
where it was.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import living_view as lv

H, W = 120, 200


def _texture(seed: int = 0, h: int = H, w: int = W) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (h, w, 3)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.0)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min()) * 255
    return smooth.astype(np.uint8)


def _disc(h: int = H, w: int = W, radius: float = 30.0) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return (np.hypot(x - w / 2, y - h / 2) < radius).astype(np.float32)


def _clip(n: int, creep: tuple[float, float], sway: float) -> tuple[list[np.ndarray], np.ndarray]:
    """Frames `T(x + f_k(x))`: the camera creeping linearly to `creep` (px) and the disc
    swaying sideways by up to `sway` px on top. Returns the frames and the disc."""
    texture = _texture()
    disc = _disc()
    frames = [texture]
    for k in range(1, n):
        t = k / (n - 1)
        flow = np.zeros((H, W, 2), np.float32)
        flow[..., 0] = creep[0] * t + sway * math.sin(math.pi * t) * disc
        flow[..., 1] = creep[1] * t
        frames.append(lv.warp(texture, flow))
    return frames, disc


def test_plants_by_category_or_best_tag_and_inherited() -> None:
    instances = [
        {"id": 1, "parent": None, "category": "shrubs", "tags": [{"label": "bush"}]},
        {"id": 2, "parent": None, "category": "ground", "tags": [{"label": "bush"}]},
        {"id": 3, "parent": None, "category": "ground", "tags": [{"label": "dirt"}]},
        {"id": 4, "parent": None, "category": "trees", "tags": [{"label": "conifer"}]},
        {"id": 5, "parent": 4, "tags": []},  # nothing of its own: its parent's verdict
        {"id": 6, "parent": 3, "category": None, "tags": []},
        {"id": 7, "parent": None, "category": "household", "tags": [{"label": "fern"}]},
    ]
    plants = lv.plant_instances(instances)
    assert plants == {1: 1.0, 2: 1.0, 4: 0.5, 5: 0.5, 7: 1.0}
    flags = lv.plant_flags(np.array([0, 1, 2, 3, 4, 5, 6, 7, 99]), plants)
    assert flags.tolist() == [0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0]


def test_an_isolated_plant_is_everything_above_ground_but_its_stem_foot() -> None:
    p = np.array([[3.0, 0.0, 0.05], [3.0, 0.0, 2.0], [0.1, 0.0, 1.0], [0.1, 0.0, 2.5]])
    assert lv.isolated_plant_flags(p).tolist() == [0.0, 1.0, 0.0, 1.0]


def test_feathered_mask_is_soft_and_covers_the_plant() -> None:
    share = _disc()
    soft = lv.feather(share)
    assert soft.shape == share.shape and soft.min() >= 0.0 and soft.max() <= 1.0
    assert soft[H // 2, W // 2] == pytest.approx(1.0, abs=1e-3)
    assert soft[0, 0] == pytest.approx(0.0, abs=1e-3)
    edge = soft[(share == 0) & (soft > 0.01)]
    assert edge.size > 0 and edge.max() < 1.0  # it fades across the outline


def test_loop_window_starts_and_ends_at_zero() -> None:
    w = lv.loop_window(41, 0.25)
    assert w[0] == 0.0 and w[-1] == 0.0
    assert np.allclose(w, w[::-1])
    assert np.all(w[10:31] == 1.0)
    assert np.all(np.diff(w[:11]) > 0)


def test_dis_recovers_a_known_backward_flow() -> None:
    texture = _texture()
    flow = np.zeros((H, W, 2), np.float32)
    flow[..., 0], flow[..., 1] = 2.0, -1.0
    frame = lv.warp(texture, flow)
    found = lv.dis_flow(frame, texture)
    inner = (slice(15, H - 15), slice(15, W - 15))
    assert np.median(found[inner][..., 0]) == pytest.approx(2.0, abs=0.15)
    assert np.median(found[inner][..., 1]) == pytest.approx(-1.0, abs=0.15)


def test_camera_creep_is_fitted_outside_the_plant_and_removed() -> None:
    frames, disc = _clip(9, creep=(1.5, -1.0), sway=3.0)
    flow = lv.dis_flow(frames[4], frames[0])  # half way: creep (0.75, -0.5), sway 3.0
    background = lv.feather(disc) < lv.NOT_PLANT_SHARE
    H_fit, offered = lv.global_motion(flow, background)
    assert offered >= lv.MIN_BACKGROUND
    camera = lv.homography_flow(H_fit, H, W)
    assert np.median(camera[..., 0]) == pytest.approx(0.75, abs=0.15)
    assert np.median(camera[..., 1]) == pytest.approx(-0.5, abs=0.15)
    local = flow - camera
    core = _disc(radius=20.0) > 0
    assert np.median(local[core][:, 0]) == pytest.approx(3.0, abs=0.4)
    assert np.abs(np.median(local[background], axis=0)).max() < 0.15


def test_motion_only_moves_only_the_plant_and_returns_to_the_render() -> None:
    frames, disc = _clip(13, creep=(2.0, 1.0), sway=3.0)
    render = frames[0].copy()
    soft = lv.feather(disc)
    moved, report = lv.motion_only(render, frames, soft, ramp=0.25)
    moved = moved["bilinear"]
    assert len(moved) == len(frames)
    assert np.array_equal(moved[0], render)
    assert np.array_equal(moved[-1], render)  # eased back onto the render: it loops
    outside = soft < 1e-3
    middle = moved[6]
    assert np.abs(middle[outside].astype(int) - render[outside].astype(int)).max() <= 1
    assert np.abs(middle[disc > 0].astype(int) - render[disc > 0].astype(int)).mean() > 3
    # Before compensation the creep shows up outside the plant; the fitted camera says how much.
    assert report.outside_mean_px > 0.5
    assert report.camera_creep_px == pytest.approx(math.hypot(2.0, 1.0), abs=0.3)
    assert report.inside_p95_px > report.outside_p95_px


def test_motion_only_scales_the_flow_to_the_render() -> None:
    import cv2

    frames, disc = _clip(7, creep=(0.0, 0.0), sway=2.0)
    big_render = cv2.resize(frames[0], (2 * W, 2 * H), interpolation=cv2.INTER_CUBIC)
    soft = lv.feather(cv2.resize(disc, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST))
    moved, report = lv.motion_only(big_render, frames, soft, ramp=0.0)
    moved = moved["bilinear"]
    assert moved[3].shape == big_render.shape
    assert report.inside_p95_px == pytest.approx(4.0, abs=0.8)  # 2 px at the model's size


def test_psnr_outside_counts_only_what_is_not_a_plant() -> None:
    render = _texture()
    mask = _disc()
    changed = render.copy()
    changed[mask > 0] = 0  # the plant redrawn: does not count
    assert lv.psnr_outside([render, changed], render, mask) == math.inf
    noisy = changed.astype(int) + 8
    value = lv.psnr_outside([render, np.clip(noisy, 0, 255).astype(np.uint8)], render, mask)
    assert value == pytest.approx(10 * math.log10(255**2 / 64), abs=0.5)


def test_upsampled_flow_scales_its_vectors() -> None:
    flow = np.zeros((10, 20, 2), np.float32)
    flow[..., 0], flow[..., 1] = 1.0, 2.0
    big = lv.upsample_flow(flow, 40, 30)
    assert big.shape == (30, 40, 2)
    assert np.allclose(big[..., 0], 2.0) and np.allclose(big[..., 1], 6.0)

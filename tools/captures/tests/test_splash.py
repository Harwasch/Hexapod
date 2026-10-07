"""The splash arm: its LoRA's keys as the loader takes them, its picture 1 (the unknown,
seen-through and pocket pixels black, the reprojected surface's photo pixels kept), the pose
check that rejects a seed whose camera turned, and the gate's summary."""

from __future__ import annotations

import math

import numpy as np

import anchor_fill as af
import anchor_models as am
from splat_render import Camera


def test_the_loras_peft_keys_are_renamed_for_the_loader() -> None:
    raw = {
        "transformer_blocks.0.attn.add_k_proj.lora_A.default.weight": 1,
        "transformer_blocks.0.attn.add_k_proj.lora_B.default.weight": 2,
        "transformer.proj_out.lora_A.weight": 3,
    }
    assert am.splash_state_dict(raw) == {
        "transformer.transformer_blocks.0.attn.add_k_proj.lora_A.weight": 1,
        "transformer.transformer_blocks.0.attn.add_k_proj.lora_B.weight": 2,
        "transformer.proj_out.lora_A.weight": 3,
    }
    assert am.LICENCES[am.SPLASH_REPO] == "Apache-2.0"


def _texture(h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(0)
    import cv2

    noise = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)


def test_the_pose_check_measures_a_turn_of_the_camera() -> None:
    import cv2

    image = _texture()
    h, w = image.shape[:2]
    cam = Camera.look_at([0.0, -3.0, 0.0], [0.0, 0.0, 0.0], fov_deg=50, width=w, height=h)
    known = np.ones((h, w), bool)
    assert af.pose_offset_deg(image, image, known, cam) < 0.2
    # The same scene from the same place, turned 3 degrees about the vertical.
    f = cam.focal
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    a = math.radians(3.0)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    turned = cv2.warpPerspective(image, k @ rot @ np.linalg.inv(k), (w, h))
    inside = cv2.warpPerspective(np.ones((h, w), np.uint8), k @ rot @ np.linalg.inv(k), (w, h)) > 0
    angle = af.pose_offset_deg(turned, image, inside, cam)
    assert angle is not None and abs(angle - 3.0) < 0.5
    # Nothing to match: no verdict (the gate rejects it).
    assert af.pose_offset_deg(np.zeros_like(image), image, known, cam) is None


def test_picture_one_blacks_out_what_the_scan_does_not_know() -> None:
    h, w = 20, 30
    render = np.full((h, w, 3), 200, np.uint8)
    known = np.zeros((h, w), bool)
    known[:, :10] = True
    weak = np.zeros((h, w), bool)
    weak[:, 10:20] = True
    surface = np.zeros((h, w), bool)
    surface[:, 15:20] = True  # the reprojected surface's photo pixels, among the weak
    unknown = np.zeros((h, w), bool)
    unknown[:, 20:28] = True
    void = np.zeros((h, w), bool)
    void[:, 28:] = True
    cam = Camera.look_at([0.0, -3.0, 0.0], [0.0, 0.0, 0.0], width=w, height=h)
    masks = af.ViewMasks(
        cam,
        known,
        weak,
        unknown,
        void,
        render,
        render.copy(),
        np.where(unknown, 1.0, np.where(weak, 0.4, 0.0)).astype(np.float32),
        np.full((h, w), 3.0),
        surface=surface,
    )

    class _Run:
        pockets = None

    out, picture = af.splash_masks(_Run(), masks)  # type: ignore[arg-type]
    black = (picture == 0).all(axis=-1)
    assert black[:, 10:15].all() and black[:, 20:].all()  # seen through, unknown, void
    assert not black[:, :10].any() and not black[:, 15:20].any()  # known, the surface kept
    assert (out.unknown == (black & ~void)).all() and not out.weak.any()
    assert out.known[:, 15:20].all() and (out.strength[out.unknown] == 1.0).all()


def test_the_gate_summary_counts_each_verdict() -> None:
    gate = [
        {"view": "a0", "seed": 17, "lpipsOk": True, "poseOk": True, "kept": True},
        {"view": "a0", "seed": 1017, "lpipsOk": False, "poseOk": True, "kept": False},
        {"view": "a1", "seed": 17, "lpipsOk": True, "poseOk": False, "kept": False},
        {"view": "a1", "seed": 1017, "kept": False, "error": "timeout"},
    ]
    s = af.splash_summary(gate)
    assert (s["frames"], s["kept"], s["rejected"], s["rejectedShare"]) == (4, 1, 3, 0.75)
    assert (s["failedLpips"], s["failedPose"], s["errors"]) == (1, 1, 1)

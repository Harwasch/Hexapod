"""The splash arm: its LoRA's keys as the loader takes them, its picture 1 (the unknown,
seen-through and pocket pixels black, the reprojected surface's photo pixels kept), the pose
check that rejects a seed whose camera turned, and the gate's summary."""

from __future__ import annotations

import math

import numpy as np
import pytest

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


def test_the_lock_frees_tokens_to_make_and_feathers_the_seam() -> None:
    w, h = 8 * am.TOKEN_PX, 4 * am.TOKEN_PX
    strength = np.zeros((h, w), np.float32)
    strength[: am.TOKEN_PX, : am.TOKEN_PX] = 1.0  # one token's worth to make, top left
    strength[3 * am.TOKEN_PX + 3, 7 * am.TOKEN_PX + 5] = 0.4  # weak: still locked
    lock = am.token_lock(strength, w, h, feather=2).reshape(4, 8)
    assert lock[0, 0] == 0.0  # free
    assert 0.0 < lock[0, 1] < lock[0, 2] < 1.0  # the seam blends over two tokens
    assert lock[0, 3:].min() == 1.0 and lock[3, 7] == 1.0
    assert (am.token_lock(np.zeros((h, w)), w, h, 2) == 1.0).all()  # nothing to make


def test_the_lock_callback_ends_locked_tokens_as_rendered() -> None:
    torch = pytest.importorskip("torch")
    x0 = torch.full((1, 3, 2), 5.0)
    eps = torch.zeros((1, 3, 2))
    lock = np.array([1.0, 0.5, 0.0], np.float32)

    class _Scheduler:
        sigmas = torch.tensor([1.0, 0.5, 0.0])

    class _Pipe:
        scheduler = _Scheduler()

    callback = am.lock_callback(x0, eps, lock)
    made = torch.full((1, 3, 2), -1.0)
    last = callback(_Pipe(), 1, None, {"latents": made})["latents"]  # next sigma 0
    assert torch.allclose(last[0, 0], torch.tensor([5.0, 5.0]))  # locked: the render
    assert torch.allclose(last[0, 1], torch.tensor([2.0, 2.0]))  # feathered: half each
    assert torch.allclose(last[0, 2], torch.tensor([-1.0, -1.0]))  # free: what was made


def _void_masks(h: int = 120, w: int = 160) -> af.ViewMasks:
    render = np.full((h, w, 3), 180, np.uint8)
    alpha = np.ones((h, w), np.float32)
    alpha[40:70, 50:90] = 0.05  # a true hole
    alpha[100, 10] = 0.0  # a speck
    weak = np.zeros((h, w), bool)
    weak[20:30, 100:150] = True  # seen through but drawn: left for the model to repair
    void = np.zeros((h, w), bool)
    void[:, 155:] = True
    alpha[void] = 0.0
    unknown = (alpha < 0.3) & ~void
    known = ~weak & ~unknown & ~void
    cam = Camera.look_at([0.0, -3.0, 0.0], [0.0, 0.0, 0.0], width=w, height=h)
    return af.ViewMasks(
        cam,
        known,
        weak,
        unknown,
        void,
        render,
        render.copy(),
        np.where(unknown, 1.0, np.where(weak, 0.4, 0.0)).astype(np.float32),
        np.full((h, w), 3.0),
        alpha=alpha,
    )


def test_the_voids_picture_blacks_only_true_holes_and_makes_a_little_round_them() -> None:
    class _Run:
        pockets = None

    masks = _void_masks()
    arm = af.SPLASH_ARMS["splash-voids"]
    out, picture = af.splash_masks(_Run(), masks, arm)  # type: ignore[arg-type]
    black = (picture == 0).all(axis=-1)
    assert black[45:65, 55:85].all()  # the hole
    assert not black[100, 10]  # the speck is not a hole worth making
    assert not black[20:30, 100:150].any()  # the seen-through pixels stay visible
    assert black[:, 155:].all() and not out.unknown[:, 155:].any()  # the open background
    # What is made reaches past the hole (the smears beside it), and nothing else.
    assert out.unknown[38:72, 48:92].all() and not out.unknown[100:, :40].any()
    assert (out.known == (~out.unknown & ~masks.void)).all() and not out.weak.any()
    assert (out.strength[out.unknown] == 1.0).all() and (out.strength[out.known] == 0.0).all()


def test_only_the_locking_arms_send_a_lock() -> None:
    def request(arm: str) -> dict:
        spec = af.SPLASH_ARMS[arm]
        r = af.EditRequest(
            f"{arm}/a0-s17",
            np.zeros((32, 32, 3), np.uint8),
            np.zeros((32, 32, 3), np.uint8),
            np.zeros((32, 32), np.float32),
            [],
            af.SPLASH_PROMPT,
            17,
            af.SPLASH_STEPS,
            hold=False,
            splash=True,
            lock=spec.feather if spec.lock else None,
        )
        return r.wire()

    assert "lock" not in request("splash")
    assert request("splash-locked")["lock"] == 2 and request("splash-voids")["lock"] == 2
    assert af.LAYERS["splash-voids"] == "anchor-splash-voids"
    assert af.ANCHOR_ARM["splash-locked"] == "refs"


def test_the_asis_arm_generates_at_about_a_megapixel_with_the_four_step_adapter() -> None:
    assert am.generation_size([1024, 592], None) == (1024, 592)
    assert am.generation_size([1024, 592], 1024 * 1024) == (1344, 768)
    spec = af.SPLASH_ARMS["splash-asis"]
    assert (spec.picture, spec.steps, spec.adapter, spec.lock, spec.whole, spec.align) == (
        "render",
        10,
        4,
        False,
        True,
        "flow",
    )


def test_the_locked_arm_shows_the_plain_render_and_makes_what_the_scan_does_not_know() -> None:
    h, w = 20, 30
    render = np.full((h, w, 3), 200, np.uint8)
    drawn = np.full((h, w, 3), 120, np.uint8)  # as drawn over black: see-through is darker
    known = np.zeros((h, w), bool)
    known[:, :10] = True
    weak = np.zeros((h, w), bool)
    weak[:, 10:20] = True
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
        np.zeros((h, w), np.float32),
        np.full((h, w), 3.0),
        drawn=drawn,
    )

    class _Run:
        pockets = None

    out, picture = af.splash_masks(_Run(), masks, af.SPLASH_ARMS["splash-locked"])  # type: ignore[arg-type]
    assert (picture == drawn).all()  # nothing painted
    assert (out.unknown == ((weak | unknown) & ~void)).all()
    assert (out.known == known).all() and (out.strength[out.unknown] == 1.0).all()


def _turned(image: np.ndarray, deg: float, shift: float) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    m[0, 2] += shift
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def test_flow_alignment_puts_a_turned_output_back_on_the_render() -> None:
    render = _texture()
    known = np.ones(render.shape[:2], bool)
    known[90:150, 120:200] = False  # a hole: what the output makes
    out = _turned(render, 1.5, 4.0)
    hom = af.match_homography(out, render, known)
    aligned = af.flow_align(out, render, known, hom)
    assert aligned is not None
    inner = known.copy()
    inner[:12], inner[-12:], inner[:, :12], inner[:, -12:] = False, False, False, False

    def err(x: np.ndarray) -> float:
        return float(np.abs(x.astype(float) - render.astype(float)).mean(axis=-1)[inner].mean())

    assert err(aligned) < 0.35 * err(out)
    assert af.flow_align(out, render, known, None) is None


def test_the_asis_gate_reports_and_keeps_an_aligned_seed() -> None:
    render = _texture()
    h, w = render.shape[:2]
    known = np.ones((h, w), bool)
    known[90:150, 120:200] = False
    cam = Camera.look_at([0.0, -3.0, 0.0], [0.0, 0.0, 0.0], fov_deg=50, width=w, height=h)
    masks = af.ViewMasks(
        cam,
        known,
        np.zeros((h, w), bool),
        ~known,
        np.zeros((h, w), bool),
        render,
        render.copy(),
        np.where(known, 0.0, 1.0).astype(np.float32),
        np.full((h, w), 3.0),
    )
    near = _turned(render, 1.0, 3.0)
    far = np.zeros_like(render)  # nothing to match: no camera
    results = [
        af.EditResult("splash-asis/a0-s17", near),
        af.EditResult("splash-asis/a0-s1017", far),
    ]
    perceptual = af.StandInPerceptual()
    cands = af.score_candidates(masks, results, [[], []], perceptual)

    from types import SimpleNamespace

    run = SimpleNamespace(
        splash_gate=[], splash_frames={"splash-asis": {"a0": {"raw": {}}}}, perceptual=perceptual
    )
    kept = af.splash_gate(run, masks, "a0", results, cands, "splash-asis")  # type: ignore[arg-type]
    assert [c.seed for c in kept] == [17]
    verdicts = {g["seed"]: g for g in run.splash_gate}
    assert verdicts[17]["aligned"] and verdicts[17]["poseDeg"] < af.SPLASH_ALIGN_DEG
    assert verdicts[17]["alignedLpips"] <= verdicts[17]["knownLpips"]
    assert not verdicts[1017]["kept"] and verdicts[1017]["poseDeg"] is None
    assert 17 in run.splash_frames["splash-asis"]["a0"]["aligned"]
    s = af.splash_summary(run.splash_gate)
    assert (s["returned"], s["poseFound"], s["poseWithin"], s["aligned"], s["kept"]) == (
        2,
        1,
        1,
        1,
        1,
    )


def _region_masks(h: int = 120, w: int = 160) -> af.ViewMasks:
    render = np.full((h, w, 3), 150, np.uint8)
    render[30:80, 40:120] = (170, 150, 120)  # the measured top's tone
    surface = np.zeros((h, w), bool)
    surface[30:80, 40:120] = True  # the rebuilt top's footprint
    surface[50:55, 70:75] = False  # a gap in it
    weak = surface.copy()
    weak[80:84, 50:110] = True  # its soft rim, joined to it
    weak[100:110, 5:20] = True  # a soft patch far away
    known = ~weak
    void = np.zeros((h, w), bool)
    void[:, 150:] = True
    known &= ~void
    cam = Camera.look_at([0.0, -3.0, 0.0], [0.0, 0.0, 0.0], width=w, height=h)
    return af.ViewMasks(
        cam,
        known,
        weak,
        np.zeros((h, w), bool),
        void,
        render,
        render.copy(),
        np.where(weak, 0.4, 0.0).astype(np.float32),
        np.full((h, w), 3.0),
        surface=surface,
        drawn=render.copy(),
    )


def test_the_region_arm_makes_the_whole_top_as_one_region() -> None:
    class _Run:
        pockets = None

    masks = _region_masks()
    out, picture = af.splash_masks(_Run(), masks, af.SPLASH_ARMS["splash-region"])  # type: ignore[arg-type]
    assert (picture == masks.drawn).all()  # nothing painted
    assert out.unknown[30:80, 40:120].all()  # the top and the gap in it
    assert out.unknown[80:84, 50:110].all()  # its soft rim
    assert not out.unknown[100:110, 5:20].any()  # a soft patch away from the top stays
    assert not out.unknown[:, 150:].any() and not out.unknown[:20].any()
    assert (out.known == (~out.unknown & ~masks.void)).all()


def test_the_region_gate_drops_a_repaint_that_changes_the_woods_tone() -> None:
    class _Run:
        pockets = None

    masks, _ = af.splash_masks(_Run(), _region_masks(), af.SPLASH_ARMS["splash-region"])  # type: ignore[arg-type]
    same = masks.render.copy()
    assert af.colour_offset(same, masks) < 1.0
    darker = same.copy()
    darker[masks.unknown] = (110, 90, 60)
    assert af.colour_offset(darker, masks) > af.SPLASH_COLOUR_DE


def test_the_inward_lock_keeps_the_outline_locked() -> None:
    w, h = 8 * am.TOKEN_PX, 8 * am.TOKEN_PX
    strength = np.zeros((h, w), np.float32)
    strength[8 : 7 * am.TOKEN_PX + 8, 8 : 7 * am.TOKEN_PX + 8] = 1.0  # off the token grid
    lock = am.token_lock(strength, w, h, feather=1, inward=True).reshape(8, 8)
    assert lock[0].min() == 1.0 and lock[:, 0].min() == 1.0  # across the edge: locked
    assert lock[1, 3] == 0.5 and lock[3, 3] == 0.0  # one token inside: half; deeper: free
    assert (am.token_lock(np.zeros((h, w)), w, h, 1, inward=True) == 1.0).all()


def _footprint_masks(h: int = 160, w: int = 200) -> af.ViewMasks:
    import cv2

    render = np.full((h, w, 3), (90, 110, 60), np.uint8)  # ground
    top = np.zeros((h, w), np.uint8)
    cv2.ellipse(top, (100, 80), (70, 45), 0, 0, 360, 1, -1)
    surface = top > 0
    render[surface] = (190, 185, 175)  # the top's grey wood
    weak = surface.copy()
    weak[125:132, 40:160] = True  # a soft fringe below it: not part of the footprint
    cam = Camera.look_at([0.0, -3.0, 0.0], [0.0, 0.0, 0.0], width=w, height=h)
    return af.ViewMasks(
        cam,
        ~weak,
        weak,
        np.zeros((h, w), bool),
        np.zeros((h, w), bool),
        render,
        render.copy(),
        np.where(weak, 0.4, 0.0).astype(np.float32),
        np.full((h, w), 3.0),
        surface=surface,
        drawn=render.copy(),
    )


def test_the_footprint_arm_makes_exactly_the_top_and_gates_its_outline() -> None:
    import cv2

    class _Run:
        pockets = None

    base = _footprint_masks()
    masks, picture = af.splash_masks(_Run(), base, af.SPLASH_ARMS["splash-footprint"])  # type: ignore[arg-type]
    assert (masks.unknown == base.surface).all()  # the top only, no fringe
    perceptual = af.StandInPerceptual()
    # A repaint inside the outline: the same shape, its wood textured.
    same = picture.copy()
    rng = np.random.default_rng(1)
    noise = rng.integers(-12, 12, same.shape)
    same[masks.unknown] = np.clip(same.astype(int) + noise, 0, 255).astype(np.uint8)[masks.unknown]
    v = af.outline_verdict(same, masks, perceptual)
    assert v["outlineOk"] and v["knownOk"], v
    # The top drawn smaller: ground colour in a ring inside the scan's outline.
    small = picture.copy()
    smaller = np.zeros(small.shape[:2], np.uint8)
    cv2.ellipse(smaller, (100, 80), (60, 37), 0, 0, 360, 1, -1)
    small[masks.unknown & (smaller == 0)] = (90, 110, 60)
    v = af.outline_verdict(small, masks, perceptual)
    assert not v["outlineOk"] and v["outlineIou"] < af.OUTLINE_IOU, v


def test_the_homography_check_measures_scale_and_shift() -> None:
    hom = np.array([[1.03, 0.0, -2.0], [0.0, 1.03, 1.0], [0.0, 0.0, 1.0]])
    scale, shift = af.homography_scale_shift(hom, 200, 100)
    assert abs(scale - 1.03) < 1e-6 and shift > 0.01
    assert af.homography_scale_shift(None, 200, 100) is None
    assert af.homography_scale_shift(np.eye(3), 200, 100) == (1.0, 0.0)

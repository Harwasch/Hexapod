"""`anchor_fill` (round 2 of the inferred-fill bake-off) on the CPU: registration and
compositing, anchored depth completion, the per-token hold of the editor, the leave-out
split, and the whole run -- anchors, sequential and joint propagation, fusion, packaging and
the held-out score -- on a table seen from below its top, with stand-in models."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from fill_scenes import photos, ring_cameras, spool_cameras, table_scene

import anchor_fill as af
import anchor_models as am
import fill_quality as fq
import fill_views as fv
import generative_fill as gf
from splat_render import Camera, load_tileset, render


def _texture(h: int = 72, w: int = 96, seed: int = 0) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (h // 4, w // 4, 3), dtype=np.uint8)
    return cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)


def _masks(render: np.ndarray, hole: np.ndarray) -> af.ViewMasks:
    cam = Camera.look_at([0, 0, 2], [0, 0, 0], width=render.shape[1], height=render.shape[0])
    void = np.zeros_like(hole)
    strength = np.where(hole, 1.0, 0.0).astype(np.float32)
    cond = render.copy()
    cond[hole] = am.KEY
    return af.ViewMasks(
        cam,
        ~hole,
        np.zeros_like(hole),
        hole,
        void,
        render,
        cond,
        strength,
        np.full(hole.shape, 2.0),
    )


def test_registration_undoes_a_small_shift_and_known_pixels_never_change() -> None:
    import cv2

    render = _texture()
    hole = np.zeros(render.shape[:2], bool)
    hole[24:48, 32:64] = True
    shifted = cv2.warpAffine(
        render, np.float32([[1, 0, 2.0], [0, 1, -1.5]]), (96, 72), borderMode=cv2.BORDER_REFLECT
    )
    drawn = np.clip(shifted.astype(float) * 1.1 + 8, 0, 255).astype(np.uint8)
    aligned, info = af.register(drawn, render, ~hole)
    before = af._mse(drawn, render, ~hole & ~af._dilate(hole, 4))
    after = af._mse(aligned, render, ~hole & ~af._dilate(hole, 4))
    assert info["warp"] is not None and after < 0.3 * before
    final = af.composite(aligned, render, ~hole)
    assert (final[~hole] == render[~hole]).all()
    assert not (final[hole] == render[hole]).all()


def test_a_drifted_seed_is_flagged_and_ranked_last() -> None:
    render = _texture()
    hole = np.zeros(render.shape[:2], bool)
    hole[24:48, 32:64] = True
    masks = _masks(render, hole)
    good = render.copy()
    good[hole] = 128
    bad = _texture(seed=5)
    results = [af.EditResult("a0-s17", good), af.EditResult("a0-s1017", bad)]
    cands = af.score_candidates(masks, results, [[], []], af.StandInPerceptual())
    assert not cands[0].drifted and cands[1].drifted
    fill = af.choose(masks, cands, "a0", "anchor")
    assert fill is not None and fill.chosen == 17
    assert (fill.image[~hole] == render[~hole]).all()


def test_harmonic_spreads_the_edge_and_leaves_what_it_cannot_reach() -> None:
    h, w = 40, 80
    known = np.zeros((h, w), bool)
    known[:, :8] = True
    known[:, 40:48] = True
    residual = np.where(np.arange(w)[None, :] >= 40, 1.0, 0.0) * np.ones((h, 1))
    target = np.zeros((h, w), bool)
    target[:, 8:40] = True
    target[:, 60:] = True  # reachable only through cells that are not to solve: stays 0
    out = af.harmonic(residual, known, target, factor=4)
    ramp = out[20, 8:40]
    assert ramp[0] < 0.15 and ramp[-1] > 0.85 and np.all(np.diff(ramp) >= -1e-3)
    assert np.allclose(out[:, 64:], 0.0)


def test_completed_depth_meets_the_scan_at_the_edge() -> None:
    h, w = 60, 90
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    truth = 2.0 + 0.01 * xx + 0.005 * yy
    hole = (np.abs(xx - 45) < 15) & (np.abs(yy - 30) < 12)
    anchor = np.where(hole, np.nan, truth)
    # The model: an affine map of the truth, with its own error growing to the right.
    pred = 0.5 * truth + 0.3 + 0.002 * xx
    depth, info = af.complete_depth(pred, anchor, hole)
    assert depth is not None and info["residual"] < 0.05
    edge = hole & ~af._dilate(~hole, 2)
    assert (
        np.nanmax(
            np.abs(
                depth[hole & ~edge & af._dilate(~hole, 4)]
                - truth[hole & ~edge & af._dilate(~hole, 4)]
            )
        )
        < 0.05
    )
    assert np.nanmax(np.abs(depth[hole] - truth[hole])) < 0.1
    # Too few anchors: no depth.
    none, _ = af.complete_depth(pred, np.where(hole, truth, np.nan) * np.nan, hole)
    assert none is None


def test_tokens_are_held_by_their_strength() -> None:
    strength = np.zeros((32, 48), np.float32)
    strength[:16, :16] = 0.45
    strength[16:, 32:] = 1.0
    strength[0, 47] = 1.0  # one pixel to make frees its whole token
    tokens = am.token_strength(strength, 48, 32)
    assert tokens.shape == (6,)
    assert np.allclose(tokens, [0.45, 0.0, 1.0, 0.0, 0.0, 1.0])
    assert am.held(tokens, 0.5).tolist() == [True, True, False, True, True, False]
    assert am.held(tokens, 0.3).tolist() == [False, True, False, True, True, False]
    assert am.held(tokens, 0.0).tolist() == [False, True, False, True, True, False]


def test_images_and_strength_survive_the_wire() -> None:
    rgb = _texture()
    assert (am.decode_png(am.encode_png(rgb)) == rgb).all()
    s = np.linspace(0, 1, 24 * 32, dtype=np.float32).reshape(24, 32)
    assert np.abs(am.decode_strength(am.encode_strength(s)) - s).max() <= 1 / 255
    filled = am.fill_holes(np.where(np.arange(20)[None, :] < 10, 2.0, np.nan) * np.ones((8, 1)))
    assert np.isfinite(filled).all() and abs(filled[4, 15] - 2.0) < 1e-6


def test_key_colour_left_behind_costs_the_seed_and_is_painted_over() -> None:
    render = _texture()
    hole = np.zeros(render.shape[:2], bool)
    hole[24:48, 32:64] = True
    masks = _masks(render, hole)
    clean = render.copy()
    clean[hole] = 128
    specks = clean.copy()
    specks[30:34, 40:44] = (255, 0, 255)
    specks[40:42, 50:56] = (235, 30, 225)
    results = [af.EditResult("a0-s17", specks), af.EditResult("a0-s1017", clean)]
    cands = af.score_candidates(masks, results, [[], []], af.StandInPerceptual())
    assert cands[0].residue > 0.02 and cands[1].residue == 0.0
    fill = af.choose(masks, cands, "a0", "anchor")
    assert fill is not None and fill.chosen == 1017
    # Forced: the speckled seed's residue is painted over and weighs nothing.
    only = af.choose(masks, cands[:1], "a0", "anchor")
    assert only is not None and only.residue is not None and only.residue[31, 41]
    assert not af.key_residue(only.image, hole).any()
    assert only.weight[31, 41] == 0.0 and only.weight[26, 34] > 0.0
    assert (only.image[~hole] == render[~hole]).all()


def test_the_chosen_seed_is_weighed_against_drifted_seeds_too() -> None:
    render = _texture()
    hole = np.zeros(render.shape[:2], bool)
    hole[24:48, 32:64] = True
    masks = _masks(render, hole)
    good = render.copy()
    good[hole] = 128
    other = _texture(seed=5)  # drifted, and a different fill in the hole
    cands = af.score_candidates(
        masks,
        [af.EditResult("a0-s17", good), af.EditResult("a0-s1017", other)],
        [[], []],
        af.StandInPerceptual(),
    )
    fill = af.choose(masks, cands, "a0", "anchor")
    assert fill is not None and fill.chosen == 17
    assert fill.weight[hole].mean() < 0.9


def test_prompts_name_the_photos_they_show() -> None:
    assert "Pictures 2 and 3 are close-up real photographs" in af.prompt_for("a table", 2)
    one = af.prompt_for("a table", 1)
    assert "Picture 2 is a close-up real photograph" in one and "Picture 3" not in one
    assert "do not copy its framing" in one
    assert "Picture 2" not in af.prompt_for("a table", 0)
    assert "smeared" in af.prompt_for("a table", 0) and "smeared" not in af.prompt_for(
        "x", 0, update=True
    )
    assert "magenta" not in af.prompt_for("a table", 2)


def test_what_is_to_make_is_given_smoothly_filled_from_around_it() -> None:
    render = _texture()
    render[:, :48] = (40, 170, 40)
    render[:, 48:] = (210, 190, 60)
    hole = np.zeros(render.shape[:2], bool)
    hole[20:50, 30:70] = True
    painted = render.copy()
    painted[hole] = (255, 0, 255)
    out = af.prefill(painted, hole, ~hole)
    assert (out[~hole] == painted[~hole]).all()
    assert not af.key_residue(out, hole).any()
    # The flat placeholder: one colour per hole, from around it.
    two = hole.copy()
    two[60:66, 5:12] = True
    flat = af.flat_fill(painted, two, ~two)
    assert len(np.unique(flat[hole].reshape(-1, 3), axis=0)) == 1
    assert (flat[62, 8] == (40, 170, 40)).all()
    assert "flat patches" in af.prompt_for("a table", 0, placeholder="flat")
    # Each side of the hole takes its own side's colour.
    assert out[35, 33, 1] > 130 and out[35, 33, 0] < 90 and out[35, 66, 0] > 160


def test_the_leave_out_holds_out_the_highest_or_lowest() -> None:
    cams = ring_cameras(6, 10.0, 2.0) + ring_cameras(4, 50.0, 2.0)
    views = [gf.RealView(f"c{k}", c) for k, c in enumerate(cams)]
    focus = fq.capture_focus(cams)
    kept, held = af.split_leave_out(views, focus, "high", 0.4)
    assert held == [6, 7, 8, 9] and len(kept) == 6
    kept, held = af.split_leave_out(views, focus, "low", 0.2)
    assert set(held) <= set(range(6)) and len(held) == 2
    assert af.split_leave_out(views, focus, "none", 0.2) == (list(range(10)), [])


def _measured_tileset(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "tileset.json"
    path.write_text(
        json.dumps(
            {
                "asset": {"version": "1.1"},
                "geometricError": 1.0,
                "root": {
                    "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    "boundingVolume": {"box": [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]},
                    "geometricError": 1.0,
                    "extras": {},
                },
            }
        )
    )
    return path


def _write_colmap(folder: Path, cams: list[Camera], names: list[str]) -> None:
    """A COLMAP sparse model of `cams` (one PINHOLE camera each), images named `names`."""
    import struct

    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for k, c in enumerate(cams):
            f.write(struct.pack("<IiQQ", k + 1, 1, c.width, c.height))  # PINHOLE
            f.write(struct.pack("<4d", c.focal, c.focal, c.width / 2, c.height / 2))
    with (folder / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for k, (c, name) in enumerate(zip(cams, names, strict=True)):
            q = gf._quaternion(c.rotation)
            t = -c.rotation @ c.centre
            f.write(struct.pack("<IdddddddI", k + 1, *q, *t, k + 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))


def test_the_command_line_runs_a_leave_out_from_poses_and_tiles(tmp_path: Path) -> None:
    import splat_tiles
    from splat_render import save_ply

    scene, _ = table_scene()
    ply = tmp_path / "scan.ply"
    save_ply(ply, scene)
    splat_tiles.convert(ply, tmp_path / "tiles", 0.0, 0.0, 0.0, opacity_min=0.0)
    measured = load_tileset(tmp_path / "tiles" / "tileset.json")
    cams = spool_cameras() + ring_cameras(5, 65.0, 2.2, phase=0.5)
    paths = photos(measured, cams, tmp_path / "frames")
    _write_colmap(tmp_path / "poses" / "sparse" / "0", cams, [p.name for p in paths])
    argv = [
        "run", str(tmp_path / "tiles" / "tileset.json"), str(tmp_path / "out"),
        "--scan", "table", "--caption", "a round wooden table on a lawn",
        "--poses", str(tmp_path / "poses"), "--frames", str(tmp_path / "frames"),
        "--leave-out", "high", "--leave-share", "0.18",
        "--editor", "standin", "--set-filler", "standin", "--depth", "oracle",
        "--renderer", "cpu", "--perceptual", "standin", "--distill-on", "none",
        "--fill-size", "96,56", "--set-size", "64,36", "--probe-size", "64,36",
        "--quality-width", "96", "--azimuths", "8", "--anchors", "2,3",
        "--propagation", "2,3", "--seeds", "2", "--prop-seeds", "2", "--set-seeds", "2",
        "--lift-stride", "2", "--score-width", "96", "--carve-width", "96",
        "--update-strengths", "0.4", "--placeholder", "flat",
    ]  # fmt: skip
    assert af.main(argv) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["setup"]["kept"] == len(cams) - 5 and report["setup"]["withheld"] > 0
    assert report["options"]["fill_size"] == [96, 56] and report["options"]["anchors"] == [2, 3]
    assert report["options"]["placeholder"] == "flat"
    for arm in af.ARMS:
        assert report["candidates"][arm]["evidence"]["gaussians"] > 0
    assert report["heldOut"]["views"]


def test_the_whole_run_fills_the_held_out_top(tmp_path: Path) -> None:
    scene, m = table_scene()
    cams = spool_cameras() + ring_cameras(5, 65.0, 2.2, phase=0.5)
    paths = photos(scene, cams, tmp_path / "frames")
    views = [gf.RealView(p.name, c, None, p) for p, c in zip(paths, cams, strict=True)]
    setup, info = af.make_setup(
        "table",
        "a round wooden table on a lawn",
        scene,
        views,
        render,
        leave_out="high",
        share=5 / len(cams),
        width=96,
        log=lambda s: None,
    )
    assert len(setup.held_out) == 5 and info["withheld"] > 0
    inner_top = m["top"] & (np.linalg.norm(scene.positions[:, :2], axis=1) < 0.3)
    assert (setup.classes[inner_top] == fq.UNKNOWN).mean() > 0.8  # withheld: to make
    # Context photos are cropped about the hole, at the photo's aspect.
    labels = setup.clusters.labels[inner_top]
    top = int(np.bincount(labels[labels >= 0]).argmax())
    members = np.flatnonzero(setup.clusters.labels == top)
    whole, crop = setup.photo(0, 2048), setup.context_photo(0, members, "top")
    assert whole is not None and crop is not None
    assert crop.shape[1] < whole.shape[1] and crop.shape[1] >= af.CROP_MIN * whole.shape[1] - 1
    assert abs(crop.shape[1] / crop.shape[0] - whole.shape[1] / whole.shape[0]) < 0.1
    # Lifted points far from every hole are floaters.
    near = af.near_holes(setup, np.array([[0.0, 0.0, 0.7], [0.0, 0.0, 3.0]]))
    assert near.tolist() == [True, False]
    options = af.Options(
        fill_size=(96, 56),
        set_size=(64, 36),
        probe_size=(64, 36),
        quality_width=96,
        azimuths=8,
        anchors=(2, 3),
        propagation=(2, 3),
        seeds=2,
        prop_seeds=2,
        set_seeds=2,
        distill=0,
        update_distill=0,
        lift_stride=2,
        score_width=96,
        carve_width=96,
    )
    editor = af.StandInEditor()
    report = af.run_arms(
        setup,
        editor,
        af.StandInSetFiller(),
        af.OracleDepth(scene, render),
        render,
        af.StandInPerceptual(),
        options,
        _measured_tileset(tmp_path / "scan"),
        tmp_path / "out",
        log=lambda s: None,
    )
    targets = report["plan"]["targets"]
    anchors = [t for t in targets if t["role"] == "anchor"]
    assert 2 <= len(anchors) <= 3 and max(t["elevation"] for t in anchors) >= 40.0
    for arm in af.ARMS:
        entry = report["candidates"][arm]
        assert entry.get("evidence"), (arm, {k: v for k, v in entry.items() if k != "fills"})
        assert entry["evidence"]["kind"] == "inferred" and entry["evidence"]["gaussians"] > 0
        assert entry["carvedFinal"] >= 0
        layer = load_tileset(tmp_path / "out" / af.LAYERS[arm] / "inferred" / "tileset.json")
        # The layer is around the table, not in the air, and nothing a camera saw through.
        assert np.percentile(layer.positions[:, 2], 90) <= 0.9
        carver = af.Carver(
            scene,
            [fv.scaled(v.camera, 96) for v in setup.views],
            render,
            2 * setup.focus.radius,
        )
        assert carver.keep(layer.positions).mean() > 0.98  # (the packed tiles round positions)
    assert report["candidates"]["vace"].get("setFrames", 0) >= 5
    # Arms differ: the photos change the anchors.
    refs = report["candidates"]["refs"]["fills"]
    norefs = report["candidates"]["norefs"]["fills"]
    assert refs[0]["context"] and not norefs[0]["context"]
    held = report["heldOut"]
    assert held["views"] and "before" in held["mean"] and all(a in held["mean"] for a in af.ARMS)
    for arm in af.ARMS:
        assert held["mean"][arm]["psnr"] > held["mean"]["before"]["psnr"]
    renders = tmp_path / "out" / "renders"
    for name in ("held-out-table.png", "anchors-table.png", "before-after-table.png"):
        assert (renders / name).exists(), name
    assert editor.calls > 0 and math.isfinite(report["timings"]["totalS"])

"""`world_model_client`: the GPU models' client, against stand-ins for the GPU.

The fakes answer as `infra/modal/world_models.py` does (same methods, same keys -- the
last test reads that file with `ast` to keep the two in step), so what is checked here is
everything but the model: encodings, sizes, ordering, fps, and that Fixer is shown the
full render while only masked pixels are lifted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import teacher_fill as tf
import world_model_client as wmc
from splat_render import Camera, Splats

APP = Path(__file__).resolve().parents[3] / "infra" / "modal" / "world_models.py"


def _gradient(h: int, w: int, shift: int = 0) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([(x + shift) % 256, y % 256, ((x + y) // 2) % 256], axis=2).astype(np.uint8)


def test_png_round_trips_exactly() -> None:
    image = _gradient(30, 50)
    assert np.array_equal(wmc.decode_png(wmc.encode_png(image)), image)


def test_a_clip_round_trips_its_frames() -> None:
    frames = [_gradient(48, 64, k * 4) for k in range(6)]
    back = wmc.decode_mp4(wmc.encode_mp4(frames, 24.0))
    assert len(back) == 6 and back[0].shape == (48, 64, 3)
    assert np.mean(np.abs(back[3].astype(int) - frames[3].astype(int))) < 6


def test_fixer_is_shown_the_full_render_and_answers_at_its_size() -> None:
    seen: list[np.ndarray] = []

    def remote(cls: str, method: str, request: dict) -> dict:
        assert (cls, method) == ("Fixer", "fix")
        (blob,) = request["images"]
        image = wmc.decode_png(blob)
        seen.append(image)
        big = np.kron(image, np.ones((2, 2, 1), np.uint8))  # Fixer answers at its own size
        return {"images": [wmc.encode_png(big)], "model": "fake"}

    rng = np.random.default_rng(0)
    n = 3000
    wall = Splats(
        np.column_stack([rng.uniform(-1, 1, n), np.zeros(n), rng.uniform(0, 2, n)]),
        np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        np.full((n, 3), 0.04),
        np.tile([0.2, 0.6, 0.3], (n, 1)),
        np.full(n, 0.9),
    )
    camera = Camera.look_at([0.0, -4.0, 1.0], [0.0, 0.0, 1.0], width=80, height=60)
    seen_opacity = np.where(wall.positions[:, 0] < 0, 1.0, 0.0)  # the right half never seen
    cond = tf.condition(wall, camera, None, seen_opacity=seen_opacity)
    assert cond.mask.any()
    filler = wmc.FixerFiller(remote=remote)
    (filled,) = tf.fill_views([cond], filler)
    full = tf.to_u8(cond.full.rgb)
    assert np.array_equal(seen[0], full)  # the unseen side, artifacts and all
    assert filled.rgb.shape == full.shape
    assert filled.accepted  # it changed nothing it was not asked to


def test_fixer_sends_its_timestep_and_names_it() -> None:
    sent: list[dict] = []

    def remote(cls: str, method: str, request: dict) -> dict:
        sent.append(request)
        return {"images": request["images"], "model": "fake"}

    rgb = np.full((8, 8, 3), 100, np.uint8)
    wmc.FixerFiller(remote=remote).fill(rgb, np.zeros((8, 8), bool))
    filler = wmc.FixerFiller(remote=remote, timestep=50)
    filler.fill(rgb, np.zeros((8, 8), bool))
    assert "timestep" not in sent[0] and sent[1]["timestep"] == 50
    assert filler.name == "nvidia-fixer-t50"


def test_video_clips_ask_every_still_once_per_seed_and_resize_back() -> None:
    calls: list[dict] = []

    def remote(cls: str, method: str, request: dict) -> dict:
        assert (cls, method) == ("Wan", "clip")
        calls.append(request)
        still = wmc.decode_png(request["image"])
        frames = [np.kron(still, np.ones((2, 2, 1), np.uint8))] * 5
        return {"mp4": wmc.encode_mp4(frames, 24.0), "fps": 24.0, "model": "fake-wan"}

    stills = [np.full((32, 48, 3), 40 * (c + 1), np.uint8) for c in range(3)]
    source = wmc.VideoClips(model="Wan", remote=remote)
    clips = source.clips(stills, [None] * 3, seeds=(1, 2))
    assert len(clips) == 6 and source.fps == 24.0
    for k, clip in enumerate(clips):
        assert len(clip) == 5 and clip[0].shape == (32, 48, 3)
        assert abs(int(clip[0].mean()) - 40 * (k % 3 + 1)) <= 3  # clip k is of still k % 3
    assert len({c["seed"] for c in calls}) == 6
    assert all(c["prompt"] == wmc.PLANT_PROMPT for c in calls)
    assert [r["frames"] for r in source.received] == [5] * 6


def test_video_clips_refuse_a_clip_at_the_wrong_rate() -> None:
    def remote(cls: str, method: str, request: dict) -> dict:
        still = wmc.decode_png(request["image"])
        return {"mp4": wmc.encode_mp4([still] * 3, 30.0), "fps": 30.0}

    source = wmc.VideoClips(model="Cosmos", remote=remote)
    assert source.fps == 16.0
    with pytest.raises(ValueError, match="fps"):
        source.clips([np.zeros((16, 16, 3), np.uint8)], [None], seeds=(1,))
    with pytest.raises(ValueError):
        wmc.VideoClips(model="Sora")


def test_the_client_calls_what_the_app_defines() -> None:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    constants = {
        t.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert constants["APP_NAME"] == wmc.APP_NAME
    methods = {
        node.name: {
            f.name
            for f in node.body
            if isinstance(f, ast.FunctionDef)
            and any(
                isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "method"
                for d in f.decorator_list
            )
        }
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert methods["Fixer"] == {"fix"} and methods["Distill"] == {"run"}
    assert methods["Wan"] == {"clip"} and methods["Cosmos"] == {"clip"}
    # segment_models' Modal clients (ModalSam2Masks, ModalSiglipEmbedder) call these.
    assert methods["SegmentMasks"] == {"masks"}
    assert methods["SegmentEmbed"] == {"embed_images", "embed_texts"}
    # The rates the client expects are the rates the app sends.
    assert constants["WAN_FPS"] == wmc.VideoClips(model="Wan", remote=None).fps  # type: ignore[arg-type]
    assert constants["COSMOS_FPS"] == wmc.VideoClips(model="Cosmos", remote=None).fps  # type: ignore[arg-type]


def test_video_clips_chain_each_call_from_the_last_frame() -> None:
    sent: list[np.ndarray] = []

    def remote(cls: str, method: str, request: dict) -> dict:
        still = wmc.decode_png(request["image"])
        sent.append(still)
        # Each frame a little brighter than the one before: the last is the next start.
        frames = [np.clip(still.astype(int) + 10 * k, 0, 255).astype(np.uint8) for k in range(4)]
        return {"mp4": wmc.encode_mp4(frames, 24.0), "fps": 24.0, "model": "fake", "seconds": 2.0}

    source = wmc.VideoClips(model="Wan", remote=remote, chain=3, steps=20)
    (clip,) = source.clips([np.full((32, 48, 3), 50, np.uint8)], [None], seeds=(1,))
    assert len(clip) == 4 + 3 + 3  # the joins' repeated frames dropped
    assert [round(float(s.mean())) for s in sent] == pytest.approx([50, 80, 110], abs=3)
    assert abs(float(clip[-1].mean()) - 140) <= 8  # MJPG shifts the colour a little
    assert [r["link"] for r in source.received] == [0, 1, 2]
    assert len({r["seed"] for r in source.received}) == 3
    assert source.name == "wan2.2-ti2v-5b-chain3"
    with pytest.raises(ValueError):
        wmc.VideoClips(chain=0)


def test_presmooth_fills_the_gaps_between_samples_and_leaves_empty_space_dark() -> None:
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[::2, 0:20:2] = (200, 100, 50)  # samples on every other pixel, left half
    out = wmc.presmooth(rgb, 1.0)
    assert out[5:35, 5:15, 0].min() > 100  # the gaps took the samples' colour
    assert out[:, 30:].max() == 0  # nothing drawn where nothing was

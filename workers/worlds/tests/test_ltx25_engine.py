"""CPU contract checks; no model inference or quality claim."""

import copy
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltx25.engine import HEIGHT, WIDTH, Engine, write_packet
from ltx25.planning import (
    PromptCache,
    Timeline,
    TransformerResidency,
    budget_prompt,
    compose_prompt,
    validate_request,
)


def request():
    return {
        "op": "generate",
        "prompt": "A forest",
        "seed": 42,
        "quality": "balanced",
        "action": {
            "motion": {"forward": 1, "right": 0, "up": 0, "yaw": 0, "pitch": 0},
            "exploration": {"mode": "walk", "cadence": "balanced", "speed": 0.5},
            "revision": 0,
            "basePrompt": "A forest",
            "promptAmendments": [],
        },
    }


class PlanningTests(unittest.TestCase):
    def test_indefinite_timeline_profile_changes_have_no_gaps_or_overlap_in_owned_frames(
        self,
    ):
        timeline = Timeline()
        end, carry = 0, 0
        for index in range(10000):
            window = timeline.next(("smooth", "responsive", "balanced")[index % 3])
            self.assertEqual(
                window.start_pixel_frame + window.prev_video_carry_frames, end
            )
            self.assertEqual(window.prev_video_carry_frames, carry)
            self.assertEqual((window.pixel_frames - 1) % 8, 0)
            self.assertGreaterEqual(window.next_video_carry_frames, 17)
            self.assertLess(window.next_video_blend_frames, window.pixel_frames - carry)
            end = window.start_pixel_frame + window.pixel_frames
            carry = window.next_video_carry_frames
        self.assertEqual(timeline.end, end)
        self.assertEqual(set(vars(timeline)), {"end", "carry"}, "No growing history")

    def test_combined_motion_and_amendments_preserve_world_and_newest_precedence(self):
        req = request()
        req["prompt"] = "A helicopter appears"
        req["action"]["promptAmendments"] = ["It starts raining", req["prompt"]]
        req["action"]["motion"]["yaw"] = -0.5
        prompt = compose_prompt(req)
        for phrase in (
            "A forest",
            "It starts raining",
            "A helicopter appears",
            "newer directions take precedence",
            "moves forward and turns left",
        ):
            self.assertIn(phrase, prompt)
        req["action"]["exploration"]["mode"] = "direct"
        self.assertNotIn("moves forward", compose_prompt(req))
        self.assertIn("turns left", compose_prompt(req))

    def test_cache_bounded_and_jitter_does_not_reencode(self):
        cache = PromptCache(4)
        calls = []

        def encode(text):
            calls.append(text)
            return object()

        req = request()
        first = cache.get(compose_prompt(req), encode)
        req["action"]["motion"]["forward"] = 0.99
        self.assertIs(cache.get(compose_prompt(req), encode), first)
        for index in range(9):
            cache.get(str(index), encode)
        self.assertEqual(len(cache.entries), 4)
        self.assertEqual(cache.encodings, 10)
        self.assertEqual(len(calls), 10)

    def test_long_world_cannot_truncate_camera_or_latest_scene_change(self):
        tokenizer = SimpleNamespace(
            encode=lambda text, **_: list(text),
            decode=lambda tokens, **_: "".join(tokens),
        )
        req = request()
        req["action"]["basePrompt"] = "forest " * 1000
        req["action"]["promptAmendments"] = ["rain " * 200, "A helicopter appears"]
        prompt, shortened = budget_prompt(req, tokenizer)
        self.assertTrue(shortened)
        self.assertLessEqual(len(tokenizer.encode(prompt)), 1023)
        self.assertIn("moves forward", prompt)
        self.assertIn("A helicopter appears", prompt)
        self.assertIn("World description: forest", prompt)
        req = request()
        prompt, shortened = budget_prompt(req, tokenizer)
        self.assertFalse(shortened)
        self.assertEqual(prompt, compose_prompt(req))

    def test_invalid_commands_rejected(self):
        cases = [
            ("revision", True),
            ("revision", -1),
            ("motion", {"yaw": float("nan")}),
            ("motion", {"forward": 2}),
            ("promptAmendments", ["x" * 3001]),
            ("promptAmendments", ["x"] * 7),
        ]
        for key, value in cases:
            req = request()
            req["action"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_request(req)
        req = request()
        req["action"]["exploration"]["speed"] = float("inf")
        with self.assertRaises(ValueError):
            validate_request(req)

    def test_transformer_residency_builds_once_and_disposes_only_on_close(self):
        model = SimpleNamespace(disposed=0)

        def dispose():
            model.disposed += 1

        model.dispose = dispose
        stage = SimpleNamespace(
            _build_transformer=lambda **_: model,
            _transformer_ctx=lambda **_: nullcontext(model),
        )
        original = stage._transformer_ctx
        resident = TransformerResidency(stage, keep=True)
        for _ in range(7):
            with stage._transformer_ctx() as current:
                self.assertIs(current, model)
        self.assertEqual(resident.loads, 1)
        self.assertEqual(model.disposed, 0)
        resident.close()
        self.assertEqual(model.disposed, 1)
        self.assertIs(stage._transformer_ctx, original)
        resident.close()
        self.assertEqual(model.disposed, 1)

    def test_cpu_mode_preserves_upstream_context_lifecycle(self):
        exits = []

        @contextmanager
        def upstream(**_):
            try:
                yield "model"
            finally:
                exits.append(True)

        stage = SimpleNamespace(_transformer_ctx=upstream)
        resident = TransformerResidency(stage, keep=False)
        for _ in range(4):
            with stage._transformer_ctx() as model:
                self.assertEqual(model, "model")
        self.assertEqual(len(exits), 4)
        self.assertEqual(resident.loads, 4)


class ArrayTensor:
    """Only the documented tensor output contract, backed by real NumPy math."""

    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def T(self):
        return ArrayTensor(self.value.T)

    def __len__(self):
        return len(self.value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return ArrayTensor(self.value.astype(np.float32))

    def clamp(self, low, high):
        return ArrayTensor(self.value.clip(low, high))

    def mul(self, value):
        return ArrayTensor(self.value * value)

    def round(self):
        return ArrayTensor(self.value.round())

    def to(self, dtype):
        return ArrayTensor(self.value.astype(dtype))

    def contiguous(self):
        return ArrayTensor(np.ascontiguousarray(self.value))

    def numpy(self):
        return self.value


TORCH = SimpleNamespace(
    cat=lambda parts, dim: ArrayTensor(
        np.concatenate([p.value for p in parts], axis=dim)
    ),
    uint8=np.uint8,
    isfinite=lambda x: np.isfinite(x.value),
    inference_mode=nullcontext,
    cuda=SimpleNamespace(synchronize=lambda: None),
)


@dataclass
class AudioPacket:
    waveform: object
    sampling_rate: int


@dataclass
class DecodedPacket:
    video: object
    audio: object


class OutputTests(unittest.TestCase):
    def packet(self):
        video = np.zeros((2, HEIGHT, WIDTH, 3), dtype=np.float32)
        video[..., 0] = 1
        video[..., 1] = 0.5
        # Stereo 24kHz -> stereo48kHz; real resampling is exercised.
        time = np.arange(2000) / 24000
        wave = np.stack(
            [0.2 * np.sin(2 * np.pi * 440 * time), 0.1 * np.sin(2 * np.pi * 880 * time)]
        )
        return DecodedPacket(
            video=ArrayTensor(video),
            audio=AudioPacket(waveform=ArrayTensor(wave), sampling_rate=24000),
        )

    def test_real_rgb_files_and_resampling_are_clock_aligned(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(sys.modules, {"torch": TORCH}),
        ):
            root = Path(temp)
            result = write_packet(self.packet(), root)
            rgb = np.fromfile(result["rawFrames"]["path"], dtype=np.uint8).reshape(
                2, HEIGHT, WIDTH, 3
            )
            np.testing.assert_array_equal(rgb[0, 0, 0], [255, 128, 0])
            audio = result["audio"]
            self.assertEqual(audio["samples"], 4000)
            pcm = np.fromfile(audio["path"], dtype="<i2").reshape(-1, 2)
            self.assertEqual(pcm.shape, (4000, 2))
            self.assertGreater(np.abs(pcm[:, 0]).max(), 6000)
            self.assertGreater(np.abs(pcm[:, 1]).max(), 3000)
            self.assertEqual(Path(result["continuationPath"]).suffix, ".png")

    def test_nan_and_unaligned_audio_fail_instead_of_silent_fake_output(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(sys.modules, {"torch": TORCH}),
        ):
            packet = self.packet()
            packet.video.value[0, 0, 0, 0] = np.nan
            with self.assertRaises(ValueError):
                write_packet(packet, Path(temp))
            packet = self.packet()
            packet.audio.waveform = ArrayTensor(np.zeros((2, 10000)))
            with self.assertRaises(ValueError):
                write_packet(packet, Path(temp))

    def test_seam_body_are_one_response_and_transient_commands_always_sample(self):
        engine = Engine.__new__(Engine)
        engine.torch = TORCH
        engine.reset()
        engine.resident = SimpleNamespace(loads=1)
        engine.residency_mode = "gpu"
        commands = []

        def decoded():
            while True:
                commands.append(copy.deepcopy(engine.request["action"]))
                engine.active_revision = engine.request["action"]["revision"]
                engine.windows += 1
                if engine.windows > 1:
                    yield self.packet()  # previous blended seam
                yield self.packet()  # newly sampled body

        engine.decoded = decoded()
        engine.seed = 42
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(sys.modules, {"torch": TORCH}),
        ):
            req = dict(request(), directory=temp)
            first = engine.generate(req)
            self.assertEqual(first["appliedRevision"], 0)
            self.assertEqual(first["generatedFrames"], 2)
            req = copy.deepcopy(req)
            req["action"]["revision"] = 3
            req["action"]["motion"]["yaw"] = 0.5
            second = engine.generate(req)
            self.assertEqual(second["appliedRevision"], 3)
            self.assertEqual(second["generatedFrames"], 4)
            self.assertEqual(second["audio"]["samples"], 8000)
            self.assertEqual(commands[-1]["motion"]["yaw"], 0.5)
            req["action"]["motion"]["yaw"] = 0
            third = engine.generate(req)
            self.assertEqual(third["renderedWindowCount"], 3)
            self.assertEqual(commands[-1]["motion"]["yaw"], 0)
            self.assertEqual(len(commands), 3)
            with self.assertRaises(ValueError):
                engine.generate(dict(req, seed=99))
            engine.reset()
            self.assertIsNone(engine.decoded)
            self.assertEqual(engine.total_frames, 0)
            self.assertEqual(len(engine.cache.entries), 0)


if __name__ == "__main__":
    unittest.main()

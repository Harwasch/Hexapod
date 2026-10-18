"""Exercise actual adapter generate methods against independent upstream fixtures.

No model or torch simulation: the fixtures only return the documented decoded
layouts/ranges and record calls. Array math and output files are real NumPy/PIL.
"""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forgewm.engine import Engine as ForgeEngine
from matrix_game.engine import Engine as MatrixEngine
from sana_wm.engine import Engine as SanaEngine


class TensorFixture:
    """Minimal tensor arithmetic for checking layout/range conversion on CPU."""
    def __init__(self, value): self.array = np.asarray(value)
    def __getitem__(self, key): return TensorFixture(self.array[key])
    def permute(self, *axes): return TensorFixture(self.array.transpose(axes))
    def clamp(self, minimum, maximum): return TensorFixture(self.array.clip(minimum, maximum))
    def __mul__(self, value): return TensorFixture(self.array * value)
    def __add__(self, value): return TensorFixture(self.array + value)
    def round(self): return TensorFixture(self.array.round())
    def byte(self): return TensorFixture(self.array.astype(np.uint8))
    def cpu(self): return self
    def numpy(self): return self.array
    def repeat(self, *shape): return TensorFixture(np.tile(self.array, shape))


TORCH = SimpleNamespace(inference_mode=nullcontext, tensor=TensorFixture)


class GenerateContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / 'input.png'
        Image.new('RGB', (12, 8), (10, 20, 30)).save(self.input)
        self.request = {'image': str(self.input), 'prompt': 'A forest', 'seed': 123,
                        'action': 'left', 'quality': 'balanced', 'directory': str(self.root / 'output')}

    def tearDown(self): self.temp.cleanup()

    def pixels(self, result):
        raw = result['rawFrames']
        return np.fromfile(raw['path'], dtype=np.uint8).reshape(raw['count'], raw['height'], raw['width'], 3)

    def test_forge_supplies_9_pixel_actions_for_3_latents_and_preserves_rgb(self):
        engine = ForgeEngine.__new__(ForgeEngine)
        engine.torch, engine.device, engine.dtype = TORCH, 'cuda', 'bf16'
        engine.pipeline = object()
        # Published Forge output is [B,T,C,H,W], 0..1. Each channel is distinct.
        video = np.zeros((1, 9, 3, 2, 3), dtype=np.float32)
        video[:, :, 0] = 1.; video[:, :, 1] = .5; video[:, :, 2] = -1.
        engine.api = SimpleNamespace(load_reference_frame=Mock(return_value='reference'),
            make_action=Mock(return_value=('mouse', 'keyboard')), build_conditional_dict=Mock(return_value='conditions'),
            infer_causal=Mock(return_value=TensorFixture(video)))
        result = engine.generate(dict(self.request, prompt=''))
        engine.api.load_reference_frame.assert_called_once_with(str(self.input), 'cuda', 'bf16', 352, 640)
        engine.api.make_action.assert_called_once_with('left', 9)
        engine.api.build_conditional_dict.assert_called_once_with(engine.pipeline, 'reference', 3, 'mouse', 'keyboard', 'bf16', 'cuda')
        engine.api.infer_causal.assert_called_once_with(engine.pipeline, 'conditions', 3, 352, 640, 'cuda', 'bf16', seed=123)
        pixels = self.pixels(result)
        self.assertEqual(pixels.shape, (9, 2, 3, 3))
        np.testing.assert_array_equal(pixels[0, 0, 0], [255, 128, 0])
        np.testing.assert_array_equal(np.array(Image.open(result['continuationPath'])), pixels[-1])
        self.assertEqual(result['fps'], 12)
        with self.assertRaises(ValueError): engine.generate(self.request)

    def test_matrix_action_callback_matches_native_six_key_schema_and_decoded_layout(self):
        engine = MatrixEngine.__new__(MatrixEngine)
        engine.torch = TORCH
        engine.inputs = SimpleNamespace(Bench_actions_universal=None)
        engine.args = SimpleNamespace(output_dir='old')
        calls = []
        # Upstream Matrix returns [C,T,H,W], -1..1, without a batch axis.
        video = np.zeros((3, 57, 2, 3), dtype=np.float32)
        video[0] = -1.; video[1] = 0.; video[2] = 1.
        def generate(prompt, image, **kwargs):
            calls.append((prompt, image.mode, kwargs))
            native = engine.inputs.Bench_actions_universal(57)
            self.assertEqual(native['keyboard_condition'].array.shape, (57, 6))
            np.testing.assert_array_equal(native['keyboard_condition'].array[0], [0, 0, 1, 0, 0, 0])
            np.testing.assert_array_equal(native['mouse_condition'].array, np.zeros((57, 2)))
            return TensorFixture(video)
        engine.pipeline = SimpleNamespace(generate=generate)
        result = engine.generate(self.request)
        self.assertEqual(calls[0][0:2], ('A forest', 'RGB'))
        self.assertEqual(calls[0][2]['num_inference_steps'], 3)
        self.assertEqual(calls[0][2]['seed'], 123)
        self.assertEqual(engine.pipeline.output_dir, self.request['directory'])
        pixels = self.pixels(result)
        self.assertEqual(pixels.shape, (57, 2, 3, 3))
        np.testing.assert_array_equal(pixels[0, 0, 0], [0, 128, 255])
        self.assertEqual(result['fps'], 17)

    def test_sana_camera_stride_and_callback_frames_are_consistent(self):
        engine = SanaEngine.__new__(SanaEngine)
        engine.torch = TORCH
        poses = np.tile(np.eye(4, dtype=np.float32), (25, 1, 1))
        engine.api = SimpleNamespace(GenerationParams=lambda **kw: SimpleNamespace(**kw),
            action_string_to_c2w=Mock(return_value=poses), resize_and_center_crop=lambda image: (image, None, None, None))
        cache_a, cache_b = {'private-old-prompt': 1}, {'private-old-prompt': 2}
        def generate(image, prompt, c2w, intrinsics, params, **kw):
            self.assertEqual(cache_a, {}); self.assertEqual(cache_b, {})
            self.assertEqual((params.num_frames - 1) // 8, 3)
            self.assertEqual(c2w.shape, (25, 4, 4))
            self.assertEqual(intrinsics.shape, (25, 4))
            np.testing.assert_array_equal(intrinsics[0], [640, 640, 640, 352])
            self.assertEqual(params.denoising_step_list, [1000, 960, 889, 727, 0])
            self.assertEqual(kw['output_mode'], 'cpu')
            # Upstream callbacks deliver [T,H,W,C] uint8, with first sink dropped.
            callback = kw['decoded_chunk_callback']
            callback(np.full((24, 2, 3, 3), 173, dtype=np.uint8), 0, 0)
            return {'n_pixel_frames': 24}
        engine.pipeline = SimpleNamespace(generate_streaming=generate,
            _streaming_stage1_prompt_cache=cache_a, _streaming_refiner_prompt_cache=cache_b)
        result = engine.generate(self.request)
        engine.api.action_string_to_c2w.assert_called_once_with('j-24')
        self.assertEqual(self.pixels(result).shape, (24, 2, 3, 3))
        self.assertTrue((self.pixels(result) == 173).all())
        self.assertEqual(result['fps'], 16)
        self.assertEqual(result['cameraIntrinsicsKind'], 'approximate-pinhole')

    def test_sana_keeps_only_current_prompt_embeddings_between_blocks(self):
        engine = SanaEngine.__new__(SanaEngine)
        engine.torch = TORCH
        engine.api = SimpleNamespace(GenerationParams=lambda **kw: kw, action_string_to_c2w=lambda *_: None,
            resize_and_center_crop=lambda image: (image,))
        cache_a, cache_b = {}, {}
        observed = []
        def generate(image, prompt, *args, **kwargs):
            observed.append((dict(cache_a), dict(cache_b)))
            cache_a[prompt] = 'stage1-embedding'; cache_b[prompt] = 'refiner-embedding'
            kwargs['decoded_chunk_callback'](np.zeros((24, 2, 2, 3), dtype=np.uint8), 0, 0)
        engine.pipeline = SimpleNamespace(generate_streaming=generate,
            _streaming_stage1_prompt_cache=cache_a, _streaming_refiner_prompt_cache=cache_b)
        engine.generate(self.request)
        engine.generate(self.request)
        engine.generate(dict(self.request, prompt='A changed scene'))
        self.assertEqual(observed[0], ({}, {}))
        self.assertEqual(observed[1], ({'A forest':'stage1-embedding'}, {'A forest':'refiner-embedding'}))
        self.assertEqual(observed[2], ({}, {}))
        self.assertEqual(set(cache_a), {'A changed scene'})
        self.assertEqual(set(cache_b), {'A changed scene'})

    def test_sana_rejects_upstream_frame_overflow(self):
        engine = SanaEngine.__new__(SanaEngine)
        engine.torch = TORCH
        engine.api = SimpleNamespace(GenerationParams=lambda **kw: kw, action_string_to_c2w=lambda *_: None,
            resize_and_center_crop=lambda image: (image,))
        def overflow(*args, **kwargs): kwargs['decoded_chunk_callback'](np.zeros((65, 2, 2, 3), dtype=np.uint8), 0, 0)
        engine.pipeline = SimpleNamespace(generate_streaming=overflow,
            _streaming_stage1_prompt_cache={}, _streaming_refiner_prompt_cache={})
        with self.assertRaises(ValueError): engine.generate(self.request)


if __name__ == '__main__': unittest.main()

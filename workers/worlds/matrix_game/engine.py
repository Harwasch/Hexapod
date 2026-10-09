"""Pinned Matrix3 distilled inference; native actions replace benchmark actions.

The bootstrap's one-line reviewed patch returns the decoded tensor instead of
terminating Python. Native long-horizon caches are NOT carried between calls.
"""
from types import SimpleNamespace
from model_support import write_frames


def action_vectors(action):
    keyboard, mouse = [0.] * 6, [0.] * 2
    if action in ('forward', 'backward', 'left', 'right'):
        keyboard[('forward', 'backward', 'left', 'right').index(action)] = 1.
    elif action in ('look_left', 'look_right'):
        mouse[1] = -.1 if action == 'look_left' else .1
    elif action in ('look_up', 'look_down'):
        mouse[0] = .1 if action == 'look_up' else -.1
    elif action != 'stop':
        raise ValueError('Unsupported native action')
    return keyboard, mouse


class Engine:
    def __init__(self, source, weights):
        import torch
        import pipeline.inference_pipeline as upstream
        import utils.utils as inputs
        from wan.configs.config import matrix_game3
        self.torch, self.inputs = torch, inputs
        self.original_actions = inputs.Bench_actions_universal
        self.args = SimpleNamespace(ckpt_dir=str(weights), vae_type='wan2.2', lightvae_pruning_rate=0.0,
            use_async_vae=False, use_int8=False, verify_quant=False, output_dir='/tmp', num_iterations=1,
            size='704*1280', save_name='worlds', use_camera=False)
        self.pipeline = upstream.MatrixGame3Pipeline(matrix_game3, str(weights), args=self.args,
                                                     use_camera=False, use_base_model=False)
        # The upstream renderer adds keyboard overlays and writes MP4. Capture
        # its already-decoded return tensor instead, without lossy recompression.
        upstream.process_video = lambda *args, **kwargs: None

    def reset(self):
        self.inputs.Bench_actions_universal = self.original_actions
        self.pipeline.vae.model.clear_cache()
        self.pipeline.output_dir = "/tmp"
        self.args.output_dir = "/tmp"

    def generate(self, request):
        from PIL import Image
        keyboard, mouse = action_vectors(request['action'])
        def actions(frame_count):
            return {'keyboard_condition': self.torch.tensor(keyboard).repeat(frame_count, 1),
                    'mouse_condition': self.torch.tensor(mouse).repeat(frame_count, 1)}
        self.inputs.Bench_actions_universal = actions
        self.args.output_dir = request['directory']
        self.pipeline.output_dir = request['directory']
        with Image.open(request['image']) as image, self.torch.inference_mode():
            video = self.pipeline.generate(request['prompt'], image.convert('RGB'), seed=request['seed'], num_inference_steps=3, args=self.args)
            if video is None:
                raise RuntimeError('Upstream returned no decoded frames')
            frames = ((video.permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
            # Pinned upstream utils/visualize.py exports at 17 fps.
            return write_frames(frames, request['directory'], 17)

"""Direct SANA-WM streaming pipeline with bounded decoded-frame callback.

Each request uses one 24-frame refined block, fixed approximate pinhole
intrinsics, and image continuation. Intrinsics are not a camera estimate.
"""
import os
from model_support import write_frames

ACTION_MAP = {'forward': 'w', 'backward': 's', 'left': 'j', 'right': 'l', 'look_left': 'a', 'look_right': 'd', 'look_up': 'i', 'look_down': 'k', 'stop': 'none'}


class Engine:
    def __init__(self, source, weights):
        os.environ['DISABLE_XFORMERS'] = '1'
        import torch
        import pyrallis
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import inference_video_scripts.wm.inference_sana_wm as upstream
        self.api, self.torch = upstream, torch
        # Upstream maps this symbolic encoder to an unpinned remote Hub name.
        # Resolve it to our explicitly installed immutable local snapshot.
        def local_text_encoder(name, device):
            if name != 'gemma-2-2b-it':
                raise ValueError('Unexpected SANA text encoder')
            tokenizer = AutoTokenizer.from_pretrained(str(weights / 'gemma2_2b'), local_files_only=True)
            tokenizer.padding_side = 'right'
            model = AutoModelForCausalLM.from_pretrained(str(weights / 'gemma2_2b'), torch_dtype=torch.bfloat16, local_files_only=True)
            return tokenizer, model.get_decoder().to(device)
        upstream.get_tokenizer_and_text_encoder = local_text_encoder
        config = pyrallis.parse(config_class=upstream.InferenceConfig,
            config_path=str(source / 'configs/sana_wm/sana_wm_streaming_1600m_720p.yaml'), args=[])
        config.vae.vae_type = 'LTX2VAE_diffusers_causal'
        config.vae.vae_pretrained = str(weights / 'ltx2_causal_vae')
        refiner = upstream.RefinerSettings(root=str(weights / 'refiner_diffusers'),
            gemma_root=str(weights / 'gemma3_12b'), sink_size=1, block_size=3, kv_max_frames=11)
        self.pipeline = upstream.SanaWMPipeline(config, str(weights / 'sana_dit/model.pt'), device='cuda', refiner=refiner)
        # No torch.compile on the causal VAE: pinned upstream documents cache
        # corruption on later chunks when compiling that decoder.

    def generate(self, request):
        import numpy as np
        from PIL import Image
        params = self.api.GenerationParams(num_frames=25, fps=16, cfg_scale=1., seed=request['seed'],
            sampling_algo='self_forcing', num_cached_blocks=2, sink_token=True,
            num_frame_per_block=3, denoising_step_list=[1000, 960, 889, 727, 0])
        c2w = self.api.action_string_to_c2w(ACTION_MAP[request['action']] + '-24')
        # Fixed 90-degree horizontal field of view, openly approximate. Avoid
        # silently downloading Pi3X or representing this as measured geometry.
        intrinsics = np.tile(np.array([640., 640., 640., 352.], dtype=np.float32), (25, 1))
        frames = []
        def collect(chunk, offset, block):
            array = np.asarray(chunk)
            if sum(len(item) for item in frames) + len(array) > 64:
                raise ValueError('Upstream exceeded bounded frame output')
            frames.append(array.copy())
        # Bound prompt-dependent caches to the current prompt. The upstream
        # refiner loads/deletes Gemma3 when encoding a cache miss; preserving
        # the same prompt avoids reloading 12B text weights on every block.
        # Changed prompts evict their predecessor; session end kills the child.
        if request['prompt'] != getattr(self, 'active_prompt', None):
            self.pipeline._streaming_stage1_prompt_cache.clear()
            self.pipeline._streaming_refiner_prompt_cache.clear()
            self.active_prompt = request['prompt']
        with Image.open(request['image']) as image, self.torch.inference_mode():
            cropped, *_ = self.api.resize_and_center_crop(image.convert('RGB'))
            self.pipeline.generate_streaming(cropped, request['prompt'], c2w, intrinsics, params,
                output_path=str(request['directory']) + '/unused.mp4', output_mode='cpu', decoded_chunk_callback=collect)
        if not frames:
            raise RuntimeError('SANA returned no decoded frames')
        result = write_frames(np.concatenate(frames), request['directory'], 16)
        result['cameraIntrinsicsKind'] = 'approximate-pinhole'
        return result

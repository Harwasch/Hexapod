"""Direct ForgeWM stage-3 (4-step) upstream API integration."""
from model_support import write_frames

ACTION_MAP = {'forward': 'forward', 'backward': 'back', 'left': 'left', 'right': 'right', 'look_left': 'turn_left', 'look_right': 'turn_right', 'look_up': 'look_up', 'look_down': 'look_down', 'stop': 'no_action'}


class Engine:
    def __init__(self, source, weights):
        import torch
        from omegaconf import OmegaConf
        import inference
        self.torch, self.api = torch, inference
        self.device, self.dtype = torch.device('cuda'), torch.bfloat16
        config = OmegaConf.merge(OmegaConf.load(source / 'configs/default.yaml'), OmegaConf.load(source / 'configs/stage3_dmd.yaml'))
        base = weights / 'MG2-base'
        config.model_kwargs.model_name = str(base)
        vae = inference.WanVAEWrapper(vae_path=str(base / 'Wan2.1_VAE.pth'),
            clip_checkpoint_path=str(base / 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
            clip_tokenizer_path=str(base / 'xlm-roberta-large'))
        self.pipeline = inference.CausalInferencePipeline(config, device=self.device, vae=vae)
        state = torch.load(weights / 'stage3/model.pt', map_location='cpu', weights_only=True)
        state = state.get('generator', state.get('generator_ema', state))
        state = {k.replace('._fsdp_wrapped_module.', '.').replace('._checkpoint_wrapped_module.', '.'): v for k, v in state.items()}
        # The published stage-3 generator must match its reviewed config exactly.
        self.pipeline.generator.load_state_dict(state, strict=True)
        self.pipeline.generator.to(device=self.device, dtype=self.dtype).eval()
        self.pipeline.vae.to(device=self.device, dtype=self.dtype).eval()

    def reset(self):
        for name in ("kv_cache1", "kv_cache_mouse", "kv_cache_keyboard", "crossattn_cache"):
            setattr(self.pipeline, name, None)
        self.pipeline.vae.model.clear_cache()

    def generate(self, request):
        if request['prompt'].strip():
            raise ValueError('ForgeWM has no text encoder; text conditioning is unsupported')
        with self.torch.inference_mode():
            pixel = self.api.load_reference_frame(request['image'], self.device, self.dtype, 352, 640)
            mouse, keyboard = self.api.make_action(ACTION_MAP[request['action']], 9)
            condition = self.api.build_conditional_dict(self.pipeline, pixel, 3, mouse, keyboard, self.dtype, self.device)
            video = self.api.infer_causal(self.pipeline, condition, 3, 352, 640, self.device, self.dtype, seed=request['seed'])
            frames = (video[0].permute(0, 2, 3, 1).clamp(0, 1) * 255).round().byte().cpu().numpy()
            return write_frames(frames, request['directory'], 12)

"""Resident Astronex inference using APIs inspected at the manifest's source SHA.

One anonymous session owns a process at a time. Model/text encoder/VAE load once;
a verified reset can retain weights for another session without its prior state.
Each request generates exactly one model block, retaining its final clean latent.
Upstream inference() resets the long-horizon KV cache on every call: this adapter
makes no claim of persistent native scene memory or exact snapshot restoration.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

MAX_MESSAGE = 64 * 1024


def validate_request(request):
    if not isinstance(request, dict) or request.get("op") != "generate":
        raise ValueError("Unsupported resident request")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError("Invalid prompt")
    seed = request.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    if request.get("quality") not in ("balanced", "quality", "low-latency", "speed"):
        raise ValueError("Invalid performance mode")
    from astronex.adapter import ACTIONS
    from astronex.controls import validate_motion
    if isinstance(request.get("action"), dict):
        validate_motion(request["action"])
    elif request.get("action") not in ACTIONS:
        raise ValueError("Invalid camera action")
    directory = Path(request["directory"])
    if not directory.is_absolute() or not directory.is_dir() or directory.is_symlink():
        raise ValueError("Invalid private output directory")
    image = request.get("image")
    if image is not None and (not isinstance(image, str) or not Path(image).is_absolute()):
        raise ValueError("Invalid image path")
    return request


# The pinned release index includes this explicitly untrained, unused readout.
# Consumer inference has action_output=False; all denoiser/control weights still
# load strictly. This is narrower than sample.py's general strict=False fallback.
UNUSED_ACTION_HEAD = {f"model.action_head.{layer}.{parameter}" for layer in (0, 1, 3) for parameter in ("weight", "bias")}


def generator_state(state):
    state = state.get("generator", state)
    normalized = {key.replace("model._fsdp_wrapped_module.", "model.", 1): value for key, value in state.items()}
    return {key: value for key, value in normalized.items() if key not in UNUSED_ACTION_HEAD}


class ResidentEngine:
    """Lazy, once-only model construction; independently testable without a GPU."""
    def __init__(self, factory):
        self.factory = factory
        self.runtime = None
        self.load_seconds = None
        self.load_count = 0
        self.blocks = 0

    def generate(self, request, on_phase=None):
        validate_request(request)
        if self.runtime is None:
            if on_phase:
                on_phase("loading-model")
            started = time.monotonic()
            self.runtime = self.factory()
            self.load_seconds = time.monotonic() - started
            self.load_count += 1
        if on_phase:
            on_phase("generating")
        result = self.runtime.generate(request)
        self.blocks += 1
        return {**result, "loadSeconds": self.load_seconds, "loadCount": self.load_count,
                "blocks": self.blocks, "residency": "reset-isolated-worker", "continuation": "last-clean-latent",
                "nativeKVContinuity": False}

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None

    def reset(self):
        if self.runtime is not None:
            self.runtime.reset()
        self.blocks = 0


class AstronexRuntime:
    def __init__(self, source: Path, weights: Path):
        # Imports deliberately stay inside the GPU subprocess. Gateway/CPU tests do
        # not import torch or consume GPU memory.
        os.environ["ASTRONEX_WEIGHTS"] = str(weights)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        sys.path.insert(0, str(source))
        os.chdir(source)
        import torch
        from omegaconf import OmegaConf
        # inference.memory calls torch.cuda.current_device() during import. Check
        # the device first so CPU-only hosts fail without importing CUDA-only code.
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required")
        self.torch = torch
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.set_grad_enabled(False)
        from inference import memory
        from inference.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
        from inference.pipelines.causal_diffusion_inference import CausalDiffusionInferencePipeline
        from utils.checkpoint_io import load_checkpoint
        memory.gpu = self.device
        config = OmegaConf.merge(OmegaConf.load(source / "utils/default_config.yaml"),
                                 OmegaConf.load(source / "inference/causal_consumer.yaml"))
        config.model_kwargs.model_root = str(weights)
        config.model_kwargs.model_name = "transformer"
        # Explicit upstream-supported initial load dtype avoids an unnecessary
        # second fp32 denoiser copy. Sample.py converts the full pipeline to BF16.
        config.model_kwargs.base_dtype = "bfloat16"
        config.vae_dtype = "bfloat16"
        config.sampling_steps = 8
        self.config = config
        self.pipeline = CausalDiffusionInferencePipeline(config, device=self.device)
        state = load_checkpoint(str(weights), map_location="cpu")
        normalized = generator_state(state)
        # Missing control weights must fail, never silently run an unconditioned model.
        self.pipeline.generator.load_state_dict(normalized, strict=True)
        del normalized, state
        gc.collect()
        self.pipeline.to(dtype=torch.bfloat16)
        if get_cuda_free_memory_gb(self.device) < 40:
            DynamicSwapInstaller.install_model(self.pipeline.text_encoder, device=self.device)
        else:
            self.pipeline.text_encoder.to(device=self.device)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)
        self.pipeline.vae_device = None
        self.pipeline.eval()
        # The actual upstream encoder already caches prompt embeddings. Restrict it
        # to the current prompt plus the negative prompt, evicting superseded text.
        self.pipeline.text_encoder.cache_limit = 2
        self.last_latent = None
        self.closed = False

    def generate(self, request):
        import numpy as np
        from PIL import Image
        from utils.camera_trajectory import parse_trajectory
        from utils.misc import set_seed
        from astronex.adapter import ACTIONS
        torch = self.torch
        pipeline = self.pipeline
        started = time.monotonic()
        set_seed(request["seed"])
        prompt = " ".join(request["prompt"].splitlines())
        wanted = {prompt, str(self.config.negative_prompt)}
        pipeline.text_encoder._cache = {key: value for key, value in pipeline.text_encoder._cache.items() if key in wanted}
        pipeline.sampling_steps = 4 if request["quality"] in ("low-latency", "speed") else 8
        directory = Path(request["directory"])
        with torch.inference_mode():
            initial_latent = self.last_latent
            if initial_latent is None and request.get("image"):
                with Image.open(request["image"]) as image:
                    # Matches sample.py Resize((480,832)), ToTensor, Normalize(.5).
                    image = image.convert("RGB").resize((832, 480), Image.Resampling.BILINEAR)
                    pixels = np.array(image, dtype=np.float32) / 127.5 - 1.0
                tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
                tensor = tensor.to(device=self.device, dtype=torch.bfloat16)
                initial_latent = pipeline.vae.encode_to_latent(tensor).to(device=self.device, dtype=torch.bfloat16)
                del tensor
            latent_frames = 7 if initial_latent is not None else 8
            noise = torch.randn([1, latent_frames, 48, 30, 52], device=self.device, dtype=torch.bfloat16)
            if isinstance(request["action"], dict):
                from astronex.controls import camera_poses
                poses = camera_poses(request["action"], latent_frames)
            else:
                poses = parse_trajectory(f"{ACTIONS[request['action']]}*{latent_frames}")
            intrinsics = np.repeat(np.array([[[0.5050505, 0, 0.5], [0, 0.89786756, 0.5], [0, 0, 1]]], dtype=np.float32), len(poses), axis=0)
            viewmats = torch.from_numpy(poses).unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
            cameras = torch.from_numpy(intrinsics).unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
            video, latents = pipeline.inference(noise=noise, text_prompts=[prompt], initial_latent=initial_latent,
                                               return_latents=True, viewmats=viewmats, Ks=cameras)
            if video.ndim != 5 or video.shape[0] != 1 or not 1 <= video.shape[1] <= 64 or tuple(video.shape[2:]) != (3, 480, 832):
                raise RuntimeError("Unexpected model output shape")
            if not torch.isfinite(video).all().item():
                raise RuntimeError("The model produced non-finite frames")
            # Tiny bounded continuation (one latent), not the long-horizon KV cache.
            self.last_latent = latents[:, -1:].detach().clone()
            pixels = video[0].clamp(0, 1).mul(255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            del video, latents, noise, initial_latent, viewmats, cameras
            pipeline.vae.model.clear_cache()
            torch.cuda.synchronize()
        # Single bounded RGB spool, no JPEG encode/decode in the video path.
        # HTTP snapshots encode only the requested latest frame on demand.
        raw_path = directory / "frames.rgb"
        pixels.tofile(raw_path)
        continuation = directory / "continuation.png"
        Image.fromarray(pixels[-1]).save(continuation, format="PNG", compress_level=1)
        elapsed = time.monotonic() - started
        return {"rawFrames": {"path": str(raw_path), "width": 832, "height": 480, "count": len(pixels), "pixelFormat": "rgb24"},
                "continuationPath": str(continuation), "fps": 24,
                "generatedFrames": len(pixels), "generationSeconds": elapsed,
                "generatedFPS": len(pixels) / elapsed, "samplingSteps": pipeline.sampling_steps,
                "latentFrames": latent_frames, "promptCacheEntries": len(pipeline.text_encoder._cache),
                "peakVRAMBytes": torch.cuda.max_memory_allocated(self.device)}

    def reset(self):
        """Erase all per-session latent/text/attention state, retain model weights."""
        self.last_latent = None
        self.pipeline.text_encoder._cache.clear()
        self.pipeline.vae.model.clear_cache()
        for name in self.pipeline._CACHES:
            setattr(self.pipeline, name, None)
        for name in ("_prope_mirror", "_prev_retrieved"):
            setattr(self.pipeline, name, [])
        self.pipeline._cache_keep = None
        self.pipeline._sink_promote = None
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats(self.device)
        gc.collect()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.last_latent = None
        self.pipeline.text_encoder._cache.clear()
        self.pipeline.vae.model.clear_cache()
        self.pipeline = None
        gc.collect()
        self.torch.cuda.empty_cache()


def serve(input_stream, output_stream, engine):
    """One bounded JSON line in, one response out. Never include raw exceptions."""
    def send(value):
        output_stream.write(json.dumps(value, separators=(",", ":")) + "\n")
        output_stream.flush()
    try:
        send({"type": "ready", "protocolVersion": 1})
        while True:
            line = input_stream.readline(MAX_MESSAGE + 1)
            if not line:
                break
            if len(line) > MAX_MESSAGE or not line.endswith("\n"):
                send({"type": "error", "code": "invalid-request", "error": "Resident request exceeded its size limit"})
                break
            try:
                request = json.loads(line)
                if request.get("op") == "shutdown":
                    send({"type": "closed"})
                    break
                if request.get("op") == "reset":
                    engine.reset()
                    send({"type": "reset", "sessionStateCleared": True})
                    continue
                result = engine.generate(request, on_phase=lambda phase: send({"type": "phase", "phase": phase}))
                send({"type": "result", "result": result})
            except Exception as error:
                # Class names, not exception strings, distinguish actionable cases
                # without leaking prompt text or filesystem paths.
                code = "gpu-out-of-memory" if type(error).__name__ == "OutOfMemoryError" else "inference-failed"
                send({"type": "error", "code": code, "error": "Resident inference failed; verify the GPU, pinned runtime and checkpoint. Private prompt/media logs are disabled."})
                # OOM/inference failure can leave cache state invalid. Restart a
                # new process explicitly rather than retrying unsafe model state.
                break
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    # Preserve a private protocol FD, then suppress all upstream/native logging.
    # redirect_stdout alone would not intercept C/CUDA code writing to FD 1.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    with open(os.devnull, "w") as quiet:
        os.dup2(quiet.fileno(), sys.stdout.fileno())
        os.dup2(quiet.fileno(), sys.stderr.fileno())
    worker_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(worker_root))
    engine = ResidentEngine(lambda: AstronexRuntime(args.source.resolve(), args.weights.resolve()))
    try:
        serve(sys.stdin, protocol, engine)
    finally:
        protocol.close()


if __name__ == "__main__":
    main()

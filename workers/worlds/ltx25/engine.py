"""LTX 2.5 continuous A/V generation against the exact manifest source revision.

Native chunk operators retain bounded video AND audio latents independently at
both diffusion stages. Controls change prompt embeddings at window boundaries;
they are not native camera tensors or a persistent 3D simulation.
"""

import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from .planning import (
    PromptCache,
    Timeline,
    TransformerResidency,
    budget_prompt,
    validate_request,
)

WIDTH, HEIGHT, FPS = 768, 448, 24


class Engine:
    def __init__(self, source, weights, residency="gpu"):
        import torch
        from ltx_core.text_encoders.gemma.gemma_assets import (
            GemmaAssets,
            build_gemma_hf_tokenizer,
        )
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.blocks import DiffusionStage
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode

        if residency not in ("gpu", "cpu"):
            raise ValueError("Unsupported residency")
        self.torch = torch
        self.device = torch.device("cuda")
        manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
        paths = {
            key: str(Path(weights) / value)
            for key, value in manifest["components"].items()
        }
        self.tokenizer = build_gemma_hf_tokenizer(
            GemmaAssets.load(paths["text_encoder"])
        )
        model_paths = ModelPaths.from_split(
            transformer_path=paths["transformer"],
            text_encoder_path=paths["text_encoder"],
            video_vae_path=paths["video_vae"],
            audio_vae_path=paths["audio_vae"],
        )
        # Gemma uses CPU block streaming even when the denoiser stays on GPU.
        self.pipe = DistilledPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=paths["upscaler"],
            loras=[],
            device=self.device,
            offload_mode=OffloadMode.CPU,
        )
        if residency == "gpu":
            self.pipe.stage = DiffusionStage.from_checkpoint(
                paths["transformer"],
                torch.bfloat16,
                self.device,
                loras=(),
                offload_mode=OffloadMode.NONE,
            )
        self.resident = TransformerResidency(self.pipe.stage, keep=residency == "gpu")
        if residency == "gpu":
            with torch.inference_mode(), self.pipe.stage._transformer_ctx():
                pass
        self.residency_mode = residency
        self.reset()

    def reset(self):
        decoded = getattr(self, "decoded", None)
        if decoded is not None:
            decoded.close()
        self.decoded = None
        self.request = None
        self.timeline = Timeline()
        self.cache = PromptCache()
        self.active_revision = 0
        self.windows = 0
        self.seed = None
        self.first_images = []
        self.total_frames = 0
        self.prompt_truncated = False

    def close(self):
        self.reset()
        self.resident.close()

    def _encode(self, prompt):
        (encoding,) = self.pipe.prompt_encoder([prompt], enhance_first_prompt=False)
        # The cache never keeps old embeddings in VRAM.
        return (
            encoding.video_encoding.detach().cpu(),
            encoding.audio_encoding.detach().cpu(),
        )

    def _chunks(self):
        from ltx_pipelines.chunks.layout import ChunkLayout
        from ltx_pipelines.chunks.state import Chunk
        from ltx_pipelines.utils.helpers import create_initial_av_latents

        while True:
            request = self.request
            action = validate_request(request)
            window = self.timeline.next(action["exploration"]["cadence"])
            prompt, self.prompt_truncated = budget_prompt(request, self.tokenizer)
            video_context, audio_context = self.cache.get(prompt, self._encode)
            video, audio = create_initial_av_latents(
                width=WIDTH // 2,
                height=HEIGHT // 2,
                frames=window.pixel_frames,
                fps=FPS,
                device=self.device,
                dtype=self.torch.bfloat16,
                video_scale_factors=self.pipe.stage.video_scale_factors,
            )
            self.active_revision = action["revision"]
            self.windows += 1
            yield Chunk(
                layout=ChunkLayout(**asdict(window)),
                video=video,
                audio=audio,
                video_context=video_context.to(self.device),
                audio_context=audio_context.to(self.device),
                make_video_conditionings=self.pipe._video_conditionings(
                    self.first_images if window.start_pixel_frame == 0 else [],
                    window.pixel_frames,
                    None,
                ),
            )

    def _start(self, request):
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.video_vae import AUTO_TILING
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.chunks import (
            decode_chunks,
            denoise_chunks,
            spatially_upsample_chunks,
        )
        from ltx_pipelines.utils.constants import (
            DISTILLED_SIGMAS,
            STAGE_2_DISTILLED_SIGMAS,
        )
        from ltx_pipelines.utils.helpers import (
            ensure_tiling_config,
            tiling_scale_factors_for_vae,
        )
        from ltx_pipelines.utils.types import ImageConditioningInput

        torch, pipe = self.torch, self.pipe
        self.seed = request["seed"]
        image = request.get("image")
        if image:
            self.first_images = pipe.image_conditioner.resolve_crf(
                [ImageConditioningInput(image, 0, 1.0)]
            )
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        noiser = GaussianNoiser(generator=generator)
        tiling = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=tiling_scale_factors_for_vae(
                pipe.video_decoder.checkpoint_path
            ),
            vae_checkpoint_path=pipe.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1, frames=97, height=HEIGHT, width=WIDTH, fps=FPS
            ),
            diffvae_optimization=pipe.video_decoder.diffvae_optimization,
            device=self.device,
        )
        base = denoise_chunks(
            self._chunks(),
            pipe.stage,
            sigmas=DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device),
            noiser=noiser,
            fps=FPS,
            noise_scale=1.0,
            **pipe._sampler_kwargs(self.seed, 10000),
        )
        upscaled = spatially_upsample_chunks(base, pipe.upsampler)
        refined = denoise_chunks(
            upscaled,
            pipe.stage,
            sigmas=STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device),
            noiser=noiser,
            fps=FPS,
            **pipe._sampler_kwargs(self.seed, 20000),
        )
        self.decoded = decode_chunks(
            refined,
            pipe.video_decoder,
            pipe.audio_decoder,
            fps=FPS,
            tiling_config=tiling,
            generator=generator,
            dtype=torch.bfloat16,
            keyframes=False,
        )

    def generate(self, request):
        validate_request(request)
        directory = Path(request["directory"]).resolve()
        if not directory.is_dir():
            raise ValueError("Private output directory required")
        if self.seed is not None and request["seed"] != self.seed:
            raise ValueError("Changing the seed requires a new session")
        self.request = request
        before = self.windows
        started = time.perf_counter()
        with self.torch.inference_mode():
            if self.decoded is None:
                self._start(request)
            packet = next(self.decoded)
            if before:
                # Native decode yields the previous blended seam, then this
                # window's body. Deliver them together: otherwise a body-only
                # request would consume transient mouse input without sampling.
                window = self.windows
                body = next(self.decoded)
                if (
                    self.windows != window
                    or packet.audio is None
                    or body.audio is None
                    or packet.audio.sampling_rate != body.audio.sampling_rate
                ):
                    raise RuntimeError("Pinned decoder packet contract changed")
                packet = replace(
                    packet,
                    video=self.torch.cat([packet.video, body.video], dim=0),
                    audio=replace(
                        packet.audio,
                        waveform=self.torch.cat(
                            [packet.audio.waveform, body.audio.waveform], dim=-1
                        ),
                    ),
                )
            if self.windows != before + 1:
                raise RuntimeError(
                    "Each exploration request must sample exactly one window"
                )
            self.torch.cuda.synchronize()
            sampling_seconds = (
                time.perf_counter() - started if self.windows > before else 0.0
            )
            result = write_packet(packet, directory)
        self.total_frames += result["generatedFrames"]
        result.update(
            appliedRevision=self.active_revision,
            continuity="latent-overlap",
            samplingSeconds=sampling_seconds,
            renderedWindowCount=self.windows,
            promptEncodingCount=self.cache.encodings,
            transformerLoadCount=self.resident.loads,
            residency=self.residency_mode,
            timelineFrames=self.total_frames,
            promptTruncated=self.prompt_truncated,
        )
        return result


def write_packet(packet, directory):
    """Convert the native bounded float FHWC packet and aligned audio to wire files."""
    import numpy as np
    import torch
    from PIL import Image

    frames = packet.video
    if (
        frames.ndim != 4
        or tuple(frames.shape[1:]) != (HEIGHT, WIDTH, 3)
        or not 1 <= len(frames) <= 97
        or not torch.isfinite(frames).all()
    ):
        raise ValueError("Invalid native video packet")
    frames = frames.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    raw_path = directory / "frames.rgb"
    np.ascontiguousarray(frames).tofile(raw_path)
    last_path = directory / "continuation.png"
    Image.fromarray(frames[-1]).save(last_path)
    audio = packet.audio
    if (
        audio is None
        or audio.waveform.ndim != 2
        or audio.waveform.shape[0] not in (1, 2)
        or not 8000 <= audio.sampling_rate <= 192000
    ):
        raise ValueError("Missing or invalid native audio packet")
    waveform = audio.waveform.detach().float().cpu()
    if (
        not torch.isfinite(waveform).all()
        or waveform.shape[-1] > audio.sampling_rate * 5
    ):
        raise ValueError("Invalid native audio samples")
    # FFmpeg provides a proper band-limited resampler; avoid dependency/ABI coupling
    # to a second PyTorch wheel solely to resample output.
    samples = len(frames) * (48000 // FPS)
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(audio.sampling_rate),
            "-ac",
            str(waveform.shape[0]),
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            "pipe:1",
        ],
        input=waveform.clamp(-1, 1).T.contiguous().numpy().astype("<f4").tobytes(),
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout
    # Native seam slicing can differ by a sample; each packet has an exact video
    # clock duration so rounding cannot accumulate over an indefinite session.
    if abs(len(encoded) // 4 - samples) > 4800:
        raise ValueError("Native audio and video duration mismatch")
    encoded = encoded[: samples * 4].ljust(samples * 4, b"\x00")
    audio_path = directory / "audio.pcm"
    audio_path.write_bytes(encoded)
    return {
        "rawFrames": {
            "path": str(raw_path),
            "width": WIDTH,
            "height": HEIGHT,
            "count": len(frames),
            "pixelFormat": "rgb24",
        },
        "continuationPath": str(last_path),
        "fps": FPS,
        "nativeKVContinuity": False,
        "generatedFrames": len(frames),
        "audio": {
            "path": str(audio_path),
            "sampleRate": 48000,
            "channels": 2,
            "sampleFormat": "s16le",
            "samples": samples,
        },
    }

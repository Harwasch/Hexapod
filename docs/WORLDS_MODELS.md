# Worlds model evidence and integration status

Verified from official source repositories and Hugging Face model metadata on
**2026-10-09 (UTC)**. Commit and checkpoint revisions below identify
what was inspected. Repository contents and publisher benchmarks are evidence of
an upstream release, not proof that this application has reproduced its results.

**Measured here:** CPU gateway tests, FFmpeg decoding of a synthetic test fixture,
HTTP authentication, validation, snapshot payloads, process cancellation and
cleanup. **Not measured here:** GPU inference, model quality, latency, VRAM,
character consistency, spatial consistency, audio quality or provider costs.
No GPU was provisioned and no Hexapod deployment was run.

## What can actually run through this application

The `astronex-world` adapter invokes the real, pinned upstream causal inference
pipeline from a resident GPU subprocess. It loads weights once per session, accepts
text or text plus one image, generates one eight-latent-frame block at a time, and
applies camera commands and changed prompts at the next block. Its final clean
latent conditions the next block without an image round trip; frames are encoded
directly to the selected transport encoder. Transformer KV caches still reset per block, so long-horizon
native memory and exact restoration are not claimed. This is an inference
integration requiring an operator's GPU/runtime setup, not a measured realtime
integration. Setup, API, limitations and
privacy behavior are in [the worker README](../workers/worlds/README.md).

Six world-generation adapters are implemented: Astronex, ForgeWM, Matrix-Game 3,
SANA-WM, HelixWorld Preview and LTX 2.5. The separate HunyuanWorld-Mirror reconstruction
worker turns selected images/replay frames into predicted geometry. All seven GPU
integrations remain unvalidated on GPU here; artifact readiness is not an inference
benchmark. Remaining world-model candidates stay disabled until an actual worker
advertises a supported implementation.

The `ltx-2.5` adapter runs the official two-stage distilled pipeline as a bounded,
indefinite sequence of overlapping windows, carrying video and audio latents at
both stages. It supports text/first-image input, continuous text/voice updates,
prompt-adapted walking/mouse/gamepad controls, and generated stereo audio. This
is not native action conditioning or a canonical world simulation. Default GPU
residency retains the transformer; four prompt embeddings are cached on CPU.
Only one future window is scheduled alongside playback. See the
[LTX setup and acceptance guide](../workers/worlds/ltx25/README.md). Seam quality,
long-session drift, inference speed and cost remain GPU-unverified.

## Candidate findings

| Candidate                                                                                                               | Verified upstream release and controls                                                                                                                                                                                                                                                                   | Licensing / exact blocker in this app                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [LTX 2.5](https://huggingface.co/Lightricks/LTX-2.5)                                                                    | Official 22B distilled A/V transformer, packed Gemma 4 encoder, image/text input and native overlapping video/audio latent chunk operators. Source v1.4.2 at `9ec55f9f22798a3198d9c923856824821bc3317e`; checkpoint revision `2356ce76915d6c48d313d7e8b25900e1dd3abaa8`.                                 | LTX 2.5 license and gated model access; see model card/component terms. Adapter implemented at 768×448 / 24 fps playback with prompt-adapted controls. No GPU execution, realtime, seamlessness or stable geography claim. Hosted LTX 2.3 extension endpoints are not used.                                                                                                                                                                 |
| [Astronex-World 1.0](https://github.com/Astronex-Robotics/Astronex-World)                                               | 5.35B Wan2.2-derived model; text and first-image conditioning; camera trajectory; 64D action conditioning with embodiment IDs; timed event caption append; 832×480 / 1280×704. Publisher reports 24 FPS on L20 48 GB and a 23.2 GB consumer configuration.                                               | Apache-2.0 code and model card. **Adapter implemented**, 832×480 consumer mode. No GPU validation. Resident model and last-latent continuation implemented; the upstream public call resets transformer KV caches, so a persistent KV-cache streaming loop remains unimplemented. The released action-output head is zero-initialized/untrained; do not expose predicted robot actions as useful outputs.                                   |
| [Matrix-Game 3.0](https://github.com/SkyworkAI/Matrix-Game/tree/756776d516233027631008761d4d40ddaea3e23a/Matrix-Game-3) | Released base/distilled 5B Unreal first-person checkpoints; image + text, native interactive actions, camera-aware long-horizon memory, 704×1280. Publisher reports 40 FPS; official example uses `torchrun ... generate.py`, `--interactive`, quantization and LightVAE.                                | v3 directory states Apache-2.0, matching the checkpoint; repository root says MIT, so use the v3 license. A/H GPUs tested upstream, 64 GB system RAM. Resident adapter implemented; GPU unverified. Real-world mixture and 28B checkpoints are described as forthcoming; do not list them as available. README clone/download snippets are inconsistent; the inspected monorepo path and `Skywork/Matrix-Game-3.0` checkpoint are verified. |
| [SANA-WM](https://github.com/NVlabs/Sana/blob/f9178744c096dcf2a2ea773da183e341bcbeb044/asset/docs/sana_wm.md)           | Official NVlabs/Sana release; 2.6B first stage plus refiner; image+text, explicit 6DoF pose/action sequences; bidirectional, chunk-causal and streaming checkpoints. 704×1280; streaming CLI defaults to 241 frames / 16 FPS. Publisher BF16 720p peak 47.3 GB; FP4 needs Blackwell and reports 29.4 GB. | Apache-2.0 repo and model cards. Also uses LTX-2 VAE/refiner and Gemma text encoders: component terms require review. Resident streaming-pipeline adapter implemented; GPU unverified; don't imply its memory use equals a standalone 2.6B model.                                                                                                                                                                                           |
| [HY-WorldPlay / HY-World 1.5](https://github.com/Tencent-Hunyuan/HY-WorldPlay)                                          | Official Tencent release; image/text, keyboard/mouse represented by actions+poses, context memory, bidirectional/AR/RL/distilled 480p checkpoints. Claimed 24 FPS. Do not confuse with HY-World 1.0 offline 3D generation.                                                                               | Custom Tencent license, not Apache/MIT. Requires HunyuanVideo-1.5 base, Qwen2.5-VL, ByT5 and gated FLUX.1-Redux-dev encoder. No worker, component downloads/permissions or performance verification here.                                                                                                                                                                                                                                   |
| [AlayaWorld](https://github.com/AlayaLab/AlayaWorld)                                                                    | Code and weights released; first-frame+camera+prompt, native 3D cache/history memory, chunk-boundary prompt switching; DA3 browser-reactor path and v1.1 AR/DMD variants. Upstream explicitly provides an interactive pipeline.                                                                          | Upstream says **academic research and non-commercial use only**, LTX-2 Community License, plus gated Gemma-3 and separate DA3/ViGeo. Do not treat as commercially cleared. Canonical code is `AlayaLab/AlayaWorld`; `alaya-lab/AlayaWorld` is a different older repo. No app adapter or licensed dependency setup.                                                                                                                          |
| [LingBot-World v1](https://github.com/Robbyant/lingbot-world)                                                           | Camera-conditioned and action-conditioned variants and fast checkpoints; image+text, 480p/720p. Official examples use 8 GPUs.                                                                                                                                                                            | Apache-2.0 code and base-camera model card. No app distributed-inference adapter; single-GPU FPS/VRAM not verified.                                                                                                                                                                                                                                                                                                                         |
| [LingBot-World v2 / Infinity](https://github.com/Robbyant/lingbot-world-v2)                                             | Newer candidate discovered during research: causal-fast 14B and 1.3B released, richer action vocabulary and text events; publisher claims 60 FPS. The 1.3B package includes DiT only and shares T5/VAE assets with 14B. Official scripts select 4 or 8 GPUs.                                             | **CC-BY-NC-SA-4.0**, unlike v1; non-commercial. Upstream explicitly says its deployment code is not released and points to SGLang/flashdreams alternatives. No app adapter. Model size alone does not prove inexpensive single-GPU operation.                                                                                                                                                                                               |
| [ForgeWM](https://github.com/asdfo123/ForgeWM)                                                                          | Released training code and stage0–3 / one-step / two-step / CrossFPS weights; image conditioning and keyboard/mouse or gamepad actions; Minecraft/GameFactory and distinct CrossFPS lineage. Publisher reports up to 72 FPS, 352×640, one H20, one-step student.                                         | Apache-2.0 repo and checkpoint. Needs Matrix-Game-2 base assets plus Forge checkpoints; resident four-step adapter implemented, GPU unverified. CrossFPS isn't an interchangeable multi-domain checkpoint. No free-form text/prompt-switching capability was verified; do not enable it.                                                                                                                                                    |
| [HelixWorld Preview v1](https://github.com/NoizAI/HelixWorld)                                                           | September release: image+prompt+camera actions and **native jointly generated audio/video** with viewpoint-dependent sound. The public `infer.py` CLI emits 121 frames at 768×512/24 fps with audio under `release/native`. Audio/video prompts are text descriptions, not media-file inputs.            | Apache-2.0 code and preview model card; separate Gemma encoder. Upstream recommends 80 GB VRAM (~70 GB BF16 run). Preview adapter and generated-audio WebRTC/Opus transport implemented; GPU unverified. One offline clip is generated, played and automatically paused. The full model/training release is not public. [Adapter setup and exact limits](../workers/worlds/helixworld/README.md).                                           |
| [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror)                                           | Separate reconstruction model: multiple images or sampled replay frames → predicted cameras, depth, Gaussian PLY, colored point-cloud PLY and an unfused depth mesh GLB where usable triangles exist.                                                                                                    | Actual pinned `WorldMirror` reconstruction worker implemented; GPU/geometry quality unverified. Custom Tencent community license with territorial and other restrictions; available for the user's research workflow. Model-relative scale and inferred surfaces, not surveyed geometry. [Worker setup and output limits](../workers/worlds/reconstruction/README.md).                                                                      |
| [Oasis 500M](https://github.com/etched-ai/open-oasis)                                                                   | Downloadable 500M subset; image or initial video frame, autoregressive game action conditioning. `generate.py` writes `video.mp4`.                                                                                                                                                                       | MIT code/model card; HF model is gated with automatic access. No text generation, arbitrary natural-language edits or full hosted Oasis model equivalence established. No app adapter/access acceptance here.                                                                                                                                                                                                                               |
| [DIAMOND](https://github.com/eloialonso/diamond)                                                                        | Released diffusion world model and policies for Atari100k with native discrete environment actions. Upstream has a CS:GO branch as a distinct domain.                                                                                                                                                    | MIT source. HF checkpoint card did not declare a license in the API response; review actual checkpoint terms. Domain-specific environment reset/context, not arbitrary prompt-to-world generation. No app gameplay loop integration.                                                                                                                                                                                                        |

The hardware and FPS values in this table are **publisher reports**, not product
estimates or benchmark results. No evidence supports choosing a “best” universal
world model from these heterogeneous tasks. Astronex was selected as the first
adapter because its actual text/image inference APIs, pinned dependencies, single-card
configuration and permissive release make a bounded real integration practical.

## Additional implemented adapters and customization

The `forge-wm`, `matrix-game-3` and `sana-wm` workers now invoke actual pinned
upstream pipeline classes in isolated resident subprocesses. All require one image;
Matrix and SANA also accept text, while Forge explicitly rejects unsupported text.
Native discrete actions apply to the next clip. Raw RGB frames reach the shared
transport without an intermediate JPEG encode/decode cycle. Forge emits the
published four-step Minecraft recipe (9 frames at 12 fps), Matrix uses its distilled
three-step first iteration (57 frames at its renderer's 17 fps), and SANA uses one
refined streaming block (24 emitted frames at 16 fps). Playback rates are not GPU
throughput claims. These adapters continue from the last decoded image, without
native KV continuity or exact resume. SANA camera intrinsics are explicitly
approximate pinhole values. The more aggressive one-step, quantized and benchmark
recipes are not silently substituted for the selected checkpoints.

The fifth generation adapter, `helix-world`, invokes the pinned HelixWorld Preview
CLI and preserves its jointly generated audio. It accepts one image, native camera
actions and a world prompt used for the upstream video/audio/relationship text
fields. It produces one offline 121-frame, 768×512, 24 fps clip, delivers generated
audio over WebRTC/Opus, and pauses after playback. The user must enable initially
muted audio. HTTP frame fallback has no audio. Explicit resume continues from the
last visual frame; audio continuity, persistent KV state and realtime inference
are not claimed. Model weights reload for each clip. See the
[Helix adapter README](../workers/worlds/helixworld/README.md) for its separate
Python3.11 environment, gated Gemma component, immutable checkpoint selection and
CPU transport tests. Neither audio-file nor video-file conditioning is enabled.

The separate `hunyuanworld-mirror` service calls the actual pinned `WorldMirror`
API for reconstruction. Its [worker README](../workers/worlds/reconstruction/README.md)
describes predicted-camera/depth outputs, Gaussian/point-cloud PLY and optional
unfused depth-mesh GLB exports. The application's reconstruction viewer and exports
do not establish surveyed scale, watertight surfaces, persistent object identity
or authoritative physics. GPU reconstruction quality and performance are pending.

Source-only bootstrap was executed for all three immutable revisions, exact
upstream function signatures were checked from those cloned files, and CPU tests
cover registry discovery, native action dimensions/mappings, private output paths,
source patches and shard validation. Hash-locked Python3.12 dependency closures
were resolved. Additional CPU fixture tests exercise every adapter generation
call path, real RGB output conversion, native control dimensions, SANA callback
stride/overflow and bounded prompt caches; they do not simulate model quality. Actual CUDA environment imports, FlashAttention build, weights,
GPU output, latency, memory and full images still require a configured GPU host.
See [model setup](../workers/worlds/MODEL_SETUP.md) for native/container setup and
precise limitations. No model weights or paid compute were acquired here.

The local customization CLI now prepares, explicitly runs, cancels and records
real Forge/SANA upstream full-training jobs, then structurally validates and
installs final compatible student checkpoints into isolated bundles. Selection
and disable operations write local configuration for the next worker start; they
do not deploy or swap an active model. Forge requires its four-stage pipeline;
SANA requires Stage1 and the appropriate distillation stages. Generic LoRA support
is not claimed, and the Matrix3 inference release has no verified training entry
point. Training has not run here. Structural checkpoint validation is not GPU or
quality validation. Custom datasets and training components must already be
prepared locally; auto-downloads are disabled.

## Source revision pins

| Source repository                         | Inspected commit                           |
| ----------------------------------------- | ------------------------------------------ |
| `Astronex-Robotics/Astronex-World`        | `27584bf1a89da01ef35a03b48d83bc818d2adf2b` |
| `SkyworkAI/Matrix-Game` (v3 subdirectory) | `756776d516233027631008761d4d40ddaea3e23a` |
| `NVlabs/Sana`                             | `f9178744c096dcf2a2ea773da183e341bcbeb044` |
| `Tencent-Hunyuan/HY-WorldPlay`            | `1588e1336e842b03b0a7860c654ebd7c46bb065e` |
| `AlayaLab/AlayaWorld`                     | `7e7dcfe5b82419d665ffdd458815553e459a8a39` |
| `Robbyant/lingbot-world`                  | `a43bec7f8091c83e9b30b16b912f6fc906236fa6` |
| `Robbyant/lingbot-world-v2`               | `1895d300d8ac936401689b26389f51cbd36530eb` |
| `asdfo123/ForgeWM`                        | `a922c6b42d2e1dcfdc367a27a07c0148cb8ed6d8` |
| `NoizAI/HelixWorld`                       | `5bc3e0e219aadac6c153748347ccfd6c4bf0e777` |
| `Tencent-Hunyuan/HunyuanWorld-Mirror`     | `c61300e47ce06db2f02c8e2551aae68be7ae3a0b` |
| `etched-ai/open-oasis`                    | `f59deef2c019c212bd0c5a3a5b986a51f3701847` |
| `eloialonso/diamond`                      | `5bcd1599755b4f2fae8e5e079e02f0728e174965` |

## Checkpoint existence and immutable revisions

These are API-verified model repository revisions, not downloaded or GPU-loaded
weights. IDs refer to actual releases, not hypothetical future checkpoints.
Component dependencies may require additional repositories and acceptance.

| Hugging Face model ID                                                                                             | Revision                                   | Model-card license / access                  |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------- |
| [Astronex-Lab/Astronex-World](https://huggingface.co/Astronex-Lab/Astronex-World)                                 | `f06ea330f2ce5cabef5fbf175287fe323ec1218c` | Apache-2.0; ungated                          |
| [Skywork/Matrix-Game-3.0](https://huggingface.co/Skywork/Matrix-Game-3.0)                                         | `382aa382a03cec057360755aa72c7735025770bb` | Apache-2.0; ungated                          |
| [Efficient-Large-Model/SANA-WM_streaming](https://huggingface.co/Efficient-Large-Model/SANA-WM_streaming)         | `c7694938c854b1fcc29468e9514f1ea2d0c4ba8e` | Apache-2.0; ungated                          |
| [Efficient-Large-Model/SANA-WM_bidirectional](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional) | `e96271d77398def8ebb9fc595e7c0056dc625ab7` | Apache-2.0; ungated                          |
| [tencent/HY-WorldPlay](https://huggingface.co/tencent/HY-WorldPlay)                                               | `f4c29235647707b571479a69b569e4166f9f5bf8` | Custom; ungated                              |
| [AlayaLab/AlayaWorld](https://huggingface.co/AlayaLab/AlayaWorld)                                                 | `e5b374e8e38e5e1025b521d879c5a6d2b1a129ca` | Custom; ungated                              |
| [robbyant/lingbot-world-base-cam](https://huggingface.co/robbyant/lingbot-world-base-cam)                         | `6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd` | Apache-2.0; ungated                          |
| [robbyant/lingbot-world-v2-1.3b-causal-fast](https://huggingface.co/robbyant/lingbot-world-v2-1.3b-causal-fast)   | `7e36a5f919f86cb4255cc9bfc30adb44963fbde1` | CC-BY-NC-SA-4.0; ungated                     |
| [ForgeWM/ForgeWM](https://huggingface.co/ForgeWM/ForgeWM)                                                         | `604011fe62d2f2ec1098ef7749c8306c937afa85` | Apache-2.0; ungated                          |
| [NoizAI/HelixWorld-preview](https://huggingface.co/NoizAI/HelixWorld-preview)                                     | `0f6aa9f329a118fcf0e16e0d315da026455ad8da` | Apache-2.0; upstream-pinned release, ungated |
| [Google Gemma3 Helix encoder](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)                  | `68f7ee4fbd59087436ada77ed2d62f373fdd4482` | Gemma terms; manual gate                     |
| [tencent/HunyuanWorld-Mirror](https://huggingface.co/tencent/HunyuanWorld-Mirror)                                 | `5574b7b0d5ac9d80e8a92976222370a0d20a57a0` | Custom Tencent community license; ungated    |
| [Etched/oasis-500m](https://huggingface.co/Etched/oasis-500m)                                                     | `4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c` | MIT; automatic gate                          |
| [eloialonso/diamond](https://huggingface.co/eloialonso/diamond)                                                   | `5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f` | Card license absent; ungated                 |

## Capability boundaries

- Text conditioning is not a guarantee that a semantic command changes an object.
- Camera pose conditioning does not mean output camera poses are available.
- A seed does not promise bitwise deterministic replay across devices or kernels.
- Native upstream memory is not an app snapshot/restore implementation.
- A reusable character library does not prove model-native identity references.
- Forge/SANA have an optional configured full-training service; generic model-specific LoRA compatibility is not claimed.
- A video frame is not a reliable mesh, depth map, splat or persistent object graph.
- Helix exposes generated audio over WebRTC/Opus; its HTTP frame fallback has no audio. Other generation adapters keep output audio disabled.
- Audio/video text prompts do not imply audio/video file conditioning. Helix accepts descriptions, not those media inputs.

The registry distinguishes model capabilities from application tools. Generated
frames, reconstructed surfaces and application-maintained world records do not
establish authoritative simulator physics or verified object identity.

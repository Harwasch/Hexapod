# Additional model workers

The ForgeWM, Matrix-Game 3 and SANA-WM adapters call pinned public upstream inference APIs. Source cloning, dependency resolution, CPU protocol tests and source-level interface inspection were performed locally. **GPU inference, FlashAttention compilation, complete container builds, checkpoint compatibility/quality and memory/performance remain unvalidated.** No weights were downloaded, GPU rented, job launched or service deployed.

| Worker ID       | Required input          | Native output recipe                                                                              | Continuation       |
| --------------- | ----------------------- | ------------------------------------------------------------------------------------------------- | ------------------ |
| `forge-wm`      | One image, empty prompt | Minecraft stage3 four-step student; 3 latents / 9 frames, 640×352 at upstream 12 fps              | Last decoded image |
| `matrix-game-3` | Image and text          | Distilled 5B, 3 denoising steps, first 57-frame iteration, 1280×704 at upstream renderer's 17 fps | Last decoded image |
| `sana-wm`       | Image and text          | Streaming stage1 + LTX2 refiner + causal VAE; 25 input / 24 emitted frames, 1280×704 at 16 fps    | Last decoded image |

All use next-clip discrete movement/look commands, with `balanced` as the only supported quality. These FPS values are playback rates, **not measured inference speed**. Native KV state does not persist between requests; exact resume is unavailable. All require an initial image. Forge has no text encoder; its inherited text-encoder class is an empty stub. SANA uses a documented approximate 90° pinhole camera, not an estimated camera pose. One selected model runs per worker.

Forge and Matrix may retain model weights for `WORLD_WARM_MODEL_SECONDS` (default 180; 0 disables). Reuse requires a reset acknowledgement clearing native KV/visual caches, Matrix's per-session action callback and output paths. SANA retains inference weights and the current prompt embeddings within a session; unchanged prompts avoid reloading its 12B Gemma3 encoder on every block. A prompt change evicts the previous embeddings. SANA closes the process at session end: its refiner module caches have not received a GPU isolation audit.

## Reproducible operator setup

Each directory has an immutable source/checkpoint manifest and a Python 3.12 hash-locked dependency closure. Separate interpreters prevent different upstream `wan`, `pipeline`, `utils`, Torch and Diffusers packages from colliding.

From repository root, prepare **source only**:

```bash
python workers/worlds/model_bootstrap.py forge-wm --source /opt/forge-source
python workers/worlds/model_bootstrap.py matrix-game-3 --source /opt/matrix-source
python workers/worlds/model_bootstrap.py sana-wm --source /opt/sana-source
```

Use separate new clone destinations. Matrix's runtime source is `/opt/matrix-source/Matrix-Game-3`. Bootstrap makes one exact reviewed Matrix patch: the final `exit()` becomes `return video`. The resident process replaces its random benchmark action callback with deterministic native keyboard/mouse tensors and skips the annotated MP4 writer to consume the real decoded tensor. No model architecture is replaced. Source checkout commands above were actually exercised at the pinned revisions.

For weights, run the same bootstrap against a **new** clone destination with `--download-weights --weights /models/MODEL_ID`. This is a large explicit download and uses each manifest's exact revision and file allowlist. Forge needs both Forge stage3 and Matrix2 base/CLIP/VAE assets. SANA needs streaming model, causal VAE, LTX2 transformer/connectors, Gemma3-12B and separate Gemma2-2B assets. Component licenses remain applicable; research use is not blocked by the app. Inference is forced offline and never auto-fetches a missing model.

Create a Python3.12 venv per model and install its `requirements.lock` with `uv pip install --require-hashes`. For SANA pass `--build-constraints workers/worlds/sana_wm/build-constraints.txt`: upstream mmcv1.7.2 needs setuptools with `pkg_resources`. Forge **and Matrix** additionally require `flash-attn==2.8.3` built with CUDA12.8 tools and the already installed Torch (`pip install --no-build-isolation flash-attn==2.8.3`). Matrix's model calls `flash_attention` directly despite an unused SDPA fallback elsewhere. No claim is made that these CUDA kernels compiled successfully here.

| Model   | Runtime source       | Checkpoint root       | Interpreter override |
| ------- | -------------------- | --------------------- | -------------------- |
| Forge   | `FORGEWM_SOURCE`     | `FORGEWM_WEIGHTS`     | `FORGEWM_PYTHON`     |
| Matrix3 | `MATRIX_GAME_SOURCE` | `MATRIX_GAME_WEIGHTS` | `MATRIX_GAME_PYTHON` |
| SANA    | `SANA_WM_SOURCE`     | `SANA_WM_WEIGHTS`     | `SANA_WM_PYTHON`     |

Interpreter defaults to `INFERENCE_PYTHON`. Select `WORLD_MODEL_ID`, provide `WORLD_GATEWAY_TOKEN`, then run `gateway.py` in the separate transport venv. See [worker API](README.md) for gateway authentication and session requests. `/health` checks artifact presence and all indexed checkpoint shards; it explicitly reports GPU inference as unverified.

An operator may build isolated RunPod-compatible images from repository root:

```bash
docker build -f workers/worlds/Dockerfile.models --build-arg MODEL=forge-wm -t worlds-forge:local .
docker build -f workers/worlds/Dockerfile.models --build-arg MODEL=matrix-game-3 -t worlds-matrix:local .
docker build -f workers/worlds/Dockerfile.models --build-arg MODEL=sana-wm -t worlds-sana:local .
```

For a Forge/SANA host that will explicitly run customization, add `--build-arg INCLUDE_TRAINING=1` to include its separately locked trainer dependencies. Matrix has no training image recipe. These commands are documentation, not actions taken. The shared Dockerfile supplies CUDA build tools for FlashAttention, a minimal runtime with a host compiler for Triton, an unprivileged worker, isolated inference/transport environments, port8789 and a healthcheck. Weights remain an externally prepared mounted volume. No infrastructure definition, registry push, paid GPU or deployment is performed by this setup.

## Real customization workflows

`customization.py` exposes **full upstream training**, not generic LoRA. Forge ships four training stages; SANA ships Stage1 SFT plus ODE/self-forcing distillation. Matrix3's public release provides inference only, so no training action is enabled. No verified world-specific LoRA install/scale workflow is claimed for these three models.

Forge recipe order: `stage0-sft` and `stage1-causal` can train independently; `stage2-distill` consumes Stage1; `stage3-student` consumes Stage2 and Stage0. An SFT Stage0 checkpoint cannot be dropped into the interactive Stage3 adapter. SANA order: `stage1-sft`, `ode`, `self-forcing-t43`, `self-forcing-t121`; use the exact reference config and required teacher/generator dependencies for each stage.

Use the model's separate `requirements-training.lock` (same Python3.12/build constraints). Published recipes use eight GPUs; preparation/run validation keeps that explicit. First copy the upstream reference YAML and edit its dataset/checkpoint paths to existing absolute local paths. Forge expects a prepared action LMDB and `ckpts/MG2-base` under the source tree (link the installed base bundle there). SANA needs prepared trajectory/latent/caption data and local teacher/student components; disable automatic `hf_dataset_repo` downloads and replace `hf://` checkpoints. Add `worlds_components: {gemma2: /absolute/path/to/gemma2_2b}` to the reviewed SANA config. The wrapper removes that field before invoking the real trainer and resolves its symbolic Gemma2 encoder to this immutable local bundle; otherwise the upstream trainer would look for an unavailable Hub cache. Its official example dataset is ~235GB with noncommercial Sekai terms. No datasets are downloaded by the job runner, and all child Hub calls run offline.

```bash
python workers/worlds/customization.py prepare --model forge-wm --stage stage3-student \
  --source /opt/forge-source --config /data/reviewed-stage3.yaml \
  --output /data/jobs/forge-stage3 --python /opt/forge-venv/bin/python
python workers/worlds/customization.py run /data/jobs/forge-stage3
# Only this explicit flag starts a local training process:
python workers/worlds/customization.py run /data/jobs/forge-stage3 --run
```

Preparation writes the exact argv, source revision, config digest and local job state. Dry-run inspection starts no subprocess. Actual execution uses a minimal child environment that excludes provider/gateway credentials, verifies eight visible CUDA devices, records `training.log` and completion/failed/interrupted state, and uses subprocess argument arrays without a shell. Completed stages are skipped on rerun. Failed stages retain logs; optimizer-level resume follows the upstream trainer and is not promised by this wrapper. The runner creates and records a dedicated training process group. An independent supervisor enforces the deadline and stops every rank if the control service dies, including descendants left by an exited launcher. `customization.py cancel JOB_DIR` checks its process identity, terminates all distributed ranks, and escalates after a bounded grace period. Only operator-approved source/config/interpreter paths should be exposed by a service.

Install a **final matching distilled student** as an isolated local bundle:

```bash
python workers/worlds/customization.py install --model forge-wm --stage stage3-student \
  --checkpoint /data/jobs/forge-stage3/checkpoints/checkpoint_model_004000/model.pt \
  --base /models/forge-wm --destination /models/forge-custom
python workers/worlds/customization.py enable --model forge-wm --bundle /models/forge-custom --env-file /data/forge-custom.env
# Source the fragment in the operator's next worker launch; it does not restart/deploy:
set -a
. /data/forge-custom.env
set +a
python workers/worlds/customization.py disable --model forge-wm --env-file /data/forge-custom.env
```

Installation checks a nonempty tensor state dictionary with `torch.load(weights_only=True)`, records SHA256, links base assets and copies the custom checkpoint. This is structural validation, not an inference/quality test. SANA uses the same commands with `--model sana-wm --stage self-forcing-t121`. Enable/disable writes the checkpoint-root selection for the next worker start; a running model is never silently swapped. Keep custom bundles and their base asset paths on the same mounted filesystem.

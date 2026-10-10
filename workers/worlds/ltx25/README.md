# LTX 2.5 continuous exploration worker

This adapter uses the official LTX 2.5 distilled video/audio model to extend a
first-person scene in overlapping chunks. Text and voice commands update the
next chunk's conditioning. Walking, looking and cruise controls become camera
instructions in the prompt; they are **not native action conditioning**, collision
simulation or a guarantee of stable geography. A session continues until paused,
stopped or an existing session/compute budget ends. “Infinite” describes the loop,
not unlimited compute, duration or spatial memory.

Source and checkpoint pins live in [manifest.json](manifest.json):

- Official source `Lightricks/LTX-2`, commit
  `9ec55f9f22798a3198d9c923856824821bc3317e` (v1.4.2).
- Checkpoints `Lightricks/LTX-2.5`, revision
  `2356ce76915d6c48d313d7e8b25900e1dd3abaa8`.
- Five BF16 components: distilled transformer, packed Gemma 4 text encoder,
  convolutional video VAE, audio VAE and spatial upscaler. About **71.1 GB decimal**
  of checkpoint storage before environments, caches and working memory.
- The packed text encoder embeds Gemma config, tokenizer and processor sidecars.
  Do not substitute a base Gemma directory or download another text encoder.
- Model access is gated and subject to the LTX 2.5 license and accompanying model
  terms. Accept these through the model host before preparing weights.

No Docker build, checkpoint download, cloud worker allocation or GPU inference was
performed for this implementation. The image recipe and native GPU path still
require build/import and real hardware acceptance. CPU tests exercise controls,
contracts, state, overlap bookkeeping and artifact checks; they do not establish
visual quality, realtime speed, VRAM fit, cost or seamless long-session output.

## Prepare source and checkpoints

Run from the Hexapod root. Source preparation downloads only the pinned Git source;
it verifies all Python source files used by the upstream packages without importing
Torch or executing model code.

```sh
python workers/worlds/ltx25/bootstrap.py --source /absolute/path/ltx25-source
```

Download **only the five paths in `manifest.json`** into a prepared checkpoint root,
preserving the `diffusion_models/`, `text_encoders/`, `vae/` and
`latent_upscale_models/` directories. For example, on an operator preparation host
with `huggingface-hub` installed and its token provided through the private secret
store (this command intentionally transfers approximately 71 GB):

```sh
python - <<'PY'
import json
from huggingface_hub import snapshot_download
manifest = json.load(open('workers/worlds/ltx25/manifest.json'))
snapshot_download(
    repo_id=manifest['checkpoint'], revision=manifest['checkpointRevision'],
    allow_patterns=[item['path'] for item in manifest['files']],
    local_dir='/absolute/path/ltx25-weights',
)
PY
python workers/worlds/ltx25/bootstrap.py \
  --source /absolute/path/ltx25-source \
  --weights /absolute/path/ltx25-weights --verify-assets
```

Verification hashes every checkpoint against pinned public repository metadata,
checks embedded text assets and writes `.worlds-ltx25-verified.json` in the checkpoint
root. This is an explicit disk-intensive preparation step. Readiness checks source
hashes and checkpoint sizes/mtime fingerprints against that receipt; keep the
verified directory immutable and mount it read-only. Reverify after changing or
copying files in a way that changes timestamps. A receipt is not a GPU-validation
certificate. The preparation token (`HF_TOKEN`, if used) is unnecessary at runtime
and must not be forwarded into the inference worker.

## Separate inference environment

Use Linux x86_64 with Python 3.11. [requirements.lock](requirements.lock) contains
an exact 65-package dependency closure resolved from the pinned upstream package
requirements. It is a version/URL lock, **not a complete wheel-hash lock**. The
image base and source/checkpoint hashes are independently pinned.

The reviewed source selects Torch **2.13.0+cu132**, torchvision **0.28.0+cu132**,
and torchaudio **2.11.0+cu132** from the official Torch test index. The current
torchaudio distribution is independently versioned and carries no Torch version
requirement; a GPU image import test remains required. Transformers is **5.14.1**:
upstream explicitly excludes 5.15 because its Gemma configuration breaks this
encoder. CUDA 13.2-compatible host drivers are required. Do not reuse the Astronex
CUDA 12.8 environment.

The upstream cuDNN override is preserved: **9.24.0.43** replaces Torch's metadata
pin **9.20.0.48**, which upstream reports lacks required sublibrary symbols.
Install the resolved dependency closure with `--no-deps` to preserve that override.
A raw `pip check` on this model environment reports this intentional discrepancy.
The Docker recipe runs import checks instead and checks the separate gateway
normally. No DiffVAE, NATTEN or custom CUDA kernel build is needed for this
convolutional-decoder path.

```sh
python3.11 -m venv /absolute/path/ltx25-venv
/absolute/path/ltx25-venv/bin/pip install --no-deps \
  -r workers/worlds/ltx25/requirements.lock
/absolute/path/ltx25-venv/bin/pip install --no-deps --no-build-isolation \
  /absolute/path/ltx25-source/packages/ltx-core \
  /absolute/path/ltx25-source/packages/ltx-pipelines
```

The gateway uses its own `requirements-streaming.lock` environment. Operator-run
Docker build recipe, from repository root:

```sh
docker build -f workers/worlds/ltx25/Dockerfile -t worlds-ltx25:local .
```

The image contains source and Python dependencies, never checkpoints or provider
credentials. Optional `--secret id=build_ca,src=/path/to/ca.pem` supports a build
proxy's private trust root. It runs as UID 10001; prepared assets need read access
for that user.

## Start on an already owned GPU

Put `WORLD_GATEWAY_TOKEN` in the worker's private environment and configure the
manager's same token. Defaults inside the image are:

```dotenv
WORLD_MODEL_ID=ltx-2.5
LTX25_SOURCE=/opt/ltx25
LTX25_WEIGHTS=/models/ltx-2.5
LTX25_PYTHON=/opt/ltx25-venv/bin/python
LTX25_RESIDENCY=gpu
```

`LTX25_RESIDENCY=gpu` (default) retains the **transformer only** across both
stages and subsequent chunks. Gemma uses CPU block streaming; native VAE and
upsampler wrappers build/dispose their GPU models when called. `cpu` also uses
upstream CPU block streaming for the transformer, recreating that wrapper twice
per window; it does not promise resident-model performance. Four recent prompt
embedding pairs are cached on CPU to avoid re-encoding unchanged instructions.
There is no cross-session model reuse.

The full checkpoint pack and working memory need substantial host RAM, including
when the transformer lives on GPU. The helper's default 64 GiB RAM limit is
insufficient for the full BF16 pack; select a deliberate larger limit such as
128 GiB and validate peak use. An upstream A100 80 GB / H100 recommendation is not
a measured capacity guarantee for this adapter.

Cadence presets bound work and overlap:

| Preset             | Frames per window | Carry frames | Video blend frames | Steady new playback |
| ------------------ | ----------------: | -----------: | -----------------: | ------------------: |
| Responsive         |                49 |           17 |                  8 |        1.33 seconds |
| Balanced (default) |                73 |           25 |                  8 |           2 seconds |
| Smooth             |                97 |           25 |                 16 |           3 seconds |

All play at 24 fps; these durations are **not generation-time measurements**.
Changing presets preserves the previous window's actual carry. Video and audio
latents continue independently at both stages; decoded overlap is blended and
audio crossfaded using native operators. Shorter windows trade overlap overhead
and potentially quality for more frequent command boundaries.

```sh
python scripts/worlds-local.py plan --model ltx-2.5 --name ltx-explore \
  --weights /absolute/path/ltx25-weights --memory-gib 128 --min-vram-gib 70
python scripts/worlds-local.py start --model ltx-2.5 --name ltx-explore \
  --weights /absolute/path/ltx25-weights --memory-gib 128 --min-vram-gib 70
python scripts/worlds-local.py stop --name ltx-explore
```

The helper refuses image pulls and only launches an already built local image.
The minimum-free-VRAM value is operator admission policy, not an inferred hardware
requirement. For native execution add `--mode native`, `--source`,
`--python /path/to/gateway-venv/bin/python` and
`--inference-python /path/to/ltx25-venv/bin/python`.

Register a model-specific gateway/profile on the manager using
`WORLD_MODEL_PROFILES_JSON`; see [Worlds deployment handoff](../../../docs/WORLDS_DEPLOYMENT.md).
Choosing LTX in the browser must not fall back to an Astronex worker. RunPod remains
the primary cloud provider, but this implementation does not provision or deploy
anything automatically during setup.

## Interaction and conditioning limits

The first window emits its body; following responses combine the previous blended
seam with the new body. Each request therefore samples one new window, including
transient mouse input. The gateway overlaps generation with playback and permits
only one future window, including in-flight work. Pause allows an in-flight window
to finish, preserves its carry and starts no further work until resume. Stop/delete
terminates the isolated process group and removes private frame/audio spools.

Walk converts combined keyboard/gamepad translation and looking into prompt text.
Direct keeps the camera position stationary while permitting looking and scene
changes. Cruise automatically advances while steering. Controls and speech/text
updates apply at the next available generation boundary, after any already queued
window. Revision feedback distinguishes received changes from displayed output.

The original scene plus the latest six amendments (at most 3,000 characters total)
form bounded scene context. The actual packed Gemma tokenizer budgets 1,023 tokens
plus BOS, preserving camera directions and prioritizing newer scene changes while
reserving world-description context. Long context can be shortened; the worker
reports `promptTruncated`. This is bounded prompt/latent continuity, not a durable
semantic memory of every instruction or previously visited place.

Worker telemetry includes actual transformer builds, prompt encodings, sampled
windows, sampling time, current residency and queued/applied revisions. The default
24 fps is playback cadence; measure generated FPS and command-to-visible latency
on the selected hardware. GPU residency reduces transformer reloads, but prompt
encoding and VAE/upsampler work still incur costs, and long sessions can drift.

## Acceptance before enabling a paid profile

Build the image and confirm imports; verify all checkpoint hashes and readiness;
then run a separately authorized GPU session. Measure first-chunk latency,
steady-state generation speed, peak GPU/host memory, movement/prompt adherence,
video/audio seam quality, pause/stop and failure cleanup over a sustained session.
Test several camera reversals and scene changes. Record whether generation keeps
up with 24 fps playback; that playback setting alone says nothing about inference
speed. Check browser audio/video timing and command queued/applied states under
backpressure. Compare CPU-offload and GPU-resident modes using measured cost.
No such GPU result is implied by the local test suite.

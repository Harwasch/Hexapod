# Worlds deployment handoff

Worlds uses **`/worlds.html`** and **`/api/v1/worlds`** on the existing Hexapod domain.
This document is a handoff, not deployment authorization. No deployment or paid GPU
validation was run; the user explicitly deferred paid tests. The other infrastructure
owner remains responsible for publishing web/API changes and provider infrastructure.

## Release boundary

The local workflows and CPU/transport contracts are implemented. Treat the GPU
services as **unvalidated deployment candidates** until the acceptance run below.
Keys alone are insufficient: each model needs its pinned sources, dependencies,
licensed checkpoints, tested image/runtime, suitable hardware and working network.
Control delivery latency is different from model reaction time. All integrated
models apply changes between blocks/clips, and none establishes realtime inference
or exact state restoration through this adapter.

Authentication is one trusted operator using the existing shared Hexapod API token.
Use one API replica with its durable SQLite ledger. Independent ledgers behind a
load balancer cannot safely coordinate ownership/cleanup. A shared browser origin
is not isolation between mutually untrusted products. Multi-user authorization and
resource ownership require further work before a multi-tenant release.

## Keys and their locations

Copy [worlds.env.example](worlds.env.example) to a gitignored `.env.worlds.local`,
restrict permissions and fill only the services in use. Load it into the API process
so both settings-backed configuration and optional AI endpoints see the values:

```bash
cp docs/worlds.env.example .env.worlds.local
chmod 600 .env.worlds.local
# From apps/api, after filling values:
uv run uvicorn app.main:app --env-file ../../.env.worlds.local --port 8000
```

| Key                                                | Location and purpose                                                                                                              |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `API_WRITE_TOKEN`                                  | API host; same value entered in Worlds Settings → session manager access token.                                                   |
| `WORLD_GATEWAY_TOKEN`                              | API host and inference worker; separate random token, at least 32 printable ASCII characters.                                     |
| `WORLD_RUNPOD_API_KEY`                             | API host only; managed creation/cleanup, hardware and explicit billing lookup. Not needed to connect an existing gateway.         |
| `WORLD_MODAL_TOKEN_ID`, `WORLD_MODAL_TOKEN_SECRET` | API host only, if using dedicated Worlds Sandbox provisioning. Do not reuse Earth configuration.                                  |
| `WORLD_LAMBDA_API_KEY`                             | API host only, if using Lambda provisioning.                                                                                      |
| `WORLDS_LLM_API_KEY`                               | API host only, if the optional text endpoint requires authentication.                                                             |
| `WORLDS_VISION_API_KEY`                            | API host only, if the optional multimodal endpoint requires authentication.                                                       |
| `WORLDS_IMAGE_API_KEY`                             | API host only, if the optional image generation/edit endpoint requires authentication.                                            |
| `WORLD_RECONSTRUCTION_GATEWAY_TOKEN`               | API host and separate Mirror reconstruction service.                                                                              |
| `WORLD_CUSTOMIZATION_GATEWAY_TOKEN`                | API host; same value as `WORLD_CUSTOMIZATION_TOKEN` on the training service.                                                      |
| `WORLD_TURN_SECRET`                                | Inference worker and TURN relay only. Worker issues temporary browser credentials.                                                |
| `HF_TOKEN`                                         | Artifact preparation environment only, when needed for authorized gated checkpoint downloads. Never passed to inference children. |

For hosted operation, use the platform secret store. Never use `VITE_` variables,
query-string tokens, committed filled files, or provider account keys inside workers.
The browser's API access token is session-scoped and excluded from library exports.
Non-secret URLs, model IDs, paths and templates also need configuration. Use the
[worker template](worlds.worker.env.example) separately from the API template.

## Model profiles and compute

RunPod is primary. The default profile is `astronex-world`. Configure additional
models with `WORLD_MODEL_PROFILES_JSON`; entries use **snake_case** configuration
keys, and must select the correct worker image/template or existing gateway:

```dotenv
WORLD_MODEL_PROFILES_JSON='{"astronex-world":{"runpod_template_id":"approved-astronex-template"},"forge-wm":{"runpod_template_id":"approved-forge-template"},"matrix-game-3":{"local_gateway_url":"http://127.0.0.1:8791"},"ltx-2.5":{"local_gateway_url":"http://127.0.0.1:8792"}}'
```

This example contains placeholders, not deployed resources. Worker-reported model
IDs must match the requested profile; workers cannot be reused across models.
A configured but unhealthy gateway does not trigger a surprise replacement paid pod.
Profiles may instead use `runpod_gateway_url`, `modal_gateway_url` or
`lambda_gateway_url`, with remote HTTPS required.

Managed creation requires the provider's explicit `WORLD_*_ALLOW_PROVISION=true`,
its credentials and approved runtime, `WORLD_DATA_DIR` on an absolute durable
volume, `WORLD_DATA_PERSISTENT=true`, `WORLD_MAX_WORKER_HOURLY_COST`, and the active
lifecycle supervisor. Set worker-count, startup, idle and lifetime limits. RunPod
reports actual rate after allocation; unknown or over-cap allocation is terminated.
An uncertain create response requires provider reconciliation before another attempt.

Modal uses a dedicated app/Sandbox and approved registry image, with an explicitly
labeled operator rate estimate. Lambda requires an approved image, local bootstrap
file, region, SSH keys and an HTTPS gateway URL template. See settings and provider
contracts in `apps/api/app/worlds`; neither path modifies existing Earth jobs.
CoreWeave/AWS/GCP/Azure provisioning is not implemented.

Keep the ledger across updates and API restarts. The API must remain running for
lease expiry and cleanup. Stopping a pod can retain storage charges; destroying it
is distinct. External gateways are not destroyed by this product. Settings → Worker
recovery exposes interrupted sessions, retained workers and failed cleanup.

Settings → Billing imports immutable provider-reference charge/credit rows and
reconciles matched worker periods against estimates. Unmatched rows remain visible;
currencies are never silently converted. Explicit RunPod lookups use its official
read-only daily billing endpoint for an owned worker and bounded time interval.
Provider-reported totals are displayed separately from imported ledger totals and
estimates; no periodic account queries or automatic invoice reconciliation runs.

## Prepare worker services

| Service               | Recipe / guide                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Astronex              | `workers/worlds/astronex/Dockerfile`; [gateway guide](../workers/worlds/README.md)                                                         |
| Forge, Matrix3, SANA  | `workers/worlds/Dockerfile.models` with selected `MODEL`; [setup](../workers/worlds/MODEL_SETUP.md)                                        |
| Helix                 | `workers/worlds/helixworld/Dockerfile`; [Python3.11 setup](../workers/worlds/helixworld/README.md)                                         |
| Mirror reconstruction | `workers/worlds/reconstruction/Dockerfile`; [service guide](../workers/worlds/reconstruction/README.md)                                    |
| Forge/SANA training   | [customization service](../workers/worlds/CUSTOMIZATION_SERVICE.md), existing eight-GPU runtime and reviewed local dataset/config profiles |

The recipes pin source and base revisions, separate conflicting model environments,
run unprivileged and omit weights from images. Astronex/Forge/Matrix/SANA/Mirror
use hash-locked dependency closures; Helix currently pins direct official requirements,
so its resolved transitive environment must also be recorded during release build.
Build/import-test final sources, prepare the exact authorized checkpoints separately,
mount weights read-only, and pin the tested image digest in each provider profile.
Some models require gated Gemma components; CUDA kernels may need compilation.
No image publication, checkpoint acquisition or provider allocation was performed.

For already owned local hardware, [worlds-local.py](../scripts/worlds-local.md)
supports inventory, launch planning, native execution and prebuilt Docker images.
It does not build/pull images or allocate cloud resources. Its configurable VRAM
floor is admission control, not a measured model memory requirement.

Inference gateway signaling must use HTTPS remotely. RunPod's HTTP proxy does not
relay WebRTC media. Configure reachable ICE, normally TURN behind NAT, and test
from the actual browser network. `WORLD_TURN_CREDENTIAL_TTL_SECONDS` must exceed
the maximum session/worker lifetime plus a margin. Relay egress contributes cost.

The default encoder is software VP8. `WORLD_VIDEO_ENCODER=nvenc` requests H264
hardware encoding with observable software-H264 fallback; `h264-software` selects
the software path. NVIDIA containers need `compute,utility,video` driver capabilities.
Raw RGB avoids JPEG transfers on the primary path, but still crosses CPU memory;
this is not GPU zero-copy. Default fixed bitrate is 3 Mbps, with bounded queues.
Hardware encode quality, fallback behavior and bandwidth need target-GPU validation.

Warm model reuse is bounded by `WORLD_WARM_MODEL_SECONDS` (180 default; 0 disables).
Astronex/Forge/Matrix reset session-private state before reuse. SANA retains weights
and current prompt embeddings within a session but ends its process at session end.
Helix reloads its offline CLI per clip. LTX 2.5 keeps its transformer resident
within a session by default, caches four prompt embeddings on CPU, carries
video/audio latents at both diffusion stages, and schedules one future window.
Its CPU-offload alternative rebuilds the transformer wrapper each stage/window.
LTX ends its process at session end; there is no cross-session warm reuse.
Its [separate runtime recipe](../workers/worlds/ltx25/README.md) requires pinned
CUDA 13.2/Python 3.11 dependencies and about 71.1 GB of prepared checkpoints. Keeping a cloud pod alive still costs money;
warm reuse is not a guarantee of lower total cost for every workload.

## Deferred GPU and endpoint acceptance

Use an explicitly approved test worker and budget; **no paid run is authorized by
this handoff**. Record hardware, image digest, checkpoint/component revisions,
resolved environment and provider rate with the results.

1. Build/import-test each final runtime. Verify pinned source, all checkpoint shards,
   auth rejection and model capabilities through read-only preflight.
2. Run first block and several continuations per model with supported inputs. Verify
   prompt/action adherence, pause/resume, visual branching, audio for Helix, actual
   recordings and clean cancellation. Forge rejects text; all added adapters need
   a starting image.
3. Measure cold/warm first-frame latency, generation FPS versus delivery FPS, control
   reaction latency, VRAM, load counts and dollars per generated minute. Verify
   session reset prevents prompt/image/cache leakage. Compare software and NVENC.
4. Verify real browser WebRTC and acknowledged controls through target-network TURN;
   force ICE failure and check fallback. Confirm Helix audio cannot silently become
   a frame-only success. Test disconnected browser and API-restart cleanup against
   the provider's actual resource list and charges.
5. Exercise the configured vision/image endpoints with consent: character synthesis,
   scene-aware voice, periodic director, highlights and identity evaluation. Run
   shared-input model comparisons and human review; heuristic scores are not truth.
6. Run Mirror from replay through downloaded PLY/GLB and browser viewing. Review
   geometry, camera consistency, artifact expiry and cancellation. Do not interpret
   self-consistency diagnostics as metric reconstruction accuracy.
7. Validate one approved Forge/SANA training recipe, rank/deadline cleanup, final
   bundle installation, enable/disable and next-worker inference. Generic LoRA and
   native video/audio-file conditioning remain unsupported by these adapters.
8. After infrastructure-owner publication, confirm existing Earth routes and APIs,
   Worlds bundle isolation, TLS, auth and limits on the shared domain.

## Local verification — 2026-10-09

- 87 Worlds frontend unit tests passed; source/test/node TypeScript and repository ESLint passed.
- Eight shared product/bundle regression tests passed.
- Generated OpenAPI/TypeScript contracts regenerated and checked.
- 158 Worlds API tests plus 11 standalone auth/config checks passed with mocked providers.
- 107 worker CPU tests passed, including real codec/transport and subprocess cleanup.
- Mirror and local-launcher CPU tests passed; model source-only bootstraps and pinned
  upstream call contracts were checked without checkpoint downloads.
- Production build and bundle checks passed: Worlds initial JavaScript is 153 kB gzip,
  with no Cesium dependency.
- 13 Chromium product/storage/experiment/voice/clip checks passed, including actual
  local recording export/decode and offline-clip planned-control ordering.
- A separate billing browser smoke check passed: estimates/imported/provider totals
  remain separate, period changes clear stale results, and lookups never allocate.
- Real Chromium received synthetic H264 video and Opus audio through local TURN/TCP,
  with acknowledged controls and cleanup. This proves software transport, not inference.
- Final model container builds/imports, NVENC hardware execution, live AI endpoints,
  paid provider operations and GPU inference/training/reconstruction were **not run**.

An earlier Astronex image build/dependency check passed before final source changes;
execution was blocked by the workspace Docker VFS storage limit. It is not evidence
that the final images build or run. The new recipes remain unbuilt. Do not call the
application GPU-validated, realtime or cost-efficient until the deferred measurements pass.

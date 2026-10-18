# Worlds explorer

Worlds is a separate, local-first neural-world product inside Hexapod. It shares
hosting and the authenticated API proxy with Earth, while keeping its own entry
point, browser library, worker ledger and model services. There is no conventional
physics engine or canonical simulated world behind the generated video.

## Open locally

Use the repository dependency setup, then run `pnpm dev` and `pnpm dev:api` in
separate terminals. Open `http://localhost:5173/worlds.html`. The interface preview
and local library work without cloud keys. **Interaction preview is procedural
art, not AI output or a model benchmark.** Failed inference never silently becomes
a preview.

Projects, reference media, characters, scenes, recordings and experiments live in
IndexedDB (`hexapod-worlds-local`), separately from the Earth catalog. Export a
library backup for durable ownership: browser storage can be cleared or evicted,
and localhost and a hosted domain have different libraries.

## Implemented workflows

| Vision                     | Implementation                                                                                                                                                                                       | Boundary                                                                                                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Multiple world models      | Real upstream adapters for Astronex, ForgeWM, Matrix-Game 3, SANA-WM, HelixWorld and LTX 2.5; source/checkpoint pins and worker recipes                                                              | CPU/source contracts checked; checkpoint execution and CUDA installation unvalidated. Other catalog candidates remain research entries.                                   |
| Continuous interaction     | Bidirectional WebRTC controls/video; combined keys, mouse-look and analog gamepad on Astronex; LTX latent-carry A/V exploration and prompt-adapted navigation; bounded queues and generation metrics | Controls apply between generation blocks. Native KV memory across blocks and realtime response are not established. Helix generates one offline clip per explicit resume. |
| Immediate voice commands   | Opt-in direct submission, continuous listening, optional scene-aware interpretation                                                                                                                  | Browser speech recognition may use the browser vendor's service. Scene interpretation requires a configured vision endpoint and consent.                                  |
| Reusable characters        | Multiple references, character analysis, portrait/full-body/scene synthesis, qualitative identity evaluation, prepared scene images                                                                  | Image preparation conditions the starting frame; integrated world models have no native identity injection or identity guarantee.                                         |
| AI-directed games          | Frame-observing director, narrative events, objective/progress assessments, evidence and replay highlights                                                                                           | Vision-model judgments are heuristic; events use supported prompts/actions and do not enforce an external game simulation.                                                |
| Replay to 3D               | Pinned HunyuanWorld-Mirror engine, authenticated job service, Gaussian/point PLY, optional depth-surface GLB, camera/diagnostic exports and browser viewing                                          | GPU reconstruction not run. Scale is model-relative; mesh is unfused and not guaranteed watertight.                                                                       |
| Model comparisons          | Sequential fixed-seed experiments, shared references/trajectories, actual frame/video capture, synchronized comparison and human ratings                                                             | Requires an approved runtime for each model and an explicit budget. Quality is not inferred from catalog claims.                                                          |
| Model Lab evaluations      | Action discovery from declared controls, automated trajectories, motion/sharpness/revisit evidence, optional character evaluation                                                                    | Image-space measurements and vision scores are evidence, not ground-truth spatial or identity accuracy.                                                                   |
| Conditioning/customization | Multi-image scene synthesis, local video-to-starting-frame preparation, Forge/SANA full training, cancellation and custom bundle selection                                                           | No native audio/video-file conditioning or generic LoRA loader in the selected adapters. Training uses upstream eight-GPU recipes and prepared datasets.                  |
| Efficient streaming        | Raw RGB pipeline, optional NVENC/H264, software fallback, bounded buffering and reset-acknowledged warm model reuse                                                                                  | Not GPU zero-copy. NVENC hardware behavior, latency/VRAM and dollar efficiency await measurements.                                                                        |
| Compute orchestration      | RunPod primary; separate Modal Sandbox and Lambda provisioning; local GPU discovery/launcher; quotas, cleanup and usage                                                                              | Operator-approved images/templates required. CoreWeave/AWS/GCP/Azure provisioning remains unimplemented.                                                                  |
| Billing                    | Durable charge/credit import and reconciliation, unmatched rows, per-currency totals; explicit read-only RunPod billing lookup                                                                       | Provider-reported daily charges, imported invoices and rate estimates remain separate. No automatic invoice ingestion or universal billing API.                           |

Research-license models are included by default as requested. Model/component
license terms remain documented; this is not a claim of commercial clearance.
See [model evidence](WORLDS_MODELS.md), [additional model setup](../workers/worlds/MODEL_SETUP.md),
[HelixWorld](../workers/worlds/helixworld/README.md),
[LTX 2.5 continuous exploration](../workers/worlds/ltx25/README.md), and
[reconstruction](../workers/worlds/reconstruction/README.md).

## Same-domain integration

Vite emits `worlds.html` into the same web build as the existing products. Existing
hosting can serve `/worlds.html`; its existing `/api/*` proxy reaches
`/api/v1/worlds/*`. The shared header links to Worlds. Worlds loads independently
of Cesium; the reconstruction viewer loads its 3D dependencies on demand.

**No deployment, DNS, hosting workflow, or existing Earth Modal infrastructure
changes were run.** The existing infrastructure owner remains responsible for
publishing. This is a trusted single-operator application using the shared API
write token, not multi-user tenancy. Sharing a domain also shares a browser
security origin; separate storage names are not a security boundary.

## Configuration and operation

See [deployment handoff](WORLDS_DEPLOYMENT.md) and the separate
[API](worlds.env.example) / [worker](worlds.worker.env.example) environment templates
for keys, model profiles, worker setup and deferred GPU acceptance. No cloud key
is needed for offline authoring. The local GPU helper's `inventory` and `plan`
commands do not allocate compute: [launcher guide](../scripts/worlds-local.md).

RunPod is primary. An existing per-model gateway takes precedence over managed
provisioning. Settings shows readiness, available hardware, quotes, worker recovery,
usage and billing without creating compute. Paid creation requires a configured
provider, approved model profile, durable ledger, running lifecycle supervisor,
rate ceiling and explicit user start. Experiments run one model at a time and
stop when pricing/allocation is uncertain.

Managed sessions renew leases; the API expires abandoned sessions and destroys
idle or over-lifetime owned workers. Defaults: 90-second session lease, 5-minute
idle allowance, 15-minute startup allowance, one-hour worker lifetime and one
concurrent worker. These require a running API and its durable ledger; they are
not a provider-enforced dollar budget. Stopping compute may retain billable storage;
destroy is a separate operation. Externally owned gateways must be stopped by their
owner. Unknown create outcomes require provider reconciliation before retrying.

Warm reuse preserves model weights only after clearing session state and receiving
a reset acknowledgement. Astronex, Forge and Matrix support this bounded reuse;
SANA closes at session end pending refiner-cache isolation validation. Helix's
public CLI reloads for each clip. No model provides exact latent save/restore.

## Optional intelligence and customization

Configure OpenAI-compatible services on the API host:

- `WORLDS_LLM_*`: prompt enhancement and game premises via chat completions.
- `WORLDS_VISION_*`: frame-aware voice commands, director, character analysis and
  evaluation, replay highlight selection via multimodal chat completions.
- `WORLDS_IMAGE_*`: image generations and reference-image edits for character and
  scene synthesis.

Each service uses `BASE_URL`, `MODEL` and optional `API_KEY`. Remote endpoints
require HTTPS. Each upload workflow explains the media being sent and requires
consent. Local services can keep those operations on the user's machine. When an
endpoint is absent, its feature stays unavailable or clearly offers a local writing
guide; there are no fabricated AI results. Replay clip export is local and needs
no model endpoint.

Model Lab → Customization connects to a separately configured training service.
Its operator supplies local profiles (model, stage, dataset/config/source/runtime
paths). Preparation displays the exact job; training requires explicit consent,
operator enablement and eight existing GPUs. Jobs have deadlines and process-group
cleanup, including service crashes. Final compatible students can be installed and
selected for the **next worker start**. Selection does not restart or deploy a worker.
See [customization setup](../workers/worlds/CUSTOMIZATION_SERVICE.md).

## Privacy and verification

Provider secrets stay on the API host. Worker children receive only allowed runtime
configuration, not provider or gateway credentials. Inputs are bounded, stripped of
image metadata and held in private job/session directories; model loading is offline.
Remote inference necessarily receives its required prompt, media and controls.
Recordings remain local until explicitly sent to an enabled service.

A saved scene is a **visual checkpoint**, not an exact neural-state resume. Helix
outputs generated audio over WebRTC/Opus; browsers start muted, and HTTP image
fallback cannot carry that audio. Reconstruction remains separate from Earth:
generated scenes are never published as measured locations.

Local verification and remaining release checks are recorded in
[WORLDS_DEPLOYMENT.md](WORLDS_DEPLOYMENT.md). No paid GPU tests, model checkpoint
downloads or deployments were performed. GPU quality, adherence, memory, latency,
reconstruction fidelity and cost efficiency remain pending by user request. Setting
API keys alone does not install models, create their runtimes or establish realtime
performance.

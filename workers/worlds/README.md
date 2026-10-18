# Worlds GPU worker

This worker is a separate product runtime, reached through the Worlds gateway in
the shared Hexapod application. It has its own worker credentials and session data
and does not use Earth's scene storage or existing Modal GPU jobs. RunPod **GPU Pods**
are the preferred persistent session host; the same image can run on local Docker.
No provisioning, image build, weight download or deployment runs automatically.

## Implemented adapters

`WORLD_MODEL_ID` selects one model per worker. All adapters call pinned upstream
inference code; GPU inference, performance and real-time behavior remain
unverified. Capabilities describe the implemented adapter, not every claim in a
publisher's research release.

| Model            | Conditioning                         | Output / execution                                         |
| ---------------- | ------------------------------------ | ---------------------------------------------------------- |
| `astronex-world` | Text; optional image                 | 832×480 video; resident block generation                   |
| `forge-wm`       | Required image; no text conditioning | 640×352 video; resident chunks                             |
| `matrix-game-3`  | Text and required image              | 1280×704 video; resident chunks                            |
| `sana-wm`        | Text and required image              | 1280×704 video; resident chunks                            |
| `helix-world`    | Text and required image              | 768×512 video with generated audio; explicit offline clips |

The released HelixWorld CLI accepts video/audio _text descriptions_, not uploaded
video or audio files. It generates one 121-frame clip with joint audio, plays it,
and pauses. Explicit resume generates another clip from its last visual frame;
audio continuity and live inference are not claimed. See the
[Helix adapter](helixworld/README.md) for its separate Python 3.11 environment,
licensed text encoder, pinned container recipe and validation limits. Forge,
Matrix and SANA use [the multi-model setup](MODEL_SETUP.md).

Astronex calls `CausalDiffusionInferencePipeline.inference` at its pinned revision.
Its subprocess loads the denoiser, text encoder and VAE once per **resident process**,
then generates one eight-latent-frame block per request. The final clean latent
conditions the next block directly. Frames are passed as a bounded RGB spool,
without an intermediate JPEG or MP4 round-trip. WebRTC encodes RGB directly;
authenticated HTTP frames and snapshots encode the latest frame to JPEG on demand.

The pinned Astronex call resets transformer KV caches between requests. Matrix
and SANA likewise lack a validated public API for extending those caches with
new live inputs across calls. These adapters do not claim persistent native world
memory or exact scene restoration. Astronex, Forge and Matrix can retain model
weights across clean sessions only after an acknowledged reset clears all
adapter-specific latent, prompt, VAE and attention state. Failed or interrupted
inference destroys the process. `WORLD_WARM_MODEL_SECONDS` defaults to 180, accepts
0–900, and zero disables reuse. At most one idle process is retained; idle expiry
and worker shutdown destroy it. SANA does not reuse its process across sessions
because its cache isolation is not fully audited. Helix reloads its offline CLI
for each explicitly requested clip. See [runtime details](RUNTIME.md).

Astronex accepts text and at most one PNG/JPEG/WebP image. Directional controls
include `forward`, `backward`, `left`, `right`, `look_left`, `look_right`,
`look_up`, `look_down`, `up`, `down` and `stop`. Held controls compose; opposing
keys cancel, and each expires two seconds after its last renewal. `move` (alias
`analog`) accepts normalized forward/right/up/yaw/pitch axes. `look` accepts bounded
mouse deltas. All are sampled at the **next generation block**, not during one
already being computed. No undocumented semantic-action mapping is invented.
`quality` and `balanced` request eight upstream sampling steps; `low-latency`
requests four. These settings have not been benchmarked on a GPU here.

The old Astronex CLI remains a diagnostic fallback. It needs `lmdb==2.2.0` because
upstream's dataset module imports LMDB; the resident path does not require it.

## Explicit local setup

The following setup is for Astronex. Use Python 3.12 on Linux with CUDA
12.8-compatible NVIDIA drivers. A 48 GB GPU is recommended as an untested
operational margin. Upstream reports 23.2 GB with its
consumer configuration, not including competing GPU workloads.

From the repository root, use separate inference and transport environments:

```bash
python3 -m venv /tmp/worlds-inference-venv
python3 -m venv /tmp/worlds-gateway-venv
/tmp/worlds-inference-venv/bin/pip install --require-hashes -r workers/worlds/astronex/requirements.lock
/tmp/worlds-gateway-venv/bin/pip install -r workers/worlds/requirements-streaming.lock
/tmp/worlds-inference-venv/bin/python workers/worlds/astronex/bootstrap.py --source /tmp/worlds-astronex
/tmp/worlds-inference-venv/bin/python workers/worlds/astronex/bootstrap.py --weights /tmp/worlds-astronex-weights
export INFERENCE_PYTHON=/tmp/worlds-inference-venv/bin/python
export ASTRONEX_SOURCE=/tmp/worlds-astronex
export ASTRONEX_WEIGHTS=/tmp/worlds-astronex-weights
export WORLD_GATEWAY_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(40))')"
/tmp/worlds-gateway-venv/bin/python workers/worlds/gateway.py
```

The Docker image creates both environments. Native installations also need a C
compiler and Python headers for the upstream Triton JIT. The token stays
server-side in the app's gateway configuration. The worker listens
on `127.0.0.1:8789` by default. Point the session service at that address using its
local gateway setting and the same token. `/health` distinguishes missing source
or checkpoint components; source readiness does not establish a successful GPU
inference run. `WORLD_IDLE_SECONDS` defaults to 300. A disconnected session is
terminated and its temporary media removed when its idle timeout expires.

For an operator-managed RunPod Pod or local Docker installation, the image recipe
is `workers/worlds/astronex/Dockerfile`, with the repository root as build context.
The build intentionally does not download weights. Mount the prepared checkpoint
read-only at `/models/astronex`, pass `WORLD_GATEWAY_TOKEN` at runtime, and expose
port 8789 through the provider's TLS proxy. Set RunPod's port to `8789/http` and use
its HTTPS proxy URL. A container's internal HTTP socket must not be exposed directly
to the Internet. See the [deployment handoff](../../docs/WORLDS_DEPLOYMENT.md) for
validation status and acceptance criteria. The base image is pinned by digest;
inference dependencies are hash-locked. Ubuntu package repositories can still
change, so pin the final tested image digest in your RunPod template.

WebRTC defaults to aiortc's software video encoder and uses an ordered acknowledged
`world-controls-v1` DataChannel. `WORLD_VIDEO_ENCODER=nvenc` opts into a real H264
NVENC attempt on the first generated frame, with permanent libx264 fallback for
that connection if initialization or encoding fails. `h264-software` selects the
same packet path using libx264. Telemetry reports the actual encoder and a
sanitized fallback reason; NVENC hardware performance is unverified. Packet modes
use a bounded fixed bitrate (`WORLD_VIDEO_BITRATE`, default 3 Mbps) and periodic
IDR frames, rather than aiortc's adaptive bitrate/keyframe feedback. Expose NVIDIA
`compute,utility,video` driver capabilities for NVENC; see [runtime details](RUNTIME.md).

There is one observer per session and a bounded latest-frame buffer. Models that
actually generate audio additionally publish a send-only Opus track; no client
camera or microphone media is accepted. Audio is initially muted in the browser.
HTTP frame fallback is video only. Remote pods behind NAT usually need your own TURN
relay: configure `WORLD_TURN_URLS`, `WORLD_TURN_SECRET` (at least 32 characters),
and optionally `WORLD_ICE_TRANSPORT_POLICY=relay`. The worker mints short-lived
TURN credentials; the shared secret stays on the worker. No third-party STUN
service is used by default. See the [worker environment template](../../docs/worlds.worker.env.example).
`WORLD_TURN_CREDENTIAL_TTL_SECONDS` defaults to 3,900 seconds and must exceed the
API's hard worker lifetime plus a connection margin.
RunPod's HTTP proxy does not relay WebRTC media; test ICE separately. The browser
falls back to authenticated frame polling and HTTP controls when needed.

## Gateway contract (version 1)

Every request requires `Authorization: Bearer <WORLD_GATEWAY_TOKEN>`. Remote
transport must use HTTPS. The browser connects through the application gateway;
this worker deliberately has no permissive CORS policy or query-string tokens.

| Request                               | Response                                                                            |
| ------------------------------------- | ----------------------------------------------------------------------------------- |
| `GET /health`                         | Status and actually implemented model/capabilities                                  |
| `POST /sessions`                      | Session with supplied anonymous ID, effective seed and loading status               |
| `GET /sessions/{id}`                  | `loading`, `generating`, `playing`, `paused` or `error`; measured delivery counters |
| `POST /sessions/{id}/actions`         | `native`, `prompt`, `pause`, `resume`; unsupported controls return 422              |
| `GET /sessions/{id}/frame`            | JPEG, `X-Frame-Index`; 204 until actual frames exist                                |
| `POST /sessions/{id}/snapshot`        | Visual checkpoint data URL and `state:null`; 409 before a frame exists              |
| `GET /sessions/{id}/transport-config` | ICE configuration with short-lived relay credentials                                |
| `POST /sessions/{id}/offer`           | SDP answer for video, supported generated audio and acknowledged controls           |
| `POST /sessions/{id}/heartbeat`       | Renew activity only when `active:true`; passive reads do not renew it               |
| `DELETE /sessions/{id}`               | 204 after state reset or process termination, and private temporary-file deletion   |

Create request:

```json
{
  "id": "anonymous-session-123",
  "modelId": "astronex-world",
  "input": { "prompt": "A woodland path", "images": [] },
  "seed": 42,
  "quality": "balanced",
  "resolution": "832x480"
}
```

Omit the seed or use null to generate one; the effective seed is returned. An image
must be a data URL, not a URL or local path. The worker rejects audio, video, multiple
images and character identity inputs. A character description may be deliberately
included in the user's main prompt, without implying reference-identity support.
Limits: one GPU session, 16 MiB JSON body, 10 MiB decoded image, 8000-character
prompt. The uploaded image is decoded and metadata stripped before inference.

Action request:

```json
{ "type": "native", "action": "forward" }
```

```json
{ "type": "prompt", "prompt": "The woodland path becomes covered in snow" }
```

Control semantics follow the selected adapter. Astronex combines held actions and
analog axes, and consumes accumulated mouse movement at the next block; releasing
one key does not cancel other keys. Other discrete adapters sample their pending
action at the next chunk. Helix accepts `values:{"planned":true}` for a one-clip
camera plan that survives a paused wait, is not erased by key release, and is
consumed once. Other adapters reject the `planned` field. Prompt updates replace
the pending caption only when the adapter supports prompt switching.

Actions do not interrupt inference already in progress. Pause halts video/audio
playback and the next generation; an in-flight GPU call finishes. Delete cancels
in-flight work. A clean completed resident call may instead be reset and returned
to the bounded warm pool. Frame and JSON responses use `Cache-Control: no-store`.

## Privacy and cleanup

Provider credentials and gateway tokens are excluded from the inference child's
environment. Weight caches are offline during inference and telemetry is disabled.
Inputs use anonymous private temporary directories. Upstream stdout/stderr are
suppressed because upstream logs can include prompts. Errors returned to the app
are bounded, generic messages; raw tracebacks are not sent. Output paths are
validated against each session's private directory. Only the
current block and continuation image remain on disk. Generated PCM is bounded in
memory and removed at teardown. Session deletion and idle cleanup clear private
latent, prompt, VAE and attention state before any warm reuse; only model weights
are retained. A reset that fails or is not acknowledged destroys the process.
Crash cleanup relies on ephemeral container/tmpfs storage; no secure erasure of GPU
or storage hardware is claimed. Remote inference necessarily receives its prompt,
image and controls.

## Verification and benchmarking

```bash
python -m pytest workers/worlds/tests -q
```

Use a CPU test environment with the streaming requirements plus pytest, NumPy
and PyYAML. Tests cover authentication, capability/input rejection, real synthetic
FFmpeg decoding, raw-frame boundaries, warm reset isolation, cancellation, planned
controls, software H264 fallback and genuine WebRTC video/Opus transport. They do
not generate an AI world. `tests/browser_transport.mjs` verifies the actual browser
transport with an explicitly synthetic RGB fixture; `WORLD_TEST_AUDIO=1` adds PCM,
and `WORLD_VIDEO_ENCODER=h264-software` selects H264. Local TURN/TCP can be used
without modifying a managed browser's networking policy.

After an operator provisions a GPU and starts this worker, this command performs
actual inference on the existing worker and records client delivery measurements:

```bash
python workers/worlds/benchmark.py --url http://127.0.0.1:8789 --seconds 120 --output /tmp/worlds-benchmark.json
```

For read-only readiness (no session creation or provisioning), use
`python workers/worlds/preflight.py --url http://127.0.0.1:8789`. Optional
`--check-config` checks exported process variables; it does not read dotenv files.
Readiness alone never proves GPU inference. The benchmark's optional
`--hourly-price` is a labeled rate-based estimate, not provider billing data.
Missing frames, model errors, and failed cleanup produce a nonzero result.

No GPU benchmark result is checked in. Presentation rates are adapter-specific
(Astronex/Helix 24, Forge 12, Matrix 17 and SANA 16 FPS); they are not measured
generation speeds. Worker `generatedFPS` and `generationSeconds` describe the
latest chunk, while `totalGeneratedFrames` and `totalGenerationSeconds` support
cumulative generation throughput. Resident generation timing includes sampling,
VAE decoding and RGB transfer/spooling, excluding separately measured initial
loading. Helix's offline CLI timing includes model loading because that boundary
is not separately measured.

Resident protocol `loadSeconds`/`loadCount` appear in session telemetry as
`modelLoadSeconds`/`modelLoadCount`. They describe the resident **process lifetime**:
one load is expected per process, and its original loading measurement remains
when clean sessions reuse it. A warm session does not imply a fresh model load.
Unavailable loading metrics remain unknown. Benchmarks separately report sampled
client delivery and time to first frame, avoiding confusion with presentation
rates or polling frequency. `peakVRAMBytes` comes from the actual GPU only; no
fallback value is invented. Snapshots remain visual checkpoints because native
latent/cache state is not exported.

The source pin has been fetched and its APIs inspected. CPU tests verify resident
reuse, cache lifetime ownership, phase messages, cancellation during startup and
inference, timeout teardown and strict output-directory boundaries. Completing
GPU validation still requires the pinned checkpoint, a CUDA-capable worker, a
successful container build/import check and a real first-block/continuation run.
Check kernel compilation, VRAM, frame validity and action response before treating
this as an operationally validated realtime service.

See [model evidence](../../docs/WORLDS_MODELS.md) for the research registry,
licenses, checkpoint pins and exact integration blockers for other models.

## LTX 2.5 continuous exploration

Use the [separate pinned LTX 2.5 worker](ltx25/README.md) for continuous overlapping
video/audio windows, live text/speech changes and prompt-adapted navigation. It
uses model id `ltx-2.5`, its own Python/CUDA environment and model-specific profile.
The inference integration and CPU contracts are implemented; GPU acceptance,
seam quality, hardware fit and cost measurements remain pending.

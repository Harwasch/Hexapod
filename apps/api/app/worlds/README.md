# Worlds API

This is a separate product namespace under `/api/v1/worlds`. It does not create
Hexapod sites, assets, reconstruction jobs, or Modal pipeline invocations.
The existing `API_WRITE_TOKEN` protects **every** endpoint, including reads of
sessions, capabilities and frames. Local development retains the API's existing
no-token mode; production requires a token.

## Connect an existing worker

Set these variables on the API process, never in a `VITE_` variable:

```sh
WORLD_DATA_DIR=./data/worlds
WORLD_LOCAL_GATEWAY_URL=http://127.0.0.1:8789
WORLD_GATEWAY_TOKEN=your-private-worker-token-at-least-32-characters
```

For hosted workers, set `WORLD_RUNPOD_GATEWAY_URL` or
`WORLD_MODAL_GATEWAY_URL` to a compatible worker's HTTPS origin. Hosted workers
require `WORLD_GATEWAY_TOKEN`. HTTP is accepted only for the local provider on
localhost, 127.0.0.1 or ::1. URLs cannot contain embedded credentials, query
strings or fragments. Gateway paths may have a prefix.

An attached worker remains externally managed: ending a model session removes
its temporary inference data, but does **not** stop provider billing. Deleting an
attached worker connection returns `detached` and explicitly says compute continues.
Stop externally managed compute in its owning runtime/provider console.

## RunPod primary provider

Actual provisioning is opt-in, with an operator-selected approved template:

```sh
API_WRITE_TOKEN=your-private-api-access-token
WORLD_DATA_DIR=/absolute/persistent/volume/worlds
WORLD_DATA_PERSISTENT=true
WORLD_MAX_WORKER_HOURLY_COST=2.00
WORLD_MAX_MANAGED_WORKERS=1
WORLD_RUNPOD_API_KEY=your-server-only-provider-key
WORLD_RUNPOD_ALLOW_PROVISION=true
WORLD_RUNPOD_TEMPLATE_ID=approved-worlds-gateway-template
WORLD_RUNPOD_GPU_TYPE=NVIDIA L40S
WORLD_RUNPOD_PORT=8789
WORLD_GATEWAY_TOKEN=your-private-worker-token-at-least-32-characters
```

These are configuration examples, not executed operations. No deployment,
provisioning, GPU benchmark, or live inference was run while implementing this
module. The template must already contain the compatible worker and its model
configuration. If `WORLD_RUNPOD_GATEWAY_URL` is also set, the existing gateway
connection takes precedence and no pod is created.

The adapter uses the official [RunPod v1 Pods REST API](https://docs.runpod.io/api-reference/pods/POST/pods),
verified against current documentation on 2026-10-08 (America/Los_Angeles). It creates one secure-cloud
GPU pod using the approved template, checks status, stops, and deletes only IDs
created by this product. RunPod account credentials are never sent to the worker;
only its separate gateway token is supplied to the container. Pod responses are
filtered to prevent provider account IDs, environment variables and internal URLs
from reaching the browser. Hourly cost comes from RunPod's returned `costPerHr`;
missing pricing stays unknown. Stopping a pod can continue storage charges;
deletion is the final teardown operation.

## Leases, recovery and cost controls

The API lifespan starts a server-owned cleanup loop when a Worlds provider is
configured or an existing Worlds ledger needs recovery. An unused Worlds module
creates no metadata directory and starts no cleanup thread. New managed
provisioning fails closed unless the lifecycle loop is alive, API authentication
is configured, the persistent-volume assertion is explicit, an hourly ceiling is
set, and the gateway token is at least 32 characters. The approved GPU setting is
an allowlist of one; browser requests cannot select a more expensive GPU.

Default limits (all durations are seconds):

| Variable | Default | Effect |
| --- | ---: | --- |
| `WORLD_SESSION_LEASE_SECONDS` | 90 | Explicit active heartbeat extends a session lease. |
| `WORLD_WORKER_IDLE_SECONDS` | 300 | Managed worker deleted after no active user renewal. |
| `WORLD_WORKER_STARTUP_SECONDS` | 900 | Initial model-startup allowance before idle deletion. |
| `WORLD_WORKER_MAX_LIFETIME_SECONDS` | 3600 | Absolute worker deadline, including retained/paused time. |
| `WORLD_REAPER_INTERVAL_SECONDS` | 15 | Cleanup scan interval. |
| `WORLD_MAX_MANAGED_WORKERS` | 1 | Atomic fleet reservation; unknown/stopped workers count. |
| `WORLD_MAX_WORKER_HOURLY_COST` | unset | Explicit USD/hour ceiling required for provisioning. |
| `WORLD_LIFECYCLE_ENABLED` | true | Disabling cleanup also disables managed provisioning. |

Send `POST /sessions/{id}/heartbeat` with `{active: true}` every 20 seconds while
there is real user engagement. Invisible/idle pages should send `active: false`
or stop heartbeats. Merely reading frames, status, readiness, or catalog does not
extend either lease. Heartbeats are forwarded to the inference worker too.
Expired leases cannot be revived. Heartbeat/cleanup decisions use SQLite write
transactions, preventing a stale scan from terminating a concurrently renewed
lease. At session expiry, the gateway receives DELETE to stop inference and
remove temporary files. Managed compute is subsequently deleted at its idle
or absolute lifetime deadline even if the browser has disappeared or the gateway
has crashed. Attached local/Modal/RunPod compute is never deleted by this reaper.

On restart the loop scans the durable ledger, processes expired leases and
retries pending teardown. Each managed create stores a random operation name
**before** its sole provider request. If the API loses that response, recovery
searches RunPod for that exact name and deletes a late allocation. It never
blindly repeats provisioning. Unresolved operations continue to reserve capacity;
a missing match does not establish that no billable worker was created. Duplicate
matches or permanently inaccessible provider APIs require console reconciliation.
No account-wide worker is modified unless its ID was created here or recovered
from an exact persisted operation name.

RunPod's v1 create API does not supply an atomic maximum-price condition. The API
checks the returned actual hourly rate immediately; unknown or over-limit pricing
causes immediate deletion before returning a usable worker. This can incur a short
charge. The maximum concurrent count and lifetime bound exposure while the API
and provider are reachable, but they are **not a provider-enforced billing cap**.
Provider outages, deletion failures, stopped-pod storage and manager downtime can
continue charges. Stop and detach decisions are reserved in the same database transaction as
session creation, so a new session cannot enter while a worker is shutting down.
Unconfirmed stops remain `stopping` and retry until acknowledged or superseded
by the idle/lifetime deletion deadline. Failed cleanup remains visible as `terminating`/`cleanupError`
and retries; it never releases reserved capacity or claims billing ended until
provider deletion succeeds. Use the provider's independent billing controls for
an account-level financial guarantee.

For production, run one API replica with the SQLite ledger on an actual persistent
volume. Multiple API processes sharing that same local volume are coordinated by
SQLite; isolated replicas with independent volumes are unsupported and would each
have a separate budget. `WORLD_DATA_PERSISTENT=true` is an operator assertion, not
filesystem durability detection. Do not point it at an ephemeral container root.
Retaining a worker only provides warm reuse until the idle/absolute deadline;
it never disables server limits.

Configuration loads repository-root `.env` by default (consistent with Hexapod's
API). To isolate this product use `WORLD_ENV_FILE=/absolute/path/worlds.env`.
Process environment overrides file values. An explicit missing file fails startup
instead of silently using defaults. The dedicated file supplies `WORLD_*` settings;
`API_WRITE_TOKEN` still belongs in the API process environment or Hexapod root
`.env`, because it protects the entire API. Restart the API to apply changes.
Keep provider/gateway keys in server configuration only; `/readiness` exposes
limits and health metadata, never their values.

## Browser contract

All JSON field names use camelCase. Routes are relative to `/api/v1/worlds`:

- `GET /providers`: `{providers, primaryProvider: "runpod"}`. Configuration is not a
  GPU availability claim; unimplemented providers remain unavailable.
- `GET /catalog`: live model declarations from configured gateways plus providers.
- `GET /readiness`: configuration, normalized model health and active lifecycle
  limits. Probes run concurrently with bounded timeouts and never provision.
- `POST /workers`: `{provider: "runpod" | "local" | "modal", gpuTypeId?}`.
- `GET /workers`, `GET /workers/{id}`, `POST /workers/{id}/stop`,
  `DELETE /workers/{id}`. Prefer ending sessions before teardown. Managed pod
  deletion also terminates sessions when a crashed gateway cannot acknowledge
  their shutdown; external detach still requires session cleanup.
- `POST /sessions`: `{workerId, modelId, prompt, seed?, quality?, resolution?, inputs?}`.
  Quality is `quality`, `balanced`, `low-latency` or `max-fps`. Inputs accepts
  `images` (image data URLs), `video` (video data URL), `characterDescription`.
  Media in the envelope is bounded to 6 MiB; the HTTP body is capped at 7 MiB
  before JSON decoding. Other control requests are capped at 128 KiB, including
  chunked requests. Individual model workers may support less.
- `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}`.
- `POST /sessions/{id}/heartbeat`: `{active: boolean}`; returns `leaseExpiresAt`,
  `hardDeadline` (Unix seconds), `heartbeatIntervalSeconds`, ID and status.
- `GET /sessions/{id}/transport-config`: worker STUN/TURN configuration for the
  authenticated session; only WebRTC fields and ephemeral credentials are relayed.
- `POST /sessions/{id}/actions`: `{type: "native" | "prompt" | "semantic" | "pause" |
  "resume", action?, prompt?, values?}`. The worker decides capability support.
- `GET /sessions/{id}/frame`: authenticated latest JPEG/PNG/WebP; 204 means no frame
  yet. `Cache-Control: no-store`. Optional telemetry headers are forwarded only
  when the worker actually supplies measurements.
- `POST /sessions/{id}/offer`: `{type: "offer", sdp}`. Returns the worker's WebRTC
  answer or an honest 501 if the worker has no WebRTC implementation.
- `POST /sessions/{id}/snapshot`: returns worker-reported resume fidelity and
  optional state. A visual checkpoint is never labeled an exact latent restore.

Gateway transport preserves the same paths except the API namespace and create
session translates `prompt` and `inputs` to `input: {prompt, ...inputs}`. Session
IDs originate on the API and must be preserved by the worker. See
`workers/worlds` for the worker implementation. The browser must fetch frames with
its authorization header and construct a blob URL; never put API tokens in URLs.

Metadata is in a separate owner-readable SQLite file. It records identifiers,
lifecycle state and measured status, never prompts, input media or cloud keys.
The ledger survives API restarts. A SQLite transaction atomically reserves each
ready worker for one active or unresolved session, including across API processes;
a competing browser receives 409 before any inference request. Generation failures preserve a session record
for explicit cleanup; deleting a missing upstream session is idempotent. Bodies
from failed upstream calls are never exposed in client errors. Redirects are
not followed, responses are bounded, and HTTP operations have finite timeouts.
The lifespan owns a bounded HTTP connection pool; authorization remains per request
and is never persisted as a shared client default.

### Approved model profiles and additional compute

The manager now selects a model runtime before allocation. `POST /workers` accepts
`modelId` (default `astronex-world`), stores it, and rejects sessions for a different
model. Set `WORLD_MODEL_PROFILES_JSON` to a JSON object of up to eight approved
profiles, for example:

```dotenv
WORLD_MODEL_PROFILES_JSON='{"matrix-game-3":{"runpod_template_id":"approved-matrix-template","runpod_gpu_type":"NVIDIA H100 80GB HBM3"},"forge-wm":{"local_gateway_url":"http://127.0.0.1:8790"},"sana-wm":{"modal_image":"registry.example/worlds/sana@sha256:APPROVED_DIGEST","modal_gpu_type":"H100","modal_hourly_cost":5.0}}'
```

Profiles accept only the typed fields in `ModelProfile`: RunPod template/GPU,
each provider's gateway URL, Modal registry image/GPU/hourly estimate, and Lambda
instance type/image/bootstrap file. Credentials and lifecycle limits remain
server-global. A second model never inherits the first model's template, image,
or gateway URL. Each approved image must implement the worker entrypoint at
`/opt/worlds/gateway.py` with interpreter `/opt/gateway-venv/bin/python`; RunPod
uses the operator template's entrypoint. Both receive `WORLD_MODEL_ID`. Model
installation, license acceptance and private weight availability remain operator
responsibilities. No worker image or weights are downloaded by the API manager.

Read-only `GET /providers/{provider}/hardware` supplies inventory and quotes.
RunPod uses its documented GraphQL `gpuTypes.lowestPrice` query restricted to
one secure-cloud GPU. Lambda uses `GET /instance-types`; its `memory_gib` is
system RAM, so the response correctly leaves GPU `memoryGB` unknown. Modal has
no supported runtime price inventory here: the operator sets a conservative
hourly estimate. `POST /providers/{provider}/quote` accepts `modelId`, optional
`gpuTypeId` and `durationMinutes`; it never allocates. Every rate declares
`pricingSource`: `provider-quote`, `operator-estimate`, or `unavailable`.

`GET /usage` reports persistent worker start/end time, observed rate and elapsed
compute estimate. Worker status reads reconcile provider-reported current rates.
`billedCostUSD` is always null because these APIs do not supply an authoritative
invoice total through the implemented contracts. Estimates omit disks, network,
tax, rounding, stopped-instance storage and historical rate changes. Do not treat
an experiment budget as a provider-enforced billing cap. The provider invoice is
authoritative. RunPod retains its post-allocation price guard; quotes do not
reserve inventory or guarantee the allocated price.

#### Lambda Cloud

The dedicated Lambda v1 REST adapter supports live inventory, launch, status,
exact-operation-name recovery, and termination. It never SSHs into instances.
Configure, in addition to the common persistent ledger/auth/lease/cost controls:

```dotenv
WORLD_LAMBDA_ALLOW_PROVISION=false
WORLD_LAMBDA_API_KEY=server-only-dedicated-key
WORLD_LAMBDA_REGION=us-east-1
WORLD_LAMBDA_INSTANCE_TYPE=gpu_1x_a100_sxm4
WORLD_LAMBDA_SSH_KEY_NAMES='["worlds-operator-key"]'
WORLD_LAMBDA_IMAGE_ID=operator-approved-image-id
WORLD_LAMBDA_BOOTSTRAP_FILE=/absolute/operator-owned/worlds-cloud-init.yaml
WORLD_LAMBDA_GATEWAY_TEMPLATE=https://{instance_id}.worlds.example.com
```

Provisioning requires an operator-approved image, bootstrap file, and HTTPS
routing template. The operator's bootstrap/routing service must register the
instance's provider ID, route DNS to it, terminate TLS, run the selected gateway,
and arrange model weights. Only `{instance_id}` is replaced in the URL; the
manager never constructs an insecure public-IP HTTP endpoint. The bootstrap
supports `{{WORLD_MODEL_ID}}` and `{{WORLD_GATEWAY_TOKEN_BASE64}}`; decode the
latter into the gateway environment using an operator-owned script, keeping it
out of logs. The cloud API key is never sent to the instance. Bootstrap content
is not exposed to browsers or stored in the metadata ledger. Keep the gateway
URL template and provider credentials available until all owned instances have
been terminated, including after a manager restart.

The launch preflight checks live regional capacity and rate against the server
ceiling. There is no stop/suspend operation: shutting down a Lambda guest does
not end billing. The manager requires explicit termination and keeps cleanup
pending until Lambda reports terminated/preempted or the instance is absent.
Asynchronous termination, provider outages and API downtime may extend charges.
An externally operated Lambda gateway can instead use
`WORLD_LAMBDA_GATEWAY_URL`; detach does not stop its billing.

#### Standalone Modal Sandboxes

This adapter uses the API's existing `modal` optional dependency and a dedicated
Sandbox app, not the Earth application's Modal deployment or shared credentials:

```dotenv
WORLD_MODAL_ALLOW_PROVISION=false
WORLD_MODAL_TOKEN_ID=dedicated-worlds-token-id
WORLD_MODAL_TOKEN_SECRET=dedicated-worlds-token-secret
WORLD_MODAL_APP_NAME=hexapod-worlds
WORLD_MODAL_IMAGE=registry.example/worlds/astronex@sha256:APPROVED_DIGEST
WORLD_MODAL_GPU_TYPE=L40S
WORLD_MODAL_HOURLY_COST=operator-conservative-total-hourly-estimate
```

The approved registry image must already contain gateway/model dependencies and
have access to its licensed weights without embedding secrets in image layers.
The first authorized allocation looks up or creates the dedicated app, starts
one Sandbox with an encrypted gateway port, and injects only the gateway bearer
secret. The provider-enforced timeout equals `WORLD_WORKER_MAX_LIFETIME_SECONDS`.
The hourly estimate must fit the global ceiling; include CPU/memory/GPU in that
operator estimate. It is not verified Modal billing. No suspension is supported;
terminate to stop the Sandbox. Allocation identity is persisted before tunnel
discovery so slow tunnel startup cannot lose ownership. The same lease, atomic
fleet capacity reservation, unknown-outcome recovery and idle cleanup apply.

Implementation references (verified against official APIs):
[RunPod GraphQL specification](https://graphql-spec.runpod.io/),
[Lambda Cloud API](https://docs.lambda.ai/api/cloud), and
[Modal Sandbox SDK](https://modal.com/docs/sdk/py/latest/Sandbox).
No provider allocation, image publication or deployment is performed by tests.

### Actual billing import and reconciliation

The authenticated `POST /worlds/billing/import` accepts up to 100 structured
invoice/export lines per request (the existing 128 KiB body limit also applies):

```json
{
  "rows": [{
    "provider": "runpod",
    "workerId": "manager-worker-uuid",
    "reference": "invoice-2026-01/line-001",
    "source": "runpod-2026-01.csv",
    "amount": "1.250000",
    "currency": "USD",
    "periodStart": "2026-01-01T00:00:00Z",
    "periodEnd": "2026-01-01T00:30:00Z",
    "kind": "charge",
    "category": "compute",
    "description": "GPU compute charge from exported invoice"
  }]
}
```

Supply exactly one `workerId` or `providerId`. Matching requires an existing
recorded worker of that same provider, and provider IDs must match uniquely.
Unmatched lines remain visible without modifying any worker. Matching is
re-evaluated on reads, so a previously unknown provisioning ID can reconcile
later. `reference` must identify an individual invoice line uniquely for that
provider; use an invoice ID plus line ID when necessary. Re-importing identical
content is idempotent. Conflicting content rejects the whole batch with 409;
previous records are immutable. Correct mistakes with a separate adjustment or
credit reference. Credits require `kind: credit` and a negative amount; charges
must be nonnegative. Amounts must be finite, at most one billion in magnitude,
and at most six decimal places. Currency codes are three uppercase letters;
currencies are never implicitly converted. Periods require time zones, positive
duration and at most 366 days. The ledger is bounded at 10,000 imported lines.

`GET /worlds/billing?offset=0&limit=100` returns paginated records, complete
per-currency totals, matched worker totals, and unmatched counts. Imported
amounts are explicitly `sourceType: operator-imported`; the server cannot verify
an uploaded invoice's authenticity or completeness. USD compute deltas compare
only the union of imported compute periods with each worker's overlapping
elapsed-rate estimate. Overlapping lines and credits do not duplicate estimated
hours. Storage/network/tax/other categories and non-USD charges do not enter that
compute delta. Rate changes remain unmeasured. `GET /usage` now also exposes
imported actual USD amounts and compute deltas separately from estimates.

For live provider data, the authenticated read-only
`GET /worlds/billing/runpod/{workerId}?startTime=...&endTime=...` queries only one
owned managed RunPod worker, for at most 31 days. The official
[RunPod public OpenAPI](https://rest.runpod.io/v1/openapi.json) specifies
`GET /billing/pods`, grouping by `podId`, daily buckets, and USD `amount` plus
optional billed milliseconds/disk GB. The adapter returns only validated fields
for the requested worker. Results carry `sourceType: provider-reported` and are
never added to imported invoice totals, avoiding double counting. Empty provider
responses mean unknown, not zero. Provider buckets can include storage charges
and are not classified as compute-only or as a final invoice. This endpoint
requires an explicit user refresh; no account-wide query or scheduled billing
poll runs automatically. Modal/Lambda exports can use the same local import
workflow; no unverified account-billing API is called for those providers.

# Real Worlds reconstruction worker

This independent service reconstructs generated frames using the actual public
[HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror)
feed-forward model. It predicts camera poses, depth, point clouds and 3D Gaussians.
The implementation calls `WorldMirror.from_pretrained(local_directory)` and
`model(views={"img": images}, cond_flags=[0,0,0], is_inference=True)` from the pinned
source; there is no synthetic geometry fallback or Earth site publication.

Outputs are a standard Gaussian `scene.ply`, colored `points.ply`, and a GLB mesh
triangulated from predicted depth surfaces where usable triangles exist. The mesh
is unfused and not watertight. Predictions have model-relative scale, not surveyed
coordinates or guaranteed physical accuracy. The existing Worlds browser viewer
opens PLY output; GLB is downloadable for external tools.

Source revision, exact checkpoint revision and license URLs are in `manifest.json`.
This is a restricted research candidate enabled by default. Its custom Tencent
community license has territorial and other restrictions; it is not an unrestricted
commercial license. The engine has **not** been run on a GPU in this workspace and
no reconstruction quality or performance result is claimed. CPU tests validate
protocols, bounded lifecycle, exports and self-consistency math only.

## Prepare locally or on an already owned GPU machine

Use a separate Python environment. The Linux PyPI Torch wheel includes its matching
CUDA runtime; the host still needs a compatible NVIDIA driver. Upstream recommends
Torch 2.4 / CUDA 12.4, and a matching official Torch 2.4 CUDA 12.4 installation may also
be used. GPU memory depends strongly on the frame count; the default is 12 views at
518 pixels. These settings are conservative bounds, not a measured VRAM guarantee.

```sh
python3 -m venv .venv-reconstruction
.venv-reconstruction/bin/pip install --require-hashes -r workers/worlds/reconstruction/requirements.lock
.venv-reconstruction/bin/python workers/worlds/reconstruction/bootstrap.py \
  --source /absolute/path/mirror-source --weights /absolute/path/mirror-weights
```

Bootstrap downloads several GB of public weights only when `--weights` is supplied.
It never provisions compute, runs inference or deploys anything. Source/checkpoint
versions are immutable pins. Inference runs offline from those prepared files and
never downloads model code or weights during a job.

Configure the worker process (use a secret environment file rather than shell
history for the token):

```dotenv
WORLD_RECONSTRUCTION_GATEWAY_TOKEN=<random-token-at-least-32-characters>
WORLD_RECONSTRUCTION_SOURCE=/absolute/path/mirror-source
WORLD_RECONSTRUCTION_WEIGHTS=/absolute/path/mirror-weights
WORLD_RECONSTRUCTION_DATA_DIR=/absolute/private/path/reconstruction-jobs
WORLD_RECONSTRUCTION_PUBLIC_URL=http://127.0.0.1:8790
WORLD_RECONSTRUCTION_BROWSER_ORIGINS=http://localhost:5173
WORLD_RECONSTRUCTION_MAX_FRAMES=12
WORLD_RECONSTRUCTION_TARGET_SIZE=518
WORLD_RECONSTRUCTION_RETENTION_SECONDS=3600
WORLD_RECONSTRUCTION_TIMEOUT_SECONDS=1800
WORLD_BIND=127.0.0.1
PORT=8790
```

Start with `.venv-reconstruction/bin/python workers/worlds/reconstruction/server.py`.
For a separately configured RunPod worker, set its public HTTPS proxy URL and the
actual Worlds browser origin. The service does not choose, allocate or bill a GPU.
Set the API server's `WORLD_RECONSTRUCTION_GATEWAY_URL` to this worker URL and use
the same `WORLD_RECONSTRUCTION_GATEWAY_TOKEN`. The normal `/worlds` **3D Worlds** UI
then submits selected sources and opens the returned geometry.

## Optional local container

`Dockerfile` pins its NVIDIA base by digest, installs hash-locked dependencies,
fetches only pinned source, runs as UID 10001 and checks authenticated artifact
readiness without inference. Its Dockerfile-specific ignore file excludes weights,
secrets and tests from the repository-root build context. The image has not been
built or GPU-tested in this workspace.

```sh
docker build -f workers/worlds/reconstruction/Dockerfile -t worlds-mirror:local .
python scripts/worlds-local.py plan --model hunyuanworld-mirror --name mirror \
  --weights /absolute/path/mirror-weights
# After preparing the token in your environment and choosing an owned GPU:
python scripts/worlds-local.py start --model hunyuanworld-mirror --name mirror \
  --weights /absolute/path/mirror-weights --gpu 0
python scripts/worlds-local.py stop --name mirror
```

The helper never builds or pulls an image. It forces the local Unix Docker socket,
checks NVIDIA inventory, binds HTTP only on loopback, mounts prepared weights
read-only, drops capabilities and bounds RAM/processes/temporary storage. Weights
must be readable by the container's UID 10001. Mirror job data uses a private
temporary `/data` mount and disappears when the container
is removed. Supply `WORLD_RECONSTRUCTION_BROWSER_ORIGINS` for browser downloads.
The native Python option and other model commands are documented in
[`scripts/worlds-local.md`](../../../scripts/worlds-local.md).

## Protocol and privacy

Authenticated routes use the gateway bearer token:

- `GET /health`: pinned artifact readiness, never an assertion of successful GPU
  inference. No model is loaded by health checks.
- `POST /reconstructions`: bounded multipart `files` and optional `metadata`; queue
  a job. Inputs are 2–120 images or one replay video. Large image sets and videos
  are uniformly sampled across their full source sequence to the configured cap.
  Video needs a decodable frame count/fps; otherwise extract frames in the browser.
- `GET /reconstructions/{id}`: actual queued/running/completed/failed state, stage,
  progress and only artifacts actually produced.
- `DELETE /reconstructions/{id}`: cancel/kill its GPU subprocess, wait for it to
  stop, remove inputs/partial outputs/results, and only then confirm deletion.

The wire limit is 65 MiB including form overhead; total media is 64 MiB. Two jobs may
be pending, only one invokes a GPU at once, and at most 32 retained jobs are allowed.
Each inference subprocess has a bounded runtime, minimal environment without
provider/API credentials, offline model loading, no stdin and no media/prompt logs.
The child stops if its parent service dies. Completed/failed jobs delete inputs
immediately; results expire at the retention deadline. An interrupted job is marked
failed on restart and its temporary media is removed. Default retention is one hour.

Result downloads use short-lived HMAC URLs scoped to job+artifact, never the worker
bearer token. Configure CORS only for the actual browser origins. Downloads expire
in five minutes and can be refreshed with job status. Expiry/deletion invalidates
existing links. Do not send signed links through analytics or third-party logging.

## Honest diagnostics

Signed `diagnosticsUrl` and `camerasUrl` outputs accompany each completed job.
Diagnostics include source sharpness/image-change heuristics, finite-point fraction,
camera spread, adjacent-view projected overlap and relative depth residual. These
measure prediction self-consistency, **not ground-truth reconstruction accuracy**.
Camera matrices use OpenCV camera-to-world convention and model-relative units.
Source images use Mirror's aspect-preserving square-pad preprocessing with a
longest-edge bound; the inferred intrinsics refer to these prepared images.
Current input pose metadata does not specify standardized coordinate/intrinsic
conventions, so it is preserved in local source bundles but not used as a model prior.

The Gaussian exporter writes sigmoid logits, log scales, normalized quaternions
and SH color coefficients for standard 3DGS/Spark readers. Outputs are capped at one
million finite points/Gaussians. Invalid or empty model predictions fail; no dummy
point is exported. Mesh triangles crossing large local depth discontinuities are
excluded. No SPZ conversion or iterative Gaussian refinement is claimed.

Run CPU tests without weights or GPUs:

```sh
apps/api/.venv/bin/python -m pytest workers/worlds/reconstruction/tests -q
```

# Local Worlds workers

`worlds-local.py` is a Linux operator helper for already owned NVIDIA GPUs. It
never builds/pulls images, downloads weights, provisions cloud resources or sends
inference requests. `inventory` reads `nvidia-smi`; `plan` only prints a launch
command and required environment variable names. Neither allocates a GPU.

```sh
python scripts/worlds-local.py inventory
python scripts/worlds-local.py plan --model astronex-world --name explore \
  --weights /absolute/path/prepared-astronex-weights
```

Docker mode supports these prebuilt local tags:

| Model               | Local image           | Build definition                                      |
| ------------------- | --------------------- | ----------------------------------------------------- |
| astronex-world      | worlds-astronex:local | workers/worlds/astronex/Dockerfile                    |
| forge-wm            | worlds-forge:local    | workers/worlds/Dockerfile.models, MODEL=forge-wm      |
| matrix-game-3       | worlds-matrix:local   | workers/worlds/Dockerfile.models, MODEL=matrix-game-3 |
| sana-wm             | worlds-sana:local     | workers/worlds/Dockerfile.models, MODEL=sana-wm       |
| ltx-2.5             | worlds-ltx25:local    | workers/worlds/ltx25/Dockerfile                       |
| helix-world         | worlds-helix:local    | workers/worlds/helixworld/Dockerfile                  |
| hunyuanworld-mirror | worlds-mirror:local   | workers/worlds/reconstruction/Dockerfile              |

Prepare the pinned model source/checkpoint/dependencies first using each model's
documentation. Put a random bearer token of at least 32 printable ASCII characters
in `WORLD_GATEWAY_TOKEN`, or `WORLD_RECONSTRUCTION_GATEWAY_TOKEN` for Mirror.
Use your existing private environment configuration; the helper does not write,
print or generate credentials. The manager API must use the same gateway token.

```sh
python scripts/worlds-local.py start --model forge-wm --name explore \
  --weights /absolute/path/prepared-forge-weights --gpu 0 --port 8789
python scripts/worlds-local.py stop --name explore
```

Docker must expose the local `/var/run/docker.sock` and support NVIDIA Container
Toolkit. The helper pins that Unix socket so a configured remote Docker context
cannot turn this command into a deployment. It refuses implicit image pulls and
uses only the fixed image tags above. Containers bind `127.0.0.1`, run with no added
capabilities, a read-only filesystem, eight CPU cores, bounded memory, 512 PIDs,
2 GiB shared memory and bounded temporary mounts. Weights mount read-only.
`--memory-gib` defaults to 64 and accepts 4–512. No arbitrary Docker arguments or
shell commands are accepted.

`--min-vram-gib` defaults to 8 (range 1–192) and checks **free** GPU memory before
launch. This is a configurable admission floor, not a model capacity guarantee or
performance measurement. Models may need much more memory or multiple GPUs; this
single-GPU helper refuses no model based on an invented performance claim. Check
the model documentation and run authenticated preflight before a real session.

For native execution, use the installed gateway interpreter and, when separate,
the model inference interpreter. Mirror uses its own single inference environment:

```sh
python scripts/worlds-local.py start --mode native --model hunyuanworld-mirror \
  --name mirror --source /absolute/path/mirror-source \
  --weights /absolute/path/mirror-weights \
  --python /absolute/path/.venv-reconstruction/bin/python --gpu 0
```

Other native models accept `--inference-python /absolute/path/model-venv/bin/python`.
For native HelixWorld, use
`--model helix-world --source /absolute/path/helix --weights /absolute/path/helix/models`
and `--inference-python` pointing to its prepared Python 3.11 environment. The
upstream CLI requires that exact source-relative model tree. Its reported 80 GB
GPU recommendation is an upstream requirement, not a workspace measurement;
choose a suitable `--min-vram-gib` floor explicitly.
Docker HelixWorld mounts the prepared tree at `/opt/helixworld/models` using the
`worlds-helix:local` image. Its recipe remains unbuilt and GPU-unvalidated in this
workspace; the helper still requires an already built local image and never pulls.
For LTX 2.5, follow [its setup guide](../workers/worlds/ltx25/README.md) and
use `--model ltx-2.5` with its separate CUDA 13.2 Python 3.11 environment.
The full BF16 pack exceeds the default 64 GiB host RAM allowance before working
memory; explicitly set an appropriate `--memory-gib` value (for example 128)
and a measured GPU admission floor. The helper forwards `LTX25_RESIDENCY`,
`WORLD_VIDEO_ENCODER` and `WORLD_VIDEO_BITRATE` when explicitly configured.
Native workers bind loopback, receive only selected runtime environment variables,
run in their own process group and have private local logs. They use the model's
job/session limits; Docker's cgroup RAM/CPU limits do not apply to native mode.

Private state defaults to `~/.local/state/hexapod-worlds` and can be selected with
`--state-dir`. Stopping verifies the recorded container ownership label, or Linux
process start time plus a random per-launch marker before signalling a native
process. It will refuse to stop an unrelated replacement process. Native logs and
completed reconstruction outputs remain local; retained outputs expire according
to the worker policy when it runs again. Docker Mirror output uses temporary data
and is discarded when the container is removed. Download wanted results first.

Launcher CPU tests mock GPU/Docker calls and start only an unconfigured local
health server; they do not assess model inference or GPU memory requirements:

```sh
apps/api/.venv/bin/python -m pytest scripts/tests/test_worlds_local.py -q
```

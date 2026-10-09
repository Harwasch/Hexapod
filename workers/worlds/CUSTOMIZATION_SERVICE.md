# Worlds customization service

`customization_gateway.py` exposes approved ForgeWM and SANA-WM training recipes to the Worlds Model Lab. It runs on an **already configured training host**. It does not provision GPUs, deploy containers, restart inference, download datasets, or implement generic LoRA training.

The service, API proxy, cancellation supervisor, and browser workflow have local tests. Actual eight-GPU training, CUDA dependency builds, trained-checkpoint inference, quality, and performance remain unvalidated. Installation validates checkpoint structure; `gpuValidated` remains false.

## Runtime and local prerequisites

Follow [MODEL_SETUP.md](MODEL_SETUP.md) to prepare the pinned source, base weights, per-model environment, and reviewed local dataset. The published recipes require **eight visible CUDA GPUs** when execution is requested. Preparing a job starts no training or GPU allocation.

Run this service with the **model inference/training interpreter**, not the lightweight transport interpreter used by `gateway.py`. Preparation imports PyYAML; checkpoint installation imports Torch. The selected training interpreter also needs the complete model-specific training dependency lock. In the model image, this is `/opt/inference/bin/python`; build configuration `INCLUDE_TRAINING=1` includes the separate training dependencies. Matrix-Game has no supported training recipe.

This CPU-only import check verifies the service environment before starting it:

```bash
/opt/inference/bin/python -c 'import torch, yaml; print("Customization service dependencies available")'
```

For an operator-created venv, replace `/opt/inference/bin/python` with that venv's absolute interpreter path. Install the corresponding `forgewm/requirements-training.lock` or `sana_wm/requirements-training.lock` using the Python version and build constraints documented in [MODEL_SETUP.md](MODEL_SETUP.md). A missing `yaml` means preparation cannot read a recipe; a missing `torch` means installation cannot validate a checkpoint. Changing the profile's `python` field alone does not supply these libraries to the service interpreter.

Keep profiles, datasets, jobs, and checkpoint bundles operator-owned. The service root is made mode `0700`. Browser requests carry IDs, never filesystem paths or command lines. Training executes fixed argument arrays without a shell, with an allowlisted environment excluding gateway/provider tokens and with Hub downloads and experiment telemetry disabled. The operator must still review the full upstream YAML, including dataset and preceding-stage checkpoint references.

## Operator profiles

Create a private JSON array, for example `/data/worlds/customization-profiles.json`:

```json
[
  {
    "id": "forge-student-v1",
    "name": "Forge student · approved local trajectories",
    "model": "forge-wm",
    "stage": "stage3-student",
    "source": "/opt/forge-source",
    "config": "/data/worlds/recipes/forge-stage3.yaml",
    "python": "/opt/inference/bin/python",
    "baseBundle": "/models/forge-wm",
    "datasetLabel": "Local trajectory set v1"
  }
]
```

All four path fields (`source`, `config`, `python`, `baseBundle`) must be absolute. Profile IDs use 1–80 letters, digits, underscores, or hyphens. `name` and `datasetLabel` are optional display labels and are visible in the browser; do not put secrets in them. The source must contain the bootstrap's `.worlds-source.json` marker for the reviewed revision. Profiles are loaded at service startup.

| Model      | Accepted stages                                                   | Stage compatible with inference installation |
| ---------- | ----------------------------------------------------------------- | -------------------------------------------- |
| `forge-wm` | `stage0-sft`, `stage1-causal`, `stage2-distill`, `stage3-student` | `stage3-student`                             |
| `sana-wm`  | `stage1-sft`, `ode`, `self-forcing-t43`, `self-forcing-t121`      | `self-forcing-t121`                          |

Each profile selects one stage. See [the stage dependencies](MODEL_SETUP.md#real-customization-workflows) before preparing subsequent stages; an SFT teacher is not interchangeable with an interactive distilled student.

For Forge, configure the encoded local LMDB `data_path`, local `model_kwargs.model_name`, and preceding-stage checkpoints. Its source checkout must expose `ckpts/MG2-base/Wan2.1_VAE.pth`, matching the upstream recipe. For SANA, disable `data.hf_dataset_repo`, replace remote checkpoint references, and supply existing absolute trajectory/latent dataset paths. Add this field to the reviewed SANA YAML:

```yaml
worlds_components:
  gemma2: /models/sana-wm/gemma2_2b
```

That directory must contain the installed pinned Gemma2 configuration, tokenizer, and weights. The wrapper removes `worlds_components` before calling upstream training and maps its symbolic Gemma2 encoder to this local bundle. All remaining YAML fields must match the selected upstream reference config; this fragment is not a complete training recipe.

## Start the optional control service

The following commands are setup examples, not a deployment performed by this project. Inject a strong secret through the operator's normal secret mechanism; the placeholder below must be replaced.

```bash
export WORLD_CUSTOMIZATION_TOKEN='REPLACE_WITH_A_RANDOM_SECRET_OF_AT_LEAST_24_CHARACTERS'
export WORLD_CUSTOMIZATION_MAX_SECONDS=21600
# Leave execution disabled while verifying profiles and connectivity.
export WORLD_CUSTOMIZATION_ALLOW_TRAINING=0
/opt/inference/bin/python workers/worlds/customization_gateway.py \
  --host 127.0.0.1 --port 8792 \
  --root /data/worlds/customization-jobs \
  --profiles /data/worlds/customization-profiles.json
```

Run from the repository root, or use an absolute script path. The default bind is loopback on port `8792`. Every endpoint requires `Authorization: Bearer <WORLD_CUSTOMIZATION_TOKEN>`. JSON request bodies are capped at 4 KiB; the listener bounds concurrent connections and header/body read time. It does not terminate TLS itself. For a manager on a different host, use the operator's authenticated HTTPS routing to this private service; do not send the token over public plaintext HTTP.

| Setting                              | Behavior                                                                               |
| ------------------------------------ | -------------------------------------------------------------------------------------- |
| `WORLD_CUSTOMIZATION_TOKEN`          | Required service secret, at least 24 characters                                        |
| `WORLD_CUSTOMIZATION_ALLOW_TRAINING` | Only the exact value `1` permits execution; preparation and inspection work without it |
| `WORLD_CUSTOMIZATION_MAX_SECONDS`    | Default 21,600 seconds; bounded to 60–172,800 seconds                                  |

Only one training job runs at a time per service. The service timer requests cancellation at the runtime limit. An independent child supervisor also enforces its deadline and terminates the training process group if the service dies. Cancellation verifies the recorded process identity before TERM, then escalates to KILL after a bounded grace period. These controls stop training ranks; they **do not stop or deallocate the host GPU instance**, whose provider charges may continue.

Run only one service instance against a job root and the corresponding GPU allocation. Existing active or unreconciled jobs block another launch. After an abrupt restart, a record may report `unknown` until the operator verifies cancellation or reconciles it; that state is not proof the process stopped. Logs and partial outputs remain on disk.

## Connect the Worlds session manager

Set these on the session manager, not in browser code:

```dotenv
WORLD_CUSTOMIZATION_GATEWAY_URL=http://127.0.0.1:8792
WORLD_CUSTOMIZATION_GATEWAY_TOKEN=THE_SAME_SECRET_AS_WORLD_CUSTOMIZATION_TOKEN
```

Loopback HTTP is appropriate only when both processes share that network namespace. Use an HTTPS URL for a remote host. Settings can be supplied through the manager's environment or its `WORLD_ENV_FILE`. The manager forwards authenticated browser requests under `/api/v1/worlds/customization` and keeps the worker token server-side. The browser uses the ordinary Worlds API write token.

Open Model Lab → Model customization. A configured service reports its approved profiles, execution setting, and runtime limit. A missing gateway configuration is shown as unavailable; the application does not invent a training result or allocate a worker.

## Job flow and endpoints

All service paths below require the bearer token. Manager paths add `/api/v1/worlds/customization` before the same suffix.

| Method and path           | Request / effect                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `GET /capabilities`       | Reviewed profiles, execution enabled flag, and limits                                                               |
| `GET /jobs`               | Public job records without source paths, argv, tokens, or log contents                                              |
| `POST /jobs`              | `{"profileId":"forge-student-v1"}` prepares an immutable job config and argv; no training starts                    |
| `GET /jobs/{id}`          | State and available local `.pt` checkpoint artifacts                                                                |
| `POST /jobs/{id}/run`     | `{"confirmTraining":true}` starts a prepared job only if the operator also enabled execution                        |
| `POST /jobs/{id}/cancel`  | `{}` requests termination of an active or unreconciled job                                                          |
| `POST /jobs/{id}/install` | `{"artifactId":"…"}` structurally validates a completed final-stage checkpoint and creates an isolated local bundle |
| `POST /jobs/{id}/enable`  | `{"artifactId":"…"}` selects an installed bundle for the next inference worker start                                |
| `POST /jobs/{id}/disable` | `{}` selects the profile's original `baseBundle` for the next start                                                 |

The UI additionally requires explicit consent before starting an eight-GPU job. The worker checks visible CUDA devices before launching the reviewed distributed trainer. Failed, cancelled, and interrupted jobs retain their local files; prepare a fresh approved job to retry. The service does not promise optimizer-state resume or automatically connect stages.

Artifacts are opaque IDs representing local `.pt` files under the job's checkpoint directory. Filenames displayed to the browser are neutral. Intermediate SANA distributed teacher metadata/files may remain available only on the host and are not presented as installable student checkpoints. Installation is allowed only for a completed activation-compatible stage. It uses CPU `torch.load(weights_only=True)` structural validation, records a SHA256 digest, copies the custom checkpoint, and links the remaining base assets. It does not measure inference quality or compatibility on a GPU.

Each job directory retains `service.json`, `job.json`, `config.yaml`, `training.log`, and outputs. Logs may be absent when preparation or the pre-launch device check fails. The service does not expose raw logs; inspect them locally as the operator. Do not edit prepared job/config files to bypass validation.

## Apply a checkpoint selection

Selection writes one service-local environment fragment per model:

- `/data/worlds/customization-jobs/selection-forge-wm.env`
- `/data/worlds/customization-jobs/selection-sana-wm.env`

For example, after reviewing the selected bundle, the operator can load the fragment into their **next** inference launch environment:

```bash
set -a
. /data/worlds/customization-jobs/selection-forge-wm.env
set +a
# Use the normal, separately managed inference-worker start procedure.
```

The variables are `FORGEWM_WEIGHTS` and `SANA_WM_WEIGHTS`. Selecting the base checkpoint writes the profile's explicit base path instead of relying on a default. Only one artifact is selected per model. This operation does not reload the current model, restart a service, deploy anything, or validate generated output. Keep custom and base bundle paths mounted and accessible to the next inference worker.

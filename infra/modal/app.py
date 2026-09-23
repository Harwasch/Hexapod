"""The Modal function `ModalAdapter` spawns. **Built and checked; never run on a GPU.**

This is the remote half: an image, a GPU, a secret, an S3 `Transfer`, and one call to
`remote.execute`. Everything that could be got wrong about what a container *does* --
fetch the inputs, run the stage, sync `checkpoint/`, upload `out/`, report back -- lives
in `tools/pipeline/remote.py` and is covered by `tools/pipeline/tests/test_remote.py`.

What this file adds is the training image, and it is worth being exact about what has and
has not been checked about it:

* **Every artifact it names exists, and was read, on 2026-09-23.** The base
  `nvidia/cuda:12.4.1-devel-ubuntu22.04` (resolved on Docker Hub, amd64 manifest present);
  torch `2.4.1+cu124` and torchvision `0.19.1+cu124` for cp310 on PyTorch's cu124 index;
  gsplat's prebuilt `1.5.3+pt24cu124` cp310 wheel, downloaded and pinned by sha256 (its
  CUDA kernels carry SASS for sm_70/75/80/86/90, which covers T4, A100, A10, H100 and --
  by same-major compatibility -- the L4's sm_89); the v1.5.3 tag of the gsplat repository
  for `examples/`; and the example's git pins, fetched at the commits it names.
* **The trainer environment resolves and the trainer accepts the pipeline's argv.** The
  same `trainer-requirements.txt`, installed into a CPU replica (CPython 3.10, CPU torch
  2.4.1), imports `simple_trainer.py` and parses `training.gsplat_argv` with its own CLI.
  `check_trainer.py` repeats that check as the last step of building this image, so an
  image whose trainer would refuse the argv fails to build rather than failing a run.
* **What has not been checked: anything that needs the GPU.** No image has been built by
  Modal from this file and no training step has run. The first `modal deploy` is where
  the build is proven (fused-ssim compiles against nvcc there, and nowhere here), and the
  smoke run in `.github/workflows/modal.yml` is where training is.

Two decisions that shape the image, both forced rather than chosen:

* **Two Pythons.** gsplat publishes its CUDA wheels for CPython 3.10 only, and this
  project is 3.12. The pipeline and `remote.execute` run under Modal's 3.12
  (`add_python`); the trainer runs under Ubuntu 22.04's own 3.10 in `/opt/trainer`, which
  `training.trainer_python` finds through `$GSPLAT_PYTHON`.
* **A CUDA *devel* base.** gsplat's example imports `fused_ssim`, a CUDA extension
  distributed only as source, so nvcc has to be in the image to build it. It is built for
  the architectures in `CUDA_ARCHS` because there is no GPU at build time to detect one.

Why one function per tier: Modal fixes a function's GPU at decoration time, so a single
`run_stage` cannot serve an L4 request and an A100 request. `FUNCTIONS` is built from the
tiers `providers.py` offers for Modal, and each one is named `run_stage_<tier>` -- which
is the name `ModalAdapter` looks up.

Deploy, and prove it, with `.github/workflows/modal.yml`: it creates the
`twin-object-storage` secret from the R2 private-bucket credentials and runs

    modal deploy infra/modal/app.py

The secret is the private bucket, not the public one. A stage reads raw uploads and
writes run outputs, and neither belongs in the bucket the world can read -- see
`apps/api/app/worker/publish.py` for which of a run's outputs are published afterwards,
and by whom.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

import modal

#: Where `tools/pipeline` and `tools/captures` go inside the container. Siblings, because
#: `captures_bridge.py` finds the packer at `../captures` relative to itself; the pipeline
#: is imported by bare module name, so it goes on `sys.path` whole.
PIPELINE_DIR = "/opt/pipeline"
CAPTURES_DIR = "/opt/captures"

#: And where they are on the machine running `modal deploy`. Both are needed and they are
#: not interchangeable: the decorators below execute *locally* at deploy time and read
#: the tier table, while the function body executes in the container and reads the
#: copies. Absolute, so `modal deploy` works from any directory.
#:
#: This whole module is imported a second time, inside every container, before the
#: function can run -- and there it is `/root/app.py`, with no repository around it.
#: `parents[2]` of that path does not exist, and computing it unconditionally crashed
#: every container on import: the first real deploy crash-looped on `IndexError: 2`
#: while the client waited for a result that could never come. In the container the
#: pipeline is the image's own copy, so that is what the tier table is read from; the
#: local-file image steps below are only read when Modal builds the image.
HERE = Path(__file__).resolve().parent
if modal.is_local():
    REPO_ROOT = HERE.parents[1]
    LOCAL_PIPELINE = REPO_ROOT / "tools" / "pipeline"
    LOCAL_CAPTURES = REPO_ROOT / "tools" / "captures"
else:
    LOCAL_PIPELINE = Path(PIPELINE_DIR)
    LOCAL_CAPTURES = Path(CAPTURES_DIR)

#: What the 3.12 side needs: the transfer, and every module `stages` imports -- which is
#: every stage, because `remote.execute` registers them all. Missing any one of these was
#: an `ImportError` before the first byte moved; the first version of this file installed
#: two of the six, and `tools/pipeline/tests/test_modal_image.py` now pins the list.
IMAGE_PACKAGES: tuple[str, ...] = (
    "boto3>=1.34",
    "numpy>=1.26",
    "pillow>=10",
    "pyyaml>=6.0",
    "imageio-ffmpeg>=0.5",
)

#: The base image. Ubuntu 22.04 because its system Python is the 3.10 gsplat's wheels are
#: built for; CUDA 12.4 because that is the toolkit the torch and gsplat wheels are.
CUDA_BASE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
TORCH_PACKAGES: tuple[str, ...] = ("torch==2.4.1+cu124", "torchvision==0.19.1+cu124")
#: gsplat's `examples/` -- `simple_trainer.py` and its `datasets/` -- at the release the
#: wheel is, since neither is in the wheel.
GSPLAT_TAG = "v1.5.3"
GSPLAT_DIR = "/opt/gsplat"
GSPLAT_TRAINER = f"{GSPLAT_DIR}/examples/simple_trainer.py"
#: `examples/requirements.txt` at v1.5.3 pins this commit.
FUSED_SSIM = "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
#: T4, A100, A10, L4/L40S, H100. fused-ssim's setup.py adds sm_75/80/89 of its own when
#: it sees no GPU; this list is what torch's extension builder adds on top.
CUDA_ARCHS = "7.5;8.0;8.6;8.9;9.0"
TRAINER_VENV = "/opt/trainer"
TRAINER_PYTHON = f"{TRAINER_VENV}/bin/python"
_PIP = f"{TRAINER_VENV}/bin/pip"

#: Six hours. A training stage that has not finished by then is wrong rather than slow,
#: and `ModalAdapter`'s own default timeout is the same number for the same reason.
TIMEOUT_S = 6 * 3600

#: No Modal-side retries. `CloudRunner` owns retrying, because it is the thing that
#: knows about attempts, ledgers and moving a stage to another provider after too many
#: preemptions. A second retry policy underneath it would make the attempt count in
#: `attempts.json` a fiction.
RETRIES = 0

_IGNORE = ["tests", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"]

image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .apt_install(
        "git",
        "python3.10",
        "python3.10-venv",
        "python3.10-dev",
        "build-essential",
        # cv2 (opencv-python-headless) still links these.
        "libglib2.0-0",
        "libgl1",
    )
    .run_commands(
        f"python3.10 -m venv {TRAINER_VENV}",
        # setuptools and wheel too: fused-ssim is built with --no-build-isolation, so it
        # builds with whatever this venv has, and Ubuntu 22.04's bundled setuptools is
        # older than the one that can build a wheel without the `wheel` package.
        f"{_PIP} install --upgrade pip setuptools wheel",
        f"{_PIP} install {' '.join(TORCH_PACKAGES)} --index-url {TORCH_INDEX}",
    )
    .add_local_file(
        HERE / "trainer-requirements.txt", "/opt/twin/trainer-requirements.txt", copy=True
    )
    .run_commands(
        # torch pinned by constraint as well as by being installed first: without it an
        # unpinned transitive requirement is free to "upgrade" torch to a CUDA build the
        # gsplat wheel was not compiled against.
        f"printf '%s\\n' {' '.join(TORCH_PACKAGES)} > /opt/twin/constraints.txt",
        f"{_PIP} install -c /opt/twin/constraints.txt -r /opt/twin/trainer-requirements.txt",
        f"{_PIP} install -c /opt/twin/constraints.txt --no-build-isolation {FUSED_SSIM}",
        f"git clone --depth 1 --branch {GSPLAT_TAG} "
        f"https://github.com/nerfstudio-project/gsplat {GSPLAT_DIR}",
        # The LPIPS network the trainer builds in its constructor downloads AlexNet's
        # weights on first use; fetched here so a run does not depend on that host.
        f"{TRAINER_PYTHON} -c 'from torchmetrics.image.lpip import "
        f'LearnedPerceptualImagePatchSimilarity as L; L(net_type="alex", normalize=True)\'',
        env={"TORCH_CUDA_ARCH_LIST": CUDA_ARCHS},
    )
    .pip_install(*IMAGE_PACKAGES)
    .env(
        {
            "GSPLAT_TRAINER": GSPLAT_TRAINER,
            "GSPLAT_PYTHON": TRAINER_PYTHON,
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(LOCAL_CAPTURES, CAPTURES_DIR, ignore=_IGNORE, copy=True)
    .add_local_dir(LOCAL_PIPELINE, PIPELINE_DIR, ignore=_IGNORE, copy=True)
    .add_local_file(HERE / "check_trainer.py", "/opt/twin/check_trainer.py", copy=True)
    .run_commands(
        # The last word before the image is accepted: does this trainer take this argv?
        f"{TRAINER_PYTHON} /opt/twin/check_trainer.py {GSPLAT_TRAINER} {PIPELINE_DIR}",
        # And does the 3.12 side import every stage (the list above, pinned by a test)?
        f"cd {PIPELINE_DIR} && python -c 'import remote, stages, captures_bridge'",
    )
)

app = modal.App("twin-pipeline")


class S3Transfer:
    """The pipeline's `Transfer`, over boto3, inside the container.

    A deliberate near-duplicate of `apps/api/app/worker/cloud.py`'s `ObjectStoreTransfer`,
    and the duplication is the cheaper of two bad options: the alternative is putting
    `apps/api` -- FastAPI, SQLAlchemy, the models -- into a GPU image so a container can
    read four object keys. `tools/pipeline` cannot hold it either, because it may not
    grow a `boto3`; that constraint is what keeps the pipeline importable by a worker
    that already has storage credentials and by a test that has none.

    The encoding is the one both other implementations use, and it has to stay that way:
    **a directory's members are objects under the key, a file is an object at the key.**
    A stage must behave the same whether its bytes came from R2, from MinIO or from a
    directory in a test.
    """

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def put(self, key: str, source: Path) -> int:
        if source.is_dir():
            moved = 0
            for member in sorted(p for p in source.rglob("*") if p.is_file()):
                relative = member.relative_to(source).as_posix()
                self._client.upload_file(str(member), self._bucket, f"{key}/{relative}")
                moved += member.stat().st_size
            return moved
        self._client.upload_file(str(source), self._bucket, key)
        return source.stat().st_size

    def get(self, key: str, target: Path) -> int:
        if self._head(key) is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self._bucket, key, str(target))
            return target.stat().st_size
        moved = 0
        for member in self._listing(f"{key}/"):
            destination = target / member[len(key) + 1 :]
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self._bucket, member, str(destination))
            moved += destination.stat().st_size
        return moved

    def exists(self, key: str) -> bool:
        return self._head(key) is not None or bool(self._listing(f"{key}/", limit=1))

    def delete(self, key: str) -> None:
        if self._head(key) is not None:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            return
        for member in self._listing(f"{key}/"):
            self._client.delete_object(Bucket=self._bucket, Key=member)

    def _head(self, key: str) -> dict[str, Any] | None:
        try:
            return dict(self._client.head_object(Bucket=self._bucket, Key=key))
        except Exception:
            # A 404 is the common case and not an error: `get` asks "is this one object
            # or a directory of them?" and this is how it finds out.
            return None

    def _listing(self, prefix: str, *, limit: int | None = None) -> list[str]:
        """Every key under `prefix`, paged to the end.

        Paged because a checkpoint or a tileset is easily more than one page, and a
        truncated listing restores half a checkpoint without saying so.
        """
        keys: list[str] = []
        token: str | None = None
        while True:
            extra = {"ContinuationToken": token} if token else {}
            page = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix, **extra)
            keys.extend(entry["Key"] for entry in page.get("Contents", []))
            if limit is not None and keys:
                return keys[:limit]
            token = page.get("NextContinuationToken") if page.get("IsTruncated") else None
            if token is None:
                return keys


def _transfer() -> S3Transfer:
    """Built from the secret, inside the container, once per call.

    `region_name="auto"` and an explicit `s3v4` signature because the target is R2:
    botocore presigns SigV2 for every region except "auto", which is the dev/prod split
    `apps/api/app/storage/s3.py` documents at length and which no config inspection can
    see.
    """
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["OBJECT_STORAGE_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OBJECT_STORAGE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["OBJECT_STORAGE_SECRET_KEY"],
        region_name=os.environ.get("OBJECT_STORAGE_REGION", "auto"),
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )
    return S3Transfer(client, os.environ["OBJECT_STORAGE_BUCKET"])


def _run(request: dict[str, Any]) -> dict[str, Any]:
    """One stage. The whole body, so every per-tier function is the same three lines.

    `sys.path` rather than an install, because the pipeline is imported by bare module
    name. The import is inside the function so that this file can be read, and the App
    built, on a machine that has no pipeline mounted -- which is every machine that is
    not a Modal container.
    """
    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)

    import remote
    import stages  # noqa: F401 - registers every shipped implementation
    from cloud import StageRequest

    parsed = StageRequest.from_dict(request)
    root = Path("/tmp") / f"stage-{parsed.stage_id}-{uuid.uuid4().hex}"  # noqa: S108
    outcome = remote.execute(parsed, _transfer(), root)
    return outcome.to_dict()


def _tiers() -> tuple[str, ...]:
    """The tiers `providers.py` offers Modal with, that `GPU_NAMES` has a name for.

    Read locally at deploy time. Named rather than inlined so CI can assert the deployed
    functions match it: a tier added to `providers.py` and silently not deployed here is
    a `submit` that fails on a lookup, hours later, on somebody's real capture.
    """
    if str(LOCAL_PIPELINE) not in sys.path:
        sys.path.insert(0, str(LOCAL_PIPELINE))
    from modal_adapter import GPU_NAMES
    from providers import provider

    offered = provider("modal")
    return tuple(t for t in (offered.tiers if offered is not None else ()) if t in GPU_NAMES)


def _register() -> dict[str, Any]:
    """One `run_stage_<tier>` per tier Modal is offered with.

    Built in a loop rather than written out, so a tier added to `providers.py` cannot be
    silently missing here -- and so the GPU strings come from `GPU_NAMES`, which is the
    table that was wrong and is now tested.
    """
    from modal_adapter import GPU_NAMES

    functions: dict[str, Any] = {}
    for tier in TIERS:
        functions[tier] = app.function(
            image=image,
            gpu=GPU_NAMES[tier],
            timeout=TIMEOUT_S,
            retries=RETRIES,
            name=f"run_stage_{tier}",
            secrets=[modal.Secret.from_name("twin-object-storage")],
        )(_run)
    return functions


#: The tiers this file deploys, and the tiers `ModalAdapter` may therefore ask for.
TIERS = _tiers()
FUNCTIONS = _register()

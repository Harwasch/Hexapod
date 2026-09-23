"""The Modal function `ModalAdapter` spawns. **Never deployed, never run.**

This is the remote half the handoff said was missing. It is deliberately the smallest
thing that could be: an image, a GPU, a secret, an S3 `Transfer`, and one call to
`remote.execute`. Everything that could be got wrong about what a container *does* --
fetch the inputs, resume from a checkpoint, sync it out on an interval, upload `out/`,
report back -- lives in `tools/pipeline/remote.py` and is covered by
`tools/pipeline/tests/test_remote.py`, which runs it here against a directory.

What is left in this file is the part no test in this repository can reach, and it is
worth being exact about which part that is:

* **That Modal accepts these decorator arguments.** `tools/pipeline/tests` cannot import
  `modal` (it is not a dependency, by design). A `modal.App` does build locally without
  credentials, so `deploy --dry-run` style checking is possible on a machine that has the
  library; nothing in CI does it today.
* **That `gpu=` takes these strings.** It does not fail locally: an App with a nonsense
  GPU name builds fine and is rejected only server-side. That is exactly how
  `gpu="A10G"` -- AWS's instance name, not a Modal tier -- survived in `GPU_NAMES` until
  someone read the documentation. `modal_adapter.GPU_NAMES` is the one source of these
  strings for that reason.
* **The image.** `IMAGE_PACKAGES` installs what the *transfer* needs and nothing more,
  so the CPU stages have a chance of working on the first try. **A training stage will
  not**: `gsplat` needs CUDA, torch and a matching gsplat build, and no version of that
  stack has ever been assembled or run here. Inventing a pinned CUDA/torch/gsplat triple
  from memory is the failure mode this repository exists to avoid, so the hook is named
  and left empty rather than filled with a guess. `TRAINING_PACKAGES` is where it goes.

Why one function per tier: Modal fixes a function's GPU at decoration time, so a single
`run_stage` cannot serve an L4 request and an A100 request. `FUNCTIONS` is built from the
tiers `providers.py` offers for Modal, and each one is named `run_stage_<tier>` -- which
is the name `ModalAdapter` looks up.

Deploy it with credentials this repository does not have:

    modal secret create twin-object-storage \\
        OBJECT_STORAGE_ENDPOINT_URL=... OBJECT_STORAGE_ACCESS_KEY=... \\
        OBJECT_STORAGE_SECRET_KEY=... OBJECT_STORAGE_BUCKET=...
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

#: Where `tools/pipeline` is mounted inside the container. The pipeline is imported by
#: bare module name (the `tools/captures` convention), so it goes on `sys.path` whole
#: rather than being installed as a package.
PIPELINE_DIR = "/opt/pipeline"

#: And where it is on the machine running `modal deploy`. Both are needed and they are
#: not interchangeable: the decorators below execute *locally* at deploy time and read
#: the tier table, while the function body executes in the container and reads the
#: mounted copy. Absolute, so `modal deploy` works from any directory.
LOCAL_PIPELINE = Path(__file__).resolve().parents[2] / "tools" / "pipeline"

#: What the container needs to move bytes and run a CPU stage. Deliberately short.
IMAGE_PACKAGES: tuple[str, ...] = ("boto3>=1.34", "numpy>=1.26")

#: The GPU stack, and the reason this file cannot claim a training run would work.
#: Empty on purpose: see the module docstring. Fill it with versions you have built,
#: not with versions you remember.
TRAINING_PACKAGES: tuple[str, ...] = ()

#: Six hours. A training stage that has not finished by then is wrong rather than slow,
#: and `ModalAdapter`'s own default timeout is the same number for the same reason.
TIMEOUT_S = 6 * 3600

#: No Modal-side retries. `CloudRunner` owns retrying, because it is the thing that
#: knows about attempts, checkpoints, ledgers and moving a stage to another provider
#: after too many preemptions. A second retry policy underneath it would make the
#: attempt count in `attempts.json` a fiction.
RETRIES = 0

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*IMAGE_PACKAGES, *TRAINING_PACKAGES)
    .add_local_dir(str(LOCAL_PIPELINE), PIPELINE_DIR, ignore=["tests", ".venv", "__pycache__"])
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

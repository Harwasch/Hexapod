"""The worker's half of the cloud seam: bytes, credentials, and which provider.

`tools/pipeline` describes a GPU stage that has to run somewhere else -- which inputs and
which checkpoint must be where the provider can read them, under which keys -- and knows
nothing about buckets. This module performs it. That is the same split
`app/worker/registration.py` set out: the pipeline describes, the worker performs, and
the pipeline keeps no `boto3` and no idea that an API exists.

Two things live here:

* **`ObjectStoreTransfer`** implements the pipeline's `Transfer` over the `ObjectStorage`
  the worker already holds. A directory's members are objects under the key, a file is an
  object *at* the key -- the same encoding `LocalTransfer` uses, so a stage behaves the
  same whether it is driven against MinIO or against a directory in a test.
* **`build_runners`** turns the worker's configuration into a `RunnerSet`: which
  adapters, in which order, how the bytes move, and how many preemptions before a stage
  stops being retried on the cheap host and is moved to the reliable one.

A deployment whose GPU box shares a filesystem with the worker -- an NFS mount, a
workstation with its own cards, a single-machine setup -- sets `WORKER_CLOUD_TRANSFER_DIR`
and the bytes go through a directory instead of the bucket. It is the same `Transfer`
either way, which is the point of it being a protocol.

It is imported by `app.worker.child`, which means the child process now needs storage
credentials where before it needed none. That is not an oversight: a stage running on
somebody else's machine has to fetch its inputs from somewhere, and the alternative --
the supervisor moving every byte on the child's behalf -- puts a 12 GB upload through a
process whose job is to hold a lease. The child still has no database session.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from app.storage import ObjectStorage
from app.worker.pipeline_bridge import (
    PROVIDERS,
    CloudRunner,
    FakeAdapter,
    LocalTransfer,
    ModalAdapter,
    Placement,
    ProviderAdapter,
    RunnerSet,
    SubprocessAdapter,
    Transfer,
    rates_from_env,
)

__all__ = [
    "ObjectStoreTransfer",
    "adapter_for",
    "build_runners",
    "check_dispatchable",
    "transfer_for",
]

#: Providers whose adapter needs a client library that is an optional extra of this
#: project. `ModalAdapter` imports `modal` inside the one function that needs it, which
#: keeps the import lazy and also keeps the failure late -- late enough to be the first
#: GPU job of a deployment rather than its start-up. `check_dispatchable` is what makes
#: it early again.
_NEEDS_CLIENT: Mapping[str, str] = {"modal": "modal"}

#: The name `infra/modal/app.py` deploys under, which `ModalAdapter` looks functions up in.
MODAL_APP = "twin-pipeline"


def check_dispatchable(providers: Sequence[str]) -> None:
    """Refuse a worker configured to dispatch somewhere it cannot reach.

    The same instinct as `create_app`'s three start-up guards, for the same reason: a
    worker that starts, polls happily, claims a real capture and only then discovers it
    has no client library has turned a configuration mistake into a failed run and an
    hour of someone's confusion. The import is the whole test -- credentials are not
    checked here, because a token that is wrong is a different failure and the provider
    is the only thing that can tell you so.
    """
    missing = sorted(
        {
            module
            for name in providers
            if (module := _NEEDS_CLIENT.get(name)) and find_spec(module) is None
        }
    )
    if missing:
        raise RuntimeError(
            f"WORKER_CLOUD_PROVIDERS names {', '.join(sorted(set(providers)))}, and this "
            f"environment has no {', '.join(missing)}. It is an optional extra of "
            "apps/api: install it (`uv sync --extra modal`, which infra/api.Dockerfile "
            "does) or drop the provider from WORKER_CLOUD_PROVIDERS. Refusing to start a "
            "worker that would claim a GPU job and then fail to dispatch it."
        )


@dataclass(frozen=True)
class ObjectStoreTransfer:
    """The pipeline's `Transfer`, over the bucket.

    Whole objects, read and written in memory, exactly as `app.worker.outputs` does: the
    same trade A7 recorded and the same place streaming belongs when a capture is big
    enough to need it. Keys are opaque strings chosen by the pipeline, so nothing here
    knows what a stage or a checkpoint is.
    """

    storage: ObjectStorage
    #: Keys the pipeline builds are already `runs/<run id>/...`; a deployment sharing a
    #: bucket between environments sets this so the two cannot collide.
    prefix: str = ""

    def _key(self, key: str) -> str:
        return f"{self.prefix.rstrip('/')}/{key}" if self.prefix else key

    def put(self, key: str, source: Path) -> int:
        target = self._key(key)
        if source.is_dir():
            moved = 0
            for member in sorted(p for p in source.rglob("*") if p.is_file()):
                data = member.read_bytes()
                relative = member.relative_to(source).as_posix()
                self.storage.put_object(f"{target}/{relative}", data, "application/octet-stream")
                moved += len(data)
            return moved
        data = source.read_bytes()
        self.storage.put_object(target, data, "application/octet-stream")
        return len(data)

    def get(self, key: str, target: Path) -> int:
        root = self._key(key)
        single = self.storage.head_object(root)
        if single is not None:
            # Streamed: what comes back from a GPU stage is a trained splat of hundreds
            # of megabytes, arriving on a worker with two gigabytes.
            return self.storage.download_file(root, target)
        moved = 0
        for member in self._listing(f"{root}/"):
            relative = member[len(root) + 1 :]
            moved += self.storage.download_file(member, target / relative)
        return moved

    def exists(self, key: str) -> bool:
        root = self._key(key)
        if self.storage.head_object(root) is not None:
            return True
        return bool(self._listing(f"{root}/", first_page_only=True))

    def delete(self, key: str) -> None:
        root = self._key(key)
        if self.storage.head_object(root) is not None:
            self.storage.delete_object(root)
            return
        for member in self._listing(f"{root}/"):
            self.storage.delete_object(member)

    def _listing(self, prefix: str, *, first_page_only: bool = False) -> list[str]:
        """Every key under `prefix`. Paged to the end, because a directory artifact can
        hold far more than one page of tiles and a truncated listing would silently
        restore half a checkpoint."""
        keys: list[str] = []
        token: str | None = None
        while True:
            page = self.storage.list_objects(prefix, continuation_token=token)
            keys.extend(entry.key for entry in page.objects)
            token = page.next_continuation_token
            if token is None or first_page_only:
                return keys


def adapter_for(
    name: str,
    transfer: Transfer,
    *,
    sandbox: Path,
    impl_modules: tuple[str, ...] = (),
    modal_app: str = "",
) -> ProviderAdapter:
    """One adapter by name. The names are `WORKER_CLOUD_PROVIDERS`' vocabulary.

    A deployment's own rates come from `PIPELINE_GPU_RATES` and are merged over the
    surveyed table, so a contract price is configuration rather than a code change --
    see `tools/pipeline/providers.py` for why only surveyed numbers are shipped.
    """
    # A surveyed rate if the price table has one for this host, and the deployment's own
    # over the top. `subprocess` is not in the table at all -- a box you already own has
    # no list price -- so for it the environment is the only source there is.
    overrides = rates_from_env()
    surveyed = {entry.name: dict(entry.rates) for entry in PROVIDERS}
    rates = {**surveyed.get(name, {}), **overrides.get(name, {})}
    if name == "subprocess":
        return SubprocessAdapter(transfer, sandbox, rates=rates, impl_modules=impl_modules)
    if name == "fake":
        return FakeAdapter(transfer, sandbox, rates=rates)
    if name == "modal":
        # Never executed. See tools/pipeline/modal_adapter.py, which says so at length.
        return ModalAdapter(modal_app or MODAL_APP, rates=rates)
    raise ValueError(
        f"unknown cloud provider {name!r}. Known: fake, subprocess, modal. A name from "
        f"the price table that has no adapter yet (runpod-*, vast) is a provider this "
        f"deployment can quote for and cannot dispatch to"
    )


def transfer_for(storage: ObjectStorage, directory: Path | None) -> Transfer:
    """A directory both machines can see, or the bucket. The stage cannot tell which."""
    if directory is None:
        return ObjectStoreTransfer(storage)
    directory.mkdir(parents=True, exist_ok=True)
    return LocalTransfer(directory)


def build_runners(
    storage: ObjectStorage,
    *,
    providers: tuple[str, ...],
    sandbox: Path,
    impl_modules: tuple[str, ...] = (),
    preemptions_before_fallback: int = 2,
    modal_app: str = "",
    poll_interval_s: float = 5.0,
    checkpoint_every_s: float = 60.0,
    transfer_dir: Path | None = None,
) -> RunnerSet:
    """CPU stages here, GPU stages on the first provider that keeps them.

    The list is preferred-first and its last entry is the one a preempted stage falls
    back to, which `Placement.of` refuses to let be interruptible. A deployment that
    names only a cheap interruptible host has no fallback, and `Placement` says so rather
    than quietly retrying it forever.
    """
    if not providers:
        return RunnerSet.local()
    transfer = transfer_for(storage, transfer_dir)
    adapters = tuple(
        adapter_for(
            name,
            transfer,
            sandbox=sandbox / name,
            impl_modules=impl_modules,
            modal_app=modal_app,
        )
        for name in providers
    )
    placement = Placement.of(*adapters, preemptions_before_fallback=preemptions_before_fallback)
    return RunnerSet.cloud(
        CloudRunner(
            placement,
            transfer,
            poll_interval_s=poll_interval_s,
            checkpoint_every_s=checkpoint_every_s,
        )
    )

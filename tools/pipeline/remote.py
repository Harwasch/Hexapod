"""What a GPU container does, minus the provider. The far side of a `ProviderAdapter`.

`run_stage.py` runs one stage in a sandbox that somebody else filled. This fills it, and
empties it afterwards: fetch the inputs and whatever checkpoint a previous attempt left,
run the stage while a syncer copies `checkpoint/` out on an interval, and upload `out/`
when it finishes. Between them those two files are the whole remote half, and neither
imports a provider SDK.

That split is deliberate and it is the only reason any of this is testable. A deployed
Modal function is a thing nobody here can execute; a function that takes a `Transfer` and
a directory is a thing `LocalTransfer` can drive in a unit test, and does. So the Modal
app in `infra/modal/` is a thin wrapper -- an image, a GPU, a secret, and a call to
`execute` -- and the behaviour that could be got wrong lives here, where it is exercised.
`SubprocessAdapter` does the same work from the *outside*, because a local process cannot
fetch its own bytes; this is the same sequence performed from the inside, which is what a
real provider's container does.

Three things are worth stating about the order, because each of them is a decision:

* **The checkpoint syncer runs beside the stage, never inside it.** The stage writes into
  `checkpoint/` and knows nothing about where that goes. This is the only arrangement
  that survives the container being taken back without warning, because a process that is
  killed does not get to run its upload -- which is the entire reason `checkpoint_every_s`
  exists rather than a single sync at the end.
* **`out/` is uploaded only on success.** A stage that raised may have written half an
  artifact, and `CloudRunner` verifies what it declared it produces against what arrived.
  Uploading the debris of a failed attempt would make the next attempt's `prepare_stage`
  wipe it anyway, after paying to move it.
* **A final sync happens after a successful run and not after a failed one.** A stage that
  finished between two intervals would otherwise lose the tail of its checkpoint; a stage
  that failed has nothing extra worth keeping, and the attempt that follows wants the last
  good checkpoint rather than whatever state the failure left.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import run_stage
from cloud import StageRequest, Transfer

__all__ = ["RemoteOutcome", "execute", "stage_sandbox"]

#: The four directories a stage is given, in the layout `run_stage._context` expects.
SANDBOX_DIRS: tuple[str, ...] = ("inputs", "out", "work", "checkpoint")


@dataclass(frozen=True)
class RemoteOutcome:
    """What the container hands back to `poll`.

    `metrics` and `summary` are the stage's own, read back out of `result.json` rather
    than passed in memory, so this path and `SubprocessAdapter`'s read the same file and
    a stage cannot report one thing locally and another remotely. The byte counts are
    here because a transfer that silently moved nothing is the failure that looks like a
    fast run.
    """

    metrics: Mapping[str, Any]
    summary: str
    fetched_bytes: int
    uploaded_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "summary": self.summary,
            "fetchedBytes": self.fetched_bytes,
            "uploadedBytes": self.uploaded_bytes,
        }


def stage_sandbox(root: Path) -> Path:
    """The four directories, created. Separate so a caller can fill one itself."""
    for name in SANDBOX_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def execute(
    request: StageRequest,
    transfer: Transfer,
    root: Path,
    *,
    impl_modules: Sequence[str] = (),
) -> RemoteOutcome:
    """Fetch, run, sync, upload. Raises whatever the stage raised, bar `TimeoutError`.

    Letting the stage's exception out is the contract: for Modal it becomes the exception
    `FunctionCall.get` re-raises, which `ModalAdapter.poll` classifies. Swallowing it and
    returning a failure dict would make every failed stage look like a successful call
    that happened to return bad news, and the adapter would have to guess.
    """
    stage_sandbox(root)
    fetched = _fetch(request, transfer, root)

    stop = threading.Event()
    syncer = threading.Thread(
        target=_sync_forever,
        args=(request, transfer, root / "checkpoint", stop, request.checkpoint_every_s),
        daemon=True,
    )
    syncer.start()
    try:
        run_stage.run(_spec(request, root, impl_modules))
    except TimeoutError as error:
        # The one exception a stage may not let out as itself. Modal 1.5.5 answers a
        # zero-timeout `FunctionCall.get` on a call that has not finished by raising the
        # *builtin* `TimeoutError()` (`modal/_functions.py`, `poll_function`), so on the
        # client a stage that timed out and a stage that is still running would be the
        # same exception. `ModalAdapter.poll` reads the builtin as "still running";
        # this is what keeps that reading true. RuntimeError rather than a class of our
        # own, because the client unpickles it and must not need this module to do so.
        raise RuntimeError(f"the stage timed out: {error!r}") from error
    finally:
        # Stopped before anything else, including on the failure path: a syncer left
        # running past its stage would keep writing a checkpoint directory that the next
        # attempt is about to be handed.
        stop.set()
        syncer.join(timeout=5.0)

    # Not fatal, and deliberately so. The stage finished; its checkpoint is insurance
    # that is no longer needed, and turning a completed GPU run into a failure because
    # the insurance could not be filed would be the expensive way to be wrong. A bucket
    # that rejects this will reject the upload below too, and that one *is* fatal.
    _sync_quietly(request, transfer, root / "checkpoint")
    uploaded = transfer.put(request.outputs_key, root / "out")
    result = _result(root)
    return RemoteOutcome(
        metrics=result.get("metrics") or {},
        summary=str(result.get("summary") or ""),
        fetched_bytes=fetched,
        uploaded_bytes=uploaded,
    )


# --- the parts -----------------------------------------------------------------------


def _fetch(request: StageRequest, transfer: Transfer, root: Path) -> int:
    """Inputs by name, then the checkpoint. A missing checkpoint is the normal case.

    `Transfer.get` answers 0 for a key that is not there, which is what makes attempt 1
    and attempt 2 the same code path: there is no "is this a resume" question here, and
    `StageContext.has_checkpoint` answers it for the stage by looking at the directory.
    """
    moved = 0
    for artifact, key in request.inputs.items():
        moved += transfer.get(key, root / "inputs" / artifact)
    moved += transfer.get(request.checkpoint_key, root / "checkpoint")
    return moved


def _spec(request: StageRequest, root: Path, impl_modules: Sequence[str]) -> Path:
    """The same JSON `SubprocessAdapter` writes, so `run_stage` has one input format."""
    spec = root / "stage.json"
    spec.write_text(
        json.dumps(
            {
                "request": request.to_dict(),
                "sandbox": str(root),
                "implModules": list(impl_modules),
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return spec


def _sync(request: StageRequest, transfer: Transfer, checkpoint: Path) -> int:
    """Copy `checkpoint/` out, unless there is nothing in it yet.

    The emptiness check is not an optimisation. `Transfer.put` on an empty directory
    would replace a good checkpoint from an earlier attempt with nothing, which is
    exactly the attempt that most needs it back.
    """
    if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        return 0
    return transfer.put(request.checkpoint_key, checkpoint)


def _sync_forever(
    request: StageRequest,
    transfer: Transfer,
    checkpoint: Path,
    stop: threading.Event,
    every: float,
) -> None:
    """Sync until told to stop. A failed sync must not take the stage down with it.

    The interval has a floor because `checkpoint_every_s` is operator-supplied and a zero
    would spin this thread against the bucket for the length of a training run.
    """
    while not stop.wait(max(0.05, every)):
        _sync_quietly(request, transfer, checkpoint)


def _sync_quietly(request: StageRequest, transfer: Transfer, checkpoint: Path) -> int:
    """`_sync`, with a failure reported and swallowed.

    A checkpoint is insurance, not the product. `sys.stdout` rather than `print` because
    stdout is this container's log -- the same convention `run_stage` states -- and
    because `print` is what the linter here refuses, for the same reason.
    """
    try:
        return _sync(request, transfer, checkpoint)
    except Exception as error:  # any failure to save insurance is the same failure
        sys.stdout.write(f"remote: checkpoint sync failed, continuing: {error!r}\n")
        sys.stdout.flush()
        return 0


def _result(root: Path) -> dict[str, Any]:
    """`result.json`, or nothing.

    A stage that returned without writing one is a `run_stage` that did not reach its
    last line, which should be impossible -- but an empty result is a poll that reports
    success with no metrics, and that is better than a container that fetched, ran,
    uploaded and then died reading a file.
    """
    path = root / "result.json"
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return document if isinstance(document, dict) else {}

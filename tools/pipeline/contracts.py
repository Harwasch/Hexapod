"""What a stage is handed, what it hands back, and what the executor records.

A stage implementation is a plain function `(StageContext) -> StageOutcome`. Everything it
is allowed to touch comes off the context; everything it wants remembered goes into the
outcome. It never learns the recipe's shape, never sees another stage, and never builds a
path -- which is what lets `impl` be swapped from a recipe file alone.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import progress
from artifacts import ArtifactDecl, ArtifactRef
from errors import StageContractError

__all__ = [
    "FANOUT_METRIC",
    "FANOUT_PARAM",
    "FanOut",
    "FanOutPart",
    "MetricValue",
    "StageContext",
    "StageOutcome",
    "StepResult",
    "fanout_role",
]

MetricValue = bool | int | float | str


# --- fanning one stage out over several remote calls ------------------------------------
#
# Block training (`blocks.py`) is one stage whose work splits into independent pieces that
# want a GPU each. Measured on the spool at `blocks: 2` on one L4, the blocks ran one after
# another inside a single attempt -- 2,530 s then 2,207 s, 2.3 h and $1.73 in all against
# ~1 h and $0.67 whole -- while Modal bills per GPU-second, so the same blocks on two GPUs
# at once cost the same seconds and end when the longer one does. The runner that already
# owns retries, preemption and the attempt ledger (`cloud.CloudRunner`) does the fanning
# out; the stage only says what the pieces are. Three roles, carried in one reserved param
# so no remote image needs a new request field to ignore:
#
#   head  the stage's usual call. A stage that can split may answer it with `FANOUT_METRIC`
#         (a `FanOut`, as JSON) instead of its outputs; one that cannot ignores the role.
#   part  one piece, on its own checkpoint key; it writes into `checkpoint/` only.
#   join  after every piece has finished and been brought home: the ordinary end of the
#         stage (for blocks: the merge, the evaluation and the held-out error).
#
# A runner that knows nothing of this (`LocalRunner`, the stub, an older worker) never sets
# the param, and the stage then does everything in one call, as before.

#: The reserved param a fanning runner adds to a request: `{"role": "head" | "part" |
#: "join", "part": <id>}`.
FANOUT_PARAM = "fanout"
#: The metric a head call answers with when it wants to be fanned out.
FANOUT_METRIC = "fanOut"


@dataclass(frozen=True)
class FanOutPart:
    """One piece: its id, and the paths under `checkpoint/` that are its result.

    `collect` is explicit rather than "whatever came back" because every part also carries
    the shared members (`FanOut.share`) and its own live snapshots, and copying those home
    from N parts at once would have them overwrite one another.
    """

    id: str
    collect: tuple[str, ...]


@dataclass(frozen=True)
class FanOut:
    """What a head call asks for: pieces, how many at once, and what each piece needs.

    `share` names the paths under the stage's `checkpoint/` every part starts with (for
    blocks, the prior and the plan); `parts` are in the order to start them -- longest
    first, so a queue longer than `parallel` does not leave the biggest piece for last.
    """

    parts: tuple[FanOutPart, ...]
    parallel: int
    share: tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps(
            {
                "parts": [{"id": p.id, "collect": list(p.collect)} for p in self.parts],
                "parallel": self.parallel,
                "share": list(self.share),
            },
            sort_keys=True,
        )

    @staticmethod
    def parse(text: object) -> FanOut | None:
        """The spec in a head call's metrics, or None when it is absent or unreadable."""
        if not isinstance(text, str):
            return None
        try:
            document = json.loads(text)
            parts = tuple(
                FanOutPart(str(entry["id"]), tuple(str(c) for c in entry.get("collect") or ()))
                for entry in document["parts"]
            )
            parallel = int(document.get("parallel") or 1)
            share = tuple(str(s) for s in document.get("share") or ())
        except (ValueError, KeyError, TypeError, AttributeError):
            return None
        ids = [part.id for part in parts]
        if not parts or len(set(ids)) != len(ids):
            return None
        return FanOut(parts=parts, parallel=max(1, parallel), share=share)


def fanout_role(params: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """(role, part id) from a request's params; (None, None) when no runner fans out."""
    value = params.get(FANOUT_PARAM)
    if not isinstance(value, Mapping):
        return None, None
    role = value.get("role")
    part = value.get("part")
    return (
        str(role) if role in ("head", "part", "join") else None,
        None if part is None else str(part),
    )


@dataclass(frozen=True)
class StageOutcome:
    """What a stage reports. Artifacts are discovered from disk, not declared here.

    A stage that fails raises; there is no failure status to forget to check.
    """

    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class StageContext:
    """The stage's whole world."""

    recipe: str
    run_id: str
    stage_id: str
    impl: str
    params: Mapping[str, Any]
    attempt: int
    gpu_tier: str | None
    out_dir: Path
    work_dir: Path
    checkpoint_dir: Path
    log_path: Path
    checkpoint_key: str
    #: Where the runner -- not the stage -- records what each attempt of this stage was
    #: placed on and what it billed. It is on the context because a runner's `_invoke` is
    #: handed the context and nothing else; a stage implementation has no business here.
    attempts_path: Path
    _inputs: Mapping[str, Path]
    _produces: Mapping[str, ArtifactDecl]
    #: Also write each log line to stdout. Set on a remote machine, where stdout is what
    #: the provider's log tail reads, so a line reaches the worker while the stage runs.
    echo: bool = False

    @property
    def inputs(self) -> Mapping[str, Path]:
        return self._inputs

    def input(self, name: str) -> Path:
        """Resolve a consumed artifact to a path. Refuses anything undeclared."""
        try:
            return self._inputs[name]
        except KeyError:
            declared = ", ".join(sorted(self._inputs)) or "nothing"
            raise StageContractError(
                f"stage {self.stage_id!r} (impl {self.impl!r}) asked for input {name!r}, which it "
                f"does not declare in `consumes`. It consumes: {declared}"
            ) from None

    def has_input(self, name: str) -> bool:
        """For `optional_consumes` -- a mask that an earlier stage may or may not produce."""
        return name in self._inputs

    def output(self, name: str) -> Path:
        """Reserve the path for a declared output, creating its parents (and, for a
        directory artifact, the directory itself). Refuses anything undeclared, so a typo
        surfaces here rather than as a missing artifact at the end of the stage."""
        try:
            decl = self._produces[name]
        except KeyError:
            declared = ", ".join(sorted(self._produces)) or "nothing"
            raise StageContractError(
                f"stage {self.stage_id!r} (impl {self.impl!r}) tried to write output {name!r}, "
                f"which it does not declare in `produces`. It produces: {declared}"
            ) from None
        path = self.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if decl.kind == "dir":
            path.mkdir(parents=True, exist_ok=True)
        return path

    def param(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)

    @property
    def has_checkpoint(self) -> bool:
        """True when a previous attempt left resumable state behind.

        `checkpoint/` is the one directory the executor does not clear between attempts,
        and `checkpoint_key` is where B1b's `CloudRunner` syncs it to object storage --
        out before an attempt, back when the attempt ends however it ends. A stage on a
        preemptible tier writes its progress here and reads it back through this flag;
        being killed is then ordinary operation rather than lost work.
        """
        return any(self.checkpoint_dir.iterdir())

    def log(self, message: str) -> None:
        line = message.rstrip("\n") + "\n"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        if self.echo:
            sys.stdout.write(line)
            sys.stdout.flush()

    def run(self, argv: Sequence[str]) -> None:
        """Run an external tool, with its output in the stage log as it is printed.

        This is LocalRunner's subprocess half: a stage that shells out (colmap, ffmpeg)
        does it here rather than each runner growing a second way to invoke stages.
        Streamed rather than collected at exit, so a long tool's progress is visible
        while it runs (`progress.py`).
        """
        self.log(f"$ {' '.join(argv)}")
        code = progress.stream(argv, cwd=self.work_dir, log=self.log)
        if code != 0:
            raise subprocess.CalledProcessError(code, list(argv))


@dataclass(frozen=True)
class StepResult:
    """One stage's run: what it produced, what it measured, where its log is.

    This is the unit A7 writes to `job_step` and A10 renders. `checkpoint_key` is set when
    the stage left resumable state; `attempt` and `preempted` are how B1 records that being
    killed on a cheap GPU is normal operation rather than a failure.
    """

    stage_id: str
    impl: str
    runner: str
    attempt: int
    duration_s: float
    artifacts: tuple[ArtifactRef, ...]
    metrics: Mapping[str, MetricValue]
    log_path: str
    checkpoint_key: str | None = None
    gpu_tier: str | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "stageId": self.stage_id,
            "impl": self.impl,
            "runner": self.runner,
            "attempt": self.attempt,
            "durationS": round(self.duration_s, 6),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metrics": dict(self.metrics),
            "logPath": self.log_path,
            "checkpointKey": self.checkpoint_key,
            "gpuTier": self.gpu_tier,
            "summary": self.summary,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n", "utf-8")

    @staticmethod
    def from_dict(document: Mapping[str, Any]) -> StepResult:
        """Read a StepResult back out of the `step.json` a previous attempt wrote.

        This is what makes a resumed run whole: A7 re-runs a recipe with the stages that
        already succeeded skipped, and their results come from here rather than from a
        second execution.
        """
        artifacts = tuple(ArtifactRef.from_dict(entry) for entry in document["artifacts"])
        metrics: dict[str, MetricValue] = {}
        for key, value in dict(document["metrics"]).items():
            metrics[str(key)] = value if isinstance(value, bool | int | float | str) else str(value)
        checkpoint_key = document.get("checkpointKey")
        gpu_tier = document.get("gpuTier")
        return StepResult(
            stage_id=str(document["stageId"]),
            impl=str(document["impl"]),
            runner=str(document["runner"]),
            attempt=int(document["attempt"]),
            duration_s=float(document["durationS"]),
            artifacts=artifacts,
            metrics=metrics,
            log_path=str(document["logPath"]),
            checkpoint_key=None if checkpoint_key is None else str(checkpoint_key),
            gpu_tier=None if gpu_tier is None else str(gpu_tier),
            summary=str(document.get("summary", "")),
        )

    @staticmethod
    def read(path: Path) -> StepResult:
        return StepResult.from_dict(json.loads(path.read_text(encoding="utf-8")))

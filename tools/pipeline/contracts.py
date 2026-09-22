"""What a stage is handed, what it hands back, and what the executor records.

A stage implementation is a plain function `(StageContext) -> StageOutcome`. Everything it
is allowed to touch comes off the context; everything it wants remembered goes into the
outcome. It never learns the recipe's shape, never sees another stage, and never builds a
path -- which is what lets `impl` be swapped from a recipe file alone.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifacts import ArtifactDecl, ArtifactRef
from errors import StageContractError

__all__ = ["MetricValue", "StageContext", "StageOutcome", "StepResult"]

MetricValue = bool | int | float | str


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
    _inputs: Mapping[str, Path]
    _produces: Mapping[str, ArtifactDecl]

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

        Nothing checkpoints yet. The contract exists now because B1 runs on preemptible
        GPUs, where being killed is ordinary: `checkpoint/` is the one directory the
        executor does not clear between attempts, and `checkpoint_key` is where B1 syncs
        it to object storage.
        """
        return any(self.checkpoint_dir.iterdir())

    def log(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip("\n") + "\n")

    def run(self, argv: Sequence[str]) -> None:
        """Run an external tool, with its output in the stage log.

        This is LocalRunner's subprocess half: a stage that shells out (colmap, ffmpeg)
        does it here rather than each runner growing a second way to invoke stages.
        """
        self.log(f"$ {' '.join(argv)}")
        completed = subprocess.run(  # noqa: S603 - argv comes from stage code, never a shell
            list(argv),
            cwd=self.work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        for line in (completed.stdout + completed.stderr).splitlines():
            self.log(line)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, list(argv))


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

"""The workdir contract: the one place that knows where anything lives on disk.

    <workdir>/
      recipe.json              the recipe exactly as executed (name, version, stages, params)
      artifacts.json           every artifact produced, with sizes and checksums
      inputs/<name>            artifacts the caller seeded before the run (the upload)
      stages/<stage id>/
        out/<artifact name>    everything this stage declared it produces, and nothing else
        work/                  scratch; never an artifact, safe to delete at any time
        checkpoint/            survives a killed attempt; the resume seam for B1
        log.txt                the stage's log
        step.json              the StepResult for the stage's last attempt
        attempts.json          B1's ledger: where every attempt ran and what it billed

Nothing outside this module builds a path by string concatenation, so B1 can relocate a
workdir (object storage, a fresh GPU container) by changing `root` and nothing else.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Workdir"]


@dataclass(frozen=True)
class Workdir:
    root: Path

    @staticmethod
    def create(root: Path) -> Workdir:
        workdir = Workdir(Path(root))
        workdir.inputs_dir.mkdir(parents=True, exist_ok=True)
        return workdir

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    @property
    def recipe_path(self) -> Path:
        return self.root / "recipe.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "artifacts.json"

    def input_path(self, name: str) -> Path:
        return self.inputs_dir / name

    def stage_dir(self, stage_id: str) -> Path:
        return self.stages_dir / stage_id

    def out_dir(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "out"

    def work_dir(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "work"

    def checkpoint_dir(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "checkpoint"

    def log_path(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "log.txt"

    def step_path(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "step.json"

    def attempts_path(self, stage_id: str) -> Path:
        """B1's attempt ledger: one entry per attempt, preempted ones included.

        Beside `step.json` rather than inside `checkpoint/` or `work/` because it has to
        outlive both an attempt and a machine -- the supervisor reads it after the run to
        fill in `jobs.cost_usd`, and a cost that counted only the attempt that succeeded
        would hide the ones that were paid for and lost.
        """
        return self.stage_dir(stage_id) / "attempts.json"

    def attempt_ledgers(self) -> tuple[Path, ...]:
        """Every stage's ledger that exists, so a whole run can be totalled."""
        if not self.stages_dir.is_dir():
            return ()
        return tuple(sorted(self.stages_dir.glob("*/attempts.json")))

    def artifact_path(self, stage_id: str, name: str) -> Path:
        return self.out_dir(stage_id) / name

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def prepare_stage(self, stage_id: str) -> None:
        """Make a stage's directories, and clear the outputs of any previous attempt.

        `out/` is wiped so a half-written output from a killed attempt can never be
        mistaken for a produced artifact. `checkpoint/` is deliberately kept: that is what
        makes a preempted stage resumable rather than restartable.
        """
        out = self.out_dir(stage_id)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        self.work_dir(stage_id).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir(stage_id).mkdir(parents=True, exist_ok=True)

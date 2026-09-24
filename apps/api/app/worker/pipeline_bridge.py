"""The seam to `tools/pipeline`, in one module.

A6's rule is that **the pipeline** must not depend on `apps/api`: no models, no session,
no HTTP. The reverse direction is fine, and this is it — the worker imports the pipeline
as a library, so the claim loop needs no second copy of the schema and the pipeline needs
no idea that a database exists.

`tools/pipeline` is a flat uv project (`executor.py`, `registry.py`, `plan.py`… at its
root, following `tools/captures`' ``package = false`` convention), so importing it is a
``sys.path`` insertion rather than a dependency. Its module names are generic enough to
collide with a worker that has its own, which is why every one of those bare imports is
in this file and nowhere else: the rest of the worker imports them from here, under
``app.worker.*``, where nothing can shadow anything.

Two things keep it honest rather than hidden:

* ``mypy_path`` in pyproject.toml points at the same directory, so mypy resolves these to
  the real files and type-checks every call across the seam (``follow_imports = "silent"``,
  because that project runs its own mypy and its errors are not this one's to report);
* ``tests/test_worker_pipeline.py`` runs a real recipe through a real worker, so a change
  to the executor's signature fails here rather than in production.

B1b added the cloud seam to the list, and the rule did not change: ``cloud``,
``adapters``, ``providers`` and ``modal_adapter`` are imported here and nowhere else in
``apps/api``. The direction of the dependency is the point — the pipeline describes a
stage that has to run somewhere else and what has to move for it to; the worker, which
holds the bucket credentials, performs it (``app/worker/cloud.py``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from app.config import REPO_ROOT

#: Where `tools/pipeline` is. Overridable because a container may lay the two out
#: differently (C1); the default is the repository layout.
PIPELINE_DIR = Path(os.environ.get("PIPELINE_DIR") or REPO_ROOT / "tools" / "pipeline").resolve()


def ensure_importable(directory: Path = PIPELINE_DIR) -> None:
    if not (directory / "executor.py").is_file():
        raise ModuleNotFoundError(
            f"tools/pipeline is not importable from {directory}. Set PIPELINE_DIR to where "
            f"the pipeline project lives, or run the worker from a full checkout."
        )
    path = str(directory)
    if path not in sys.path:
        sys.path.insert(0, path)


ensure_importable()

from adapters import FakeAdapter, LocalTransfer, SubprocessAdapter  # noqa: E402
from artifacts import ArtifactRef  # noqa: E402
from cloud import (  # noqa: E402
    AttemptLedger,
    CloudRunner,
    Placement,
    ProviderAdapter,
    RunCost,
    Transfer,
    run_cost,
)
from contracts import StepResult  # noqa: E402
from errors import PipelineError, PreemptedError, StageFailedError  # noqa: E402
from executor import RunResult, execute  # noqa: E402
from modal_adapter import ModalAdapter  # noqa: E402
from plan import Plan, PlannedStage, plan_recipe  # noqa: E402
from progress import latest as latest_progress  # noqa: E402
from progress import tail as tail_of  # noqa: E402
from providers import PROVIDERS, Provider, Rate, rates_from_env, with_rates  # noqa: E402
from recipe import Recipe, load_recipe, recipe_dir  # noqa: E402
from runners import RunnerSet  # noqa: E402
from workdir import Workdir  # noqa: E402

__all__ = [
    "PIPELINE_DIR",
    "PROVIDERS",
    "ArtifactRef",
    "AttemptLedger",
    "CloudRunner",
    "FakeAdapter",
    "LocalTransfer",
    "ModalAdapter",
    "PipelineError",
    "Placement",
    "Plan",
    "PlannedStage",
    "PreemptedError",
    "Provider",
    "ProviderAdapter",
    "Rate",
    "Recipe",
    "RunCost",
    "RunResult",
    "RunnerSet",
    "StageFailedError",
    "StepResult",
    "SubprocessAdapter",
    "Transfer",
    "Workdir",
    "ensure_importable",
    "execute",
    "latest_progress",
    "load_recipe",
    "plan_recipe",
    "rates_from_env",
    "recipe_dir",
    "run_cost",
    "tail_of",
    "with_rates",
]

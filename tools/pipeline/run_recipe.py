"""Run a recipe from the command line.

    uv run python run_recipe.py splat-ingest --workdir /tmp/run-1 \
        --runner stub --seed upload=./fixture.ply

A recipe that declares `site_context` (photo-reconstruct) gets an empty one -- a site's
first epoch, nothing to align to -- unless `--seed site_context=<dir>` names one (a
directory holding `site_context.json`, laid out as `alignment.py` describes).

The worker (A7) calls `executor.execute_recipe` directly rather than shelling out to this;
this exists so a recipe can be run and inspected by hand, which is how A8 developed
Lane 1's stages and how B2 will develop Lane 2's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import alignment
from errors import PipelineError
from executor import execute_recipe
from plan import plan_recipe
from recipe import load_recipe
from runners import RunnerSet

_RUNNERS = {"stub": RunnerSet.stubbed, "local": RunnerSet.local}


def _seed(values: list[str]) -> dict[str, Path]:
    seeds: dict[str, Path] = {}
    for value in values:
        name, _, path = value.partition("=")
        if not name or not path:
            raise SystemExit(f"--seed expects name=path, got {value!r}")
        seeds[name] = Path(path)
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", help="a shipped recipe name, or a path to a .yaml")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--runner", choices=sorted(_RUNNERS), default="stub")
    parser.add_argument("--seed", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--plan-only", action="store_true", help="validate the recipe and print the plan"
    )
    args = parser.parse_args(argv)
    document: dict[str, object]
    try:
        if args.plan_only:
            plan = plan_recipe(load_recipe(args.recipe))
            document = {
                "recipe": plan.recipe.name,
                "stages": [
                    {"id": s.id, "impl": s.impl.name, "inputs": dict(s.inputs)} for s in plan.stages
                ],
                "gpuStages": list(plan.gpu_stages),
            }
        else:
            seeds = _seed(args.seed)
            if alignment.SITE_CONTEXT in load_recipe(args.recipe).inputs and (
                alignment.SITE_CONTEXT not in seeds
            ):
                seed_empty_site_context(args.workdir)
            result = execute_recipe(args.recipe, args.workdir, _RUNNERS[args.runner](), seeds)
            document = {
                "recipe": result.recipe,
                "stages": [step.to_dict() for step in result.steps],
                "artifacts": [artifact.to_dict() for artifact in result.artifacts],
            }
    except PipelineError as error:
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 1
    sys.stdout.write(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return 0


def seed_empty_site_context(workdir: Path) -> Path:
    """`inputs/site_context/site_context.json`, empty, unless one is already there."""
    target = workdir / "inputs" / alignment.SITE_CONTEXT
    target.mkdir(parents=True, exist_ok=True)
    path = target / alignment.SITE_CONTEXT_FILE
    if not path.exists():
        path.write_text(json.dumps(alignment.empty_site_context()) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    raise SystemExit(main())

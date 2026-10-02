"""Run exactly one stage, here, from a JSON spec. The far side of `SubprocessAdapter`.

This is what a GPU container runs. It has no recipe, no plan, no workdir and no idea what
came before it or comes after: it is handed a `StageRequest` and a sandbox with the
stage's inputs and whatever checkpoint a previous attempt left, it calls the registered
implementation, and it writes what the implementation reported to `result.json`.

It does **not** sync its own checkpoint. The stage writes into `checkpoint/` and a syncer
beside the process copies that out on an interval -- which is the only arrangement that
survives the process being killed without warning, since a process that is killed does
not get to run its upload.

`print` is the log: stdout is captured to the sandbox's `log.txt`, which `logs()` tails.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import registry
from cloud import StageRequest
from contracts import StageContext, StageOutcome


def _context(request: StageRequest, sandbox: Path) -> StageContext:
    impl = registry.lookup(request.impl)
    if impl is None:
        raise SystemExit(
            f"impl {request.impl!r} is not registered here. Registered: "
            f"{', '.join(registry.known_impls()) or 'none'}"
        )
    return StageContext(
        recipe=request.recipe,
        run_id=request.run_id,
        stage_id=request.stage_id,
        impl=request.impl,
        params=dict(request.params),
        attempt=request.attempt,
        gpu_tier=request.tier,
        out_dir=sandbox / "out",
        work_dir=sandbox / "work",
        checkpoint_dir=sandbox / "checkpoint",
        log_path=sandbox / "stage.log",
        checkpoint_key=request.checkpoint_key,
        attempts_path=sandbox / "attempts.json",
        _inputs={name: sandbox / "inputs" / name for name in request.inputs},
        _produces={decl.name: decl for decl in request.produces},
        echo=True,
    )


def run(spec_path: Path) -> int:
    document: dict[str, Any] = json.loads(spec_path.read_text(encoding="utf-8"))
    for module in document.get("implModules") or ():
        import_module(str(module))
    request = StageRequest.from_dict(document["request"])
    sandbox = Path(str(document["sandbox"]))
    context = _context(request, sandbox)
    impl = registry.lookup(request.impl)
    assert impl is not None  # _context already refused an unknown impl
    sys.stdout.write(
        f"run_stage: {request.impl} for stage {request.stage_id!r}, attempt {request.attempt}, "
        f"{'resuming' if context.has_checkpoint else 'from scratch'}\n"
    )
    # `echo` puts each stage log line on stdout as it is written, so one `logs()` tail
    # shows both, and shows them while the stage runs rather than after it ends.
    outcome: StageOutcome = impl.fn(context)
    (sandbox / "result.json").write_text(
        json.dumps({"metrics": dict(outcome.metrics), "summary": outcome.summary}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one stage from a JSON spec.")
    parser.add_argument("spec", type=Path, help="the spec SubprocessAdapter wrote")
    return run(parser.parse_args(argv).spec)


if __name__ == "__main__":
    raise SystemExit(main())

"""Run gsplat's `simple_trainer.py` unchanged, and end it once held-out quality is flat.

**Not executed on a GPU in this repository yet**, like everything that needs CUDA here.
What is tested (`tests/test_convergence.py`) is the mechanism, against a stand-in trainer
whose loop has v1.5.3's shape; the decision itself is `convergence.py`, shared with the
stage and unit-tested. The image build runs `--self-check` against the real trainer.

**Why a wrapper and not a patch.** The repository ships no fork of gsplat's examples:
the image clones `examples/` at the v1.5.3 tag and the pipeline drives it by argv
(`training.gsplat_argv`), with its own torch code beside it where it needs more
(`holdout_error.py`). This follows suit. It runs the trainer as `python simple_trainer.py`
would -- `runpy` with `__name__ == "__main__"` and the trainer's directory first on
`sys.path` -- so the trainer's own `__main__` block builds its config, applies
`adjust_steps` and imports its bilateral grid exactly as it always does. The one thing
changed is the last call that block makes: `from gsplat.distributed import cli` is
replaced, before the trainer runs, by a `cli` that wraps two methods of the `Runner`
class it is handed `main` from, then calls the real `cli`:

* `Runner.eval(step)` runs as it is (it writes `stats/val_step<step>.json` and prints its
  line), then reads the PSNR back out of that file and asks `convergence.decide`. On a
  stop it appends `step + 2` to the config's `ply_steps`, `save_steps` and `eval_steps`
  -- lists v1.5.3's loop re-reads at every step (`if step in [i - 1 for i in
  cfg.ply_steps]`) -- so on the very next step the trainer itself exports the PLY (its
  appearance bake included), saves the checkpoint and `train_step` stats, and evaluates;
  after that evaluation the loop is ended by an exception only this file raises.
* `Runner.train()` catches that exception, and nothing else.

So every file the pipeline reads is written by gsplat's own code, in its own place, and
the run that stops early looks, to `training.parse_metrics` and `training.latest_ply`,
like a run whose schedule was that long. `converge.json` beside `stats/` says what
happened. The rendered val images `eval()` writes are deleted as they are scored: nothing
reads them, and a tail of evaluations would otherwise leave hundreds of full-size PNGs.

Single GPU only: gsplat's `cli` spawns one process per device when there are several, and
a class patched here would not reach them. The pipeline's tiers are all one GPU; with
more, the hook says so and the trainer runs its full schedule.

Usage: `$GSPLAT_PYTHON converge_trainer.py --trainer <simple_trainer.py> [rule] -- <the
trainer's own argv, from its strategy on>`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import convergence

#: What `--self-check` requires of the trainer's source: the call this file replaces, the
#: method it wraps, and the loop re-reading the three step lists it appends to.
REQUIRED_SOURCE = (
    "from gsplat.distributed import cli",
    "cli(main, cfg, verbose=True)",
    'def eval(self, step: int, stage: str = "val")',
    "def train(self)",
    "step in [i - 1 for i in cfg.ply_steps]",
    "step in [i - 1 for i in cfg.save_steps]",
    "step in [i - 1 for i in cfg.eval_steps]",
    "self.stats_dir}/{stage}_step{step:04d}.json",
    "refine_stop_iter",
)


class _ConvergedError(Exception):
    """Raised after the evaluation that follows a stop decision; caught by `train`."""


class _State:
    def __init__(self, rule: convergence.Rule, result_dir: Path | None) -> None:
        self.rule = rule
        self.result_dir = result_dir
        self.hooked = False
        self.reason = "the trainer never reached gsplat.distributed.cli"
        self.history: list[tuple[int, float]] = []
        self.decided_at: int | None = None
        self.stopped_at: int | None = None
        self.gain_db: float | None = None
        self.last: str | None = None
        self.max_steps: int | None = None
        self.refine_stop: int | None = None
        self.window: int | None = None
        self.started = time.time()

    def report(self) -> dict[str, object]:
        return {
            "version": 1,
            "hooked": self.hooked,
            "reason": self.reason,
            "rule": self.rule.to_dict(),
            "maxSteps": self.max_steps,
            "refineStopIter": self.refine_stop,
            "windowSteps": self.window,
            "evaluations": len(self.history),
            "lastDecision": self.last,
            "decidedAtStep": None if self.decided_at is None else self.decided_at + 1,
            "stoppedAtStep": None if self.stopped_at is None else self.stopped_at + 1,
            "stoppedEarly": self.stopped_at is not None,
            "gainDb": self.gain_db,
            "wallSeconds": round(time.time() - self.started, 1),
        }

    def write(self) -> None:
        if self.result_dir is None:
            return
        try:
            self.result_dir.mkdir(parents=True, exist_ok=True)
            (self.result_dir / convergence.REPORT).write_text(
                json.dumps(self.report(), indent=2) + "\n", encoding="utf-8"
            )
        except OSError as error:
            sys.stderr.write(f"converge: could not write {convergence.REPORT}: {error}\n")


def install(runner: Any, cfg: Any, state: _State) -> None:
    """Wrap `runner.eval` and `runner.train` (a class) for `cfg`'s run. See the docstring.

    Read off `cfg` after the trainer's own `adjust_steps`, so every number here is the
    scaled one the loop really uses: `max_steps`, the strategy's `refine_stop_iter`, and
    the rule's window multiplied by the same `steps_scaler`.
    """
    original_eval = runner.eval
    original_train = runner.train
    scaler = float(getattr(cfg, "steps_scaler", 1.0) or 1.0)
    max_steps = int(cfg.max_steps)
    refine_stop = min(int(getattr(cfg.strategy, "refine_stop_iter", 0)), max_steps)
    window = max(1, int(state.rule.window * scaler))
    state.max_steps, state.refine_stop, state.window = max_steps, refine_stop, window

    def eval(self: Any, step: int, stage: str = "val") -> None:
        original_eval(self, step, stage)
        if stage != "val" or getattr(self, "world_rank", 0) != 0:
            return
        psnr = _read_psnr(Path(self.stats_dir) / f"{stage}_step{step:04d}.json")
        _drop_renders(Path(getattr(self, "render_dir", "")), stage, step)
        if psnr is not None:
            state.history.append((step, psnr))
        if state.decided_at is not None:
            state.stopped_at = step
            raise _ConvergedError
        decision = convergence.decide(
            state.history, start=refine_stop, window=window, min_gain_db=state.rule.min_gain_db
        )
        state.last = decision.reason
        state.gain_db = decision.gain_db
        # Room for one more step, which is the one the trainer saves on.
        if decision.stop and step + 2 < max_steps:
            state.decided_at = step
            for name in ("ply_steps", "save_steps", "eval_steps"):
                getattr(cfg, name).append(step + 2)
            sys.stdout.write(
                f"converge: held-out PSNR gained {decision.gain_db} dB over the last "
                f"{window} steps (< {state.rule.min_gain_db} dB); saving at step "
                f"{step + 2} of {max_steps} and stopping\n"
            )
            sys.stdout.flush()

    def train(self: Any) -> None:
        try:
            original_train(self)
        except _ConvergedError:
            done = None if state.stopped_at is None else state.stopped_at + 1
            sys.stdout.write(f"converge: stopped after {done} of {max_steps} steps\n")
            sys.stdout.flush()

    runner.eval = eval
    runner.train = train


def _read_psnr(path: Path) -> float | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = document.get("psnr") if isinstance(document, dict) else None
    return float(value) if isinstance(value, int | float) else None


def _drop_renders(render_dir: Path, stage: str, step: int) -> None:
    if not render_dir.is_dir():
        return
    for path in render_dir.glob(f"{stage}_step{step}_*.png"):
        path.unlink(missing_ok=True)


def _result_dir(argv: list[str]) -> Path | None:
    for flag in ("--result_dir", "--result-dir"):
        if flag in argv and argv.index(flag) + 1 < len(argv):
            return Path(argv[argv.index(flag) + 1])
    return None


def self_check(trainer: Path) -> int:
    """At image build: the trainer still has the shape this hooks, and gsplat's `cli`
    is where it is. No GPU, no training."""
    source = trainer.read_text(encoding="utf-8")
    missing = [text for text in REQUIRED_SOURCE if text not in source]
    if missing:
        sys.stderr.write(f"converge: {trainer} no longer has {missing}\n")
        return 1
    distributed = importlib.import_module("gsplat.distributed")

    if not callable(getattr(distributed, "cli", None)):
        sys.stderr.write("converge: gsplat.distributed.cli is missing\n")
        return 1
    sys.stdout.write(f"converge: {trainer} has the loop this hooks; gsplat cli found\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    own, rest = (
        (args[: args.index("--")], args[args.index("--") + 1 :]) if "--" in args else (args, [])
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--trainer", type=Path, default=None)
    parser.add_argument("--every", type=int, default=convergence.DEFAULT_EVERY)
    parser.add_argument("--window", type=int, default=convergence.DEFAULT_WINDOW)
    parser.add_argument(
        "--min-gain-db", dest="min_gain_db", type=float, default=convergence.DEFAULT_MIN_GAIN_DB
    )
    parser.add_argument("--self-check", dest="self_check", action="store_true")
    options = parser.parse_args(own)
    trainer = options.trainer or (
        Path(os.environ["GSPLAT_TRAINER"]) if os.environ.get("GSPLAT_TRAINER") else None
    )
    if trainer is None:
        sys.stderr.write("converge: no --trainer and no $GSPLAT_TRAINER\n")
        return 2
    trainer = trainer.resolve()
    sys.path.insert(0, str(trainer.parent))
    if options.self_check:
        return self_check(trainer)
    state = _State(
        convergence.Rule(
            every=options.every, window=options.window, min_gain_db=options.min_gain_db
        ),
        _result_dir(rest),
    )
    try:
        distributed: Any = importlib.import_module("gsplat.distributed")
    except ImportError as error:
        distributed = None
        state.reason = f"gsplat.distributed does not import ({error}); trained the full schedule"
    if distributed is not None:
        real_cli = distributed.cli

        def cli(fn: Any, cfg: Any, verbose: bool = False) -> Any:
            runner = getattr(fn, "__globals__", {}).get("Runner")
            devices = _device_count()
            if runner is None:
                state.reason = (
                    "the trainer's main has no Runner beside it; trained the full schedule"
                )
            elif devices > 1:
                state.reason = (
                    f"{devices} GPUs: gsplat spawns a process each, which a patch here "
                    f"does not reach; trained the full schedule"
                )
            else:
                install(runner, cfg, state)
                state.hooked = True
                state.reason = "hooked"
            return real_cli(fn, cfg, verbose=verbose)

        distributed.cli = cli
    sys.argv = [str(trainer), *rest]
    try:
        runpy.run_path(str(trainer), run_name="__main__")
    except SystemExit as exit_:
        code = exit_.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    finally:
        state.write()
    return 0


def _device_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:  # no torch, or no driver: not more than one GPU
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

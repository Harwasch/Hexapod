"""Best-effort observations of the trainer, isolated from its numerical work.

CPython 3.10, stdlib only. The means optimizer supplies a tick without replacing the
loop; its original call and return are preserved. The writer supplies the loop-local
losses (v1.5.3 simple_trainer.py:739-747). Quality observations are detached and no RNG is touched,
and every observation is guarded independently of the training call it surrounds.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
import uuid
from pathlib import Path
from types import MethodType
from typing import Any

# Checked on the quality-patched pinned source by the image self-check.
REQUIRED_SOURCE = (
    'self.optimizers["means"]',
    "optimizer.step()",
    'self.writer.add_scalar("train/loss", loss.item(), step)',
    'self.writer.add_scalar("train/l1loss", l1loss.item(), step)',
    'self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)',
    "cfg.sh_degree_interval",
)
EVERY = 200
TAG = "live-train: "


def number(value: Any, digits: int = 6) -> float | None:
    try:
        result = float(value)
        return round(result, digits) if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def item(container: Any, key: str) -> Any:
    # torch.nn.ParameterDict supports indexing, but is not a dict and has no get().
    with contextlib.suppress(Exception):
        return container[key]
    return None


def emit(payload: dict[str, Any]) -> None:
    """A broken log sink must never interrupt a step, even while unwinding a stop."""
    with contextlib.suppress(Exception):
        line = TAG + json.dumps({"v": 1, **payload}, separators=(",", ":"), allow_nan=False)
        if len(line.encode()) < 512:
            print(line, flush=True)  # noqa: T201 - the worker's telemetry transport


def install(runner: Any, cfg: Any, timing: Any, state: Any = None) -> None:
    """Install through `instrument`, after convergence and timing, including rule-off runs."""
    with contextlib.suppress(Exception):
        _install(runner, cfg, timing, state)


def _install(runner: Any, cfg: Any, timing: Any, state: Any) -> None:
    original_train = getattr(runner, "train", None)
    original_eval: Any = getattr(runner, "eval", None)
    if not callable(original_train):
        return
    attempt = uuid.uuid4().hex[:12]
    losses: dict[str, float | None] = {}
    step = 0
    last_step = 0
    last_time = 0.0

    def send(kind: str, **fields: Any) -> None:
        emit({"k": kind, "run": attempt, **fields})

    def steps_time() -> float:
        return (
            time.monotonic()
            - getattr(timing, "train_entered", time.monotonic())
            - getattr(timing, "eval_in_train_s", 0.0)
            - getattr(timing, "ply_s", 0.0)
        )

    def sample(self: Any) -> None:
        nonlocal last_step, last_time
        with contextlib.suppress(Exception):
            now = steps_time()
            optimizer: Any = item(getattr(self, "optimizers", None), "means")
            groups = getattr(optimizer, "param_groups", [])
            means = item(getattr(self, "splats", None), "means")
            mem = None
            with contextlib.suppress(Exception):
                import torch

                mem = number(torch.cuda.max_memory_allocated() / 1024**3)
            interval = getattr(cfg, "sh_degree_interval", None)
            degree = getattr(cfg, "sh_degree", None)
            send(
                "s",
                step=step,
                t=number(time.monotonic() - timing.started),
                loss=dict(losses) if losses else None,
                gs=None
                if means is None
                else len(means) + getattr(self, "hex_settings", {}).get("far_count", 0),
                lr=number(groups[0].get("lr"), 10) if groups else None,
                sh=min(max(0, step - 1) // interval, degree)
                if interval and degree is not None
                else None,
                its=number((step - last_step) / (now - last_time)) if now > last_time else None,
                mem=mem,
            )
            last_step, last_time = step, now
        # Independent of the core sample: optional quality failures never hide it.
        with contextlib.suppress(Exception):
            import quality_trainer

            payload = quality_trainer.live_metrics(self, step)
            if payload:
                print(  # noqa: T201 - the worker's telemetry transport
                    "live-metrics: " + json.dumps(payload, allow_nan=False), flush=True
                )

    def train(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal step, last_step, last_time, attempt
        restore: list[tuple[Any, str, Any]] = []
        with contextlib.suppress(Exception):
            attempt = uuid.uuid4().hex[:12]
            losses.clear()
            strategy = getattr(cfg, "strategy", None)
            scaler = getattr(cfg, "steps_scaler", 1.0) or 1.0
            rule = getattr(state, "rule", None) if getattr(state, "hooked", False) else None
            send(
                "p",
                **{
                    "maxSteps": getattr(cfg, "max_steps", None),
                    "refineStart": getattr(strategy, "refine_start_iter", None),
                    "refineStop": getattr(strategy, "refine_stop_iter", None),
                    "refineEvery": getattr(strategy, "refine_every", None),
                    "capMax": getattr(strategy, "cap_max", None),
                    "shDegree": getattr(cfg, "sh_degree", None),
                    "shInterval": getattr(cfg, "sh_degree_interval", None),
                    "rule": None
                    if rule is None
                    else {
                        "every": max(1, int(rule.every * scaler)),
                        "window": max(1, int(rule.window * scaler)),
                        "minGainDb": rule.min_gain_db,
                    },
                },
            )
        with contextlib.suppress(Exception):
            writer: Any = getattr(self, "writer", None)
            scalar = getattr(writer, "add_scalar", None)
            if callable(scalar):

                def add_scalar(tag: str, value: Any, *a: Any, **kw: Any) -> Any:
                    result = scalar(tag, value, *a, **kw)
                    with contextlib.suppress(Exception):
                        if tag.startswith("train/") and "loss" in tag:
                            key = tag[6:].replace("loss", "") or "total"
                            # Bounded by the transport, while preserving component names.
                            if len(key) <= 24 and (key in losses or len(losses) < 8):
                                losses[key] = number(value)
                    return result

                restore.append((writer, "add_scalar", scalar))
                writer.add_scalar = add_scalar
        with contextlib.suppress(Exception):
            optimizer: Any = item(getattr(self, "optimizers", None), "means")
            original_step = getattr(optimizer, "step", None)
            if callable(original_step):

                def tick(_optimizer: Any, *a: Any, **kw: Any) -> Any:
                    nonlocal step
                    result = original_step(*a, **kw)
                    step += 1
                    if step == 1 or step % EVERY == 0:
                        sample(self)
                    return result

                restore.append((optimizer, "step", original_step))
                # ExponentialLR wraps a bound method at train entry; preserve that shape.
                optimizer.step = MethodType(tick, optimizer)
        step = last_step = 0
        last_time = 0.0
        try:
            return original_train(self, *args, **kwargs)
        finally:
            if step != last_step:
                sample(self)
            with contextlib.suppress(Exception):
                stopped = getattr(state, "stopped_at", None)
                if stopped is not None:
                    send(
                        "x",
                        step=stopped + 1,
                        maxSteps=getattr(cfg, "max_steps", None),
                        gain=number(getattr(state, "gain_db", None)),
                    )
            for obj, name, value in restore:
                with contextlib.suppress(Exception):
                    setattr(obj, name, value)

    def evaluate(self: Any, index: int, stage: str = "val") -> Any:
        try:
            return original_eval(self, index, stage)
        finally:
            with contextlib.suppress(Exception):
                if stage == "val":
                    psnr = None
                    with contextlib.suppress(Exception):
                        path = Path(getattr(self, "stats_dir", "")) / f"val_step{index:04d}.json"
                        psnr = number(json.loads(path.read_text(encoding="utf-8")).get("psnr"))
                    active = getattr(state, "hooked", False)
                    send(
                        "e",
                        step=index + 1,
                        psnr=psnr,
                        gain=number(getattr(state, "gain_db", None)) if active else None,
                        decision=(
                            getattr(state, "last", None)
                            if active and getattr(state, "last", None) != "densifying"
                            else None
                        ),
                    )

    runner.train = train
    if callable(original_eval):
        runner.eval = evaluate

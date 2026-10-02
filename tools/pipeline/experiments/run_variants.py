"""Run one capture through named parameter variants, one job each, and tabulate them.

The question it answers is "does this switch make *our* captures better, and at what
cost?" -- which a benchmark scene cannot answer, because a curated benchmark is not a
phone video. Each variant is a `params` override for `POST /api/v1/phone/captures/{id}/
process`, started only when the one before it has finished (the API refuses a second
active job on one capture, and the worker ran one job at a time when this was written),
then polled on `GET /api/v1/jobs/{id}` until it ends. What is read back is exactly what
the job records:

* **wall time** -- here, from the POST to the terminal status, so queueing counts -- and
  the job's own `durationS`;
* **per-stage seconds** -- each step's `durationS`;
* **cost** -- the sum of the steps' `stageCostUsd` (every attempt, preempted ones
  included) and the job's `costUsd`;
* **held-out quality** -- the `train` step's `psnr` / `ssim` / `lpips` (gsplat's `val`
  split, every 8th frame, never trained on), `ccPsnr` when a bilateral grid ran, and the
  `quality` step's `keepPct` and `heldOutPsnr` when it wrote them;
* **gaussians** -- the `train` step's count.

Two things to know before reading a table it wrote. Every variant re-runs `pose` (each
is a fresh `process`), so run-to-run noise from COLMAP and from training is inside every
difference: give the baseline `repeat: 2` to see how big it is before believing a 0.2 dB
win. And every run writes the capture's quality verdict, so the phone's view of the
capture is whichever variant finished last.

The phone key comes from `$TWIN_PHONE_KEY` and nowhere else -- never an argument, which
would land in shell history and `ps`. Usage, from `tools/pipeline`:

    TWIN_PHONE_KEY=... uv run python -m experiments.run_variants \\
        --api https://<api host> --variants experiments/variants/spool-table.yaml \\
        --out runs/spool-variants

`--dry-run` prints each variant's merged params and starts nothing; `--resume` skips the
variants already finished in `<out>.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "KEY_ENV",
    "TERMINAL",
    "Api",
    "ApiError",
    "Plan",
    "Variant",
    "load_plan",
    "markdown",
    "merge_params",
    "summarise",
]

#: Where the phone key is read from. Only there.
KEY_ENV = "TWIN_PHONE_KEY"
#: `RunStatus` values a job rests in when it will not change again.
TERMINAL = frozenset({"complete", "error", "cancelled"})
#: A variant name: something a table cell and a file name can both hold.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
#: The stage order a photo-reconstruct run goes in, for the per-stage columns.
STAGE_ORDER = (
    "normalize",
    "pose",
    "mask",
    "train",
    "compensate",
    "georeference",
    "quality",
    "place",
    "package",
    "thumbnail",
    "ground_samples",
    "manifest",
    "register",
)


class PlanError(ValueError):
    """A variant file that cannot be run as written."""


@dataclass(frozen=True)
class Variant:
    name: str
    #: Stage id -> params, already merged over the plan's `base`.
    params: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Plan:
    capture: str | None
    recipe: str
    variants: tuple[Variant, ...]
    baseline: str | None


def merge_params(
    base: Mapping[str, Mapping[str, Any]], over: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """`over` on top of `base`, stage by stage. A value of `None` removes the key, so a
    variant can take a base setting away rather than only add to it; a stage left empty
    is dropped, since the phone's whitelist refuses a stage it was sent nothing for no
    better than one it was sent something wrong for."""
    merged: dict[str, dict[str, Any]] = {stage: dict(values) for stage, values in base.items()}
    for stage, values in over.items():
        target = merged.setdefault(stage, {})
        for name, value in values.items():
            if value is None:
                target.pop(name, None)
            else:
                target[name] = value
    return {stage: values for stage, values in merged.items() if values}


def _stage_params(where: str, value: object) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PlanError(f"{where} must map stage ids to params, not {value!r}")
    out: dict[str, dict[str, Any]] = {}
    for stage, params in value.items():
        if not isinstance(params, Mapping):
            raise PlanError(f"{where}.{stage} must be a mapping of params, not {params!r}")
        out[str(stage)] = dict(params)
    return out


def load_plan(text: str) -> Plan:
    """A variant file: YAML (or JSON, which is YAML). Either a bare list of
    `{name, params, repeat?}`, or a mapping with `variants` and optionally `capture`,
    `recipe` (default photo-reconstruct), `base` params every variant starts from, and
    `baseline`, the variant the others' deltas are against (default: one named
    `baseline`, if there is one)."""
    document = yaml.safe_load(text)
    if isinstance(document, list):
        document = {"variants": document}
    if not isinstance(document, Mapping):
        raise PlanError("a variant file is a list of variants or a mapping with `variants`")
    unknown = set(document) - {"capture", "recipe", "base", "variants", "baseline"}
    if unknown:
        raise PlanError(f"unknown keys {sorted(unknown)}")
    base = _stage_params("base", document.get("base"))
    raw = document.get("variants")
    if not isinstance(raw, list) or not raw:
        raise PlanError("`variants` must be a non-empty list")
    variants: list[Variant] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise PlanError(f"variant {index} must be a mapping with `name` and `params`")
        extra = set(item) - {"name", "params", "repeat"}
        if extra:
            raise PlanError(f"variant {index}: unknown keys {sorted(extra)}")
        name = str(item.get("name", ""))
        if not _NAME.match(name):
            raise PlanError(
                f"variant {index}: name {name!r} must be 1-64 of letters, digits and ._+- "
                f"(it names a table row and a file)"
            )
        repeat = item.get("repeat", 1)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or not 1 <= repeat <= 5:
            raise PlanError(f"variant {name}: repeat must be a whole number from 1 to 5")
        params = merge_params(base, _stage_params(f"variant {name}.params", item.get("params")))
        for copy in range(repeat):
            variants.append(Variant(name if copy == 0 else f"{name}#{copy + 1}", params))
    names = [variant.name for variant in variants]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise PlanError(f"variant names must be unique; repeated: {duplicates}")
    baseline = document.get("baseline")
    if baseline is None and "baseline" in names:
        baseline = "baseline"
    if baseline is not None and str(baseline) not in names:
        raise PlanError(f"baseline {baseline!r} is not one of the variants")
    capture = document.get("capture")
    return Plan(
        capture=None if capture is None else str(capture),
        recipe=str(document.get("recipe", "photo-reconstruct")),
        variants=tuple(variants),
        baseline=None if baseline is None else str(baseline),
    )


# --- reading a job back ------------------------------------------------------------------


def _number(value: object) -> float | None:
    """A JSON number, or a Decimal the API serialised as a string; None for anything else."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _steps(job: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Each stage's step, the latest attempt if a stage has several rows."""
    found: dict[str, Mapping[str, Any]] = {}
    for step in job.get("steps") or []:
        if isinstance(step, Mapping) and isinstance(step.get("stageId"), str):
            found[step["stageId"]] = step
    return found


def _metric(step: Mapping[str, Any] | None, name: str) -> float | None:
    if step is None:
        return None
    metrics = step.get("metrics")
    return _number(metrics.get(name)) if isinstance(metrics, Mapping) else None


def summarise(
    variant: str,
    job: Mapping[str, Any],
    *,
    wall_s: float | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One table row from a `JobRead` document. Anything the job did not record is None,
    never 0: a variant that failed before `train` has no PSNR, not a PSNR of zero."""
    steps = _steps(job)
    train = steps.get("train")
    quality = steps.get("quality")
    stage_seconds = {
        stage: seconds
        for stage, step in steps.items()
        if (seconds := _metric(step, "durationS")) is not None
    }
    stage_costs = [cost for step in steps.values() if (cost := _metric(step, "stageCostUsd"))]
    gaussians = _metric(train, "gaussians")
    return {
        "variant": variant,
        "jobId": job.get("id"),
        "status": job.get("status"),
        "error": job.get("error"),
        "wallS": None if wall_s is None else round(wall_s, 1),
        "jobDurationS": _number(job.get("durationS")),
        "stageSeconds": {
            stage: stage_seconds[stage]
            for stage in (*STAGE_ORDER, *sorted(set(stage_seconds) - set(STAGE_ORDER)))
            if stage in stage_seconds
        },
        "stageCostUsd": round(sum(stage_costs), 4) if stage_costs else None,
        "jobCostUsd": _number(job.get("costUsd")),
        "trainTier": (train or {}).get("metrics", {}).get("tier") if train else None,
        "iterations": _metric(train, "iterations"),
        "psnr": _metric(train, "psnr"),
        "ssim": _metric(train, "ssim"),
        "lpips": _metric(train, "lpips"),
        "ccPsnr": _metric(train, "ccPsnr"),
        "gaussians": None if gaussians is None else int(gaussians),
        "keepPct": _metric(quality, "keepPct"),
        "heldOutPsnr": _metric(quality, "heldOutPsnr"),
        "params": dict(params) if params is not None else job.get("params"),
    }


def _cell(value: object, digits: int) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _delta(value: object, reference: object, digits: int) -> str:
    a, b = _number(value), _number(reference)
    return "" if a is None or b is None else f" ({a - b:+.{digits}f})"


def _minutes(value: object) -> str:
    number = _number(value)
    return "-" if number is None else f"{number / 60:.1f}"


def markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str | None = None,
    title: str = "Variant runs",
    note: str = "",
) -> str:
    """The results as Markdown: a headline table (quality with deltas against
    `baseline`, cost, time) and a per-stage minutes table."""
    reference = next((row for row in rows if row.get("variant") == baseline), None)
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines += [
        "Quality is gsplat's held-out `val` split (every 8th registered frame, never trained "
        "on). "
        + (f"Brackets are the difference from `{baseline}`. " if reference is not None else "")
        + "Every variant re-poses the capture, so the differences include run-to-run noise.",
        "",
        "| variant | status | PSNR | SSIM | LPIPS | CC PSNR | gaussians | keep % "
        "| wall min | stage $ | job $ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        base = reference if reference is not None and row is not reference else None

        def with_delta(
            key: str,
            digits: int,
            row: Mapping[str, Any] = row,
            base: Mapping[str, Any] | None = base,
        ) -> str:
            text = _cell(row.get(key), digits)
            return text + (_delta(row.get(key), base.get(key), digits) if base else "")

        gaussians = row.get("gaussians")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('variant')}`",
                    str(row.get("status") or "-"),
                    with_delta("psnr", 2),
                    with_delta("ssim", 3),
                    with_delta("lpips", 3),
                    _cell(row.get("ccPsnr"), 2),
                    "-" if gaussians is None else f"{int(gaussians):,}",
                    with_delta("keepPct", 1),
                    _minutes(row.get("wallS")),
                    _cell(row.get("stageCostUsd"), 3),
                    _cell(row.get("jobCostUsd"), 3),
                ]
            )
            + " |"
        )
    stages = [
        stage
        for stage in STAGE_ORDER
        if any(stage in (row.get("stageSeconds") or {}) for row in rows)
    ]
    extra = sorted(
        {stage for row in rows for stage in (row.get("stageSeconds") or {})} - set(STAGE_ORDER)
    )
    stages += extra
    if stages:
        lines += [
            "",
            "Minutes per stage (`durationS`):",
            "",
            "| variant | " + " | ".join(stages) + " |",
            "|---|" + "---|" * len(stages),
        ]
        for row in rows:
            seconds = row.get("stageSeconds") or {}
            lines.append(
                f"| `{row.get('variant')}` | "
                + " | ".join(_minutes(seconds.get(stage)) for stage in stages)
                + " |"
            )
    failed = [row for row in rows if row.get("status") not in (None, "complete")]
    if failed:
        lines += ["", "Did not complete:", ""]
        lines += [
            f"- `{row.get('variant')}` ({row.get('status')}): {row.get('error') or 'no error text'}"
            for row in failed
        ]
    lines += ["", "Params each variant sent:", ""]
    lines += [
        f"- `{row.get('variant')}`: `{json.dumps(row.get('params'), sort_keys=True)}`"
        for row in rows
    ]
    return "\n".join(lines) + "\n"


# --- the API -------------------------------------------------------------------------------


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


Opener = Callable[[urllib.request.Request, float], Any]


def _open(request: urllib.request.Request, timeout: float) -> Any:
    # The scheme is checked in `Api.__init__`; urllib honours HTTPS_PROXY and the system
    # CA bundle, and nothing here turns either off.
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310


class Api:
    """The two routes this needs, over the standard library: no client dependency."""

    def __init__(self, base: str, key: str, *, opener: Opener = _open) -> None:
        if not base.startswith(("https://", "http://")):
            raise ValueError(f"--api must be an http(s) URL, not {base!r}")
        if not key:
            raise ValueError(f"no phone key: set ${KEY_ENV}")
        self._base = base.rstrip("/")
        self._key = key
        self._opener = opener

    def _call(self, method: str, path: str, body: object | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            f"{self._base}{path}", data=data, method=method
        )
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", f"Bearer {self._key}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with self._opener(request, 60.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise ApiError(error.code, error.read().decode("utf-8", "replace")) from error

    def process(self, capture: str, recipe: str, params: Mapping[str, Any]) -> dict[str, Any]:
        result = self._call(
            "POST",
            f"/api/v1/phone/captures/{capture}/process",
            {"recipe": recipe, "params": dict(params)},
        )
        return dict(result)

    def job(self, job_id: str) -> dict[str, Any]:
        return dict(self._call("GET", f"/api/v1/jobs/{job_id}"))

    def stop(self, capture: str) -> None:
        self._call("POST", f"/api/v1/phone/captures/{capture}/stop")


def _busy(error: ApiError) -> bool:
    """The capture already has a job queued or running: wait, don't fail the variant."""
    return error.status == 409 and "already has job" in error.body


def _say(text: str) -> None:
    stamp = datetime.now(tz=UTC).strftime("%H:%M:%S")
    sys.stdout.write(f"[{stamp}] {text}\n")
    sys.stdout.flush()


def _run_one(
    api: Api,
    capture: str,
    recipe: str,
    variant: Variant,
    *,
    poll_s: float,
    job_timeout_s: float,
    busy_timeout_s: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    waited = clock()
    while True:
        try:
            job = api.process(capture, recipe, variant.params)
            break
        except ApiError as error:
            if not _busy(error):
                _say(f"{variant.name}: refused -- {error}")
                return {
                    **summarise(variant.name, {}, params=variant.params),
                    "status": "refused",
                    "error": str(error),
                }
            if clock() - waited > busy_timeout_s:
                return {
                    **summarise(variant.name, {}, params=variant.params),
                    "status": "not-started",
                    "error": f"the capture stayed busy for {busy_timeout_s:.0f} s: {error}",
                }
            _say(f"{variant.name}: the capture has a job running; waiting")
            sleep(poll_s)
    started = clock()
    job_id = str(job["id"])
    _say(f"{variant.name}: job {job_id} queued")
    last = ""
    while str(job.get("status")) not in TERMINAL:
        if clock() - started > job_timeout_s:
            _say(f"{variant.name}: over {job_timeout_s:.0f} s; stopping it")
            try:
                api.stop(capture)
            except ApiError as error:
                _say(f"{variant.name}: stop failed: {error}")
            row = summarise(variant.name, job, wall_s=clock() - started, params=variant.params)
            return {**row, "status": "timeout"}
        sleep(poll_s)
        try:
            job = api.job(job_id)
        except (ApiError, OSError) as error:  # a blip in polling is not the job failing
            _say(f"{variant.name}: poll failed ({error}); retrying")
            continue
        running = [
            str(step.get("stageId"))
            for step in job.get("steps") or []
            if step.get("status") == "in-progress"
        ]
        state = f"{job.get('status')} {','.join(running)}"
        if state != last:
            _say(f"{variant.name}: {state}")
            last = state
    return summarise(variant.name, job, wall_s=clock() - started, params=variant.params)


def _write(out: Path, rows: Sequence[Mapping[str, Any]], plan: Plan, capture: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "capture": capture,
        "recipe": plan.recipe,
        "baseline": plan.baseline,
        "writtenAt": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "rows": list(rows),
    }
    out.with_suffix(".json").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(
        markdown(rows, baseline=plan.baseline, title=f"Variant runs: capture {capture}"),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api", help="API base URL, e.g. https://api.example.com")
    parser.add_argument("--variants", type=Path, required=True, help="variant YAML/JSON")
    parser.add_argument("--capture", help="capture id; overrides the file's `capture`")
    parser.add_argument("--out", type=Path, default=Path("runs") / "variants")
    parser.add_argument("--only", action="append", default=[], help="run just these names")
    parser.add_argument("--poll", type=float, default=30.0, help="seconds between polls")
    parser.add_argument("--job-timeout", type=float, default=4 * 3600.0)
    parser.add_argument("--busy-timeout", type=float, default=6 * 3600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    plan = load_plan(args.variants.read_text(encoding="utf-8"))
    capture = args.capture or plan.capture
    if not capture:
        parser.error("no capture id: give --capture or put `capture:` in the variant file")
    variants = [v for v in plan.variants if not args.only or v.name in args.only]
    if args.dry_run:
        for variant in variants:
            sys.stdout.write(f"{variant.name}: {json.dumps(variant.params, sort_keys=True)}\n")
        return 0
    if not args.api:
        parser.error("--api is required unless --dry-run")
    api = Api(args.api, os.environ.get(KEY_ENV, ""))

    rows: list[dict[str, Any]] = []
    existing = args.out.with_suffix(".json")
    if args.resume and existing.is_file():
        rows = [
            row
            for row in json.loads(existing.read_text(encoding="utf-8")).get("rows", [])
            if row.get("status") in TERMINAL
        ]
    done = {row["variant"] for row in rows}
    for variant in variants:
        if variant.name in done:
            _say(f"{variant.name}: already finished in {existing}; skipped")
            continue
        row = _run_one(
            api,
            capture,
            plan.recipe,
            variant,
            poll_s=args.poll,
            job_timeout_s=args.job_timeout,
            busy_timeout_s=args.busy_timeout,
        )
        rows.append(row)
        # After every variant, so a run stopped halfway still has its table.
        _write(args.out, rows, plan, capture)
        _say(
            f"{variant.name}: {row['status']} psnr={row['psnr']} gaussians={row['gaussians']} "
            f"cost=${row['stageCostUsd']}"
        )
    _write(args.out, rows, plan, capture)
    sys.stdout.write(f"wrote {args.out.with_suffix('.md')} and {args.out.with_suffix('.json')}\n")
    return 0 if all(row.get("status") == "complete" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

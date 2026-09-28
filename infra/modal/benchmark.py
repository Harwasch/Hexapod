"""Calibrate the `train` stage on a scene with published numbers, on a Modal GPU. Opt-in.

What `smoke.py` does for "does the GPU path work", this does for "is it as good as it
should be": the production `train` stage (CloudRunner -> ModalAdapter -> the deployed
`run_stage_<tier>` -> `remote.execute` in the training image, bytes through the private
R2 bucket) on Tanks and Temples `truck`, with the paper's own poses, and the result set
beside 3DGS's published PSNR / SSIM / LPIPS. The scene, the archive's pinned sha256, the
numbers and where they were read are in `tools/pipeline/experiments/benchmark.py`.

It is never run by default. `.github/workflows/modal-benchmark.yml` runs it when the
head commit's message carries a `[benchmark]` token (or `[benchmark:recipe]`, or
`[benchmark:gsplat-default+pose_opt]` -- see `benchmark.parse_spec`). Cost, estimated
rather than measured -- it has not run yet: 30k steps of the `default` strategy on 251
frames at 979x546 is ~15-30 min of L4 at $0.80/h, $0.20-0.40, plus the cold start the
smoke already pays; the runner downloads the 683 MB archive in about half a minute.

Needs what `smoke.py` needs (Modal token, the `OBJECT_STORAGE_*` variables) and no
COLMAP: there is no pose stage.

    uv run --project tools/pipeline --with modal==1.5.5 --with boto3 \\
        python infra/modal/benchmark.py --scene truck --spec gsplat-default --tier l4

`--rehearse --archive <zip>` runs everything but Modal on a local archive with the test
suite's stand-in trainer: it proves the extraction, the recipe and the report, not
training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PIPELINE = REPO / "tools" / "pipeline"
sys.path.insert(0, str(PIPELINE))

import stages  # noqa: E402,F401 - registers every shipped implementation
from cloud import AttemptLedger, CloudRunner, Placement  # noqa: E402
from executor import execute  # noqa: E402
from experiments import benchmark  # noqa: E402
from recipe import Recipe  # noqa: E402
from runners import RunnerSet  # noqa: E402
from workdir import Workdir  # noqa: E402


def _s3_transfer() -> object:
    """The container's own `S3Transfer`, from `app.py`, as `smoke.py` builds it."""
    spec = importlib.util.spec_from_file_location("modal_app", Path(__file__).parent / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_app"] = module
    spec.loader.exec_module(module)
    return module._transfer()


def _download(url: str, target: Path) -> Path:
    """Stream `url` to `target`, unless a file of the right size is already there."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    sys.stdout.write(f"benchmark: downloading {url}\n")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as sink:  # noqa: S310 - pinned https URL
        shutil.copyfileobj(response, sink, length=1 << 22)
    partial.replace(target)
    return target


def cloud(work: Path, *, app: str, rehearse: bool) -> tuple[object, CloudRunner]:
    """The transfer and the `CloudRunner` a driver trains through: the deployed Modal
    app and the private R2 bucket, or -- rehearsing -- a subprocess and a directory.

    Shared with `minnetonka.py`, so a scene with known poses goes to the GPU one way.
    """
    if rehearse:
        from adapters import LocalTransfer, SubprocessAdapter

        transfer: object = LocalTransfer(work / "bucket")
        adapter: object = SubprocessAdapter(transfer, work / "sandbox")  # type: ignore[arg-type]
        poll = 0.2
    else:
        from modal_adapter import ModalAdapter

        transfer = _s3_transfer()
        adapter = ModalAdapter(app)
        poll = 15.0
    return transfer, CloudRunner(Placement.of(adapter), transfer, poll_interval_s=poll)  # type: ignore[arg-type]


def execute_remotely(
    recipe: Recipe, workdir: Workdir, transfer: object, runner: CloudRunner
) -> None:
    """`execute` on the cloud runner, then the run's own `runs/<id>` prefix removed from
    the bucket whatever happened -- what the run keeps, the caller copies out first."""
    run_id = workdir.root.name
    try:
        execute(recipe, workdir, RunnerSet.cloud(runner))
    finally:
        try:
            transfer.delete(f"runs/{run_id}")  # type: ignore[attr-defined]
        except Exception as error:  # cleanup must not hide the run's own failure
            sys.stderr.write(f"cloud: could not delete runs/{run_id}: {error!r}\n")


def stand_in_params(extra_args: list[str]) -> dict[str, object]:
    """Rehearsal: the test suite's stand-in trainer in place of gsplat. Trains nothing."""
    return {
        "trainer": str(PIPELINE / "tests" / "gsplat_stand_in.py"),
        "python": sys.executable,
        "iterations": 300,
        "extra_args": [*extra_args, "--gaussians", "2000"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", default="truck", choices=sorted(benchmark.SCENES))
    parser.add_argument(
        "--spec", default="", help="CONFIG[+switch...], e.g. gsplat-default or recipe+pose_opt"
    )
    parser.add_argument("--tier", default="l4")
    parser.add_argument("--work", type=Path, default=Path("benchmark-work"))
    parser.add_argument("--archive", type=Path, help="an already-downloaded archive")
    parser.add_argument("--app", default="twin-pipeline")
    parser.add_argument("--rehearse", action="store_true")
    args = parser.parse_args()

    scene = benchmark.SCENES[args.scene]
    config, switches = benchmark.parse_spec(args.spec)
    work = args.work.resolve()
    archive = args.archive
    if archive is None:
        archive = work / "archive" / Path(scene.archive_url).name
        if not (archive.is_file() and archive.stat().st_size == scene.archive_bytes):
            _download(scene.archive_url, archive)

    run_id = f"bench-{scene.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    workdir = Workdir.create(work / run_id)
    # -> inputs/frames and inputs/poses, the two artifacts the train stage consumes.
    benchmark.extract(archive, scene, workdir.inputs_dir, verify=not args.rehearse)
    params = benchmark.train_params(scene, config, switches)
    recipe = Recipe.from_dict(
        {
            "name": f"benchmark-{scene.name}",
            "version": 1,
            "inputs": ["frames", "poses"],
            "stages": [
                {
                    "id": "train",
                    "impl": "gsplat",
                    "params": params,
                    "gpu": {"tier": args.tier, "preemptible": False},
                }
            ],
        }
    )
    if args.rehearse:
        recipe = recipe.with_params({"train": stand_in_params(list(params["extra_args"]))})
    transfer, runner = cloud(work, app=args.app, rehearse=args.rehearse)
    started = time.monotonic()
    execute_remotely(recipe, workdir, transfer, runner)

    metrics = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    ledger = AttemptLedger.read(workdir.attempts_path("train"))
    result = benchmark.compare(scene, config, metrics, switches)
    result["tier"] = args.tier
    result["wallSeconds"] = round(time.monotonic() - started, 1)
    cost = {"tier": args.tier, "billedSeconds": round(ledger.billed_s, 1), "usd": ledger.usd}
    result["cost"] = cost
    report = benchmark.markdown(result, cost=cost)
    if args.rehearse:
        report = (
            "> Rehearsal: the stand-in trained nothing; the numbers are made up.\n\n" + report
        )
    (workdir.root / "benchmark.json").write_text(json.dumps(result, indent=1) + "\n")
    (workdir.root / "benchmark.md").write_text(report)
    sys.stdout.write(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(report)
    if result["verdict"] == "fail":
        sys.stderr.write(f"benchmark: FAILED -- {json.dumps(result['delta'])}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

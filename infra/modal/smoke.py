"""Prove the GPU path, cheaply: a real `train` stage on Modal, dispatched the production way.

Everything here is the production code path except the input. The frames are 40 renders
of the committed synthetic tree (`tools/pipeline/tests/tree_frames.py`, 640x480) -- no
dataset to download, no licence to check, and a scene whose answer is known. Then:

    pose        COLMAP, on this machine (the CI runner), exactly as the worker runs it
    train       gsplat on a Modal GPU: CloudRunner -> ModalAdapter.submit -> the deployed
                run_stage_<tier> -> remote.execute in the training image, the bytes moving
                through the private R2 bucket exactly as a worker's would
    georeference, place, package, thumbnail   back on this machine

and it asserts on what came back: a `trained.ply` with gaussians in it, metrics read from
gsplat's own stats file, a packaged tileset, and an attempt ledger with billed seconds.
The transfer keys live under `runs/<run id>/` in the private bucket and are deleted at
the end, pass or fail.

Needs: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (read by the Modal SDK), the
OBJECT_STORAGE_* variables the Modal secret `twin-object-storage` holds, `colmap` on PATH,
and `modal` and `boto3` importable. `.github/workflows/modal.yml` provides all of them.

    uv run --project tools/pipeline --with modal==1.5.5 --with boto3 \\
        python infra/modal/smoke.py --iterations 500 --tier l4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PIPELINE = REPO / "tools" / "pipeline"
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "tests"))

import gaussians  # noqa: E402
import stages  # noqa: E402,F401 - registers every shipped implementation
import tree_frames  # noqa: E402
from cloud import AttemptLedger, CloudRunner, Placement  # noqa: E402
from executor import execute  # noqa: E402
from modal_adapter import ModalAdapter  # noqa: E402
from recipe import Recipe  # noqa: E402
from runners import RunnerSet  # noqa: E402
from workdir import Workdir  # noqa: E402

FIXTURE_PLY = REPO / "data" / "tiles" / "synthetic-tree" / "source" / "splat.ply"


def _s3_transfer() -> object:
    """The container's own `S3Transfer`, from `app.py`, so both ends move bytes one way."""
    spec = importlib.util.spec_from_file_location("modal_app", Path(__file__).parent / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_app"] = module
    spec.loader.exec_module(module)
    return module._transfer()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--tier", default="l4")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--work", type=Path, default=Path("smoke-work"))
    parser.add_argument("--app", default="twin-pipeline")
    parser.add_argument(
        "--rehearse",
        action="store_true",
        help=(
            "everything but Modal: a local subprocess instead of a GPU container, a "
            "directory instead of R2, and the test suite's stand-in instead of gsplat. "
            "Proves the recipe, the transfer keys and the placement -- not training"
        ),
    )
    args = parser.parse_args()

    run_id = f"smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    workdir = Workdir.create(args.work.resolve() / run_id)
    frames = workdir.input_path("frames")
    tree_frames.render_orbit(FIXTURE_PLY, frames, count=args.frames)

    recipe = Recipe.from_dict(
        {
            "name": "modal-smoke",
            "version": 1,
            "inputs": ["frames"],
            "stages": [
                {"id": "pose", "impl": "colmap", "params": {"matcher": "exhaustive"}},
                {
                    "id": "train",
                    "impl": "gsplat",
                    "params": {"iterations": args.iterations},
                    "gpu": {"tier": args.tier, "preemptible": False},
                },
                {
                    "id": "georeference",
                    "impl": "exif_gps",
                    # The rendered frames carry no EXIF, so this is the capture-coordinate
                    # fallback a GPS-less phone video takes: the synthetic tree's own spot.
                    "params": {"lat": 28.0389, "lon": -82.6966},
                },
                {"id": "place", "impl": "place_splat"},
                {"id": "package", "impl": "splat_tiles"},
                {"id": "thumbnail", "impl": "splat_thumbnail"},
            ],
        }
    )
    if args.rehearse:
        from adapters import LocalTransfer, SubprocessAdapter

        transfer: object = LocalTransfer(args.work.resolve() / "bucket")
        adapter: object = SubprocessAdapter(transfer, args.work.resolve() / "sandbox")  # type: ignore[arg-type]
        stand_in = PIPELINE / "tests" / "gsplat_stand_in.py"
        train = recipe.stages[1]
        recipe = recipe.with_params(
            {
                train.id: {
                    "trainer": str(stand_in),
                    "python": sys.executable,
                    "extra_args": ["--gaussians", "2000"],
                }
            }
        )
        poll = 0.2
    else:
        transfer = _s3_transfer()
        adapter = ModalAdapter(args.app)
        poll = 10.0
    cloud = CloudRunner(Placement.of(adapter), transfer, poll_interval_s=poll)  # type: ignore[arg-type]
    started = time.monotonic()
    try:
        execute(recipe, workdir, RunnerSet.cloud(cloud))
    finally:
        # The transfer keys are this run's and nobody else's; leave the bucket as found.
        try:
            transfer.delete(f"runs/{run_id}")  # type: ignore[attr-defined]
        except Exception as error:  # cleanup must not hide the run's own failure
            sys.stderr.write(f"smoke: could not delete runs/{run_id}: {error!r}\n")

    metrics = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    trained = gaussians.read_splat(workdir.out_dir("train") / "trained.ply")
    ledger = AttemptLedger.read(workdir.attempts_path("train"))
    tileset = workdir.out_dir("package") / "splat" / "tileset.json"
    summary = {
        "runId": run_id,
        "tier": args.tier,
        "iterations": metrics.get("iterations"),
        "requestedIterations": args.iterations,
        "gaussians": trained.count,
        "psnr": metrics.get("psnr"),
        "ssim": metrics.get("ssim"),
        "metricsSource": metrics.get("source"),
        "attempts": [entry.to_dict() for entry in ledger.entries],
        "billedSeconds": round(ledger.billed_s, 1),
        "usd": ledger.usd,
        "wallSeconds": round(time.monotonic() - started, 1),
        "tileset": tileset.is_file(),
    }
    sys.stdout.write(json.dumps(summary, indent=1) + "\n")
    report = os.environ.get("GITHUB_STEP_SUMMARY")
    if report:
        with open(report, "a", encoding="utf-8") as handle:
            handle.write(
                "## Modal GPU smoke\n\n```json\n" + json.dumps(summary, indent=1) + "\n```\n"
            )

    if args.rehearse:
        sys.stdout.write("smoke: rehearsal only -- the stand-in trained nothing\n")
    problems = []
    if trained.count < 1000:
        problems.append(f"only {trained.count} gaussians came back")
    if metrics.get("source") != "stats" or metrics.get("psnr") is None:
        problems.append(f"no PSNR read from gsplat's stats file: {metrics}")
    if metrics.get("iterations") != args.iterations:
        problems.append(f"trained {metrics.get('iterations')} of {args.iterations} steps")
    if not tileset.is_file():
        problems.append("the placed splat did not package")
    for problem in problems:
        sys.stderr.write(f"smoke: FAILED -- {problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

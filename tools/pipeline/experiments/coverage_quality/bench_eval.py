"""Evaluation-only adapter for frozen bench runs. Execute inside GSPLAT_PYTHON only.

No private data leaves /bench: output contains grid sufficient statistics, photometric
metrics and metadata, never source images, PLYs, or rendered RGB/depth images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import patch_table, write_outputs
from .fitting import compact_rows, plot_fit
from .gpu import camera_from_item, scene_from_ply
from .progress import Progress
from .renderers import GsplatBackend
from .selection import (
    choose_test_views,
    frame_metadata,
    metric_scale,
    select_frames,
    spread_test_views,
)
from .views import image_ratio, metadata_cameras, read_views, single_view_source


def optional(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def require_lpips_weights(hub: Path) -> Path:
    weights = hub / "checkpoints/alexnet-owt-7be5be79.pth"
    if not weights.is_file():
        raise FileNotFoundError(f"offline LPIPS weights absent: {weights}; no download allowed")
    return weights


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    progress = Progress(args.time_budget)
    rows: list[dict[str, Any]] = []
    try:
        return _evaluate(args, progress, rows)
    except Exception as error:
        progress.log(f"FAIL {type(error).__name__}: {error}; writing partial derived outputs")
        if not (args.out / "patches.csv").is_file():
            write_outputs(compact_rows(rows, args.label), args.out, fit=False)
        previous = optional(args.out / "provenance.json")
        partial = {
            **previous,
            "status": "partial",
            "error": f"{type(error).__name__}: {error}",
            "scene": args.label,
            "rows": previous.get("rows", len(rows)),
            "phaseSeconds": progress.used,
            "seconds": time.monotonic() - progress.started,
        }
        (args.out / "provenance.json").write_text(json.dumps(partial, indent=2) + "\n")
        raise


def _evaluate(
    args: argparse.Namespace, progress: Progress, row_sink: list[dict[str, Any]]
) -> dict[str, Any]:
    with progress.phase("setup", "metadata and GPU imports"):
        import distortion
        import training

        started = progress.started
        run, frozen, output = args.run, args.scene, args.out
        result = optional(run / "result.json")
        if result.get("status") != "complete":
            raise ValueError("bench training run is not complete")
        params = result.get("trainParams") or {}
        test_report = optional(run / "test.json")
        renderer = result.get(
            "testRenderer", "antialiased" if params.get("antialiased") else "classic"
        )
        if renderer not in ("classic", "antialiased"):
            raise ValueError("unsupported recorded test renderer")
        if args.label == "beach" and renderer != "classic":
            raise ValueError("recorded Beach run requires classic renderer")
        scale, scale_source = metric_scale(
            {
                "manifest": optional(frozen / "manifest.json"),
                "full_data": optional(frozen / "full_data.json"),
                "poses": optional(frozen / "poses/poses.json"),
            },
            attested_scale=args.metres_per_unit,
            attested_source=args.scale_source,
        )
        trainer = Path(os.environ["GSPLAT_TRAINER"])
        sys.path.insert(0, str(trainer.parent))
        import torch
        from datasets.colmap import Dataset, Parser
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        torch.set_num_threads(4)
    degree = int(params.get("ship_sh_degree", test_report.get("shDegree", 3)))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="coverage-eval-") as temporary:
        with progress.phase("setup", "camera metadata only; skip SfM points/tracks and dense maps"):
            views = read_views(frozen / "poses")
            train, test, selection_source = select_frames(
                list(views.images),
                optional(run / "frame_selection.json"),
                optional(frozen / "split.json"),
                result,
                test_report,
            )
            test = spread_test_views(choose_test_views(sorted(test), args.max_test_frames))
            metadata = frame_metadata(optional(frozen / "frames_meta.json"))
            ratio = image_ratio(
                views, frozen / "frames", params.get("train_max_side"), reference=sorted(train)[0]
            )
            rectify = distortion.enabled(frozen / "poses")
            progress.log(
                f"metadata read: {len(views.images)} poses / {len(views.cameras)} calibrations; "
                "no sparse points, tracks, training RGB or dense maps loaded; "
                f"models={sorted({c.model for c in views.cameras.values()})}"
            )
        with progress.phase("cameras", f"{len(train)} training / {len(test)} test"):
            cameras = metadata_cameras(
                views, train, ratio, metadata, scale, explicit_rectification=rectify
            )
            test_cameras = {
                c.name: c
                for c in metadata_cameras(
                    views, test, ratio, metadata, scale, explicit_rectification=rectify
                )
            }
        ply = run / "trained.ply"
        with progress.phase("ply", str(ply)):
            scene = scene_from_ply(ply, degree)
            if scene.sh_degree != degree:
                raise ValueError("trained.ply SH bands disagree with recorded shipping degree")
            scene.xyz *= scale
            scene.scales *= scale
            progress.log(
                f"PLY loaded once: {len(scene.xyz):,} splats; SH={degree}; "
                f"renderer={renderer}; packed={params.get('packed', True)}"
            )
        with progress.phase("lpips", "offline cached AlexNet initialization"):
            weights = require_lpips_weights(Path(torch.hub.get_dir()))
            progress.log(f"LPIPS baked weights: {weights}; downloads forbidden")
            previous_download = torch.hub.download_url_to_file

            def refuse_download(*a: Any, **kw: Any) -> None:
                progress.log("LPIPS attempted a download: refusing immediately")
                raise RuntimeError("offline LPIPS: image must contain all weights")

            torch.hub.download_url_to_file = refuse_download
            try:
                perceptual = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=True
                ).cuda()
            finally:
                torch.hub.download_url_to_file = previous_download

        def lpips(a: np.ndarray, b: np.ndarray) -> float:
            def tensor(value: np.ndarray) -> Any:
                return torch.from_numpy(value.astype(np.float32)).permute(2, 0, 1)[None].cuda()

            with torch.no_grad():
                perceptual.reset()
                return float(perceptual(tensor(a), tensor(b)).item())

        def tests() -> Any:
            for name in test:
                progress.log(f"test load {name}; build only this view's image/remap")
                with (
                    progress.phase("test_render", f"single-view assembly/decode {name}"),
                    tempfile.TemporaryDirectory(dir=temporary, prefix="view-") as one,
                ):
                    root = Path(one)
                    single_view_source(views, name, frozen, root / "source")
                    data = training.build_dataset(
                        root / "source/frames",
                        root / "source/poses",
                        root / "data",
                        max_side=params.get("train_max_side"),
                    )
                    parsed = Parser(str(data), factor=1, normalize=False, test_every=1)
                    dataset = Dataset(parsed, split="val")
                    if parsed.image_names != [name]:
                        raise ValueError("single-view Parser admitted unexpected frames")
                    item = dataset[0]
                    camera = camera_from_item(name, item, metadata.get(name, {}))
                    camera.c2w = camera.c2w.copy()
                    camera.c2w[:3, 3] *= scale
                    expected = test_cameras[name]
                    tolerance = 4 * np.finfo(np.float32).eps
                    if (camera.width, camera.height) != (
                        expected.width,
                        expected.height,
                    ) or not np.allclose(camera.k, expected.k, rtol=tolerance, atol=tolerance):
                        raise ValueError("single-view trainer K/ROI differs from metadata geometry")
                    truth = item["image"].numpy().astype(np.float32) / 255
                    mask = item["mask"].numpy().astype(bool) if "mask" in item else None
                    del parsed, dataset  # Release dense maps before the next view.
                yield camera, truth, mask

        completed: list[str] = []
        checkpoint_count = 0
        checkpoint_rows: list[dict[str, Any]] = []

        def checkpoint(rows: list[dict[str, Any]], name: str) -> None:
            nonlocal checkpoint_count
            with progress.phase("write", f"compact newly completed patches {name}"):
                fresh = compact_rows([dict(row) for row in rows[checkpoint_count:]], args.label)
                checkpoint_rows.extend(fresh)
            snapshot = checkpoint_rows
            if not completed:
                with progress.phase("fit", "first completed view checkpoint"):
                    write_outputs(snapshot, output)
            else:
                with progress.phase("write", f"incremental table after {name}"):
                    write_outputs(snapshot, output, fit=False)
            completed.append(name)
            checkpoint_count = len(rows)
            (output / "provenance.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "scene": args.label,
                        "rows": len(snapshot),
                        "completedTestNames": completed,
                        "testNames": test,
                        "metresPerUnit": scale,
                        "scaleSource": scale_source,
                        "scaleUncertain": args.label == "beach",
                        "fitNote": "first-view checkpoint until final/partial refit",
                        "phaseSeconds": progress.used,
                    },
                    indent=2,
                )
                + "\n"
            )
            progress.log(f"checkpoint persisted: {len(snapshot)} rows / {len(completed)} views")

        backend = GsplatBackend(
            antialiased=renderer == "antialiased", packed=bool(params.get("packed", True))
        )
        stopped = None
        try:
            patch_table(
                scene,
                cameras,
                tests(),
                backend,
                lpips=lpips,
                max_patches_per_view=args.max_patches,
                progress=progress,
                rows=row_sink,
                on_view=checkpoint,
            )
        except TimeoutError as error:
            if checkpoint_count < 8:
                raise
            stopped = str(error)
            del row_sink[checkpoint_count:]  # Fits use fully completed test views only.
            progress.log(f"evaluation stopping for final partial fit: {stopped}")
        rows = checkpoint_rows[:checkpoint_count]
        with progress.phase("fit", f"{len(rows)} measured rows"):
            fits = write_outputs(rows, output)
            if not fits or not fits.get("error"):
                raise ValueError("rehearsal/evaluation yielded too few usable patches to fit")
        with progress.phase("write", "plots and provenance"):
            plot_fit(rows, fits, output)
        if not (output / "fit_error.png").is_file():
            raise ValueError("no monotone combined-error fit; inspect derived table")
        with progress.phase("write", "PLY checksum"), ply.open("rb") as stream:
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(1 << 20), b""):
                checksum.update(block)
        provenance = {
            "status": "partial" if stopped else "complete",
            "stopReason": stopped,
            "completedTestNames": completed,
            "scene": args.label,
            "run": run.name,
            "imageId": os.environ.get("COVERAGE_IMAGE_ID"),
            "renderer": renderer,
            "shDegree": degree,
            "trainingNames": train,
            "testNames": test,
            "selectionSource": selection_source,
            "metresPerUnit": scale,
            "scaleSource": scale_source,
            "scaleUncertain": args.label == "beach",
            "metadataReader": "COLMAP camera/image headers only; sparse points and tracks skipped",
            "cameraModels": sorted({c.model for c in views.cameras.values()}),
            "pixelReader": "one held-out image at a time through trainer Parser/Dataset",
            "readerSha256": hashlib.sha256(
                Path(__file__).with_name("views.py").read_bytes()
            ).hexdigest(),
            "parserSha256": hashlib.sha256(
                (trainer.parent / "datasets/colmap.py").read_bytes()
            ).hexdigest(),
            "plySha256": checksum.hexdigest(),
            "trainerSha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
            "trainParams": params,
            "gaussians": len(scene.xyz),
            "patchM": 0.1,
            "maxPatchesPerView": args.max_patches,
            "rows": len(rows),
            "kernelEncoding": "exact credit sums for grid, plus counts; no angle quantization",
            "kernelGrid": "cones=10,20,25,30,45,60; powers=.5,1,2,4; cone-major",
            "exposureField": "frames_meta exposure.ev100 (EV); absent stays null",
            "blurField": "frames_meta blur; absent stays null",
            "phaseSeconds": progress.used,
            "visibility": (
                "all training cameras; frustum prefilter, batch=4, side=160; once/run depth cache"
            ),
            "lpipsWeights": str(weights),
            "seconds": time.monotonic() - started,
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        progress.log(f"write complete: {len(rows)} rows -> {output}")
        return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-budget", type=float, default=360)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", choices=("c935", "beach"), required=True)
    parser.add_argument("--max-test-frames", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=64)
    parser.add_argument("--metres-per-unit", type=float)
    parser.add_argument("--scale-source")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

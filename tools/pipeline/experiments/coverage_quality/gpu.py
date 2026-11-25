"""Later GPU calibration entry point. Importing this module never constructs a Modal app.

Requires a private, assembled gsplat dataset in the trained PLY's metric frame. Run
under GSPLAT_PYTHON, with GSPLAT_TRAINER set, as holdout_error.py does. No URLs, S3 keys,
API calls or stage/recipe registration. The optional Modal wrapper uses a research
volume and the existing pinned training image, GPU names and reservation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import patch_table, write_outputs
from .renderers import Camera, GsplatBackend, Scene


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--ply", type=Path)
    result.add_argument(
        "--metres-per-unit",
        type=float,
        required=True,
        help="verified model scale; ARKit c935 = 1; unresolved scale is blocked",
    )
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--trainer", type=Path, default=os.environ.get("GSPLAT_TRAINER"))
    result.add_argument(
        "--trained-test-every",
        type=int,
        required=True,
        help="actual training split, 0 for c935; never invent a held-out split",
    )
    result.add_argument("--retrain-with-split", action="store_true")
    result.add_argument("--test-every", type=int, default=8)
    result.add_argument("--steps", type=int, default=60000)
    result.add_argument("--cap", type=int, default=5626152)
    result.add_argument("--sh-degree", type=int, default=3, choices=range(4))
    result.add_argument("--data-factor", type=int, default=1)
    result.add_argument("--antialiased", action="store_true")
    result.add_argument("--unpacked", action="store_true")
    result.add_argument("--max-patches", type=int, default=256)
    result.add_argument(
        "--max-test-frames",
        type=int,
        default=0,
        help="0 = all; otherwise deterministic evenly spaced subset",
    )
    result.add_argument(
        "--external-test-dataset",
        type=Path,
        help="genuinely unused frames, with poses registered in SAME metric frame",
    )
    result.add_argument(
        "--frame-meta",
        type=Path,
        help="optional JSON name -> {blur, exposure}; absent remains null",
    )
    return result


def scene_from_ply(path: Path, degree: int) -> Scene:
    import holdout_error

    columns = holdout_error.read_ply(path)
    xyz = np.stack([columns[k] for k in ("x", "y", "z")], 1)
    scales = np.exp(np.stack([columns[f"scale_{i}"] for i in range(3)], 1))
    quats = np.stack([columns[f"rot_{i}"] for i in range(4)], 1)
    opacity = 1 / (1 + np.exp(-np.clip(columns["opacity"].astype(float), -80, 80)))
    dc = np.stack([columns[f"f_dc_{i}"] for i in range(3)], 1)[:, None]
    colors, used = holdout_error.shipped_sh(dc, holdout_error.sh_rest(columns), degree)
    return Scene(xyz, scales, quats, opacity, colors, used)


def camera_from_item(name: str, item: dict[str, Any], meta: dict[str, Any]) -> Camera:
    h, w = item["image"].shape[:2]
    return Camera(
        name,
        item["camtoworld"].numpy(),
        item["K"].numpy(),
        w,
        h,
        float(meta["blur"]) if meta.get("blur") is not None else float("nan"),
        float(meta["exposure"]) if meta.get("exposure") is not None else float("nan"),
    )


def make_dataset(data: Any, factory: Any, split: str) -> Any:
    """Mirror train_split.py's all-training seam; upstream Dataset cannot modulo zero."""
    if data.test_every != 0:
        return factory(data, split=split)
    data.test_every = 1
    try:
        dataset = factory(data, split=split)
    finally:
        data.test_every = 0
    dataset.indices = (
        np.arange(len(data.image_names)) if split == "train" else np.array([], dtype=int)
    )
    return dataset


def run(args: argparse.Namespace) -> dict[str, Any]:
    import training

    if args.trainer is None:
        raise ValueError("--trainer or GSPLAT_TRAINER required")
    if args.test_every < 2 or args.trained_test_every == 1 or args.trained_test_every < 0:
        raise ValueError("test_every >= 2; trained-test-every is 0 or >= 2")
    if args.trained_test_every == 0 and not (args.retrain_with_split or args.external_test_dataset):
        raise ValueError("all-frame model: retrain with a split or supply truly unused frames")
    if args.metres_per_unit <= 0 or not np.isfinite(args.metres_per_unit):
        raise ValueError("verified positive metres-per-unit required")
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.trainer.resolve().parent))
    import torch
    from datasets.colmap import Dataset, Parser
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    torch.set_num_threads(4)
    split = args.test_every if args.retrain_with_split else args.trained_test_every
    data = Parser(str(args.dataset), factor=args.data_factor, normalize=False, test_every=split)
    metadata = json.loads(args.frame_meta.read_text()) if args.frame_meta else {}
    trainset = make_dataset(data, Dataset, "train")
    train_names = [str(data.image_names[int(i)]) for i in trainset.indices]
    started = time.monotonic()
    provenance: dict[str, Any] = {
        "trainingNames": train_names,
        "testEvery": split,
        "metricFrameRequired": True,
        "metresPerUnit": args.metres_per_unit,
        "antialiased": args.antialiased,
        "packed": not args.unpacked,
        "dataFactor": args.data_factor,
        "trainer": str(args.trainer),
        "shDegree": args.sh_degree,
        "splitEvidence": "fresh retrain"
        if args.retrain_with_split
        else "operator training provenance",
    }
    if args.retrain_with_split:
        result_dir = args.out / "retrain"
        if result_dir.exists():
            raise ValueError("fresh retrain output already exists; use a new --out")
        argv = training.gsplat_argv(
            sys.executable,
            args.trainer,
            args.dataset,
            result_dir,
            strategy="mcmc",
            max_steps=30000,
            steps_scaler=args.steps / 30000,
            data_factor=args.data_factor,
            cap_max=args.cap,
            antialiased=args.antialiased,
            packed=not args.unpacked,
            extra=["--test_every", str(split), "--sh_degree", str(args.sh_degree)],
        )
        provenance["retrainArgv"] = argv
        # Trusted repository-built argv, never a command from an artifact/JSON file.
        subprocess.run(argv, check=True, timeout=6 * 3600)  # noqa: S603
        args.ply = training.latest_ply(result_dir)
    if args.ply is None or not args.ply.is_file():
        raise ValueError("a trained PLY is required")
    # Streaming hash: do not double memory on a multi-GB trained PLY.
    with args.ply.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        provenance["plySha256"] = digest.hexdigest()
    scene = scene_from_ply(args.ply, args.sh_degree)
    scene.xyz *= args.metres_per_unit
    scene.scales *= args.metres_per_unit
    # Training dataset returns decoded RGB too, but only keep camera metadata.
    cameras = [
        camera_from_item(name, trainset[j], metadata.get(name, {}))
        for j, name in enumerate(train_names)
    ]
    for camera in cameras:
        camera.c2w[:3, 3] *= args.metres_per_unit
    if args.external_test_dataset:
        test_data = Parser(
            str(args.external_test_dataset), factor=args.data_factor, normalize=False, test_every=0
        )
        testset = make_dataset(test_data, Dataset, "train")
        if set(train_names) & set(test_data.image_names):
            raise ValueError(
                "external test names overlap training; use unique, provenance-backed names"
            )
        provenance["externalTestFrameWarning"] = (
            "Operator must verify unused images and common frame"
        )
    else:
        test_data, testset = data, make_dataset(data, Dataset, "val")
    names = [str(test_data.image_names[int(i)]) for i in testset.indices]
    if not names:
        raise ValueError("no held-out images")
    chosen = (
        np.linspace(0, len(names) - 1, min(args.max_test_frames, len(names)), dtype=int)
        if args.max_test_frames
        else np.arange(len(names))
    )
    provenance["testNames"] = [names[i] for i in chosen]
    provenance["gaussians"] = len(scene.xyz)
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # torchmetrics uses the trainer's AlexNet LPIPS. Pre-stage its weights in the image
    # cache before an offline run; never substitute an L1 score under the LPIPS name.
    perceptual = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()

    def lpips(a: np.ndarray, b: np.ndarray) -> float:
        def tensor(image: np.ndarray) -> Any:
            return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None].cuda()

        with torch.no_grad():
            return float(perceptual(tensor(a), tensor(b)).item())

    def tests() -> Any:
        # Generator prevents all native-resolution test RGBs residing in memory.
        for index in chosen:
            item = testset[int(index)]
            name = names[int(index)]
            camera = camera_from_item(name, item, metadata.get(name, {}))
            camera.c2w[:3, 3] *= args.metres_per_unit
            truth = item["image"].numpy().astype(np.float32) / 255
            mask = item["mask"].numpy().astype(bool) if "mask" in item else None
            yield camera, truth, mask

    backend = GsplatBackend(antialiased=args.antialiased, packed=not args.unpacked)
    rows = patch_table(
        scene, cameras, tests(), backend, lpips=lpips, max_patches_per_view=args.max_patches
    )
    fits = write_outputs(rows, args.out)
    plots(rows, fits, args.out)
    provenance.update(seconds=time.monotonic() - started, rows=len(rows))
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return fits


def plots(rows: list[dict[str, Any]], fits: dict[str, Any], out: Path) -> None:
    """GPU image may add matplotlib, isolated from the CPU dry-run dependencies."""
    import importlib

    plt = importlib.import_module("matplotlib.pyplot")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, key in zip(axes, ("nearest_deg", "best_mm_px", "parallax_deg"), strict=True):
        ax.scatter([r[key] for r in rows], [r["error"] for r in rows], s=2, alpha=0.15)
        ax.set(xlabel=key, ylabel="held-out 0.8 L1 + 0.2 (1-SSIM)")
    fig.suptitle(f"Exploratory fit R² = {fits['error'].get('r2')}")
    fig.tight_layout()
    fig.savefig(out / "error_geometry.png", dpi=120)
    plt.close(fig)


def build_modal_app() -> tuple[Any, dict[str, Any]]:
    """Explicit opt-in construction for later deployment; never called by CPU tools.

    Reuse infra/modal/app.py's pinned image (including both sibling source trees),
    reservation and tier names. A mounted research volume carries prepared inputs and
    outputs; no production Transfer/stage registration or bucket writes.
    """
    import modal

    from modal_adapter import GPU_NAMES, GPU_RESERVATION

    repo = Path(__file__).resolve().parents[4]
    source = repo / "infra/modal/app.py" if modal.is_local() else Path("/root/app.py")
    spec = importlib.util.spec_from_file_location("coverage_training_image", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load repository training image definition")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = module.image  # Trainer requirements already include matplotlib and LPIPS weights.
    app = modal.App("hex-coverage-quality-research")
    volume = modal.Volume.from_name("hex-coverage-quality-research", create_if_missing=False)

    def evaluate(arguments: list[str]) -> dict[str, Any]:
        # The outer image interpreter is 3.12, while CUDA dependencies are in 3.10.
        env = {
            **os.environ,
            "PYTHONPATH": "/opt/pipeline",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
        subprocess.run(  # noqa: S603 - explicit operator CLI, no shell or data commands
            [os.environ["GSPLAT_PYTHON"], "-m", "experiments.coverage_quality.gpu", *arguments],
            env=env,
            check=True,
            timeout=6 * 3600,
        )
        volume.commit()
        return {"status": "complete", "output": "research volume; see --out"}

    functions = {}
    for tier in ("l4", "l40s"):
        functions[tier] = app.function(
            image=image,
            gpu=GPU_NAMES[tier],
            cpu=GPU_RESERVATION[0],
            memory=GPU_RESERVATION[1],
            timeout=6 * 3600,
            retries=0,
            volumes={"/research": volume},
            name=f"calibrate_{tier}",
        )(evaluate)
    return app, functions


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()

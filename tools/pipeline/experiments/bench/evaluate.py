"""GPU evaluator, invoked as a script with GSPLAT_PYTHON (Python 3.10).

Keep imports here compatible with the trainer interpreter. The full dataset is built
outside the training workdir, after training. Parser sees its complete model/images;
only the frozen test names are fetched from Dataset. This avoids assuming Parser can
map a full COLMAP image list onto a directory containing only test images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from holdout_error import read_ply, sh_rest, shipped_sh


def region_info(scene: Path) -> dict[str, Any]:
    """Hook for future annotated masks. Undistortion masks are NOT subject masks."""
    candidates = [
        p.relative_to(scene).as_posix()
        for p in scene.rglob("*")
        if p.is_file() and any(word in p.name.lower() for word in ("mask", "region", "subject"))
    ]
    meta = scene / "frames_meta.json"
    if meta.is_file() and any(
        word in meta.read_text().lower() for word in ('"mask', '"region', '"subject')
    ):
        candidates.append("frames_meta.json: mask/region/subject metadata")
    return {
        "regions": None,
        "regionsReason": (
            "No per-frame region/subject masks in frozen inputs."
            if not candidates
            else "Mask candidates found; no annotated-mask adapter registered."
        ),
        "regionMaskCandidates": candidates,
    }


def full_sh_columns(
    shipped: dict[str, np.ndarray], raw_path: Path | None
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """Compare the same geometry. Never call an uncropped/different splat a colour gap.

    The raw export retains bands removed by the stage. Cropping preserves row order,
    so select exactly those original rows; no nearest-neighbour match or tolerance.
    If identity is ambiguous/different, leave the diagnostic unavailable.
    """
    if raw_path is None:
        return None, "No matching full-SH export (for example, a merged block run)."
    raw = read_ply(raw_path)
    identity = [
        "x",
        "y",
        "z",
        "opacity",
        *[f"scale_{i}" for i in range(3)],
        *[f"rot_{i}" for i in range(4)],
        *[f"f_dc_{i}" for i in range(3)],
    ]
    if not all(np.array_equal(shipped[k], raw[k]) for k in identity):

        def row_keys(columns: dict[str, np.ndarray]) -> np.ndarray:
            values = np.stack([columns[k] for k in identity], axis=1).astype("<f4")
            return np.ascontiguousarray(values).view(np.dtype((np.void, 4 * len(identity)))).ravel()

        keep = np.isin(row_keys(raw), row_keys(shipped))
        # Ambiguous duplicate rows (potentially different SH), reordering and altered
        # geometry all fail this exact check rather than manufacturing a colour gap.
        if not all(np.array_equal(shipped[k], raw[k][keep]) for k in identity):
            return None, "Raw export geometry differs from shipped PLY; full-SH gap unavailable."
        raw = {k: v[keep] for k, v in raw.items()}
    return raw, None


def lookup_frames(image_names: list[str], indices: Any, names: list[str]) -> dict[str, int]:
    """Each frozen test name -> its item in gsplat's dataset, or refuse the whole set."""
    if not names:
        raise ValueError("no test names")
    if len(set(names)) != len(names):
        raise ValueError("test names duplicated")
    items = {image_names[int(index)]: item for item, index in enumerate(indices)}
    missing = sorted(set(names) - items.keys())
    if missing:
        raise ValueError(f"test names missing from the gsplat dataset: {missing}")
    return {name: items[name] for name in names}


def summarise(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float | None]:
    """Mean of each metric over the frames that have it; None where none do."""
    summary: dict[str, float | None] = {}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def evaluate(
    data_dir: Path,
    ply: Path,
    raw_ply: Path | None,
    trainer: Path,
    names: list[str],
    degree: int,
    antialiased: bool,
) -> dict[str, Any]:
    sys.path.insert(0, str(trainer.resolve().parent))
    import torch
    from datasets.colmap import Dataset, Parser
    from gsplat import rasterization
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    columns = read_ply(ply)
    raw, reason = full_sh_columns(columns, raw_ply)
    device = torch.device("cuda")

    def tensor(keys: list[str]) -> Any:
        return torch.from_numpy(np.stack([columns[k] for k in keys], axis=1).astype(np.float32)).to(
            device
        )

    means = tensor(["x", "y", "z"])
    scales = tensor([f"scale_{i}" for i in range(3)]).exp()
    quats = tensor([f"rot_{i}" for i in range(4)])
    opacities = tensor(["opacity"])[:, 0].sigmoid()

    def colors(source: dict[str, np.ndarray], requested: int) -> tuple[Any, int]:
        dc = np.stack([source[f"f_dc_{i}"] for i in range(3)], axis=1)[:, None]
        coefficients, used = shipped_sh(dc, sh_rest(source), requested)
        return torch.from_numpy(coefficients).to(device), used

    shown, used_degree = colors(columns, degree)
    full = colors(raw, 3) if raw is not None else None
    parser = Parser(data_dir=str(data_dir), factor=1, normalize=False, test_every=1)
    dataset = Dataset(parser, split="val")
    lookup = lookup_frames(parser.image_names, dataset.indices, names)
    metrics = {
        "psnr": PeakSignalNoiseRatio(data_range=1.0).to(device),
        "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        "lpips": LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device),
    }
    per_frame = []
    with torch.no_grad():
        for name in names:
            data = dataset[lookup[name]]
            real = data["image"].to(device).float()[None] / 255.0
            height, width = real.shape[1:3]

            def render(
                coefficients: Any,
                sh_degree: int,
                data: Any = data,
                width: int = width,
                height: int = height,
            ) -> Any:
                rgb, _, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=coefficients,
                    viewmats=torch.linalg.inv(data["camtoworld"][None].to(device)),
                    Ks=data["K"][None].to(device),
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    packed=False,
                    rasterize_mode="antialiased" if antialiased else "classic",
                )
                rgb = rgb.clamp(0.0, 1.0)
                # Same invalid undistorted pixels as simple_trainer's rasterize_splats.
                if "mask" in data:
                    rgb = rgb * data["mask"].to(device)[None, ..., None]
                return rgb.permute(0, 3, 1, 2)

            prediction = render(shown, used_degree)
            target = real.permute(0, 3, 1, 2)
            row: dict[str, Any] = {"name": name}
            for key, metric in metrics.items():
                metric.reset()
                row[key] = float(metric(prediction, target).item())
            row["psnrFullSh"] = None
            if full is not None:
                metrics["psnr"].reset()
                row["psnrFullSh"] = float(metrics["psnr"](render(*full), target).item())
            row["psnrFullShGap"] = None if full is None else row["psnrFullSh"] - row["psnr"]
            per_frame.append(row)
    result: dict[str, Any] = {
        "frames": len(per_frame),
        "perFrame": per_frame,
        "shDegree": used_degree,
        "fullShReason": reason,
        "pixels": "whole image, black background; no coverage threshold",
        "lpipsNet": "alex",
        "lpipsNormalize": True,
    }
    result.update(summarise(per_frame, ("psnr", "ssim", "lpips", "psnrFullSh", "psnrFullShGap")))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "ply", "trainer", "split", "scene", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--raw-ply", type=Path)
    parser.add_argument("--degree", type=int, default=0)
    parser.add_argument("--antialiased", action="store_true")
    args = parser.parse_args()
    names = json.loads(args.split.read_text())["test"]
    result = evaluate(
        args.data, args.ply, args.raw_ply, args.trainer, names, args.degree, args.antialiased
    )
    result.update(region_info(args.scene))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

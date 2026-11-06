"""Nine classic gsplat renders; one model on the GPU, bounded PLY host buffers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import splat_io
from captures_bridge import unpack_spz
from experiments.bench.console_maths import VIEWS, psnr
from experiments.bench.evaluate import lookup_frames
from experiments.bench.scene import write_json
from holdout_error import sh_rest


def panels(output: Path, settings: dict[str, Any]) -> None:
    pairs = [(view, "trained", "snapshot", f"comparison-{view}.png") for view in VIEWS]
    pairs.append(("orbit", "trained", "baseline", "baseline-comparison-orbit.png"))
    for view, left, right, filename in pairs:
        images = [Image.open(output / f"{key}-{view}.png").convert("RGB") for key in (left, right)]
        width, height = images[0].size
        canvas = Image.new("RGB", (width * 2, height + 42), "black")
        draw = ImageDraw.Draw(canvas)
        for i, (key, image) in enumerate(zip((left, right), images, strict=True)):
            row = settings["sources"][key]
            canvas.paste(image, (i * width, 42))
            draw.text(
                (i * width + 4, 4),
                f"{row['runId'][:8]} {key} / {view}\n"
                f"{row['count']:,} splats / SH{row['shDegree']}"
                + (" / SYNTHETIC" if settings.get("rehearsal") else ""),
                fill="white",
            )
        canvas.save(output / filename)


def load_model(source: dict[str, Any]) -> dict[str, Any]:
    import torch

    count, degree = source["count"], source["shDegree"]
    shapes = {
        "means": (3,),
        "quats": (4,),
        "scales": (3,),
        "opacities": (),
        "colors": ((degree + 1) ** 2, 3),
    }
    tensors = {
        key: torch.empty((count, *shape), dtype=torch.float32, device="cuda")
        for key, shape in shapes.items()
    }
    path = Path(source["path"])
    chunks = (
        [(0, unpack_spz(path.read_bytes(), sh=True))]
        if path.suffix == ".spz"
        else splat_io.SplatReader(path).chunks()
    )
    for start, columns in chunks:

        def stack(keys: list[str], columns: dict[str, np.ndarray] = columns) -> np.ndarray:
            return np.stack([columns[k] for k in keys], axis=1).astype(np.float32)

        dc = stack([f"f_dc_{i}" for i in range(3)])[:, None]
        rest = sh_rest(columns)
        values = {
            "means": stack(["x", "y", "z"]),
            "quats": stack([f"rot_{i}" for i in range(4)]),
            "scales": stack([f"scale_{i}" for i in range(3)]),
            "opacities": np.asarray(columns["opacity"], dtype=np.float32),
            "colors": dc if rest is None else np.concatenate((dc, rest), axis=1),
        }
        for key, value in values.items():
            tensors[key][start : start + len(dc)].copy_(torch.from_numpy(value))
    tensors["scales"].exp_()
    tensors["opacities"].sigmoid_()
    return tensors


def render_model(model: dict[str, Any], camera: dict[str, Any], degree: int) -> np.ndarray:
    import torch
    from gsplat import rasterization

    with torch.no_grad():
        rgb, _, _ = rasterization(
            **model,
            viewmats=torch.tensor([camera["viewmat"]], dtype=torch.float32, device="cuda"),
            Ks=torch.tensor([camera["K"]], dtype=torch.float32, device="cuda"),
            width=camera["width"],
            height=camera["height"],
            near_plane=camera["near"],
            far_plane=camera["far"],
            sh_degree=degree,
            packed=True,
            rasterize_mode="classic",
        )
        return np.asarray(rgb[0].clamp(0, 1).cpu().numpy())


def cameras_from_dataset(settings: dict[str, Any], output: Path) -> dict[str, Any]:
    sys.path.insert(0, str(Path(settings["trainer"]).parent))
    from datasets.colmap import Dataset, Parser

    parser = Parser(data_dir=settings["data"], factor=1, normalize=False, test_every=1)
    dataset = Dataset(parser, split="val")
    lookup = lookup_frames(parser.image_names, dataset.indices, settings["frames"])
    cameras = {"orbit": settings["orbit"]}
    for key, name in zip(VIEWS[:3], settings["frames"], strict=True):
        data = dataset[lookup[name]]
        height, width = data["image"].shape[:2]
        cameras[key] = {
            "frame": name,
            "viewmat": np.linalg.inv(data["camtoworld"].numpy()).tolist(),
            "K": data["K"].numpy().tolist(),
            "width": width,
            "height": height,
            "near": 0.01,
            "far": 1e10,
            "intrinsicsSource": "pinned gsplat Dataset undistorted ROI",
        }
        if "mask" in data:
            np.save(output / f"{key}-mask.npy", data["mask"].numpy().astype(bool))
    return cameras


def run(settings: dict[str, Any], output: Path) -> None:
    import torch

    torch.set_num_threads(2)
    cameras = cameras_from_dataset(settings, output)
    write_json(
        output / "cameras.json",
        {
            "cameras": cameras,
            "liveCameras": settings["liveCameras"],
            "renderer": "gsplat classic",
            "background": "black",
            "projection": "pinhole, undistorted training ROI",
            "comparison": "same frozen-solve world pose; no alignment, PPISP or colour correction",
        },
    )
    scores = {}
    for label in ("trained", "snapshot", "baseline"):
        source = settings["sources"][label]
        model = load_model(source)
        for key in ("orbit",) if label == "baseline" else VIEWS:
            rgb = render_model(model, cameras[key], source["shDegree"])
            Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(output / f"{label}-{key}.png")
            if key == "orbit":
                continue
            ref_path = output / f"{key}-reference.npy"
            if label == "trained":
                np.save(ref_path, rgb)
            else:
                reference = np.load(ref_path)
                mask_path = output / f"{key}-mask.npy"
                mask = np.load(mask_path) if mask_path.exists() else np.ones(rgb.shape[:2], bool)
                if not mask.any():
                    raise ValueError("camera has no valid pixels")
                scores[key] = {
                    "frame": cameras[key]["frame"],
                    "psnrSnapshotVsTrained": psnr(rgb[mask], reference[mask]),
                    "validPixels": int(mask.sum()),
                }
                ref_path.unlink()
                mask_path.unlink(missing_ok=True)
        del model
        torch.cuda.empty_cache()
    panels(output, settings)
    write_json(
        output / "scores.json",
        {
            "trainingCameras": scores,
            "psnrProtocol": "clamped float RGB [0,1], valid pixels; null=identical",
            "note": "Agreement with trained render, not photographs. Spark is not exercised.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.settings.read_text()), args.out)


if __name__ == "__main__":
    main()

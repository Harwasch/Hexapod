"""Small, source-checked extension seam for the pinned upstream training loop.

We insert one initialization call, request expected depth, and add losses immediately
before backward. The build checks every anchor against the pinned source. No upstream
training loop is vendored. Math here implements per-image robust scale/shift alignment,
view-facing depth normals, and a scale-axis ratio penalty independently.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import block_trainer
import quality_stack

ANCHORS = (
    "class Runner:",
    "        self.stage = Stage()",
    'render_mode="RGB+ED" if cfg.depth_loss else "RGB",',
    "            loss.backward()",
)


def patch_source(source: str, settings: Path, trainer: Path) -> str:
    for anchor in ANCHORS:
        if source.count(anchor) != 1:
            raise ValueError(f"quality trainer seam moved: expected one {anchor!r}")
    bootstrap = (
        f"import sys\nsys.path.insert(0, {str(trainer.parent)!r})\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import quality_trainer as _hex_quality\n"
        f"_hex_settings = _hex_quality.read_settings({str(settings)!r})\n"
    )
    futures = list(re.finditer(r"(?m)^from __future__ import [^\n]+\n", source))
    insertion = futures[-1].end() if futures else 0
    source = source[:insertion] + bootstrap + source[insertion:]
    source = source.replace(
        ANCHORS[1], ANCHORS[1] + "\n        _hex_quality.prepare(self, globals(), _hex_settings)"
    )
    source = source.replace(
        ANCHORS[2], 'render_mode="RGB+ED" if cfg.depth_loss or _hex_settings["priors"] else "RGB",'
    )
    return source.replace(
        ANCHORS[3],
        "            loss = _hex_quality.loss(self, step, data, depths, alphas, Ks, loss)\n"
        + ANCHORS[3],
    )


def read_settings(path: str) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text()))


def prepare(runner: Any, namespace: dict[str, Any], settings: dict[str, Any]) -> None:
    runner.hex_settings = settings
    runner.hex_manifest = {}
    if settings["priors"]:
        manifest = json.loads((Path(settings["cache"]) / "manifest.json").read_text())
        if manifest["revision"] != quality_stack.MOGE_REVISION:
            raise ValueError("prior cache revision changed")
        runner.hex_manifest = manifest["frames"]
    if settings["far_field"]:
        ring = block_trainer.load_ring(Path(settings["ring"]))
        namespace["rasterization"] = block_trainer.ring_rasterization(
            namespace["rasterization"], ring
        )
        real_export = namespace["export_splats"]
        torch = importlib.import_module("torch")

        def export(**kwargs: Any) -> Any:
            # Plain standard 3DGS fields even with UT: background is appended just as
            # foreground SH gaussians. A marker is added only by our final exporter.
            for key, field in (
                ("means", "means"),
                ("scales", "log_scales"),
                ("quats", "quats"),
                ("opacities", "logit_opacity"),
                ("sh0", "f_dc"),
            ):
                extra = torch.as_tensor(
                    ring[field], device=kwargs[key].device, dtype=kwargs[key].dtype
                )
                if key == "sh0":
                    extra = extra[:, None, :]
                kwargs[key] = torch.cat([kwargs[key], extra])
            kwargs["shN"] = torch.cat(
                [
                    kwargs["shN"],
                    kwargs["shN"].new_zeros((len(ring["means"]), *kwargs["shN"].shape[1:])),
                ]
            )
            return real_export(**kwargs)

        namespace["export_splats"] = export


@lru_cache(maxsize=8)
def prior(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in ("depth", "normal", "mask")}


def aligned_depth(prediction: Any, target: Any) -> tuple[Any, Any]:
    """Differentiable residual with a detached robust alignment (three IRLS solves)."""
    torch = importlib.import_module("torch")
    with torch.no_grad():
        x, y = prediction.detach().float(), target.detach().float()
        if x.max() - x.min() <= torch.finfo(x.dtype).eps:
            return prediction * 0, prediction.new_tensor(0.0)
        design = torch.stack([x, torch.ones_like(x)], 1)
        weights = torch.ones_like(x)
        fit = x.new_tensor([1.0, 0.0])
        for _ in range(3):
            matrix = design * weights.sqrt()[:, None]
            fit = torch.linalg.lstsq(matrix, y * weights.sqrt()).solution
            residual = (design @ fit - y).abs()
            scale = residual.median().clamp_min(torch.finfo(x.dtype).eps)
            weights = (1.5 * scale / residual.clamp_min(torch.finfo(x.dtype).eps)).clamp_max(1)
        # Constant rendered depths cannot establish a scale. No negative-scale inversion.
        if (
            not torch.isfinite(fit).all()
            or fit[0] <= 0
            or x.max() - x.min() <= torch.finfo(x.dtype).eps
        ):
            return prediction * 0, prediction.new_tensor(0.0)
    return prediction * fit[0] + fit[1] - target, fit[0]


def normals(depth: Any, intrinsic: Any) -> Any:
    torch = importlib.import_module("torch")
    h, w = depth.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing="ij"
    )
    rays = torch.stack(
        [
            (x - intrinsic[0, 2]) / intrinsic[0, 0],
            (y - intrinsic[1, 2]) / intrinsic[1, 1],
            torch.ones_like(x),
        ],
        -1,
    )
    points = rays * depth[..., None]
    normal = torch.linalg.cross(
        points[1:-1, 2:] - points[1:-1, :-2], points[2:, 1:-1] - points[:-2, 1:-1]
    )
    normal = torch.nn.functional.normalize(normal, dim=-1)
    return normal * torch.where((normal * points[1:-1, 1:-1]).sum(-1, keepdim=True) > 0, -1.0, 1.0)


def loss(
    runner: Any, step: int, data: Any, depths: Any, alphas: Any, intrinsic: Any, total: Any
) -> Any:
    torch = importlib.import_module("torch")
    cfg = runner.hex_settings
    terms = {name: total.new_zeros(()) for name in ("depth", "normal", "gradient", "anisotropy")}
    if cfg["aniso_weight"]:
        scales = runner.splats["scales"]
        terms["anisotropy"] = (
            (torch.exp(scales.max(-1).values - scales.min(-1).values) - cfg["aniso_ratio"])
            .clamp_min(0)
            .mean()
        )
        total = total + cfg["aniso_weight"] * terms["anisotropy"]
    if cfg["priors"] and step >= cfg["start_step"]:
        item = int(data["image_id"].reshape(-1)[0])
        name = runner.parser.image_names[int(runner.trainset.indices[item])]
        cached = prior(str(Path(cfg["cache"]) / runner.hex_manifest[name]))
        prediction = depths[0, ..., 0]
        target = torch.as_tensor(cached["depth"], device=prediction.device, dtype=prediction.dtype)
        valid = (
            torch.as_tensor(cached["mask"], device=prediction.device).bool()
            & torch.isfinite(target)
            & (target > 0)
            & torch.isfinite(prediction)
            & (alphas[0, ..., 0] > 0.05)
        )
        if int(valid.sum()) >= 3:
            residual, scale = aligned_depth(prediction[valid], target[valid])
            denominator = target[valid].median().clamp_min(1e-6)
            terms["depth"] = residual.abs().mean() / denominator
            total = total + cfg["depth_weight"] * terms["depth"]
            if cfg["gradient_weight"]:
                error = (prediction * scale - target) / denominator
                pieces = []
                for axis in (0, 1):
                    pair = (
                        valid[1:, :] & valid[:-1, :] if axis == 0 else valid[:, 1:] & valid[:, :-1]
                    )
                    if pair.any():
                        pieces.append(torch.diff(error, dim=axis)[pair].abs().mean())
                if pieces:
                    terms["gradient"] = torch.stack(pieces).mean()
                    total = total + cfg["gradient_weight"] * terms["gradient"]
            if cfg["normal_weight"]:
                # Normals derived from rendered depth work with MCMC and UT; they do
                # not assume planar splats or modify the PLY representation.
                normal = normals(prediction, intrinsic[0])
                reference = torch.as_tensor(
                    cached["normal"], device=prediction.device, dtype=prediction.dtype
                )[1:-1, 1:-1]
                interior = (
                    valid[1:-1, 1:-1]
                    & valid[:-2, 1:-1]
                    & valid[2:, 1:-1]
                    & valid[1:-1, :-2]
                    & valid[1:-1, 2:]
                )
                interior &= torch.isfinite(reference).all(-1) & (reference.norm(dim=-1) > 0.5)
                if interior.any():
                    terms["normal"] = (1 - (normal * reference).sum(-1).clamp(-1, 1))[
                        interior
                    ].mean()
                    total = total + cfg["normal_weight"] * terms["normal"]
    if step % 100 == 0 or step + 1 == runner.cfg.max_steps:
        with torch.no_grad():
            scales = runner.splats["scales"]
            count = len(scales)
            far = cfg.get("far_count", 0)
            if cfg["far_field"]:
                center = scales.new_tensor(cfg["center"])
                far += int(
                    (
                        torch.linalg.vector_norm(runner.splats["means"] - center, dim=-1)
                        > 5 * cfg["radius"]
                    ).sum()
                )
            payload = {
                "farFieldShare": far / max(1, count + cfg.get("far_count", 0)),
                "step": step,
                "splatCount": count + cfg.get("far_count", 0),
                "elongationShare": float(
                    (torch.exp(scales.max(-1).values - scales.min(-1).values) > 10).sum()
                )
                / max(1, count + cfg.get("far_count", 0)),
                "lossTerms": {k: float(v.detach()) for k, v in terms.items()},
            }
        Path(cfg["loss_report"]).write_text(json.dumps(payload))
        if cfg["emit_metrics"]:
            sys.stdout.write("live-metrics: " + json.dumps(payload) + "\n")
            sys.stdout.flush()
    return total


def cpu_self_check() -> None:
    """Build-time gradient and standard-PLY contract check, with the image's torch."""
    import tempfile

    import gaussians

    torch = importlib.import_module("torch")
    x = torch.arange(1.0, 31.0, requires_grad=True)
    target = 2 * x.detach() + 3
    target[10] += 0.1
    residual, _ = aligned_depth(x, target)
    residual.abs().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise ValueError("aligned depth did not produce finite gradients")
    z = torch.full((7, 9), 5.0, requires_grad=True)
    normal = normals(z, torch.tensor([[50.0, 0, 4], [0, 50.0, 3], [0, 0, 1.0]]))
    normal.sum().backward()
    if z.grad is None or not torch.isfinite(z.grad).all():
        raise ValueError("rendered depth normals did not produce finite gradients")
    if not torch.allclose(normal[..., 2], torch.full((5, 7), -1.0)):
        raise ValueError("depth normals changed camera convention")
    export = importlib.import_module("gsplat").export_splats
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "plain.ply"
        export(
            means=torch.zeros(4, 3),
            scales=torch.zeros(4, 3),
            quats=torch.tensor([[1.0, 0, 0, 0]]).repeat(4, 1),
            opacities=torch.zeros(4),
            sh0=torch.zeros(4, 1, 3),
            shN=torch.zeros(4, 15, 3),
            format="ply",
            save_to=str(path),
        )
        splat = gaussians.read_splat(path, sh_degree=3)
        if splat.count != 4 or splat.sh_degree != 3:
            raise ValueError("upstream export no longer produces a plain SH3 3DGS PLY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--cpu-maths", action="store_true")
    parser.add_argument("--trainer", type=Path, required=True)
    args = parser.parse_args()
    patched = patch_source(args.trainer.read_text(), Path("settings.json"), args.trainer)
    compile(patched, str(args.trainer), "exec")
    if not args.self_check:
        raise ValueError("this command only checks the upstream seam")
    if args.cpu_maths:
        cpu_self_check()
    sys.stdout.write("quality trainer seam: OK\n")


if __name__ == "__main__":
    main()

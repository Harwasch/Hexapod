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
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

import block_trainer
import photometric_maths
import prior_cache
import prior_maths
import quality_stack

ANCHORS = (
    "class Runner:",
    "        self.stage = Stage()",
    'render_mode="RGB+ED" if cfg.depth_loss else "RGB",',
    "            loss.backward()",
    "            renders, alphas, info = self.stage.render(",
)


def patch_source(
    source: str, settings: Path, trainer: Path, *, display_referred: bool = False
) -> str:
    for anchor in ANCHORS:
        if source.count(anchor) != 1:
            raise ValueError(f"quality trainer seam moved: expected one {anchor!r}")
    if display_referred:
        anchor = "        # Losses & Metrics."
        if source.count(anchor) != 1:
            raise ValueError("quality trainer PPISP initialization seam moved")
        source = source.replace(
            anchor, "        _hex_quality.configure_photometry(self)\n" + anchor
        )
    bootstrap = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import quality_trainer as _hex_quality\n"
        # Helper imports (block/converge_trainer) also prepend the pipeline. Set
        # examples first *after* those imports, so datasets gets upstream's exif.
        f"sys.path.insert(0, {str(trainer.parent)!r})\n"
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
    # Assignment evaluates its RHS before releasing the old LHS. Upstream's info
    # holds the previous frame's large intersection arrays until the new render
    # has already allocated its own. Backward/strategy/eval are finished by here.
    source = source.replace(
        ANCHORS[4],
        "            pixels, Ks, masks = _hex_quality.step_inputs(self, pixels, Ks, masks)\n"
        "            height, width = pixels.shape[1:3]\n"
        "            num_train_rays_per_step = pixels.shape[0] * height * width\n"
        "            info = None\n" + ANCHORS[4],
    )
    return source.replace(
        ANCHORS[3],
        "            loss = _hex_quality.loss(self, step, data, depths, alphas, Ks, loss)\n"
        + ANCHORS[3]
        + "\n            _hex_quality.record_memory(self, step, info)",
    )


def read_settings(path: str) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text()))


def assert_import_origins(trainer: Path) -> None:
    """Audit every top-level collision, including future additions to either tree."""

    def modules(root: Path) -> set[str]:
        return {
            p.stem if p.suffix == ".py" else p.name
            for p in root.iterdir()
            if p.suffix == ".py" or (p.is_dir() and not p.name.startswith("."))
        }

    examples = trainer.parent.resolve()
    for name in modules(examples) & modules(Path(__file__).parent):
        if name.startswith("_"):
            continue
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(examples):
            raise ImportError(f"upstream module {name!r} shadowed by {origin}")


def import_patched_trainer(trainer: Path, settings: Path) -> types.ModuleType:
    """Execute the real patched module without entering __main__ or creating a Runner.

    The image has all dependencies but no GPU. Import errors and module collisions
    must fail its build before a paid prepass can start.
    """
    source = patch_source(
        trainer.read_text(),
        settings,
        trainer,
        display_referred=bool(read_settings(str(settings)).get("ppisp_display_referred")),
    )
    module = types.ModuleType("_hexapod_quality_import_check")
    module.__file__ = str(trainer)
    sys.modules[module.__name__] = module
    exec(compile(source, str(trainer), "exec", dont_inherit=True), module.__dict__)  # noqa: S102
    assert_import_origins(trainer)
    return module


def prepare(runner: Any, namespace: dict[str, Any], settings: dict[str, Any]) -> None:
    runner.hex_settings = settings
    runner.hex_manifest = {}
    if settings["priors"]:
        manifest = json.loads((Path(settings["cache"]) / "manifest.json").read_text())
        if (
            manifest["revision"] != quality_stack.MOGE_REVISION
            or manifest.get("format") != prior_maths.CACHE_FORMAT
        ):
            raise ValueError("prior cache revision or format changed")
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


def freeze_display_reference(module: Any) -> None:
    # Learned per-frame exposure/colour remain trainable with upstream's mean
    # regularization. Camera-wide tone and vignette cannot be represented by a PLY.
    module.crf_params.requires_grad_(False)
    module.vignetting_params.requires_grad_(False)


def configure_photometry(runner: Any) -> None:
    if not runner.hex_settings.get("ppisp_display_referred"):
        return
    torch = importlib.import_module("torch")
    module = runner.post_processing_module
    if runner.cfg.post_processing != "ppisp" or runner.cfg.ppisp_use_controller:
        raise ValueError("display-referred PPISP needs ppisp with no controller")
    if module.crf_params.shape[1:] != (3, 4) or module.vignetting_params.shape[1:] != (3, 5):
        raise ValueError("PPISP camera parameter schema moved")
    with torch.no_grad():
        module.crf_params.copy_(module.crf_params.new_tensor(photometric_maths.identity_crf()))
        module.vignetting_params.zero_()
    freeze_display_reference(module)


@lru_cache(maxsize=8)
def prior(path: str) -> dict[str, Any]:
    return prior_cache.load(Path(path))


def resize_image(value: Any, size: tuple[int, int], *, nearest: bool = False) -> Any:
    """HWC/CHW-free interface for an HxW or HxWxC tensor, preserving autograd."""
    if tuple(value.shape[:2]) == size:
        return value
    torch = importlib.import_module("torch")
    channels = value.ndim == 3
    image = value.permute(2, 0, 1)[None] if channels else value[None, None]
    resized = torch.nn.functional.interpolate(
        image.float(),
        size=size,
        mode="nearest-exact" if nearest else "bilinear",
        **({} if nearest else {"align_corners": False, "antialias": True}),
    )
    return resized[0].permute(1, 2, 0) if channels else resized[0, 0]


def resize_intrinsic(intrinsic: Any, sx: float, sy: float) -> Any:
    scaled = intrinsic.clone()
    scaled[..., 0, :] *= sx
    scaled[..., 1, :] *= sy
    return scaled


def step_inputs(runner: Any, pixels: Any, intrinsic: Any, masks: Any) -> tuple[Any, Any, Any]:
    """Bound only training steps: evaluation/export keep the frozen camera resolution.

    This is an explicit quality/runtime tradeoff, not an automatic OOM retry. It
    leaves the splat budget and schedule unchanged and does not crop the camera.
    """
    h, w = pixels.shape[1:3]
    height, width, sx, sy = prior_maths.resize_geometry(
        h, w, runner.hex_settings.get("step_max_side", 0)
    )
    if (h, w) == (height, width):
        return pixels, intrinsic, masks
    pixels = resize_image(pixels[0], (height, width))[None]
    masks = (
        None
        if masks is None
        else resize_image(masks[0], (height, width), nearest=True)[None].bool()
    )
    return pixels, resize_intrinsic(intrinsic, sx, sy), masks


def record_memory(runner: Any, step: int, info: dict[str, Any]) -> None:
    """After-backward snapshots every 1k completed steps; never retain a GPU tensor.

    Peaks are cumulative (do not reset gsplat's own high-water counters). Current
    allocated/reserved sizes distinguish live growth from allocator caching. The
    atomic file and stdout survive an exception later in training.
    """
    if (step + 1) % 1000 and step + 1 != runner.cfg.max_steps:
        return
    torch = importlib.import_module("torch")
    with torch.no_grad():
        radii = info.get("radii")
        sample = {
            "step": step + 1,
            "peakAllocatedGb": torch.cuda.max_memory_allocated() / 1e9,
            "peakReservedGb": torch.cuda.max_memory_reserved() / 1e9,
            "allocatedGb": torch.cuda.memory_allocated() / 1e9,
            "reservedGb": torch.cuda.memory_reserved() / 1e9,
            "splatCount": len(runner.splats["means"]) + runner.hex_settings.get("far_count", 0),
            "width": info.get("width"),
            "height": info.get("height"),
            "tileIntersections": int(info["isect_ids"].numel()) if "isect_ids" in info else None,
            "maxRadiusPx": float(radii.detach().max())
            if radii is not None and radii.numel()
            else None,
        }
    path = Path(runner.hex_settings["loss_report"]).with_name("train_memory.json")
    document = json.loads(path.read_text()) if path.exists() else {"samples": []}
    document["samples"].append(sample)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(document))
    temporary.replace(path)
    if runner.hex_settings["emit_metrics"]:
        sys.stdout.write("live-metrics: " + json.dumps({"trainMemoryCurve": [sample]}) + "\n")
        sys.stdout.flush()


def aligned_depth(prediction: Any, target: Any) -> tuple[Any, Any]:
    """Detached three-pass IRLS alignment; differentiable residual, no tall GPU solve."""
    torch = importlib.import_module("torch")
    with torch.no_grad():
        x, y = prediction.detach().float(), target.detach().float()
        epsilon = torch.finfo(x.dtype).eps
        weights = torch.ones_like(x)
        for _ in range(3):
            slope, offset, variance = prior_maths.weighted_affine(x, y, weights, epsilon)
            residual = (x * slope + offset - y).abs()
            scale = residual.median().clamp_min(epsilon)
            weights = (1.5 * scale / residual.clamp_min(epsilon)).clamp_max(1)
        valid = torch.isfinite(slope) & torch.isfinite(offset) & (slope > 0) & (variance > epsilon)
        # Tensor selection avoids host synchronization for fit validity. Constant
        # depths still contribute zero residual/gradient, as before.
        slope = torch.where(valid, slope, torch.zeros_like(slope))
        offset = torch.where(valid, offset, torch.zeros_like(offset))
    return torch.where(valid, prediction * slope + offset - target, prediction * 0), slope


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
        h, w, sx, sy = prior_maths.resize_geometry(
            prediction.shape[0], prediction.shape[1], cfg.get("prior_max_side", 0)
        )
        size = (h, w)
        prediction = resize_image(prediction, size)
        intrinsic = resize_intrinsic(intrinsic, sx, sy)

        def reference_tensor(key: str, *, nearest: bool = False) -> Any:
            # Resize on CPU before transfer: the mmap cache never owns a CUDA tensor.
            value = torch.as_tensor(cached[key])
            return resize_image(value, size, nearest=nearest).to(
                device=prediction.device, dtype=prediction.dtype
            )

        target = reference_tensor("depth")
        valid = (
            reference_tensor("mask", nearest=True).bool()
            & torch.isfinite(target)
            & (target > 0)
            & torch.isfinite(prediction)
            & (resize_image(alphas[0, ..., 0], size, nearest=True) > 0.05)
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
                reference = reference_tensor("normal")[1:-1, 1:-1]
                interior = (
                    valid[1:-1, 1:-1]
                    & valid[:-2, 1:-1]
                    & valid[2:, 1:-1]
                    & valid[1:-1, :-2]
                    & valid[1:-1, 2:]
                )
                interior &= torch.isfinite(reference).all(-1) & (reference.norm(dim=-1) > 0.5)
                if tuple(cached["normal"].shape[:2]) != size:
                    reference = torch.nn.functional.normalize(reference, dim=-1)
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

    import numpy as np

    import gaussians
    import outofcore

    if outofcore._lerp(np.float32(1), np.float32(3), 0.75, np.dtype("float32")) != 2.5:
        raise ValueError("image NumPy percentile interpolation failed")

    torch = importlib.import_module("torch")
    camera = types.SimpleNamespace(
        crf_params=torch.nn.Parameter(torch.randn(2, 3, 4)),
        vignetting_params=torch.nn.Parameter(torch.randn(2, 3, 5)),
    )
    configure_photometry(
        types.SimpleNamespace(
            hex_settings={"ppisp_display_referred": True},
            cfg=types.SimpleNamespace(post_processing="ppisp", ppisp_use_controller=False),
            post_processing_module=camera,
        )
    )
    if camera.crf_params.requires_grad or camera.vignetting_params.requires_grad:
        raise ValueError("PPISP display reference did not freeze")
    if torch.count_nonzero(camera.vignetting_params):
        raise ValueError("PPISP vignetting is not identity")
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
    # The local venv has no torch; execute the actual resize/backward seam on the
    # image's CPU as well as testing its camera geometry with NumPy locally.
    pixels = torch.full((1, 90, 160, 3), 0.5, requires_grad=True)
    intrinsic = torch.tensor([[[120.0, 0, 80], [0, 120.0, 45], [0, 0, 1.0]]])
    masks = torch.ones((1, 90, 160), dtype=torch.bool)
    small, scaled, mask = step_inputs(
        types.SimpleNamespace(hex_settings={"step_max_side": 80}), pixels, intrinsic, masks
    )
    small.sum().backward()
    if small.shape != (1, 45, 80, 3) or mask.shape != (1, 45, 80):
        raise ValueError("training resize did not bound RGB/masks")
    if pixels.grad is None or not torch.isfinite(pixels.grad).all():
        raise ValueError("training resize lost its gradient")
    if not torch.allclose(scaled[0, :2], intrinsic[0, :2] / 2):
        raise ValueError("training resize changed camera rays")
    depth = torch.full((45, 80), 5.0, requires_grad=True)
    coarse = resize_image(depth, (22, 40))
    coarse_k = resize_intrinsic(scaled, 40 / 80, 22 / 45)
    normals(coarse, coarse_k[0]).sum().backward()
    if depth.grad is None or not torch.isfinite(depth.grad).all():
        raise ValueError("coarse normal comparison lost its gradient")
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
    if not args.self_check:
        raise ValueError("this command only checks the upstream seam")
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        settings = Path(directory) / "settings.json"
        settings.write_text(json.dumps({"ppisp_display_referred": True}))
        import_patched_trainer(args.trainer, settings)
    if args.cpu_maths:
        cpu_self_check()
    sys.stdout.write("quality trainer seam: OK\n")


if __name__ == "__main__":
    main()

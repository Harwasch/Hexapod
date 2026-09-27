"""A calibration scene: our exact `train` stage on data with published numbers.

The spool-table capture tells us how a switch changes *our* capture; it cannot tell us
whether 23 dB is what this trainer should get, because no one else has trained on it.
This does: one standard scene, its published poses, the pipeline's own `train` stage
(`stages.gsplat` -- the dataset it builds, the argv it passes, the metrics it reads
back) on a Modal GPU, and the result set beside the paper's.

**The scene is Tanks and Temples `truck`**, from the archive the 3DGS authors released
with their paper -- `tandt_db.zip`, 682,628,995 bytes, sha256 below (downloaded and
hashed 2026-09-27). Its 251 frames are already 979x546, and its `sparse/0` is the COLMAP
model the paper trained and evaluated on, so nothing here re-poses anything. Chosen over
a Mip-NeRF 360 scene because the only download of those is the 12.5 GB `360_v2.zip`.

**The published numbers** are 3DGS's own, "Ours-30k" on Truck in Appendix D of
Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH
2023; arXiv:2308.04079, https://arxiv.org/abs/2308.04079), read from the PDF 2026-09-27:
Table 8 PSNR **25.187**, Table 7 SSIM **0.879**, Table 9 LPIPS **0.148**. The split is
the same one: the paper's `--eval` holds out every 8th image sorted by name
(`llffhold = 8`), and gsplat v1.5.3's parser holds out `index % 8 == 0` of the images
sorted by name -- 32 of 251 either way (checked by loading this archive with that parser
on CPU: val starts 000001, 000009, 000017). 3DGS's LPIPS is VGG, so the benchmark asks
gsplat for `--lpips_net vgg` (its default is AlexNet; gsplat's own `benchmarks/
compression/mcmc_tt.sh` does the same "to align with other benchmarks").

gsplat's `simple_trainer.py default` is its reproduction of 3DGS; its docs
(`docs/source/tests/eval.rst` at v1.5.3) show it within about 0.15 dB of the paper per
Mip-NeRF 360 scene. So the `gsplat-default` config is the one to compare, and a gap much
bigger than that is ours: the argv, the dataset, or the one deliberate difference, that
this stage trains with `--no-normalize-world-space`. The `recipe` config trains with
photo-reconstruct's own settings (MCMC, 500k cap) instead, to see what those cost or buy
on a scene with a known answer.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CONFIGS",
    "SCENES",
    "Published",
    "Scene",
    "compare",
    "extract",
    "markdown",
]

PAPER = (
    "Kerbl et al. 2023, 3D Gaussian Splatting for Real-Time Radiance Field Rendering, "
    "Appendix D, Tables 7-9 (Ours-30k)"
)
PAPER_URL = "https://arxiv.org/abs/2308.04079"

#: How far below the published PSNR a result may land and still be called a match. gsplat
#: reproduces 3DGS to about 0.15 dB a scene; 0.5 leaves room for a different GPU and for
#: run-to-run noise, and not for a real defect.
MATCH_DB = 0.5
#: Below this, the run is a failure worth a red job, not a line to read.
FAIL_DB = 1.0


@dataclass(frozen=True)
class Published:
    psnr: float
    ssim: float
    lpips: float
    lpips_net: str
    source: str
    url: str


@dataclass(frozen=True)
class Scene:
    name: str
    archive_url: str
    archive_bytes: int
    archive_sha256: str
    #: The scene's directory inside the archive, with a trailing slash.
    prefix: str
    images: int
    published: Published
    #: Train-stage params every config of this scene needs (data_factor, lpips net, ...).
    params: Mapping[str, Any] = field(default_factory=dict)


SCENES: dict[str, Scene] = {
    "truck": Scene(
        name="truck",
        archive_url=(
            "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
        ),
        archive_bytes=682_628_995,
        archive_sha256="816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1",
        prefix="tandt/truck/",
        images=251,
        published=Published(
            psnr=25.187,
            ssim=0.879,
            lpips=0.148,
            lpips_net="vgg",
            source=PAPER,
            url=PAPER_URL,
        ),
        # The archive's frames are the resolution the paper trained at (979x546), so
        # nothing is shrunk; LPIPS on VGG, as the paper measured it.
        params={"data_factor": 1, "extra_args": ["--lpips_net", "vgg"]},
    ),
}

#: What each config trains with, over the scene's own params. The full schedule both
#: times: `schedule_full_at` is absent, so no capture-size scaling applies.
CONFIGS: dict[str, dict[str, Any]] = {
    # gsplat's 3DGS reproduction, as `examples/benchmarks/basic.sh` runs it: the
    # `default` strategy, 30k steps. The one the published numbers are comparable to.
    "gsplat-default": {"iterations": 30_000, "strategy": "default"},
    # photo-reconstruct's Standard train stage (recipes/photo-reconstruct.yaml): MCMC
    # under a budget measured from the scene (gaussian_budget.py puts Truck at 1.52M at
    # its 979 px) and stopped when held-out PSNR goes flat (convergence.py). The budget's
    # floor, density and ceilings are the stage's defaults, which the recipe restates.
    "recipe": {
        "iterations": 30_000,
        "strategy": "mcmc",
        "cap_max": "auto",
        "budget_max": 2_000_000,
        "converge": True,
    },
    # The recipe before the budget: a fixed 500k and the full 30k. Measured 25.12 dB
    # against the paper's 25.19, so `recipe` against this is what the budget and the
    # stopping rule buy on a scene with a known answer.
    "recipe-500k": {"iterations": 30_000, "strategy": "mcmc", "cap_max": 500_000},
}


def train_params(
    scene: Scene, config: str, switches: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The `train` stage's params for `config` on `scene`, plus any extra switches (for
    trying `pose_opt` and friends on a scene with a known answer)."""
    if config not in CONFIGS:
        raise ValueError(f"config {config!r} is not one of {sorted(CONFIGS)}")
    extra = dict(switches or {})
    params: dict[str, Any] = {**scene.params, **CONFIGS[config], **extra}
    params["extra_args"] = [
        *scene.params.get("extra_args", []),
        *extra.get("extra_args", []),
    ]
    return params


#: The switches a benchmark spec may turn on: the boolean ones `stages.gsplat` reads.
SPEC_SWITCHES = frozenset({"antialiased", "depth_loss", "pose_opt", "app_opt", "bilateral_grid"})


def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """`CONFIG[+switch...]`, the text inside a commit message's `[benchmark:...]` token.

    Empty is `gsplat-default` with nothing on; `recipe+pose_opt` is the recipe's settings
    with pose refinement. Anything unrecognised is refused by name rather than ignored,
    since a benchmark that silently ran without the switch asked for answers a different
    question.
    """
    parts = [part.strip() for part in spec.strip().split("+")]
    config = parts[0] or "gsplat-default"
    if config not in CONFIGS:
        raise ValueError(f"benchmark config {config!r} is not one of {sorted(CONFIGS)}")
    switches: dict[str, Any] = {}
    for name in parts[1:]:
        if name not in SPEC_SWITCHES:
            raise ValueError(f"benchmark switch {name!r} is not one of {sorted(SPEC_SWITCHES)}")
        switches[name] = True
    return config, switches


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive: Path, scene: Scene, target: Path, *, verify: bool = True) -> tuple[Path, Path]:
    """The scene's `images/` as the `frames` input and its `sparse/0/*.bin` as `poses`.

    Checked against the pinned sha256 first (unless `verify` is off, for a rehearsal on a
    made-up archive): a changed archive is a different benchmark, and its numbers would
    be compared against a paper that never saw it. Only the three COLMAP binaries go into
    `poses` -- the archive's `project.ini` is COLMAP GUI state, not the model.
    """
    if verify:
        found = sha256_of(archive)
        if found != scene.archive_sha256:
            raise ValueError(
                f"{archive} has sha256 {found}, not the {scene.archive_sha256} this "
                f"benchmark was pinned to; its numbers would not be comparable"
            )
    frames = target / "frames"
    poses = target / "poses"
    for directory in (frames, poses):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    images_prefix = f"{scene.prefix}images/"
    sparse_prefix = f"{scene.prefix}sparse/0/"
    wanted = {"cameras.bin", "images.bin", "points3D.bin"}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = info.filename
            if info.is_dir():
                continue
            leaf = name.rsplit("/", 1)[-1]
            if name.startswith(images_prefix) and "/" not in name[len(images_prefix) :]:
                destination = frames / leaf
            elif name.startswith(sparse_prefix) and leaf in wanted:
                destination = poses / leaf
            else:
                continue
            with bundle.open(info) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    count = sum(1 for path in frames.iterdir() if path.is_file())
    missing = sorted(wanted - {path.name for path in poses.iterdir()})
    if missing or count == 0:
        raise ValueError(
            f"{archive} has no {scene.prefix} scene in the expected layout: "
            f"{count} images, missing {missing}"
        )
    if verify and count != scene.images:
        raise ValueError(f"{scene.name}: expected {scene.images} images, found {count}")
    return frames, poses


def _num(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def compare(
    scene: Scene,
    config: str,
    metrics: Mapping[str, Any],
    switches: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Our `train_metrics.json` beside the published numbers, and a verdict.

    The verdict is on PSNR alone, and only for `gsplat-default` with no switches: that is
    the run the paper's numbers describe. `match` within `MATCH_DB` below (or anywhere
    above), `below` past it, `fail` past `FAIL_DB` -- or when there is no PSNR at all.
    Any other run is `reference`: reported beside the paper, not judged by it.
    """
    published = scene.published
    ours = {key: _num(metrics.get(key)) for key in ("psnr", "ssim", "lpips")}
    delta: dict[str, float | None] = {}
    for key, value in ours.items():
        delta[key] = None if value is None else round(value - getattr(published, key), 4)
    comparable = config == "gsplat-default" and not switches
    psnr_delta = delta["psnr"]
    if psnr_delta is None:
        verdict = "fail"
    elif not comparable:
        verdict = "reference"
    elif psnr_delta >= -MATCH_DB:
        verdict = "match"
    elif psnr_delta >= -FAIL_DB:
        verdict = "below"
    else:
        verdict = "fail"
    settings = metrics.get("settings")
    held_out = metrics.get("heldOut")
    return {
        "scene": scene.name,
        "config": config,
        "switches": dict(switches or {}),
        "comparable": comparable,
        "verdict": verdict,
        "ours": ours,
        "published": {
            "psnr": published.psnr,
            "ssim": published.ssim,
            "lpips": published.lpips,
            "lpipsNet": published.lpips_net,
            "source": published.source,
            "url": published.url,
        },
        "delta": delta,
        "gaussians": metrics.get("gaussians"),
        "iterations": metrics.get("iterations"),
        "trainSeconds": metrics.get("trainSeconds"),
        "peakMemoryGb": metrics.get("peakMemoryGb"),
        "valFrames": held_out.get("valFrames") if isinstance(held_out, Mapping) else None,
        "settings": settings if isinstance(settings, Mapping) else None,
    }


def markdown(result: Mapping[str, Any], *, cost: Mapping[str, Any] | None = None) -> str:
    """The job summary: one table, ours against the paper, and what the verdict means."""
    ours, published, delta = result["ours"], result["published"], result["delta"]

    def cell(value: object, digits: int) -> str:
        number = _num(value)
        return "-" if number is None else f"{number:.{digits}f}"

    rows = [
        ("PSNR", "psnr", 2, 3),
        ("SSIM", "ssim", 3, 3),
        (f"LPIPS ({published['lpipsNet']})", "lpips", 3, 3),
    ]
    spec = "+".join([str(result["config"]), *sorted(result.get("switches") or {})])
    lines = [
        f"## Benchmark: {result['scene']} ({spec}) -- **{result['verdict']}**",
        "",
        "| | ours | published | difference |",
        "|---|---|---|---|",
    ]
    for label, key, digits, published_digits in rows:
        lines.append(
            f"| {label} | {cell(ours[key], digits)} | "
            f"{cell(published[key], published_digits)} | "
            f"{'-' if delta[key] is None else f'{delta[key]:+.{digits}f}'} |"
        )
    gaussians = result.get("gaussians")
    lines += [
        "",
        f"Held out: {result.get('valFrames') or '-'} frames (every 8th by name, as the paper). "
        f"Iterations {result.get('iterations') or '-'}; gaussians "
        f"{'-' if gaussians is None else f'{int(gaussians):,}'}; train "
        f"{cell(result.get('trainSeconds'), 0)} s; peak memory "
        f"{cell(result.get('peakMemoryGb'), 1)} GB.",
    ]
    if cost:
        lines.append(
            f"Billed {cell(cost.get('billedSeconds'), 0)} s on `{cost.get('tier')}`"
            + (f", ${cost['usd']:.3f}." if _num(cost.get("usd")) is not None else ".")
        )
    lines += [
        "",
        f"Published: {published['source']}, {published['url']}.",
        "",
        (
            f"`match`: PSNR no more than {MATCH_DB} dB below the paper (gsplat's own "
            f"reproduction lands within ~0.15 dB a scene). `below`: up to {FAIL_DB} dB "
            f"under -- worth a look. `fail`: more than that, or nothing measured. "
            f"`reference`: a config the paper did not run, reported for comparison only."
        ),
    ]
    if result.get("settings"):
        lines += ["", f"Settings: `{dict(result['settings'])}`"]
    return "\n".join(lines) + "\n"

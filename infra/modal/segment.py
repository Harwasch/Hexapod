"""Scene segmentation on a Modal GPU: `tools/captures/segment_scene.py` with SAM 2.1 and
SigLIP 2 on CUDA, for scans already published as tilesets.

docs/SCENE_OBJECTS.md is the method; this file only puts it on a GPU. The container gets
the same `tools/captures` code the CPU runs, fetches a scan's tileset (every tile, so the
binding covers parents too) from its public URL, runs the CLI, and hands back
`instances.json`, `instances.emb` and the run's summary, with `report.json` and
`compare.png` (`instances_report.py`: what the new run and the published `instances.json`
each cover, and a contact sheet of both by object and by category, drawn by gsplat). Nothing
is written to any bucket: the caller decides what to keep.

**Renderer.** The views the mask and image models see are rasterized by gsplat
(`--renderer gsplat`, the views' labels still from the CPU's samples), without the
floaters larger than `MAX_SCALE_M`; the image is the one `infra/modal/fill.py`'s gsplat jobs
use (torch 2.4 cu124 and gsplat's prebuilt wheel, Python 3.10), with a transformers that
carries SAM 2 and SigLIP 2 and still runs on that torch.

No secrets: both models are public (Apache-2.0) and download from Hugging Face into the
weights volume once. Run from the repository root (`.github/workflows/segment.yml` does):

    modal run infra/modal/segment.py                    # the scans in SCANS, in parallel
    modal run infra/modal/segment.py --names spool --views 24

**What a call reserves, and what that costs.** An L4, and cores and memory sized from the
scan before it is spawned (`sizing`; its constants, and the runs each comes from, are
together below). The scan's tileset.json is read here first -- it names every tile's
gaussians -- and the views a run can keep follow from `--views`, `MAX_VIEWS` and the
coverage rounds. The cores are a few render processes' worth whatever the scan, because
SAM 2.1's masks on the GPU set the pace; the memory is what the main process holds (the
models, the scan and the copy its views are drawn from, the views kept for `describe`)
plus the render processes. Both go to `Function.with_options` (modal 1.5.5, the version
`.github/workflows/segment.yml` pins) as `(request, limit)`: Modal bills max(request,
used), and the limit -- `CPU_HEADROOM` and `MEMORY_HEADROOM` times the request -- is a hard
ceiling, so an estimate that is low costs a little more, or throttles, instead of failing.
The render processes follow from the request, never the limit:
`segment_scene.default_workers` derives them from what it is told (`--cpus`,
`--memory-gb`) -- the reserved cores less `MAIN_PROCESS_CORES`, no more than the memory
left after the scan and the models holds -- because inside the container the host's cores
and memory are visible, not the reservation, and a hard `--workers 24` on 8 cores forks
three renders per core and pays for the burst.
Every run logs its peaks (`usage` in its summary.json, and one printed line) so the
estimate can be tuned.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import modal

APP_NAME = "hexapod-segment"
app = modal.App(APP_NAME)
WEIGHTS = modal.Volume.from_name("hexapod-world-model-weights", create_if_missing=True)
HF_HOME = "/weights/hf"
CAPTURES = "/root/captures"

if modal.is_local():
    LOCAL_CAPTURES = Path(__file__).resolve().parents[2] / "tools" / "captures"
else:
    LOCAL_CAPTURES = Path(CAPTURES)

#: The published scans to segment, by short name: their public tileset URLs.
PUBLIC = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs"
SCANS: dict[str, str] = {
    "spool": f"{PUBLIC}/8e1cc115-cb80-4af2-81fc-dccaf6b65891/package/splat/tileset.json",
    "pumpkin": f"{PUBLIC}/430c1932-5b6a-47b1-bb71-bb7fa2fec86b/package/splat/tileset.json",
    "camp": f"{PUBLIC}/50c25673-0940-4574-9b96-0b21362f83ca/package/splat/tileset.json",
}

#: gsplat's prebuilt wheel (as `infra/modal/fill.py`): torch 2.4, CUDA 12.4, Python 3.10.
GSPLAT_WHEEL = (
    "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/"
    "gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"
    "#sha256=01e1fd63dc69c9945e70158c818c3bb07fedf4aabcf020e6608d264cf27cc5dd"
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.4.1+cu124",
        "torchvision==0.19.1+cu124",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        # gsplat imports `packaging`, which nothing else here installs.
        "numpy==1.26.4",
        "ninja",
        "jaxtyping",
        "rich",
        "packaging",
        GSPLAT_WHEEL,
        # SAM 2, SigLIP 2 and Qwen3-VL; the 5.x line needs torch 2.5.
        "transformers==4.57.6",
        "sentencepiece",
        # Loads Qwen3-VL straight onto the GPU (`device_map`), not through host memory.
        "accelerate==1.10.1",
        # tools/captures/pyproject.toml's dependencies, at versions built for numpy 1.26.
        "pillow>=10",
        "laspy[lazrs]>=2.5",
        "pyproj>=3.6",
        "scipy>=1.11,<1.16",
        "opencv-python-headless==4.10.0.84",
    )
    .env({"HF_HOME": HF_HOME})
    .add_local_dir(
        LOCAL_CAPTURES,
        CAPTURES,
        ignore=["**/.venv/**", "**/__pycache__/**", "tests/**", "**/*.pyc"],
    )
)


#: **SAM 3** (candidate C's own segmenter, `facebook/sam3`, gated) needs transformers 5, so
#: torch >= 2.5, for which gsplat has no prebuilt wheel (its wheels stop at torch 2.4). Its
#: runs (`SEEDED_VARIANTS`) have an image of their own without gsplat, and are seeded with
#: an earlier run's `cache.tar` (`--seed`): that run's gsplat views, class-free masks and
#: cameras, and its VLM's vocabulary, so only SAM 3's masks are new (and the two candidates
#: differ in nothing else). The Hugging Face token is the Modal secret's (`HF_SECRET`, by
#: the name `.github/workflows/segment.yml` finds, as fill.yml does).
HF_SECRET = modal.Secret.from_name(os.environ.get("HEXAPOD_HF_SECRET", "huggingface"))
#: The keys a token may sit under in that secret (`_hf_token` copies it to HF_TOKEN).
HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")
SAM3_REPO = "facebook/sam3"
sam3_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.7.1+cu126",
        "torchvision==0.22.1+cu126",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "numpy==1.26.4",
        "packaging",
        "rich",
        # SAM 3 (Sam3VideoModel, Sam3Model), SigLIP 2 and SAM 2; 5.x needs torch >= 2.5.
        "transformers==5.19.0",
        "huggingface_hub>=1.31,<3",
        "sentencepiece",
        "accelerate>=1.1,<2",
        "pillow>=10",
        "laspy[lazrs]>=2.5",
        "pyproj>=3.6",
        "scipy>=1.11,<1.16",
        "opencv-python-headless==4.10.0.84",
    )
    .env({"HF_HOME": HF_HOME})
    .add_local_dir(
        LOCAL_CAPTURES,
        CAPTURES,
        ignore=["**/.venv/**", "**/__pycache__/**", "tests/**", "**/*.pyc"],
    )
)


def _hf_token() -> list[str]:
    """HF_TOKEN set from whichever key the secret uses; the keys present (names only)."""
    present = sorted(k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper())
    keys = [k for k in HF_TOKEN_KEYS if os.environ.get(k)]
    keys += sorted(k for k, v in os.environ.items() if v.startswith("hf_"))
    if keys:
        os.environ["HF_TOKEN"] = os.environ[keys[0]]
    return present


#: Gaussians larger than this (largest axis, metres) are left out of the views: the camp's
#: floaters (0.4% of it), which gsplat draws as blobs over a view from outside.
MAX_SCALE_M = 0.5
#: Coverage rounds after the first lift, and views per round (`segment_scene.coverage_views`).
COVERAGE_ROUNDS = 2
COVERAGE_VIEWS = 96
#: Also embed every crop kind and score `segment_scene.describe_variants` (variants.json,
#: variants.npz): to compare how instances are described. It costs a third more time.
VARIANTS = False

#: The segmentation bake-off's candidates that run here (`--variant`), each a script of
#: tools/captures with segment_scene's arguments and its own. A variant's call has a hard
#: `VARIANT_TIMEOUT_S`, so a run that hangs costs at most that; it keeps its cache
#: (`cache.tar`: every view's masks and image, the refine pass's box masks, the names) so it
#: can be re-assembled on a CPU without another GPU run.
VARIANT_SCRIPTS: dict[str, tuple[str, ...]] = {
    "ground-first": (
        "segment_ground_first.py",
        "--masks", "segment_models:Sam2LargeMasks",
        "--boxes", "segment_models:Sam2BoxMasks",
        "--namer", "segment_models:QwenNamer",
    ),
    # Candidate C itself: SAM 3, in its own image, seeded with the stand-in's run
    # (`SEEDED_VARIANTS`: `[segment|names=spool,pumpkin|variant=concept-first|seed=<run>]`).
    "concept-first": (
        "concept_scene.py",
        "--concepts", "concept_models:Sam3Concepts",
        "--masks", "segment_models:Sam2LargeMasks",
        "--seeded",
    ),
    # Candidate C with Grounding DINO + SAM 2 standing in for SAM 3: never C itself. Run it
    # with `views=64` (no coverage rounds): spool or pumpkin ~7 min, ~$0.13 each.
    "concept-first-standin": (
        "concept_scene.py",
        "--vlm", "concept_models:QwenVocabulary",
        "--concepts", "concept_models:GroundedSam2Concepts",
        "--masks", "segment_models:Sam2LargeMasks",
        "--stand-in",
    ),
}  # fmt: skip
VARIANT_TIMEOUT_S = 50 * 60
#: Variants that run in the SAM 3 image from a seed (`segment_seeded`), each call stopped
#: after `SEEDED_TIMEOUT_S` (run 37537406212: ~4 min a scan, the models' load, SAM 3 on 64
#: views and the instances' portraits drawn on the CPU). `check_sam3` runs first, on CPU only: the
#: secret's token can read the gated weights (else nothing on a GPU starts), the weights
#: into the volume, and SAM 3's calls on two of the seed's views.
SEEDED_VARIANTS = frozenset({"concept-first"})
SEEDED_TIMEOUT_S = 12 * 60
SAM3_CHECK_TIMEOUT_S = 15 * 60
SAM3_CHECK_CPU = 4.0
SAM3_CHECK_MEMORY_MIB = 16 * 1024
#: What a variant adds to the main process's memory: SAM 2.1 large and Qwen3-VL 4B's host
#: side (their weights go to the GPU), and the refine pass's views.
VARIANT_BYTES = 4 * (1 << 30)


# ------------------------------------------------------------------- what a run reserves
#
# Every constant of the estimate is here, with the run it was calibrated from. v2 has not
# been measured at these sizes, so the memory terms are derived from segment_scene's code
# (what it holds, and when) and the ceiling over them is generous. Each run logs its peaks
# -- `usage` in its summary.json, beside `sizing` (the estimate it ran on, term by term in
# `termsGiB`), and one `<name>: peak ...` line -- and the constants are re-tuned from those:
#   * BASE_BYTES: `mainPeakGiB` of a small scan (spool, pumpkin), less its `scan` and
#     `views` terms, which are small there;
#   * SCAN_BYTES_PER_GAUSSIAN: the slope of `mainPeakGiB` (less the views term) against
#     `gaussians`, from a small scan to the camp;
#   * RENDER_WORKER_BYTES (and segment_scene's, which must match): `workerPeakGiB`, what
#     a render worker held of its own (not `workerResidentGiB`, which counts the pages it
#     shares with the main process from the fork);
#   * MEMORY_HEADROOM: down towards 1.5 once `containerPeakGiB` (or main + workers) has
#     stayed near the request on all three scans;
#   * the cores: `gpuBusyShare` near 1 with `coresUsed` under the request says the masks
#     set the pace, as assumed; a GPU that idles while `coresUsed` sits at the request says
#     the renders do (RENDER_S_PER_VIEW is too low; `run.timingsS.renderS` is the time the
#     GPU waited for views).

GIB = 1 << 30
MIB = 1 << 20
GPU = "L4"

#: Modal's per-second rates, per hour: the L4, a core, a GiB. They reproduce both v1 camp
#: bills (2026-10-02, 252 views): 8 cores and 32 GiB, 1,416 s for $0.56, against 32 cores
#: and 96 GiB, 1,217 s for $1.04 -- 14% faster for 86% more.
L4_PER_HOUR = 0.80
CORE_PER_HOUR = 0.047
GIB_PER_HOUR = 0.008

#: **Cores.** The L4 is paced by SAM 2.1's masks (that v1 camp run: 695 s over 252 views,
#: 2.8 s a view at 32 points a side), while a render process takes ~8 s a view (the same
#: run, 24 of them): the renders overlap the masks, which is why 32 cores were only 14%
#: faster than 8. v2 adds work on the GPU's side of each view (gsplat's raster), not the
#: CPU's, so render processes enough to feed the masks, with `RENDER_SLACK` for slow views,
#: are enough at any scan size: a view's samples are capped (`splat_render.render`'s
#: `sample_budget`) and a local view sees one footprint, so a larger scan has more views,
#: not slower ones. Plus `MAIN_PROCESS_CORES` for the process that rasterizes, masks and
#: votes each view.
MASK_S_PER_VIEW = 2.8
RENDER_S_PER_VIEW = 8.0
RENDER_SLACK = 1.25
#: segment_scene.MAIN_PROCESS_CORES: of the cores requested, the ones it does not render on.
MAIN_PROCESS_CORES = 2
RENDER_WORKERS = math.ceil(RENDER_S_PER_VIEW / MASK_S_PER_VIEW * RENDER_SLACK)  # 4
CPU_CORES = float(RENDER_WORKERS + MAIN_PROCESS_CORES)  # 6
#: The CPU limit over the request: room for the single-process phases' bursts (the lift's
#: KD-tree queries use every core they see), throttled past it.
CPU_HEADROOM = 1.5
MIN_CPU_CORES, MAX_CPU_CORES = 4.0, 16.0

#: **Memory: what the main process holds at its peak** -- the end of the last coverage
#: round's renders, the views all kept and the render processes all running. Afterwards
#: the renders are gone, and what the lift, `describe` and the tile binding add is less
#: than they held. Unmeasured for v2, so derived from segment_scene's code:
#: the process with its models (`load_masks`, `load_embedder`): Python, numpy, scipy, torch
#: 2.4 with a CUDA context and its libraries (~3 GiB resident), SAM 2.1 tiny and SigLIP 2
#: base, whose weights are on the GPU once loaded ...
BASE_BYTES = 4 * GIB
#: ... per gaussian (the tileset's leaves, which `splat_render.load_tileset` loads): the
#: scan as `Splats` in float64 (positions 24, rotations 32, scales 24, colours 24,
#: opacities 8: 112), the copy its views are drawn from (`segment`: `splats.take(keep)`
#: without the floaters past `--max-scale-m`, 112 more), the cell of each in both (int32,
#: 4 + 4), the scale test's `largest` and `keep` (8 + 8) and the `SplatIndex` order (8).
#: The transients (the tiles' concatenation, the cells' and the index's sorts, the gsplat
#: upload) come before the views and the render processes are held ...
SCAN_BYTES_PER_GAUSSIAN = 256
#: ... per view kept for `describe` (`segment_scene.View`, 512 x 384): the image and the
#: CPU's samples (uint8 x 3 each), the cell and its purity per pixel (int32, float32), and
#: the view's votes (`_Votes`: 20 bytes a visible cell, up to ~50k cells) ...
VIEW_WIDTH, VIEW_HEIGHT = 512, 384  # segment_scene.VIEW_WIDTH, VIEW_HEIGHT
VOTE_BYTES = 1 << 20
VIEW_BYTES = VIEW_WIDTH * VIEW_HEIGHT * (3 + 3 + 4 + 4) + VOTE_BYTES
#: (segment_scene.MAX_VIEWS: the plan's views in all, local ones included) ...
MAX_VIEWS = 480
#: ... and per render process (segment_scene's, the room `default_workers` divides): a
#: camp view renders in 1.6-1.9 GB at most (v1, 2026-10-02), its per-sample arrays capped
#: by `render`'s sample budget; past 62.5M gaussians a whole-scan view's culling (40 bytes
#: a gaussian) is more.
RENDER_WORKER_BYTES = 2.5e9
RENDER_BYTES_PER_GAUSSIAN = 40
#: The memory limit over the request. Nothing of v2's memory has been measured, so it is
#: twice the estimate: the limit is not billed (only use above the request is), and a run
#: OOM-killed at its limit wastes the whole run, while one that needs twice the estimate
#: under it costs `GIB_PER_HOUR` a GiB more. Bring it down once the logged peaks agree.
MEMORY_HEADROOM = 2.0
MIN_MEMORY_MIB, MAX_MEMORY_MIB = 16 * 1024, 128 * 1024
MAX_MEMORY_LIMIT_MIB = 192 * 1024
#: A leaf tile without `extras.gaussians` counts as the package stage's most
#: (`tile_gaussians`); a scan whose tileset.json cannot be read here is sized as the camp
#: (22,577,243 leaf gaussians in 514 tiles).
TILE_GAUSSIANS = 100_000
FALLBACK_GAUSSIANS = 22_577_243

#: **Expected, not measured.** The estimate at the defaults (24 views, 2 coverage rounds:
#: up to 672 views), the scans' sizes from their tileset.json (2026-10-03). The times are
#: guesses: the camp's v2 run (segment.yml run 37100026825) took 2,351 s on 32 cores and
#: 96 GiB, about $2.00, and 4 render processes are taken to add ~15% (8 cores did to v1);
#: the small scans ~250 views of 2.8 s, the models' loading and `describe`.
#:
#:   scan     leaf gaussians  tiles  cores (limit)  GiB (limit)  ~time   ~$/h   ~$ a run
#:   spool           153,566      3     6 (9)        16 (32)    15 min  1.21    0.30
#:   pumpkin         387,813      7     6 (9)        16 (32)    15 min  1.21    0.30
#:   camp         22,577,243    514     6 (9)        22 (44)    45 min  1.26    0.94
#:   (2x camp)    45,000,000      -     6 (9)        27 (54)       -    1.30      -


def scan_size(document: dict) -> dict[str, int]:
    """A tileset.json's size: its tiles (content URIs, parents included), its leaf gaussians
    -- what segment_scene loads (`splat_render.load_tileset`) -- and its parents', from the
    `extras.gaussians` that `splat_tiles.convert` writes on every tile (a leaf without one
    counts as `TILE_GAUSSIANS`)."""
    tiles = leaves = parents = 0
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        children = tile.get("children", [])
        if tile.get("content", {}).get("uri"):
            tiles += 1
            count = tile.get("extras", {}).get("gaussians")
            if children:
                parents += int(count or 0)
            else:
                leaves += TILE_GAUSSIANS if count is None else int(count)
        stack.extend(children)
    return {"tiles": tiles, "gaussians": leaves, "parentGaussians": parents}


def planned_views(views: int, coverage_rounds: int) -> int:
    """The most views a run renders and keeps: `plan_views` makes `views` and local views up
    to `MAX_VIEWS` in all (how many local ones is known only once planned, so the most), and
    each coverage round up to `COVERAGE_VIEWS` more."""
    return max(views, MAX_VIEWS) + max(coverage_rounds, 0) * COVERAGE_VIEWS


def render_worker_bytes(gaussians: int) -> float:
    """One render process's peak (segment_scene.render_worker_bytes)."""
    return max(RENDER_WORKER_BYTES, RENDER_BYTES_PER_GAUSSIAN * float(gaussians))


def sizing(
    gaussians: int,
    *,
    views: int = 24,
    coverage_rounds: int = COVERAGE_ROUNDS,
    extra_bytes: float = 0.0,
) -> dict:
    """The reservation for a scan of `gaussians` leaf gaussians: requests, limits and the
    estimate's terms (GiB), as the summary keeps them.

    Memory: `BASE_BYTES`, the scan (`SCAN_BYTES_PER_GAUSSIAN`), the views it can keep
    (`planned_views` of `VIEW_BYTES`) and the render processes (`render_worker_bytes`),
    rounded up to a GiB within `MIN_MEMORY_MIB` and `MAX_MEMORY_MIB`; the limit
    `MEMORY_HEADROOM` times that. Cores: `CPU_CORES` within `MIN_CPU_CORES` and
    `MAX_CPU_CORES`, whatever the scan; the limit `CPU_HEADROOM` times that. `workers` is
    what `segment_scene.default_workers` forks from the request -- never the limit -- the
    cores less `MAIN_PROCESS_CORES`, as many as the memory left after the rest holds."""
    cores = min(max(CPU_CORES, MIN_CPU_CORES), MAX_CPU_CORES)
    worker = render_worker_bytes(gaussians)
    count = planned_views(views, coverage_rounds)
    terms = {
        "base": float(BASE_BYTES),
        "scan": float(SCAN_BYTES_PER_GAUSSIAN) * gaussians,
        "views": float(count * VIEW_BYTES),
    }
    if extra_bytes:
        terms["variant"] = float(extra_bytes)
    held = sum(terms.values())
    renders = max(1, int(cores) - MAIN_PROCESS_CORES)
    memory_mib = math.ceil((held + renders * worker) / GIB) * 1024
    memory_mib = min(max(memory_mib, MIN_MEMORY_MIB), MAX_MEMORY_MIB)
    workers = max(1, min(renders, int((memory_mib * MIB - held) // worker)))
    terms["workers"] = workers * worker
    return {
        "gaussians": int(gaussians),
        "views": count,
        "cores": cores,
        "coresLimit": cores * CPU_HEADROOM,
        "memoryMiB": memory_mib,
        "memoryLimitMiB": max(
            memory_mib, min(int(memory_mib * MEMORY_HEADROOM), MAX_MEMORY_LIMIT_MIB)
        ),
        "workers": workers,
        "termsGiB": {k: round(v / GIB, 2) for k, v in terms.items()},
        "dollarsPerHour": round(
            L4_PER_HOUR + cores * CORE_PER_HOUR + memory_mib / 1024 * GIB_PER_HOUR, 3
        ),
    }


def options(plan: dict) -> dict:
    """`Function.with_options`' arguments for a `sizing`: (request, limit) pairs."""
    return {
        "gpu": GPU,
        "cpu": (float(plan["cores"]), float(plan["coresLimit"])),
        "memory": (int(plan["memoryMiB"]), int(plan["memoryLimitMiB"])),
    }


#: What the function's decorator reserves -- a call spawned without `with_options` -- the
#: estimate for a camp-sized scan. `main` spawns each scan on its own.
DEFAULT_SIZING = sizing(FALLBACK_GAUSSIANS)


def sizing_line(name: str, size: dict | None, plan: dict) -> str:
    """The line `main` prints for a scan before spawning it."""
    tiles = "?" if size is None else size["tiles"]
    return (
        f"{name}: {tiles} tiles, {plan['gaussians']:,} gaussians, up to {plan['views']} views"
        f" -> {plan['cores']:g} cores (limit {plan['coresLimit']:g}),"
        f" {plan['memoryMiB'] / 1024:g} GiB (limit {plan['memoryLimitMiB'] / 1024:g}) on an"
        f" {GPU}, {plan['workers']} render workers; ~${plan['dollarsPerHour']:.2f}/h"
    )


# ------------------------------------------------------------------- what a run used

#: nvidia-smi's GPU utilization (%) and memory (MiB), sampled this often (seconds) while
#: segment_scene runs.
GPU_SAMPLE_S = 2
GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used",
    "--format=csv,noheader,nounits",
    "-l",
    str(GPU_SAMPLE_S),
)
#: segment_scene.CGROUP_PEAK_FILES: the container's peak memory, cgroup v2 then v1.
CGROUP_PEAK_FILES = (
    "/sys/fs/cgroup/memory.peak",
    "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
)


@contextlib.contextmanager
def gpu_samples(path: Path) -> Iterator[None]:
    """nvidia-smi's samples (`GPU_QUERY`) into `path` while the block runs; none where
    there is no nvidia-smi."""
    process = None
    try:
        with path.open("w", encoding="utf-8") as out:
            process = subprocess.Popen(GPU_QUERY, stdout=out, stderr=subprocess.DEVNULL)
    except OSError:
        process = None
    try:
        yield
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def gpu_usage(text: str) -> dict[str, float]:
    """nvidia-smi's samples (`utilization.gpu, memory.used` a line) as the GPU's busy share
    -- their mean: each is the share of nvidia-smi's sample period a kernel ran -- and its
    peak memory; empty without samples."""
    busy: list[float] = []
    used: list[float] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        try:
            utilisation, memory = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        busy.append(utilisation)
        used.append(memory)
    if not busy:
        return {}
    return {
        "gpuBusyShare": round(sum(busy) / len(busy) / 100, 3),
        "gpuPeakGiB": round(max(used) / 1024, 2),
        "gpuSamples": len(busy),
    }


def container_peak_gib(files: tuple[str, ...] = CGROUP_PEAK_FILES) -> float | None:
    """The container's peak memory (every process in it, shared pages once), or None
    where no cgroup file keeps it."""
    for name in files:
        try:
            return round(int(Path(name).read_text().split()[0]) / GIB, 2)
        except (OSError, ValueError, IndexError):
            continue
    return None


def call_usage(run: dict | None, gpu_text: str) -> dict:
    """What a call used at its peak: segment_scene's `usage` from its summary (`run`: its
    main process, what a render worker held of its own, CPU seconds, wall time, cores
    busy), the
    container's peak over the whole call (the fetch and the report too) where the cgroup
    keeps it, the largest process the call ran (`resource`'s RUSAGE_CHILDREN: there even
    when segment_scene failed), and the GPU's busy share and peak memory."""
    import resource

    usage = dict((run or {}).get("usage", {}))
    largest = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # KiB on Linux
    usage["largestProcessGiB"] = round(largest * 1024 / GIB, 2)
    container = container_peak_gib()
    if container is not None:
        usage["containerPeakGiB"] = container
    usage.update(gpu_usage(gpu_text))
    return usage


def usage_line(name: str, summary: dict) -> str:
    """The line `main` prints for a scan once it is done: its peaks against its sizing."""
    plan = summary.get("sizing") or {}
    usage = summary.get("usage") or {}

    def gib(key: str) -> str:
        value = usage.get(key)
        return "?" if value is None else f"{value:g} GiB"

    busy = usage.get("gpuBusyShare")
    return (
        f"{name}: peak {gib('mainPeakGiB')} main, {gib('workerPeakGiB')} a render worker,"
        f" {gib('containerPeakGiB')} container; requested {plan.get('memoryMiB', 0) / 1024:g}"
        f" GiB (limit {plan.get('memoryLimitMiB', 0) / 1024:g}); {usage.get('coresUsed', '?')}"
        f" of {plan.get('cores', 0):g} cores busy; GPU busy"
        f" {'?' if busy is None else f'{busy:.0%}'}; {summary.get('totalS', '?')} s"
    )


def segment_argv(
    tiles: Path,
    work: Path,
    *,
    views: int,
    cpus: float,
    memory_mib: int,
    renderer: str = "gsplat",
    coverage_rounds: int = COVERAGE_ROUNDS,
    cache: Path | None = None,
    variant: str = "",
) -> list[str]:
    """`segment_scene.py`'s command line for one scan in `work` (the call's scratch
    directory: variants, debug sheets and the run's summary, `run.json`, go there): the
    views drawn by `renderer` without the floaters past `MAX_SCALE_M`, `coverage_rounds`
    rounds of `COVERAGE_VIEWS`, and the reservation's request (`cpus`, `memory_mib`; never
    the limit), not a worker count, so the render processes follow the memory the scan
    leaves (`default_workers`). A `variant` (`VARIANT_SCRIPTS`) runs its script instead,
    writing into `work/out` with a check sheet (`check.png`)."""
    if variant:
        script, *own = VARIANT_SCRIPTS[variant]
        head = [
            sys.executable, script, str(tiles / "tileset.json"), str(tiles),
            "--out", str(work / "out"), "--check", str(work / "check.png"), *own,
        ]  # fmt: skip
    else:
        head = [
            sys.executable, "segment_scene.py", str(tiles / "tileset.json"), str(tiles),
            "--masks", "segment_models:Sam2Masks",
        ]  # fmt: skip
    return [
        *head,
        "--embedder",
        "segment_models:SiglipEmbedder",
        "--vocabulary",
        "data/open_vocabulary.txt",
        "--views",
        str(views),
        "--renderer",
        renderer,
        "--max-scale-m",
        str(MAX_SCALE_M),
        "--coverage-rounds",
        str(coverage_rounds),
        "--coverage-views",
        str(COVERAGE_VIEWS),
        *(["--variants", str(work / "variants.json")] if VARIANTS and not variant else []),
        *([] if variant else ["--debug-dir", str(work / "debug")]),
        "--cpus",
        str(int(cpus)),
        "--memory-gb",
        f"{memory_mib / 1024:g}",
        "--summary",
        str(work / "run.json"),
        *(["--cache", str(cache)] if cache is not None else []),
    ]


#: The public bucket answers Python's default user agent with 403 (Cloudflare's bot rules);
#: curl's is let through.
USER_AGENT = "curl/8.5.0 (hexapod-segment)"


def _get(url: str, timeout: float) -> bytes:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as error:
        # HTTPError does not pickle back to the caller; say what it was instead.
        raise RuntimeError(f"GET {url}: HTTP {error.code} {error.reason}") from None


def _fetch(url: str, out: Path) -> int:
    """The tileset and every tile it names, parents included."""
    import concurrent.futures

    out.mkdir(parents=True, exist_ok=True)
    base = url.rsplit("/", 1)[0]
    if not url.startswith("https://"):
        raise ValueError(f"not an https URL: {url}")
    (out / "tileset.json").write_bytes(_get(url, 120))
    document = json.loads((out / "tileset.json").read_text(encoding="utf-8"))
    uris: list[str] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if uri := tile.get("content", {}).get("uri"):
            uris.append(uri)
        stack.extend(tile.get("children", []))

    def get(uri: str) -> None:
        (out / uri).parent.mkdir(parents=True, exist_ok=True)
        (out / uri).write_bytes(_get(f"{base}/{uri}", 600))

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        list(pool.map(get, uris))
    return len(uris)


@app.function(
    image=image,
    gpu=GPU,
    # Views render in forked processes, as many as the request holds
    # (`segment_scene.default_workers`), while the GPU masks them. Each call in `main`
    # overrides both with its scan's own (request, limit) (`sizing`, `with_options`).
    cpu=options(DEFAULT_SIZING)["cpu"],
    memory=options(DEFAULT_SIZING)["memory"],
    volumes={"/weights": WEIGHTS},
    timeout=3 * 3600,
)
def segment_scan(
    name: str,
    url: str,
    views: int = 24,
    keep_masks: bool = False,
    renderer: str = "gsplat",
    coverage_rounds: int = COVERAGE_ROUNDS,
    plan: dict | None = None,
    variant: str = "",
) -> dict:
    """Segment one published scan; returns the files (bytes) and the run's summary.
    `keep_masks`: also return `masks.tar` (every view's masks and the cameras, the
    `--cache` files that are not views), to lift again elsewhere without a GPU.
    `plan` is the `sizing` this call was spawned on (default `DEFAULT_SIZING`, the
    decorator's), which the container cannot see for itself: its request goes to
    segment_scene, and the summary keeps it beside what the run used (`usage`)."""
    return _segment(name, url, views, keep_masks, renderer, coverage_rounds, plan, variant)


@app.function(
    image=sam3_image,
    gpu=GPU,
    cpu=options(DEFAULT_SIZING)["cpu"],
    memory=options(DEFAULT_SIZING)["memory"],
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=SEEDED_TIMEOUT_S,
)
def segment_seeded(name: str, url: str, seed: bytes, views: int, plan: dict, variant: str) -> dict:
    """A `SEEDED_VARIANTS` run of one scan in the SAM 3 image: `seed` is an earlier run's
    `cache.tar` for it (its views' gsplat images, class-free masks and cameras, its VLM's
    vocabulary), so only the concept masks are new; the rest as `segment_scan`."""
    _hf_token()
    return _segment(name, url, views, False, "gsplat", 0, plan, variant, seed=seed)


@app.function(
    image=sam3_image,
    cpu=SAM3_CHECK_CPU,
    memory=SAM3_CHECK_MEMORY_MIB,
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=SAM3_CHECK_TIMEOUT_S,
)
def check_sam3(frames: list[bytes], vocabulary: bytes | None = None) -> dict:
    """Before any GPU: whose token the secret holds, whether it can read `SAM3_REPO`
    (gated), the weights into the volume, and SAM 3's calls on `frames` (`.npz` images,
    `rgb`) on the CPU (`concept_models.smoke_sam3`), asked for `vocabulary`'s things and
    cover (a seed's `names.json`) when given. `ok` only when all of it worked."""
    import io
    import traceback

    import numpy as np
    from huggingface_hub import HfApi, auth_check, snapshot_download

    out: dict = {"repo": SAM3_REPO, "secretKeys": _hf_token(), "ok": False}
    try:
        out["user"] = HfApi().whoami().get("name")
    except Exception as error:  # noqa: BLE001 - reported, the check goes on
        out["whoamiError"] = f"{type(error).__name__}: {error}"[:1000]
    try:
        auth_check(SAM3_REPO)
        out["access"] = True
    except Exception as error:  # noqa: BLE001 - this is the answer
        out["access"] = False
        out["error"] = f"{type(error).__name__}: {error}"[:2000]
        return out
    started = time.time()
    snapshot_download(SAM3_REPO, allow_patterns=["*.json", "*.txt", "model.safetensors"])
    WEIGHTS.commit()
    out["downloadS"] = round(time.time() - started, 1)
    sys.path.insert(0, CAPTURES)
    os.chdir(CAPTURES)
    try:
        import concept_models

        import concept_scene

        images = [np.load(io.BytesIO(b))["rgb"] for b in frames]
        concepts = None
        if vocabulary is not None:
            listed = Path(tempfile.mkdtemp()) / "names.json"
            listed.write_bytes(vocabulary)
            concepts = concept_scene.load_concepts(listed)
        started = time.time()
        out["smoke"] = concept_models.smoke_sam3(images, concepts, device="cpu")
        out["smokeS"] = round(time.time() - started, 1)
        out["ok"] = True
    except Exception:  # noqa: BLE001 - the trace is the answer
        out["smokeError"] = traceback.format_exc()[-6000:]
    return out


#: What of an earlier run's `cache.tar` seeds a `SEEDED_VARIANTS` run: never its concept
#: masks (`boxmask-*`, named by the things alone), which the new segmenter makes again.
SEED_FILES = ("raster-", "masks-", "cameras.json", "names.json")


def _segment(
    name: str,
    url: str,
    views: int,
    keep_masks: bool,
    renderer: str,
    coverage_rounds: int,
    plan: dict | None,
    variant: str,
    seed: bytes | None = None,
) -> dict:
    """`segment_scan`'s and `segment_seeded`'s call: fetch, segment, report, return."""
    import io
    import tarfile

    plan = plan or DEFAULT_SIZING
    started = time.time()
    with tempfile.TemporaryDirectory() as work:
        tiles = Path(work) / "tiles"
        cache = Path(work) / "cache"
        seeded: list[str] = []
        if seed is not None:
            cache.mkdir()
            with tarfile.open(fileobj=io.BytesIO(seed)) as tar:
                for member in tar.getmembers():
                    plain = member.isfile() and "/" not in member.name and ".." not in member.name
                    if plain and member.name.startswith(SEED_FILES):
                        (cache / member.name).write_bytes(tar.extractfile(member).read())
                        seeded.append(member.name)
        count = _fetch(url, tiles)
        fetched = time.time() - started
        # What is published now, to compare against (none before a first publish).
        before = Path(work) / "published" / "instances.json"
        before.parent.mkdir()
        try:
            before.write_bytes(_get(url.rsplit("/", 1)[0] + "/instances.json", 120))
        except RuntimeError:
            before = None
        gpu = Path(work) / "gpu.csv"
        with gpu_samples(gpu):
            run = subprocess.run(  # noqa: S603 - fixed argv; only paths we made vary
                segment_argv(
                    tiles,
                    Path(work),
                    views=views,
                    cpus=plan["cores"],
                    memory_mib=plan["memoryMiB"],
                    renderer=renderer,
                    coverage_rounds=coverage_rounds,
                    cache=cache if keep_masks or variant else None,
                    variant=variant,
                )
                + (
                    ["--cameras", str(cache / "cameras.json"), "--vlm", str(cache / "names.json")]
                    if seed is not None
                    else []
                ),
                cwd=CAPTURES,
                capture_output=True,
                text=True,
            )
        WEIGHTS.commit()
        log = run.stdout[-20000:] + run.stderr[-20000:]
        try:
            segmentation = json.loads((Path(work) / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):  # it failed before its summary
            segmentation = {}
        gpu_text = gpu.read_text(encoding="utf-8") if gpu.exists() else ""
        if run.returncode != 0:
            # What it used up to there still says whether it was the memory.
            usage = call_usage(segmentation, gpu_text)
            return {"name": name, "ok": False, "sizing": plan, "usage": usage, "log": log}
        segmented = time.time()
        written = Path(work) / "out" if variant else tiles
        files = {
            k: (written / k).read_bytes()
            for k in ("instances.json", "instances.emb")
            if (written / k).exists()
        }
        # The new run against the published one: coverage, categories and a contact sheet.
        new = Path(work) / "new" / "instances.json"
        new.parent.mkdir()
        new.write_bytes(files["instances.json"])
        report = subprocess.run(  # noqa: S603 - fixed argv; only paths we made vary
            [
                sys.executable,
                "instances_report.py",
                str(tiles),
                *([str(before)] if before else []),
                str(new),
                "--sheet",
                str(Path(work) / "compare.png"),
                "--max-scale-m",
                str(MAX_SCALE_M),
                *(["--gsplat"] if renderer == "gsplat" and seed is None else []),
                "--out",
                str(Path(work) / "report.json"),
            ],
            cwd=CAPTURES,
            capture_output=True,
            text=True,
        )
        log += "\n--- report\n" + report.stdout[-20000:] + report.stderr[-20000:]
        for k in (
            "compare.png",
            "report.json",
            "variants.json",
            "variants.npz",
            "debug/views.jpg",
            "debug/crops.jpg",
            "check.png",
            "check.legend.json",
        ):
            if (Path(work) / k).exists():
                files[k.rsplit("/", 1)[-1]] = (Path(work) / k).read_bytes()
        if (keep_masks or variant) and cache.exists():
            tar = Path(work) / "masks.tar"
            kept = ("masks-*.npz", "raster-*.npz", "boxmask-*.npz") if variant else ("masks-*.npz",)
            with tarfile.open(tar, "w") as out:
                for pattern in kept:
                    for path in sorted(cache.glob(pattern)):
                        out.add(path, arcname=path.name)
                for path in (cache / "cameras.json", cache / "names.json"):
                    if path.exists():
                        out.add(path, arcname=path.name)
            files["cache.tar" if variant else "masks.tar"] = tar.read_bytes()
        return {
            "name": name,
            "ok": True,
            "variant": variant,
            **({"seededFiles": len(seeded)} if seed is not None else {}),
            "tiles": count,
            "sizing": plan,
            "fetchS": round(fetched, 1),
            "segmentS": round(segmented - started - fetched, 1),
            "totalS": round(time.time() - started, 1),
            # Peaks over the whole call (the report too), to tune `sizing`'s constants.
            "usage": call_usage(segmentation, gpu_text),
            "run": {k: v for k, v in segmentation.items() if k != "usage"},
            "files": files,
            "log": log,
        }


#: `check_sam3`'s rate: its cores and memory at Modal's prices (no GPU).
CHECK_DOLLARS_PER_HOUR = (
    SAM3_CHECK_CPU * CORE_PER_HOUR + SAM3_CHECK_MEMORY_MIB / 1024 * GIB_PER_HOUR
)


def _seeds(folder: Path | None, names: list[str]) -> dict[str, bytes]:
    """Each scan's seed: `<folder>/<scan>/cache.tar`, an earlier run's (its artifact)."""
    if folder is None:
        raise SystemExit("a seeded variant needs --seed: a folder of an earlier run's caches")
    seeds = {}
    for name in names:
        path = folder / name / "cache.tar"
        if not path.exists():
            raise SystemExit(f"{name}: no seed at {path}")
        seeds[name] = path.read_bytes()
    return seeds


def _seed_file(seed: bytes, name: str) -> bytes | None:
    """One file of a seed (`names.json`: its vocabulary), or None."""
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(seed)) as tar:
        try:
            return tar.extractfile(name).read()
        except KeyError:
            return None


def _seed_frames(seed: bytes, count: int) -> list[bytes]:
    """`count` of a seed's view images (`raster-*.npz`, as stored), for `check_sam3`."""
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(seed)) as tar:
        members = sorted(
            (m for m in tar.getmembers() if m.isfile() and m.name.startswith("raster-")),
            key=lambda m: m.name,
        )
        return [tar.extractfile(m).read() for m in members[:count]]


@app.local_entrypoint()
def main(
    names: str = ",".join(SCANS),
    views: int = 24,
    out: str = "segment-out",
    keep_masks: bool = False,
    renderer: str = "gsplat",
    coverage_rounds: int = COVERAGE_ROUNDS,
    variant: str = "",
    seed: str = "",
    check_only: bool = False,
    check_frames: int = 2,
) -> None:
    """Segment the named scans in parallel containers; write each result under `out/`.
    `variant`: a bake-off candidate (`VARIANT_SCRIPTS`) instead of segment_scene, each call
    stopped after `VARIANT_TIMEOUT_S`. A `SEEDED_VARIANTS` one needs `seed`, a folder with an
    earlier run's `<scan>/cache.tar` (its `segmentation` artifact): `check_sam3` runs first,
    on the CPU, and only when it passes does a GPU start (`segment_seeded`, each call stopped
    after `SEEDED_TIMEOUT_S`); what it found is `out/sam3-check.json`. `check_only`: that
    check alone, SAM 3 asked for the first scan's own vocabulary on `check_frames` of its
    views, and no GPU (to see what a change to SAM 3's prompts finds before a run).

    Each scan's reservation follows its size (`sizing`): its tileset.json is read here
    first -- a few kB, hundreds for the camp -- for its tiles and gaussians, and it is
    spawned on `segment_scan.with_options(gpu=, cpu=(request, limit), memory=(request,
    limit))` (`options`), a container pool of its own. Once done, a line says what it used
    against that (`usage_line`); summary.json keeps both.
    """
    chosen = [n.strip() for n in names.split(",") if n.strip()]
    failed = []
    calls = []
    seeds: dict[str, bytes] = {}
    if variant in SEEDED_VARIANTS:
        seeds = _seeds(Path(seed) if seed else None, chosen)
        rate = CHECK_DOLLARS_PER_HOUR * SAM3_CHECK_TIMEOUT_S / 3600
        sys.stdout.write(
            f"check_sam3 (CPU): stopped after {SAM3_CHECK_TIMEOUT_S} s: at most ${rate:.2f}\n"
        )
        first = seeds[chosen[0]]
        count = check_frames if check_only else 2
        check = check_sam3.remote(_seed_frames(first, count), _seed_file(first, "names.json"))
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "sam3-check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
        sys.stdout.write(json.dumps(check, indent=1) + "\n")
        if not check.get("ok"):
            why = check.get("error") or check.get("smokeError") or "see sam3-check.json"
            raise SystemExit(f"SAM 3 is not usable here, so no GPU started: {why}")
        if check_only:
            return
    for name in chosen:
        url = SCANS[name]
        try:
            size = scan_size(json.loads(_get(url, 120)))
        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
            # The container fetches the same URL and will say what is wrong with it.
            sys.stdout.write(f"{name}: could not read its size ({error}); sized as the camp\n")
            size = None
        gaussians = FALLBACK_GAUSSIANS if size is None else size["gaussians"]
        if variant and variant not in VARIANT_SCRIPTS:
            raise SystemExit(f"variant {variant!r}: one of {', '.join(VARIANT_SCRIPTS)}")
        plan = sizing(
            gaussians, views=views, coverage_rounds=coverage_rounds,
            extra_bytes=VARIANT_BYTES if variant else 0,
        )  # fmt: skip
        sys.stdout.write(sizing_line(name, size, plan) + "\n")
        chosen_options = options(plan)
        if variant in SEEDED_VARIANTS:
            chosen_options["timeout"] = SEEDED_TIMEOUT_S
            worst = plan["dollarsPerHour"] * SEEDED_TIMEOUT_S / 3600
            sys.stdout.write(
                f"{name}: {variant}, seeded, stopped after {SEEDED_TIMEOUT_S} s: at most ${worst:.2f}\n"
            )
            call = segment_seeded.with_options(**chosen_options).spawn(
                name, url, seeds[name], views, plan, variant
            )
            calls.append((name, plan, call))
            continue
        if variant:
            chosen_options["timeout"] = VARIANT_TIMEOUT_S
            worst = plan["dollarsPerHour"] * VARIANT_TIMEOUT_S / 3600
            sys.stdout.write(
                f"{name}: {variant}, stopped after {VARIANT_TIMEOUT_S} s: at most ${worst:.2f}\n"
            )
        calls.append(
            (
                name,
                plan,
                segment_scan.with_options(**chosen_options).spawn(
                    name, url, views, keep_masks, renderer, coverage_rounds, plan, variant
                ),
            )
        )
    for name, plan, call in calls:
        try:
            result = call.get()
        except Exception as error:  # noqa: BLE001 - e.g. a container killed at its memory limit
            # The other scans' results are still written.
            log = f"{type(error).__name__}: {error}"
            result = {"name": name, "ok": False, "sizing": plan, "log": log}
        folder = Path(out) / result["name"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "log.txt").write_text(result["log"], encoding="utf-8")
        for file, data in result.get("files", {}).items():
            (folder / file).write_bytes(data)
        summary = {k: v for k, v in result.items() if k not in ("files", "log")}
        (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        sys.stdout.write(json.dumps(summary) + "\n")
        sys.stdout.write(usage_line(result["name"], summary) + "\n")
        if not result["ok"]:
            failed.append(result["name"])
    if failed:
        raise SystemExit(f"segmentation failed for {', '.join(failed)}")

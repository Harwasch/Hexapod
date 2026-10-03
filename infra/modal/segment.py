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

**What a call reserves, and what that costs.** An L4 with `CPU_CORES` cores and
`MEMORY_MIB` of memory -- chosen on cost, from the camp (below) -- and, for a scan of more
than `LARGE_TILES` tiles, `LARGE_CPU_CORES` and `LARGE_MEMORY_MIB` through
`Function.with_options` (modal 1.5.5, the version `.github/workflows/segment.yml` pins).
The render processes are not a fixed number: `segment_scene.default_workers` derives them
from the reservation it is told (`--cpus`, `--memory-gb`) -- one per reserved core, no more
than the memory left after the scan and the models are loaded holds at
`RENDER_WORKER_BYTES` each -- because inside the container the host's cores and memory
are visible, not the reservation, and a hard `--workers 24` on 8 cores forks three
renders per core and pays for the burst (Modal bills max(reserved, used)).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
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
        # SAM 2 and SigLIP 2; the 5.x line needs torch 2.5.
        "transformers==4.57.6",
        "sentencepiece",
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


#: The reservation every call gets, chosen on cost. Measured on the camp (22.6M gaussians,
#: ~110 m, 252 views; 2026-10-02, L4): **8 cores and 32 GiB, 1,416 s for $0.56**, against
#: 32 cores and 96 GiB, 1,217 s for $1.04 -- 14% faster for 86% more, because SAM 2.1's
#: masks on the GPU (695 s of the camp's) are the bulk and the CPU renders overlap them.
#: At Modal's per-second rates -- L4 $0.80/h, a core $0.047/h, a GiB $0.008/h, which
#: reproduce both figures -- 8 cores and 32 GiB are $0.63/h of CPU and memory beside the
#: L4, and the camp's run is $0.32 of GPU, $0.15 of cores and $0.10 of memory.
CPU_CORES = 8.0
MEMORY_MIB = 32 * 1024

#: A scan of more tiles than this gets the large reservation: twice the camp, whose
#: tileset.json names 514 (22.7M gaussians; `tiles_in` counts the LOD parents as well as
#: the leaves, each at most 100k gaussians -- the package stage's `tile_gaussians`), so
#: about 45M gaussians. What grows with the scan is what the main process holds --
#: the scan whole, its index, every view kept for `describe` (segment_scene's docstring:
#: the camp ran "on 15 GB") -- and at twice the camp that alone leaves 32 GiB room for one
#: render at `RENDER_WORKER_BYTES`, so the renders, not the masks, would set the pace,
#: and a little more would not fit at all. The view count does not grow past
#: `MAX_VIEWS` (480), so time grows about with the views: **expected, not measured**, about
#: twice the camp's 24 min at 16 cores and 64 GiB, ~$1.6 a scan ($0.80/h of L4 and $1.27/h
#: of cores and memory).
LARGE_TILES = 1000
LARGE_CPU_CORES = 16.0
LARGE_MEMORY_MIB = 64 * 1024


def tiles_in(document: dict) -> int:
    """How many tiles (content URIs, parents included) a tileset.json names."""
    count = 0
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("content", {}).get("uri"):
            count += 1
        stack.extend(tile.get("children", []))
    return count


def reservation(tiles: int) -> tuple[float, int]:
    """(cores, MiB) for a scan of `tiles` tiles: the default, or the large one past
    `LARGE_TILES` (see there for why, and what it is expected to cost)."""
    if tiles > LARGE_TILES:
        return LARGE_CPU_CORES, LARGE_MEMORY_MIB
    return CPU_CORES, MEMORY_MIB


#: Gaussians larger than this (largest axis, metres) are left out of the views: the camp's
#: floaters (0.4% of it), which gsplat draws as blobs over a view from outside.
MAX_SCALE_M = 0.5
#: Coverage rounds after the first lift, and views per round (`segment_scene.coverage_views`).
COVERAGE_ROUNDS = 2
COVERAGE_VIEWS = 96
#: Also embed every crop kind and score `segment_scene.describe_variants` (variants.json,
#: variants.npz): to compare how instances are described. It costs a third more time.
VARIANTS = False


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
) -> list[str]:
    """`segment_scene.py`'s command line for one scan in `work` (the call's scratch
    directory: variants and debug sheets go there): the views drawn by `renderer` without
    the floaters past `MAX_SCALE_M`, `coverage_rounds` rounds of `COVERAGE_VIEWS`, and the
    reservation, not a worker count, so the render processes follow the memory the scan
    leaves (`default_workers`)."""
    return [
        sys.executable,
        "segment_scene.py",
        str(tiles / "tileset.json"),
        str(tiles),
        "--masks",
        "segment_models:Sam2Masks",
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
        *(["--variants", str(work / "variants.json")] if VARIANTS else []),
        "--debug-dir",
        str(work / "debug"),
        "--cpus",
        str(int(cpus)),
        "--memory-gb",
        f"{memory_mib / 1024:g}",
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
    gpu="L4",
    # Views render in forked processes, as many as this reservation holds
    # (`segment_scene.default_workers`), while the GPU masks them. A large scan's call
    # overrides both (`reservation`, `Function.with_options`).
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
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
    cpus: float = CPU_CORES,
    memory_mib: int = MEMORY_MIB,
) -> dict:
    """Segment one published scan; returns the files (bytes) and the run's summary.
    `keep_masks`: also return `masks.tar` (every view's masks and the cameras, the
    `--cache` files that are not views), to lift again elsewhere without a GPU.
    `cpus` and `memory_mib` are what this call reserved -- the caller chose them -- which
    the container cannot see for itself."""
    import tarfile

    started = time.time()
    with tempfile.TemporaryDirectory() as work:
        tiles = Path(work) / "tiles"
        cache = Path(work) / "cache"
        count = _fetch(url, tiles)
        fetched = time.time() - started
        # What is published now, to compare against (none before a first publish).
        before = Path(work) / "published" / "instances.json"
        before.parent.mkdir()
        try:
            before.write_bytes(_get(url.rsplit("/", 1)[0] + "/instances.json", 120))
        except RuntimeError:
            before = None
        run = subprocess.run(  # noqa: S603 - fixed argv; only paths we made vary
            segment_argv(
                tiles,
                Path(work),
                views=views,
                cpus=cpus,
                memory_mib=memory_mib,
                renderer=renderer,
                coverage_rounds=coverage_rounds,
                cache=cache if keep_masks else None,
            ),
            cwd=CAPTURES,
            capture_output=True,
            text=True,
        )
        WEIGHTS.commit()
        log = run.stdout[-20000:] + run.stderr[-20000:]
        if run.returncode != 0:
            return {"name": name, "ok": False, "log": log}
        segmented = time.time()
        files = {
            k: (tiles / k).read_bytes()
            for k in ("instances.json", "instances.emb")
            if (tiles / k).exists()
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
                *(["--gsplat"] if renderer == "gsplat" else []),
                "--out",
                str(Path(work) / "report.json"),
            ],
            cwd=CAPTURES,
            capture_output=True,
            text=True,
        )
        log += "\n--- report\n" + report.stdout[-20000:] + report.stderr[-20000:]
        for k in ("compare.png", "report.json", "variants.json", "variants.npz", "debug/views.jpg", "debug/crops.jpg"):
            if (Path(work) / k).exists():
                files[k.rsplit("/", 1)[-1]] = (Path(work) / k).read_bytes()
        if keep_masks and cache.exists():
            tar = Path(work) / "masks.tar"
            with tarfile.open(tar, "w") as out:
                for path in sorted(cache.glob("masks-*.npz")) + [cache / "cameras.json"]:
                    if path.exists():
                        out.add(path, arcname=path.name)
            files["masks.tar"] = tar.read_bytes()
        return {
            "name": name,
            "ok": True,
            "tiles": count,
            "cpus": cpus,
            "memoryMiB": memory_mib,
            "fetchS": round(fetched, 1),
            "segmentS": round(segmented - started - fetched, 1),
            "totalS": round(time.time() - started, 1),
            "files": files,
            "log": log,
        }


@app.local_entrypoint()
def main(
    names: str = ",".join(SCANS),
    views: int = 24,
    out: str = "segment-out",
    keep_masks: bool = False,
    renderer: str = "gsplat",
    coverage_rounds: int = COVERAGE_ROUNDS,
) -> None:
    """Segment the named scans in parallel containers; write each result under `out/`.

    Each scan's reservation follows its size (`reservation`): its tileset.json is read
    here first -- a few kB -- to count its tiles, and a scan past `LARGE_TILES` is spawned
    on `segment_scan.with_options(cpu=..., memory=...)`, a container pool of its own.
    """
    chosen = [n.strip() for n in names.split(",") if n.strip()]
    failed = []
    calls = []
    for name in chosen:
        url = SCANS[name]
        try:
            tiles = tiles_in(json.loads(_get(url, 120)))
        except (RuntimeError, OSError, ValueError, KeyError) as error:
            # The container fetches the same URL and will say what is wrong with it.
            sys.stdout.write(f"{name}: could not count tiles ({error}); default reservation\n")
            tiles = 0
        cpus, memory_mib = reservation(tiles)
        function = (
            segment_scan
            if (cpus, memory_mib) == (CPU_CORES, MEMORY_MIB)
            else segment_scan.with_options(cpu=cpus, memory=memory_mib)
        )
        sys.stdout.write(
            f"{name}: {tiles} tiles -> {cpus:g} cores, {memory_mib // 1024} GiB on an L4\n"
        )
        calls.append(
            function.spawn(
                name, url, views, keep_masks, renderer, coverage_rounds, cpus, memory_mib
            )
        )
    for call in calls:
        result = call.get()
        folder = Path(out) / result["name"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "log.txt").write_text(result["log"], encoding="utf-8")
        for file, data in result.get("files", {}).items():
            (folder / file).write_bytes(data)
        summary = {k: v for k, v in result.items() if k not in ("files", "log")}
        (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        sys.stdout.write(json.dumps(summary) + "\n")
        if not result["ok"]:
            failed.append(result["name"])
    if failed:
        raise SystemExit(f"segmentation failed for {', '.join(failed)}")

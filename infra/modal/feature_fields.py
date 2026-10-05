"""Bake-off candidate B on a Modal GPU: objects from a scale-conditioned feature field
(`tools/captures/feature_fields.py`, whose docstring is the method), for the published scans
that have their photos and poses -- the spool and the pumpkin.

One call per scan, on an L4:

1. the scan's tileset (every tile, as `infra/modal/segment.py` fetches it) from its public
   URL, and the capture's photos, COLMAP poses and placement from the private bucket
   (`runs/<job>/normalize/frames/`, `runs/<job>/pose/poses/`, `runs/<job>/place/
   placement.json`, the Modal secret `twin-object-storage`);
2. `feature_fields.py train`: SAM 2.1 hiera-large masks on the photos (renders if the photos
   do not match the splat), then the field (`STEPS` steps);
3. `feature_fields.py finish`: the tree, the ground and its cover classes, SigLIP 2
   descriptions, the tile binding, an overview sheet drawn by gsplat.

Nothing is written to any bucket. The call returns `instances.json`, `instances.emb`,
`overview.png`, both halves' summaries and `field.npz` (the trained field, so the tree can
be re-read on a CPU without another GPU run), and what the call cost: its wall time at the
reservation's rates (`cost`).

**Cost (estimated before the first run; the coordinator's cap for the whole candidate is
$4).** An L4 with 4 cores and 16 GiB: $0.80 + 4 x $0.047 + 16 x $0.008 = $1.12 an hour.
Per scan: fetching ~1 min, SAM on at most 128 photos ~8 min (2.8 s a view at 32 points a
side with the tiny checkpoint, `segment.py`; the large one's encoder adds ~0.5 s), the field
(3,000 steps) ~3 min, the rest ~4 min: ~16 min, ~$0.30. Each call stops at `TIMEOUT_S` (45
min, $0.84), so the two scans cannot pass $1.68 together.

Run from the repository root (`.github/workflows/segment-feature-fields.yml` does):

    modal run infra/modal/feature_fields.py --names spool,pumpkin --out ff-out
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "hexapod-feature-fields"
app = modal.App(APP_NAME)
WEIGHTS = modal.Volume.from_name("hexapod-world-model-weights", create_if_missing=True)
STORAGE = modal.Secret.from_name("twin-object-storage")
HF_HOME = "/weights/hf"
CAPTURES = "/root/captures"
PIPELINE = "/root/pipeline"

if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    LOCAL_CAPTURES = ROOT / "tools" / "captures"
    LOCAL_SFM = ROOT / "tools" / "pipeline" / "sfm.py"
else:
    LOCAL_CAPTURES, LOCAL_SFM = Path(CAPTURES), Path(PIPELINE) / "sfm.py"

PUBLIC = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs"
#: Scans with photos: (public tileset URL, the pipeline run whose photos and poses made it).
SCANS: dict[str, tuple[str, str]] = {
    "spool": (
        f"{PUBLIC}/8e1cc115-cb80-4af2-81fc-dccaf6b65891/package/splat/tileset.json",
        "8e1cc115-cb80-4af2-81fc-dccaf6b65891",
    ),
    "pumpkin": (
        f"{PUBLIC}/430c1932-5b6a-47b1-bb71-bb7fa2fec86b/package/splat/tileset.json",
        "430c1932-5b6a-47b1-bb71-bb7fa2fec86b",
    ),
}

GPU = "L4"
CORES = 4.0
MEMORY_MIB = 16 * 1024
TIMEOUT_S = 45 * 60
#: Modal's rates (per hour), as infra/modal/segment.py states them.
L4_PER_HOUR = 0.80
CORE_PER_HOUR = 0.047
GIB_PER_HOUR = 0.008
DOLLARS_PER_HOUR = L4_PER_HOUR + CORES * CORE_PER_HOUR + MEMORY_MIB / 1024 * GIB_PER_HOUR
STEPS = 3000
MAX_FRAMES = 128

#: As infra/modal/segment.py: gsplat's prebuilt wheel, torch 2.4 cu124, Python 3.10.
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
        "numpy==1.26.4",
        "ninja",
        "jaxtyping",
        "rich",
        "packaging",
        GSPLAT_WHEEL,
        "transformers==4.57.6",
        "sentencepiece",
        "pillow>=10",
        "laspy[lazrs]>=2.5",
        "pyproj>=3.6",
        "scipy>=1.11,<1.16",
        "opencv-python-headless==4.10.0.84",
        "boto3>=1.34",
    )
    .env({"HF_HOME": HF_HOME})
    .add_local_file(LOCAL_SFM, f"{PIPELINE}/sfm.py")
    .add_local_dir(
        LOCAL_CAPTURES,
        CAPTURES,
        ignore=["**/.venv/**", "**/__pycache__/**", "tests/**", "**/*.pyc"],
    )
)

#: The public bucket answers Python's default user agent with 403; curl's is let through.
USER_AGENT = "curl/8.5.0 (hexapod-feature-fields)"


def _get(url: str, timeout: float) -> bytes:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"GET {url}: HTTP {error.code} {error.reason}") from None


def _fetch_tiles(url: str, out: Path) -> int:
    """The tileset and every tile it names (parents too: the binding covers them)."""
    import concurrent.futures

    if not url.startswith("https://"):
        raise ValueError(f"not an https URL: {url}")
    out.mkdir(parents=True, exist_ok=True)
    base = url.rsplit("/", 1)[0]
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


def _fetch_capture(job: str, out: Path) -> dict[str, int]:
    """The run's photos, COLMAP model and placement from the private bucket."""
    import concurrent.futures
    import os

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["OBJECT_STORAGE_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OBJECT_STORAGE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["OBJECT_STORAGE_SECRET_KEY"],
        region_name=os.environ.get("OBJECT_STORAGE_REGION", "auto"),
    )
    bucket = os.environ["OBJECT_STORAGE_BUCKET"]
    counts: dict[str, int] = {}
    wanted: list[tuple[str, Path]] = []
    for name, prefix in (
        ("frames", f"runs/{job}/normalize/frames/"),
        ("poses", f"runs/{job}/pose/poses/"),
    ):
        (out / name).mkdir(parents=True, exist_ok=True)
        pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        found = [
            (item["Key"], out / name / item["Key"][len(prefix) :])
            for page in pages
            for item in page.get("Contents", [])
            if item["Key"][len(prefix) :] and "/" not in item["Key"][len(prefix) :]
        ]
        counts[name] = len(found)
        wanted += found
    wanted.append((f"runs/{job}/place/placement.json", out / "placement.json"))

    def get(entry: tuple[str, Path]) -> None:
        client.download_file(bucket, entry[0], str(entry[1]))

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        list(pool.map(get, wanted))
    return counts


def _run(argv: list[str], log: list[str]) -> int:
    started = time.time()
    done = subprocess.run(argv, cwd=CAPTURES, capture_output=True, text=True, check=False)
    log.append(f"$ {' '.join(argv[1:])}  ({time.time() - started:.0f} s, exit {done.returncode})")
    log.append(done.stdout[-30000:])
    log.append(done.stderr[-30000:])
    return done.returncode


@app.function(
    image=image,
    gpu=GPU,
    cpu=(CORES, CORES * 1.5),
    memory=(MEMORY_MIB, MEMORY_MIB * 2),
    volumes={"/weights": WEIGHTS},
    secrets=[STORAGE],
    timeout=TIMEOUT_S,
)
def field_scan(name: str, url: str, job: str, steps: int = STEPS) -> dict:
    """One scan, both halves; returns its files (bytes), summaries, log and cost."""
    started = time.time()
    log: list[str] = []
    files: dict[str, bytes] = {}
    summary: dict = {"name": name, "url": url, "job": job, "gpu": GPU, "steps": steps}
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        tiles = root / "tiles"
        summary["tiles"] = _fetch_tiles(url, tiles)
        capture = root / "capture"
        try:
            summary["capture"] = _fetch_capture(job, capture)
            photos = [
                "--frames", str(capture / "frames"), "--poses", str(capture / "poses"),
                "--placement", str(capture / "placement.json"),
            ]  # fmt: skip
        except Exception as error:  # noqa: BLE001 - no photos: the field trains on renders
            summary["capture"] = f"{type(error).__name__}: {error}"
            photos = []
        summary["fetchS"] = round(time.time() - started, 1)
        python = sys.executable
        ok = _run(
            [python, "feature_fields.py", "train", str(tiles / "tileset.json"),
             "--out", str(root / "field.npz"), "--steps", str(steps),
             "--max-frames", str(MAX_FRAMES), "--summary", str(root / "train.json"),
             "--sheet", str(root / "train-sheet.jpg"), *photos],
            log,
        ) == 0  # fmt: skip
        WEIGHTS.commit()
        summary["trainS"] = round(time.time() - started - summary["fetchS"], 1)
        if ok:
            out = root / "out"
            ok = _run(
                [python, "feature_fields.py", "finish", str(tiles / "tileset.json"),
                 "--field", str(root / "field.npz"), "--out", str(out),
                 "--embedder", "segment_models:SiglipEmbedder",
                 "--vocabulary", "data/open_vocabulary.txt", "--gsplat",
                 "--cpus", str(int(CORES)), "--memory-gb", str(MEMORY_MIB / 1024)],
                log,
            ) == 0  # fmt: skip
            WEIGHTS.commit()
            for file in ("instances.json", "instances.emb", "overview.png", "summary.json"):
                if (out / file).exists():
                    files[file] = (out / file).read_bytes()
        for file in ("field.npz", "train.json", "train-sheet.jpg"):
            if (root / file).exists():
                files[file] = (root / file).read_bytes()
    elapsed = time.time() - started
    summary.update(
        ok=ok,
        totalS=round(elapsed, 1),
        cost={
            "dollarsPerHour": round(DOLLARS_PER_HOUR, 3),
            "dollars": round(elapsed / 3600 * DOLLARS_PER_HOUR, 3),
            "basis": f"{GPU}, {CORES:g} cores, {MEMORY_MIB // 1024} GiB requested, wall time",
        },
    )
    return {"summary": summary, "files": files, "log": "\n".join(log)}


@app.local_entrypoint()
def main(names: str = "spool,pumpkin", out: str = "ff-out", steps: int = STEPS) -> None:
    """The named scans in parallel containers; each result under `out/<name>/`."""
    chosen = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in chosen if n not in SCANS]
    if unknown:
        raise SystemExit(f"no photos known for {unknown}; scans: {sorted(SCANS)}")
    calls = [(n, field_scan.spawn(n, SCANS[n][0], SCANS[n][1], steps)) for n in chosen]
    failed, total = [], 0.0
    for name, call in calls:
        try:
            result = call.get()
        except Exception as error:  # noqa: BLE001 - the other scan's result is still written
            result = {"summary": {"name": name, "ok": False}, "files": {}, "log": repr(error)}
        folder = Path(out) / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "log.txt").write_text(result["log"], encoding="utf-8")
        for file, data in result["files"].items():
            (folder / file).write_bytes(data)
        summary = result["summary"]
        (folder / "run.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        sys.stdout.write(json.dumps(summary) + "\n")
        total += float(summary.get("cost", {}).get("dollars", 0.0))
        if not summary.get("ok"):
            failed.append(name)
    sys.stdout.write(f"estimated cost of this run: ${total:.2f}\n")
    if failed:
        raise SystemExit(f"feature fields failed for {', '.join(failed)}")

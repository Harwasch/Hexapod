"""Concept-first segmentation (bake-off candidate C) on a Modal GPU:
`tools/captures/concept_scene.py` for scans already published as tilesets.

docs/SCENE_OBJECTS.md §3 and the module's docstring are the method; this file only puts it
on a GPU, as `infra/modal/segment.py` does for the class-free pipeline (the same image:
torch 2.4 cu124, gsplat's wheel, transformers 4.57, which carries Qwen3-VL, Grounding DINO,
SAM 2 and SigLIP 2). Each call fetches a scan's tileset and every tile from its public URL,
runs the CLI, and hands back `instances.json`, `instances.emb`, `vocabulary.json`, the
run's summary and the overview renders (by object, by ground class, and the views the
vocabulary was read from). Nothing is written to any bucket: publishing is
`publish-instances.yml`'s, into `variants/objects/<name>/`.

**Segmenters.** `sam3` is the method (SAM 3; gated weights, and transformers 5, which this
image does not have: `prepare` says whether the `huggingface` secret's account may read
them, and a `sam3` run stops there, before any GPU, when it may not). `standin` is
Grounding DINO + SAM 2.1 + SigLIP 2 (`concept_models.GroundedSam2Concepts`), marked
`standIn` in `instances.json`: what can run while SAM 3 is gated.

**Money.** The owner's cap for this candidate is $3 of GPU. `main` refuses to start when the
worst case -- every scan's call running to `TIMEOUT_S` at `DOLLARS_PER_HOUR` -- is over
`--budget`. An L4 with 6 cores and 16 GiB (spool and pumpkin, as `segment.py` sizes them):
~$1.21/h; a scan is expected at 15-20 min (~$0.35), and is stopped at 40 min (~$0.81).
`prepare` (weights into the volume, the access check) runs on 2 CPU cores, no GPU.

    modal run infra/modal/segment_concepts.py --names spool,pumpkin --segmenter standin
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "hexapod-segment-concepts"
app = modal.App(APP_NAME)
WEIGHTS = modal.Volume.from_name("hexapod-world-model-weights", create_if_missing=True)
HF_HOME = "/weights/hf"
CAPTURES = "/root/captures"
#: The Modal secret with the Hugging Face token: `huggingface`, or what CI finds by prefix.
HF_SECRET = modal.Secret.from_name(os.environ.get("HEXAPOD_HF_SECRET", "huggingface"))
HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")

if modal.is_local():
    LOCAL_CAPTURES = Path(__file__).resolve().parents[2] / "tools" / "captures"
else:
    LOCAL_CAPTURES = Path(CAPTURES)

#: The scans this candidate runs on (the owner's choice: not the camp), by their legacy
#: public tileset URLs (`segment.py`'s `SCANS`).
PUBLIC = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs"
SCANS: dict[str, str] = {
    "spool": f"{PUBLIC}/8e1cc115-cb80-4af2-81fc-dccaf6b65891/package/splat/tileset.json",
    "pumpkin": f"{PUBLIC}/430c1932-5b6a-47b1-bb71-bb7fa2fec86b/package/splat/tileset.json",
}
USER_AGENT = "curl/8.5.0 (hexapod-segment)"

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
        # Qwen3-VL (4.57.0+), Grounding DINO, SAM 2 and SigLIP 2; the 5.x line needs torch 2.5.
        "transformers==4.57.6",
        "sentencepiece",
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

#: The models a run downloads (`prepare` fetches them into the volume, off the GPU).
VOCABULARY_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
MODELS = (
    VOCABULARY_MODEL,
    "IDEA-Research/grounding-dino-base",
    "facebook/sam2.1-hiera-tiny",
    "google/siglip2-base-patch16-224",
)
SAM3_REPO = "facebook/sam3"
SEGMENTERS = {
    "standin": "concept_models:GroundedSam2Concepts",
    "sam3": "concept_models:Sam3Concepts",
}
#: What each segmenter's run is published as (`<scan>/variant.json`, which
#: publish-instances.yml registers in `extras.variants.objects`): only SAM 3's is candidate
#: C; the stand-in has a name of its own, so it can never be published as C.
VARIANTS = {
    "sam3": {
        "name": "concept-first",
        "label": "C · Concept first",
        "about": (
            "A vision-language model lists the scene's things and ground cover; SAM 3 finds "
            "each of them in every view, the masks are voted onto the splats, and what nobody "
            "named comes from a class-free pass."
        ),
    },
    "standin": {
        "name": "concept-first-standin",
        "label": "C (stand-in) · Concept first, Grounding DINO + SAM 2",
        "about": (
            "Concept first as C, with Grounding DINO and SAM 2 standing in for SAM 3 (whose "
            "weights are gated): a vision-language model lists the things and ground cover, "
            "each is found in every view and voted onto the splats."
        ),
    },
}

#: The reservation, as `segment.py` sizes spool and pumpkin: an L4, 6 cores (4 render
#: processes), 16 GiB (limits 1.5x and 2x).
GPU = "L4"
CORES = 6.0
MEMORY_MIB = 16 * 1024
#: Modal's rates per hour (modal.com/pricing): the L4, a core, a GiB.
L4_PER_HOUR = 0.80
CORE_PER_HOUR = 0.047
GIB_PER_HOUR = 0.008
DOLLARS_PER_HOUR = L4_PER_HOUR + CORES * CORE_PER_HOUR + MEMORY_MIB / 1024 * GIB_PER_HOUR
#: A call is stopped at this; the budget check counts every call running to it.
TIMEOUT_S = 40 * 60
#: `prepare`: 2 cores and 8 GiB, no GPU, at most this long.
PREPARE_TIMEOUT_S = 20 * 60
PREPARE_PER_HOUR = 2 * CORE_PER_HOUR + 8 * GIB_PER_HOUR

#: The views of a run: whole-scan views (rings and observers; spool and pumpkin, ~6 m
#: across, are within one view's footprint, so they get no local views) and at most
#: `MAX_VIEWS` with local ones. 64 rather than `segment_scene`'s 24: there are no coverage
#: rounds here, and a view costs ~3 s of the L4 (SAM 2's grid most of it).
VIEWS = 64
MAX_VIEWS = 160
MAX_SCALE_M = 0.5


def worst_case_dollars(scans: int) -> float:
    """What the run could cost at most: every scan's call to `TIMEOUT_S`, and `prepare`."""
    return scans * TIMEOUT_S / 3600 * DOLLARS_PER_HOUR + PREPARE_TIMEOUT_S / 3600 * PREPARE_PER_HOUR


def _hf_token() -> list[str]:
    """HF_TOKEN from whichever key the secret uses; the keys present (names only)."""
    present = sorted(k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper())
    keys = [k for k in HF_TOKEN_KEYS if os.environ.get(k)]
    keys += sorted(k for k, v in os.environ.items() if v.startswith("hf_"))
    if keys:
        os.environ["HF_TOKEN"] = os.environ[keys[0]]
    return present


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=PREPARE_TIMEOUT_S,
)
def prepare() -> dict:
    """Off the GPU: whether the secret's account may read SAM 3, and every model a run
    needs downloaded into the weights volume (so the GPU does not wait for them)."""
    from huggingface_hub import HfApi, snapshot_download

    started = time.time()
    out: dict = {"secretKeys": _hf_token()}
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        out["account"] = api.whoami().get("name")
    except Exception as error:  # noqa: BLE001 - reported, not raised
        out["account"] = f"error: {type(error).__name__}"
    try:
        api.auth_check(SAM3_REPO)
        out["sam3"] = "ok"
    except Exception as error:  # noqa: BLE001 - reported, not raised
        out["sam3"] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
    for repo in MODELS:
        snapshot_download(
            repo,
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja"],
        )
    WEIGHTS.commit()
    out["models"] = list(MODELS)
    out["seconds"] = round(time.time() - started, 1)
    return out


def _get(url: str, timeout: float) -> bytes:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"GET {url}: HTTP {error.code} {error.reason}") from None


def _fetch(url: str, out: Path) -> int:
    """The tileset and every tile it names, parents included."""
    import concurrent.futures

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


def concept_argv(
    tiles: Path, work: Path, *, segmenter: str, views: int, max_views: int
) -> list[str]:
    """`concept_scene.py`'s command line for one scan."""
    return [
        sys.executable,
        "concept_scene.py",
        str(tiles / "tileset.json"),
        str(tiles),
        "--vocabulary",
        "concept_models:QwenVocabulary",
        "--concepts",
        SEGMENTERS[segmenter],
        "--masks",
        "segment_models:Sam2Masks",
        "--embedder",
        "segment_models:SiglipEmbedder",
        "--words",
        "data/open_vocabulary.txt",
        "--views",
        str(views),
        "--max-views",
        str(max_views),
        "--renderer",
        "gsplat",
        "--max-scale-m",
        str(MAX_SCALE_M),
        "--cpus",
        str(int(CORES)),
        "--memory-gb",
        f"{MEMORY_MIB / 1024:g}",
        *(["--stand-in"] if segmenter == "standin" else []),
        "--out",
        str(work / "out"),
        "--overviews",
        str(work / "out" / "overviews"),
        "--summary",
        str(work / "run.json"),
    ]


@app.function(
    image=image,
    gpu=GPU,
    cpu=(CORES, CORES * 1.5),
    memory=(MEMORY_MIB, MEMORY_MIB * 2),
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=TIMEOUT_S,
)
def segment_scan(name: str, url: str, segmenter: str, views: int, max_views: int) -> dict:
    """Segment one published scan by concepts; the files (bytes) and the run's summary."""
    _hf_token()
    started = time.time()
    with tempfile.TemporaryDirectory() as work_dir:
        work = Path(work_dir)
        tiles = work / "tiles"
        count = _fetch(url, tiles)
        fetched = time.time() - started
        run = subprocess.run(  # noqa: S603 - fixed argv; only paths we made vary
            concept_argv(tiles, work, segmenter=segmenter, views=views, max_views=max_views),
            cwd=CAPTURES,
            capture_output=True,
            text=True,
            check=False,
        )
        log = run.stdout[-30000:] + run.stderr[-30000:]
        try:
            summary = json.loads((work / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = {}
        files: dict[str, bytes] = {}
        out = work / "out"
        if out.exists():
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    files[str(path.relative_to(out))] = path.read_bytes()
        seconds = round(time.time() - started, 1)
        return {
            "name": name,
            "ok": run.returncode == 0 and "instances.json" in files,
            "segmenter": segmenter,
            "tiles": count,
            "fetchS": round(fetched, 1),
            "totalS": seconds,
            "dollars": round(seconds / 3600 * DOLLARS_PER_HOUR, 3),
            "run": summary,
            "files": files,
            "log": log,
        }


@app.local_entrypoint()
def main(
    names: str = ",".join(SCANS),
    segmenter: str = "standin",
    views: int = VIEWS,
    max_views: int = MAX_VIEWS,
    out: str = "segment-out",
    budget: float = 3.0,
) -> None:
    """`prepare`, then every named scan in its own container; each result under `out/`
    (`<scan>/instances.json`, ... as `segment.py` writes them, for `publish-instances.yml`)."""
    chosen = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in chosen if n not in SCANS]
    if unknown:
        raise SystemExit(f"unknown scans {unknown}: this candidate runs on {sorted(SCANS)}")
    if segmenter not in SEGMENTERS:
        raise SystemExit(f"segmenter {segmenter!r}: one of {sorted(SEGMENTERS)}")
    worst = worst_case_dollars(len(chosen))
    sys.stdout.write(
        f"{len(chosen)} scans on an {GPU} at ${DOLLARS_PER_HOUR:.2f}/h, stopped at "
        f"{TIMEOUT_S // 60} min each: at most ${worst:.2f} (budget ${budget:.2f})\n"
    )
    if worst > budget:
        raise SystemExit(f"the worst case (${worst:.2f}) is over the budget (${budget:.2f})")
    prepared = prepare.remote()
    sys.stdout.write("prepare: " + json.dumps(prepared) + "\n")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "prepare.json").write_text(json.dumps(prepared, indent=1), encoding="utf-8")
    if segmenter == "sam3" and prepared.get("sam3") != "ok":
        raise SystemExit(
            f"SAM 3 is gated and the secret's account ({prepared.get('account')}) has no "
            f"access: {prepared.get('sam3')}. Request it at https://huggingface.co/{SAM3_REPO}."
        )
    calls = [
        (name, segment_scan.spawn(name, SCANS[name], segmenter, views, max_views))
        for name in chosen
    ]
    failed = []
    spent = prepared.get("seconds", 0) / 3600 * PREPARE_PER_HOUR
    for name, call in calls:
        try:
            result = call.get()
        except Exception as error:  # noqa: BLE001 - e.g. killed at its timeout or memory limit
            result = {"name": name, "ok": False, "log": f"{type(error).__name__}: {error}"}
        folder = Path(out) / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "log.txt").write_text(result.get("log", ""), encoding="utf-8")
        for file, data in result.get("files", {}).items():
            (folder / file).parent.mkdir(parents=True, exist_ok=True)
            (folder / file).write_bytes(data)
        summary = {k: v for k, v in result.items() if k not in ("files", "log")}
        (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        if result["ok"]:
            (folder / "variant.json").write_text(
                json.dumps(VARIANTS[segmenter], indent=1, ensure_ascii=False), encoding="utf-8"
            )
        spent += float(result.get("dollars", TIMEOUT_S / 3600 * DOLLARS_PER_HOUR))
        sys.stdout.write(
            f"{name}: {'ok' if result['ok'] else 'FAILED'} in {result.get('totalS', '?')} s, "
            f"~${result.get('dollars', '?')}\n"
        )
        if not result["ok"]:
            failed.append(name)
    sys.stdout.write(f"spent ~${spent:.2f} (from wall time at the reservation's rates)\n")
    (Path(out) / "cost.json").write_text(json.dumps({"dollars": round(spent, 3)}), encoding="utf-8")
    if failed:
        raise SystemExit(f"concept segmentation failed for {', '.join(failed)}")

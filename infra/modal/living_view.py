"""Living view bake-off on Modal: slight, anchored wind on the plants of a still render.

The owner's idea: the viewer shows the measured splat; when the camera stops, a video model
adds slight wind to the plants, conditioned on the exact render of the splat, in short segments
that start again from that render (so nothing drifts); when the camera moves, the plain splat.
Which model gives believable, slight, anchored motion, and is the model's *motion* applied to
our full-resolution render (`tools/captures/living_view.py`, run on the clips afterwards)
better than its pixels? Nothing here is written to any bucket or published.

Steps (`--steps`, comma-separated, in this order):

    access    CPU. Which Hub repositories the arms need the token in the `huggingface` secret
              can read (`auth_check` and one file's metadata each), and which account it is.
              The model cards of the gated LTX repositories come back with it.
    download  CPU, one container per arm: its weights into the volume
              `hexapod-living-view-weights` -- never while holding a GPU.
    starts    L4. Two rest views per scene, drawn with gsplat at 1280x704 over a pale sky, each
              with a feathered plant mask (a label render of the plant gaussians): the
              Minnetonka tree from its drone orbit's ring (the whole tree above its lawn, less
              the stem's foot), the camp's shrubs from the places on its clearing and trail the
              capture walked (`instances.json` categories and best tags,
              `living_view.plant_instances`). Kept in the volume `hexapod-living-view` for the
              arms and returned.
    arms      One GPU container per arm, every start in it, one seed, the same prompt (adapted
              per model): a gentle breeze, leaves and thin branches sway slightly, a static
              locked-off tripod camera, nothing else changes.

The arms (licences in `ARMS`):

    ltx      LTX-2.5 distilled (Lightricks/LTX-2.5-Diffusers, diffusers at the commit its own
             Space pins) with the Cinemagraph LoRA, two stages as that Space runs them: 8
             sigmas at half size, x2 latent upsample, 3 at 1280x704; 97 frames at 24 fps. H200.
    causal   Causal Forcing (zhuhz22/Causal-Forcing, frame-wise, on Wan 2.1 T2V 1.3B): text to
             video, started from our frame by pre-filling its causal KV cache with the render's
             latent (the repository's own `initial_latent` path), 16 latent frames streamed
             after it (65 frames at 16 fps, 832x480). Time to first motion is the render
             encoded, the cache filled, the first new latent frame denoised and decoded. L40S.
    flf      Wan 2.1 FLF2V 14B (diffusers): first and last frame both our render, so the clip
             returns exactly; 848x464 (480p area), 65 frames at 16 fps, 30 steps. H100.
    wan      The control: Wan 2.2 TI2V-5B as `world_models.Wan` runs it (1280x704, 50 steps,
             guidance 5), 97 frames at 24 fps. H100.

Costs: every GPU function is `single_use_containers` (no idle window billed) and runs every
start in one warm container. The entrypoint refuses arms whose worst case (timeout x rate)
does not fit in `--budget-left`, and reports each call's wall time and dollars (Modal's list
prices, read 2026-10-08) for `bakeoff/living-view/cost-ledger.md`.

    modal run infra/modal/living_view.py --steps access,download,starts --arms wan,flf,causal
    modal run infra/modal/living_view.py --steps arms --arms wan,flf --budget-left 8
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "hexapod-living-view"
app = modal.App(APP_NAME)
CAPTURES = "/root/captures"

if modal.is_local():
    LOCAL_CAPTURES = Path(__file__).resolve().parents[2] / "tools" / "captures"
else:
    LOCAL_CAPTURES = Path(CAPTURES)

#: Wan 2.2's weights are already cached here by `world_models.Wan` (HF_HOME=/weights/hf).
WEIGHTS = modal.Volume.from_name("hexapod-world-model-weights", create_if_missing=True)
#: This bake-off's weights (about 190 GB; delete the volume when it is decided).
LV_WEIGHTS = modal.Volume.from_name(
    "hexapod-living-view-weights", create_if_missing=True, version=2
)
#: The starts (renders, masks, cameras) the arms read.
RESULTS = modal.Volume.from_name("hexapod-living-view", create_if_missing=True, version=2)
HF_SECRET = modal.Secret.from_name(os.environ.get("HEXAPOD_HF_SECRET", "huggingface"))
HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")

PUBLIC = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev"
SCANS: dict[str, str] = {
    "tree": f"{PUBLIC}/sites/minnetonka-tree/splat/tileset.json",
    "camp": f"{PUBLIC}/runs/50c25673-0940-4574-9b96-0b21362f83ca/package/splat/tileset.json",
}
USER_AGENT = "curl/8.5.0 (hexapod-living-view)"

START_SIZE = (1280, 704)
FOV_DEG = 60.0
EYE_HEIGHT_M = 1.6
#: The camp's places on its clearing and trail (dream.py's EYES, tileset ENU metres).
CAMP_EYES = [(9.0, 18.3), (-4.3, 1.7), (0.0, 10.0)]
#: Gaussians larger than this are not drawn (the camp's edge floaters, dream.py's rule).
MAX_SCALE_M = 1.0
SKY = (0.78, 0.82, 0.86)  # teacher_materials.SKY

#: What each scene is, for the prompts (a start may override it once seen: `START_SCENES`).
SCENE_TEXT: dict[str, str] = {
    "tree": "A single large leafy deciduous tree standing alone on a summer day, pale sky behind",
    "camp": (
        "Dense green shrubs, ferns and young trees at the edge of a forest campsite, "
        "overcast daylight"
    ),
}
START_SCENES: dict[str, str] = {}
MOTION_TEXT = "A gentle breeze: the leaves and thin branches sway slightly and settle."
STATIC_TEXT = (
    "Static locked-off tripod shot, no camera movement, no pan, no zoom. Nothing else changes."
)
NEGATIVE = (
    "camera motion, panning, zooming, handheld shake, cuts, scene change, strong wind, "
    "people appearing, text, watermark, blur, flicker, low quality"
)
LTX_TRIGGER = "CINEMAGRAPH_MOTION"

# --- the arms -------------------------------------------------------------------------------

LTX_REPO = "Lightricks/LTX-2.5-Diffusers"
LTX_LORA_REPO = "Lightricks/LTX-2.5-22b-LoRA-Cinemagraph"
LTX_LORA_FILE = "ltx-2.5-22b-lora-cinemagraph-0.9.safetensors"
LTX_SINGLE_REPO = "Lightricks/LTX-2.5"
#: diffusers' main at LTX-2.5's merge (PR #14447), as Lightricks' own Space pins it.
LTX_DIFFUSERS_COMMIT = "7564fb016dabda0c943416190fc92398c50b1b20"
LTX_FRAMES = 97
LTX_FPS = 24.0
LTX_LORA_SCALE = 1.0

CF_REPO = "zhuhz22/Causal-Forcing"
CF_CHECKPOINT = "framewise/causal_forcing.pt"
CF_BASE = "Wan-AI/Wan2.1-T2V-1.3B"
CF_CODE = "https://github.com/thu-ml/Causal-Forcing.git"
CF_COMMIT = "8202903297918bcd12810fc4230a4db7fe065e64"
CF_LATENT_FRAMES = 16
CF_SIZE = (832, 480)
CF_FPS = 16.0

FLF_REPO = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
FLF_SIZE = (848, 464)
FLF_FRAMES = 65
FLF_FPS = 16.0
FLF_STEPS = 30
FLF_GUIDANCE = 5.5
#: The repository's scheduler shift (16) is its 720p one; 480p takes a smaller one.
FLF_SHIFT = 5.0

WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_SIZE = (1280, 704)
WAN_FRAMES = 97
WAN_FPS = 24.0

#: Modal list prices, $/s (modal.com/pricing, read 2026-10-08).
GPU_PER_S = {"H200": 0.001261, "H100": 0.001097, "L40S": 0.000542, "L4": 0.000222, "": 0.0}
CPU_CORE_PER_S = 0.0000131
MEMORY_GIB_PER_S = 0.00000222

#: Per arm: its GPU and reservation (for the dollars), timeout, what it needs readable, the
#: licence of the model and of the code it runs.
ARMS: dict[str, dict] = {
    "ltx": {
        "gpu": "H200",
        "cpu": 8.0,
        "memoryGiB": 96,
        "timeoutS": 1500,
        "needs": (LTX_REPO, LTX_LORA_REPO),
        "model": f"{LTX_REPO} (distilled) + {LTX_LORA_REPO}",
        "licence": "LTX-2.x Community License (commercial use free under $10M revenue)",
    },
    "causal": {
        "gpu": "L40S",
        "cpu": 4.0,
        "memoryGiB": 48,
        "timeoutS": 900,
        "needs": (CF_REPO, CF_BASE),
        "model": f"{CF_REPO} ({CF_CHECKPOINT}) on {CF_BASE}",
        "licence": "Apache-2.0 (weights and code; base Wan 2.1 Apache-2.0)",
    },
    "flf": {
        "gpu": "H100",
        "cpu": 8.0,
        "memoryGiB": 64,
        "timeoutS": 1800,
        "needs": (FLF_REPO,),
        "model": FLF_REPO,
        "licence": "Apache-2.0",
    },
    "wan": {
        "gpu": "H100",
        "cpu": 4.0,
        "memoryGiB": 32,
        "timeoutS": 1500,
        "needs": (WAN_REPO,),
        "model": WAN_REPO,
        "licence": "Apache-2.0",
    },
}
STARTS_RESERVATION = {"gpu": "L4", "cpu": 4.0, "memoryGiB": 32, "timeoutS": 2400}
DOWNLOAD_RESERVATION = {"gpu": "", "cpu": 2.0, "memoryGiB": 8, "timeoutS": 3 * 3600}

#: One file per repository whose metadata proves the token can read it.
ACCESS: dict[str, str] = {
    LTX_REPO: "model_index.json",
    LTX_LORA_REPO: LTX_LORA_FILE,
    LTX_SINGLE_REPO: "vae/ltx-2.5-video-vae-bf16.safetensors",
    CF_REPO: CF_CHECKPOINT,
    CF_BASE: "config.json",
    FLF_REPO: "model_index.json",
    WAN_REPO: "model_index.json",
    "TencentARC/RollingForcing": "README.md",
    "krea/krea-realtime-video": "README.md",
}


def rate_per_s(reservation: dict) -> float:
    """Dollars per second of a container with this reservation."""
    return (
        GPU_PER_S[reservation["gpu"]]
        + reservation["cpu"] * CPU_CORE_PER_S
        + reservation["memoryGiB"] * MEMORY_GIB_PER_S
    )


def prompt_for(arm: str, start: str) -> str:
    scene = START_SCENES.get(start) or SCENE_TEXT[start.split("-")[0]]
    if arm == "ltx":
        return f"{LTX_TRIGGER}. {scene}. {MOTION_TEXT} Tripod locked-off static camera."
    return f"{scene}. {MOTION_TEXT} {STATIC_TEXT}"


# --- images ---------------------------------------------------------------------------------

GSPLAT_WHEEL = (
    "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/"
    "gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"
    "#sha256=01e1fd63dc69c9945e70158c818c3bb07fedf4aabcf020e6608d264cf27cc5dd"
)

#: dream.py's gsplat job image (torch 2.4 cu124, gsplat's wheel, the captures).
job_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.4.1+cu124", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "numpy==1.26.4",
        "ninja",
        "jaxtyping",
        "rich",
        "packaging",
        GSPLAT_WHEEL,
        "pillow>=10",
        "laspy[lazrs]>=2.5",
        "pyproj>=3.6",
        "scipy>=1.11,<1.16",
        "opencv-python-headless==4.10.0.84",
        "matplotlib>=3.8,<3.10",
    )
    .add_local_dir(
        LOCAL_CAPTURES,
        CAPTURES,
        ignore=["**/.venv/**", "**/__pycache__/**", "tests/**", "**/*.pyc"],
    )
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]>=1.23,<2")
    .env({"HF_XET_CHUNK_CACHE_SIZE_BYTES": "0"})
)

#: `world_models.video_image` as it is (so its layers are cached): Wan 2.2 and FLF2V.
video_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "diffusers==0.40.0",
        "transformers>=5,<6",
        "accelerate>=1.6",
        "ftfy",
        "sentencepiece",
        "protobuf",
        "imageio[ffmpeg]>=2.37",
        "pillow",
        "huggingface_hub",
        "cosmos_guardrail==0.3.2",
    )
    .env({"HF_HOME": "/weights/hf"})
)

#: LTX-2.5 as Lightricks' Space runs it (its requirements.txt): diffusers at the merge commit,
#: transformers 5.14.1 (Gemma 4), and peft for the LoRA.
ltx_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0")
    .pip_install(
        f"diffusers @ git+https://github.com/huggingface/diffusers@{LTX_DIFFUSERS_COMMIT}",
        "transformers==5.14.1",
        "huggingface_hub>=1.23.0,<2.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "peft",
        "av",
        "imageio[ffmpeg]>=2.37",
        "pillow",
        "numpy",
        "scipy",
    )
    .env({"HF_HOME": "/lv/hf"})
)

#: Causal Forcing's inference environment (its requirements, the parts inference imports),
#: its code at `CF_COMMIT`. No flash-attn: its attention falls back to torch's SDPA.
cf_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "diffusers==0.31.0",
        "transformers==4.49.0",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "huggingface_hub>=0.26,<1.0",
        "safetensors",
        "numpy==1.26.4",
        "omegaconf",
        "einops",
        "easydict",
        "ftfy",
        "regex",
        "tqdm",
        "imageio",
        "imageio-ffmpeg",
        "av==13.1.0",
        "sentencepiece",
        "lmdb",
        "opencv-python-headless==4.10.0.84",
        "pillow",
    )
    .run_commands(
        f"git clone {CF_CODE} /opt/causal-forcing && git -C /opt/causal-forcing checkout {CF_COMMIT}"
    )
)


# --- shared helpers --------------------------------------------------------------------------


def _hf_token() -> str | None:
    """HF_TOKEN set from whichever key the secret uses (`world_models._hf_token`)."""
    keys = [k for k in HF_TOKEN_KEYS if os.environ.get(k)]
    keys += sorted(k for k, v in os.environ.items() if v.startswith("hf_"))
    if keys:
        os.environ["HF_TOKEN"] = os.environ[keys[0]]
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ[keys[0]])
    return os.environ.get("HF_TOKEN")


def _u8(frames: object) -> object:
    import numpy as np

    return np.clip(np.round(np.asarray(frames, np.float32) * 255), 0, 255).astype(np.uint8)


def _mp4(frames: object, fps: float) -> bytes:
    """Near-lossless H.264 (CRF 14): the flow is measured on these frames afterwards."""
    import imageio.v2 as imageio
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        writer = imageio.get_writer(
            f.name, fps=fps, codec="libx264", ffmpeg_params=["-crf", "14"], macro_block_size=1
        )
        for frame in frames:  # type: ignore[attr-defined]
            writer.append_data(np.asarray(frame))
        writer.close()
        return Path(f.name).read_bytes()


def _start_image(name: str) -> object:
    from PIL import Image

    return Image.open(f"/data/starts/{name}.png").convert("RGB")


def _need(paths: list[str]) -> None:
    """Fail at once (before any model loads) when an input is missing."""
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing: {missing}")


def _clip_entry(frames: object, fps: float, seconds: float, first: float, size: list[int]) -> dict:
    import torch

    return {
        "mp4": _mp4(frames, fps),
        "fps": fps,
        "frames": len(frames),  # type: ignore[arg-type]
        "size": size,
        "seconds": round(seconds, 2),
        "firstMotionSeconds": round(first, 2),
        "peakMemoryGB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }


# --- access and downloads ----------------------------------------------------------------------


@app.function(image=download_image, secrets=[HF_SECRET], timeout=300)
def access() -> dict:
    """Which repositories the secret's token reads: `auth_check` and the metadata of one file
    each; the account; the model cards of the LTX repositories it can read."""
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url

    token = _hf_token()
    api = HfApi(token=token)
    out: dict = {"repos": {}, "cards": {}}
    try:
        out["account"] = api.whoami().get("name")
    except Exception as error:  # noqa: BLE001 - reported, not raised
        out["account"] = f"error: {error}"
    for repo, filename in ACCESS.items():
        try:
            api.auth_check(repo)
            meta = get_hf_file_metadata(hf_hub_url(repo, filename), token=token)
            out["repos"][repo] = f"ok ({filename}: {meta.size} bytes)"
        except Exception as error:  # noqa: BLE001 - reported, not raised
            out["repos"][repo] = f"{type(error).__name__}: {str(error).splitlines()[0][:240]}"
    for repo in (LTX_LORA_REPO, LTX_REPO, LTX_SINGLE_REPO):
        if str(out["repos"].get(repo, "")).startswith("ok"):
            with tempfile.TemporaryDirectory() as tmp:
                path = hf_hub_download(repo, "README.md", local_dir=tmp, token=token)
                out["cards"][repo] = Path(path).read_text(encoding="utf-8")[:60000]
    return out


def _ltx_patterns(token: str | None) -> list[str]:
    """LTX-2.5-Diffusers without what this run does not load: `transformer_full/` (the
    non-distilled DiT, 38 GB), the root distilled LoRA, and the second copy of each sharded
    set (only the shards its index names)."""
    from huggingface_hub import hf_hub_download

    patterns = [
        "model_index.json",
        "README.md",
        "scheduler/*",
        "tokenizer/*",
        "processor/*",
        "text_encoder/*",
        "vae/*",
        "audio_vae/*",
        "vocoder/*",
        "latent_upsampler/*",
        "duration_head/*",
        "prompt_enhancer/*",
        # Not used, but named by model_index.json; small enough to keep loading simple.
        "temporal_latent_upsampler/*",
        "diffusion_decoder/*",
    ]
    for sub in ("transformer", "connectors"):
        index = f"{sub}/diffusion_pytorch_model.safetensors.index.json"
        path = hf_hub_download(LTX_REPO, index, token=token)
        shards = sorted(set(json.loads(Path(path).read_text())["weight_map"].values()))
        patterns += [f"{sub}/config.json", index] + [f"{sub}/{s}" for s in shards]
    return patterns


def _download_plan(arm: str, token: str | None) -> list[tuple[str, list[str]]]:
    if arm == "ltx":
        return [(LTX_REPO, _ltx_patterns(token)), (LTX_LORA_REPO, [LTX_LORA_FILE, "README.md"])]
    if arm == "causal":
        base = [
            "config.json",
            "diffusion_pytorch_model.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "google/umt5-xxl/*",
        ]
        return [(CF_BASE, base), (CF_REPO, [CF_CHECKPOINT, "README.md"])]
    if arm == "flf":
        parts = ["image_encoder", "image_processor", "scheduler", "text_encoder", "tokenizer"]
        parts += ["transformer", "vae"]
        return [(FLF_REPO, ["model_index.json"] + [f"{p}/*" for p in parts])]
    if arm == "wan":
        return []  # cached in hexapod-world-model-weights by world_models.Wan
    raise ValueError(f"unknown arm {arm!r}")


@app.function(
    image=download_image,
    secrets=[HF_SECRET],
    volumes={"/lv": LV_WEIGHTS},
    cpu=DOWNLOAD_RESERVATION["cpu"],
    memory=DOWNLOAD_RESERVATION["memoryGiB"] * 1024,
    timeout=DOWNLOAD_RESERVATION["timeoutS"],
)
def download(arm: str) -> dict:
    """An arm's weights into /lv/<repo>/ (CPU only), committed."""
    from huggingface_hub import snapshot_download

    started = time.time()
    token = _hf_token()
    done = []
    for repo, patterns in _download_plan(arm, token):
        local = Path("/lv") / repo
        snapshot_download(
            repo, local_dir=str(local), allow_patterns=patterns, token=token, max_workers=8
        )
        size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
        done.append({"repo": repo, "bytes": size})
        LV_WEIGHTS.commit()
    return {"arm": arm, "repos": done, "seconds": round(time.time() - started, 1)}


# --- starts ----------------------------------------------------------------------------------


def _get(url: str, timeout: float) -> bytes:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(f"GET {url}: HTTP {error.code} {error.reason}") from None
        time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def _fetch(url: str, out: Path) -> Path:
    """A tileset's leaves and its instances.json (dream.py's `_fetch`)."""
    import concurrent.futures

    out.mkdir(parents=True, exist_ok=True)
    base = url.rsplit("/", 1)[0]
    (out / "tileset.json").write_bytes(_get(url, 120))
    document = json.loads((out / "tileset.json").read_text(encoding="utf-8"))
    uris: list[str] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        stack.extend(tile.get("children", []))
        uri = tile.get("content", {}).get("uri")
        if uri and not tile.get("children"):
            uris.append(uri)
    if instances := document["root"].get("extras", {}).get("instances", {}).get("uri"):
        uris.append(instances)

    def get(uri: str) -> None:
        (out / uri).parent.mkdir(parents=True, exist_ok=True)
        (out / uri).write_bytes(_get(f"{base}/{PurePosixPath(uri)}", 600))

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(get, uris))
    return out / "tileset.json"


def _leaves_with_ids(tileset: Path, doc: dict | None) -> tuple[object, object]:
    """Every leaf gaussian (as `splat_render.load_tileset`) and its instance id (0 when
    instances.json does not list the tile), bound by the tiles' position checksums."""
    import numpy as np
    from rig_tiles import glb_spz
    from splat_render import Splats, _from_columns
    from splat_tiles import unpack_spz
    from synthetic_tree import checksum_positions

    document = json.loads(tileset.read_text(encoding="utf-8"))
    leaves: list[str] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
        else:
            leaves.append(tile["content"]["uri"])
    # A checksum is `fnv1a32:<count>:<hash>`, and FNV in Python is slow over 20M gaussians:
    # a tile whose count no other listed tile has is matched by its count alone.
    by_count: dict[int, list[str]] = {}
    for key in (doc or {}).get("tiles", {}):
        by_count.setdefault(int(key.split(":")[1]), []).append(key)
    parts, ids = [], []
    for uri in sorted(leaves):
        columns = unpack_spz(glb_spz(tileset.parent / uri))
        n = len(columns["x"])
        tile_ids = np.zeros(n, np.int64)
        if doc is not None and by_count.get(n):
            if len(by_count[n]) == 1:
                runs = doc["tiles"][by_count[n][0]]
            else:
                positions = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
                positions = positions.astype("<f4")
                positions[positions == 0.0] = 0.0
                runs = doc["tiles"].get(checksum_positions(positions))
            if runs is not None:
                pairs = np.asarray(runs, np.int64).reshape(-1, 2)
                decoded = np.repeat(pairs[:, 0], pairs[:, 1])
                if decoded.size == n:
                    tile_ids = decoded
        parts.append(_from_columns(columns))
        ids.append(tile_ids)
    return Splats.concat(parts), np.concatenate(ids)


def _candidates(scene: str, positions: object, plant: object, vfov: float) -> list[dict]:
    """Rest views to choose from: `{"name", "eye", "target", "place", "bearing"}`. The tree:
    a ring round its crown at the distance that frames it, at eye height and mid-crown (its
    drone orbit's tiers). The camp: the clearing and trail places, 16 bearings, level and
    10 degrees down."""
    import math

    import numpy as np

    p = np.asarray(positions)
    out = []
    if scene == "tree":
        crown = p[np.asarray(plant) > 0]
        if len(crown) < 100:
            raise ValueError(f"only {len(crown)} plant gaussians: is the origin the stem's foot?")
        cx, cy = np.median(crown[:, 0]), np.median(crown[:, 1])
        lo, hi = np.percentile(crown[:, 2], [2, 98])
        mid = float(lo + 0.5 * (hi - lo))
        frame_d = 0.5 * (hi - lo) / math.tan(math.radians(vfov) / 2)
        for scale in (0.75, 1.0):
            for height in (1.7, mid):
                for bearing in range(0, 360, 30):
                    b = math.radians(bearing)
                    d = frame_d * scale
                    out.append(
                        {
                            "name": f"r{d:.1f}-h{height:.1f}-b{bearing:03d}",
                            "eye": [cx + d * math.sin(b), cy + d * math.cos(b), height],
                            "target": [float(cx), float(cy), mid],
                            "place": f"h{height:.1f}",
                            "bearing": bearing,
                        }
                    )
        return out
    for k, (x, y) in enumerate(CAMP_EYES):
        near = np.hypot(p[:, 0] - x, p[:, 1] - y) < 1.5
        if near.sum() < 50:
            near = np.hypot(p[:, 0] - x, p[:, 1] - y) < 4.0
        ground = float(np.percentile(p[near, 2], 10)) if near.any() else float(np.median(p[:, 2]))
        for bearing in np.arange(0, 360, 22.5):
            for pitch in (0.0, -10.0):
                b, t = math.radians(bearing), math.radians(pitch)
                z = ground + EYE_HEIGHT_M
                out.append(
                    {
                        "name": f"eye{k}-b{bearing:05.1f}-p{pitch:+03.0f}",
                        "eye": [x, y, z],
                        "target": [
                            x + 10 * math.cos(t) * math.sin(b),
                            y + 10 * math.cos(t) * math.cos(b),
                            z + 10 * math.sin(t),
                        ],
                        "place": f"eye{k}",
                        "bearing": float(bearing),
                    }
                )
    return out


@app.function(
    image=job_image,
    gpu="L4",
    cpu=STARTS_RESERVATION["cpu"],
    memory=STARTS_RESERVATION["memoryGiB"] * 1024,
    timeout=STARTS_RESERVATION["timeoutS"],
    volumes={"/data": RESULTS},
    single_use_containers=True,
)
def starts(scene: str, keep: int = 2) -> dict:
    """`keep` rest views of a scene: render, feathered plant mask, an overlay to check the
    mask by eye, the camera, and every candidate's score (and a sheet of them)."""
    import math

    import numpy as np

    sys.path.insert(0, CAPTURES)
    import living_view as lv
    from splat_render import Camera, GsplatRenderer, Splats
    from world_model_client import encode_png

    started = time.time()
    with tempfile.TemporaryDirectory() as work:
        tileset = _fetch(SCANS[scene], Path(work) / "scan")
        doc_path = tileset.parent / "instances.json"
        doc = json.loads(doc_path.read_text(encoding="utf-8")) if doc_path.exists() else None
        splats, ids = _leaves_with_ids(tileset, doc)
    fetched = time.time() - started
    if scene == "tree":
        weight = lv.isolated_plant_flags(splats.positions)
    else:
        weight = lv.plant_flags(ids, lv.plant_instances(doc["instances"]))
    keep_rows = np.flatnonzero(splats.scales.max(axis=1) <= MAX_SCALE_M)
    splats, weight = splats.take(keep_rows), weight[keep_rows]
    label = Splats(
        splats.positions,
        splats.rotations,
        splats.scales,
        np.stack([(weight > 0).astype(np.float64), weight, np.ones_like(weight)], axis=1),
        splats.opacities,
    )
    renderer = GsplatRenderer(keep=2)
    w, h = START_SIZE
    vfov = 2 * math.degrees(math.atan(math.tan(math.radians(FOV_DEG) / 2) * h / w))

    def look(c: dict, width: int, height: int) -> Camera:
        return Camera.look_at(c["eye"], c["target"], fov_deg=FOV_DEG, width=width, height=height)

    scored = []
    thumbs = []
    for c in _candidates(scene, splats.positions, weight, vfov):
        cam = look(c, 320, 176)
        lab = renderer(label, cam)
        cover = lab.rgb[..., 2]
        share = np.where(cover > 1e-3, lab.rgb[..., 0] / np.maximum(cover, 1e-3), 0.0)
        framing = np.where(cover > 1e-3, lab.rgb[..., 1] / np.maximum(cover, 1e-3), 0.0)
        plant_px = share >= lv.PLANT_SHARE
        depth = lab.depth[plant_px]
        median = float(np.median(depth)) if depth.size else 0.0
        near = float((lab.depth[cover > 0.5] < 1.0).mean()) if (cover > 0.5).any() else 1.0
        plant_fraction = float(plant_px.mean())
        if scene == "tree":
            score = plant_fraction * (1.0 - near)
        else:
            coverage = float((cover > 0.5).mean())
            score = float(framing.mean()) * coverage * min(1.0, median / 2.5) * (1.0 - near)
        scored.append(
            {**c, "plantFraction": plant_fraction, "medianPlantDepth": median, "near": near,
             "score": score}
        )  # fmt: skip
        thumbs.append((c["name"], renderer(splats, cam, background=SKY).rgb))
    order = sorted(range(len(scored)), key=lambda k: -scored[k]["score"])
    chosen: list[int] = []
    for k in order:  # another place, or another bearing at least 90 degrees round
        s = scored[k]
        if all(
            s["place"] != scored[o]["place"]
            or 90 <= abs(s["bearing"] - scored[o]["bearing"]) <= 270
            for o in chosen
        ):
            chosen.append(k)
        if len(chosen) == keep:
            break
    files: dict[str, bytes] = {}
    views = []
    RESULTS.reload()
    folder = Path("/data/starts")
    folder.mkdir(parents=True, exist_ok=True)
    for rank, k in enumerate(chosen, start=1):
        s = scored[k]
        cam = look(s, w, h)
        rgb = renderer(splats, cam, background=SKY).rgb
        lab = renderer(label, cam)
        cover = lab.rgb[..., 2]
        share = np.where(cover > 1e-3, lab.rgb[..., 0] / np.maximum(cover, 1e-3), 0.0)
        soft = lv.feather(share)
        name = f"{scene}-{rank}"
        u8 = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
        mask_u8 = np.clip(np.round(soft * 255), 0, 255).astype(np.uint8)
        tint = u8.astype(np.float32)
        tint = (
            tint * (1 - 0.45 * soft[..., None]) + np.array([255, 40, 160]) * 0.45 * soft[..., None]
        )
        files[f"{name}.png"] = encode_png(u8)
        files[f"{name}-mask.png"] = encode_png(np.repeat(mask_u8[..., None], 3, axis=2))
        files[f"{name}-overlay.png"] = encode_png(np.clip(tint, 0, 255).astype(np.uint8))
        view = {
            **s,
            "start": name,
            "candidate": s["name"],
            "camera": cam.to_json(),
            "maskShare": round(float((share >= lv.PLANT_SHARE).mean()), 4),
        }
        files[f"{name}.json"] = json.dumps(view, indent=1).encode()
        views.append(view)
    for filename, data in files.items():
        (folder / filename).write_bytes(data)
    RESULTS.commit()
    # The candidates at a glance, best first, scores stamped.
    import cv2

    tiles = []
    for k in order:
        t = np.clip(np.round(thumbs[k][1] * 255), 0, 255).astype(np.uint8)[..., ::-1].copy()
        cv2.putText(t, f"{scored[k]['name']} {scored[k]['score']:.3f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)  # fmt: skip
        tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i : i + 6], axis=1) for i in range(0, len(tiles), 6)]
    ok, sheet = cv2.imencode(".jpg", np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        files[f"{scene}-candidates.jpg"] = sheet.tobytes()
    return {
        "scene": scene,
        "gaussians": len(keep_rows),
        "plantGaussians": int((weight > 0).sum()),
        "views": views,
        "candidates": [scored[k] for k in order],
        "files": files,
        "fetchSeconds": round(fetched, 1),
        "seconds": round(time.time() - started, 1),
    }


# --- arm: LTX-2.5 + Cinemagraph LoRA ------------------------------------------------------------


@app.function(
    image=ltx_image,
    gpu=ARMS["ltx"]["gpu"],
    cpu=ARMS["ltx"]["cpu"],
    memory=ARMS["ltx"]["memoryGiB"] * 1024,
    timeout=ARMS["ltx"]["timeoutS"],
    volumes={"/lv": LV_WEIGHTS, "/data": RESULTS},
    secrets=[HF_SECRET],
    single_use_containers=True,
)
def ltx(request: dict) -> dict:
    """Every start through LTX-2.5 distilled with the Cinemagraph LoRA, two stages."""
    started = time.time()
    _hf_token()
    model_dir = f"/lv/{LTX_REPO}"
    lora_dir = f"/lv/{LTX_LORA_REPO}"
    names = [s["name"] for s in request["starts"]]
    _need([f"{model_dir}/model_index.json", f"{lora_dir}/{LTX_LORA_FILE}"])
    _need([f"/data/starts/{n}.png" for n in names])
    import torch
    from diffusers import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline, LTX2Pipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )

    notes = []
    try:  # the prompt enhancer (a 12B Gemma) is not used: not loaded when the pipeline allows
        pipe = LTX2Pipeline.from_pretrained(
            model_dir, dtype=torch.bfloat16, prompt_enhancer=None, processor=None
        )
    except Exception as error:  # noqa: BLE001 - load it after all
        notes.append(f"prompt_enhancer=None refused ({type(error).__name__}); loaded whole")
        pipe = LTX2Pipeline.from_pretrained(model_dir, dtype=torch.bfloat16)
    scale = float(request.get("loraScale", LTX_LORA_SCALE))
    pipe.load_lora_weights(lora_dir, weight_name=LTX_LORA_FILE, adapter_name="cinemagraph")
    pipe.set_adapters(["cinemagraph"], [scale])
    upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        model_dir, subfolder="latent_upsampler", dtype=torch.bfloat16
    )
    pipe.to("cuda")
    upsampler.to("cuda")
    pipe.vae.enable_tiling()
    i2v = LTX2ImageToVideoPipeline(
        scheduler=pipe.scheduler,
        vae=pipe.vae,
        audio_vae=pipe.audio_vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        connectors=pipe.connectors,
        transformer=pipe.transformer,
        vocoder=pipe.vocoder,
        processor=getattr(pipe, "processor", None),
        prompt_enhancer=getattr(pipe, "prompt_enhancer", None),
        duration_head=getattr(pipe, "duration_head", None),
    )
    upsample = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=upsampler)
    load = time.time() - started
    guidance = float(request.get("guidance", 1.0))
    w, h = START_SIZE
    clips = {}
    for start in request["starts"]:
        name = start["name"]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        generator = torch.Generator("cuda").manual_seed(int(request.get("seed", 1)))
        shared = {
            "image": _start_image(name),
            "prompt": start["prompt"],
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "frame_rate": LTX_FPS,
            "guidance_scale": guidance,
            "audio_guidance_scale": 1.0,
            "stg_scale": 0.0,
            "audio_stg_scale": 0.0,
            "modality_scale": 1.0,
            "audio_modality_scale": 1.0,
            "guidance_rescale": 0.0,
            "audio_guidance_rescale": 0.0,
            "spatio_temporal_guidance_blocks": None,
            "generator": generator,
            "return_dict": False,
        }
        s1, s1_audio = i2v(
            height=h // 2,
            width=w // 2,
            num_frames=LTX_FRAMES,
            sigmas=DISTILLED_SIGMA_VALUES,
            output_type="latent",
            **shared,
        )
        up = upsample(latents=s1, output_type="latent", return_dict=False)[0]
        video, _audio = i2v(
            num_frames=LTX_FRAMES,
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            latents=up,
            audio_latents=s1_audio,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            output_type="np",
            **shared,
        )
        torch.cuda.synchronize()
        seconds = time.time() - t0
        frames = _u8(video[0])
        size = [int(frames.shape[2]), int(frames.shape[1])]
        clips[name] = _clip_entry(frames, LTX_FPS, seconds, seconds, size)
    return {
        "arm": "ltx",
        "model": ARMS["ltx"]["model"],
        "settings": {
            "diffusers": LTX_DIFFUSERS_COMMIT,
            "loraScale": scale,
            "guidance": guidance,
            "sigmas": "distilled: 8 at half size, x2 latent upsample, 3 at full size",
            "frames": LTX_FRAMES,
            "fps": LTX_FPS,
            "seed": int(request.get("seed", 1)),
            "decoder": "convolutional VAE (tiled)",
        },
        "notes": notes,
        "loadSeconds": round(load, 1),
        "clips": clips,
        "containerSeconds": round(time.time() - started, 1),
    }


# --- arm: Causal Forcing ---------------------------------------------------------------------


@app.function(
    image=cf_image,
    gpu=ARMS["causal"]["gpu"],
    cpu=ARMS["causal"]["cpu"],
    memory=ARMS["causal"]["memoryGiB"] * 1024,
    timeout=ARMS["causal"]["timeoutS"],
    volumes={"/lv": LV_WEIGHTS, "/data": RESULTS},
    single_use_containers=True,
)
def causal(request: dict) -> dict:
    """Every start through Causal Forcing (frame-wise), started from the render: the render's
    latent pre-fills the causal KV cache (`initial_latent`), then latent frames stream. With
    `contexts` [1, 3]: once from the image alone (the repository's I2V path) and once from a
    still clip of it three latent frames long (video extension)."""
    started = time.time()
    names = [s["name"] for s in request["starts"]]
    base = f"/lv/{CF_BASE}"
    _need([f"{base}/diffusion_pytorch_model.safetensors", f"/lv/{CF_REPO}/{CF_CHECKPOINT}"])
    _need([f"/data/starts/{n}.png" for n in names])
    code = Path("/opt/causal-forcing")
    (code / "wan_models").mkdir(exist_ok=True)
    link = code / "wan_models" / CF_BASE.split("/")[1]
    if not link.exists():
        link.symlink_to(base)
    os.chdir(code)
    sys.path.insert(0, str(code))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline

    torch.set_grad_enabled(False)
    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load("configs/causal_forcing_dmd_framewise.yaml"),
    )
    pipe = CausalInferencePipeline(config, device="cuda")
    state = torch.load(f"/lv/{CF_REPO}/{CF_CHECKPOINT}", map_location="cpu")
    key = "generator_ema" if "generator_ema" in state else "generator"
    weights = {
        k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in state[key].items()
    }
    missing, unexpected = pipe.generator.load_state_dict(weights, strict=False)
    if missing:  # a generator left partly at its base weights is not Causal Forcing
        raise RuntimeError(f"{len(missing)} generator keys missing, e.g. {missing[:5]}")
    del state, weights
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to("cuda")
    pipe.generator.to("cuda")
    pipe.vae.to("cuda")

    class CachedText(torch.nn.Module):
        """The prompt's embedding once: a viewer would hold it, not recompute it per stop."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner
            self.cache: dict = {}

        def forward(self, text_prompts: list[str]) -> dict:
            key = tuple(text_prompts)
            if key not in self.cache:
                self.cache[key] = self.inner(text_prompts=text_prompts)
            return self.cache[key]

    pipe.text_encoder = CachedText(pipe.text_encoder)
    load = time.time() - started
    width, height = CF_SIZE
    seed = int(request.get("seed", 1))
    contexts = [int(c) for c in request.get("contexts", [1])]

    def latent_of(image: object, frames: int) -> torch.Tensor:
        """The render as a still clip `1 + 4 (frames - 1)` pixel frames long, encoded."""
        rgb = np.asarray(image.resize((width, height)), np.float32) / 127.5 - 1.0  # type: ignore[attr-defined]
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None, :, None]  # (1, 3, 1, h, w)
        x = x.repeat(1, 1, 1 + 4 * (frames - 1), 1, 1).to("cuda", torch.bfloat16)
        return pipe.vae.encode_to_latent(x).to("cuda", torch.bfloat16)

    def run(prompt: str, initial: torch.Tensor, n: int) -> tuple[np.ndarray, float]:
        torch.manual_seed(seed)
        noise = torch.randn(
            [1, n, 16, height // 8, width // 8],
            generator=torch.Generator("cuda").manual_seed(seed),
            device="cuda",
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        t0 = time.time()
        video, _latents = pipe.inference(
            noise=noise, text_prompts=[prompt], initial_latent=initial, return_latents=True
        )
        torch.cuda.synchronize()
        seconds = time.time() - t0
        pipe.vae.model.clear_cache()
        frames = (video[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255).round()
        return np.clip(frames, 0, 255).astype(np.uint8), seconds

    clips = {}
    for start in request["starts"]:
        name, prompt = start["name"], start["prompt"]
        image = _start_image(name)
        pipe.text_encoder(text_prompts=[prompt])  # the embedding, held (not timed)
        for ctx in contexts:
            label = name if ctx == contexts[0] else f"{name}~ctx{ctx}"
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            initial = latent_of(image, ctx)
            torch.cuda.synchronize()
            encode = time.time() - t0
            # Time to first motion: encode, fill the cache, one new latent frame, decode.
            _first, first = run(prompt, initial, 1)
            frames, seconds = run(prompt, initial, CF_LATENT_FRAMES)
            # The pixel frames of the still context, beyond the render itself, are not shown.
            frames = frames[4 * (ctx - 1) :]
            entry = _clip_entry(frames, CF_FPS, encode + seconds, encode + first, [width, height])
            entry["contextLatentFrames"] = ctx
            clips[label] = entry
    return {
        "arm": "causal",
        "model": ARMS["causal"]["model"],
        "settings": {
            "code": f"{CF_CODE}@{CF_COMMIT}",
            "config": "configs/causal_forcing_dmd_framewise.yaml (4 denoising steps a frame)",
            "weights": key,
            "missingKeys": len(missing),
            "unexpectedKeys": len(unexpected),
            "latentFrames": CF_LATENT_FRAMES,
            "fps": CF_FPS,
            "size": [width, height],
            "contexts": contexts,
            "seed": seed,
            "attention": "torch SDPA (no flash-attn)",
        },
        "loadSeconds": round(load, 1),
        "clips": clips,
        "containerSeconds": round(time.time() - started, 1),
    }


# --- arm: Wan 2.1 FLF2V ----------------------------------------------------------------------


@app.function(
    image=video_image,
    gpu=ARMS["flf"]["gpu"],
    cpu=ARMS["flf"]["cpu"],
    memory=ARMS["flf"]["memoryGiB"] * 1024,
    timeout=ARMS["flf"]["timeoutS"],
    volumes={"/lv": LV_WEIGHTS, "/data": RESULTS},
    secrets=[HF_SECRET],
    single_use_containers=True,
)
def flf(request: dict) -> dict:
    """Every start through Wan 2.1 FLF2V with the render as first and last frame."""
    started = time.time()
    model_dir = f"/lv/{FLF_REPO}"
    names = [s["name"] for s in request["starts"]]
    _need([f"{model_dir}/model_index.json"] + [f"/data/starts/{n}.png" for n in names])
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    image_encoder = CLIPVisionModel.from_pretrained(
        model_dir, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_dir, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=FLF_SHIFT
    )
    pipe.to("cuda")
    load = time.time() - started
    w, h = FLF_SIZE
    seed = int(request.get("seed", 1))
    clips = {}
    for start in request["starts"]:
        name = start["name"]
        image = _start_image(name).resize((w, h))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        frames = pipe(
            image=image,
            last_image=image,
            prompt=start["prompt"],
            negative_prompt=NEGATIVE,
            height=h,
            width=w,
            num_frames=FLF_FRAMES,
            guidance_scale=FLF_GUIDANCE,
            num_inference_steps=FLF_STEPS,
            generator=torch.Generator("cuda").manual_seed(seed),
            output_type="np",
        ).frames[0]
        torch.cuda.synchronize()
        seconds = time.time() - t0
        clips[name] = _clip_entry(_u8(frames), FLF_FPS, seconds, seconds, [w, h])
    return {
        "arm": "flf",
        "model": ARMS["flf"]["model"],
        "settings": {
            "size": [w, h],
            "frames": FLF_FRAMES,
            "fps": FLF_FPS,
            "steps": FLF_STEPS,
            "guidance": FLF_GUIDANCE,
            "flowShift": FLF_SHIFT,
            "seed": seed,
        },
        "loadSeconds": round(load, 1),
        "clips": clips,
        "containerSeconds": round(time.time() - started, 1),
    }


# --- arm: Wan 2.2 TI2V-5B (the control) --------------------------------------------------------


@app.function(
    image=video_image,
    gpu=ARMS["wan"]["gpu"],
    cpu=ARMS["wan"]["cpu"],
    memory=ARMS["wan"]["memoryGiB"] * 1024,
    timeout=ARMS["wan"]["timeoutS"],
    volumes={"/weights": WEIGHTS, "/data": RESULTS},
    secrets=[HF_SECRET],
    single_use_containers=True,
)
def wan(request: dict) -> dict:
    """Every start through Wan 2.2 TI2V-5B, as `world_models.Wan.clip` runs it."""
    started = time.time()
    _hf_token()
    names = [s["name"] for s in request["starts"]]
    _need([f"/data/starts/{n}.png" for n in names])
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(
        WAN_REPO, vae=vae, torch_dtype=torch.bfloat16
    ).to("cuda")
    load = time.time() - started
    w, h = WAN_SIZE
    seed = int(request.get("seed", 1))
    clips = {}
    for start in request["starts"]:
        name = start["name"]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        frames = pipe(
            image=_start_image(name).resize((w, h)),
            prompt=start["prompt"],
            negative_prompt=NEGATIVE,
            height=h,
            width=w,
            num_frames=WAN_FRAMES,
            guidance_scale=5.0,
            num_inference_steps=50,
            generator=torch.Generator("cuda").manual_seed(seed),
            output_type="np",
        ).frames[0]
        torch.cuda.synchronize()
        seconds = time.time() - t0
        clips[name] = _clip_entry(_u8(frames), WAN_FPS, seconds, seconds, [w, h])
    return {
        "arm": "wan",
        "model": ARMS["wan"]["model"],
        "settings": {
            "size": [w, h],
            "frames": WAN_FRAMES,
            "fps": WAN_FPS,
            "steps": 50,
            "guidance": 5.0,
            "seed": seed,
        },
        "loadSeconds": round(load, 1),
        "clips": clips,
        "containerSeconds": round(time.time() - started, 1),
    }


ARM_FUNCTIONS = {"ltx": ltx, "causal": causal, "flf": flf, "wan": wan}


# --- the run --------------------------------------------------------------------------------


def _write(folder: Path, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)


@app.local_entrypoint()
def main(
    steps: str = "access,starts",
    arms: str = "wan,flf,causal,ltx",
    starts_names: str = "tree-1,tree-2,camp-1,camp-2",
    seed: int = 1,
    contexts: str = "1,3",
    budget_left: float = 0.0,
    out: str = "lv-out",
) -> None:
    """The steps, everything under `out/` with `out/summary.json` (each call's wall time and
    estimated dollars in `summary["costs"]`)."""
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    wanted_steps = [s for s in steps.split(",") if s]
    wanted_arms = [a for a in arms.split(",") if a]
    for a in wanted_arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}")
    summary: dict = {"steps": wanted_steps, "arms": wanted_arms, "costs": []}

    def dump() -> None:
        (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    if str(LOCAL_CAPTURES) not in sys.path:
        sys.path.insert(0, str(LOCAL_CAPTURES))
    from modal_calls import SpawnedCalls

    calls = SpawnedCalls()

    def cost(label: str, reservation: dict, seconds: float, extra: dict | None = None) -> None:
        dollars = seconds * rate_per_s(reservation)
        row = {"call": label, "gpu": reservation["gpu"] or "CPU", "seconds": round(seconds, 1)}
        row |= {"dollars": round(dollars, 4)} | (extra or {})
        summary["costs"].append(row)
        sys.stdout.write(f"cost {json.dumps(row)}\n")
        dump()

    def run() -> None:
        readable = {a: True for a in wanted_arms}
        if "access" in wanted_steps:
            t0 = time.time()
            report = access.remote()
            cost("access", {"gpu": "", "cpu": 0.125, "memoryGiB": 0.125}, time.time() - t0)
            cards = report.pop("cards", {})
            for repo, text in cards.items():
                (folder / "cards").mkdir(exist_ok=True)
                (folder / "cards" / f"{repo.replace('/', '__')}.md").write_text(text)
            summary["access"] = report
            sys.stdout.write(f"access: {json.dumps(report, indent=1)}\n")
            for a in wanted_arms:
                readable[a] = all(
                    str(report["repos"].get(r, "")).startswith("ok") for r in ARMS[a]["needs"]
                )
            summary["readable"] = readable
            dump()
        if "download" in wanted_steps:
            spawned = []
            for a in wanted_arms:
                if not readable[a]:
                    sys.stdout.write(f"download {a}: skipped, not readable\n")
                    continue
                spawned.append((a, time.time(), calls.spawn(f"download {a}", download, a)))
            summary["downloads"] = []
            for a, t0, call in spawned:
                try:
                    result = calls.get(call, timeout=DOWNLOAD_RESERVATION["timeoutS"] + 600)
                except Exception as error:  # noqa: BLE001 - the other arms go on
                    summary["downloads"].append({"arm": a, "error": repr(error)[:2000]})
                    cost(f"download {a} (failed)", DOWNLOAD_RESERVATION, time.time() - t0)
                    continue
                summary["downloads"].append(result)
                cost(f"download {a}", DOWNLOAD_RESERVATION, time.time() - t0, {"note": "CPU"})
        if "starts" in wanted_steps:
            spawned = [
                (s, time.time(), calls.spawn(f"starts {s}", starts, s)) for s in ("tree", "camp")
            ]
            summary["starts"] = []
            for s, t0, call in spawned:
                try:
                    result = calls.get(call, timeout=STARTS_RESERVATION["timeoutS"] + 600)
                except Exception as error:  # noqa: BLE001
                    summary["starts"].append({"scene": s, "error": repr(error)[:2000]})
                    cost(f"starts {s} (failed)", STARTS_RESERVATION, time.time() - t0)
                    continue
                _write(folder / "starts", result.pop("files"))
                summary["starts"].append(result)
                cost(f"starts {s}", STARTS_RESERVATION, time.time() - t0)
        if "arms" in wanted_steps:
            names = [n for n in starts_names.split(",") if n]
            left = budget_left
            spawned = []
            for a in wanted_arms:
                worst = ARMS[a]["timeoutS"] * rate_per_s(ARMS[a])
                if not readable[a]:
                    sys.stdout.write(f"arm {a}: skipped, not readable\n")
                    summary.setdefault("skipped", {})[a] = "not readable"
                    continue
                if worst > left:
                    sys.stdout.write(f"arm {a}: worst case ${worst:.2f} > ${left:.2f} left\n")
                    summary.setdefault("skipped", {})[a] = f"worst ${worst:.2f} > left ${left:.2f}"
                    continue
                left -= worst
                request = {
                    "starts": [{"name": n, "prompt": prompt_for(a, n)} for n in names],
                    "seed": seed,
                    "contexts": [int(c) for c in contexts.split(",") if c],
                }
                spawned.append((a, time.time(), calls.spawn(f"arm {a}", ARM_FUNCTIONS[a], request)))
                dump()
            summary["results"] = {}
            for a, t0, call in spawned:
                try:
                    result = calls.get(call, timeout=ARMS[a]["timeoutS"] + 900)
                except Exception as error:  # noqa: BLE001 - one failed arm does not stop the rest
                    summary["results"][a] = {"error": repr(error)[:4000]}
                    cost(f"arm {a} (failed)", ARMS[a], time.time() - t0)
                    continue
                for name, clip in result["clips"].items():
                    (folder / "clips" / a).mkdir(parents=True, exist_ok=True)
                    (folder / "clips" / a / f"{name}.mp4").write_bytes(clip.pop("mp4"))
                summary["results"][a] = result | {
                    "licence": ARMS[a]["licence"],
                    "prompts": {n: prompt_for(a, n) for n in names},
                }
                cost(
                    f"arm {a}",
                    ARMS[a],
                    time.time() - t0,
                    {"containerSeconds": result.get("containerSeconds")},
                )

    try:
        with calls.guard():
            run()
        problems = [f"{f['call']}: {f['detail']}" for f in calls.failures()]
        if problems:
            raise RuntimeError(
                f"{len(problems)} problem(s), everything else collected: " + "; ".join(problems)
            )
    finally:
        summary["calls"] = calls.report()
        summary["dollars"] = round(sum(r["dollars"] for r in summary["costs"]), 4)
        dump()
        sys.stdout.write(f"estimated dollars this run: {summary['dollars']}\n")

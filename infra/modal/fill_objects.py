"""The inferred fill's object round on Modal: the pumpkins' unseen undersides completed by 3D
object-completion models (`tools/captures/object_fill.py`, `object_models.py`), registered to
the real scan, the measured splats untouched.

`objects:<scan>` runs `object_fill.py run` on the scan with one method (`--methods`): the job
(gsplat renders, quality, registration, carving, grades, sheets) on an L4, the model in its own
GPU class. The scan is the asset's **current** tileset (its segmentation variants beside it);
the photos, poses and placement come from the pipeline run in the private bucket, as round 2's
`anchor:<scan>` jobs fetch them (`infra/modal/fill.py`). Each run also holds out the lowest
cameras (round 2's `leaveout:pumpkin`) and scores the held-out photos; round 2's own leave-out
layers, when given (`--compare-dir`, a fill.yml artifact), are scored alone and with the object
layer on top.

Nothing starts without `--budget-usd`, nor when the worst case is past what is left of it.

Run from the repository root (`.github/workflows/fill-objects.yml` does, on a push to
`wm-fill-objects` whose message carries an `[objects|...]` token):

    modal run infra/modal/fill_objects.py --check            # which models the token can read
    modal run infra/modal/fill_objects.py --jobs objects:pumpkin --methods trellis \\
        --budget-usd 7 --spent-usd 0.01

**Methods.** `trellis`: TRELLIS (v1) image-large, MIT, its DINOv2 encoder Apache-2.0 -- the
stand-in while SAM 3D Objects (`sam3d`) and Pixal3D / TRELLIS.2 (`pixal3d`, `trellis2`) wait on
gated Meta weights (facebook/sam-3d-objects; facebook/dinov3-vitl16-pretrain-lvd1689m), see
`object_models`.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "hexapod-fill-objects"
app = modal.App(APP_NAME)
WEIGHTS = modal.Volume.from_name("hexapod-world-model-weights", create_if_missing=True)
CAPTURES = "/root/captures"

if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    LOCAL_CAPTURES = ROOT / "tools" / "captures"
else:
    LOCAL_CAPTURES = Path(CAPTURES)

HF_SECRET = modal.Secret.from_name(os.environ.get("HEXAPOD_HF_SECRET", "huggingface"))
STORAGE_SECRET = modal.Secret.from_name("twin-object-storage")
USER_AGENT = "curl/8.5.0 (hexapod-fill-objects)"
GPU_RATES = {"L4": 0.80, "L40S": 1.95, "A100-80GB": 3.40, "H100": 3.95}

#: The scans: the asset (its current tileset is resolved through the API), the URL to fall back
#: on, the pipeline run whose photos and poses are in the private bucket, the segmentation
#: variant whose objects are completed, and a caption.
OBJECT_SCANS: dict[str, dict[str, str]] = {
    "pumpkin": {
        "asset": "695e9556-b376-48e5-ada5-1e9e7009cb21",
        "url": (
            "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs/"
            "430c1932-5b6a-47b1-bb71-bb7fa2fec86b/p46e74dc0f9df49af/package/splat/tileset.json"
        ),
        "job": "430c1932-5b6a-47b1-bb71-bb7fa2fec86b",
        "variant": "concept-first",
        "concept": "pumpkin",
        "caption": "an orange pumpkin and a red pumpkin resting on a bed of dry straw",
    },
}

# --- the access check ------------------------------------------------------------------------------

#: The access check and the weights prefetch: the Hub client alone, on a CPU.
_hub_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=0.34,<1.0"
)
access_image = _hub_image.add_local_file(
    LOCAL_CAPTURES / "object_models.py", "/root/object_models.py"
)
prefetch_image = _hub_image.env({"HF_HOME": "/weights/hf"}).add_local_file(
    LOCAL_CAPTURES / "object_models.py", "/root/object_models.py"
)


@app.function(
    image=access_image, secrets=[HF_SECRET], cpu=1.0, memory=1024, timeout=300
)
def objects_access(methods: list[str]) -> dict:
    """Whose token the secret holds and which repositories each method reads it can read
    (`object_models.access`); nothing is downloaded."""
    sys.path.insert(0, "/root")
    import object_models

    keys = sorted(k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper())
    out = object_models.access(object_models.find_token(), methods)
    out["secretKeys"] = keys
    return out


@app.function(
    image=prefetch_image,
    secrets=[HF_SECRET],
    volumes={"/weights": WEIGHTS},
    cpu=4.0,
    memory=8192,
    timeout=3600,
)
def objects_prefetch(repos: list[str]) -> dict[str, float]:
    """Each repository into the weights volume once, on a CPU, before a GPU waits on it."""
    sys.path.insert(0, "/root")
    import object_models
    from huggingface_hub import snapshot_download

    token = object_models.find_token()
    seconds = {}
    for repo in repos:
        started = time.time()
        snapshot_download(repo, token=token, max_workers=16)
        WEIGHTS.commit()
        seconds[repo] = round(time.time() - started, 1)
    return seconds


# --- the stand-in model: TRELLIS (v1) image-large on an L4 ---------------------------------------------

#: TRELLIS's own install (its setup.sh `--basic --xformers --spconv`), for gaussians only:
#: torch 2.4.0 with xformers' wheel for it, spconv, utils3d at TRELLIS's pinned commit, the
#: repository at a fixed commit with its FlexiCubes submodule; `rembg` is a stub (the masks are
#: given as alpha, so TRELLIS never removes a background), and so are `open3d` (imported by the
#: text-to-3D pipeline only) and kaolin's `check_tensor` (FlexiCubes' debug check). The rest of
#: what `trellis` imports at load time (crawled from its sources at this commit): torch,
#: torchvision, PIL, numpy, easydict, tqdm, safetensors, huggingface_hub, transformers,
#: plyfile, utils3d, spconv, xformers.
TRELLIS_COMMIT = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
UTILS3D = (
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git"
    "@9a4eb15e4021b67b12c460c7057d642626897ec8"
)
#: Two one-line definitions, each written by `echo` (single quotes outside, double inside).
REMBG_STUB = (
    'def new_session(*a, **k): raise RuntimeError("rembg is not used: masks are given as alpha")',
    'def remove(*a, **k): raise RuntimeError("rembg is not used: masks are given as alpha")',
)
#: `trellis.pipelines` imports its text-to-3D pipeline, which imports open3d (unused here).
#: FlexiCubes (the mesh decoder, built but never run here) imports kaolin's tensor check.
KAOLIN_STUB = "def check_tensor(*a, **k): return True"
#: Its class body names `o3d.geometry.TriangleMesh` in annotations, so those names exist.
OPEN3D_STUB = (
    'geometry = type("geometry", (), {"TriangleMesh": object, "VoxelGrid": object}); '
    'utility = type("utility", (), {"Vector3dVector": list})'
)
trellis_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "xformers==0.0.27.post2", index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "numpy==1.26.4",
        "spconv-cu120==2.3.6",
        "pillow",
        "imageio",
        "easydict",
        "scipy",
        "opencv-python-headless==4.10.0.84",
        "trimesh",
        "plyfile",
        "safetensors",
        "huggingface_hub>=0.26,<1",
        "tqdm",
        "moderngl",
        "transformers==4.46.3",
        UTILS3D,
    )
    .run_commands(
        "git clone https://github.com/microsoft/TRELLIS.git /opt/trellis",
        f"cd /opt/trellis && git checkout {TRELLIS_COMMIT} && git submodule update --init --recursive",
        "mkdir -p /opt/stubs/rembg /opt/stubs/open3d /opt/stubs/kaolin/utils",
        f"echo '{REMBG_STUB[0]}' > /opt/stubs/rembg/__init__.py",
        f"echo '{REMBG_STUB[1]}' >> /opt/stubs/rembg/__init__.py",
        f"echo '{OPEN3D_STUB}' > /opt/stubs/open3d/__init__.py",
        "touch /opt/stubs/kaolin/__init__.py /opt/stubs/kaolin/utils/__init__.py",
        f"echo '{KAOLIN_STUB}' > /opt/stubs/kaolin/utils/testing.py",
        "python -c \"import sys; sys.path.insert(0, '/opt/stubs'); import rembg, open3d; "
        'open3d.geometry.TriangleMesh; from kaolin.utils.testing import check_tensor"',
    )
    .env(
        {
            "PYTHONPATH": "/opt/trellis:/opt/stubs",
            "HF_HOME": "/weights/hf",
            "TORCH_HOME": "/weights/torch",
            "ATTN_BACKEND": "xformers",
            "SPARSE_ATTN_BACKEND": "xformers",
            "SPARSE_BACKEND": "spconv",
            "SPCONV_ALGO": "native",
        }
    )
    .add_local_file(LOCAL_CAPTURES / "object_models.py", "/root/object_models.py")
)
#: What the guard plans with: a TRELLIS call (one to three 518-px frames, 12 + 12 steps) on an
#: L4 (run 37550643965: 6-7 s each), its load (weights from the volume, DINOv2 through
#: torch.hub; 54-74 s), the idle tail and the cap.
GEN_GPU = {"trellis": "L4"}
GEN_CALL_S = {"trellis": 15.0}
GEN_LOAD_S = 180.0
GEN_IDLE_S = 60
GEN_CAP_S = 900
GEN_CONTAINERS = 1


def _models():  # noqa: ANN202 - object_models, where it was copied
    sys.path.insert(0, "/root")
    import object_models

    object_models.find_token()
    return object_models


@app.cls(
    image=trellis_image,
    gpu=GEN_GPU["trellis"],
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    memory=32768,
    timeout=GEN_CAP_S,
    startup_timeout=20 * 60,
    scaledown_window=GEN_IDLE_S,
    max_containers=GEN_CONTAINERS,
)
class GenTrellis:
    """TRELLIS (v1) image-large (MIT; DINOv2, Apache-2.0): the object from 1-3 masked photos
    (`object_models.generate_trellis`)."""

    @modal.enter()
    def load(self) -> None:
        import traceback

        started = time.time()
        self.error = ""
        try:
            self.om = _models()
            self.pipe = self.om.load_trellis()
            WEIGHTS.commit()  # DINOv2's first download, kept
        except Exception:  # noqa: BLE001 - every call reports it (no reload loop)
            self.error = traceback.format_exc()[-3000:]
        self.load_seconds = round(time.time() - started, 1)

    @modal.method()
    def generate(self, request: dict) -> dict:
        done = {"loadSeconds": self.load_seconds, "gpu": GEN_GPU["trellis"]}
        if self.error:
            return {
                "error": f"TRELLIS did not load:\n{self.error[-1400:]}",
                "seconds": 0.0,
                **done,
            }
        started = time.time()
        try:
            return {**self.om.generate_trellis(self.pipe, request), **done}
        except Exception as error:  # noqa: BLE001 - reported with its seconds
            import traceback

            return {
                "error": f"{error!r}\n{traceback.format_exc()[-1500:]}",
                "seconds": round(time.time() - started, 1),
                **done,
            }


GEN_CLASSES = {"GenTrellis": GenTrellis}
#: Each method's repositories to prefetch.
GEN_REPOS = {"trellis": ["microsoft/TRELLIS-image-large"]}


# --- the job --------------------------------------------------------------------------------------------

#: round 2's job image (`infra/modal/fill.py` `anchorfill_image`): gsplat on torch 2.4.1
#: (Python 3.10), LPIPS and DreamSim for the leave-out scores, boto3 for the private bucket.
GSPLAT_WHEEL = (
    "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/"
    "gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"
    "#sha256=01e1fd63dc69c9945e70158c818c3bb07fedf4aabcf020e6608d264cf27cc5dd"
)
_gsplat_base = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.4.1+cu124", index_url="https://download.pytorch.org/whl/cu124"
    )
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
        "pytest>=8",
    )
)
objects_job_image = (
    _gsplat_base.apt_install("git")
    .run_commands(
        "pip install 'torchvision==0.19.1' --index-url https://download.pytorch.org/whl/cu124",
        "pip install 'transformers==4.57.1' 'huggingface_hub>=0.34,<1.0' 'boto3' "
        "'lpips==0.1.4' 'dreamsim==0.2.1' 'open-clip-torch==2.32.0' 'timm==1.0.15' "
        "'peft==0.15.2' 'ftfy' 'regex'",
        "python -c \"import torch, numpy; assert torch.__version__.startswith('2.4.1'), "
        "torch.__version__; assert numpy.__version__.startswith('1.26'), numpy.__version__\"",
    )
    .env({"HF_HOME": "/weights/hf", "TORCH_HOME": "/weights/torch"})
    .add_local_dir(
        LOCAL_CAPTURES,
        CAPTURES,
        ignore=["**/.venv/**", "**/__pycache__/**", "**/*.pyc", "tests/**"],
    )
)
#: The job's L4: expected minutes (run 37553626539 took 9 with its fetches; round 2's
#: free-space voxels add about a minute per object and setup) and its timeout.
JOB_MIN = 18
JOB_CAP_MIN = 75
#: Objects and setups the estimate plans for: 4 objects (two large, two small), two setups
#: (every camera; the leave-out's), the frames shared in about a third of them; per object
#: and seed, all its frames together and each alone (up to 1 + 3 requests).
PLAN_OBJECTS = 4
PLAN_SETUPS = 1.7
PLAN_GROUPS = 4
WORST_FACTOR = 1.5


def _get(url: str, timeout: float) -> bytes:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(
                    f"GET {url}: HTTP {error.code} {error.reason}"
                ) from None
        time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def _fetch_scan(url: str, out: Path, extra: list[str]) -> Path:
    """The tileset, every tile (the leaves the pipeline loads, the parents its layer names
    nothing of) and `extra` files beside it."""
    import concurrent.futures

    out.mkdir(parents=True, exist_ok=True)
    base = url.rsplit("/", 1)[0]
    (out / "tileset.json").write_bytes(_get(url, 120))
    document = json.loads((out / "tileset.json").read_text(encoding="utf-8"))
    uris: list[str] = list(extra)
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        stack.extend(tile.get("children", []))
        uri = tile.get("content", {}).get("uri")
        if uri:
            uris.append(uri)

    def get(uri: str) -> None:
        (out / uri).parent.mkdir(parents=True, exist_ok=True)
        (out / uri).write_bytes(_get(f"{base}/{uri}", 600))

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(get, uris))
    return out / "tileset.json"


def _private_client():  # noqa: ANN202 - boto3's client
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["OBJECT_STORAGE_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OBJECT_STORAGE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["OBJECT_STORAGE_SECRET_KEY"],
        region_name=os.environ.get("OBJECT_STORAGE_REGION", "auto"),
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )


def _fetch_private(prefix: str, out: Path) -> int:
    """Every object under `prefix` in the private bucket into `out` (fill.py's)."""
    import concurrent.futures

    client = _private_client()
    bucket = os.environ["OBJECT_STORAGE_BUCKET"]
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        keys += [
            o["Key"] for o in page.get("Contents", []) if not o["Key"].endswith("/")
        ]

    def get(key: str) -> None:
        dest = out / (key[len(prefix) :].lstrip("/") or PurePosixPath(key).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(dest))

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        list(pool.map(get, keys))
    return len(keys)


def _tree(folder: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(folder)): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _tar(folder: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(folder, arcname="inferred")
    return buf.getvalue()


def _spawn(cls: str, method: str, request: dict) -> object:
    if cls not in GEN_CLASSES:
        raise ValueError(f"no class {cls!r}")
    return getattr(GEN_CLASSES[cls](), method).spawn(request)


def _wait(call: object) -> dict:
    """A call back within an hour (its class's cap, and the queue before it), else cancelled."""
    try:
        return call.get(timeout=3600)  # type: ignore[attr-defined]
    except TimeoutError:
        call.cancel(terminate_containers=True)  # type: ignore[attr-defined]
        raise


@app.function(
    image=objects_job_image,
    gpu="L4",
    cpu=8.0,
    memory=65536,
    timeout=JOB_CAP_MIN * 60,
    volumes={"/weights": WEIGHTS},
    secrets=[STORAGE_SECRET],
)
def run_objects(
    scan: str, method: str, options: dict, compare: dict[str, bytes]
) -> dict:
    """`object_fill.py run` on one scan with one method: the layer to publish (every camera),
    the leave-out scored (its lowest cameras held out), grades and sheets."""
    os.chdir(CAPTURES)
    sys.path.insert(0, CAPTURES)
    import attach_sidecars
    import object_fill as of

    of.BACKEND = (_spawn, _wait)
    started = time.time()
    setup = OBJECT_SCANS[scan]
    files: dict[str, bytes] = {}
    timings: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        try:
            url = attach_sidecars.resolve_asset(setup["asset"])["url"]
        except Exception as error:  # noqa: BLE001 - the known URL, and why
            url = setup["url"]
            timings["resolveError"] = repr(error)[:300]
        timings["tileset"] = url
        variant = f"variants/objects/{setup['variant']}/instances.json"
        tileset = _fetch_scan(url, root / "scan", [variant])
        timings["fetchS"] = round(time.time() - started, 1)
        run = f"runs/{setup['job']}"
        t = time.time()
        counts = {
            "poses": _fetch_private(f"{run}/pose/poses/", root / "poses"),
            "frames": _fetch_private(f"{run}/normalize/frames/", root / "frames"),
            "placement": _fetch_private(f"{run}/place/placement.json", root / "place"),
        }
        timings["privateS"] = round(time.time() - t, 1)
        timings["privateFiles"] = counts
        placement = next((root / "place").rglob("*.json"), None)
        out = root / "out"
        argv = [
            "run",
            str(tileset),
            str(out),
            "--instances",
            str(root / "scan" / variant),
        ]
        argv += ["--poses", str(root / "poses"), "--frames", str(root / "frames")]
        if placement is not None:
            argv += ["--placement", str(placement)]
        argv += [
            "--scan",
            scan,
            "--caption",
            setup["caption"],
            "--concept",
            setup["concept"],
        ]
        argv += ["--method", method, "--renderer", "gsplat"]
        argv += ["--leave-out", options.get("leave_out", "low")]
        argv += ["--seeds", options.get("seeds", "1,2")]
        for name, data in compare.items():
            path = root / "compare" / f"{name}.tar.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            argv += ["--compare", f"{name}={path}"]
        t = time.time()
        code, log = 1, ""
        try:
            code = of.main(argv)
        except (Exception, SystemExit):  # noqa: BLE001 - reported back with what it wrote
            import traceback

            log = traceback.format_exc()
            sys.stderr.write(log)
        timings["fillS"] = round(time.time() - t, 1)
        report_path = out / "report.json"
        result = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else None
        )
        if (out / "renders").exists():
            files.update({f"renders/{k}": v for k, v in _tree(out / "renders").items()})
        if (out / "debug").exists():
            # Each object's registered model and what each test removed (object_fill).
            files.update({f"debug/{k}": v for k, v in _tree(out / "debug").items()})
        if report_path.exists():
            files["report.json"] = report_path.read_bytes()
        for layer in sorted(out.glob("*/inferred/tileset.json")):
            files[f"{layer.parent.parent.name}/inferred.tar.gz"] = _tar(layer.parent)
        files["measured-tileset.json"] = tileset.read_bytes()
    timings["totalS"] = round(time.time() - started, 1)
    return {
        "kind": "objects",
        "scan": scan,
        "method": method,
        "ok": code == 0 and result is not None,
        "argv": argv,
        "result": result,
        "timings": timings,
        "files": files,
        "log": log,
    }


# --- the budget -----------------------------------------------------------------------------------------


def estimate_cost(jobs: list[str], method: str, options: dict) -> dict:
    """What the jobs should cost, and the worst case the guard holds them to. Per job: its
    model calls (objects x setups x seeds, `GEN_CALL_S` each), the model's container starts
    (load and idle tail), the job's L4. The worst case: every call and start at `WORST_FACTOR`
    times (five objects), one hung call to its cap, every job at its timeout."""
    seeds = len([s for s in str(options.get("seeds", "1,2")).split(",") if s])
    gpu = GEN_GPU[method]
    rate = GPU_RATES[gpu] / 3600
    rate_l4 = GPU_RATES["L4"] / 3600
    usd, worst = {}, {}
    for job in jobs:
        calls = PLAN_OBJECTS * PLAN_SETUPS * seeds * PLAN_GROUPS
        worst_calls = 5 * 2 * seeds * PLAN_GROUPS
        start = GEN_CONTAINERS * (GEN_LOAD_S + GEN_IDLE_S)
        usd[f"{job} {method} calls"] = calls * GEN_CALL_S[method] * rate
        usd[f"{job} {method} starts"] = start * rate
        usd[f"{job} L4"] = JOB_MIN * 60 * rate_l4
        worst[f"{job} {method} calls"] = (
            WORST_FACTOR * worst_calls * GEN_CALL_S[method] * rate
        )
        worst[f"{job} {method} starts"] = WORST_FACTOR * start * rate
        worst[f"{job} L4"] = JOB_CAP_MIN * 60 * rate_l4
        worst[f"{job} hung call"] = GEN_CAP_S * rate
    cpu = 0.05  # access check, prefetch, image builds (CPU)
    usd["cpu"] = cpu
    worst["cpu"] = 2 * cpu
    return {
        "usd": {k: round(v, 3) for k, v in usd.items()},
        "worst": {k: round(v, 3) for k, v in worst.items()},
        "totalUsd": round(sum(usd.values()), 2),
        "worstUsd": round(sum(worst.values()), 2),
    }


def actual_cost(results: list[dict], method: str) -> dict:
    """What the jobs cost from what they report: the model's call seconds plus, per container
    start (a distinct load time), its load and idle tail; each job's L4 wall time. A floor:
    Modal also bills image pulls and the CPU side."""
    rate = GPU_RATES[GEN_GPU[method]] / 3600
    seconds, loads, usd = 0.0, set(), {}
    for r in results:
        report = r.get("result") or {}
        for call in report.get("calls") or []:
            seconds += float(call.get("seconds") or 0.0)
            if call.get("loadSeconds") is not None:
                loads.add(float(call["loadSeconds"]))
        if r.get("timings", {}).get("totalS"):
            usd[f"{r['kind']}:{r['scan']} L4"] = (
                r["timings"]["totalS"] / 3600 * GPU_RATES["L4"]
            )
    if seconds or loads:
        usd[method] = (seconds + sum(loads) + GEN_IDLE_S * max(1, len(loads))) * rate
    return {
        "usd": {k: round(v, 3) for k, v in usd.items()},
        "modelSeconds": round(seconds, 1),
        "starts": sorted(loads),
        "totalUsd": round(sum(usd.values()), 2),
    }


@app.local_entrypoint()
def main(
    check: bool = False,
    jobs: str = "",
    methods: str = "",
    budget_usd: float = 0.0,
    spent_usd: float = 0.0,
    options: str = "",
    compare_dir: str = "",
    out: str = "objects-out",
) -> None:
    """`check`: which repositories each method reads the token can read (`access.json`).
    `jobs` (`objects:<scan>`, comma-separated) with each of `methods` (`+`-joined): the
    estimate (`objects-estimate.json`) first, refused past `budget_usd - spent_usd`; then
    access, prefetch and the jobs, each method's layer as `objects-<scan>-<layer>/
    inferred.tar.gz` (as publish-fill.yml takes them), the cost (`objects-cost.json`).
    `options`: `k=v;...` (`seeds`, `leave_out`). `compare_dir`: a fill.yml artifact whose
    `leaveout-<scan>-<layer>/inferred.tar.gz` layers are scored beside the object layers."""
    sys.path.insert(0, str(LOCAL_CAPTURES))
    import object_models

    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    wanted = [m.strip() for m in methods.split("+") if m.strip()]
    if check:
        result = objects_access.remote(wanted or list(object_models.REPOS))
        (folder / "access.json").write_text(
            json.dumps(result, indent=1), encoding="utf-8"
        )
        sys.stdout.write(f"access: {json.dumps(result, indent=1)}\n")
    every = [j.strip() for j in jobs.split(",") if j.strip()]
    if not every:
        return
    opts = dict(p.partition("=")[::2] for p in options.split(";") if p)
    for job in every:
        kind, _, scan = job.partition(":")
        if kind != "objects" or scan not in OBJECT_SCANS:
            raise SystemExit(f"job {job!r}: objects:<{'|'.join(OBJECT_SCANS)}>")
    for method in wanted:
        if method not in GEN_GPU:
            gates = object_models.GATES.get(method)
            raise SystemExit(
                f"method {method!r} has no GPU class here"
                + (f" (gated: accept {gates} first)" if gates else "")
            )
    for method in wanted:
        estimate = estimate_cost(every, method, opts)
        estimate.update(
            budgetUsd=budget_usd, spentUsd=spent_usd, options=opts, method=method
        )
        (folder / f"objects-estimate-{method}.json").write_text(
            json.dumps(estimate, indent=1)
        )
        sys.stdout.write(f"{method}, estimated: {json.dumps(estimate)}\n")
        if budget_usd <= 0:
            raise SystemExit(
                "the object jobs run only with --budget-usd set (the guard)"
            )
        if estimate["worstUsd"] > budget_usd - spent_usd:
            raise SystemExit(
                f"the worst case ${estimate['worstUsd']} (estimated ${estimate['totalUsd']}) is "
                f"more than the ${budget_usd - spent_usd:.2f} left of the budget: not started"
            )
        access = objects_access.remote([method])
        (folder / f"objects-access-{method}.json").write_text(
            json.dumps(access, indent=1)
        )
        if not access["methods"][method]["readable"]:
            raise SystemExit(
                f"the token cannot read what {method} needs: {access['repos']}"
            )
        seconds = objects_prefetch.remote(GEN_REPOS[method])
        sys.stdout.write(f"weights fetched: {json.dumps(seconds)}\n")
        compare: dict[str, bytes] = {}
        if compare_dir:
            for path in sorted(
                Path(compare_dir).glob("leaveout-*-anchor-*/inferred.tar.gz")
            ):
                compare[path.parent.name.split("-", 2)[2]] = path.read_bytes()
        sys.stdout.write(f"round-2 leave-out layers to compare: {sorted(compare)}\n")
        results, failed = [], []
        calls = [(job.partition(":")[2], method, opts, compare) for job in every]
        for result in run_objects.starmap(calls, return_exceptions=True):
            if isinstance(result, BaseException):
                failed.append(repr(result))
                sys.stdout.write(f"object job raised: {result!r}\n")
                continue
            results.append(result)
            base = f"objects-{result['scan']}"
            sub = folder / f"{base}-{method}"
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "log.txt").write_text(result["log"], encoding="utf-8")
            for name, data in result.get("files", {}).items():
                if name.endswith("/inferred.tar.gz"):
                    target = (
                        folder / f"{base}-{name.split('/', 1)[0]}" / "inferred.tar.gz"
                    )
                else:
                    target = sub / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            brief = {k: v for k, v in result.items() if k not in ("files", "log")}
            (sub / "result.json").write_text(
                json.dumps(brief, indent=1), encoding="utf-8"
            )
            report = result.get("result") or {}
            headline = {
                k: report.get(k)
                for k in ("silhouette", "freeSpace", "freeSpaceLeaveOut")
            }
            headline["heldOut"] = (report.get("heldOut") or {}).get("mean")
            sys.stdout.write(
                f"{base} {method}: ok={result['ok']} {json.dumps(headline)[:4000]}\n"
            )
            if not result["ok"]:
                failed.append(f"{base}-{method}")
        cost = actual_cost(results, method)
        cost["estimate"] = estimate["totalUsd"]
        (folder / f"objects-cost-{method}.json").write_text(json.dumps(cost, indent=1))
        sys.stdout.write(
            f"{method}, cost from what the jobs report: {json.dumps(cost)}\n"
        )
        if failed:
            raise SystemExit(f"failed: {', '.join(failed)}")

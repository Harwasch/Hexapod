"""Teacher B on Modal: `tools/captures/teacher_fill.py` (drop test, fill from outside) with
NVIDIA Fixer as the filler, on a GPU, for scans already published as tilesets.

docs/WORLD_MODEL_RUNBOOK.md is the order of runs; this file only puts them on Modal. Each
job runs the teacher_fill CLI in a CPU container (the renders are numpy) with the same
`tools/captures` code the CPU runs; `world_model_client.FixerFiller` calls the `Fixer`
class below, which holds the model on an L40S (`world_model_client.LOCAL_CLASSES`, so
nothing has to be deployed first). Nothing is written to any bucket: the strips, reports
and inferred tilesets come back to the caller.

**No NGC container and no secrets.** Fixer's own Dockerfile starts from NGC's
`cosmos-predict2-container:1.2`, which needs an NGC key. That container is
cosmos-predict2's environment, and cosmos-predict2's repository builds the same
environment from its `uv.lock` on a public CUDA base (its Dockerfile); `Fixer` uses that,
then Fixer's Dockerfile lines on top. The weights (`nvidia/Fixer`, not gated) download
into the shared weights volume once.

Run from the repository root (`.github/workflows/fill.yml` does):

    modal run infra/modal/fill.py --jobs drop:yard,drop:spool
    modal run infra/modal/fill.py --jobs fill:camp --distill 1500
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "hexapod-fill"
app = modal.App(APP_NAME)
WEIGHTS = modal.Volume.from_name("hexapod-world-model-weights", create_if_missing=True)
CAPTURES = "/root/captures"
YARD = "/root/yard"

if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    LOCAL_CAPTURES = ROOT / "tools" / "captures"
    LOCAL_YARD = ROOT / "data" / "tiles" / "synthetic-yard" / "splat"
else:
    LOCAL_CAPTURES, LOCAL_YARD = Path(CAPTURES), Path(YARD)

PUBLIC = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs"
#: Scans by short name: a public tileset URL, or a path in the container (the yard ships).
SCANS: dict[str, str] = {
    "yard": f"{YARD}/tileset.json",
    "spool": f"{PUBLIC}/8e1cc115-cb80-4af2-81fc-dccaf6b65891/package/splat/tileset.json",
    "pumpkin": f"{PUBLIC}/430c1932-5b6a-47b1-bb71-bb7fa2fec86b/package/splat/tileset.json",
    "camp": f"{PUBLIC}/50c25673-0940-4574-9b96-0b21362f83ca/package/splat/tileset.json",
}
#: The dropped cube's half size (m) per scan: the test's region, not a fill setting.
DROP_HALF_SIZE_M = {"yard": 0.4, "spool": 0.15, "pumpkin": 0.15, "camp": 0.4}
#: Fixer runs at 16:9 (1024x576); views are rendered at that aspect so it is not stretched.
DROP_SIZE = (640, 360)
FILL_SIZE = (1024, 576)

#: The public bucket answers Python's default user agent with 403; curl's is let through.
USER_AGENT = "curl/8.5.0 (hexapod-fill)"

# --- Fixer -----------------------------------------------------------------------------------

FIXER_REPO = "https://github.com/nv-tlabs/Fixer.git"
FIXER_COMMIT = "b39dfcaf4eeec90dc943b057ff368c16252c6c6e"
#: cosmos-predict2 at the commit that is its 1.0.9 (what Fixer's Dockerfile pip-installs).
COSMOS_REPO = "https://github.com/nvidia-cosmos/cosmos-predict2.git"
COSMOS_COMMIT = "661da4774b0ca41d082a0ecbeb47550bcf07e03f"
FIXER_RESOLUTION = 1024
MODEL_PKL = "/work/models/pretrained/pretrained_fixer.pkl"
FIXER_TIMESTEP = 250

fixer_image = (
    # cosmos-predict2's Dockerfile base; its lockfile is for Python 3.10.
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04", add_python="3.10")
    .apt_install("git", "curl", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install uv==0.8.12",
        f"git clone {COSMOS_REPO} /cosmos && git -C /cosmos checkout {COSMOS_COMMIT}",
        # The repository's locked environment (torch 2.6 cu126, transformer-engine, apex,
        # flash-attn as prebuilt wheels), into the image's own Python. `--frozen`, not
        # `--locked`: Modal's PyPI mirror as the default index reads as a stale lock.
        "cd /cosmos && UV_PROJECT_ENVIRONMENT=$(python -c 'import sys; print(sys.prefix)') "
        "uv sync --frozen --inexact --no-install-project --extra cu126",
        # Fixer's Dockerfile.cosmos, then its repository at the pinned commit.
        'pip install --no-deps "cosmos-predict2==1.0.9"',
        "pip install lpips natsort",
        f"git clone {FIXER_REPO} /work/fixer && git -C /work/fixer checkout {FIXER_COMMIT}",
    )
)


@app.cls(
    image=fixer_image,
    gpu="L40S",
    volumes={"/work/models": WEIGHTS},
    memory=49152,
    timeout=3600,
    scaledown_window=120,
)
class Fixer:
    """The same contract as `world_models.Fixer`: `{"images": [png]}` -> `{"images": [png],
    "model"}`, each output the size of its input."""

    @modal.enter()
    def load(self) -> None:
        import torch

        _snapshot("nvidia/Fixer", Path("/work/models"), "pretrained/pretrained_fixer.pkl")
        sys.path.insert(0, "/work/fixer/src")
        import inference_pretrained_model as fixer  # Fixer's own script, as a module

        torch.set_grad_enabled(False)
        self.fixer = fixer
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.width, self.height = fixer.get_resolution_size(FIXER_RESOLUTION)
        self.model = fixer.load_and_compile_model(
            model_path=MODEL_PKL,
            timestep=FIXER_TIMESTEP,
            vae_skip_connection=False,  # as the README runs it
            batch_size=1,
            device=self.device,
            dtype=self.dtype,
            compile=False,
        )
        self.model.set_eval()
        self.calls = 0
        self.seconds = 0.0
        self.load_report = _load_report(self.model, MODEL_PKL)

    @modal.method()
    def fix(self, request: dict) -> dict:
        return self._fix(request)

    def _fix(self, request: dict) -> dict:
        import torch
        from PIL import Image

        started = time.time()
        out = []
        for blob in request["images"]:
            image = Image.open(io.BytesIO(blob)).convert("RGB")
            size = image.size
            x = self.fixer.preprocess_image(
                image.resize((self.width, self.height), Image.BILINEAR), self.device, self.dtype
            )
            with torch.no_grad():
                y = self.fixer.model_inference(
                    self.model, 1, self.height, self.width, self.dtype, self.device, x=x
                )
            buffer = io.BytesIO()
            self.fixer.postprocess_output(y, size).save(buffer, format="PNG")
            out.append(buffer.getvalue())
        self.calls += len(out)
        self.seconds += time.time() - started
        return {"images": out, "model": f"nvidia/Fixer@{FIXER_COMMIT[:7]}"}

    @modal.method()
    def examples(self) -> dict[str, bytes]:
        """The repository's own example renders and what this setup makes of them: the
        check that the model is sound, apart from what our renders look like."""
        out = {}
        for path in sorted(Path("/work/fixer/examples").glob("*.png")):
            blob = path.read_bytes()
            out[f"{path.stem}-in.png"] = blob
            out[f"{path.stem}-out.png"] = self._fix({"images": [blob]})["images"][0]
        out["load.json"] = json.dumps(self.load_report, indent=1).encode()
        return out


def _load_report(model: object, path: str) -> dict:
    """How much of the checkpoint the model took: Fixer loads it with `strict=False`, so a
    key mismatch (another cosmos-predict2) would leave the base weights silently."""
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    report = {}
    for part in ("unet", "vae"):
        own = getattr(model, part).state_dict()
        theirs = checkpoint[f"state_dict_{part}"]
        same = [k for k in theirs if k in own]
        equal = sum(
            1
            for k in same
            if own[k].shape == theirs[k].shape
            and torch.equal(own[k].detach().cpu(), theirs[k].to(own[k].dtype))
        )
        report[part] = {
            "modelKeys": len(own),
            "checkpointKeys": len(theirs),
            "matched": len(same),
            "equalAfterLoad": equal,
            "missing": sorted(set(own) - set(theirs))[:8],
            "unexpected": sorted(set(theirs) - set(own))[:8],
        }
    return report


def _snapshot(repo: str, local: Path, marker: str) -> None:
    """The Hub repo in the weights volume, downloaded once."""
    if (local / marker).exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=str(local), token=os.environ.get("HF_TOKEN"))
    WEIGHTS.commit()


# --- Distill (gsplat), as world_models.Distill ----------------------------------------------

GSPLAT_WHEEL = (
    "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/"
    "gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"
    "#sha256=01e1fd63dc69c9945e70158c818c3bb07fedf4aabcf020e6608d264cf27cc5dd"
)
distill_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.4.1+cu124", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy==1.26.4", "ninja", "jaxtyping", "rich", GSPLAT_WHEEL)
    .add_local_file(LOCAL_CAPTURES / "distill_fill.py", "/root/distill_fill.py")
)


@app.cls(image=distill_image, gpu="L40S", timeout=3600, scaledown_window=60)
class Distill:
    @modal.method()
    def run(self, request: dict) -> dict:
        sys.path.insert(0, "/root")
        import distill_fill

        return distill_fill.run(request)


# --- The teacher_fill jobs ------------------------------------------------------------------

job_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # tools/captures/pyproject.toml's dependencies.
        "numpy>=1.26",
        "pillow>=10",
        "laspy[lazrs]>=2.5",
        "pyproj>=3.6",
        "scipy>=1.11",
        "opencv-python-headless>=4.10",
    )
    .add_local_dir(LOCAL_YARD, YARD)
    .add_local_dir(
        LOCAL_CAPTURES,
        CAPTURES,
        ignore=["**/.venv/**", "**/__pycache__/**", "tests/**", "**/*.pyc"],
    )
)


def _get(url: str, timeout: float) -> bytes:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"GET {url}: HTTP {error.code} {error.reason}") from None


def _fetch(url: str, out: Path) -> Path:
    """The tileset, its leaves and its view cones (what teacher_fill reads)."""
    import concurrent.futures

    if not url.startswith("https://"):
        source = Path(url)
        shutil.copytree(source.parent, out)
        return out / source.name
    out.mkdir(parents=True, exist_ok=True)
    base = url.rsplit("/", 1)[0]
    (out / "tileset.json").write_bytes(_get(url, 120))
    document = json.loads((out / "tileset.json").read_text(encoding="utf-8"))
    uris: list[str] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
        elif uri := tile.get("content", {}).get("uri"):
            uris.append(uri)
    if cones := document["root"].get("extras", {}).get("viewCones", {}).get("uri"):
        uris.append(cones)

    def get(uri: str) -> None:
        (out / uri).parent.mkdir(parents=True, exist_ok=True)
        (out / uri).write_bytes(_get(f"{base}/{uri}", 600))

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        list(pool.map(get, uris))
    return out / "tileset.json"


def _remote_classes() -> None:
    sys.path.insert(0, CAPTURES)
    import world_model_client

    world_model_client.LOCAL_CLASSES.update(Fixer=Fixer, Distill=Distill)


def _teacher_fill(argv: list[str]) -> tuple[int, dict | None, str]:
    """`teacher_fill.main(argv)` in this process; its JSON and its log."""
    import teacher_fill

    stdout, stderr = io.StringIO(), io.StringIO()
    code = 1
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = teacher_fill.main(argv)
        except Exception:  # reported back; the other jobs go on
            import traceback

            traceback.print_exc()
    text = stdout.getvalue()
    start = text.find("{")
    try:
        result = json.loads(text[start:]) if start >= 0 else None
    except json.JSONDecodeError:
        result = None
    return code, result, (text[-20000:] + stderr.getvalue()[-20000:])


def _tree(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _tar(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as out:
        out.add(folder, arcname=folder.name)
    return buffer.getvalue()


@app.function(image=job_image, cpu=8.0, memory=65536, timeout=4 * 3600)
def run_job(kind: str, scan: str, filler: str, options: dict) -> dict:
    """One `teacher_fill.py drop|fill` on one scan with one filler (`telea` or `fixer`).
    Returns its report, the strips, and for `fill` the inferred layer (a tar.gz) and the
    measured tileset.json with the layer linked."""
    _remote_classes()
    os.chdir(CAPTURES)
    started = time.time()
    spec = "telea" if filler == "telea" else "world_model_client:FixerFiller"
    files: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        tileset = _fetch(SCANS[scan], root / "scan")
        timings = {"fetchS": round(time.time() - started, 1)}
        document = json.loads(tileset.read_text(encoding="utf-8"))
        logs = []
        if "viewCones" not in document["root"].get("extras", {}):
            import view_cones

            t = time.time()
            view_cones.build_view_cones(tileset, tileset.parent, tileset)
            timings["viewConesS"] = round(time.time() - t, 1)
            logs.append("view cones backfilled from the leaves")
        save = root / "save"
        t = time.time()
        if kind == "drop":
            width, height = DROP_SIZE
            argv = [
                "drop",
                str(tileset),
                "--filler",
                spec,
                "--half-size-m",
                str(options.get("half_size_m", DROP_HALF_SIZE_M.get(scan, 0.15))),
                "--views",
                str(options.get("views", 4)),
            ]
        else:
            width, height = FILL_SIZE
            out = root / "inferred"
            argv = ["fill", str(tileset), str(out), "--filler", spec]
            argv += ["--views", str(options.get("views", 8)), "--mode", "ring"]
            if options.get("distill"):
                argv += ["--distill", str(options["distill"]), "--distill-on", "modal"]
        argv += ["--width", str(width), "--height", str(height), "--save", str(save)]
        code, result, log = _teacher_fill(argv)
        timings["teacherS"] = round(time.time() - t, 1)
        logs.append(log)
        if save.exists():
            files.update({f"strips/{k}": v for k, v in _tree(save).items()})
        if kind == "fill" and code == 0 and (root / "inferred" / "tileset.json").exists():
            import teacher_fill

            files["inferred.tar.gz"] = _tar(root / "inferred")
            teacher_fill.link_inferred(tileset, root / "inferred" / "tileset.json")
            files["measured-tileset.json"] = tileset.read_bytes()
        timings["totalS"] = round(time.time() - started, 1)
        return {
            "kind": kind,
            "scan": scan,
            "filler": filler,
            "ok": code == 0 and result is not None,
            "argv": argv,
            "result": result,
            "timings": timings,
            "files": files,
            "log": "\n".join(logs),
        }


@app.local_entrypoint()
def main(
    jobs: str = "drop:yard,drop:spool",
    fillers: str = "telea,fixer",
    views: int = 0,
    distill: int = 0,
    out: str = "fill-out",
    selftest: bool = False,
) -> None:
    """Every `kind:scan` in `jobs` with every filler, in parallel containers; each result
    under `out/<kind>-<scan>-<filler>/`, and `out/summary.json`. `selftest`: also Fixer on
    its repository's examples, under `out/selftest/`."""
    if selftest:
        folder = Path(out) / "selftest"
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in Fixer().examples.remote().items():
            (folder / name).write_bytes(data)
    calls = []
    for job in (j.strip() for j in jobs.split(",") if j.strip()):
        kind, _, scan = job.partition(":")
        if kind not in ("drop", "fill") or scan not in SCANS:
            raise SystemExit(f"job {job!r}: drop|fill:<{'|'.join(SCANS)}>")
        options: dict = {"views": views} if views else {}
        if kind == "fill" and distill:
            options["distill"] = distill
        calls += [(kind, scan, f.strip(), options) for f in fillers.split(",") if f.strip()]
    summary, failed = [], []
    for result in run_job.starmap(calls, return_exceptions=True):
        if isinstance(result, BaseException):
            failed.append(repr(result))
            sys.stdout.write(f"job raised: {result!r}\n")
            continue
        folder = Path(out) / f"{result['kind']}-{result['scan']}-{result['filler']}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "log.txt").write_text(result["log"], encoding="utf-8")
        for name, data in result.get("files", {}).items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(data)
        brief = {k: v for k, v in result.items() if k not in ("files", "log")}
        (folder / "report.json").write_text(json.dumps(brief, indent=1), encoding="utf-8")
        summary.append(brief)
        sys.stdout.write(json.dumps({k: brief[k] for k in ("kind", "scan", "filler", "ok")}))
        sys.stdout.write(" " + json.dumps(_headline(brief)) + "\n")
        if not result["ok"]:
            failed.append(folder.name)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


def _headline(brief: dict) -> dict:
    result = brief.get("result") or {}
    if brief["kind"] == "drop":
        return {"heldOut": result.get("heldOut"), "timings": brief["timings"]}
    keep = ("gaussians", "views", "meanConfidence", "perView", "distill")
    return {**{k: result.get(k) for k in keep}, "timings": brief["timings"]}

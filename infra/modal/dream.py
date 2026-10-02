"""Dream clips on Modal: Wan 2.2 and Cosmos-Predict2 video from views of published scans,
and the video teacher for materials (C2) fitted to what they generate.

docs/WORLD_MODEL_RUNBOOK.md (section 8) is what this found; this file only runs it. The
video models are the deployed `hexapod-world-models` app's `Wan` and `Cosmos` classes
(`infra/modal/world_models.py`, deployed first by `.github/workflows/dream.yml`), called the
way the teachers call them (`tools/captures/world_model_client.py`). Nothing is written to
any bucket: clips, contact sheets, stills and fitted materials come back to the caller.

* **Starts** (`starts`, an L4): the scan's leaves rasterized with gsplat
  (`splat_render.GsplatRenderer`) from eye-level candidates -- a ring of bearings at a few
  places on the ground -- scored by how much of the frame the scan covers at a useful
  depth; the best views rendered at 1280x704 over a pale sky (`teacher_materials.SKY`).
* **Clips**: every start x prompt x model, one seed each, saved as the model's mp4 with a
  contact sheet (`sheet`: frames across the clip and where it moved).
* **Materials** (`materials`, an L4 + CPUs per instance): the camp's skin for one in-place
  instance (`skin_scene.build --only`), then `teacher_materials.py world` -- the bearing it
  is seen best from, a gsplat still, Wan clips chained from each clip's last frame, tracked
  and fitted. The records are `fitted-generated`.

Run from the repository root, after `modal deploy infra/modal/world_models.py`:

    modal run infra/modal/dream.py --scans camp,pumpkin --models Wan,Cosmos
    modal run infra/modal/dream.py --scans "" --instances 225,244 --chain 3
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "hexapod-dream"
WORLD_MODELS_APP = "hexapod-world-models"
app = modal.App(APP_NAME)
CAPTURES = "/root/captures"

if modal.is_local():
    LOCAL_CAPTURES = Path(__file__).resolve().parents[2] / "tools" / "captures"
else:
    LOCAL_CAPTURES = Path(CAPTURES)

PUBLIC = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs"
SCANS: dict[str, str] = {
    "pumpkin": f"{PUBLIC}/430c1932-5b6a-47b1-bb71-bb7fa2fec86b/package/splat/tileset.json",
    "camp": f"{PUBLIC}/50c25673-0940-4574-9b96-0b21362f83ca/package/splat/tileset.json",
}
USER_AGENT = "curl/8.5.0 (hexapod-dream)"

#: Where the eye-level candidates stand (tileset local ENU, metres; the ground is found
#: under each) and how many views of each scan are kept. The camp's are on its dirt clearing
#: (instance 1) and the trail into it; the pumpkin patch is looked at from a ring around it.
EYES: dict[str, list[tuple[float, float]]] = {
    "camp": [(9.0, 18.3), (-4.3, 1.7), (0.0, 10.0)],
}
RING: dict[str, tuple[float, float, float]] = {"pumpkin": (0.0, 0.0, 4.5)}  # centre, radius
KEEP: dict[str, int] = {"camp": 2, "pumpkin": 1}
EYE_HEIGHT_M = 1.6
#: Starts are drawn without gaussians larger than this (the camp's edge floaters).
MAX_SCALE_M = 1.0
START_SIZE = (1280, 704)
FOV_DEG = 60.0

PROMPTS: dict[str, str] = {
    "gentle": (
        "A gentle breeze moves the leaves and branches of the trees and plants; they sway "
        "softly and settle. Everything else is still."
    ),
    "gusty": (
        "Strong gusty wind: the trees, bushes and plants bend and whip back and forth, their "
        "leaves and needles shaking hard with every gust, then recoil between gusts."
    ),
    "rain": (
        "Light rain is falling: fine raindrops streak down through the air, the leaves "
        "glisten and drip, and the trees and plants are almost still."
    ),
}

#: The camp's in-place plants fitted by the materials job (instances.json ids, level 0):
#: two knee-high bushes on the clearing's edge, a 4 m bush, a 3.8 m conifer, a 9.7 m tree.
CAMP_INSTANCES = (225, 244, 113, 414, 231)

GSPLAT_WHEEL = (
    "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/"
    "gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"
    "#sha256=01e1fd63dc69c9945e70158c818c3bb07fedf4aabcf020e6608d264cf27cc5dd"
)

#: `infra/modal/fill.py`'s gsplat job image (torch 2.4 cu124, gsplat's wheel, the captures'
#: dependencies), with matplotlib for the spectrum plots.
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

sheet_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy>=1.26", "opencv-python-headless>=4.10", "pillow>=10"
)


# --- fetching ------------------------------------------------------------------------------


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


def _fetch(url: str, out: Path, every: bool = False) -> Path:
    """The tileset, its leaves (`every`: its merged parents too, which the skin is bound to)
    and its instances.json."""
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
        if uri and (every or not tile.get("children")):
            uris.append(uri)
    if instances := document["root"].get("extras", {}).get("instances", {}).get("uri"):
        uris.append(instances)

    def get(uri: str) -> None:
        (out / uri).parent.mkdir(parents=True, exist_ok=True)
        (out / uri).write_bytes(_get(f"{base}/{PurePosixPath(uri)}", 600))

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(get, uris))
    return out / "tileset.json"


# --- start frames --------------------------------------------------------------------------


def _ground(positions: object, x: float, y: float) -> float:
    """The ground under (x, y): a low percentile of the scan's heights within 1.5 m."""
    import numpy as np

    p = np.asarray(positions)
    near = np.hypot(p[:, 0] - x, p[:, 1] - y) < 1.5
    if near.sum() < 50:
        near = np.hypot(p[:, 0] - x, p[:, 1] - y) < 4.0
    return float(np.percentile(p[near, 2], 10)) if near.any() else float(np.median(p[:, 2]))


def _candidates(scan: str, positions: object) -> list[dict]:
    """Eye-level cameras to choose from: `{"eye", "target", "name"}`."""
    import math

    out = []
    if scan in RING:
        cx, cy, radius = RING[scan]
        ground = _ground(positions, cx, cy)
        for bearing in range(0, 360, 45):
            b = math.radians(bearing)
            eye = (cx + radius * math.sin(b), cy + radius * math.cos(b))
            g = _ground(positions, *eye)
            out.append(
                {
                    "name": f"ring{bearing:03d}",
                    "eye": [eye[0], eye[1], g + 1.4],
                    "target": [cx, cy, ground + 0.3],
                }
            )
    for k, (x, y) in enumerate(EYES.get(scan, [])):
        g = _ground(positions, x, y)
        for bearing in range(0, 360, 45):
            b = math.radians(bearing)
            out.append(
                {
                    "name": f"eye{k}-{bearing:03d}",
                    "eye": [x, y, g + EYE_HEIGHT_M],
                    "target": [x + 10 * math.sin(b), y + 10 * math.cos(b), g + EYE_HEIGHT_M + 1.5],
                }
            )
    return out


@app.function(image=job_image, gpu="L4", cpu=8.0, memory=65536, timeout=3600)
def starts(scan: str) -> dict:
    """The best eye-level views of a scan, drawn with gsplat over a sky: PNGs, cameras and
    the scores of every candidate."""
    import numpy as np

    sys.path.insert(0, CAPTURES)
    from splat_render import Camera, GsplatRenderer, load_tileset
    from teacher_materials import SKY, to_u8
    from world_model_client import encode_png

    started = time.time()
    with tempfile.TemporaryDirectory() as work:
        tileset = _fetch(SCANS[scan], Path(work) / "scan")
        splats = load_tileset(tileset)
    total = len(splats)
    splats = splats.take(np.flatnonzero(splats.scales.max(axis=1) <= MAX_SCALE_M))
    renderer = GsplatRenderer()
    scored = []
    for c in _candidates(scan, splats.positions):
        cam = Camera.look_at(c["eye"], c["target"], fov_deg=FOV_DEG, width=320, height=176)
        frame = renderer(splats, cam)
        covered = frame.alpha > 0.5
        depth = frame.depth[covered]
        coverage = float(covered.mean())
        median = float(np.median(depth)) if depth.size else 0.0
        near = float((depth < 1.0).mean()) if depth.size else 1.0
        score = coverage * min(1.0, median / 6.0) * (1.0 - near)
        scored.append(
            {**c, "coverage": coverage, "medianDepth": median, "near": near, "score": score}
        )
    scored.sort(key=lambda s: -s["score"])
    chosen: list[dict] = []
    for s in scored:  # distinct places, or bearings at least 90 degrees apart
        if all(
            s["name"].split("-")[0] != o["name"].split("-")[0]
            or 90 <= abs(int(s["name"][-3:]) - int(o["name"][-3:])) <= 270
            for o in chosen
        ):
            chosen.append(s)
        if len(chosen) == KEEP.get(scan, 1):
            break
    files: dict[str, bytes] = {}
    views = []
    for s in chosen:
        w, h = START_SIZE
        cam = Camera.look_at(s["eye"], s["target"], fov_deg=FOV_DEG, width=w, height=h)
        rgb = renderer(splats, cam, background=SKY).rgb
        name = f"{scan}-{s['name']}"
        files[f"{name}.png"] = encode_png(to_u8(rgb))
        views.append({"name": name, "camera": cam.to_json(), **s})
    # The candidates at a glance.
    thumbs = []
    for s in scored:
        cam = Camera.look_at(s["eye"], s["target"], fov_deg=FOV_DEG, width=320, height=176)
        thumbs.append(to_u8(renderer(splats, cam, background=SKY).rgb))
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.concatenate(thumbs[i : i + 4], axis=1) for i in range(0, len(thumbs), 4)]
    files[f"{scan}-candidates.png"] = encode_png(np.concatenate(rows, axis=0))
    return {
        "scan": scan,
        "gaussians": total,
        "drawn": len(splats),
        "views": views,
        "candidates": scored,
        "files": files,
        "seconds": round(time.time() - started, 1),
    }


# --- contact sheets ------------------------------------------------------------------------


@app.function(image=sheet_image, cpu=2.0, memory=8192, timeout=600)
def sheet(clip: bytes, title: str, suffix: str = ".mp4") -> bytes:
    """A clip at a glance: eight frames across it (time stamped) and where it moved (the
    mean absolute difference from its first frame, brightest where most)."""
    import cv2
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(clip)
        f.flush()
        capture = cv2.VideoCapture(f.name)
        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(bgr)
        capture.release()
    if not frames:
        raise ValueError(f"{title}: no frames")
    h, w = frames[0].shape[:2]
    tw = 400
    th = round(h * tw / w)
    picks = np.linspace(0, len(frames) - 1, 8).round().astype(int)
    tiles = []
    for k in picks:
        t = cv2.resize(frames[k], (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(
            t, f"{k / fps:4.2f} s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        tiles.append(t)
    first = frames[0].astype(np.float32)
    motion = np.mean([np.abs(f.astype(np.float32) - first).mean(axis=2) for f in frames], axis=0)
    scale = max(float(np.percentile(motion, 99.5)), 1.0)
    heat = cv2.applyColorMap(
        np.clip(motion / scale * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    heat = cv2.resize(heat, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.putText(
        heat, "motion vs frame 0", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    blank = np.zeros_like(heat)
    grid = [tiles[0:4], tiles[4:8], [heat, blank, blank, blank]]
    body = np.concatenate([np.concatenate(r, axis=1) for r in grid], axis=0)
    band = np.full((36, body.shape[1], 3), 24, np.uint8)
    text = f"{title}  ({len(frames)} frames, {len(frames) / fps:.2f} s at {fps:g} fps)"
    cv2.putText(band, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2)
    ok, png = cv2.imencode(".png", np.concatenate([band, body], axis=0))
    if not ok:
        raise RuntimeError("png encode failed")
    return png.tobytes()


# --- materials (C2) ------------------------------------------------------------------------


def _run_main(module: str, argv: list[str]) -> tuple[int, str]:
    import importlib
    import traceback

    main = importlib.import_module(module).main
    out = io.StringIO()
    code = 1
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            code = main(argv)
        except SystemExit as exit_:
            code = int(exit_.code or 0) if not isinstance(exit_.code, str) else 1
            out.write(f"SystemExit: {exit_.code}\n")
        except Exception:  # noqa: BLE001 - reported back; the other instances go on
            traceback.print_exc()
    return code, out.getvalue()[-40000:]


@app.function(image=job_image, gpu="L4", cpu=8.0, memory=98304, timeout=4 * 3600)
def materials(scan: str, instance: int, options: dict) -> dict:
    """The skin of one instance, then `teacher_materials.py world` on it: its report,
    materials.json, still, clips, spectrum and log."""
    sys.path.insert(0, CAPTURES)
    os.chdir(CAPTURES)
    import skin_scene

    started = time.time()
    timings: dict[str, float] = {}
    files: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        tileset = _fetch(SCANS[scan], root / "scan", every=True)
        timings["fetchS"] = round(time.time() - started, 1)
        tiles = tileset.parent
        doc_path = tiles / "instances.json"
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        t = time.time()
        built = skin_scene.build(
            skin_scene.read_tiles(tiles, doc), doc["instances"], only=[instance]
        )
        skin_scene.write_skin(root / "skin", built)
        timings["skinS"] = round(time.time() - t, 1)
        save = root / "save"
        argv = [
            "world",
            str(tiles),
            str(root / "skin"),
            "--instance",
            str(instance),
            "--instances",
            str(doc_path),
            "--model",
            options.get("model", "Wan"),
            "--seeds",
            str(options.get("seeds", 1)),
            "--chain",
            str(options.get("chain", 1)),
            "--strength",
            str(options.get("strength", 0.1)),
            "--renderer",
            "gsplat",
            "--auto-bearing",
            "--materials",
            str(root / "materials.json"),
            "--report",
            str(root / "report.json"),
            "--save",
            str(save),
        ]
        if options.get("prompt"):
            argv += ["--prompt", options["prompt"]]
        t = time.time()
        code, log = _run_main("teacher_materials", argv)
        timings["worldS"] = round(time.time() - t, 1)
        for name in ("materials.json", "report.json"):
            if (root / name).exists():
                files[name] = (root / name).read_bytes()
        files["skin.json"] = (root / "skin" / "skin.json").read_bytes()
        if save.exists():
            for avi in sorted(save.rglob("clip-*.avi")):
                _to_mp4(avi)
            for p in sorted(save.rglob("*")):
                if p.is_file():
                    files[p.relative_to(save).as_posix()] = p.read_bytes()
    timings["totalS"] = round(time.time() - started, 1)
    return {
        "scan": scan,
        "instance": instance,
        "ok": code == 0,
        "argv": argv,
        "timings": timings,
        "files": files,
        "log": log,
    }


def _to_mp4(avi: Path) -> None:
    """The world command's MJPG clip as a (much smaller) MPEG-4 one beside it; the AVI goes."""
    import cv2

    capture = cv2.VideoCapture(str(avi))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    writer = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(avi.with_suffix(".mp4")), fourcc, fps, (w, h))
        writer.write(frame)
    capture.release()
    if writer is not None:
        writer.release()
        avi.unlink()


# --- the run --------------------------------------------------------------------------------


def _write(folder: Path, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)


@app.local_entrypoint()
def main(
    scans: str = "camp,pumpkin",
    prompts: str = "gentle,gusty,rain",
    models: str = "Wan,Cosmos",
    instances: str = ",".join(str(i) for i in CAMP_INSTANCES),
    chain: int = 3,
    seeds: int = 1,
    strength: float = 0.1,
    out: str = "dream-out",
) -> None:
    """Starts of every scan, a clip per start x prompt x model, contact sheets, and the
    camp's `instances` fitted (C2) from Wan clips `chain` calls long. Everything under
    `out/`, with `out/summary.json`."""
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    def dump() -> None:  # after every result: a failed run still leaves what it got
        dump()

    try:
        access = modal.Function.from_name(WORLD_MODELS_APP, "access").remote()
    except Exception as error:  # noqa: BLE001 - reported
        access = {"error": repr(error)}
    summary["access"] = access
    sys.stdout.write(f"hugging face access: {json.dumps(access)}\n")

    jobs = [int(i) for i in instances.split(",") if i.strip()]
    options = {"model": "Wan", "seeds": seeds, "chain": chain, "strength": strength}
    material_calls = [materials.spawn("camp", i, options) for i in jobs]

    names = [s.strip() for s in scans.split(",") if s.strip()]
    summary["starts"] = []
    views: list[tuple[str, bytes]] = []
    for result in starts.map(names, return_exceptions=True):
        if isinstance(result, BaseException):
            summary["starts"].append({"error": repr(result)})
            sys.stdout.write(f"starts raised: {result!r}\n")
            continue
        _write(folder / "starts", result["files"])
        brief = {k: v for k, v in result.items() if k != "files"}
        summary["starts"].append(brief)
        for v in result["views"]:
            views.append((v["name"], result["files"][f"{v['name']}.png"]))
        sys.stdout.write(f"starts {result['scan']}: {[v['name'] for v in result['views']]}\n")
        dump()

    classes = {m: modal.Cls.from_name(WORLD_MODELS_APP, m)() for m in models.split(",") if m}
    calls = []
    for name, png in views:
        for p in (p.strip() for p in prompts.split(",") if p.strip()):
            for model, cls in classes.items():
                for seed in range(1, seeds + 1):
                    request = {"image": png, "prompt": PROMPTS[p], "seed": seed}
                    calls.append(
                        (f"{name}-{p}-{model.lower()}-s{seed}", model, cls.clip.spawn(request))
                    )
    summary["clips"] = []
    sheets = []
    for label, model, call in calls:
        started = time.time()
        try:
            response = call.get()
        except Exception as error:  # noqa: BLE001 - one failed model does not stop the rest
            summary["clips"].append({"clip": label, "model": model, "error": repr(error)[:2000]})
            sys.stdout.write(f"clip {label}: {error!r}\n"[:2000])
            continue
        (folder / "clips").mkdir(exist_ok=True)
        (folder / "clips" / f"{label}.mp4").write_bytes(response["mp4"])
        entry = {k: v for k, v in response.items() if k != "mp4"}
        entry |= {"clip": label, "waitedS": round(time.time() - started, 1)}
        summary["clips"].append(entry)
        sheets.append((label, sheet.spawn(response["mp4"], f"{label} ({response.get('model')})")))
        sys.stdout.write(f"clip {label}: {json.dumps(entry)}\n")
        dump()
    for label, call in sheets:
        try:
            (folder / "sheets").mkdir(exist_ok=True)
            (folder / "sheets" / f"{label}.png").write_bytes(call.get())
        except Exception as error:  # noqa: BLE001
            sys.stdout.write(f"sheet {label}: {error!r}\n")

    summary["materials"] = []
    merged: dict[int, dict] = {}
    for call in material_calls:
        try:
            result = call.get()
        except Exception as error:  # noqa: BLE001
            summary["materials"].append({"error": repr(error)[:2000]})
            sys.stdout.write(f"materials raised: {error!r}\n")
            continue
        sub = folder / "materials" / f"instance-{result['instance']}"
        sub.mkdir(parents=True, exist_ok=True)
        _write(sub, result["files"])
        (sub / "log.txt").write_text(result["log"], encoding="utf-8")
        report = json.loads(result["files"].get("report.json", b"[]") or b"[]")
        brief = {k: v for k, v in result.items() if k not in ("files", "log")} | {"report": report}
        summary["materials"].append(brief)
        dump()
        if "materials.json" in result["files"]:
            for record in json.loads(result["files"]["materials.json"])["materials"]:
                merged[int(record["instance"])] = record
        for clip in sorted(sub.rglob("clip-*.mp4")):
            sheets_dir = folder / "sheets"
            sheets_dir.mkdir(exist_ok=True)
            label = f"camp-c2-instance-{result['instance']}-{clip.stem}"
            try:
                png = sheet.remote(clip.read_bytes(), label, clip.suffix)
                (sheets_dir / f"{label}.png").write_bytes(png)
            except Exception as error:  # noqa: BLE001
                sys.stdout.write(f"sheet {label}: {error!r}\n")
        sys.stdout.write(
            f"materials {result['instance']}: ok={result['ok']} {json.dumps(report)[:3000]}\n"
        )
    (folder / "materials.json").write_text(
        json.dumps(
            {
                "format": "hexapod.materials",
                "version": 1,
                "materials": [merged[k] for k in sorted(merged)],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    dump()

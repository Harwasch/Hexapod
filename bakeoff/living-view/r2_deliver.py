"""Round 2's files for the page (the coordinator builds it), from the runs' artifacts.

    r2_deliver.py OUT STARTS RUN [RUN ...]

STARTS is round 1's starts folder (`{start}.png`, `{start}-mask.png`). Each RUN is an unpacked artifact of `.github/workflows/living-view.yml` (`summary.json` with
`r2` results, `*-summary.json` of the world models, `upscale-summary.json`; `clips/<arm>/
<start>.mp4`, `upscaled/<label>/<arm>/<start>.mp4`); later runs replace earlier ones. Under OUT:

    clips/{start}-{arm}-a.mp4      the model's pixels at 1280x704 (letterboxed if the aspect
                                   differs), H.264 High, yuv420p, +faststart
    clips/{start}-{arm}-u2k.mp4    FlashVSR at 2560x1408 (as the upscaler wrote it)
    clips/{start}-{arm}-s2k.mp4    SeedVR2-3B at 2560x1408
    clips/{start}-{arm}-u4k|s4k.mp4  3840x2112
    numbers.json                   per "{start}|{arm}": timings, model size, settings, prompt,
                                   motion and drift numbers, upscale times; per arm: a summary
                                   and the read by eye (`EYE` below)

The motion numbers, on the model's own frames (at most 640 px wide for the flow): plant motion
(DIS flow against frame 0, inside the plant mask, render px at 1280 wide, mean and p95),
non-plant motion on textured pixels and the camera creep (a homography fitted there), and the
drift: PSNR of the non-plant pixels against our render at the first and the last frame.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "captures"))

import living_view as lv

GRID = (1280, 704)
FLOW_WIDTH = 640
LIMITS = {"a": 8, "u2k": 8, "s2k": 8, "u4k": 14, "s4k": 14}  # MB
KINDS = {
    "flashvsr-2560x1408": "u2k",
    "seedvr2-2560x1408": "s2k",
    "flashvsr-3840x2112": "u4k",
    "seedvr2-3840x2112": "s4k",
}
#: Round-1 LTX clips (scene prompts) are re-upscaled as "ltx-r1".
ARM_ALIASES = {"ltx": "ltx-r1"}
LICENCES = {
    "ltx": "LTX-2.x Community License (commercial use free under $10M revenue)",
    "waypoint": "weights Apache-2.0; inference code (HF repo .py) GPL-3.0",
    "mg3": "Apache-2.0 (weights and code)",
    "yume": "Apache-2.0 (weights and code)",
}
#: Filled in by hand after looking at the clips frame by frame.
EYE: dict[str, str] = {}


def licence(arm: str) -> str:
    return LICENCES[arm.split("-")[0]]


def ffmpeg_encode(frames: list[np.ndarray], fps: float, out: Path, limit_mb: float) -> int:
    """H.264 High, yuv420p, +faststart, from CRF 18 up until it fits."""
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    for crf in (18, 20, 22, 24, 26, 28, 30, 32):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "slow", "-profile:v", "high", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out),
        ]  # fmt: skip
        subprocess.run(cmd, input=np.stack(frames).astype(np.uint8).tobytes(), check=True)
        if out.stat().st_size <= limit_mb * 1024 * 1024:
            return crf
    return crf


def letterbox(frames: list[np.ndarray], width: int, height: int) -> list[np.ndarray]:
    import cv2

    h, w = frames[0].shape[:2]
    k = min(width / w, height / h)
    fw, fh = min(width, round(w * k / 2) * 2), min(height, round(h * k / 2) * 2)
    out = []
    for f in frames:
        small = cv2.resize(f, (fw, fh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        if (fw, fh) == (width, height):
            out.append(small)
            continue
        canvas = np.zeros((height, width, 3), np.uint8)
        top, left = (height - fh) // 2, (width - fw) // 2
        canvas[top : top + fh, left : left + fw] = small
        out.append(canvas)
    return out


def render_for(start: Path, arm: str, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Our render and its soft plant mask in the clip's geometry (Waypoint saw a 2:1 crop)."""
    import cv2

    render = lv.read_png(start.with_suffix(".png"))
    mask = lv.read_mask(start.parent / f"{start.name}-mask.png")
    if arm.startswith("waypoint"):
        h, w = render.shape[:2]
        crop = w // 2
        top = (h - crop) // 2
        render, mask = render[top : top + crop], mask[top : top + crop]
    w, h = size
    return (
        cv2.resize(render, (w, h), interpolation=cv2.INTER_AREA),
        cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA),
    )


def metrics(frames: list[np.ndarray], render: np.ndarray, mask: np.ndarray) -> dict:
    import cv2

    h, w = frames[0].shape[:2]
    k = min(1.0, FLOW_WIDTH / w)
    fw, fh = round(w * k), round(h * k)
    small = [cv2.resize(f, (fw, fh), interpolation=cv2.INTER_AREA) for f in frames]
    m_small = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    background = (m_small < lv.NOT_PLANT_SHARE) & (grad / 4.0 >= lv.MIN_GRADIENT)
    step = max(1, len(small) // 24)  # about 24 frames sampled across the clip
    flows, creep = [], 0.0
    scale = (GRID[0] / fw, GRID[0] / fw)  # in render px at 1280 wide
    for f in small[step::step]:
        flow = lv.dis_flow(f, small[0])
        H, n = lv.global_motion(flow, background)
        if n:
            cam = lv.homography_flow(H, fh, fw)
            creep = max(creep, float(np.hypot(cam[..., 0], cam[..., 1]).mean() * scale[0]))
        flows.append(flow)
    inside_mean, inside_p95, out_mean, out_p95 = lv.motion_stats(
        flows, m_small, scale, outside=background
    )
    not_plant = mask < lv.NOT_PLANT_SHARE

    def psnr(frame: np.ndarray) -> float | None:
        if not not_plant.any():
            return None
        err = ((frame.astype(np.float32) - render.astype(np.float32)) ** 2)[not_plant].mean()
        return round(float(10 * np.log10(255.0**2 / max(err, 1e-6))), 2)

    return {
        "plantMotionMeanPx": round(inside_mean, 2),
        "plantMotionP95Px": round(inside_p95, 2),
        "nonPlantMotionMeanPx": round(out_mean, 2) if background.sum() >= 200 else None,
        "nonPlantMotionP95Px": round(out_p95, 2) if background.sum() >= 200 else None,
        "cameraCreepPx": round(creep, 2) if background.sum() >= 200 else None,
        "driftPsnrFirstDb": psnr(frames[0]),
        "driftPsnrLastDb": psnr(frames[-1]),
        "flowSize": [fw, fh],
    }


def load(runs: list[Path]) -> dict:
    """clips[(arm, start)] -> {"path", "entry", "gpu", "dollars"...}; upscales[(kind, arm,
    start)] -> {"path", "entry"}."""
    clips: dict = {}
    ups: dict = {}
    calls: list = []
    for run in runs:
        for summary_path in sorted(run.glob("*summary.json")):
            summary = json.loads(summary_path.read_text())
            calls += [dict(c, run=run.name) for c in summary.get("costs", [])]
            for label, result in (summary.get("r2") or {}).items():
                if "clips" not in result:
                    continue
                for key, entry in result["clips"].items():
                    arm, start = key.split("/")
                    clips[(arm, start)] = {
                        "path": run / "clips" / arm / f"{start}.mp4",
                        "entry": entry,
                        "call": label,
                        "gpu": result.get("gpu"),
                        "settings": result.get("settings", {}),
                        "loadSeconds": result.get("loadSeconds"),
                    }
            world = summary.get("run")
            if world and "clips" in world:
                for key, entry in world["clips"].items():
                    arm, start = key.split("/")
                    clips[(arm, start)] = {
                        "path": run / "clips" / arm / f"{start}.mp4",
                        "entry": entry,
                        "call": f"run {arm.split('-')[0]}",
                        "gpu": world.get("gpu"),
                        "settings": {k: v for k, v in world.items() if k != "clips"},
                        "loadSeconds": world.get("loadSeconds"),
                    }
            for label, result in (summary.get("upscaled") or {}).items():
                kind = KINDS.get(label)
                if not kind:
                    continue
                for key, entry in result.get("clips", {}).items():
                    arm, start = key.split("/")
                    ups[(kind, ARM_ALIASES.get(arm, arm), start)] = {
                        "path": run / "upscaled" / label / arm / f"{start}.mp4",
                        "entry": entry,
                        "attention": result.get("attention"),
                        "label": label,
                    }
    return {"clips": clips, "ups": ups, "calls": calls}


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    starts_dir = Path(argv[1])
    runs = [Path(a) for a in argv[2:]]
    data = load(runs)
    numbers: dict = {}
    (out / "clips").mkdir(parents=True, exist_ok=True)
    for (arm, start), clip in sorted(data["clips"].items()):
        frames, fps = lv.read_clip(clip["path"])
        size = (frames[0].shape[1], frames[0].shape[0])
        render, mask = render_for(starts_dir / start, arm, size)
        row = {k: v for k, v in clip["entry"].items() if k != "mp4"}
        row |= metrics(frames, render, mask)
        row |= {"gpu": clip["gpu"], "licence": licence(arm), "settings": clip["settings"]}
        row["loadSeconds"] = clip["loadSeconds"]
        grid = letterbox(frames, *GRID)
        row["crf"] = {"a": ffmpeg_encode(grid, fps, out / "clips" / f"{start}-{arm}-a.mp4", 8)}
        numbers[f"{start}|{arm}"] = row
        print(start, arm, json.dumps({k: row[k] for k in row if "Px" in k or "Psnr" in k}))
    for (kind, arm, start), up in sorted(data["ups"].items()):
        target = out / "clips" / f"{start}-{arm}-{kind}.mp4"
        shutil.copyfile(up["path"], target)
        row = numbers.setdefault(f"{start}|{arm}", {"licence": licence(arm)})
        entry = {k: v for k, v in up["entry"].items() if k != "mp4"}
        row.setdefault("upscale", {})[kind] = entry | {
            "attention": up.get("attention"),
            "bytes": target.stat().st_size,
        }
    (out / "numbers.json").write_text(json.dumps({"clips": numbers, "eye": EYE}, indent=1))
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    big = [p.name for p in (out / "clips").glob("*.mp4") if p.stat().st_size > 14 * 1024 * 1024]
    print(f"{out}: {total / 1e6:.1f} MB, files over 14 MB: {big}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

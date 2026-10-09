"""Round 2's files for the page (the coordinator builds it), from the runs' artifacts.

    r2_deliver.py OUT STARTS RUN [RUN ...]
    r2_deliver.py --summary OUT/numbers.json     (the per-arm summary and notes again)

OUT is the delivery folder (`.../living-view-bakeoff/r2`); STARTS is round 1's starts folder
(`{start}.png`, `{start}-mask.png`). Each RUN is an unpacked artifact of
`.github/workflows/living-view.yml` (`summary.json` with `r2` results, `*-summary.json` of the
world models, `upscale-summary.json`; `clips/<arm>/<start>.mp4`,
`upscaled/<label>/<arm>/<start>.mp4`); a later run replaces an earlier one's clip.

    clips/{start}-{arm}-a.mp4      the model's pixels at 1280x704 (letterboxed if the aspect
                                   differs), H.264 High, yuv420p, +faststart
    clips/{start}-{arm}-u2k.mp4    FlashVSR at 2560x1408 (as the upscaler wrote it)
    clips/{start}-{arm}-s2k.mp4    SeedVR2-3B at 2560x1408
    clips/{start}-{arm}-u4k|s4k.mp4  3840x2112
    numbers.json                   per "{start}|{arm}": timings, dollars, model size, settings,
                                   prompt, motion and drift numbers, upscale times, where each
                                   file went; per arm: a summary and the read by eye (`EYE`)

OUT holds at most `LIMIT_MB` (stills included): the (arm, kind) groups go in `TIERS` order,
each group (its starts) whole or not at all; whatever does not fit goes to the sibling folder
`r2-more/` under the same names.

The winner's base clip (ltx-p1) is delivered as ltx-base; round 1's LTX clip, re-upscaled, as
ltx-r1. Clips of an `-h100` arm are timings only (`gpuCompare`).

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
LIMIT_MB = 150
KINDS = {
    "flashvsr-2560x1408": "u2k",
    "seedvr2-2560x1408": "s2k",
    "flashvsr-3840x2112": "u4k",
    "seedvr2-3840x2112": "s4k",
}
#: Delivered under another arm name: the winner's base clip, round 1's LTX clip.
DELIVER_AS = {"ltx-p1": "ltx-base", "ltx": "ltx-r1"}
STARTS = ("tree-1", "tree-2", "camp-1", "camp-2")
#: What goes in OUT first: the brief's priority. Part 1 a-b with FlashVSR 2560, then the prompt
#: variants, then Part 2 idle, then Part 1 c-d. Next come the upscales of the loop and of round
#: 1's LTX, SeedVR2, the other upscales and the pans. In practice OUT ends after Part 1 c-d's
#: grid copies, and everything later goes to r2-more/.
TIERS = [
    ("ltx-base", "a"),
    ("ltx-base", "u2k"),
    ("ltx-s1", "a"),
    ("ltx-s1", "u2k"),
    ("ltx-p2", "a"),
    ("ltx-p3", "a"),
    ("mg3-idle", "a"),
    ("waypoint-idle", "a"),
    ("yume-idle", "a"),
    ("ltx-loop", "a"),
    ("ltx-chunk", "a"),
    ("ltx-loop", "u2k"),
    ("ltx-r1", "u2k"),
    ("ltx-p5", "a"),
    ("ltx-base", "s2k"),
    ("ltx-r1", "s2k"),
    ("ltx-chunk", "u2k"),
    ("mg3-idle", "u2k"),
    ("yume-idle", "u2k"),
    ("waypoint-idle", "u2k"),
    ("mg3-pan", "a"),
    ("waypoint-pan", "a"),
    ("yume-pan", "a"),
    ("ltx-p4", "a"),
    ("ltx-p5", "u2k"),
    ("ltx-p2", "u2k"),
    ("ltx-p3", "u2k"),
    ("ltx-p4", "u2k"),
]
#: The grid copies (1280x704) are previews: at most 3 MB each (CRF 18 up), which only touches
#: the 60 fps Waypoint clips and Matrix-Game's.
GRID_MB = 3
LICENCES = {
    "ltx": "LTX-2.x Community License (commercial use free under $10M revenue)",
    "waypoint": "weights Apache-2.0; inference code (HF repo .py) GPL-3.0",
    "mg3": "Apache-2.0 (weights and code)",
    "yume": "Apache-2.0 (weights and code)",
}
#: Filled in by hand after looking at the clips frame by frame (contact sheets of frames 0,
#: 1/3, 2/3 and the end with the mean change, and full-size crops around seams and wraps).
EYE: dict[str, str] = {
    "ltx-base": (
        "The researched generic prompt (ltx-p1) at round 1's recipe. It looks like round 1's "
        "scene-prompted clip. The near foliage still swings visibly (20-57 px p95), while trunks, "
        "the sign, the path and the sky stay put (creep at most 0.7 px). Fast fronds smear (the "
        "model's own motion blur), the rest is sharp. FlashVSR at 2560 sharpens the still parts "
        "well and keeps the smear. SeedVR2 at 2560 exists for camp-2, tree-1 and camp-1; "
        "tree-2 ran out of memory."
    ),
    "ltx-s1": (
        "Stage 1 alone at 640x352 runs faster than real time (3.7 s for 4 s of video) but is "
        "soft: the big fern motion on camp-1 turns to mush. The still parts hold. FlashVSR's 4x "
        "adds invented detail, and it stays visibly softer than ltx-base."
    ),
    "ltx-chunk": (
        "Two 2 s chunks, the second continued from the first's last 9 frames: first motion "
        "after 6.1 s instead of 10.7 s. The seam (frame 49) is a small visible step: texture "
        "and brightness shift, and the motion jumps 1.6-2x the median. camp-2's camera creeps "
        "5 px across it. Each chunk sways less than one 4 s clip."
    ),
    "ltx-loop": (
        "The render as both end keyframes. This is the gentlest arm (2-4 px p95): the leaves "
        "breathe rather than swing, and the last frame wraps back onto the first with no "
        "visible jump. The closest to the owner's 'slight', at the same 10.7 s per 4 s clip."
    ),
    "ltx-p2": (
        "The coordinator's simple prompt. At a glance it is the same as ltx-p1: a little less "
        "swing on camp-2's near conifer, a little more on the trees and camp-1's ferns, and "
        "more creep on camp-2 (2.1 px against 0.7)."
    ),
    "ltx-p3": (
        "Auto-captioned by the pipeline's own enhancer: no better and slower. The captions "
        "misdescribe views (tree-1's crown as 'high-angle, looking down'). camp-1's left tree "
        "stirs, and tree-1's sky drifts 9.5 dB by the end."
    ),
    "ltx-p4": (
        "ltx-p1 at LoRA 0.6. More motion everywhere, and camp-2's camera creeps 8 px. Worse."
    ),
    "ltx-p5": (
        "ltx-p1 at LoRA 1.4. About a quarter less swing than ltx-p1 (mean p95 24 against "
        "32 px) with the same stillness elsewhere; otherwise it looks the same. The better of "
        "the two LoRA strengths tried."
    ),
    "mg3-idle": (
        "Frozen. With no action the plants do not move at all (0.5 px p95); the frame is only "
        "re-rendered through its VAE (a little sharper, the trees' colour shifted). A still, "
        "not a living view. 0.23x real time on an H100 with PyTorch attention."
    ),
    "mg3-pan": (
        "A slow yaw (0.45 degrees a frame) with consistent geometry for most of the clip. The "
        "plants still do not sway, and new content smears in at the edge."
    ),
    "waypoint-idle": (
        "Real time on an H100 (first frame 0.4-1.0 s, 21-47 fps, after a 5-minute compile), "
        "but the world wanders. tree-1 holds (frozen). The camp views switch to a darker, "
        "flatter look of its own from the first generated frame, then creep 6-7 px. "
        "tree-2 turns into a different scene within 4 s: a yellowed tree, scattered leaves and "
        "a wall. It only sees a 2:1 crop at 1024x512, and its 2560 upscale (x3) stays soft. "
        "Two of its four upscales (camp-1, tree-2) did not fit in the last run's wall."
    ),
    "waypoint-pan": (
        "Mouse x 0.1 is a fast pan, and it invents a blank grey wall to the right of camp-1. "
        "Not usable."
    ),
    "yume-idle": (
        "Not idle. The camera drifts (16-20 px), the whole frame is re-rendered and smeared, "
        "the plants churn (27-71 px p95), and by the end it sits 15-20 dB away from the "
        "render. 0.13x real time."
    ),
    "yume-pan": (
        "Pans right and invents plausible, smeary foliage. The camera control works; the "
        "stillness does not."
    ),
    "ltx-r1": (
        "Round 1's LTX clip (scene prompts), upscaled again to 2560 with FlashVSR's real "
        "kernels. SeedVR2 at 2560 exists for camp-2 and camp-1; tree-1 ran out of memory and "
        "tree-2 was past the wall."
    ),
    "upscale-flashvsr": (
        "With the real Block-Sparse-Attention kernels: 4.5 fps at 2560x1408 on an A100 (round "
        "1's PyTorch fallback did 0.6 fps). Crisp, natural detail on the still parts (leaves, "
        "bark, the sign). Motion blur from the model stays blur. The better upscaler. At "
        "3840x2112 (x3) a 97-frame clip runs out of the A100's 80 GB (2560 peaks at 37 GB), so "
        "there are no 4K clips: that needs the clip in temporal chunks, or the kernels built "
        "for an H200."
    ),
    "upscale-seedvr2": (
        "At 2560x1408 the foliage turns waxy and painted: smooth plastic leaf shapes and lifted "
        "blacks, clearly worse than FlashVSR. 1.1 fps on an H200, peaking at 113-124 GB, so "
        "3840x2112 is out of reach on one GPU. Chunked clips (33 frames, 8 cross-faded) show no "
        "visible seams."
    ),
}


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
            shift = float(np.hypot(cam[..., 0], cam[..., 1]).mean() * scale[0])
            if np.isfinite(shift) and shift < 2 * GRID[0]:  # a degenerate fit is no camera
                creep = max(creep, shift)
        flows.append(flow)
    inside_mean, inside_p95, out_mean, out_p95 = lv.motion_stats(
        flows, m_small, scale, outside=background
    )
    not_plant = mask < lv.NOT_PLANT_SHARE
    textured = background.sum() >= 200

    def psnr(frame: np.ndarray) -> float | None:
        if not not_plant.any():
            return None
        err = ((frame.astype(np.float32) - render.astype(np.float32)) ** 2)[not_plant].mean()
        return round(float(10 * np.log10(255.0**2 / max(err, 1e-6))), 2)

    return {
        "plantMotionMeanPx": round(inside_mean, 2),
        "plantMotionP95Px": round(inside_p95, 2),
        "nonPlantMotionMeanPx": round(out_mean, 2) if textured else None,
        "nonPlantMotionP95Px": round(out_p95, 2) if textured else None,
        "cameraCreepPx": round(creep, 2) if textured else None,
        "driftPsnrFirstDb": psnr(frames[0]),
        "driftPsnrLastDb": psnr(frames[-1]),
        "flowSize": [fw, fh],
    }


def load(runs: list[Path]) -> dict:
    """clips[(arm, start)] -> {"path", "entry", "gpu", "rate"...}; ups[(kind, arm, start)]."""
    clips: dict = {}
    ups: dict = {}
    for run in runs:
        for summary_path in sorted(run.glob("*summary.json")):
            summary = json.loads(summary_path.read_text())
            rates = {
                c["call"]: c["dollars"] / c["seconds"]
                for c in summary.get("costs", [])
                if c.get("seconds")
            }
            calls = {c["call"]: c for c in summary.get("costs", [])}
            for label, result in (summary.get("r2") or {}).items():
                for key, entry in result.get("clips", {}).items():
                    arm, start = key.split("/")
                    clips[(arm, start)] = {
                        "path": run / "clips" / arm / f"{start}.mp4",
                        "entry": entry,
                        "call": label,
                        "run": run.name,
                        "callDollars": calls.get(label, {}).get("dollars"),
                        "clipsInCall": len(result["clips"]),
                        "rate": rates.get(label),
                        "gpu": result.get("gpu"),
                        "settings": result.get("settings", {}),
                        "loadSeconds": result.get("loadSeconds"),
                    }
            world = summary.get("run")
            if world and "clips" in world:
                label = next((c for c in calls if c.startswith("run ")), None)
                for key, entry in world["clips"].items():
                    arm, start = key.split("/")
                    clips[(arm, start)] = {
                        "path": run / "clips" / arm / f"{start}.mp4",
                        "entry": entry,
                        "call": label,
                        "run": run.name,
                        "callDollars": calls.get(label, {}).get("dollars"),
                        "clipsInCall": len(world["clips"]),
                        "rate": rates.get(label),
                        "gpu": world.get("gpu"),
                        "settings": {k: v for k, v in world.items() if k != "clips"},
                        "loadSeconds": world.get("loadSeconds"),
                    }
            for label, result in (summary.get("upscaled") or {}).items():
                kind = KINDS.get(label)
                if not kind:
                    continue
                call = f"upscale {result.get('upscaler', label.split('-')[0])}"
                for key, entry in result.get("clips", {}).items():
                    arm, start = key.split("/")
                    ups[(kind, DELIVER_AS.get(arm, arm), start)] = {
                        "path": run / "upscaled" / label / arm / f"{start}.mp4",
                        "entry": entry,
                        "attention": result.get("attention"),
                        "label": label,
                        "run": run.name,
                        "rate": rates.get(call),
                        "loadSeconds": result.get("loadSeconds"),
                    }
    return {"clips": clips, "ups": ups}


def video_seconds(entry: dict) -> float | None:
    if entry.get("videoSeconds"):
        return entry["videoSeconds"]
    if entry.get("frames") and entry.get("fps"):
        return round(entry["frames"] / entry["fps"], 3)
    return None


def place(more: Path, out: Path, files: dict, reserved: float) -> dict:
    """Move whole (arm, kind) groups from `more` into `out` in TIERS order while they fit."""
    used = reserved
    where: dict = {}
    order = TIERS + sorted(k for k in files if k not in TIERS)
    for group in order:
        names = files.get(group, [])
        if not names:
            continue
        size = sum((more / "clips" / n).stat().st_size for n in names) / 1e6
        target = out if used + size <= LIMIT_MB else more
        if target is out:
            used += size
            for n in names:
                shutil.move(str(more / "clips" / n), str(out / "clips" / n))
        for n in names:
            where[n] = f"{target.name}/clips/{n}"
    return {"where": where, "usedMB": round(used, 1)}


def mean(values: list) -> float | None:
    values = [v for v in values if isinstance(v, int | float)]
    return round(sum(values) / len(values), 3) if values else None


def real_time(row: dict) -> float | None:
    """Seconds of video per second of generation (Waypoint: its sustained rate over 60 fps)."""
    if row.get("realTimeFactor") is not None:
        return row["realTimeFactor"]
    if row.get("sustainedFps") and row.get("fps"):
        return round(row["sustainedFps"] / row["fps"], 3)
    return None


def summarise(numbers: dict) -> dict:
    """Per arm: what it made, how fast, how it moved and drifted, its upscales, the read by eye."""
    arms: dict = {}
    for key, row in numbers.items():
        if "sameAs" in row:
            continue
        arms.setdefault(key.partition("|")[2], []).append(row)
    summary = {}
    for arm, rows in sorted(arms.items()):
        made = [r for r in rows if "plantMotionP95Px" in r]
        summary[arm] = {
            "clips": len(made),
            "worked": (len(made) == len(STARTS)) if made else None,
            "gpu": made[0].get("gpu") if made else None,
            "size": made[0].get("size") if made else None,
            "fps": made[0].get("fps") if made else None,
            "frames": made[0].get("frames") if made else None,
            "prompt": (made[0].get("prompt") or made[0].get("caption")) if made else None,
            "seconds": mean([r.get("seconds") for r in made]),
            "firstMotionSeconds": mean(
                [r.get("firstMotionSeconds") or r.get("firstFrameSeconds") for r in made]
            ),
            "realTimeFactor": mean([real_time(r) for r in made]),
            "sustainedFps": mean([r.get("sustainedFps") for r in made]),
            "dollarsWarm": mean([r.get("dollarsWarm") for r in made]),
            "plantMotionP95Px": mean([r.get("plantMotionP95Px") for r in made]),
            "plantMotionMeanPx": mean([r.get("plantMotionMeanPx") for r in made]),
            "cameraCreepPxMax": max(
                [r["cameraCreepPx"] for r in made if r.get("cameraCreepPx") is not None],
                default=None,
            ),
            "driftDropDbMean": mean(
                [
                    r["driftPsnrFirstDb"] - r["driftPsnrLastDb"]
                    for r in made
                    if r.get("driftPsnrFirstDb") is not None
                ]
            ),
            "upscale": {
                kind: {
                    "clips": len(done),
                    "seconds": mean([u.get("seconds") for u in done]),
                    "framesPerSecond": mean([u.get("framesPerSecond") for u in done]),
                    "peakMemoryGB": mean([u.get("peakMemoryGB") for u in done]),
                    "dollarsWarm": mean([u.get("dollarsWarm") for u in done]),
                }
                for kind in sorted({k for r in rows for k in r.get("upscale", {})})
                for done in [[r["upscale"][kind] for r in rows if kind in r.get("upscale", {})]]
            },
            "licence": licence(arm),
            "eye": EYE.get(arm),
        }
    return summary


def main(argv: list[str]) -> int:
    if argv[0] == "--summary":  # rebuild the per-arm summary and the notes of a numbers.json
        path = Path(argv[1])
        document = json.loads(path.read_text())
        document["arms"] = summarise(document["clips"])
        document["eye"] = EYE
        path.write_text(json.dumps(document, indent=1))
        return 0
    out = Path(argv[0])
    more = out.parent / f"{out.name}-more"
    starts_dir = Path(argv[1])
    data = load([Path(a) for a in argv[2:]])
    for folder in (out / "clips", more / "clips"):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    numbers: dict = {}
    gpu_compare: dict = {}
    files: dict = {}
    for (arm, start), clip in sorted(data["clips"].items()):
        entry = {k: v for k, v in clip["entry"].items() if k != "mp4"}
        seconds = entry.get("seconds")
        warm = round(seconds * clip["rate"], 4) if seconds and clip["rate"] else None
        if arm.endswith("-h100"):
            gpu_compare[f"{start}|{arm}"] = entry | {"gpu": clip["gpu"], "dollarsWarm": warm}
            continue
        name = DELIVER_AS.get(arm, arm)
        frames, fps = lv.read_clip(clip["path"])
        size = (frames[0].shape[1], frames[0].shape[0])
        render, mask = render_for(starts_dir / start, arm, size)
        row = entry | metrics(frames, render, mask)
        vs = video_seconds(entry)
        row |= {
            "gpu": clip["gpu"],
            "licence": licence(arm),
            "settings": clip["settings"],
            "loadSeconds": clip["loadSeconds"],
            "run": clip["run"],
            "call": clip["call"],
            "callDollars": clip["callDollars"],
            "clipsInCall": clip["clipsInCall"],
            "dollarsWarm": warm,
            "videoSeconds": vs,
            "realTimeFactor": round(vs / seconds, 3) if vs and seconds else None,
        }
        if name != arm:
            row["deliveredAs"] = name
            numbers[f"{start}|{arm}"] = {"sameAs": f"{start}|{name}"}
        target = f"{start}-{name}-a.mp4"
        grid = letterbox(frames, *GRID)
        row["crf"] = {"a": ffmpeg_encode(grid, fps, more / "clips" / target, GRID_MB)}
        files.setdefault((name, "a"), []).append(target)
        numbers[f"{start}|{name}"] = row
        print(start, name, json.dumps({k: row[k] for k in row if "Px" in k or "Psnr" in k}))
    for (kind, arm, start), up in sorted(data["ups"].items()):
        target = f"{start}-{arm}-{kind}.mp4"
        shutil.copyfile(up["path"], more / "clips" / target)
        files.setdefault((arm, kind), []).append(target)
        row = numbers.setdefault(f"{start}|{arm}", {"licence": licence(arm)})
        entry = {k: v for k, v in up["entry"].items() if k != "mp4"}
        seconds = entry.get("seconds")
        row.setdefault("upscale", {})[kind] = entry | {
            "attention": up.get("attention"),
            "bytes": (more / "clips" / target).stat().st_size,
            "dollarsWarm": round(seconds * up["rate"], 4) if seconds and up["rate"] else None,
            "upscalerLoadSeconds": up["loadSeconds"],
            "run": up["run"],
        }
    stills = sum(p.stat().st_size for p in (out / "stills").glob("*")) / 1e6
    placed = place(more, out, files, stills + 1.0)  # 1 MB kept for numbers.json
    for key, row in numbers.items():
        start, _, arm = key.partition("|")
        row["files"] = {
            kind: placed["where"][n]
            for (a, kind), names in files.items()
            if a == arm
            for n in names
            if n.startswith(f"{start}-")
        }
    summary = summarise(numbers)
    document = {
        "clips": numbers,
        "arms": summary,
        "gpuCompare": gpu_compare,
        "placement": {"limitMB": LIMIT_MB, "usedMB": placed["usedMB"], "overflow": more.name},
        "eye": EYE,
    }
    (out / "numbers.json").write_text(json.dumps(document, indent=1))
    for folder in (out, more):
        total = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        big = [p.name for p in folder.rglob("*.mp4") if p.stat().st_size > 14 * 1024 * 1024]
        print(f"{folder}: {total / 1e6:.1f} MB, files over 14 MB: {big}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

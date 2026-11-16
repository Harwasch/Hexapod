"""Render tiny, explicitly synthetic capture replays; no device or network needed."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[3] / "apps/ios/Fixtures/bundles"
WIDTH, HEIGHT, COUNT, FPS = 192, 144, 24, 12
SEED = 1701
OBJECTS = [
    {"label": "ground", "centre": [0, 0, 0], "extents": [20, 0, 20]},
    {"label": "shed", "centre": [-1.2, 0.75, 0], "extents": [1.4, 1.5, 1.4]},
    {"label": "trunk", "centre": [1.2, 0.8, 0], "extents": [0.3, 1.6, 0.3]},
    {"label": "crown", "centre": [1.2, 2, 0], "extents": [1.5, 1.5, 1.5]},
]


def camera(index: int) -> F64:
    centre = np.array([-0.8 + 1.6 * index / (COUNT - 1), 1.8, 5.5])
    back = centre - [0, 1.1, 0]
    back /= np.linalg.norm(back)
    right = np.cross([0, 1, 0], back)
    right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, np.cross(back, right), back))
    transform[:3, 3] = centre
    return transform


def render(transform: F64, width: int, height: int) -> tuple[F64, npt.NDArray[np.int64]]:
    """Analytic intersections; ray parameter is optical-axis depth, not range.

    Integer pixel coordinates denote sample centres. Depth intrinsics are RGB K
    scaled by 1/4 (no half-pixel offset), as used by the fixture's consumers.
    """
    y, x = np.indices((height, width))
    directions = (
        np.stack(
            (
                (x - width / 2) / (width * 0.85),
                -(y - height / 2) / (width * 0.85),
                -np.ones_like(x),
            ),
            axis=-1,
        )
        @ transform[:3, :3].T
    )
    origin = transform[:3, 3]
    candidates: list[F64] = []
    with np.errstate(divide="ignore", invalid="ignore"):
        plane = -origin[1] / directions[..., 1]
        hit = origin + plane[..., None] * directions
        candidates.append(
            np.where((plane > 0) & (np.abs(hit[..., [0, 2]]) <= 10).all(-1), plane, np.inf)
        )
        low = np.array([-1.9, 0, -0.7])
        high = np.array([-0.5, 1.5, 0.7])
        a, b = (low - origin) / directions, (high - origin) / directions
        near, far = np.minimum(a, b).max(-1), np.maximum(a, b).min(-1)
        candidates.append(np.where((near > 0) & (near <= far), near, np.inf))
        # Vertical, capped cylinder. The crown overlaps its top cap.
        offset = origin - [1.2, 0, 0]
        aa = directions[..., 0] ** 2 + directions[..., 2] ** 2
        bb = 2 * (offset[0] * directions[..., 0] + offset[2] * directions[..., 2])
        cc = offset[0] ** 2 + offset[2] ** 2 - 0.15**2
        disc = bb**2 - 4 * aa * cc
        side = (-bb - np.sqrt(np.maximum(disc, 0))) / (2 * aa)
        yy = origin[1] + side * directions[..., 1]
        trunk = np.where((disc >= 0) & (side > 0) & (yy >= 0) & (yy <= 1.6), side, np.inf)
        for cap_y in (0, 1.6):
            cap = (cap_y - origin[1]) / directions[..., 1]
            p = offset + cap[..., None] * directions
            trunk = np.minimum(
                trunk,
                np.where((cap > 0) & (p[..., 0] ** 2 + p[..., 2] ** 2 <= 0.15**2), cap, np.inf),
            )
        candidates.append(trunk)
        offset = origin - [1.2, 2, 0]
        aa = (directions**2).sum(-1)
        bb = 2 * (directions * offset).sum(-1)
        cc = float(offset @ offset) - 0.75**2
        disc = bb**2 - 4 * aa * cc
        sphere = (-bb - np.sqrt(np.maximum(disc, 0))) / (2 * aa)
        candidates.append(np.where((disc >= 0) & (sphere > 0), sphere, np.inf))
    distances = np.stack(candidates)
    return distances.min(0), distances.argmin(0)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_archive(folder: Path, target: Path) -> None:
    with tarfile.open(target, "w", format=tarfile.USTAR_FORMAT) as archive:
        for path in sorted(folder.iterdir()):
            payload = path.read_bytes()
            entry = tarfile.TarInfo(path.name)
            entry.size = len(payload)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(payload))


def generate(root: Path = ROOT) -> None:
    palette = np.array([[145, 120, 75], [190, 65, 45], [95, 55, 25], [40, 145, 55]], dtype=np.uint8)
    # Fixed seed controls tiny GPS jitter; scene geometry itself has no randomness.
    rng = np.random.default_rng(SEED)
    fixes = [
        {
            "timestamp": 1794830400 + i,
            "lat": 37.25 + float(rng.uniform(-1e-6, 1e-6)),
            "lon": -122.1,
            "altitudeM": 100,
            "horizontalAccuracyM": 5 + i,
            "verticalAccuracyM": 8,
        }
        for i in range(2)
    ]
    rgb_frames = []
    records: list[dict[str, Any]] = []
    depths = []
    probes = []
    for index in range(COUNT):
        transform = camera(index)
        distance, labels = render(transform, WIDTH, HEIGHT)
        rgb = palette[labels]
        rgb[~np.isfinite(distance)] = [130, 180, 225]
        rgb_frames.append(rgb.tobytes())
        depth, labels = render(transform, WIDTH // 4, HEIGHT // 4)
        depths.append(depth.astype("<f2").tobytes())
        records.append(
            {
                "index": index,
                "timestamp": index / FPS,
                "transform": transform.ravel().tolist(),
                "intrinsics": [WIDTH * 0.85, 0, WIDTH / 2, 0, WIDTH * 0.85, HEIGHT / 2, 0, 0, 1],
                "imageWidth": WIDTH,
                "imageHeight": HEIGHT,
                "trackingState": "normal",
                "depth": {
                    "file": f"depth_{index:08d}.f16",
                    "confidence": f"depth_{index:08d}.u8",
                    "width": WIDTH // 4,
                    "height": HEIGHT // 4,
                },
            }
        )
        for label, obj in enumerate(OBJECTS):
            pixels = np.argwhere((labels == label) & np.isfinite(depth))
            v, u = pixels[len(pixels) // 2]
            z = depth[v, u]
            local = (
                np.array(
                    [
                        (u - WIDTH / 8) / (WIDTH * 0.85 / 4),
                        -(v - HEIGHT / 8) / (WIDTH * 0.85 / 4),
                        -1,
                    ]
                )
                * z
            )
            probes.append(
                {
                    "frame": index,
                    "label": obj["label"],
                    "pixel": [int(u), int(v)],
                    "world": (transform[:3, :3] @ local + transform[:3, 3]).tolist(),
                }
            )
    for name, normal in (("yard-small", 22), ("yard-limited", 3)):
        destination = root / name
        folder = destination / "bundle"
        folder.mkdir(parents=True, exist_ok=True)
        # One encoder thread and pool; no timestamps or host paths in the bitstream.
        subprocess.run(  # noqa: S603
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{WIDTH}x{HEIGHT}",
                "-r",
                str(FPS),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx265",
                "-threads",
                "1",
                "-x265-params",
                "pools=none:frame-threads=1:wpp=0:info=0:log-level=error",
                "-pix_fmt",
                "yuv420p",
                "-tag:v",
                "hvc1",
                "-map_metadata",
                "-1",
                str(folder / "rgb.mov"),
            ],
            input=b"".join(rgb_frames),
            check=True,
        )
        frames = []
        for index, record in enumerate(records):
            frame = dict(record)
            frame["trackingState"] = (
                "limited:initializing"
                if index < 2
                else "normal"
                if index < normal + 2
                else "limited:excessiveMotion"
            )
            frames.append(frame)
            (folder / frame["depth"]["file"]).write_bytes(depths[index])
            confidence = np.where(
                np.isfinite(np.frombuffer(depths[index], dtype="<f2")), 2, 0
            ).astype(np.uint8)
            (folder / frame["depth"]["confidence"]).write_bytes(confidence.tobytes())
        (folder / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
        (folder / "location.jsonl").write_text("".join(json.dumps(f) + "\n" for f in fixes))
        manifest = {
            "version": 1,
            "format": "hex-capture",
            "device": "SYNTHETIC",
            "startedAt": "2026-11-16T12:00:00Z",
            "frameCount": COUNT,
            "depthFrameCount": COUNT,
            "smoothedDepthFrameCount": 0,
            "video": "rgb.mov",
            "frameLog": "frames.jsonl",
            "locationLog": "location.jsonl",
            "location": fixes[0],
            "files": [
                {"path": p.name, "bytes": p.stat().st_size}
                for p in sorted(folder.iterdir())
                if p.name != "manifest.json"
            ],
        }
        write_json(folder / "manifest.json", manifest)
        write_json(
            destination / "ground_truth.json",
            {
                "synthetic": True,
                "seed": SEED,
                "units": "metres",
                "objects": OBJECTS,
                "counts": {"shed": 1, "tree": 1},
                "treeHeight": 2.75,
                "normalFrames": normal,
                "depthIntrinsicsScale": 0.25,
                "probes": probes,
            },
        )
        write_archive(folder, destination / "capture.hexcap")


if __name__ == "__main__":
    generate()

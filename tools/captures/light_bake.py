"""Local Living Light experiment. Heavy teacher imports only in `teacher` (uv --group light).

Frozen scan, sRGB input; Marigold Lighting returns LINEAR albedo. No image generation or
photometric flattening substitutes for the teacher. Files and weights stay in the cache.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from splat_render import Camera, load_tileset, render

MODEL = "prs-eth/marigold-iid-lighting-v1-1"
SOURCE = (
    "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs/"
    "ec644db5-ceaf-4c01-bf89-b261623327c2/p63eb06dede59cf4a/package/splat/tileset.json"
)


def prepare(folder: Path, size: int = 512) -> None:
    """Read-only download of this bounded scan; deterministic capture-like oblique views."""
    folder.mkdir(parents=True, exist_ok=True)
    tiles = folder / "scan"
    tiles.mkdir(exist_ok=True)
    document = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(SOURCE, headers={"User-Agent": "living-light-research"}),
            timeout=60,
        ).read()
    )
    (tiles / "tileset.json").write_text(json.dumps(document))
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        stack.extend(tile.get("children", []))
        uri = tile.get("content", {}).get("uri")
        if uri:
            path = tiles / uri
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(
                    urllib.request.urlopen(
                        urllib.request.Request(
                            SOURCE.rsplit("/", 1)[0] + "/" + uri,
                            headers={"User-Agent": "living-light-research"},
                        ),
                        timeout=60,
                    ).read()
                )
    splats = load_tileset(tiles / "tileset.json")
    print(
        "splats",
        len(splats),
        "bounds",
        splats.positions.min(0),
        splats.positions.max(0),
        flush=True,
    )
    cameras = [
        Camera.look_at(eye, [0, 0, -0.05], width=size, height=size, fov_deg=49)
        for eye in ([4, -5, 3.5], [-4, -5, 3.5], [4, 4, 3.5], [-4, 4, 3.5])
    ]
    for i, camera in enumerate(cameras):
        start = time.monotonic()
        frame = render(
            splats, camera, max_samples=48, sample_budget=18_000_000, background=(0.62, 0.70, 0.78)
        )
        Image.fromarray(np.uint8(np.clip(frame.rgb, 0, 1) * 255)).save(folder / f"view-{i}.png")
        np.savez_compressed(folder / f"view-{i}.npz", depth=frame.depth, alpha=frame.alpha)
        print("render", i, time.monotonic() - start, flush=True)
    (folder / "cameras.json").write_text(json.dumps([c.to_json() for c in cameras], indent=2))


def teacher(folder: Path, steps: int = 5, threads: int = 8) -> None:
    """Run the real IID teacher on CPU, deterministic seeds; preserve raw linear outputs."""
    import torch
    from diffusers import MarigoldIntrinsicsPipeline

    torch.set_num_threads(threads)
    start = time.monotonic()
    pipe = MarigoldIntrinsicsPipeline.from_pretrained(MODEL, torch_dtype=torch.float32)
    pipe.to("cpu")
    pipe.enable_attention_slicing()
    times = []
    for i, path in enumerate(sorted(folder.glob("view-*.png"))):
        out = folder / f"teacher-{i}.npz"
        if out.exists():
            continue
        now = time.monotonic()
        result = pipe(
            Image.open(path).convert("RGB"),
            num_inference_steps=steps,
            ensemble_size=1,
            processing_resolution=512,
            generator=torch.Generator().manual_seed(20261025),
        )
        prediction = result.prediction
        print("prediction", prediction.shape, flush=True)
        np.savez_compressed(out, prediction=prediction)
        times.append({"view": i, "seconds": time.monotonic() - now})
        albedo = prediction[0]
        srgb = np.where(
            albedo <= 0.0031308, 12.92 * albedo, 1.055 * np.maximum(albedo, 0) ** (1 / 2.4) - 0.055
        )
        Image.fromarray(np.uint8(np.clip(srgb, 0, 1) * 255)).save(folder / f"albedo-{i}.png")
        print(times[-1], flush=True)
    (folder / "teacher-times.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "steps": steps,
                "threads": threads,
                "totalSeconds": time.monotonic() - start,
                "views": times,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "teacher", "bake"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.folder)
    elif args.stage == "teacher":
        teacher(args.folder, args.steps)
    else:
        bake(args.folder)


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    return np.where(
        value <= 0.0031308, value * 12.92, 1.055 * np.maximum(value, 0) ** (1 / 2.4) - 0.055
    )


def sun_path(count: int = 32, latitude: float = 45.21) -> np.ndarray:
    """Equinox, solar hours 07–17, local ENU. Same analytic path as the lab player."""
    hour = np.linspace(-75, 75, count) * np.pi / 180
    lat = latitude * np.pi / 180
    return np.stack(
        [-np.sin(hour), -np.sin(lat) * np.cos(hour), np.cos(lat) * np.cos(hour)], axis=1
    )


def fit_albedo(
    splats,
    cameras: list[Camera],
    predictions: list[np.ndarray],
    depths: list[np.ndarray],
    alphas: list[np.ndarray],
):
    """Weighted multi-view fit; depth rejects hidden surfaces. Unseen splats stay flagged.

    Each visible centre supplies a linear RGB observation. This is the least-squares
    constant albedo under a visibility/coverage weight, without colour equalisation or
    spatial smoothing. Occlusion tolerance is two splat radii + one projected pixel.
    """
    from splat_render import _rotation_matrices

    sums = np.zeros((len(splats), 3))
    weights = np.zeros(len(splats))
    observer = np.zeros_like(sums)
    for camera, albedo, depth, alpha in zip(cameras, predictions, depths, alphas, strict=True):
        uv, z = camera.project(splats.positions)
        pixel = np.floor(uv).astype(int)
        x, y = pixel.T
        valid = (z > 0) & (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
        ids = np.flatnonzero(valid)
        x, y = x[ids], y[ids]
        tolerance = 2 * splats.scales[ids].max(1) + z[ids] / camera.focal
        visible = (np.abs(z[ids] - depth[y, x]) < tolerance) & (alpha[y, x] > 0.8)
        ids, x, y = ids[visible], x[visible], y[visible]
        w = alpha[y, x]
        sums[ids] += albedo[y, x] * w[:, None]
        weights[ids] += w
        direction = camera.centre - splats.positions[ids]
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        observer[ids] += direction * w[:, None]
    albedo = srgb_to_linear(splats.colours)
    seen = weights > 0
    albedo[seen] = sums[seen] / weights[seen, None]
    rot = _rotation_matrices(splats.rotations)
    normal = rot[np.arange(len(splats)), :, np.argmin(splats.scales, axis=1)]
    # Sign ambiguity is resolved ONCE from capture-like observers, never the live camera.
    orientation = np.where(seen[:, None], observer, np.array([0.0, 0.0, 1.0]))
    normal[np.sum(normal * orientation, axis=1) < 0] *= -1
    return albedo, normal, np.minimum(weights / len(cameras), 1)


def trace_visibility(
    grid: np.ndarray, origin: np.ndarray, cell: float, points: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Beer–Lambert transmittance through scan-derived density, no external occluders.

    Two-cell start offset avoids self-occlusion. Sampling ends beyond the bounded volume;
    outside it is empty. grid stores extinction per metre. Runtime never traces rays.
    """
    from scipy.ndimage import map_coordinates

    result = np.empty((len(points), len(directions)), np.float32)
    distance = float(np.linalg.norm(np.asarray(grid.shape) * cell))
    base = (points - origin) / cell
    for j, direction in enumerate(directions):
        optical = np.zeros(len(points))
        for t in np.arange(2.0 * cell, distance, cell):
            pos = base + direction * (t / cell)
            optical += (
                map_coordinates(grid, pos.T, order=1, mode="constant", cval=0, prefilter=False)
                * cell
            )
        result[:, j] = np.exp(-optical)
    return result


def bake_visibility(splats, cell: float = 0.04):
    """Splat density splatted to a bounded grid; trace once per occupied voxel, then bind."""
    from scipy.ndimage import gaussian_filter

    low = splats.positions.min(0) - 3 * cell
    index = np.floor((splats.positions - low) / cell).astype(int)
    shape = index.max(0) + 4
    grid = np.zeros(shape, np.float32)
    # Surface footprint / voxel area gives optical mass. Small overlapping gaussians add.
    scales = np.sort(splats.scales, axis=1)
    mass = -np.log1p(-np.minimum(splats.opacities, 0.99)) * (
        2 * np.pi * scales[:, 1] * scales[:, 2] / cell**3
    )
    np.add.at(grid, tuple(index.T), mass)
    grid = gaussian_filter(grid, 0.55)
    unique, inverse = np.unique(index, axis=0, return_inverse=True)
    points = low + unique * cell
    sun = trace_visibility(grid, low, cell, points, sun_path())
    # Uniform-solid-angle hemisphere quadrature (12 deterministic Fibonacci directions).
    z = (np.arange(12) + 0.5) / 12
    angle = np.arange(12) * (np.pi * (3 - np.sqrt(5)))
    sky_dirs = np.stack(
        [np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z], 1
    )
    sky = trace_visibility(grid, low, cell, points, sky_dirs).mean(1)
    return sun[inverse], sky[inverse]


def pack_sidecar(albedo, normals, visibility, sky, confidence) -> bytes:
    """48 B/splat: sRGB8+confidence, snorm8 normal+sky, 32 visibility bytes, magic+version.

    Colour is quantised in sRGB to preserve dark albedo precision; the shader decodes to
    linear before lighting. Normals have no camera dependence. 3 RGBA32UI GPU streams.
    """
    n = len(albedo)
    rows = np.zeros((n, 48), np.uint8)
    rows[:, :3] = np.rint(np.clip(linear_to_srgb(albedo), 0, 1) * 255).astype(np.uint8)
    rows[:, 3] = np.rint(np.clip(confidence, 0, 1) * 255).astype(np.uint8)
    rows[:, 4:7] = np.rint(np.clip(normals, -1, 1) * 127 + 128).astype(np.uint8)
    rows[:, 7] = np.rint(np.clip(sky, 0, 1) * 255).astype(np.uint8)
    rows[:, 8:40] = np.rint(np.clip(visibility, 0, 1) * 255).astype(np.uint8)
    rows[:, 40:44] = np.array([76, 76, 48, 49], np.uint8)  # LL01
    return rows.tobytes()


def bake(folder: Path) -> None:
    from rig_tiles import glb_spz
    from splat_render import splats_from_columns
    from splat_tiles import tile_offsets, unpack_spz
    from synthetic_tree import checksum_positions

    start = time.monotonic()
    cameras = [Camera.from_json(c) for c in json.loads((folder / "cameras.json").read_text())]
    predictions = [np.load(folder / f"teacher-{i}.npz")["prediction"][0] for i in range(4)]
    frames = [np.load(folder / f"view-{i}.npz") for i in range(4)]
    full = load_tileset(folder / "scan/tileset.json")
    visibility, sky = bake_visibility(full)
    print("visibility seconds", time.monotonic() - start, flush=True)
    # All LODs get the same fitted teacher and nearest full-detail visibility, keyed to
    # the exact original position checksum and order. No source colour/geometry changes.
    from scipy.spatial import cKDTree

    tree = cKDTree(full.positions)
    document = json.loads((folder / "scan/tileset.json").read_text())
    offsets = tile_offsets(document)
    output = folder / "sidecars"
    output.mkdir(exist_ok=True)
    manifest = {
        "version": 1,
        "bytesPerSplat": 48,
        "source": SOURCE,
        "sunPath": sun_path().tolist(),
        "tiles": {},
        "cellM": 0.04,
    }
    for uri, offset in offsets.items():
        splats = splats_from_columns(unpack_spz(glb_spz(folder / "scan" / uri)), offset)
        albedo, normals, confidence = fit_albedo(
            splats, cameras, predictions, [f["depth"] for f in frames], [f["alpha"] for f in frames]
        )
        _, nearest = tree.query(splats.positions, workers=2)
        checksum = checksum_positions(splats.positions)
        name = uri.replace("/", "_") + ".light.bin"
        (output / name).write_bytes(
            pack_sidecar(albedo, normals, visibility[nearest], sky[nearest], confidence)
        )
        manifest["tiles"][checksum] = {
            "uri": name,
            "count": len(splats),
            "observedFraction": float(np.mean(confidence > 0)),
        }
        print(uri, len(splats), "observed", float(np.mean(confidence > 0)), flush=True)
    manifest["bakeSeconds"] = time.monotonic() - start
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

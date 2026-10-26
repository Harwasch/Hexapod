"""CPU-only simulated branch wind -> sparse voxel Fourier sidecar (docs/LIVING_SPECTRA.md).

Run from tools/captures:
  uv run python living_spectra.py ../../.lab-data/minnetonka-tree \
      ../../.lab-out/living-spectra/bake
No learned model or measured wind: all motion is explicitly simulated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from rig_tiles import tile_positions
from splat_tiles import tile_offsets


def spectrum(samples: np.ndarray, fps: float, k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Real series [time, voxel, xyz] -> normalized complex amplitudes [voxel,k,xyz].

    Keep DC plus globally strongest bins, reserving four terms above 1.5 Hz for twigs.
    Interior rFFT bins are doubled: playback is Re(C exp(+i 2π f t)), no implicit factor.
    """
    n = len(samples)
    if not 1 <= k <= n // 2 + 1:
        raise ValueError("K exceeds rFFT bins")
    ft = np.fft.rfft(samples, axis=0) / n
    ft[1 : (len(ft) - 1 if n % 2 == 0 else len(ft))] *= 2
    frequencies = np.fft.rfftfreq(n, 1 / fps)
    power = np.sum(abs(ft) ** 2, axis=(1, 2))
    high = np.flatnonzero(frequencies > 1.5)
    reserved = high[np.argsort(power[high])[-min(4, k - 1) :]] if k > 1 else []
    chosen = {0, *reserved}
    for i in np.argsort(power)[::-1]:
        if len(chosen) >= k:
            break
        chosen.add(int(i))
    bins = np.array(sorted(chosen))
    return frequencies[bins], ft[bins].transpose(1, 0, 2)


def inverse(frequencies: np.ndarray, coefficients: np.ndarray, times: np.ndarray) -> np.ndarray:
    basis = np.exp(2j * np.pi * times[:, None] * frequencies[None, :])
    return np.einsum("tk,vkc->tvc", basis, coefficients).real


def write_sidecar(out: Path, meta: dict, indices: np.ndarray, coefficients: np.ndarray) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    packed = np.stack((coefficients.real, coefficients.imag), axis=-1).astype("<f2")
    if not np.isfinite(packed).all():
        raise ValueError("Non-finite or overflowing coefficients")
    binary = indices.astype("<u4").tobytes() + packed.tobytes()
    header = {
        **meta,
        "format": "hexapod.living-spectra",
        "version": 1,
        "binary": "spectra.bin",
        "encoding": "float16-le",
        "occupied": len(indices),
        "coefficientOffset": 4 * len(indices),
        "byteLength": len(binary),
        "motionEvidence": "simulated",
    }
    (out / "spectra.bin").write_bytes(binary)
    (out / "spectra.json").write_text(json.dumps(header, indent=2) + "\n")
    return header


def read_sidecar(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    meta = json.loads(path.read_text())
    data = (path.parent / meta["binary"]).read_bytes()
    n, k = meta["occupied"], len(meta["frequenciesHz"])
    if len(data) != n * (4 + k * 12) or meta["coefficientOffset"] != n * 4:
        raise ValueError("Invalid binary length or offset")
    indices = np.frombuffer(data, dtype="<u4", count=n)
    values = np.frombuffer(data, dtype="<f2", offset=n * 4).reshape(n, k, 3, 2)
    return meta, indices, values[..., 0].astype(float) + 1j * values[..., 1].astype(float)


def scan_points(folder: Path) -> np.ndarray:
    tileset = json.loads((folder / "tileset.json").read_text())
    offsets = tile_offsets(tileset)
    uris: list[str] = []

    def walk(tile: dict) -> None:
        if tile.get("children"):
            for child in tile["children"]:
                walk(child)
        elif "content" in tile:
            uris.append(tile["content"]["uri"])

    walk(tileset["root"])
    return np.concatenate([tile_positions(folder / uri, offsets.get(uri)) for uri in uris])


def lattice(points: np.ndarray, resolution: int = 128, budget: int = 50_000):
    """All eight interpolation corners occupied by measured splats, x fastest.

    Expand extent, never drop occupied cells. One empty border protects the upper boundary.
    The resolution remains 128³; report resulting cell size rather than imply finer detail.
    """
    low, high = points.min(axis=0), points.max(axis=0)
    side = float(max(high - low)) * 1.05
    corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)])
    while True:
        origin = (low + high - side) / 2
        cell = side / (resolution - 1)
        base = np.floor((points - origin) / cell).astype(np.int32)
        # Deduplicate base cells first, before expanding their eight corners.
        base = np.unique(base, axis=0)
        coords = np.unique((base[:, None, :] + corners).reshape(-1, 3), axis=0)
        if len(coords) <= budget:
            break
        side *= 1.12
    indices = coords[:, 0] + resolution * (coords[:, 1] + resolution * coords[:, 2])
    order = np.argsort(indices)
    return origin, side, indices[order], origin + coords[order] * cell


def simulate_nodes(rig: dict, fps: int = 30, seconds: int = 10):
    """Hierarchical rotations about parent joints, driven by periodic gusts and damping.

    Five periods of warmup. 120 Hz semi-implicit steps, last period sampled at 30 Hz.
    Frequencies grow from trunk to tips; drag and damping are artistic priors, not fitted.
    """
    nodes = rig["nodes"]
    rest = np.array([n["position"] for n in nodes], dtype=float)
    parents = np.array([n["parent"] for n in nodes])
    if any(p >= i or p < -1 for i, p in enumerate(parents)):
        raise ValueError("Rig must be parent-first and acyclic")
    depth = np.zeros(len(nodes))
    for i, p in enumerate(parents):
        if p >= 0:
            depth[i] = depth[p] + 1
    rng = np.random.default_rng(2510)
    phase = rng.uniform(0, 2 * np.pi, (len(nodes), 3))
    trunk = np.array([n["band"] == "trunk" for n in nodes])
    frequency = np.where(trunk, 0.35 + depth * 0.012, 0.7 + depth * 0.045)
    omega = (2 * np.pi * frequency)[:, None]
    gain = np.where(trunk, 0.0015, 0.008)[:, None]
    q = np.zeros((len(nodes), 3))
    velocity = q.copy()
    angles = []
    dt = 1 / 120
    for step in range(seconds * 120 * 6):
        t = step * dt
        # Correlated slow forcing plus branch-specific travelling phase and twig excitation.
        gust = (
            0.65
            + 0.28 * np.sin(2 * np.pi * 0.2 * t)
            + 0.22 * np.sin(2 * np.pi * 0.5 * t - rest[:, 0, None] * 0.45)
        )
        target = gain * gust * np.array([0.35, 1.0, 0.15])
        target += gain * (
            0.45 * np.sin(2 * np.pi * 0.7 * t + phase)
            + 0.3 * np.sin(2 * np.pi * 1.1 * t + phase * 1.7)
        )
        target += gain * 0.18 * np.sin(2 * np.pi * 2.3 * t + phase * 2)
        target[parents < 0] = 0
        velocity += (omega**2 * (target - q) - 2 * 0.22 * omega * velocity) * dt
        q += velocity * dt
        if step >= seconds * 120 * 5 and step % (120 // fps) == 0:
            angles.append(q.copy())
    local = (
        Rotation.from_rotvec(np.array(angles).reshape(-1, 3))
        .as_matrix()
        .reshape(seconds * fps, len(nodes), 3, 3)
    )
    rotations = local.copy()
    positions = np.broadcast_to(rest, (seconds * fps, *rest.shape)).copy()
    for i, p in enumerate(parents):
        if p < 0:
            continue
        rotations[:, i] = rotations[:, p] @ local[:, i]
        positions[:, i] = positions[:, p] + np.einsum(
            "tij,j->ti", rotations[:, p], rest[i] - rest[p]
        )
    return rest, parents, rotations, positions


def attach(points: np.ndarray, rest: np.ndarray, parents: np.ndarray):
    """Smooth four-segment attachment, selected by distance to the branch, not just its tip."""
    start = rest[np.maximum(parents, 0)]
    segment = rest - start
    distance = np.empty((len(points), len(rest)), dtype=np.float32)
    for i in range(len(rest)):
        alpha = np.clip((points - start[i]) @ segment[i] / max(segment[i] @ segment[i], 1e-9), 0, 1)
        distance[:, i] = np.linalg.norm(points - start[i] - alpha[:, None] * segment[i], axis=1)
    ids = np.argsort(distance, axis=1)[:, :4]
    d = np.take_along_axis(distance, ids, axis=1)
    weights = 1 / (d**2 + 0.06**2) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    return ids, weights


def bake(source: Path, out: Path) -> dict:
    rig = json.loads((source / "source/rig.json").read_text())
    points = scan_points(source / "splat")
    origin, side, indices, voxels = lattice(points)
    print(
        f"{len(points)} splats; {len(indices)} occupied corners; cell {side / 127:.4f} m",
        flush=True,
    )
    rest, parents, rotations, positions = simulate_nodes(rig)
    ids, weights = attach(voxels, rest, parents)
    # Empty-space corners still follow the local branch. The base fades to an exact pin.
    pin = np.clip((voxels[:, 2] - rest[:, 2].min()) / 0.65, 0, 1)
    pin = pin * pin * (3 - 2 * pin)
    samples = np.zeros((300, len(voxels), 3), dtype=np.float32)
    for j in range(ids.shape[1]):
        node = ids[:, j]
        offset = voxels - rest[node]
        moved = np.einsum("tvij,vj->tvi", rotations[:, node], offset) + positions[:, node]
        samples += ((moved - voxels) * weights[None, :, j, None] * pin[None, :, None]).astype(
            np.float32
        )
    # Fine flutter is spatially smooth within the voxel field and fades near woody segments.
    nearest, _ = cKDTree(rest).query(voxels)
    flutter = np.clip(nearest / 0.4, 0, 1) * pin * 0.008
    t = np.arange(300) / 30
    phase = voxels @ np.array([3.1, 2.7, 1.9])
    samples[:, :, 1] += (np.sin(2 * np.pi * 2.3 * t[:, None] + phase) * flutter).astype(np.float32)
    samples[:, :, 2] += (np.sin(2 * np.pi * 3.1 * t[:, None] + phase * 1.3) * flutter * 0.6).astype(
        np.float32
    )
    frequencies, coefficients = spectrum(samples, 30)
    reconstructed = inverse(frequencies, coefficients, t)
    metrics = {
        "rmsReconstructionErrorM": float(np.sqrt(np.mean((samples - reconstructed) ** 2))),
        "peakDisplacementM": float(np.max(np.linalg.norm(samples, axis=2))),
        "rmsDisplacementM": float(np.sqrt(np.mean(samples**2))),
        "cellM": side / 127,
        "splats": len(points),
        "nodes": len(rest),
    }
    meta = write_sidecar(
        out,
        {
            "origin": origin.tolist(),
            "size": [side] * 3,
            "resolution": 128,
            "frequenciesHz": frequencies.tolist(),
            "durationS": 10,
            "sampleRateHz": 30,
            "referenceBearingDeg": 90,
            "sourceChecksum": rig["canonicalChecksum"],
            "sourceNote": rig["sourceNote"],
            "metrics": metrics,
        },
        indices,
        coefficients,
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    bake(args.source, args.out)

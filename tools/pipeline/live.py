"""Watching a reconstruction happen: cameras as they are solved, splats as they train.

A capture spends most of its life in two long stages -- the pose solve (minutes on a CPU
box) and training (up to an hour on a GPU) -- and until now a person could only watch a
progress bar. This module is what lets the viewer show the work itself. It has two
halves, one on each side of the stage log:

**In the container**, beside the running tool:

* `MapperWatch` watches the directory COLMAP's incremental mapper writes snapshots into
  (`--Mapper.snapshot_path` / `--Mapper.snapshot_images_freq`, both in COLMAP 3.9.1's
  `colmap mapper -h`), and every `every_s` logs the newest one as a `live-cameras:` line:
  each registered camera's centre, view direction and up, and a sample of the sparse
  points with their colours. Measured on 3.9.1: a snapshot is a directory of
  `cameras.bin`, `images.bin` and `points3D.bin` named after a millisecond timestamp that
  overflows a 32-bit int (`-528822934`), so snapshots are ordered by mtime, never by name;
  and each is **normalised** the way the mapper normalises after a global bundle
  adjustment (centred, extent about 10), so successive snapshots sit roughly on top of one
  another rather than drifting. `emit_model` is the same line for any finished model, so a
  global mapper (GLOMAP) can call it too.
* `SplatWatch` watches gsplat's `ply/` directory for the intermediate splats asked for by
  `--ply_steps` (v1.5.3 writes `point_cloud_<i>.ply` at step `i` for every listed step,
  scaled by `--steps_scaler`, as well as the final one), keeps the `max_gaussians` with
  the highest opacity x volume, packs them as SPZ (the packer `splat_tiles.py` already
  uses for the map, which Spark reads) into `checkpoint/live/`, and logs a `live-splat:`
  line naming the object key the checkpoint syncer will upload it under. The full
  intermediate PLY is deleted once it is packed: it is hundreds of megabytes and nothing
  else reads it, and `training.latest_ply` then only ever sees the final step's.

**In the worker**, `latest` reads the newest of each line back out of a stage log, which
is how the supervisor copies them onto the running step (`metrics.live`) for the API.

Both lines are single JSON objects on one line. Positions are quantised to int16 against an
origin and a scale carried in the line and packed as base64, so the whole cameras line --
400 cameras and 1,500 points at most -- is about 25 kB: small enough for a log line, and
for a row in the database.
"""

from __future__ import annotations

import base64
import contextlib
import itertools
import json
import math
import re
import shutil
import struct
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import sfm
from captures_bridge import SplatFormatError, pack_spz, read_ply, sigmoid

__all__ = [
    "CAMERAS_TAG",
    "LIVE_DIR",
    "MAX_CAMERAS",
    "MAX_GAUSSIANS",
    "MAX_POINTS",
    "SPLAT_TAG",
    "TRAIN_FRACTIONS",
    "MapperWatch",
    "SplatWatch",
    "decode_cameras",
    "emit_model",
    "encode_model",
    "latest",
    "parse_line",
    "snapshot_every",
    "snapshot_splat",
    "train_ply_steps",
]

CAMERAS_TAG = "live-cameras: "
SPLAT_TAG = "live-splat: "
#: The directory under a stage's `checkpoint/` that live snapshots are written into.
LIVE_DIR = "live"
MAX_CAMERAS = 400
MAX_POINTS = 1_500
MAX_GAUSSIANS = 100_000
#: Where in the schedule an intermediate splat is written, as fractions of it.
TRAIN_FRACTIONS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75)
#: A line the parser accepts is at most this long, whatever wrote it.
MAX_LINE = 96_000
#: SPZ v2 stores positions as 24-bit fixed point with 12 fractional bits: +/-2048.
_SPZ_LIMIT = 2_000.0
_INT16 = 32_767.0

Log = Callable[[str], None]

#: One sequence across every watcher in the process, so a line's `seq` only goes up.
_SEQ = itertools.count(1)


# --- cameras ------------------------------------------------------------------------


def snapshot_every(frames: int) -> int:
    """`--Mapper.snapshot_images_freq`: about sixty snapshots over a full solve.

    A snapshot rewrites the whole reconstruction, so one per registered image would be
    thousands of files for a long video; the watcher only reads one every few seconds
    anyway, and deletes the ones it has passed.
    """
    return max(1, math.ceil(max(frames, 1) / 60))


def _unit(vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / np.where(norms > 0, norms, 1.0), dtype=np.float64)


def _read_points_sample(
    path: Path, limit: int, seed: int = 0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], int]:
    """(xyz, rgb, total) for a sample of `points3D.bin`, preferring well-tracked points.

    One pass over the file in memory: a record is 43 bytes of point and a track whose
    length it states, so it cannot be read as one array. Raises on a truncated file,
    which is what a snapshot still being written looks like.
    """
    raw = path.read_bytes()
    (count,) = struct.unpack_from("<Q", raw, 0)
    offset = 8
    record = struct.Struct("<Q3d3BdQ")  # id, xyz, rgb, error, track length: 51 bytes
    rows: list[tuple[Any, ...]] = []
    for _ in range(count):
        row = record.unpack_from(raw, offset)
        rows.append(row)
        offset += record.size + 8 * row[8]
    if offset > len(raw):
        raise ValueError(f"{path} is truncated")
    table = np.asarray(rows, dtype=np.float64).reshape(-1, 9)
    xyz = table[:, 1:4]
    rgb = table[:, 4:7].astype(np.uint8)
    tracks = table[:, 8]
    candidates = np.flatnonzero(tracks >= 3)
    if candidates.size < min(limit, count):
        candidates = np.arange(count)
    if candidates.size > limit:
        candidates = np.sort(np.random.default_rng(seed).choice(candidates, limit, replace=False))
    return xyz[candidates], rgb[candidates], int(count)


def _b64(array: npt.NDArray[Any]) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def encode_model(
    model: sfm.Model,
    points: npt.NDArray[np.float64],
    colours: npt.NDArray[np.uint8],
    *,
    total_points: int | None = None,
    frames: int | None = None,
    final: bool = False,
    max_cameras: int = MAX_CAMERAS,
) -> dict[str, Any]:
    """The `live-cameras` payload for one model: cameras and a point sample, quantised.

    `cameras` is base64 of 12 bytes per camera, little-endian: int16 x, y, z of the
    centre (`origin + q / 32767 * scale`), then int8 x, y, z of the view direction and of
    the camera's up (`/ 127`). `points` is 9 bytes per point: int16 x, y, z, then uint8
    r, g, b. Cameras are in image-name order -- capture order, for frames cut from a video.
    """
    images = list(model.images)
    if len(images) > max_cameras:
        keep = np.unique(np.linspace(0, len(images) - 1, max_cameras).round().astype(int))
        images = [images[int(i)] for i in keep]
    if images:
        rotations = np.stack([image.rotation for image in images])
        centres = np.stack([image.centre for image in images])
        # COLMAP cameras look down +z with +y down the image: the view direction in the
        # world is R's third row, and the camera's up is minus its second.
        forward = _unit(rotations[:, 2, :])
        up = _unit(-rotations[:, 1, :])
    else:
        centres = np.zeros((0, 3))
        forward = up = np.zeros((0, 3))
    anchor = centres if len(centres) else points
    origin = np.median(anchor, axis=0) if len(anchor) else np.zeros(3)
    reach = float(np.max(np.linalg.norm(centres - origin, axis=1))) if len(centres) else 0.0
    if len(points):
        distance = np.linalg.norm(points - origin, axis=1)
        # A stray point hundreds of units out would cost every other point its precision.
        limit = max(reach, float(np.percentile(distance, 98)) * 1.5, 1e-6)
        inside = distance <= limit
        points, colours = points[inside], colours[inside]
        reach = max(reach, float(distance[inside].max()) if inside.any() else 0.0)
    scale = max(reach, 1e-6) * 1.001
    q_cam = np.round((centres - origin) / scale * _INT16).astype("<i2")
    q_dir = np.round(np.hstack([forward, up]) * 127.0).astype("<i1")
    cam_bytes = np.hstack([q_cam.view(np.uint8).reshape(-1, 6), q_dir.view(np.uint8)])
    q_pts = np.round((points - origin) / scale * _INT16).astype("<i2")
    pt_bytes = np.hstack([q_pts.view(np.uint8).reshape(-1, 6), colours.astype(np.uint8)])
    camera = model.cameras[0] if model.cameras else None
    estimate = sfm.camera_up(model)
    return {
        "v": 1,
        "seq": next(_SEQ),
        "final": final,
        "registered": model.registered,
        "frames": frames,
        "origin": [round(float(v), 6) for v in origin],
        "scale": round(scale, 6),
        "up": None if estimate is None else [round(v, 5) for v in estimate.up],
        "aspect": (
            round(camera.width / camera.height, 4) if camera and camera.height > 0 else None
        ),
        "cameraCount": len(images),
        "cameras": _b64(cam_bytes.reshape(-1)),
        "pointCount": int(points.shape[0]),
        "pointsTotal": int(total_points if total_points is not None else points.shape[0]),
        "points": _b64(pt_bytes.reshape(-1)),
    }


def read_snapshot(
    directory: Path,
    *,
    frames: int | None = None,
    final: bool = False,
    max_cameras: int = MAX_CAMERAS,
    max_points: int = MAX_POINTS,
) -> dict[str, Any]:
    """`encode_model` of a COLMAP model directory. Raises while it is still being written."""
    model = sfm.read_model(directory)
    points, colours, total = _read_points_sample(directory / "points3D.bin", max_points)
    return encode_model(
        model,
        points,
        colours,
        total_points=total,
        frames=frames,
        final=final,
        max_cameras=max_cameras,
    )


def cameras_line(payload: Mapping[str, Any]) -> str:
    return CAMERAS_TAG + json.dumps(payload, separators=(",", ":"))


def emit_model(
    log: Log,
    directory: Path,
    *,
    frames: int | None = None,
    final: bool = True,
    max_cameras: int = MAX_CAMERAS,
    max_points: int = MAX_POINTS,
) -> dict[str, Any] | None:
    """Log one `live-cameras` line for a finished model -- for any mapper, not just this one.

    Never raises: a line the viewer does not get is a line the viewer does not draw, and
    must not fail a pose solve that has just succeeded.
    """
    try:
        payload = read_snapshot(
            directory,
            frames=frames,
            final=final,
            max_cameras=max_cameras,
            max_points=max_points,
        )
    except (OSError, ValueError, struct.error) as error:
        log(f"live: could not read the model at {directory.name} for the viewer: {error}")
        return None
    log(cameras_line(payload))
    return payload


class MapperWatch:
    """A thread that logs the newest mapper snapshot, at most once every `every_s`.

    Use it around the mapper run: `with MapperWatch(dir, ctx.log, frames=n): ctx.run(...)`.
    The directory is created on entry (COLMAP writes into it but does not make it) and
    removed on exit. Snapshots the watcher has moved past are deleted as it goes, so a
    long solve does not fill the disk with copies of itself.
    """

    def __init__(
        self,
        directory: Path,
        log: Log,
        *,
        frames: int | None = None,
        every_s: float = 5.0,
        poll_s: float = 1.0,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
        max_cameras: int = MAX_CAMERAS,
        max_points: int = MAX_POINTS,
    ) -> None:
        self.directory = directory
        self._log = log
        self._frames = frames
        self._every_s = every_s
        self._poll_s = poll_s
        self._enabled = enabled
        self._clock = clock
        self._max_cameras = max_cameras
        self._max_points = max_points
        self._last_emit = float("-inf")
        self._last_seen: tuple[float, str] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.emitted = 0

    def __enter__(self) -> MapperWatch:
        if not self._enabled:
            return self
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True)
        self._thread = threading.Thread(target=self._run, daemon=True, name="live-cameras")
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=30.0)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_s):
            with contextlib.suppress(Exception):  # watching must never fail the stage
                self.poll()

    def _snapshots(self) -> list[tuple[float, Path]]:
        found: list[tuple[float, Path]] = []
        with contextlib.suppress(OSError):
            for entry in self.directory.iterdir():
                points = entry / "points3D.bin"
                with contextlib.suppress(OSError):
                    found.append((points.stat().st_mtime, entry))
        return sorted(found, key=lambda item: (item[0], item[1].name))

    def poll(self) -> bool:
        """Log the newest snapshot if one is due. True when a line was logged."""
        if self._clock() - self._last_emit < self._every_s:
            return False
        snapshots = self._snapshots()
        if not snapshots:
            return False
        mtime, newest = snapshots[-1]
        if self._last_seen == (mtime, newest.name):
            return False
        try:
            payload = read_snapshot(
                newest,
                frames=self._frames,
                max_cameras=self._max_cameras,
                max_points=self._max_points,
            )
        except (OSError, ValueError, struct.error):
            return False  # still being written; the next tick reads it
        self._log(cameras_line(payload))
        self._last_emit = self._clock()
        self._last_seen = (mtime, newest.name)
        self.emitted += 1
        for _mtime, older in snapshots[:-1]:
            shutil.rmtree(older, ignore_errors=True)
        return True


# --- splats -------------------------------------------------------------------------


def train_ply_steps(max_steps: int, fractions: Iterable[float] = TRAIN_FRACTIONS) -> list[int]:
    """The extra `--ply_steps`, unscaled like `max_steps` (the trainer scales them all)."""
    steps = {int(max_steps * fraction) for fraction in fractions if 0.0 < fraction < 1.0}
    return sorted(step for step in steps if 0 < step < max_steps)


def ply_indices(live_steps: Iterable[int], steps_scaler: float) -> set[int]:
    """The file indices v1.5.3 writes for them: `point_cloud_{int(s * f) - 1}.ply`."""
    return {int(step * steps_scaler) - 1 for step in live_steps if int(step * steps_scaler) > 0}


def snapshot_splat(
    ply: Path, target: Path, *, max_gaussians: int = MAX_GAUSSIANS
) -> tuple[int, int, int]:
    """Pack the `max_gaussians` most opaque-and-largest gaussians of `ply` as SPZ.

    Returns (kept, of, bytes). Degree-0 colour only: a preview does not need the view-
    dependent terms, and they are most of an SPZ. Raises `SplatFormatError` on a PLY the
    trainer has not finished writing.
    """
    data = read_ply(ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    log_scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)
    alpha = sigmoid(data["opacity"])
    score = alpha * np.exp(np.clip(log_scales.sum(axis=1), -60.0, 60.0))
    good = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(score)
        & np.isfinite(log_scales).all(axis=1)
        & (np.abs(xyz) < _SPZ_LIMIT).all(axis=1)
    )
    candidates = np.flatnonzero(good)
    if candidates.size > max_gaussians:
        top = np.argpartition(-score[candidates], max_gaussians - 1)[:max_gaussians]
        candidates = np.sort(candidates[top])
    sh0 = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)[candidates]
    quat = np.stack([data["rot_1"], data["rot_2"], data["rot_3"], data["rot_0"]], axis=1)
    blob = pack_spz(
        xyz[candidates],
        sh0,
        data["opacity"][candidates],
        log_scales[candidates],
        quat[candidates],
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(blob)
    partial.replace(target)
    return int(candidates.size), int(xyz.shape[0]), len(blob)


_PLY_STEP = re.compile(r"point_cloud_(\d+)\.ply$")


@dataclass(frozen=True)
class _Pending:
    index: int
    path: Path
    size: int


class SplatWatch:
    """A thread that turns each intermediate PLY into a small SPZ in `checkpoint/live/`.

    `indices` are the file indices to snapshot (`ply_indices`); any other PLY -- the
    final one above all -- is never touched. A file is packed once its size has held
    still between two polls (the trainer writes it in one go, but a reader can still
    land in the middle), and a pack that fails because the file is short is retried.
    On exit whatever is still pending is packed, and every intermediate PLY is removed.
    """

    def __init__(
        self,
        ply_dir: Path,
        live_dir: Path,
        log: Log,
        *,
        indices: Iterable[int],
        total: int,
        key_prefix: str,
        up: Sequence[float] | None = None,
        max_gaussians: int = MAX_GAUSSIANS,
        poll_s: float = 2.0,
        enabled: bool = True,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._ply_dir = ply_dir
        self._live_dir = live_dir
        self._log = log
        self._indices = set(indices)
        self._total = total
        self._key_prefix = key_prefix.rstrip("/")
        self._up = None if up is None else [round(float(v), 5) for v in up]
        self._max_gaussians = max_gaussians
        self._poll_s = poll_s
        self._enabled = enabled and bool(self._indices)
        self._now = now
        self._sizes: dict[int, int] = {}
        self._done: set[int] = set()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.emitted: list[dict[str, Any]] = []

    def __enter__(self) -> SplatWatch:
        # This attempt's snapshots only: an earlier attempt's (or the preview's, on a
        # Refine) would otherwise be uploaded again as if they were this run's.
        shutil.rmtree(self._live_dir, ignore_errors=True)
        if self._enabled:
            self._thread = threading.Thread(target=self._run, daemon=True, name="live-splat")
            self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=120.0)
        if self._enabled:
            with contextlib.suppress(Exception):
                self.poll(settled=True)
        for index, path in self._intermediates():
            if index in self._indices:
                path.unlink(missing_ok=True)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_s):
            with contextlib.suppress(Exception):  # watching must never fail the stage
                self.poll()

    def _intermediates(self) -> list[tuple[int, Path]]:
        found: list[tuple[int, Path]] = []
        with contextlib.suppress(OSError):
            for path in self._ply_dir.glob("point_cloud_*.ply"):
                match = _PLY_STEP.search(path.name)
                if match is not None:
                    found.append((int(match.group(1)), path))
        return sorted(found)

    def poll(self, *, settled: bool = False) -> bool:
        """Pack the newest intermediate PLY that is ready. True when one was logged.

        `settled` skips the size check: the trainer has exited, so nothing is still
        being written.
        """
        with self._lock:
            ready: list[_Pending] = []
            for index, path in self._intermediates():
                if index not in self._indices or index in self._done:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if settled or (size > 0 and self._sizes.get(index) == size):
                    ready.append(_Pending(index, path, size))
                self._sizes[index] = size
            if not ready:
                return False
            newest = ready[-1]
            # An older one that was never shown is not worth showing now.
            for skipped in ready[:-1]:
                self._done.add(skipped.index)
                skipped.path.unlink(missing_ok=True)
            return self._pack(newest)

    def _pack(self, pending: _Pending) -> bool:
        step = pending.index + 1
        name = f"splat_{step:06d}_{int(self._now())}.spz"
        try:
            kept, of, size = snapshot_splat(
                pending.path, self._live_dir / name, max_gaussians=self._max_gaussians
            )
        except (SplatFormatError, OSError, ValueError, KeyError) as error:
            self._sizes.pop(pending.index, None)
            if isinstance(error, KeyError):
                # A PLY without the columns a splat has will never pack; say so once.
                self._done.add(pending.index)
                self._log(f"live: {pending.path.name} is not a splat this can pack: {error}")
            return False
        self._done.add(pending.index)
        pending.path.unlink(missing_ok=True)
        for older in self._live_dir.glob("splat_*.spz"):
            if older.name != name:
                older.unlink(missing_ok=True)
        payload = {
            "v": 1,
            "step": step,
            "total": self._total,
            "count": kept,
            "of": of,
            "bytes": size,
            "key": f"{self._key_prefix}/{name}",
            "up": self._up,
        }
        self._log(SPLAT_TAG + json.dumps(payload, separators=(",", ":")))
        self.emitted.append(payload)
        return True


# --- reading the lines back ---------------------------------------------------------


def _floats(value: object, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == count
        and all(isinstance(v, int | float) and math.isfinite(v) for v in value)
    )


def _count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_cameras(payload: Mapping[str, Any]) -> bool:
    cameras, points = payload.get("cameras"), payload.get("points")
    return (
        payload.get("v") == 1
        and isinstance(cameras, str)
        and isinstance(points, str)
        and len(cameras) + len(points) <= MAX_LINE
        and _floats(payload.get("origin"), 3)
        and isinstance(payload.get("scale"), int | float)
        and float(payload["scale"]) > 0
        and (payload.get("up") is None or _floats(payload.get("up"), 3))
        and _count(payload.get("cameraCount"))
        and _count(payload.get("pointCount"))
        and len(cameras) == 4 * math.ceil(12 * payload["cameraCount"] / 3)
        and len(points) == 4 * math.ceil(9 * payload["pointCount"] / 3)
    )


def _valid_splat(payload: Mapping[str, Any]) -> bool:
    key = payload.get("key")
    return (
        payload.get("v") == 1
        and isinstance(key, str)
        and len(key) <= 512
        and key.startswith("runs/")
        and f"/checkpoint/{LIVE_DIR}/" in key
        and key.endswith(".spz")
        and ".." not in key
        and _count(payload.get("step"))
        and _count(payload.get("count"))
        and (payload.get("total") is None or _count(payload.get("total")))
        and (payload.get("up") is None or _floats(payload.get("up"), 3))
    )


_TAGS: tuple[tuple[str, str, Callable[[Mapping[str, Any]], bool]], ...] = (
    ("cameras", CAMERAS_TAG, _valid_cameras),
    ("splat", SPLAT_TAG, _valid_splat),
)


def _decode(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def parse_line(line: str, following: Sequence[str] = ()) -> tuple[str, dict[str, Any]] | None:
    """(`cameras` or `splat`, payload) for a live line, or None.

    A provider's log service may split one long line across entries, which the tail
    turns into consecutive lines; when a tagged line does not parse by itself, the lines
    after it are joined on until it does (or `MAX_LINE` is reached).
    """
    for kind, tag, valid in _TAGS:
        at = line.find(tag)
        if at < 0:
            continue
        text = line[at + len(tag) :]
        payload = _decode(text)
        for extra in following:
            if payload is not None or len(text) > MAX_LINE:
                break
            text += extra
            payload = _decode(text)
        if payload is not None and len(text) <= MAX_LINE and valid(payload):
            return kind, payload
        return None
    return None


def latest(text: str) -> dict[str, dict[str, Any]]:
    """The newest valid line of each kind in a log: `{"cameras": ..., "splat": ...}`."""
    found: dict[str, dict[str, Any]] = {}
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if "live-" not in line:
            continue
        parsed = parse_line(line, lines[index + 1 : index + 8])
        if parsed is not None and parsed[0] not in found:
            found[parsed[0]] = parsed[1]
            if len(found) == len(_TAGS):
                break
    return found


def decode_cameras(
    payload: Mapping[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(centres, forward, up) back out of a `live-cameras` payload. For tests and tools."""
    raw = np.frombuffer(base64.b64decode(payload["cameras"]), dtype=np.uint8).reshape(-1, 12)
    q = raw[:, :6].copy().view("<i2").astype(np.float64)
    d = raw[:, 6:].copy().view("<i1").astype(np.float64) / 127.0
    origin = np.asarray(payload["origin"], dtype=np.float64)
    centres = origin + q / _INT16 * float(payload["scale"])
    return centres, d[:, :3], d[:, 3:]

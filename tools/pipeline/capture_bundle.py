"""Hex capture v1: strict archive parsing and ARKit camera-to-COLMAP conversion.

No archive member is extracted with tarfile.extract: links, paths, duplicates,
undeclared payloads, expanded sizes and malformed frame identities fail closed.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import video

F64 = npt.NDArray[np.float64]
NAME = re.compile(r"[A-Za-z0-9_.-]{1,99}\Z")
# Resource bounds, not quality thresholds. No subsampling inside these bounds.
MAX_BYTES = 128 * 1024**3
MAX_MANIFEST = 64 * 1024**2
MAX_FRAMES = 1_000_000
MAX_LINE = 16384
MAX_CAPTURE_INFO = 64 * 1024
MAX_DIAGNOSTIC_NAMES = 10
# Input precision bounds for float32 ARKit poses, not reconstruction quality gates.
# See docs/CAPTURE_BUNDLE.md for the drift envelope and synthetic measurements.
ROTATION_ERROR_MAX = 1e-3
HOMOGENEOUS_ERROR_MAX = 1e-6


@dataclass(frozen=True)
class Bundle:
    directory: Path
    manifest: dict[str, Any]
    frames: list[dict[str, Any]]
    locations: list[dict[str, Any]]
    location: dict[str, object] | None


def safe_name(name: object) -> str:
    if not isinstance(name, str) or not NAME.fullmatch(name) or name in (".", ".."):
        raise ValueError("bundle names must be flat ASCII filenames")
    return name


def integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def matrix(value: object, size: int) -> F64:
    if not isinstance(value, list) or len(value) != size * size:
        raise ValueError("incorrect matrix dimensions")
    if any(type(v) not in (int, float) for v in value):
        raise ValueError("matrix entries must be numbers")
    result = np.asarray(value, dtype=np.float64).reshape(size, size)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite matrix")
    return result


def world_to_camera(transform: F64) -> tuple[F64, F64]:
    """Invert the nearest proper rotation; keep ARKit's metric Y-up world and centre.

    Raw observations stay in frames_meta; both seeding and fixed-pose checks call this
    conversion so that COLMAP always sees the same corrected, orthonormal rotation.
    """
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("camera transform must be a proper rigid transform; error=nonfinite/shape")
    rotation, centre = transform[:3, :3], transform[:3, 3]
    with np.errstate(over="ignore", invalid="ignore"):
        error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant = float(np.linalg.det(rotation))
    row_error = float(np.max(np.abs(transform[3] - [0, 0, 0, 1])))
    if (
        not math.isfinite(error)
        or not math.isfinite(determinant)
        or error > ROTATION_ERROR_MAX
        or determinant <= 0
        or row_error > HOMOGENEOUS_ERROR_MAX
    ):
        raise ValueError(
            "camera transform must be a proper rigid transform; "
            f"orthonormality error={error:.9g}, det={determinant:.9g}, "
            f"last-row error={row_error:.9g}"
        )
    u, _, vt = np.linalg.svd(rotation)
    # Polar/SVD projection onto SO(3), explicitly enforcing det +1.
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    corrected = u @ vt
    converted = np.diag([1.0, -1.0, -1.0]) @ corrected.T
    return converted, -converted @ centre


def camera_to_world(rotation: F64, translation: F64) -> F64:
    result = np.eye(4)
    result[:3, :3] = rotation.T @ np.diag([1.0, -1.0, -1.0])
    result[:3, 3] = -rotation.T @ translation
    return result


def quaternion(rotation: F64) -> F64:
    """Hamilton w,x,y,z via the symmetric eigenproblem, stable also at 180 degrees."""
    r = rotation
    k = (
        np.array(
            [
                [
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[0, 1] + r[1, 0],
                    r[0, 2] + r[2, 0],
                    r[2, 1] - r[1, 2],
                ],
                [
                    r[0, 1] + r[1, 0],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[1, 2] + r[2, 1],
                    r[0, 2] - r[2, 0],
                ],
                [
                    r[0, 2] + r[2, 0],
                    r[1, 2] + r[2, 1],
                    r[2, 2] - r[0, 0] - r[1, 1],
                    r[1, 0] - r[0, 1],
                ],
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
            ]
        )
        / 3
    )
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def validate_frame(frame: dict[str, Any], index: int, previous: float) -> float:
    if integer(frame.get("index"), "index") != index:
        raise ValueError("frame indices must be consecutive from zero")
    timestamp = frame.get("timestamp")
    if (
        not isinstance(timestamp, int | float)
        or isinstance(timestamp, bool)
        or not math.isfinite(timestamp)
        or timestamp <= previous
        or (index == 0 and timestamp != 0)
    ):
        raise ValueError("timestamps must start at zero and strictly increase")
    try:
        transform = matrix(frame.get("transform"), 4)
        if frame.get("trackingState") == "normal":
            world_to_camera(transform)
    except ValueError as error:
        if frame.get("trackingState") != "normal":
            raise ValueError(f"frame {index}: {error}") from error
        detail = str(error).removeprefix("camera transform must be a proper rigid transform; ")
        raise ValueError(
            f"camera transform must be a proper rigid transform; frame {index}; error={detail}"
        ) from error
    k = matrix(frame.get("intrinsics"), 3)
    if (
        k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.array_equal(k[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1])
    ):
        raise ValueError("intrinsics must be an unskewed pinhole matrix")
    integer(frame.get("imageWidth"), "imageWidth", minimum=1)
    integer(frame.get("imageHeight"), "imageHeight", minimum=1)
    if not isinstance(frame.get("trackingState"), str):
        raise ValueError("missing tracking state")
    validate_exposure(frame, index)
    return float(timestamp)


def finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def validate_exposure(frame: dict[str, Any], index: int) -> None:
    """Optional per-frame camera exposure (app 2026-11+); older bundles omit every field."""
    duration = frame.get("exposureDurationS")
    if duration is not None and not (finite_number(duration) and 0 < duration <= 1):
        raise ValueError(f"frame {index}: exposureDurationS must be in (0, 1] seconds")
    offset = frame.get("exposureOffsetEV")
    if offset is not None and not finite_number(offset):
        raise ValueError(f"frame {index}: exposureOffsetEV must be a finite number")
    iso = frame.get("iso")
    if iso is not None and not (finite_number(iso) and iso > 0):
        raise ValueError(f"frame {index}: iso must be positive")
    gains = frame.get("whiteBalanceGains")
    if gains is not None and not (
        isinstance(gains, list)
        and len(gains) == 3
        and all(finite_number(g) and g > 0 for g in gains)
    ):
        raise ValueError(f"frame {index}: whiteBalanceGains must be three positive numbers")


VIDEO_FORMAT_KINDS = {"default", "4k"}
#: Why the app's 4K probe fell back: no depth within its budget, or depth of another aspect.
FALLBACK_REASONS = {"noDepth", "depthAspect"}
EXPOSURE_MODES = {"auto", "locked", "custom"}
WHITE_BALANCE_MODES = {"auto", "locked"}


def validate_capture_settings(manifest: dict[str, Any], first: dict[str, Any]) -> None:
    """Optional manifest records of the ARKit video format and the exposure lock."""
    video_format = manifest.get("videoFormat")
    if video_format is not None:
        if not isinstance(video_format, dict) or video_format.get("kind") not in VIDEO_FORMAT_KINDS:
            raise ValueError("videoFormat.kind must be one of " + str(sorted(VIDEO_FORMAT_KINDS)))
        size = (
            integer(video_format.get("width"), "videoFormat.width", minimum=1),
            integer(video_format.get("height"), "videoFormat.height", minimum=1),
        )
        integer(video_format.get("framesPerSecond"), "videoFormat.framesPerSecond", minimum=1)
        # ARKit reports the format landscape, as capturedImage is stored.
        if size != (first["imageWidth"], first["imageHeight"]):
            raise ValueError("videoFormat size differs from the recorded frames")
        fallback = video_format.get("fallback")
        if fallback is not None:
            if not isinstance(fallback, dict) or fallback.get("reason") not in FALLBACK_REASONS:
                raise ValueError(
                    "videoFormat.fallback.reason must be " + str(sorted(FALLBACK_REASONS))
                )
            integer(fallback.get("imageWidth"), "videoFormat.fallback.imageWidth", minimum=1)
            integer(fallback.get("imageHeight"), "videoFormat.fallback.imageHeight", minimum=1)
            for key in ("depthWidth", "depthHeight"):
                if fallback.get(key) is not None:
                    integer(fallback[key], f"videoFormat.fallback.{key}", minimum=1)
    control = manifest.get("exposureControl")
    if control is not None:
        if (
            not isinstance(control, dict)
            or control.get("mode") not in EXPOSURE_MODES
            or control.get("whiteBalance") not in WHITE_BALANCE_MODES
        ):
            raise ValueError("exposureControl needs a known mode and whiteBalance")
        for key, upper in (("durationS", 1.0), ("iso", math.inf)):
            value = control.get(key)
            if value is not None and not (finite_number(value) and 0 < value <= upper):
                raise ValueError(f"exposureControl.{key} must be positive")
        shortfall = control.get("shortfallEV")
        if shortfall is not None and not (finite_number(shortfall) and shortfall >= 0):
            raise ValueError("exposureControl.shortfallEV must be a non-negative number")


#: The largest frame the bundle normalizer still extracts losslessly: the 4:3 1920x1440
#: ARKit default. 720 such PNGs are ~4 GB; the same count at 3840x2160 is ~11-13 GB, beyond
#: what the 20 GB capture worker can hold beside the archive. Larger frames are written as
#: JPEG at ffmpeg -q:v 2, the quality the ordinary video path already trains from.
MAX_PNG_PIXELS = 1920 * 1440
JPEG_QUALITY = 2


def frame_suffix(width: int, height: int) -> str:
    """The extension normalize gives a bundle's frames at this size."""
    return ".png" if width * height <= MAX_PNG_PIXELS else ".jpg"


def frame_name(index: int, suffix: str = ".png") -> str:
    return f"frame_{index:08d}{suffix}"


def read_locations(
    directory: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, object] | None]:
    """Retain all fixes; locate from the first fix with the lowest positive uncertainty."""
    locations: list[dict[str, Any]] = []
    best = None
    if manifest.get("locationLog") is not None:
        if manifest["locationLog"] != "location.jsonl":
            raise ValueError("v1 locationLog must be location.jsonl")
        with (directory / "location.jsonl").open() as log:
            while line := log.readline(MAX_LINE + 1):
                if len(line) > MAX_LINE or len(locations) >= MAX_FRAMES:
                    raise ValueError("location log exceeds limits")
                fix = json.loads(line)
                fields = (
                    "timestamp",
                    "lat",
                    "lon",
                    "altitudeM",
                    "horizontalAccuracyM",
                    "verticalAccuracyM",
                )
                if (
                    not isinstance(fix, dict)
                    or any(
                        type(fix.get(key)) not in (int, float) or not math.isfinite(fix[key])
                        for key in fields
                    )
                    or not (-90 <= fix["lat"] <= 90 and -180 <= fix["lon"] <= 180)
                ):
                    raise ValueError("invalid bundle location fix")
                locations.append(fix)
                if fix["horizontalAccuracyM"] > 0 and (
                    best is None or fix["horizontalAccuracyM"] < best["horizontalAccuracyM"]
                ):
                    best = fix
    if manifest.get("location") != best:
        raise ValueError("manifest location does not match the location log summary")
    location = (
        None
        if best is None
        else video.Location(
            lat=best["lat"],
            lon=best["lon"],
            altitude_m=best["altitudeM"] if best["verticalAccuracyM"] >= 0 else None,
            source="capture-bundle-corelocation",
            accuracy_m=best["horizontalAccuracyM"],
        ).to_dict()
    )
    return locations, location


def validate_live_splat(path: Path) -> None:
    """Bounded validation of the optional preview export; never initializes training."""
    properties = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    with path.open("rb") as stream:
        lines = []
        for _ in range(32):
            line = stream.readline(256)
            lines.append(line)
            if line == b"end_header\n":
                break
        header = b"".join(lines).decode("ascii")
        expected = ["ply", "format binary_little_endian 1.0"]
        if header.splitlines()[:2] != expected or lines[-1] != b"end_header\n":
            raise ValueError("invalid live splat header")
        fields = [line for line in header.splitlines()[2:-1] if not line.startswith("comment ")]
        if not fields or not fields[0].startswith("element vertex "):
            raise ValueError("invalid live splat vertex count")
        count = int(fields[0].removeprefix("element vertex "))
        if not 0 <= count <= 32768 or fields[1:] != [
            "property float " + name for name in properties
        ]:
            raise ValueError("invalid live splat properties/count")
        payload = stream.read(32768 * 17 * 4 + 1)
        if (
            len(payload) != count * 17 * 4
            or not np.isfinite(np.frombuffer(payload, dtype="<f4")).all()
        ):
            raise ValueError("invalid live splat payload")


def diagnostic_names(names: set[str]) -> str:
    ordered = sorted(names)
    result = json.dumps(ordered[:MAX_DIAGNOSTIC_NAMES])
    remaining = len(ordered) - MAX_DIAGNOSTIC_NAMES
    return result + (f" (+{remaining} more)" if remaining > 0 else "")


def validate_capture_info(path: Path) -> None:
    """App metadata is optional and opaque to reconstruction, but bounded JSON."""
    with path.open("rb") as stream:
        data = stream.read(MAX_CAPTURE_INFO + 1)
    if len(data) > MAX_CAPTURE_INFO:
        raise ValueError(f"capture-info.json exceeds {MAX_CAPTURE_INFO} byte limit")

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    try:
        value = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    except (ValueError, RecursionError) as error:
        raise ValueError("capture-info.json must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("capture-info.json must contain a JSON object")


def read_bundle(directory: Path) -> Bundle:
    manifest_path = directory / "manifest.json"
    if manifest_path.stat().st_size > MAX_MANIFEST:
        raise ValueError("manifest too large")
    manifest = json.loads(manifest_path.read_text())
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("version")) is not int
        or manifest["version"] != 1
        or manifest.get("format") != "hex-capture"
    ):
        raise ValueError("unsupported capture bundle version/format")
    if manifest.get("video") != "rgb.mov" or manifest.get("frameLog") != "frames.jsonl":
        raise ValueError("v1 requires rgb.mov and frames.jsonl")
    if not isinstance(manifest.get("device"), str) or not isinstance(
        manifest.get("startedAt"), str
    ):
        raise ValueError("missing capture provenance")
    started = datetime.fromisoformat(manifest["startedAt"])
    if started.utcoffset() != timedelta(0):
        raise ValueError("startedAt must be an ISO 8601 UTC timestamp")
    count = integer(manifest.get("frameCount"), "frameCount", minimum=1)
    if count > MAX_FRAMES:
        raise ValueError("too many frames")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("missing file inventory")
    inventory: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid file entry")
        name = safe_name(entry.get("path"))
        if name in inventory or name == "manifest.json":
            raise ValueError("duplicate or reserved filename")
        size = integer(entry.get("bytes"), "bytes")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"inventoried file missing or not a regular file: {name}")
        actual_size = path.stat().st_size
        if actual_size != size:
            raise ValueError(
                f"file size/inventory mismatch: {name}; actual={actual_size}, inventoried={size}"
            )
        inventory[name] = size
    actual = {p.name for p in directory.iterdir()}
    if sum(inventory.values()) > MAX_BYTES:
        raise ValueError("bundle too large")
    required = {"rgb.mov", "frames.jsonl"}
    if missing_required := required - set(inventory):
        raise ValueError(
            f"referenced files missing from inventory={diagnostic_names(missing_required)}"
        )
    if empty_required := {name for name in required if inventory[name] == 0}:
        raise ValueError(f"required bundle files empty={diagnostic_names(empty_required)}")
    frames: list[dict[str, Any]] = []
    used = {"rgb.mov", "frames.jsonl"}
    # Known optional app metadata, opaque to reconstruction. Existing inventory
    # checks still enforce a regular file, exact size and the total byte bound.
    if "capture-info.json" in inventory:
        validate_capture_info(directory / "capture-info.json")
        used.add("capture-info.json")
    if manifest.get("locationLog") is not None:
        if manifest["locationLog"] != "location.jsonl":
            raise ValueError("v1 locationLog must be location.jsonl")
        used.add("location.jsonl")
    if "liveSplat" in manifest:
        if manifest["liveSplat"] != "livesplat.ply":
            raise ValueError("liveSplat must reference inventoried livesplat.ply")
        used.add("livesplat.ply")
    counts = {"depth": 0, "smoothedDepth": 0}
    previous = -1.0
    with (directory / "frames.jsonl").open() as log:
        while line := log.readline(MAX_LINE + 1):
            if len(line) > MAX_LINE or len(frames) >= MAX_FRAMES:
                raise ValueError("frame log exceeds resource limits")
            frame = json.loads(line)
            if not isinstance(frame, dict):
                raise ValueError("frame must be an object")
            previous = validate_frame(frame, len(frames), previous)
            if frames and (frame["imageWidth"], frame["imageHeight"]) != (
                frames[0]["imageWidth"],
                frames[0]["imageHeight"],
            ):
                raise ValueError("video dimensions changed")
            for key in counts:
                depth = frame.get(key)
                if depth is None:
                    continue
                if not isinstance(depth, dict):
                    raise ValueError("invalid depth reference")
                pixels = integer(depth.get("width"), "depth width", minimum=1) * integer(
                    depth.get("height"), "depth height", minimum=1
                )
                for field, size in (("file", pixels * 2), ("confidence", pixels)):
                    name = safe_name(depth.get(field))
                    if name in used:
                        raise ValueError(f"depth reference reused: {name}")
                    used.add(name)
                    # Accumulate missing names for the aggregate diagnostic below.
                    if name in inventory and inventory[name] != size:
                        raise ValueError(
                            f"depth reference wrong size: {name}; "
                            f"expected={size}, inventoried={inventory[name]}"
                        )
                counts[key] += 1
            frames.append(frame)
    if frames:
        validate_capture_settings(manifest, frames[0])
    unreferenced = set(inventory) - used
    missing = used - set(inventory)
    if len(frames) != count or unreferenced or missing:
        raise ValueError(
            f"capture bundle mismatch: frames.jsonl line count={len(frames)}; "
            f"manifest frameCount={count}; "
            f"inventory files not referenced={diagnostic_names(unreferenced)}; "
            f"referenced files missing from inventory={diagnostic_names(missing)}"
        )
    if actual != set(inventory) | {"manifest.json"}:
        raise ValueError(
            "undeclared bundle files="
            + diagnostic_names(actual - set(inventory) - {"manifest.json"})
        )
    # Validate optional payload contents only after their references are reconciled.
    locations, location = read_locations(directory, manifest)
    if "liveSplat" in manifest:
        validate_live_splat(directory / "livesplat.ply")
    for key, actual_count in counts.items():
        if integer(manifest.get(key + "FrameCount"), key + "FrameCount") != actual_count:
            raise ValueError("depth count mismatch")
    return Bundle(directory, manifest, frames, locations, location)


def unpack(archive: Path, destination: Path) -> Bundle:
    destination.mkdir(parents=True, exist_ok=False)
    try:
        total = 0
        seen: set[str] = set()
        with tarfile.open(archive, mode="r|") as tar:
            for member in tar:
                name = safe_name(member.name)
                if not member.isfile() or member.issparse() or name in seen:
                    raise ValueError("bundle contains a link, duplicate or special member")
                seen.add(name)
                total += member.size
                if total > MAX_BYTES or len(seen) > 4 * MAX_FRAMES + 4:
                    raise ValueError("bundle exceeds resource limits")
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError("missing archive member")
                with source, (destination / name).open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        return read_bundle(destination)
    except Exception:
        shutil.rmtree(destination)
        raise

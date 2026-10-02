"""The Minnetonka tree: a real photogrammetric tree, from its published photos to a splat.

Living Survey milestone M0 (docs/LIVING_WORLD.md section 9). This module is the pure part:
which files to fetch and how to check them, what the dataset's own metadata says about each
photo, how a COLMAP model is filtered and written, and -- the part that decides whether a
GPU is worth paying for -- whether a set of camera poses agrees with the photos it claims
to describe. `infra/modal/minnetonka.py` is the driver that does the I/O; the post-train
half (level, scale, isolate, rig, site.json) is `tools/captures/real_tree.py`.

**The dataset** is "Single Tree -- High-Density Photogrammetry Dataset", Matthew Guertin,
2020, CC BY 4.0 (LICENSE, CITATION.cff and the Hugging Face card all say so): 812 JPEGs of
one tree in Minnetonka, MN, 5464x3640, DJI Mavic 2 Pro. Docs, manifests and RealityScan's
solved cameras are on GitHub; the images and `colmap/points3D.txt` on Hugging Face. Both
are pinned below to the revision that was read on 2026-09-28, and every file is checked
against a sha256 before it is used: the images against the LFS object id the Hub reports
for that revision (the same digests as the GitHub repo's `manifest/checksums.sha256`), the
three COLMAP text files against digests taken here.

**The published poses do not describe the photos they are named after.** Measured
2026-09-28 against the DJI metadata in every `The_Tree` JPEG (read from the first
192 KiB of each file, no full download), `The_Tree` only, 659 posed:

* solved pitch within 12 deg of the gimbal's pitch for 318 of 659 (a correct pose is
  within a degree or two of it);
* camera height against the barometric height above take-off: correlation 0.40 overall,
  0.90 on those 318 and -0.14 on the other 341;
* gimbal heading against solved heading: no consistent offset at all (median residual
  67 deg even on the 318);
* consecutive photos (4-8 s apart) are 5.8 model units apart at the median against 19.7
  for random pairs -- so the order is partly there, but 301 of 658 steps jump more than
  8 units (~3 m);
* two photos looked at by eye: `The_Tree-86.jpg` is taken from above the crown looking
  down, and its pose (also in `poses/xmp/The_Tree-86.xmp`) is at the lowest tier looking
  up 8 deg; `The_Tree-542.jpg` is a low shot looking up at the trunk, and its pose is at
  the top of the orbit looking down 43 deg.

The mismatch is in the XMP sidecars' names, upstream of the COLMAP conversion (the
converter pairs by file name and cannot see it; its in-frame check passes for any camera
that looks at the middle of an orbit). So the default pose source here is **COLMAP 4.2 on
the photos themselves** -- the repository's own `pose` stage -- and the published model
is kept only behind `pose_gate`, which runs on the metadata before any GPU is paid for and
refuses it by name. The four `suspect_cameras.txt` entries are dropped either way.

**Scale and gravity come from the drone, not from GPS.** The dataset's author tried GPS
and found no usable scale (the whole capture spans ~17 m of GPS footprint, and consumer
fixes carry metres of error). Every photo also carries `RelativeAltitude` (barometric
height above take-off, 0.1 m steps) and the gimbal's pitch and yaw, which the gimbal holds
against its own IMU and compass. Those three give, from any pose set:

* **up**: the direction `u` for which `asin(view . u)` matches the gimbal pitch -- linear
  least squares on `view . u = sin(pitch)`, the pitch of every photo being a measurement
  of gravity in that photo's frame;
* **scale**: metres per model unit, the slope of barometric height against `u . centre`
  (one offset per flight, so a second take-off does not bend the fit);
* **heading**: the rotation about `u` that best turns solved headings into compass ones.

Each is checked against a second measurement before it is trusted: `u` against the
direction the barometric fit finds on its own, the pitch and height residuals against a
bar a correct model clears easily, and the heading residuals against a compass. That is
`fit_frame`, and its verdict is the gate.
"""

from __future__ import annotations

import io
import json
import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import resolution
import sfm
from recipe import load_recipe

F64 = npt.NDArray[np.float64]

__all__ = [
    "COLMAP_FILES",
    "GROUPS",
    "HF_REPO",
    "HF_REVISION",
    "SUSPECT_CAMERAS",
    "DroneMeta",
    "FrameFit",
    "PosedView",
    "RemoteFile",
    "TextModel",
    "fit_frame",
    "frame_name",
    "plan_download",
    "read_drone_meta",
    "read_text_model",
]

# --- the dataset, pinned ------------------------------------------------------------------

HF_REPO = "Matt1up/tree-minnetonka-photogrammetry"
#: The Hub revision read on 2026-09-28 (`GET /api/datasets/<repo>` -> `sha`). Every URL
#: below names it, so a later push to the dataset cannot change what a run downloads.
HF_REVISION = "5f9de5e4a1be429b192a928cf1359c066dadb4b3"
GITHUB_REPO = "https://github.com/Matt1Up/tree-photogrammetry-dataset"
#: The docs repository's HEAD on 2026-09-28; its `poses/colmap/{cameras,images}.txt` are
#: byte-identical to the Hub's `colmap/` copies at `HF_REVISION` (same sha256 below).
GITHUB_COMMIT = "942f050f7bb7fed530e61c80b3c72e5674986b48"

ATTRIBUTION = "Matthew Guertin"
ATTRIBUTION_URL = "https://mattguertin.com"
LICENSE_NAME = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
#: The tree, as the dataset's README gives it (44.944 N, -93.426 W) to the precision
#: docs/CAPTURES.md recorded.
LATITUDE = 44.944565
LONGITUDE = -93.425903


@dataclass(frozen=True)
class Group:
    """One capture tier: how many images, how many bytes, and when."""

    images: int
    bytes: int
    date: str


#: From the Hub's tree listing at `HF_REVISION` (sizes are the LFS objects').
GROUPS: dict[str, Group] = {
    "The_Tree": Group(images=659, bytes=13_961_267_697, date="2020-07-20"),
    "Original_low": Group(images=85, bytes=659_282_963, date="2020-07-18"),
    "Original_mid": Group(images=68, bytes=481_341_460, date="2020-07-18"),
}

#: `colmap/` on the Hub at `HF_REVISION`: name -> (bytes, sha256), hashed 2026-09-28.
COLMAP_FILES: dict[str, tuple[int, str]] = {
    "cameras.txt": (56_476, "94f6a342eccd25ec453263a65b3897e6bc2586d884617cb1cf42a6e4fa6d232b"),
    "images.txt": (91_457, "8cb7c7ff01b542638044d1dc7d105b0ed1fe59a27120fd577301eaecd22d16be"),
    "points3D.txt": (
        58_072_943,
        "f38ceae8edc4953aec869e20b8210ae2dbe1c58780ece6cfa41c846541c10288",
    ),
}

#: `poses/suspect_cameras.txt` at `GITHUB_COMMIT`: solved at ~9.2 mm-equivalent focal,
#: which the Mavic 2 Pro's fixed 28 mm-equivalent lens cannot be. Dropped before training.
SUSPECT_CAMERAS: tuple[str, ...] = (
    "The_Tree-88.jpg",
    "The_Tree-137.jpg",
    "The_Tree-223.jpg",
    "The_Tree-235.jpg",
)


def tree_url(path: str) -> str:
    """The Hub API's listing of `path` at the pinned revision."""
    return f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REVISION}/{path}"


def resolve_url(path: str) -> str:
    """A file's bytes at the pinned revision."""
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{path}"


_NEXT = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def next_page(link_header: str | None) -> str | None:
    """The Hub paginates listings with an RFC 8288 `Link: <url>; rel="next"` header."""
    if not link_header:
        return None
    found = _NEXT.search(link_header)
    return found.group(1) if found else None


# --- which files ------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return resolve_url(self.path)


def parse_listing(entries: Iterable[Mapping[str, Any]]) -> list[RemoteFile]:
    """Files out of the Hub's `tree` listing. An image without an LFS sha256 is refused:
    the digest is what every download is checked against."""
    out: list[RemoteFile] = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs")
        oid = lfs.get("oid") if isinstance(lfs, Mapping) else None
        if not isinstance(oid, str) or not re.fullmatch(r"[0-9a-f]{64}", oid):
            raise ValueError(f"{entry.get('path')!r} has no LFS sha256 in the Hub listing")
        out.append(RemoteFile(path=str(entry["path"]), size=int(entry["size"]), sha256=oid))
    return out


_FRAME = re.compile(r"^(?P<group>[A-Za-z_]+)-(?P<number>\d+)\.jpg$")


def group_of(name: str) -> str:
    found = _FRAME.match(name)
    if not found:
        raise ValueError(f"{name!r} is not a <group>-<number>.jpg name")
    return found.group("group")


def frame_number(name: str) -> int:
    found = _FRAME.match(name)
    if not found:
        raise ValueError(f"{name!r} is not a <group>-<number>.jpg name")
    return int(found.group("number"))


def frame_name(name: str) -> str:
    """`The_Tree-7.jpg` -> `The_Tree-0007.jpg`.

    Zero-padded so that sorting by name is sorting by capture order: the numbers follow
    the photos' own timestamps, and COLMAP's sequential matcher and gsplat's held-out split
    both go by name. `The_Tree-10` sorts before `The_Tree-2` otherwise.
    """
    return f"{group_of(name)}-{frame_number(name):04d}.jpg"


def plan_download(files: Sequence[RemoteFile], groups: Sequence[str]) -> list[RemoteFile]:
    """The images of `groups`, in capture order, after checking the listing is the pinned one.

    A group with a different count or byte total than `GROUPS` records is refused: that is
    a different dataset from the one these numbers, the gate's calibration and the cost
    estimate were read off.
    """
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown or not groups:
        raise ValueError(f"groups must be some of {sorted(GROUPS)}, not {list(groups)}")
    chosen: list[RemoteFile] = []
    for group in groups:
        members = [
            file
            for file in files
            if file.path.startswith("images/") and _FRAME.match(file.name)
            if group_of(file.name) == group
        ]
        expected = GROUPS[group]
        total = sum(file.size for file in members)
        if len(members) != expected.images or total != expected.bytes:
            raise ValueError(
                f"{group}: the listing has {len(members)} images, {total:,} bytes; the pinned "
                f"revision has {expected.images}, {expected.bytes:,}"
            )
        chosen.extend(members)
    return sorted(chosen, key=lambda file: (group_of(file.name), frame_number(file.name)))


def planned_bytes(files: Sequence[RemoteFile]) -> int:
    return sum(file.size for file in files)


# --- what each photo says about itself --------------------------------------------------


@dataclass(frozen=True)
class DroneMeta:
    """What the DJI XMP packet in one photo records. Every field is optional because a
    photo that lacks one is still a photo; `fit_frame` uses only the ones that have it."""

    relative_altitude_m: float | None = None
    absolute_altitude_m: float | None = None
    gimbal_pitch_deg: float | None = None
    gimbal_yaw_deg: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    taken: str | None = None

    def seconds(self) -> float | None:
        if not self.taken:
            return None
        try:
            return datetime.fromisoformat(self.taken[:19]).timestamp()
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relativeAltitudeM": self.relative_altitude_m,
            "absoluteAltitudeM": self.absolute_altitude_m,
            "gimbalPitchDeg": self.gimbal_pitch_deg,
            "gimbalYawDeg": self.gimbal_yaw_deg,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "taken": self.taken,
        }

    @staticmethod
    def from_dict(value: Mapping[str, Any]) -> DroneMeta:
        def number(key: str) -> float | None:
            raw = value.get(key)
            return (
                float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
            )

        taken = value.get("taken")
        return DroneMeta(
            relative_altitude_m=number("relativeAltitudeM"),
            absolute_altitude_m=number("absoluteAltitudeM"),
            gimbal_pitch_deg=number("gimbalPitchDeg"),
            gimbal_yaw_deg=number("gimbalYawDeg"),
            latitude=number("latitude"),
            longitude=number("longitude"),
            taken=taken if isinstance(taken, str) else None,
        )


#: How much of the start of a JPEG to read for its XMP. The DJI packet sits after the
#: EXIF block and a Lightroom thumbnail; in this dataset it starts as late as ~49 KiB.
META_BYTES = 192 * 1024


def _xmp_value(text: str, prefix: str, name: str) -> str | None:
    """`prefix:name="v"` (attribute form, what the drone writes) or
    `<prefix:name>v</prefix:name>` (element form, what the repaired low/mid tiers carry)."""
    found = re.search(rf'{prefix}:{name}="([^"]*)"', text) or re.search(
        rf"<{prefix}:{name}>([^<]*)</{prefix}:{name}>", text
    )
    return found.group(1).strip() if found else None


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_drone_meta(data: bytes) -> DroneMeta:
    """The DJI fields out of a JPEG's XMP. Reads text, not structure: the packet is ASCII
    inside an APP1 segment, and a regex over it is what every DJI metadata tool does."""
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta>", start + 1) if start >= 0 else -1
    if start < 0:
        return DroneMeta()
    text = data[start : end if end > 0 else len(data)].decode("utf-8", errors="replace")
    taken = _xmp_value(text, "xmp", "CreateDate") or _xmp_value(text, "exif", "DateTimeOriginal")
    return DroneMeta(
        relative_altitude_m=_float(_xmp_value(text, "drone-dji", "RelativeAltitude")),
        absolute_altitude_m=_float(_xmp_value(text, "drone-dji", "AbsoluteAltitude")),
        gimbal_pitch_deg=_float(_xmp_value(text, "drone-dji", "GimbalPitchDegree")),
        gimbal_yaw_deg=_float(_xmp_value(text, "drone-dji", "GimbalYawDegree")),
        latitude=_float(_xmp_value(text, "drone-dji", "GpsLatitude")),
        longitude=_float(_xmp_value(text, "drone-dji", "GpsLongitude")),
        taken=taken,
    )


def shrink_jpeg(data: bytes, target: Path, max_side: int) -> tuple[int, int]:
    """Write `data` to `target` with its long side at most `max_side`; returns the size.

    Lanczos, quality 95, the same choices as `training._shrink`, and the stored pixels as
    they are (no EXIF rotation): these are landscape drone frames with orientation 1, and
    the model was posed on the stored pixels. The XMP is not carried over -- it has been
    read by then (`read_drone_meta`) and is kept beside the frames in `meta.json`.
    """
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        picture = image.convert("RGB")
        if size != (width, height):
            picture = picture.resize(size, Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        picture.save(target, format="JPEG", quality=95)
    return size


# --- the published COLMAP model ------------------------------------------------------------


@dataclass(frozen=True)
class TextCamera:
    id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class TextImage:
    id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str


@dataclass(frozen=True)
class TextModel:
    cameras: dict[int, TextCamera]
    images: tuple[TextImage, ...]
    points: sfm.Points3D


_MODEL_IDS = {name: model_id for model_id, (name, _) in sfm.CAMERA_MODELS.items()}


def _data_lines(path: Path) -> list[str]:
    return [
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    ]


def read_text_model(directory: Path) -> TextModel:
    """`cameras.txt`, `images.txt`, `points3D.txt` as COLMAP writes them.

    Tracks are not carried: this dataset's `images.txt` has no 2D points (its POINTS2D
    lines are empty), so every track is empty too, and gsplat's parser reads tracks only
    for the depth loss. An image line is recognised by shape -- ten fields, the last a
    `.jpg` -- rather than by position, so an empty POINTS2D line cannot shift the pairing.
    """
    cameras: dict[int, TextCamera] = {}
    for line in _data_lines(directory / "cameras.txt"):
        parts = line.split()
        if not parts:
            continue
        model = parts[1]
        if model not in _MODEL_IDS:
            raise ValueError(f"camera model {model} is not one sfm.CAMERA_MODELS knows")
        cameras[int(parts[0])] = TextCamera(
            id=int(parts[0]),
            model=model,
            width=int(parts[2]),
            height=int(parts[3]),
            params=tuple(float(v) for v in parts[4:]),
        )
    images: list[TextImage] = []
    for line in _data_lines(directory / "images.txt"):
        parts = line.split()
        if len(parts) != 10 or not parts[9].lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        values = [float(v) for v in parts[1:8]]
        images.append(
            TextImage(
                id=int(parts[0]),
                qvec=(values[0], values[1], values[2], values[3]),
                tvec=(values[4], values[5], values[6]),
                camera_id=int(parts[8]),
                name=parts[9],
            )
        )
    ids: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    error: list[float] = []
    with (directory / "points3D.txt").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.split(maxsplit=8)
            if len(parts) < 8:
                continue
            ids.append(int(parts[0]))
            xyz.append((float(parts[1]), float(parts[2]), float(parts[3])))
            rgb.append((int(parts[4]), int(parts[5]), int(parts[6])))
            error.append(float(parts[7]))
    count = len(ids)
    points = sfm.Points3D(
        ids=np.asarray(ids, dtype=np.uint64),
        xyz=np.asarray(xyz, dtype=np.float64).reshape(count, 3),
        rgb=np.asarray(rgb, dtype=np.uint8).reshape(count, 3),
        error=np.asarray(error, dtype=np.float64),
        track=np.zeros((0, 2), dtype=np.uint32),
        track_offsets=np.zeros(count + 1, dtype=np.int64),
    )
    return TextModel(cameras=cameras, images=tuple(images), points=points)


def select_images(model: TextModel, rename: Mapping[str, str]) -> TextModel:
    """The images named in `rename` (old name -> new name), renamed; cameras no image
    uses any more dropped. Everything else -- the suspects, the groups not downloaded, the
    photos that failed to fetch -- simply is not in `rename`."""
    images = tuple(
        TextImage(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=rename[image.name],
        )
        for image in model.images
        if image.name in rename
    )
    used = {image.camera_id for image in images}
    cameras = {cid: camera for cid, camera in model.cameras.items() if cid in used}
    return TextModel(cameras=cameras, images=images, points=model.points)


def write_binary_model(model: TextModel, directory: Path) -> None:
    """`cameras.bin`, `images.bin` and `points3D.bin`, in the layout `sfm.read_model` and
    gsplat's parser read. Images carry no 2D points (see `read_text_model`)."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(model.cameras)))
        for camera in sorted(model.cameras.values(), key=lambda c: c.id):
            expected = len(sfm.CAMERA_MODELS[_MODEL_IDS[camera.model]][1])
            if len(camera.params) != expected:
                raise ValueError(
                    f"camera {camera.id}: {camera.model} takes {expected} params, "
                    f"not {len(camera.params)}"
                )
            handle.write(
                struct.pack(
                    "<IiQQ", camera.id, _MODEL_IDS[camera.model], camera.width, camera.height
                )
            )
            handle.write(struct.pack(f"<{expected}d", *camera.params))
    with (directory / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(model.images)))
        for image in sorted(model.images, key=lambda i: i.id):
            handle.write(
                struct.pack("<IdddddddI", image.id, *image.qvec, *image.tvec, image.camera_id)
            )
            handle.write(image.name.encode("utf-8") + b"\0")
            handle.write(struct.pack("<Q", 0))
    points = model.points
    if len(points.track):
        sfm.write_points3d(directory / "points3D.bin", points)
        return
    # Every track empty: one packed record per point, written in one go rather than
    # 1.2M `struct.pack` calls.
    record = np.dtype(
        [("id", "<u8"), ("xyz", "<f8", 3), ("rgb", "u1", 3), ("error", "<f8"), ("track", "<u8")]
    )
    rows = np.zeros(len(points), dtype=record)
    rows["id"] = points.ids
    rows["xyz"] = points.xyz
    rows["rgb"] = points.rgb
    rows["error"] = points.error
    with (directory / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        handle.write(rows.tobytes())


# --- does a pose set agree with the photos? ------------------------------------------------


@dataclass(frozen=True)
class PosedView:
    """One posed photo: its centre, where it looks, and which way its image-up points."""

    name: str
    centre: F64
    view: F64
    up: F64


def posed_views(model: sfm.Model) -> list[PosedView]:
    """COLMAP's convention: `x_cam = R x_world + t`, the camera looks down +z and its
    image +y points down, so the view is `R[2]` and image-up is `-R[1]` in the world."""
    views: list[PosedView] = []
    for image in model.images:
        rotation = image.rotation
        views.append(
            PosedView(
                name=image.name,
                centre=image.centre,
                view=rotation[2, :].copy(),
                up=-rotation[1, :].copy(),
            )
        )
    return views


#: A new flight starts when two photos are further apart in time than this: a battery
#: change or a landing, after which the barometer's zero is a new take-off.
SEGMENT_GAP_S = 120.0
#: Photos pitched steeper than this say little about heading.
HEADING_MAX_PITCH_DEG = 70.0

#: What a correct pose set clears. Gimbal pitch is levelled by the gimbal's own IMU to a
#: fraction of a degree; the barometer reports 0.1 m steps and drifts a few tenths over a
#: flight; the compass is good to a few degrees. The published model fails every one of
#: these by an order of magnitude (module docstring), so the bars are not tuned to a margin.
GATE = {
    "metaCoverageMin": 0.9,
    "pitchMedianAbsMaxDeg": 3.0,
    "pitchP90AbsMaxDeg": 6.0,
    "upAgreementMaxDeg": 3.0,
    "heightRmsMaxM": 0.6,
    "heightInlierMin": 0.9,
    "heightSpanMinM": 2.0,
    "headingMedianAbsMaxDeg": 8.0,
    "headingP90AbsMaxDeg": 20.0,
}


def _robust_lstsq(
    design: F64, target: F64, *, floor: float, rounds: int = 4
) -> tuple[F64, npt.NDArray[np.bool_]]:
    """Least squares that drops rows beyond 3 robust sigmas (and never inside `floor`)."""
    keep = np.ones(target.shape[0], dtype=bool)
    solution = np.zeros(design.shape[1])
    for _ in range(rounds):
        if keep.sum() < design.shape[1] + 1:
            break
        solution, *_ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
        residual = target - design @ solution
        sigma = 1.4826 * float(np.median(np.abs(residual[keep])))
        keep = np.abs(residual) <= max(3.0 * sigma, floor)
    return solution, keep


def look_at_point(centres: F64, views: F64) -> F64:
    """The point nearest every optical axis, least squares: what an orbit looks at."""
    system = np.zeros((3, 3))
    rhs = np.zeros(3)
    for centre, view in zip(centres, views, strict=True):
        unit = view / np.linalg.norm(view)
        projector = np.eye(3) - np.outer(unit, unit)
        system += projector
        rhs += projector @ centre
    return np.asarray(np.linalg.solve(system, rhs), dtype=np.float64)


def _wrap(degrees: F64) -> F64:
    return np.asarray((degrees + 180.0) % 360.0 - 180.0, dtype=np.float64)


def _angle_deg(a: F64, b: F64) -> float:
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def rotation_about_z(degrees_clockwise: float) -> F64:
    """Clockwise seen from above, the way a compass heading turns."""
    angle = -math.radians(degrees_clockwise)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def flights(times: Sequence[float], gap_s: float = SEGMENT_GAP_S) -> list[int]:
    """A flight index per photo (in the order given): a new one after every gap."""
    order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
    out = [0] * len(times)
    flight = 0
    previous: float | None = None
    for index in order:
        moment = float(times[int(index)])
        if previous is not None and moment - previous > gap_s:
            flight += 1
        out[int(index)] = flight
        previous = moment
    return out


@dataclass
class FrameFit:
    """The similarity from a pose set's frame into metres, east/north/up about the tree's
    foot, and how far the photos' own metadata agrees with that pose set."""

    posed: int
    with_meta: int
    metres_per_unit: float
    up: list[float]
    up_from_height: list[float]
    up_from_image: list[float]
    up_agreement_deg: float
    heading_offset_deg: float
    look_at: list[float]
    ground_offset_m: float
    matrix: list[list[float]]
    residuals: dict[str, float]
    checks: dict[str, bool]
    worst: list[dict[str, Any]] = field(default_factory=list)
    flights: int = 1

    @property
    def verdict(self) -> str:
        return "pass" if all(self.checks.values()) else "fail"

    @property
    def failed(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok]

    def apply(self, points: F64) -> F64:
        """Model-frame points into the metric frame this fit defines."""
        matrix = np.asarray(self.matrix, dtype=np.float64)
        return np.asarray(points @ matrix[:3, :3].T + matrix[:3, 3], dtype=np.float64)

    def roi(self, radius_m: float) -> dict[str, Any]:
        """The train stage's `roi` (model frame): a sphere round what the orbit looks at."""
        return {"center": list(self.look_at), "radius": radius_m / self.metres_per_unit}

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "failed": self.failed,
            "posed": self.posed,
            "withMeta": self.with_meta,
            "flights": self.flights,
            "metresPerUnit": self.metres_per_unit,
            "up": self.up,
            "upFromHeight": self.up_from_height,
            "upFromImage": self.up_from_image,
            "upAgreementDeg": self.up_agreement_deg,
            "headingOffsetDeg": self.heading_offset_deg,
            "lookAt": self.look_at,
            "groundOffsetM": self.ground_offset_m,
            "matrix": self.matrix,
            "residuals": self.residuals,
            "checks": self.checks,
            "gate": dict(GATE),
            "worst": self.worst,
            "note": (
                "matrix takes the pose set's frame to metres, x east, y north, z up, origin "
                "on the axis the cameras look at, z = 0 at the main flight's take-off "
                "height. Up from gimbal pitch, scale from barometric height, heading from "
                "the gimbal compass (experiments/minnetonka.py)."
            ),
        }


def fit_frame(views: Sequence[PosedView], metas: Mapping[str, DroneMeta]) -> FrameFit:
    """Up, scale, heading and origin for `views` from the photos' DJI metadata, and the
    checks that say whether the pose set and the photos describe the same flight.

    Raises when too few photos carry metadata to fit anything at all: that is not a
    verdict on the poses, it is a missing input.
    """
    paired = [
        (view, metas[view.name])
        for view in views
        if view.name in metas
        and metas[view.name].gimbal_pitch_deg is not None
        and metas[view.name].relative_altitude_m is not None
    ]
    if len(paired) < 12:
        raise ValueError(
            f"only {len(paired)} of {len(views)} posed photos carry gimbal pitch and "
            f"barometric height; cannot fit a frame"
        )
    centres = np.stack([view.centre for view, _ in paired])
    directions = np.stack([view.view / np.linalg.norm(view.view) for view, _ in paired])
    image_ups = np.stack([view.up for view, _ in paired])
    pitch = np.array([meta.gimbal_pitch_deg for _, meta in paired], dtype=np.float64)
    height = np.array([meta.relative_altitude_m for _, meta in paired], dtype=np.float64)
    times = [meta.seconds() for _, meta in paired]
    flight = (
        flights([t if t is not None else 0.0 for t in times])
        if all(t is not None for t in times)
        else [0] * len(paired)
    )
    count = max(flight) + 1
    offsets = np.zeros((len(paired), count))
    offsets[np.arange(len(paired)), flight] = 1.0

    # Up from gimbal pitch: view . u = sin(pitch), linear in u.
    solution, _ = _robust_lstsq(
        directions, np.sin(np.radians(pitch)), floor=math.sin(math.radians(2))
    )
    up = solution / np.linalg.norm(solution)
    # Up from barometric height alone: h = a . C + b_flight, u = a / |a|.
    both, _ = _robust_lstsq(np.hstack([centres, offsets]), height, floor=0.3)
    up_height = both[:3] / np.linalg.norm(both[:3])
    mean_image_up = image_ups.mean(axis=0)
    up_image = mean_image_up / np.linalg.norm(mean_image_up)
    # Scale with the up fixed: h = s (u . C) + b_flight.
    along = centres @ up
    scaled, inliers = _robust_lstsq(np.hstack([along[:, None], offsets]), height, floor=0.3)
    slope = float(scaled[0])
    # A slope of the wrong sign is a verdict, not a crash: the pose set and the photos
    # disagree about which way is up. The magnitude is kept so the matrix stays a
    # similarity, and the `height` check fails.
    scale = abs(slope) if slope != 0 else 1.0
    height_residual = height - np.hstack([along[:, None], offsets]) @ scaled
    solved_pitch = np.degrees(np.arcsin(np.clip(directions @ up, -1.0, 1.0)))
    pitch_residual = solved_pitch - pitch

    level = sfm.rotation_onto_z(up)
    levelled = directions @ level.T
    solved_heading = np.degrees(np.arctan2(levelled[:, 0], levelled[:, 1]))
    yaw = np.array(
        [np.nan if meta.gimbal_yaw_deg is None else meta.gimbal_yaw_deg for _, meta in paired]
    )
    usable = np.isfinite(yaw) & (np.abs(solved_pitch) < HEADING_MAX_PITCH_DEG)
    difference = _wrap(yaw - solved_heading)
    offset = 0.0
    heading_residual = np.full(len(paired), np.nan)
    if usable.sum() >= 3:
        keep = usable.copy()
        for _ in range(4):
            offset = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(difference[keep]))))))
            residual = _wrap(difference - offset)
            sigma = 1.4826 * float(np.median(np.abs(residual[keep])))
            keep = usable & (np.abs(residual) <= max(3.0 * sigma, 5.0))
            if keep.sum() < 3:
                keep = usable
                break
        heading_residual = np.where(usable, _wrap(difference - offset), np.nan)

    rotation = rotation_about_z(offset) @ level
    look_at = look_at_point(centres, directions)
    main = int(np.bincount(np.asarray(flight)).argmax())
    ground_offset = float(scaled[1 + main])
    translation = -scale * rotation @ look_at + np.array(
        [0.0, 0.0, scale * float(up @ look_at) + ground_offset]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation

    finite_heading = heading_residual[np.isfinite(heading_residual)]
    residuals = {
        "pitchMedianAbsDeg": round(float(np.median(np.abs(pitch_residual))), 3),
        "pitchP90AbsDeg": round(float(np.percentile(np.abs(pitch_residual), 90)), 3),
        "heightRmsM": round(float(np.sqrt(np.mean(height_residual[inliers] ** 2))), 4),
        "heightInlierFraction": round(float(inliers.mean()), 4),
        "headingMedianAbsDeg": (
            round(float(np.median(np.abs(finite_heading))), 3) if finite_heading.size else 180.0
        ),
        "headingP90AbsDeg": (
            round(float(np.percentile(np.abs(finite_heading), 90)), 3)
            if finite_heading.size
            else 180.0
        ),
        "heightCorrelation": round(float(np.corrcoef(along, height)[0, 1]), 4),
        "heightSpanM": round(float(np.ptp(height)), 2),
    }
    agreement = _angle_deg(up, up_height)
    checks = {
        "metaCoverage": len(paired) >= GATE["metaCoverageMin"] * len(views),
        "pitch": residuals["pitchMedianAbsDeg"] <= GATE["pitchMedianAbsMaxDeg"]
        and residuals["pitchP90AbsDeg"] <= GATE["pitchP90AbsMaxDeg"],
        "upAgreement": agreement <= GATE["upAgreementMaxDeg"],
        "height": slope > 0
        and residuals["heightRmsM"] <= GATE["heightRmsMaxM"]
        and residuals["heightInlierFraction"] >= GATE["heightInlierMin"],
        # A scale needs a baseline: photos at one height say nothing about metres.
        "heightSpan": residuals["heightSpanM"] >= GATE["heightSpanMinM"],
        "heading": residuals["headingMedianAbsDeg"] <= GATE["headingMedianAbsMaxDeg"]
        and residuals["headingP90AbsDeg"] <= GATE["headingP90AbsMaxDeg"],
    }
    badness = np.abs(pitch_residual) / 3.0 + np.abs(height_residual) / 0.5
    worst = [
        {
            "name": paired[int(index)][0].name,
            "pitchResidualDeg": round(float(pitch_residual[int(index)]), 2),
            "heightResidualM": round(float(height_residual[int(index)]), 3),
        }
        for index in np.argsort(-badness)[:20]
    ]
    return FrameFit(
        posed=len(views),
        with_meta=len(paired),
        metres_per_unit=scale,
        up=[float(v) for v in up],
        up_from_height=[float(v) for v in up_height],
        up_from_image=[float(v) for v in up_image],
        up_agreement_deg=round(agreement, 3),
        heading_offset_deg=round(offset, 3),
        look_at=[float(v) for v in look_at],
        ground_offset_m=ground_offset,
        matrix=[[float(v) for v in row] for row in matrix],
        residuals=residuals,
        checks=checks,
        worst=worst,
        flights=count,
    )


def frame_document(
    fit: FrameFit, model: sfm.Model, metas: Mapping[str, DroneMeta]
) -> dict[str, Any]:
    """`frame.json`: the fit and its verdict, plus what the post-train step needs from the
    pose set -- every camera in metres (the crown sits inside their ring; the GSD is their
    distance to it over the focal), the focal in training pixels, and the take-off's
    height above sea level as the drone reckoned it."""
    document = fit.to_dict()
    document["camerasM"] = [[round(float(v), 3) for v in row] for row in fit.apply(model.centres())]
    focals = [camera.focal_px for camera in model.cameras]
    document["focalPx"] = float(np.median(focals)) if focals else None
    if model.cameras:
        document["imageSize"] = [model.cameras[0].width, model.cameras[0].height]
    document["takeoffMslM"] = takeoff_msl(metas.values())
    return document


def gps_centre(metas: Iterable[DroneMeta]) -> tuple[float, float] | None:
    """The median GPS fix of the photos: reported beside the documented coordinate, not
    used for placement (consumer fixes carry metres of shared bias)."""
    fixes = [(m.latitude, m.longitude) for m in metas if m.latitude and m.longitude]
    if not fixes:
        return None
    return (
        float(np.median([f[0] for f in fixes])),
        float(np.median([f[1] for f in fixes])),
    )


def takeoff_msl(metas: Iterable[DroneMeta]) -> float | None:
    """Median of AbsoluteAltitude - RelativeAltitude: the take-off's height above sea
    level as the drone reckoned it (GPS/barometer, metres of error; cosmetic only)."""
    values = [
        m.absolute_altitude_m - m.relative_altitude_m
        for m in metas
        if m.absolute_altitude_m is not None and m.relative_altitude_m is not None
    ]
    return float(np.median(values)) if values else None


# --- what the GPU stages are asked for -----------------------------------------------------


#: Over photo-reconstruct's own `pose` params (COLMAP 4.2, 4096 features at 1600 px).
#:
#: `sequential`, with the image's vocabulary-tree loop closure, because these photos are
#: named in capture order (`frame_name`), 4-8 s and ~6 deg of orbit apart: exhaustive
#: would be 214k pairs for 655 photos where this is ~15 per photo plus the loop pairs.
#: Fifteen neighbours rather than COLMAP's ten because each orbit tier is ~60 photos and
#: its neighbours are the only overlap until the loop closes. `min_registered_fraction`
#: 0.7 and no rematches because the stage's answer to a short sequential result is an
#: exhaustive fill-in, and measured here (51 of these frames at 1600 px, COLMAP 3.9.1 on
#: 4 shared cores) exhaustive matching ran ~8 min per 1,275 pairs -- 214k pairs would not
#: fit the 6-hour limit. A model short of 70 % is reported and fails the gate instead.
POSE_OVERRIDES: dict[str, Any] = {
    "matcher": "sequential",
    "sequential_overlap": 15,
    "min_registered_fraction": 0.7,
    "rematches": 0,
    "live": False,
}


def pose_params(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """photo-reconstruct's own `pose` params with `POSE_OVERRIDES` and the run's own."""
    stage = next(s for s in load_recipe("photo-reconstruct").stages if s.id == "pose")
    params: dict[str, Any] = {**stage.params, **POSE_OVERRIDES}
    params.update(overrides or {})
    return params


#: EGM2008 geoid height near Minneapolis, approximately (+-2 m): the ellipsoid is this far
#: above mean sea level here. Only the site's recorded centre height uses it; the splat is
#: clamped to the viewer's terrain.
GEOID_UNDULATION_M = -27.6


def capture_descriptor() -> dict[str, Any]:
    """What `tools/captures/real_tree.py` needs to know about this capture that is not in
    its splat or its poses: where it is, what it is of, who made it and under what
    licence. real_tree.py itself holds nothing about any one capture."""
    return {
        "slug": "minnetonka-tree",
        "name": "Minnetonka tree (real capture, Living Survey)",
        "subject": "one mature deciduous tree in Minnetonka, Minnesota",
        "photos": "drone photographs",
        "capturedBy": ATTRIBUTION,
        "conditions": "in still air",
        "dataset": "Single Tree -- High-Density Photogrammetry Dataset, CC BY 4.0",
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "geoidUndulationM": GEOID_UNDULATION_M,
        "attribution": ATTRIBUTION,
        "attributionUrl": ATTRIBUTION_URL,
        "licenseName": LICENSE_NAME,
        "licenseUrl": LICENSE_URL,
        "sourceUrl": GITHUB_REPO,
        # The pinned group's date; train.json's sessions say which days a splat used.
        "captured": GROUPS["The_Tree"].date,
        "pipeline": (
            "tools/pipeline pose (COLMAP) + train (gsplat MCMC) on Modal via "
            "infra/modal/minnetonka.py"
        ),
    }


def train_params(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """photo-reconstruct's own Standard `train` params, unchanged -- 30k steps, gsplat
    MCMC, `cap_max: auto`, `converge: true`, `blocks: auto`, gsplat's default SH degree
    3 -- plus the run's own (`roi`, `budget_max`, ...).

    Nothing is fixed for this capture. The gaussian count is the density budget
    (gaussian_budget.py: the surface in finest-seen training pixels), clamped only by the
    memory of the GPU the stage runs on at the frames' real size; a budget one GPU cannot
    hold trains as blocks (blocks.py). m0 pinned `blocks: 1` and a `budget_max` of 2M,
    which is a count, not a quality."""
    stage = next(s for s in load_recipe("photo-reconstruct").stages if s.id == "train")
    params: dict[str, Any] = dict(stage.params)
    params.update(overrides or {})
    return params


def train_tier() -> str:
    """The GPU the recipe trains on: what this capture trains on unless a run says."""
    stage = next(s for s in load_recipe("photo-reconstruct").stages if s.id == "train")
    return str(stage.gpu.tier) if stage.gpu is not None else "l4"


def frame_size_params() -> tuple[int, int]:
    """`normalize`'s `auto_base` and `auto_ceiling`: the rule every capture's frames are
    sized by (`resolution.py`), which this capture's are too."""
    stage = next(s for s in load_recipe("photo-reconstruct").stages if s.id == "normalize")
    return (
        int(stage.params.get("auto_base", resolution.BASE)),
        int(stage.params.get("auto_ceiling", resolution.CEILING)),
    )


# --- capture sessions: which photos train ------------------------------------------------


def sessions(metas: Mapping[str, DroneMeta]) -> dict[str, str]:
    """Each photo's capture session, `YYYY-MM-DD/<flight>`: the day on the camera's clock,
    and the flight -- `flights`' index over the whole capture in time order, a new one
    after every gap of more than `SEGMENT_GAP_S` (a landing or a battery change). A photo
    without a timestamp is `unknown`. Generic: any photo set whose EXIF/XMP carries a
    capture time has sessions, and training on a subset of them is how a capture made
    over several visits (leaves that moved, light that changed) can be tested for it."""
    timed = {
        name: seconds
        for name, meta in metas.items()
        if (seconds := meta.seconds()) is not None and meta.taken
    }
    names = sorted(timed)
    flight = flights([timed[name] for name in names]) if names else []
    out = {name: "unknown" for name in metas}
    for name, index in zip(names, flight, strict=True):
        out[name] = f"{str(metas[name].taken)[:10]}/{index}"
    return out


def session_counts(metas: Mapping[str, DroneMeta]) -> dict[str, int]:
    """How many photos each session holds, in order."""
    counts: dict[str, int] = {}
    for label in sessions(metas).values():
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def names_in(metas: Mapping[str, DroneMeta], wanted: Sequence[str]) -> set[str]:
    """The photos in any of `wanted`: a whole day (`2020-07-20`) or one flight
    (`2020-07-20/2`). A selector that matches no photo is refused, so a typo cannot
    quietly train on nothing."""
    labels = sessions(metas)
    chosen: set[str] = set()
    for selector in wanted:
        hits = {
            name
            for name, label in labels.items()
            if label == selector or label.startswith(selector.rstrip("/") + "/")
        }
        if not hits:
            raise ValueError(
                f"no photo is in session {selector!r}; the sessions are {session_counts(metas)}"
            )
        chosen |= hits
    return chosen


def days(counts: Mapping[str, int]) -> dict[str, int]:
    """Photo counts by day, from `session_counts`' labels."""
    out: dict[str, int] = {}
    for label, count in counts.items():
        day = label.split("/", 1)[0]
        out[day] = out.get(day, 0) + count
    return out


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")

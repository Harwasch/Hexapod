"""Two epochs of the committed synthetic tree, flown and filmed, registered and scored.

The milestone-3 demo (M3b): epoch 1 is a drone survey, and a second epoch -- another
drone flight, or an iPhone video -- is aligned to it and scored against the truth. Every
epoch runs the shipped photo-reconstruct stages up to `place` (`normalize`, `pose`,
`georeference`, `align`, then `place` over the pose model's sparse points standing in
for a trained splat, which needs a GPU), on pycolmap 4.2, with the shipped parameters.

**These are synthetic captures, built to carry exactly the metadata real ones do:**

* the drone's photos are `tree_frames` renders named `DJI_0000.JPG`..., with an EXIF block
  (make DJI, a GPS fix at the true camera centre plus seeded noise of `RTK_SIGMA_M` per
  axis) and a DJI XMP packet saying RTK fixed (`RtkFlag` 50) with that same sigma -- so
  `normalize` detects `rtk-drone` and the pose priors take their sigma from the file, as
  a real M3E's would (`drone_frames`: field names from ExifTool's DJI table, no real M3E
  file read);
* the phone's video is the renders encoded by ffmpeg as H.264 in a `.mov` with the
  QuickTime keys an iPhone writes: make Apple, one ISO 6709 location `PHONE_GPS_OFFSET_M`
  from the orbit's centre, a declared horizontal accuracy, and no per-frame GPS;
* check targets are eight surveyed ground marks (`TARGETS_ENU`) with declared survey
  sigmas, and their clicks are their true projections plus seeded noise of `PIXEL_SIGMA`,
  on the original upload names -- **synthetic clicks**, not a detector's or a person's.

Each epoch has its own parent directory, because `tree_frames` writes its truth beside
the frames. Nothing here is a byte-gated fixture: the tree is only read.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gps_frames
import imageio_ffmpeg
import numpy as np
import numpy.typing as npt
from PIL import Image

import alignment
import drone_frames
import exif
import gaussians
import sfm
import tree_frames
from conftest import FIXTURE_PLY, make_recipe
from contracts import StageContext, StageOutcome
from executor import execute
from recipe import load_recipe
from registry import stage_impl
from runners import LocalRunner, RunnerSet
from stages import TRAINED_PLY
from workdir import Workdir

F64 = npt.NDArray[np.float64]

#: Where the scene is on the globe: the synthetic tree's own catalog coordinate.
ORIGIN = (28.0389, -82.6966, 10.8)
#: Ground marks round the tree (east, north, up metres about ORIGIN; the ground is z 0).
TARGETS_ENU: dict[str, tuple[float, float, float]] = {
    "T1": (4.0, 4.0, 0.0),
    "T2": (-4.0, 4.0, 0.0),
    "T3": (-4.0, -4.0, 0.0),
    "T4": (4.0, -4.0, 0.0),
    "T5": (6.0, 0.0, 0.0),
    "T6": (0.0, 6.0, 0.0),
    "T7": (-6.0, 0.0, 0.0),
    "T8": (0.0, -6.0, 0.0),
}
#: The drone's RTK fix error, per axis, written into its fixes *and* declared in its XMP.
RTK_SIGMA_M = 0.02
#: The targets' declared survey sigma, and the noise they were "surveyed" with.
TARGET_SIGMA_H_M, TARGET_SIGMA_V_M = 0.005, 0.01
#: A click's error, per pixel axis.
PIXEL_SIGMA = 0.5
#: Where the phone's one location is, east and north of the orbit's centre.
PHONE_GPS_OFFSET_M = (8.0, 5.0)
#: The horizontal accuracy the phone declares for that location (larger than its error).
PHONE_ACCURACY_M = 10.0
#: The fps the video is encoded at and normalized back at, one frame per render.
VIDEO_FPS = 6
#: A click must be this far inside the frame (pixels).
MARGIN_PX = 5.0

ORBITS: dict[str, dict[str, float]] = {
    "e1": {"count": 40, "radius_m": 9.0, "camera_height_m": 3.6, "phase_deg": 0.0},
    "e2_drone": {"count": 30, "radius_m": 11.0, "camera_height_m": 5.0, "phase_deg": 20.0},
    "e2_phone": {"count": 30, "radius_m": 11.0, "camera_height_m": 5.0, "phase_deg": 20.0},
    "unrelated": {"count": 30, "radius_m": 11.0, "camera_height_m": 5.0, "phase_deg": 0.0},
}
#: The unrelated scene's own ground, and where its capture says it is (by hand).
UNRELATED_GROUND_SEED = 12345


@stage_impl(
    "t_sparse_as_trained",
    consumes=("poses",),
    produces=(TRAINED_PLY,),
    summary="the pose model's sparse points as a splat, standing in for training",
)
def t_sparse_as_trained(ctx: StageContext) -> StageOutcome:
    """Training needs a GPU; `place` needs a `trained.ply` in the pose model's frame.
    The sparse points are exactly that, so placing them tests the placement."""
    points = sfm.read_points(ctx.input("poses"))
    gaussians.write_ply(ctx.output(TRAINED_PLY.name), points_as_gaussians(points))
    return StageOutcome(metrics={"points": int(points.shape[0])})


def points_as_gaussians(points: F64) -> dict[str, npt.NDArray[np.float32]]:
    count = points.shape[0]
    columns: dict[str, Any] = {axis: points[:, i] for i, axis in enumerate("xyz")}
    columns.update({f"f_dc_{i}": np.zeros(count) for i in range(3)})
    columns.update({"opacity": np.full(count, 2.0)})
    columns.update({f"scale_{i}": np.full(count, -4.0) for i in range(3)})
    columns.update({"rot_0": np.ones(count), "rot_1": np.zeros(count)})
    columns.update({"rot_2": np.zeros(count), "rot_3": np.zeros(count)})
    return {name: np.ascontiguousarray(v, dtype=np.float32) for name, v in columns.items()}


# --- the captures ---------------------------------------------------------------------------


def _splice(path: Path, *segments: bytes) -> None:
    raw = path.read_bytes()
    if raw[:2] != b"\xff\xd8":
        raise ValueError(f"{path.name} is not a JPEG")
    path.write_bytes(raw[:2] + b"".join(segments) + raw[2:])


def _app1(payload: bytes) -> bytes:
    return b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload


def _tag_drone(path: Path, fix: tuple[float, float, float] | None, *, rtk: bool) -> None:
    """EXIF (make DJI, the GPS fix when given) and a DJI XMP packet, spliced after SOI
    so the rendered scan is untouched."""
    tags = Image.Exif()
    tags[0x010F] = "DJI"
    tags[0x0110] = "M3E"
    if fix is not None:
        tags[0x8825] = drone_frames.gps_ifd(*fix)
    # With no fix the packet still says where the flight was (the scene), not null island.
    lat, lon, alt = fix if fix is not None else ORIGIN
    xmp = drone_frames.dji_xmp(
        rtk_flag=50 if rtk else None,
        std=(RTK_SIGMA_M, RTK_SIGMA_M, RTK_SIGMA_M),
        lat=lat,
        lon=lon,
        abs_alt=alt,
        rel_alt=alt - ORIGIN[2],
        calibrated_focal=None,
    )
    _splice(
        path,
        _app1(b"Exif\x00\x00" + tags.tobytes()),
        _app1(b"http://ns.adobe.com/xap/1.0/\0" + xmp.encode()),
    )


def _render(root: Path, name: str, *, empty: bool = False) -> tuple[Path, tree_frames.Truth]:
    orbit = ORBITS[name]
    cloud = (
        tree_frames.Cloud(
            position=np.zeros((0, 3)),
            colour=np.zeros((0, 3), dtype=np.uint8),
            radius_m=np.zeros(0),
        )
        if empty
        else None
    )
    frames = root / name / "render" / "frames"
    truth = tree_frames.render_orbit(
        FIXTURE_PLY,
        frames,
        count=int(orbit["count"]),
        radius_m=orbit["radius_m"],
        camera_height_m=orbit["camera_height_m"],
        phase_deg=orbit["phase_deg"],
        cloud=cloud,
        ground_seed=UNRELATED_GROUND_SEED if empty else tree_frames.GROUND_SEED,
    )
    return frames, truth


def render_drone(
    root: Path, name: str, *, rtk: bool = True, gps: bool = True, seed: int = 0
) -> tuple[Path, tree_frames.Truth, dict[str, str]]:
    """A drone photo set: (upload dir, truth, original upload name -> rendered name)."""
    frames, truth = _render(root, name, empty=not gps)
    upload = root / name / "upload"
    upload.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    originals: dict[str, str] = {}
    for index, pose in enumerate(truth.poses):
        original = f"DJI_{index:04d}.JPG"
        shutil.copyfile(frames / pose.name, upload / original)
        fix = None
        if gps:
            east, north, up = pose.centre + rng.normal(scale=RTK_SIGMA_M, size=3)
            fix = gps_frames.enu_to_geodetic(float(east), float(north), float(up), *ORIGIN)
        _tag_drone(upload / original, fix, rtk=rtk)
        originals[original] = pose.name
    return upload, truth, originals


def phone_location() -> tuple[float, float, float]:
    """The phone's one location: the orbit's centre moved by `PHONE_GPS_OFFSET_M`."""
    return gps_frames.enu_to_geodetic(*PHONE_GPS_OFFSET_M, 0.0, *ORIGIN)


def iso6709(lat: float, lon: float, height: float) -> str:
    return f"{lat:+09.6f}{lon:+010.6f}{height:+08.3f}/"


def render_phone(root: Path, name: str = "e2_phone") -> tuple[Path, tree_frames.Truth, Path]:
    """An iPhone-style `.mov`: (upload dir, truth, the renders it was encoded from)."""
    frames, truth = _render(root, name)
    upload = root / name / "upload"
    upload.mkdir(parents=True, exist_ok=True)
    clip = upload / "IMG_0042.MOV"
    lat, lon, height = phone_location()
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-y",
         "-framerate", str(VIDEO_FPS), "-i", str(frames / "frame_%04d.jpg"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-movflags", "use_metadata_tags",
         "-metadata", f"com.apple.quicktime.location.ISO6709={iso6709(lat, lon, height)}",
         "-metadata", f"com.apple.quicktime.location.accuracy.horizontal={PHONE_ACCURACY_M}",
         "-metadata", "com.apple.quicktime.make=Apple",
         "-metadata", "com.apple.quicktime.model=iPhone 15 Pro",
         str(clip)],
        capture_output=True, check=True,
    )  # fmt: skip
    return upload, truth, frames


def surveyed_targets(seed: int = 7) -> list[dict[str, object]]:
    """The eight marks as a survey reports them: truth plus noise of the declared sigma,
    on the same `msl-unverified` datum the EXIF-defined frame has."""
    rng = np.random.default_rng(seed)
    out: list[dict[str, object]] = []
    for label, enu in TARGETS_ENU.items():
        noise = rng.normal(scale=(TARGET_SIGMA_H_M, TARGET_SIGMA_H_M, TARGET_SIGMA_V_M))
        east, north, up = (float(v) for v in np.asarray(enu) + noise)
        lat, lon, height = gps_frames.enu_to_geodetic(east, north, up, *ORIGIN)
        out.append(
            {
                "label": label,
                "role": "check",
                "lat": lat,
                "lon": lon,
                "height": height,
                "heightDatum": "msl-unverified",
                "sigmaHM": TARGET_SIGMA_H_M,
                "sigmaVM": TARGET_SIGMA_V_M,
            }
        )
    return out


def observations(
    truth: tree_frames.Truth, originals: Mapping[str, str], seed: int = 11
) -> list[dict[str, object]]:
    """Each target clicked in every frame that sees it: its true projection plus noise,
    on the original upload name, in that original's pixels (COLMAP's continuous
    convention, which `tree_frames` renders in)."""
    rng = np.random.default_rng(seed)
    intrinsics = truth.intrinsics
    rendered = truth.by_name()
    out: list[dict[str, object]] = []
    for original in sorted(originals):
        pose = rendered[originals[original]]
        for label, enu in TARGETS_ENU.items():
            camera = pose.rotation @ np.asarray(enu) + pose.translation
            if camera[2] <= 0.1:
                continue
            u = intrinsics.fx * camera[0] / camera[2] + intrinsics.cx
            v = intrinsics.fy * camera[1] / camera[2] + intrinsics.cy
            u, v = (float(c) for c in np.asarray([u, v]) + rng.normal(scale=PIXEL_SIGMA, size=2))
            if not (
                MARGIN_PX <= u <= intrinsics.width - MARGIN_PX
                and MARGIN_PX <= v <= intrinsics.height - MARGIN_PX
            ):
                continue
            out.append(
                {
                    "target": label,
                    "image": original,
                    "px": [u, v],
                    "imageSize": [intrinsics.width, intrinsics.height],
                }
            )
    return out


# --- running an epoch -----------------------------------------------------------------------


@dataclass
class Epoch:
    name: str
    workdir: Workdir
    truth: tree_frames.Truth
    #: Kept frame name -> the rendered pose it is.
    kept: dict[str, str]
    alignment: dict[str, Any] = field(default_factory=dict)
    georef: dict[str, Any] = field(default_factory=dict)
    #: `fingerprint_of` when it was first built (epoch 1 only: what later epochs must
    #: not change).
    fingerprint: str | None = None

    @property
    def origin(self) -> tuple[float, float, float]:
        return (float(self.georef["lat"]), float(self.georef["lon"]), float(self.georef["height"]))

    def path(self, stage: str, name: str) -> Path:
        return self.workdir.artifact_path(stage, name)


def write_site_context(
    directory: Path,
    *,
    reference: Epoch | None,
    targets: list[dict[str, object]],
    observations: list[dict[str, object]],
) -> Path:
    """`site_context/` as the worker will seed it (M3b.2): the reference epoch's frames,
    poses, placement and georef copied in, its verdict and uncertainty beside them."""
    directory.mkdir(parents=True, exist_ok=True)
    document = alignment.empty_site_context("synthetic-tree")
    if reference is not None:
        target = directory / "reference"
        shutil.copytree(reference.path("normalize", "frames"), target / "frames")
        shutil.copytree(reference.path("pose", "poses"), target / "poses")
        shutil.copyfile(reference.path("place", "placement.json"), target / "placement.json")
        shutil.copyfile(reference.path("georeference", "georef.json"), target / "georef.json")
        document["reference"] = {
            "epochId": reference.name,
            "jobId": None,
            "observedAt": None,
            "platform": reference.alignment["platform"],
            "verdict": reference.alignment["verdict"],
            "uncertaintyM": reference.alignment["uncertaintyM"],
            "frames": "reference/frames",
            "poses": "reference/poses",
            "placement": "reference/placement.json",
            "georef": "reference/georef.json",
        }
    document["targets"] = targets
    document["observations"] = observations
    (directory / alignment.SITE_CONTEXT_FILE).write_text(json.dumps(document, indent=1) + "\n")
    return directory


def _shipped(stage_id: str) -> dict[str, Any]:
    recipe: dict[str, Any] = load_recipe("photo-reconstruct").to_dict()
    for stage in recipe["stages"]:
        if stage["id"] == stage_id:
            params = dict(stage.get("params") or {})
            return {"id": stage["id"], "impl": stage["impl"], "params": params}
    raise KeyError(stage_id)


def run_epoch(
    root: Path,
    name: str,
    *,
    upload: Path,
    site_context: Path,
    frame_count: int,
    video: bool,
    params: Mapping[str, Mapping[str, object]] | None = None,
) -> Workdir:
    """The shipped stages normalize, pose, georeference, align and place (GPU blocks
    dropped; training replaced by the sparse points), on this machine."""
    stages = [_shipped(s) for s in ("normalize", "pose", "georeference", "align")]
    # One frame per render: every frame of the clip, every photo.
    stages[0]["params"].update({"select": "all", "keep": frame_count, "keep_video": frame_count})
    if video:
        stages[0]["params"]["fps"] = VIDEO_FPS
    stages.append({"id": "train", "impl": "t_sparse_as_trained"})
    stages.append(_shipped("place"))
    for stage in stages:
        stage["params"] = {**stage.get("params", {}), **dict((params or {}).get(stage["id"], {}))}
    workdir = Workdir.create(root / name / "run")
    shutil.copytree(upload, workdir.input_path("upload"), dirs_exist_ok=True)
    shutil.copytree(site_context, workdir.input_path("site_context"), dirs_exist_ok=True)
    execute(
        make_recipe(stages, name="photo-reconstruct-local", inputs=["upload", "site_context"]),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    return workdir


def _load(epoch: Epoch) -> Epoch:
    epoch.alignment = json.loads(epoch.path("align", "epoch_alignment.json").read_text())
    epoch.georef = json.loads(epoch.path("georeference", "georef.json").read_text())
    return epoch


def _kept_photos(workdir: Workdir, originals: Mapping[str, str]) -> dict[str, str]:
    meta = json.loads(workdir.artifact_path("normalize", "frames_meta.json").read_text())
    return {record["name"]: originals[record["original"]] for record in meta["frames"]}


def _kept_video(workdir: Workdir, rendered: Path, truth: tree_frames.Truth) -> dict[str, str]:
    """Kept frame i is render i: asserted, by which render each kept frame is nearest."""
    kept = sorted(p for p in workdir.artifact_path("normalize", "frames").iterdir())
    names = [pose.name for pose in truth.poses]
    if len(kept) != len(names):
        raise AssertionError(f"normalize kept {len(kept)} frames of a {len(names)}-frame clip")

    def small(path: Path) -> F64:
        with Image.open(path) as image:
            return np.asarray(image.convert("L").resize((80, 60)), dtype=np.float64)

    renders = [small(rendered / name) for name in names]
    for index, path in enumerate(kept):
        frame = small(path)
        nearest = int(np.argmin([np.abs(frame - r).mean() for r in renders]))
        if nearest != index:
            raise AssertionError(f"kept frame {path.name} is nearest render {nearest}, not {index}")
    return {path.name: name for path, name in zip(kept, names, strict=True)}


def build(root: Path) -> dict[str, Epoch]:
    """e1 (RTK drone + check targets, a site's first epoch), e2_drone (RTK drone + check
    targets, aligned to e1), e2_phone (iPhone .mov, aligned to e1), e2_phone_framed (the
    same clip told e1's origin as its site frame, as M2 runs it), and unrelated (another
    scene, placed by hand, aligned to e1: it must not)."""
    targets = surveyed_targets()
    epochs: dict[str, Epoch] = {}

    upload, truth, originals = render_drone(root, "e1", seed=1)
    context = write_site_context(
        root / "e1" / "context", reference=None, targets=targets,
        observations=observations(truth, originals, seed=21),
    )  # fmt: skip
    workdir = run_epoch(root, "e1", upload=upload, site_context=context,
                        frame_count=len(truth.poses), video=False)  # fmt: skip
    e1 = _load(Epoch("e1", workdir, truth, _kept_photos(workdir, originals)))
    epochs["e1"] = e1
    fingerprint = e1.fingerprint = fingerprint_of(e1)

    upload, truth, originals = render_drone(root, "e2_drone", seed=2)
    context = write_site_context(
        root / "e2_drone" / "context", reference=e1, targets=targets,
        observations=observations(truth, originals, seed=22),
    )  # fmt: skip
    workdir = run_epoch(root, "e2_drone", upload=upload, site_context=context,
                        frame_count=len(truth.poses), video=False)  # fmt: skip
    epochs["e2_drone"] = _load(Epoch("e2_drone", workdir, truth, _kept_photos(workdir, originals)))

    upload, truth, rendered = render_phone(root)
    context = write_site_context(root / "e2_phone" / "context", reference=e1, targets=[],
                                 observations=[])  # fmt: skip
    for name, extra in (
        ("e2_phone", None),
        ("e2_phone_framed", {"georeference": {"site_frame": site_frame_of(e1)},
                             "pose": {"site_frame": site_frame_of(e1)},
                             "align": {"site_frame": site_frame_of(e1)}}),
    ):  # fmt: skip
        workdir = run_epoch(root, name, upload=upload, site_context=context,
                            frame_count=len(truth.poses), video=True, params=extra)  # fmt: skip
        epochs[name] = _load(Epoch(name, workdir, truth, _kept_video(workdir, rendered, truth)))

    upload, truth, originals = render_drone(root, "unrelated", rtk=False, gps=False, seed=3)
    context = write_site_context(root / "unrelated" / "context", reference=e1, targets=[],
                                 observations=[])  # fmt: skip
    workdir = run_epoch(
        root, "unrelated", upload=upload, site_context=context, frame_count=len(truth.poses),
        video=False, params={"georeference": {"lat": ORIGIN[0], "lon": ORIGIN[1]}},
    )  # fmt: skip
    epochs["unrelated"] = _load(
        Epoch("unrelated", workdir, truth, _kept_photos(workdir, originals))
    )
    if fingerprint_of(e1) != fingerprint:
        raise AssertionError("aligning later epochs changed epoch 1's files")
    return epochs


def site_frame_of(epoch: Epoch) -> dict[str, object]:
    """The `site_frame` param M2's worker would send a later epoch of `epoch`'s site."""
    lat, lon, height = epoch.origin
    return {"siteId": "synthetic-tree", "lat": lat, "lon": lon, "height": height,
            "heightDatum": "msl-unverified", "epsg": 32617}  # fmt: skip


def fingerprint_of(epoch: Epoch) -> str:
    """One hash over the epoch's frames, poses and placement."""
    digest = hashlib.sha256()
    for root in (
        epoch.path("normalize", "frames"),
        epoch.path("pose", "poses"),
        epoch.path("place", "placement.json"),
    ):
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        for path in files:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


# --- scoring against the truth ---------------------------------------------------------------


def placement_frame(epoch: Epoch) -> sfm.Similarity | None:
    """The similarity `place` applied to this epoch's pose model, when it did not
    recentre (an unaligned, located epoch's placement is not a fixed frame)."""
    frame = (
        epoch.alignment["frame"]
        if epoch.alignment.get("frameSource") == "epoch-alignment"
        else epoch.georef.get("frame")
    )
    if not isinstance(frame, dict) or frame.get("recentre") is not False:
        return None
    return sfm.Similarity(
        scale=float(frame["scale"]),
        rotation=np.asarray(frame["rotation"], dtype=np.float64),
        translation=np.asarray(frame["translationM"], dtype=np.float64),
    )


def truth_centre_errors(epoch: Epoch) -> F64 | None:
    """Each placed camera centre against its true one, east/north/up about the epoch's
    own origin (the truth reframed from ORIGIN, exactly). None when not placed in a
    fixed frame."""
    similarity = placement_frame(epoch)
    if similarity is None:
        return None
    model = sfm.read_model(epoch.path("pose", "poses"))
    rendered = epoch.truth.by_name()
    names = [image.name for image in model.images if image.name in epoch.kept]
    placed = similarity.apply(np.stack([model.by_name()[n].centre for n in names]))
    true = exif.reframe_enu(
        np.stack([rendered[epoch.kept[n]].centre for n in names]), ORIGIN, epoch.origin
    )
    return np.asarray(np.linalg.norm(placed - true, axis=1), dtype=np.float64)


def truth_centre_rmse(epoch: Epoch) -> float | None:
    errors = truth_centre_errors(epoch)
    return None if errors is None else float(np.sqrt(np.mean(errors**2)))


def placed_scene_distance(epoch: Epoch) -> float:
    """Median distance of `place`'s output (the sparse points, placed) to the scene they
    were rendered from: the nearest fixture gaussian, or the textured ground at z 0,
    both reframed from ORIGIN to the epoch's origin."""
    placed = gaussians.read_splat(epoch.path("place", "canonical.ply"))
    sample = placed.xyz[:: max(1, placed.count // 400)].astype(np.float64)
    fixture = gaussians.read_splat(FIXTURE_PLY).xyz.astype(np.float64)
    in_scene = exif.reframe_enu(sample, epoch.origin, ORIGIN)
    nearest = np.sqrt(((in_scene[:, None, :] - fixture[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    ground = np.abs(in_scene[:, 2])
    return float(np.median(np.minimum(nearest, ground)))


def gnss_offset_truth() -> float:
    """How far the phone's location is from where its cameras' centroid truly is."""
    return math.hypot(*PHONE_GPS_OFFSET_M)

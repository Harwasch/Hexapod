"""Structure from motion: running COLMAP, reading what it wrote, and scoring it.

Everything here is pure or a subprocess argv, so the `pose` stage in `stages.py` stays a
thin layer that resolves artifacts. Three things live here that are worth naming:

* **The `pose` stage runs `sequential` only with loop detection, and only as a first
  try.** A0 #7 ran
  both on the same 40-frame closed orbit: exhaustive registered 40/40, sequential without
  loop detection registered **2**, because the last frame never meets the first. With a
  vocabulary tree (`vocab_tree_path`) sequential matching queries every tenth frame
  against the whole capture, which is what closes an orbit, and it matches a small
  multiple of the frame count in pairs rather than its square. It is still not trusted
  on its own: the README's 100 real frames registered 76 in its run and 98 when the same
  frames were re-run on 2026-09-26 -- the mapper is not repeatable run to run -- so the
  `pose` stage checks the registered fraction and falls back to exhaustive matching when
  it is short (see `colmap` in stages.py, and `matching_plan` for the measurements).
* **the focal length.** A0 #7 measured COLMAP's self-calibrated focal 3.1% low on this
  fixture. `camera_params` therefore exists: given a focal prior (EXIF, ARKit) the stage
  fixes it and says so, and given none it records `focalPriorPx: null` together with the
  measured bias, so a downstream metric scale is never read as if it were surveyed.
* **`umeyama`**, because a COLMAP reconstruction is only defined up to a similarity. Any
  comparison against known poses has to solve for that similarity first, and the scale it
  returns *is* the scale bias -- it is not a nuisance parameter to throw away.
* **`model_aligner` is read through `--transform_path`, never through `--output_path`.**
  Measured on the COLMAP 3.9.1 installed here (`apt-get install -y colmap` on Ubuntu
  24.04), on the 40-frame rendered orbit, aligning to the cameras' true centres: the
  model written to `--output_path` **is not georeferenced**. Its camera centres sit a
  median of 9.315 m from the reference -- the same 9.3 m COLMAP prints as its own
  "Alignment error", on a 9 m orbit -- while the eight numbers written to
  `--transform_path`, applied by hand to those same centres, put them 0.009 m from it.
  The transform is right to millimetres and the model it wrote beside it is not.

  Be precise about the mechanism, because the obvious test for it gives the wrong
  answer: the output is **not** a byte-for-byte copy of the input. `cameras.bin` is
  identical, but `images.bin` and `points3D.bin` both differ -- COLMAP rewrites them and
  the result is still unaligned. So a check that hashes the whole model and finds a
  difference will conclude the alignment was applied. It was not. The only sound test is
  the one above: compare the output model's centres against the reference directly.

  Nothing here consumes that model or that number; `alignment_residuals` recomputes the
  residual from the transform, which is the quantity `georef.json` records anyway.

`read_model` is a reader for COLMAP's own binary format rather than a shell out to
`model_converter`: the stage needs the camera count and the poses to write `poses.json`,
and a second subprocess to get them would make a 20 MB `points3D.txt` on a real capture.
The reader is held honest by `tests/test_pose_colmap.py`, which scores the poses it
returns against known ground truth -- a misread quaternion does not produce a 0.4 degree
median error.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import struct
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

__all__ = [
    "MATCHERS",
    "VOCAB_TREE_ENV",
    "Camera",
    "Image",
    "MatchPass",
    "Model",
    "Points3D",
    "Similarity",
    "UpEstimate",
    "alignment_residuals",
    "camera_up",
    "colmap_available",
    "colmap_exe",
    "colmap_version",
    "count_pairs",
    "feature_extractor_argv",
    "mapper_argv",
    "matcher_argv",
    "matching_plan",
    "model_aligner_argv",
    "quat_to_matrix",
    "read_model",
    "read_points",
    "read_points3d",
    "read_similarity",
    "rotation_angle_deg",
    "rotation_onto_z",
    "umeyama",
    "vocab_tree_path",
    "write_points3d",
    "write_ref_positions",
]

F64 = npt.NDArray[np.float64]

#: The matchers this stage will run, and what each one is for.
MATCHERS: dict[str, str] = {
    "exhaustive": "every pair; the only one that closes an orbit (A0 #7: 40/40)",
    "sequential": (
        "neighbours in filename order, plus vocabulary-tree loop closure when given a "
        "tree; without one, 2/40 on a closed orbit"
    ),
    "spatial": "neighbours by GPS; needs per-image location priors in the database",
}

#: Where the image says its vocabulary tree is. `infra/modal/app.py` downloads COLMAP's
#: published Flickr100K 32K-word tree into the CPU image at build time, checks its sha256,
#: and sets this; a machine without it (the worker, CI) matches exhaustively instead.
VOCAB_TREE_ENV = "COLMAP_VOCAB_TREE"

#: COLMAP camera model id -> (name, parameter names). Only the models this stage asks
#: for are listed; an unlisted id is read as its raw parameter vector and named by id.
CAMERA_MODELS: dict[int, tuple[str, tuple[str, ...]]] = {
    0: ("SIMPLE_PINHOLE", ("f", "cx", "cy")),
    1: ("PINHOLE", ("fx", "fy", "cx", "cy")),
    2: ("SIMPLE_RADIAL", ("f", "cx", "cy", "k")),
    3: ("RADIAL", ("f", "cx", "cy", "k1", "k2")),
    4: ("OPENCV", ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")),
}


class ColmapMissingError(RuntimeError):
    """No COLMAP on this machine. The stage says where to get one rather than crashing."""


def colmap_exe() -> str:
    """The absolute path to `colmap`, from `$COLMAP_BIN` or `$PATH`.

    Absolute on purpose: the argv goes to `subprocess` with no shell, so a bare name
    would be resolved by the child's environment rather than by this decision, which is
    also what ruff's S607 is pointing at.
    """
    configured = os.environ.get("COLMAP_BIN")
    found = configured if configured else shutil.which("colmap")
    if not found or not Path(found).exists():
        raise ColmapMissingError(
            "colmap is not on this machine. Install it (`apt-get install -y colmap` on "
            "Ubuntu 24.04 gives 3.9.1 built without CUDA, which is enough -- A0 #7 "
            "measured the CPU-only path at 40/40 registered in 48.3 s on 4 cores) or "
            "point $COLMAP_BIN at one."
        )
    return str(Path(found).resolve())


def colmap_version() -> str:
    """The version string COLMAP prints in its own help banner, e.g. `3.9.1`.

    Recorded in `poses.json` because 3.9 and 3.10 are not the same mapper, and a pose set
    nobody can attribute to a version is a pose set nobody can reproduce.
    """
    completed = subprocess.run(  # noqa: S603 - absolute argv from colmap_exe, no shell
        [colmap_exe(), "-h"], capture_output=True, text=True, check=False
    )
    match = re.search(r"COLMAP\s+(\S+)", completed.stdout + completed.stderr)
    return match.group(1) if match else "unknown"


def colmap_available() -> bool:
    try:
        colmap_exe()
    except ColmapMissingError:
        return False
    return True


def vocab_tree_path(configured: object = None) -> Path | None:
    """The vocabulary tree for loop detection: the `vocab_tree` param, else the env var.

    None when neither names a file that exists. Loop detection is what separates a
    sequential match that closes an orbit from one that registers 2 of 40, so a missing
    tree is not something to run without: the caller matches exhaustively instead and
    says why.

    The tree must be the FLANN-format `vocab_tree_flickr100K_words32K.bin` that COLMAP
    3.9.1 reads. COLMAP 3.11 moved to a faiss format with differently named files
    (`vocab_tree_faiss_*`), which 3.9.1 cannot load.
    """
    raw = str(configured) if configured else os.environ.get(VOCAB_TREE_ENV, "")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path.resolve() if path.is_file() else None


# --- argv ---------------------------------------------------------------------------


def feature_extractor_argv(
    database: Path,
    images: Path,
    *,
    camera_model: str = "SIMPLE_RADIAL",
    camera_params: Sequence[float] | None = None,
    single_camera: bool = True,
    max_image_size: int = 2400,
    max_features: int = 8192,
    first_octave: int | None = None,
    num_threads: int | None = None,
    use_gpu: bool = False,
) -> list[str]:
    """SIFT over every frame, into one database.

    `single_camera` is on by default because a frame set from one clip *is* one camera,
    and letting COLMAP fit one intrinsic per frame throws away the only constraint that
    makes a 40-frame orbit over-determined.

    `first_octave` is COLMAP's own -1 unless given: -1 doubles the image before building
    the scale space, so the finest octave alone is four times the frame's pixels.
    `num_threads` None is COLMAP's -1, every core the process can see.
    """
    argv = [
        colmap_exe(),
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--ImageReader.single_camera",
        "1" if single_camera else "0",
        "--ImageReader.camera_model",
        camera_model,
        "--SiftExtraction.use_gpu",
        "1" if use_gpu else "0",
        "--SiftExtraction.max_image_size",
        str(max_image_size),
        "--SiftExtraction.max_num_features",
        str(max_features),
    ]
    if camera_params is not None:
        argv += ["--ImageReader.camera_params", ",".join(f"{v:.10g}" for v in camera_params)]
    if first_octave is not None:
        argv += ["--SiftExtraction.first_octave", str(first_octave)]
    if num_threads is not None:
        argv += ["--SiftExtraction.num_threads", str(num_threads)]
    return argv


def matcher_argv(
    database: Path,
    matcher: str = "exhaustive",
    *,
    use_gpu: bool = False,
    overlap: int | None = None,
    vocab_tree: Path | None = None,
    num_threads: int | None = None,
) -> list[str]:
    """The matching pass. For `sequential`, `overlap` neighbours in each direction and,
    with a `vocab_tree`, loop detection -- which is what closes an orbit that sequential
    matching alone would leave open (A0 #7's 2/40).

    Every matcher skips a pair the database already holds, which is what makes an
    `exhaustive` pass after a `sequential` one a fill-in: it matches only the pairs the
    first pass did not. `clear_matches` is how a pass is made to start over instead.
    """
    if matcher not in MATCHERS:
        raise ValueError(
            f"unknown matcher {matcher!r}; this stage runs one of {', '.join(sorted(MATCHERS))}"
        )
    argv = [
        colmap_exe(),
        f"{matcher}_matcher",
        "--database_path",
        str(database),
        "--SiftMatching.use_gpu",
        "1" if use_gpu else "0",
    ]
    if matcher == "sequential":
        if overlap is not None:
            argv += ["--SequentialMatching.overlap", str(overlap)]
        if vocab_tree is not None:
            argv += [
                "--SequentialMatching.loop_detection",
                "1",
                "--SequentialMatching.vocab_tree_path",
                str(vocab_tree),
            ]
    if num_threads is not None:
        argv += ["--SiftMatching.num_threads", str(num_threads)]
    return argv


def mapper_argv(
    database: Path,
    images: Path,
    output: Path,
    *,
    refine_focal_length: bool = True,
    refine_extra_params: bool = True,
    random_seed: int = 0,
    num_threads: int | None = None,
) -> list[str]:
    """Incremental SfM. `refine_focal_length=False` is how a focal prior is *held*.

    `random_seed` is COLMAP's own default, 0, unless a retry asks for another: the
    mapper's initial-pair search is randomised, and which pair it starts from decides
    whether an orbit closes (see `colmap` in stages.py).
    """
    argv = [
        colmap_exe(),
        "mapper",
        "--random_seed",
        str(random_seed),
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(output),
        "--Mapper.ba_refine_focal_length",
        "1" if refine_focal_length else "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "1" if refine_extra_params else "0",
    ]
    if num_threads is not None:
        argv += ["--Mapper.num_threads", str(num_threads)]
    return argv


@dataclass(frozen=True)
class MatchPass:
    """One matching pass the `pose` stage may run, and why it is in the plan."""

    matcher: str
    loop_detection: bool = False
    #: Forget every earlier match first (`clear_matches`), so this pass re-matches rather
    #: than filling in.
    clear: bool = False
    why: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "matcher": self.matcher,
            "loopDetection": self.loop_detection,
            "clear": self.clear,
            "why": self.why,
        }


def matching_plan(
    requested: str,
    *,
    source_format: str | None,
    vocab_tree: bool,
    rematches: int = 1,
) -> tuple[MatchPass, ...]:
    """Which matching passes to run, in order; the stage stops at the first that is enough.

    `auto` is the recipe's choice. A frame set that came out of a **video** is in time
    order, so each frame's neighbours by name are its neighbours in space, and sequential
    matching with vocabulary-tree loop detection matches a few pairs per frame where
    exhaustive matches every pair. Measured 2026-09-26 on 4 cores of the development
    container (COLMAP 3.9.1, 4096 features, 1600 px; the machine was shared with another
    test run for part of it, so these are upper bounds):

        capture                      matcher          pairs  extract  match   map  total  reg.
        fixture orbit, 87 @ 640x480  exhaustive       3,741    17 s  195 s  33 s  245 s  87/87
        fixture orbit, 87 @ 640x480  sequential+loop  1,192    10 s   93 s  27 s  131 s  87/87
        real frames, 100 @ 3008x2000 sequential+loop  1,372   109 s  348 s 150 s  607 s  100/100
        README's 100 @ 1080x1920    sequential+loop  1,400    79 s  271 s  82 s  432 s  98/100

    On the 3008x2000 frames (a fisheye walk round a backhoe, in capture order, which SIFT
    sees at 1600 px), the exhaustive fill-in of the other 3,578 pairs took a further
    553 s and mapping on all 4,950 took 266 s (98/100): exhaustive from the start would
    have been about twice the sequential run. When the fallback does run, it costs the
    sequential pass and one mapping on top of plain exhaustive -- on the fixture, 316 s
    against 245 s. A **photo set** has no such order --
    `frame_0003` may be across the scene from `frame_0004` -- so it is matched
    exhaustively. Sequential is never planned without a vocabulary tree (A0 #7's 2/40);
    `sequential` asked for by name on a machine with no tree is matched exhaustively, and
    the plan says so.

    A sequential pass is always followed by an exhaustive one **without** clearing: the
    matchers skip pairs the database already holds, so the fallback costs only the pairs
    sequential did not match. Then `rematches` exhaustive passes that do clear, the
    existing answer to RANSAC's thread-order nondeterminism (`_best_reconstruction`).
    """
    if requested not in (*MATCHERS, "auto"):
        raise ValueError(
            f"unknown matcher {requested!r}; this stage runs auto or one of "
            f"{', '.join(sorted(MATCHERS))}"
        )
    passes: list[MatchPass] = []
    wants_sequential = requested == "sequential" or (
        requested == "auto" and source_format == "video"
    )
    if wants_sequential and vocab_tree:
        passes.append(
            MatchPass(
                "sequential",
                loop_detection=True,
                why=(
                    "frames from a video, in time order: neighbours plus vocabulary-tree "
                    "loop closure"
                    if requested == "auto"
                    else "asked for by name, with vocabulary-tree loop closure"
                ),
            )
        )
        passes.append(
            MatchPass(
                "exhaustive",
                why=(
                    "fallback: sequential registered too few frames, so every pair it "
                    "skipped is matched as well"
                ),
            )
        )
    elif wants_sequential:
        passes.append(
            MatchPass(
                "exhaustive",
                why=(
                    f"no vocabulary tree on this machine (${VOCAB_TREE_ENV}), and "
                    f"sequential matching without loop detection registered 2 of 40 on a "
                    f"closed orbit (A0 #7)"
                ),
            )
        )
    elif requested == "spatial":
        passes.append(MatchPass("spatial", why="asked for by name"))
    else:
        passes.append(
            MatchPass(
                "exhaustive",
                why=(
                    "asked for by name"
                    if requested == "exhaustive"
                    else f"frames from {source_format or 'an unknown source'}, in no "
                    f"guaranteed order: every pair"
                ),
            )
        )
    last = passes[-1].matcher
    passes += [
        MatchPass(last, clear=True, why="rematch: RANSAC across threads is not deterministic")
    ] * max(0, rematches)
    return tuple(passes)


def count_pairs(database: Path) -> tuple[int, int]:
    """(pairs matched, pairs geometrically verified) in a COLMAP database.

    Every pair a matcher tried has a `matches` row, empty or not (87 frames exhaustive
    wrote 3,741 = 87 * 86 / 2), which is what makes this the cost of a matching pass.
    """
    with sqlite3.connect(database) as connection:
        tried = connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        verified = connection.execute(
            "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"
        ).fetchone()[0]
    return int(tried), int(verified)


def clear_matches(database: Path) -> None:
    """Forget every match and two-view geometry, keeping the features.

    `*_matcher` skips pairs the database already has, so this is what makes a second
    matching pass actually re-match. Features stay: extraction is deterministic, and it
    is the matching's geometric verification (RANSAC across threads) that is not.
    """
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM matches")
        connection.execute("DELETE FROM two_view_geometries")


def model_aligner_argv(
    model: Path,
    output: Path,
    ref_images: Path,
    transform: Path,
    *,
    max_error_m: float,
    min_common_images: int = 3,
) -> list[str]:
    """Estimate the similarity that takes this model into the frame `ref_images` is in.

    Three of these flags are decisions rather than defaults:

    * **`--ref_is_gps 0`** with positions already in metres, and **`--alignment_type
      custom`**. COLMAP will take latitude and longitude directly (`--ref_is_gps 1`
      with `ecef` or `enu`), but then *COLMAP* chooses the frame: `enu` puts the origin
      at the first reference image (measured), and `ecef` produces coordinates of order
      6.4e6 m. Handing it metres about an origin this project picked means the frame the
      transform lands in is one the rest of the pipeline already knows.
    * **`--alignment_max_error` must be greater than zero.** COLMAP 3.9.1 refuses the run
      outright otherwise -- "You must provide a maximum alignment error > 0" -- so there
      is no non-robust mode to fall back to. It is a RANSAC inlier threshold in the units
      of `ref_images`, which here is metres.
    * **`--output_path` is written and never read.** See the module docstring: on 3.9.1
      the model it writes there is not georeferenced, though it is not a byte copy of the
      input either -- two of its three files change and the result is still unaligned.
    """
    if max_error_m <= 0.0:
        raise ValueError(
            f"alignment_max_error must be > 0; COLMAP 3.9.1 refuses {max_error_m!r} with "
            f'"You must provide a maximum alignment error > 0". It is a RANSAC inlier '
            f"threshold in metres, so it is the size of the GPS error being tolerated"
        )
    return [
        colmap_exe(),
        "model_aligner",
        "--input_path",
        str(model),
        "--output_path",
        str(output),
        "--ref_images_path",
        str(ref_images),
        "--ref_is_gps",
        "0",
        "--alignment_type",
        "custom",
        "--merge_image_and_ref_origins",
        "0",
        "--transform_path",
        str(transform),
        "--min_common_images",
        str(min_common_images),
        "--alignment_max_error",
        f"{max_error_m:.10g}",
    ]


def write_ref_positions(path: Path, positions: Mapping[str, Sequence[float]]) -> int:
    """`<image name> <x> <y> <z>` per line, which is what `--ref_images_path` reads.

    Sorted by name so two runs over the same capture hand COLMAP the same file. Returns
    how many lines were written, because "how many frames had a position" is the first
    number anybody asks about a georeference.
    """
    lines = [
        f"{name} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f}"
        for name, values in sorted(positions.items())
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def read_similarity(path: Path) -> Similarity:
    """Read what COLMAP wrote to `--transform_path`.

    Eight numbers on one line: `scale`, then a w-first quaternion, then a translation,
    applied as `x' = scale * R(q) @ x + t`. That spelling is not in COLMAP's help text;
    it was read off 3.9.1's own output and checked against `umeyama` on the same
    correspondences, which agreed to 7 mm on the rendered orbit.
    """
    values = [float(token) for token in path.read_text(encoding="utf-8").split()]
    if len(values) != 8:
        raise ValueError(
            f"{path.name} holds {len(values)} numbers, not the 8 COLMAP 3.9.1 writes "
            f"(scale, w-first quaternion, translation). A different COLMAP may write a "
            f"4x4 matrix instead, and this reader has to learn that before trusting it"
        )
    return Similarity(
        scale=values[0],
        rotation=quat_to_matrix(values[1:5]),
        translation=np.asarray(values[5:8], dtype=np.float64),
    )


def alignment_residuals(
    similarity: Similarity,
    centres: Mapping[str, Sequence[float]],
    reference: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Per-image distance, in metres, between an aligned camera centre and its reference.

    Recomputed here rather than scraped out of COLMAP's log for two reasons: on 3.9.1 that
    number is computed against the model it failed to transform, and a residual is
    provenance -- it belongs in `georef.json` per image, not in a line of stderr.

    What it measures is *consistency*, not accuracy. Every fix in a capture shares
    whatever bias the receiver had, and a bias common to all of them moves the whole
    reconstruction without changing a single residual. That is why the stage floors the
    uncertainty it reports rather than quoting this number straight.
    """
    out: dict[str, float] = {}
    for name, target in reference.items():
        centre = centres.get(name)
        if centre is None:
            continue
        moved = similarity.apply(np.asarray([centre], dtype=np.float64))[0]
        out[name] = float(np.linalg.norm(moved - np.asarray(target, dtype=np.float64)))
    return out


# --- the model COLMAP wrote ---------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]
    param_names: tuple[str, ...]

    @property
    def focal_px(self) -> float:
        """`fx`, or the single `f` of a one-focal model. The number A0 found 3.1% low."""
        named = dict(zip(self.param_names, self.params, strict=False))
        for key in ("fx", "f"):
            if key in named:
                return float(named[key])
        return float(self.params[0]) if self.params else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "params": dict(zip(self.param_names, self.params, strict=False))
            or {f"p{i}": v for i, v in enumerate(self.params)},
        }


@dataclass(frozen=True)
class Image:
    """One registered frame: world-to-camera rotation as a w-first quaternion, and `t`.

    COLMAP's convention exactly -- `x_cam = R(q) @ x_world + t` -- so the camera centre
    is `-R^T t` and nothing here silently flips a handedness.
    """

    id: int
    name: str
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    points2d: int
    points3d: int

    @property
    def rotation(self) -> F64:
        return quat_to_matrix(self.qvec)

    @property
    def centre(self) -> F64:
        return -self.rotation.T @ np.asarray(self.tvec, dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "qvec": list(self.qvec),
            "tvec": list(self.tvec),
            "cameraId": self.camera_id,
            "points2D": self.points2d,
            "points3D": self.points3d,
            "centre": [float(v) for v in self.centre],
        }


@dataclass(frozen=True)
class Model:
    cameras: tuple[Camera, ...]
    images: tuple[Image, ...]
    points3d: int
    mean_track_length: float

    @property
    def registered(self) -> int:
        return len(self.images)

    def by_name(self) -> dict[str, Image]:
        return {image.name: image for image in self.images}

    def centres(self) -> F64:
        if not self.images:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([image.centre for image in self.images])


def read_model(directory: Path) -> Model:
    """Read `cameras.bin`, `images.bin` and `points3D.bin` out of a COLMAP sparse model.

    Note the capital D: COLMAP writes `points3D.bin`, which the artifact-name grammar in
    `artifacts.py` (lowercase only) cannot express. That is why `POSES.stub_members` does
    not name it -- a lowercase lookalike in the stub would be a file name no real run
    ever produces.
    """
    cameras = _read_cameras(directory / "cameras.bin")
    images = _read_images(directory / "images.bin")
    points, tracks = _read_points3d(directory / "points3D.bin")
    return Model(
        cameras=cameras,
        images=images,
        points3d=points,
        mean_track_length=(tracks / points) if points else 0.0,
    )


def _read_cameras(path: Path) -> tuple[Camera, ...]:
    out: list[Camera] = []
    with path.open("rb") as handle:
        for _ in range(_u64(handle)):
            camera_id, model_id, width, height = struct.unpack("<IiQQ", handle.read(24))
            name, names = CAMERA_MODELS.get(model_id, (f"MODEL_{model_id}", ()))
            count = len(names) if names else _fallback_param_count(model_id)
            params = struct.unpack(f"<{count}d", handle.read(8 * count))
            out.append(
                Camera(
                    id=camera_id,
                    model=name,
                    width=width,
                    height=height,
                    params=params,
                    param_names=names,
                )
            )
    return tuple(out)


def _fallback_param_count(model_id: int) -> int:
    raise ValueError(
        f"COLMAP camera model id {model_id} is not one this reader knows the parameter "
        f"count for, so the rest of cameras.bin cannot be located. Add it to CAMERA_MODELS"
    )


def _read_images(path: Path) -> tuple[Image, ...]:
    out: list[Image] = []
    with path.open("rb") as handle:
        for _ in range(_u64(handle)):
            image_id, *pose, camera_id = struct.unpack("<IdddddddI", handle.read(64))
            name = _read_cstring(handle)
            count = _u64(handle)
            raw = handle.read(24 * count)
            ids = np.frombuffer(raw, dtype="<u8").reshape(-1, 3)[:, 2] if count else np.zeros(0)
            out.append(
                Image(
                    id=image_id,
                    name=name,
                    qvec=(pose[0], pose[1], pose[2], pose[3]),
                    tvec=(pose[4], pose[5], pose[6]),
                    camera_id=camera_id,
                    points2d=count,
                    # 2**64-1 is COLMAP's "this observation has no 3D point".
                    points3d=int(np.count_nonzero(ids != np.uint64(0xFFFFFFFFFFFFFFFF))),
                )
            )
    return tuple(sorted(out, key=lambda image: image.name))


def read_points(directory: Path) -> F64:
    """The sparse model's 3D points, as an (n, 3) array in the model's own frame."""
    out: list[tuple[float, float, float]] = []
    with (directory / "points3D.bin").open("rb") as handle:
        for _ in range(_u64(handle)):
            record = handle.read(43)  # id u64, xyz 3 x f64, rgb 3 x u8, error f64
            out.append(struct.unpack_from("<3d", record, 8))
            handle.read(8 * _u64(handle))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


@dataclass(frozen=True)
class Points3D:
    """Every field of `points3D.bin`, as arrays, so that a subset can be written back.

    `track_offsets[i]:track_offsets[i + 1]` indexes point `i`'s rows of `track`, each an
    `(image_id, point2D_idx)` pair: the observations gsplat's parser turns into
    `point_indices`, which is what its depth loss projects into each frame.
    """

    ids: npt.NDArray[np.uint64]
    xyz: F64
    rgb: npt.NDArray[np.uint8]
    error: F64
    track: npt.NDArray[np.uint32]
    track_offsets: npt.NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def track_of(self, index: int) -> npt.NDArray[np.uint32]:
        return self.track[self.track_offsets[index] : self.track_offsets[index + 1]]

    def subset(self, keep: npt.NDArray[np.bool_]) -> Points3D:
        """The points where `keep` is true, their tracks with them."""
        lengths = np.diff(self.track_offsets)
        rows = np.repeat(keep, lengths)
        offsets = np.concatenate([[0], np.cumsum(lengths[keep])]).astype(np.int64)
        return Points3D(
            ids=self.ids[keep],
            xyz=self.xyz[keep],
            rgb=self.rgb[keep],
            error=self.error[keep],
            track=self.track[rows],
            track_offsets=offsets,
        )


def read_points3d(path: Path) -> Points3D:
    """`points3D.bin` in full: id, xyz, rgb, error and track, per point."""
    ids: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    error: list[float] = []
    tracks: list[npt.NDArray[np.uint32]] = []
    offsets = [0]
    with path.open("rb") as handle:
        for _ in range(_u64(handle)):
            point_id, x, y, z, r, g, b, err = struct.unpack("<Q3d3Bd", handle.read(43))
            length = _u64(handle)
            track = np.frombuffer(handle.read(8 * length), dtype="<u4").reshape(-1, 2)
            ids.append(point_id)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
            error.append(err)
            tracks.append(track)
            offsets.append(offsets[-1] + length)
    return Points3D(
        ids=np.asarray(ids, dtype=np.uint64),
        xyz=np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        rgb=np.asarray(rgb, dtype=np.uint8).reshape(-1, 3),
        error=np.asarray(error, dtype=np.float64),
        track=(
            np.concatenate(tracks).astype(np.uint32)
            if tracks
            else np.zeros((0, 2), dtype=np.uint32)
        ),
        track_offsets=np.asarray(offsets, dtype=np.int64),
    )


def write_points3d(path: Path, points: Points3D) -> int:
    """Write `points3D.bin` in COLMAP's binary layout. Returns the point count.

    `images.bin` beside it may still name a point this file no longer holds, as the 3D
    point behind one of its 2D features. gsplat's parser reads the points and their
    tracks and never that cross-reference, so a file written here is fit for a trainer's
    dataset; it is not a COLMAP model to hand back to COLMAP, and it never goes into the
    `poses` artifact.
    """
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for index in range(len(points)):
            track = points.track_of(index)
            handle.write(
                struct.pack(
                    "<Q3d3Bd",
                    int(points.ids[index]),
                    *(float(v) for v in points.xyz[index]),
                    *(int(v) for v in points.rgb[index]),
                    float(points.error[index]),
                )
            )
            handle.write(struct.pack("<Q", int(track.shape[0])))
            handle.write(np.ascontiguousarray(track, dtype="<u4").tobytes())
    return len(points)


@dataclass(frozen=True)
class UpEstimate:
    """Which way is up in a reconstruction, from how the camera was held.

    `consistency` is the length of the mean of the per-frame up vectors: 1.0 when every
    frame was held at the same roll and pitch, falling as they disagree. It is reported
    rather than thresholded -- a capture walking round a tree tilts the phone a little,
    one filming the ground from above tilts it a lot, and the number says which.
    """

    up: tuple[float, float, float]
    consistency: float
    frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "camera-up",
            "up": [round(v, 6) for v in self.up],
            "consistency": round(self.consistency, 4),
            "frames": self.frames,
            "note": (
                "the mean of each registered frame's up (-y of the camera, in the model's "
                "frame): nerfstudio's `up` orientation and gsplat's similarity_from_cameras "
                "use the same estimate. Right when the capture was filmed roughly upright"
            ),
        }


def camera_up(model: Model) -> UpEstimate | None:
    """The mean camera-up direction of a reconstruction, in its own frame.

    COLMAP cameras look down +z with +y pointing *down* the image, so a frame's up in the
    world is `-R[1, :]` for its world-to-camera rotation `R`. Averaged over every
    registered frame this is a robust estimate of gravity for footage taken the way
    people hold phones -- upright, roughly level -- which is what nerfstudio's default
    `orientation_method="up"` and gsplat's `similarity_from_cameras` both rely on. It is
    no estimate at all of *heading*: nothing in a reconstruction from images says which
    way north is.
    """
    if not model.images:
        return None
    ups = np.stack([-image.rotation[1, :] for image in model.images])
    mean = ups.mean(axis=0)
    length = float(np.linalg.norm(mean))
    if length == 0.0:
        return None
    unit = mean / length
    return UpEstimate(
        up=(float(unit[0]), float(unit[1]), float(unit[2])),
        consistency=length,
        frames=len(model.images),
    )


def rotation_onto_z(up: Sequence[float] | F64) -> F64:
    """The smallest proper rotation that takes the unit vector `up` onto +z.

    Rodrigues about `up x z`. When `up` is already (anti)parallel to z the axis is
    undefined, so +z is the identity and -z is a half turn about x -- the same choice
    `gaussians.UP_AXES["-z"]` makes, so a y-up estimate and a y-up file land the same.
    """
    u = np.asarray(up, dtype=np.float64)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    c = float(u @ z)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(u, z)
    s = float(np.linalg.norm(axis))
    k = axis / s
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.asarray(np.eye(3) + s * kx + (1.0 - c) * (kx @ kx), dtype=np.float64)


def _read_points3d(path: Path) -> tuple[int, int]:
    """(point count, total track length). The points themselves are not needed here."""
    points = 0
    tracks = 0
    with path.open("rb") as handle:
        for _ in range(_u64(handle)):
            handle.read(43)  # id, xyz, rgb, error
            length = _u64(handle)
            handle.read(8 * length)
            points += 1
            tracks += length
    return points, tracks


def _u64(handle: BinaryIO) -> int:
    return int(struct.unpack("<Q", handle.read(8))[0])


def _read_cstring(handle: BinaryIO) -> str:
    out = bytearray()
    while (byte := handle.read(1)) not in (b"\x00", b""):
        out += byte
    return out.decode("utf-8", errors="replace")


# --- geometry -----------------------------------------------------------------------


def quat_to_matrix(qvec: Sequence[float]) -> F64:
    """w-first quaternion to a rotation matrix. COLMAP's `qvec` order, not scipy's."""
    w, x, y, z = (float(v) for v in qvec)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        raise ValueError("a zero quaternion is not a rotation")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_angle_deg(a: F64, b: F64) -> float:
    """The angle of the rotation that takes `a` to `b`, in degrees."""
    trace = float(np.trace(a.T @ b))
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


@dataclass(frozen=True)
class Similarity:
    """`target ~= scale * rotation @ source + translation`.

    `scale` is not a nuisance term. A reconstruction from images alone has no metric
    scale at all, so this number *is* the answer to "how far off was the size", and the
    focal bias A0 measured shows up in it.
    """

    scale: float
    rotation: F64
    translation: F64

    def apply(self, points: F64) -> F64:
        return (self.scale * (self.rotation @ points.T)).T + self.translation


def umeyama(source: F64, target: F64) -> Similarity:
    """Least-squares similarity from `source` to `target` (Umeyama 1991), with scale.

    Includes the reflection guard the original paper is about, which is the same trap A0
    fell into from the other side: an improper "rotation" fits the points better and is
    not a rotation, and the symptom downstream is a constant, plausible-looking error.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"umeyama needs two (n, 3) arrays; got {source.shape} and {target.shape}")
    if len(source) < 3:
        raise ValueError(f"a similarity needs at least three points; got {len(source)}")
    mu_source, mu_target = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mu_source, target - mu_target
    covariance = (b.T @ a) / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    variance = float((a**2).sum() / len(source))
    scale = float((singular @ np.diag(correction)) / variance) if variance > 0 else 1.0
    return Similarity(
        scale=scale, rotation=rotation, translation=mu_target - scale * rotation @ mu_source
    )

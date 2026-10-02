"""COLMAP 4.2 for the `pose` stage, through pycolmap: extraction, matching, mapping.

**What it is for.** `pose` / `colmap` runs apt's COLMAP 3.9.1 CLI. `colmap: "4.2"` runs
the same three steps -- SIFT extraction, sequential (with vocabulary-tree loop detection)
or exhaustive matching, incremental mapping -- on COLMAP 4.2 instead, so that the two can
be compared on one real capture with nothing else changed: the stage's matching plan, the
exhaustive fallback, the mapper seeds, `mapper: global`, `poses.json` and its metrics
are all the same code. Between 3.9.1 and 4.2 (COLMAP's CHANGELOG.rst): 3.11 replaced FLANN
with faiss for matching and retrieval; 4.0 merged GLOMAP; 4.1 made the exhaustive matcher
faster (IVF scalar quantisation) and fixed a sequential loop-detection hang; 4.1.1 fixed a
~4-6x matching slowdown from an OpenMP critical section in RANSAC (a 4.x regression, not
one 3.9.1 had); 4.2 added the 6-point shared-focal relative-pose solver, MSAC with Sampson
refinement for two-view geometry and analytical BA Jacobians ("~1.2-1.55x faster
incremental mapping").

**Measured here** (2026-09-28, 4 cores of the development container, otherwise idle; the
pose stage end to end on each, same frames, same params; errors against the known poses
after a similarity fit, medians; two runs each except the last, which is one):

    frames                          params     COLMAP  extract  match    map  reg.   rot deg trans
    orbit 40 @ 640x480, exhaustive  recipe*    3.9.1     2.1 s  17.9 s  4.7 s  40/40  0.10 0.18%
                                               4.2.0     1.9 s   3.3 s  2.6 s  40/40  0.12 0.19%
    orbit 40 @ 640x480, exhaustive  defaults   3.9.1     8.0 s  67.3 s 11.9 s  40/40  0.06 0.09%
                                               4.2.0     6.8 s  14.8 s  5.7 s  40/40  0.06 0.11%
    video 30 @ 640x480, seq.+loop   defaults   3.9.1     6.1 s  39.8 s  2.2 s  30/30  0.12 0.20%
                                               4.2.0     5.0 s  13.6 s  1.4 s  30/30  0.11 0.15%
    video 30 @ 640x480, seq.+loop   recipe*    3.9.1     1.7 s  12.7 s  1.3 s  30/30  0.15 0.25%
                                               4.2.0     1.5 s   3.6 s  4.1 s  30/30^ 0.12 0.22%
    video 87 @ 1600x1200, seq.+loop recipe*    3.9.1    26.6 s 255.3 s 193 s   87/87  0.019 0.034%
                                               4.2.0    21.4 s 188.9 s  83 s   87/87  0.013 0.023%

    * photo-reconstruct's 4096 features, 1600 px, first_octave 0
    ^ after the exhaustive fill-in: sequential alone registered 15 (below)

So on these frames 4.2's pose error is within run-to-run noise of 3.9.1's (a little
worse on the exhaustive orbit, better on the videos), it maps 2-2.3x faster where mapping
is long, and matches 1.35x (1600 px) to 5x (640 px) faster. One real difference: on the
30 small frames with the recipe's `first_octave: 0` (~800 features each), 4.2's
two-view verification kept fewer inliers per neighbour pair (118 against 140 on
average) and its sequential mapping registered **15 of 30** where 3.9.1's registered
30; the stage's exhaustive fill-in then registered 30/30, as it is there to. It did not
happen at the stage defaults, nor on the 1600 px frames. Pycolmap's CPU wheel uses every
core: extraction ran at ~3.7 and matching at ~3.4 cores' CPU time per wall second.
None of this is a real capture; the spool comparison is what decides the default.

**Why pycolmap and not a COLMAP 4.2 binary.** The Modal CPU image already installs
pycolmap 4.2.0's manylinux wheel for the global mapper (`global_sfm.py`), and it exposes
everything this stage asks of the CLI: `extract_features` with a single camera, a camera
model and prior, `max_image_size`, `max_num_features`, `first_octave` and threads;
`match_sequential` with `overlap`, loop detection and a vocabulary tree (in two passes, to
pair the frames 3.9.1 pairs -- `_matching` says why); `match_exhaustive`
and `match_spatial`, which skip pairs the database already holds, so the fill-in fallback
behaves as it does on 3.9.1; and `incremental_mapping` with a seed, held intrinsics,
threads and snapshots every N frames for the live viewer. The alternatives were
conda-forge (a second package manager in the image), or building COLMAP 4 from source
(CMake, Ceres, Boost, CGAL, faiss, onnxruntime; 15-25 minutes per image change).

**The vocabulary tree changes with the version.** 3.11 moved retrieval to faiss, and a
FLANN tree -- the `vocab_tree_flickr100K_words32K.bin` 3.9.1 reads -- is not one 4.x
loads. COLMAP 4 downloads a default faiss tree (the 256K-word one) on first use, but only
when built with download support, which the pycolmap wheel is not (its default path is
empty). So the image carries **the faiss build of the same Flickr100K 32K-word tree**,
from the same 3.11.1 release, at `$COLMAP4_VOCAB_TREE`: the same vocabulary size as 3.9.1's,
so the comparison is of the versions, not of the trees.

It runs in a **subprocess** (this file is also a script), as `global_sfm.py` does: a C++
abort in the new build is then a failed step the stage reports, not a dead stage process,
and the pipeline itself never imports pycolmap. The model it writes is trimmed to
COLMAP 3.9.1's own three files (`global_sfm.KEEP`), so `georeference`'s `model_aligner`
(still 3.9.1) and everything downstream read it exactly as they read 3.9.1's.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# The same interpreter, the same wheel and the same trimmed model the global mapper uses.
from global_sfm import KEEP, PYCOLMAP_VERSION, python

__all__ = [
    "VERSIONS",
    "VOCAB_TREE_ENV",
    "extract_argv",
    "main",
    "map_argv",
    "match_argv",
    "python",
    "version",
    "vocab_tree_path",
]

#: What the stage's `colmap` param may name: apt's CLI, or pycolmap's build.
VERSIONS = ("3.9", "4.2")
#: Where the image says the faiss-format vocabulary tree is (`infra/modal/app.py`).
VOCAB_TREE_ENV = "COLMAP4_VOCAB_TREE"


def vocab_tree_path(configured: object = None) -> Path | None:
    """The faiss tree for 4.2's loop detection: the `vocab_tree` param, else the env var.

    None when neither names a file, and then the stage matches exhaustively and says why,
    exactly as it does on 3.9.1 without a tree.
    """
    raw = str(configured) if configured else os.environ.get(VOCAB_TREE_ENV, "")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path.resolve() if path.is_file() else None


def version(interpreter: str) -> str | None:
    """The COLMAP version pycolmap was built from (`4.2.0`), or None without pycolmap.

    Asked of the interpreter that will run the steps, before any of them, so a run that
    asked for 4.2 on a machine without it stops with that said rather than three steps
    into a traceback -- and never quietly becomes a 3.9.1 run in an A/B.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - our own interpreter and script, no shell
            [*_script(interpreter, "version")], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    found = completed.stdout.strip().splitlines()
    return found[-1] if completed.returncode == 0 and found else None


def _script(interpreter: str, command: str) -> list[str]:
    return [interpreter, str(Path(__file__).resolve()), command]


def extract_argv(
    interpreter: str,
    database: Path,
    images: Path,
    *,
    camera_model: str = "SIMPLE_RADIAL",
    camera_params: tuple[float, ...] | None = None,
    max_image_size: int = 2400,
    max_features: int = 8192,
    first_octave: int | None = None,
    num_threads: int | None = None,
) -> list[str]:
    """`sfm.feature_extractor_argv`, on 4.2: one camera for the clip, SIFT on the CPU."""
    argv = [
        *_script(interpreter, "extract"),
        "--database",
        str(database),
        "--images",
        str(images),
        "--camera-model",
        camera_model,
        "--max-image-size",
        str(max_image_size),
        "--max-features",
        str(max_features),
    ]
    if camera_params is not None:
        argv += ["--camera-params", ",".join(f"{v:.10g}" for v in camera_params)]
    if first_octave is not None:
        argv += ["--first-octave", str(first_octave)]
    if num_threads is not None:
        argv += ["--threads", str(num_threads)]
    return argv


def match_argv(
    interpreter: str,
    database: Path,
    matcher: str,
    *,
    overlap: int | None = None,
    vocab_tree: Path | None = None,
    num_threads: int | None = None,
) -> list[str]:
    """`sfm.matcher_argv`, on 4.2. A `vocab_tree` turns on sequential loop detection."""
    if matcher not in ("exhaustive", "sequential", "spatial"):
        raise ValueError(f"unknown matcher {matcher!r}")
    argv = [*_script(interpreter, "match"), "--database", str(database), "--matcher", matcher]
    if matcher == "sequential":
        if overlap is not None:
            argv += ["--overlap", str(overlap)]
        if vocab_tree is not None:
            argv += ["--vocab-tree", str(vocab_tree)]
    if num_threads is not None:
        argv += ["--threads", str(num_threads)]
    return argv


def map_argv(
    interpreter: str,
    database: Path,
    images: Path,
    output: Path,
    *,
    refine_focal_length: bool = True,
    refine_extra_params: bool = True,
    random_seed: int = 0,
    num_threads: int | None = None,
    snapshot_path: Path | None = None,
    snapshot_every: int = 0,
) -> list[str]:
    """`sfm.mapper_argv`, on 4.2: incremental mapping into numbered models under `output`."""
    argv = [
        *_script(interpreter, "map"),
        "--database",
        str(database),
        "--images",
        str(images),
        "--output",
        str(output),
        "--seed",
        str(random_seed),
    ]
    if not refine_focal_length:
        argv.append("--hold-focal")
    if not refine_extra_params:
        argv.append("--hold-extra")
    if num_threads is not None:
        argv += ["--threads", str(num_threads)]
    if snapshot_path is not None and snapshot_every > 0:
        argv += ["--snapshot-path", str(snapshot_path), "--snapshot-every", str(snapshot_every)]
    return argv


# --- the script half, run under an interpreter with pycolmap ------------------------------


def _extraction(pycolmap: Any, parsed: argparse.Namespace) -> tuple[Any, Any]:
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = parsed.camera_model
    if parsed.camera_params:
        reader.camera_params = parsed.camera_params
    options = pycolmap.FeatureExtractionOptions()
    options.use_gpu = False
    options.num_threads = parsed.threads
    options.max_image_size = parsed.max_image_size
    options.sift.max_num_features = parsed.max_features
    if parsed.first_octave is not None:
        options.sift.first_octave = parsed.first_octave
    return reader, options


def _matching(pycolmap: Any, parsed: argparse.Namespace) -> tuple[Any, list[Any]]:
    """The matching options and the pairing pass(es) to run with them, in order.

    Sequential is **two** passes, to match the pairs 3.9.1 matches. 3.9.1's
    `quadratic_overlap` (on by default in both) *adds* frames i+1, i+2, i+4, ... i+2^9 to
    the linear i+1 ... i+overlap; 4.2's replaces the linear window with them
    (`SequentialPairGenerator::Next`, colmap/controllers/pairing.cc) -- on the 30-frame
    fixture 180 pairs where 3.9.1 matched 275, and it registered 15 of 30 where 3.9.1
    registered all 30. So the linear window first, then the quadratic one with loop
    detection; each pass skips pairs the database already holds.
    """
    options = pycolmap.FeatureMatchingOptions()
    options.use_gpu = False
    options.num_threads = parsed.threads
    if parsed.matcher != "sequential":
        return options, [
            pycolmap.SpatialPairingOptions()
            if parsed.matcher == "spatial"
            else pycolmap.ExhaustivePairingOptions()
        ]
    passes = []
    for quadratic in (False, True):
        pairing = pycolmap.SequentialPairingOptions()
        pairing.num_threads = parsed.threads
        pairing.quadratic_overlap = quadratic
        if parsed.overlap is not None:
            pairing.overlap = parsed.overlap
        if quadratic and parsed.vocab_tree is not None:
            pairing.loop_detection = True
            pairing.vocab_tree_path = parsed.vocab_tree
        passes.append(pairing)
    return options, passes


def _mapping(pycolmap: Any, parsed: argparse.Namespace) -> Any:
    options = pycolmap.IncrementalPipelineOptions()
    options.random_seed = parsed.seed
    options.num_threads = parsed.threads
    options.ba_refine_focal_length = not parsed.hold_focal
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = not parsed.hold_extra
    options.ba_use_gpu = False
    if parsed.snapshot_path is not None and parsed.snapshot_every > 0:
        options.snapshot_path = parsed.snapshot_path
        options.snapshot_frames_freq = parsed.snapshot_every
    return options


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COLMAP 4.2 (pycolmap) for the pose stage")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--database", type=Path, required=True)
    extract.add_argument("--images", type=Path, required=True)
    extract.add_argument("--camera-model", default="SIMPLE_RADIAL")
    extract.add_argument("--camera-params", default="")
    extract.add_argument("--max-image-size", type=int, default=2400)
    extract.add_argument("--max-features", type=int, default=8192)
    extract.add_argument("--first-octave", type=int, default=None)
    extract.add_argument("--threads", type=int, default=-1)
    match = commands.add_parser("match")
    match.add_argument("--database", type=Path, required=True)
    match.add_argument("--matcher", choices=("exhaustive", "sequential", "spatial"))
    match.add_argument("--overlap", type=int, default=None)
    match.add_argument("--vocab-tree", type=Path, default=None)
    match.add_argument("--threads", type=int, default=-1)
    mapper = commands.add_parser("map")
    mapper.add_argument("--database", type=Path, required=True)
    mapper.add_argument("--images", type=Path, required=True)
    mapper.add_argument("--output", type=Path, required=True)
    mapper.add_argument("--seed", type=int, default=0)
    mapper.add_argument("--threads", type=int, default=-1)
    mapper.add_argument("--hold-focal", action="store_true")
    mapper.add_argument("--hold-extra", action="store_true")
    mapper.add_argument("--snapshot-path", type=Path, default=None)
    mapper.add_argument("--snapshot-every", type=int, default=0)
    commands.add_parser("version")
    check = commands.add_parser("self-check")
    check.add_argument("--vocab-tree", type=Path, default=None)
    return parser


def _self_check(pycolmap: Any, parsed: argparse.Namespace) -> int:
    """Every option this file sets exists on this pycolmap, and it is the pinned one.

    pybind11 option objects refuse an attribute they do not have, so building each one
    the way a run will is a check that an upgrade has not renamed a field under us.
    """
    assert pycolmap.__version__ == PYCOLMAP_VERSION, pycolmap.__version__
    for name in ("extract_features", "match_sequential", "match_exhaustive"):
        assert callable(getattr(pycolmap, name)), name
    assert callable(pycolmap.incremental_mapping)
    common = {"database": Path("d"), "threads": 4}
    _extraction(
        pycolmap,
        argparse.Namespace(
            **common,
            camera_model="SIMPLE_RADIAL",
            camera_params="1000,800,600,0",
            max_image_size=1600,
            max_features=4096,
            first_octave=0,
        ),
    )
    for matcher in ("sequential", "exhaustive", "spatial"):
        _matching(
            pycolmap,
            argparse.Namespace(**common, matcher=matcher, overlap=10, vocab_tree=Path("t")),
        )
    _mapping(
        pycolmap,
        argparse.Namespace(
            **common,
            seed=1,
            hold_focal=True,
            hold_extra=False,
            snapshot_path=Path("s"),
            snapshot_every=3,
        ),
    )
    words = ""
    if parsed.vocab_tree is not None:
        # Loaded, not just found: a FLANN tree (3.9.1's) is refused right here --
        # "Failed to read faiss index ... legacy flann-based index" -- not mid-capture.
        index = pycolmap.VisualIndex.read(parsed.vocab_tree)
        assert index.num_visual_words() > 0, parsed.vocab_tree
        words = f", vocabulary tree {index.num_visual_words()} words"
    sys.stdout.write(f"colmap4: {pycolmap.COLMAP_version} ok{words}\n")
    return 0


def main(args: list[str] | None = None) -> int:
    parsed = _parser().parse_args(args)
    try:
        import pycolmap  # type: ignore[import-not-found,unused-ignore]
    except ImportError as error:
        sys.stdout.write(
            f"colmap4: pycolmap is not installed for {sys.executable} ({error}); the CPU "
            f"image installs pycolmap=={PYCOLMAP_VERSION}, or set $COLMAP_GLOBAL_PYTHON\n"
        )
        return 3
    if parsed.command == "version":
        # pycolmap's own version is the COLMAP release it was built from.
        sys.stdout.write(f"{pycolmap.__version__}\n")
        return 0
    if parsed.command == "self-check":
        return _self_check(pycolmap, parsed)
    cpu = pycolmap.Device.cpu
    began = time.monotonic()
    if parsed.command == "extract":
        reader, options = _extraction(pycolmap, parsed)
        pycolmap.extract_features(
            parsed.database,
            parsed.images,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader,
            extraction_options=options,
            device=cpu,
        )
    elif parsed.command == "match":
        options, passes = _matching(pycolmap, parsed)
        run = {
            "sequential": pycolmap.match_sequential,
            "spatial": pycolmap.match_spatial,
            "exhaustive": pycolmap.match_exhaustive,
        }[parsed.matcher]
        for pairing in passes:
            run(parsed.database, matching_options=options, pairing_options=pairing, device=cpu)
    else:
        # The CLI's `--random_seed` seeds COLMAP's global PRNG as well as the mapper's.
        pycolmap.set_random_seed(parsed.seed)
        parsed.output.mkdir(parents=True, exist_ok=True)
        models = pycolmap.incremental_mapping(
            parsed.database, parsed.images, parsed.output, _mapping(pycolmap, parsed)
        )
        for model in sorted(p for p in parsed.output.iterdir() if p.is_dir()):
            for extra in (p for p in model.iterdir() if p.is_file() and p.name not in KEEP):
                extra.unlink()
        sizes = sorted((r.num_reg_images() for r in models.values()), reverse=True)
        sys.stdout.write(f"colmap4: {len(sizes)} model(s), registered {sizes}\n")
    sys.stdout.write(
        f"colmap4: {parsed.command} on {pycolmap.COLMAP_version} in "
        f"{time.monotonic() - began:.1f} s\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

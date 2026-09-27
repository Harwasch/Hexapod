"""Global SfM for the `pose` stage: GLOMAP, as COLMAP 4 ships it, through pycolmap.

**What it is.** GLOMAP (Pan et al., ECCV 2024) solves every camera at once -- rotation
averaging, then a global positioning of cameras and points, then bundle adjustment --
where COLMAP's incremental `mapper` registers frames one at a time and re-runs bundle
adjustment as it grows. It was a separate project; COLMAP **4.0** (2026) merged it in as
`colmap global_mapper` (and `automatic_reconstructor --mapper GLOBAL`), and pycolmap
exposes the same pipeline as `pycolmap.global_mapping` beside `calibrate_view_graph`
(COLMAP's `view_graph_calibrator`).

**How it gets onto the CPU box, and why this way.** Ubuntu 24.04's `colmap` package is
3.9.1, which has no global mapper, and building COLMAP 4 from source in the Modal image
means CMake, Ceres, Boost, CGAL, FLANN/faiss and a 15-25 minute compile on every image
change. **pycolmap publishes prebuilt manylinux wheels** (4.2.0: CPython 3.10-3.14,
x86-64, glibc >= 2.28; the CPU box is Ubuntu 24.04 with glibc 2.39), so the image gets it
with one `pip install pycolmap==4.2.0` -- about 135 MB unpacked, seconds to install --
into the Python the pipeline already runs under. Nothing else changes: extraction and
matching stay on the apt COLMAP 3.9.1 that every pose measurement here was made with (its
vocabulary tree included, whose FLANN format 3.11+ cannot read), and pycolmap 4.2 opens the
3.9.1 database, upgrading a *copy* of it (it adds the rig and frame tables 3.12 introduced).
Its model is COLMAP's usual `cameras.bin`/`images.bin`/`points3D.bin` -- plus `rigs.bin`
and `frames.bin`, which are removed so the `poses` artifact stays a model 3.9.1's
`model_aligner` reads exactly as it reads its own (checked: `model_analyzer` 3.9.1 reads
the global model).

It runs in a **subprocess** (this file is also a script) rather than imported: a C++
failure in a new mapper is then a failed attempt the stage falls back from, not a dead
stage process, and the pipeline does not need pycolmap to import.

**Measured here, on 4 CPU cores** (2026-09-27, the committed tree rendered on a closed
40-frame orbit, the same COLMAP 3.9.1 database for both, scored against the known poses
after a similarity fit):

    mapper                       registered  median rot  median trans   map time
    incremental (3.9.1)          40/40       0.122 deg   0.207% extent  3.0 s
    global (pycolmap 4.2.0)      40/40       0.104 deg   0.172% extent  1.7 s
    global + calibrate_view_graph 40/40      9-100 deg   17-91% extent  1.0-1.8 s

The last row is why `calibrate_view_graph` is **not** run, although COLMAP's docs
recommend it where focal priors are missing: on this scene (a textured ground plane under
a tree) it re-estimated the pairwise geometry into reconstructions that registered every
frame, had sub-pixel reprojection error, every point in front of its cameras -- and were
wrong, differently on each run. Registration would not have caught it. Without it the
result was identical over three runs and better than incremental's.

The fixture is small enough that both mappers take seconds; the 67 s the real 87-frame
video spent in incremental mapping is where a global mapper's saving would be, and it is
**unmeasured on a real capture**. So `mapper: global` is opt-in, and a global result that
registers fewer than `min_registered_fraction` of the frames falls back to the incremental
mapper on the same matches, in the same stage.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

__all__ = ["KEEP", "PYCOLMAP_VERSION", "PYTHON_ENV", "argv", "main", "python"]

#: The wheel `infra/modal/app.py` installs, and the version the numbers above are for.
PYCOLMAP_VERSION = "4.2.0"
#: An interpreter with pycolmap, when it is not the one running the pipeline.
PYTHON_ENV = "COLMAP_GLOBAL_PYTHON"
#: The files of a model the `poses` artifact keeps: COLMAP 3.9.1's own set.
KEEP = frozenset({"cameras.bin", "images.bin", "points3D.bin"})


def python() -> str:
    """`$COLMAP_GLOBAL_PYTHON`, else this interpreter (the Modal CPU image's case)."""
    return os.environ.get(PYTHON_ENV) or sys.executable


def argv(
    interpreter: str,
    database: Path,
    images: Path,
    output: Path,
    *,
    refine_focal_length: bool = True,
    random_seed: int = 0,
    num_threads: int | None = None,
) -> list[str]:
    """The command that maps `database` globally into numbered models under `output`."""
    command = [
        interpreter,
        str(Path(__file__).resolve()),
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
        command.append("--hold-focal")
    if num_threads is not None:
        command += ["--threads", str(num_threads)]
    return command


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="global SfM over a COLMAP database")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--hold-focal", action="store_true")
    parsed = parser.parse_args(args)
    try:
        import pycolmap  # type: ignore[import-not-found,unused-ignore]
    except ImportError as error:
        sys.stdout.write(
            f"global_sfm: pycolmap is not installed for {sys.executable} ({error}); the "
            f"CPU image installs pycolmap=={PYCOLMAP_VERSION}, or set ${PYTHON_ENV}\n"
        )
        return 3
    # pycolmap upgrades a 3.9.1 database's schema as it opens it; the original stays as
    # the incremental mapper, and any later matching pass, expect it.
    copy = parsed.output.parent / f"{parsed.output.name}.db"
    shutil.copyfile(parsed.database, copy)
    parsed.output.mkdir(parents=True, exist_ok=True)
    options = pycolmap.GlobalPipelineOptions()
    options.random_seed = parsed.seed
    options.num_threads = parsed.threads
    options.mapper.random_seed = parsed.seed
    options.mapper.global_positioning.use_gpu = False
    options.mapper.bundle_adjustment.ceres.use_gpu = False
    options.mapper.bundle_adjustment.refine_focal_length = not parsed.hold_focal
    options.mapper.bundle_adjustment.refine_principal_point = False
    began = time.monotonic()
    models = pycolmap.global_mapping(copy, parsed.images, parsed.output, options)
    seconds = time.monotonic() - began
    for model in sorted(p for p in parsed.output.iterdir() if p.is_dir()):
        for extra in (p for p in model.iterdir() if p.is_file() and p.name not in KEEP):
            extra.unlink()
    copy.unlink(missing_ok=True)
    sizes = sorted((r.num_reg_images() for r in models.values()), reverse=True)
    sys.stdout.write(
        f"global_sfm: pycolmap {pycolmap.__version__}: {len(sizes)} model(s), registered "
        f"{sizes} in {seconds:.1f} s\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

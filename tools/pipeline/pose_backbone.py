"""Coverage backbone and sparse register planning; no fixed frame-count cap.

The incremental estimate fits t = a*N^2 to production ecbcdda1 (954 registrations
in 6180 s, 2026-11-09). This is an extrapolation of a partial run, not a wall-time
guarantee. Registration's 1.5 s/view allowance exceeds the bench CPU smoke's
5.76/4 s (FULL_DATA.md on codex/feat-quality-campaign); 4.2 extraction/matching use
the 179-frame production measurements in README.md. All estimates exclude transfers
and retry passes, and assume 4096 features / 1600 px / octave 0 on cpu4.
"""

from __future__ import annotations

import bisect
import json
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import global_sfm
import sfm
from contracts import StageContext

MIN_FRAMES = 200
BA_ITERATIONS = 10
BA_SECONDS = 120


def enabled(mode: str, frames: int, source: str | None) -> bool:
    """Small captures keep the existing path, including videos; unknown sets are photos."""
    if mode not in ("auto", "off"):
        raise ValueError(f"backbone must be auto or off, not {mode!r}")
    return mode == "auto" and frames >= MIN_FRAMES


def estimate_seconds(
    frames: int, backbone: int, *, ordered: bool = True, engine: str = "4.2"
) -> dict[str, float]:
    """Both paths, with the selected coverage count supplied (never inferred from N)."""
    if not 0 < backbone <= frames:
        raise ValueError("pose estimate requires 0 < backbone <= frames")
    mapping = 6180.0 / 954**2
    # 49/179 seconds for sequential matching; 3.7/87 seconds extraction. Photo
    # matching has N*B registration pairs and B*(B-1)/2 backbone pairs instead.
    extract = frames * (3.7 / 87)
    selection = frames * 0.06  # keyframes.py's measured ~50-60 ms/candidate
    match_s = 49 if engine == "4.2" else 266  # README's 179-frame production runs
    pair_s = (match_s / 179) / 20  # linear + quadratic neighbourhood, ~20 pairs/view
    backbone_pairs = backbone * 20 if ordered else backbone * (backbone - 1) / 2
    register_pairs = (frames - backbone) * (8 if ordered else backbone)
    return {
        "incrementalS": extract
        + mapping * frames**2
        + pair_s * (frames * 20 if ordered else frames * (frames - 1) / 2),
        "backboneS": selection
        + extract
        + mapping * backbone**2
        + pair_s * (backbone_pairs + register_pairs)
        + 1.5 * (frames - backbone)
        + BA_SECONDS,
    }


def registration_pairs(
    names: Sequence[str], anchors: set[str], *, ordered: bool
) -> list[tuple[str, str]]:
    """Video: four solved anchors each side, as in the bench; photos: all anchors.

    Photo filenames convey no temporal relationship. All-to-backbone is conservative
    O(NB), with no native-to-native pairs or new all-frame mapper.
    """
    indices = [i for i, name in enumerate(names) if name in anchors]
    result: list[tuple[str, str]] = []
    for i, name in enumerate(names):
        if name in anchors:
            continue
        at = bisect.bisect_left(indices, i)
        neighbours = indices[max(0, at - 4) : at + 4] if ordered else indices
        result.extend((name, names[j]) for j in neighbours)
    return result


def register(
    ctx: StageContext,
    database: Path,
    frames: Path,
    model: Path,
    output: Path,
    *,
    four: bool,
    interpreter: str,
    threads: int | None,
    ordered: bool,
) -> dict[str, object]:
    """Same database and feature indices as the solved backbone; never re-keypoint it."""
    names = sorted(p.name for p in frames.iterdir() if p.is_file())
    anchors = set(sfm.read_model(model).by_name())
    pairs = registration_pairs(names, anchors, ordered=ordered)
    pair_file = ctx.work_dir / "registration-pairs.txt"
    pair_file.write_text("".join(f"{a} {b}\n" for a, b in pairs), encoding="utf-8")
    seconds: dict[str, float] = {}

    def run(phase: str, argv: list[str]) -> None:
        began = time.monotonic()
        ctx.run(argv)
        seconds[phase] = time.monotonic() - began

    if not pairs:
        shutil.copytree(model, output, dirs_exist_ok=True)
    elif four:
        report = ctx.work_dir / "registration.json"
        run(
            "registrationWall",
            [
                interpreter,
                str(Path(__file__).resolve()),
                "--database",
                str(database),
                "--model",
                str(model),
                "--output",
                str(output),
                "--pairs",
                str(pair_file),
                "--report",
                str(report),
                "--threads",
                str(threads or -1),
            ],
        )
        details = json.loads(report.read_text())
        seconds.update(details["timings"])
    else:
        run("registrationMatch", sfm.registration_matcher_argv(database, pair_file, threads))
        registered = ctx.work_dir / "registered"
        triangulated = ctx.work_dir / "triangulated"
        for path in (registered, triangulated, output):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        run("register", sfm.image_registrator_argv(database, model, registered, threads))
        run(
            "triangulate",
            sfm.point_triangulator_argv(database, frames, registered, triangulated, threads),
        )
        run("bundleAdjust", sfm.bundle_adjuster_argv(triangulated, output, threads))
    final = sfm.read_model(output)
    missing = sorted(set(names) - final.by_name().keys())
    if anchors - final.by_name().keys():
        raise ValueError("registration lost solved backbone cameras")
    return {
        "solvedBackbone": len(anchors),
        "attempted": len(names) - len(anchors),
        "registered": final.registered - len(anchors),
        "failed": len(missing),
        "failures": [
            {
                "name": name,
                "reason": "not registered by COLMAP (no usable absolute pose)",
            }
            for name in missing
        ],
        "pairs": len(pairs),
        "timings": seconds,
        "bundleMaxIterations": BA_ITERATIONS,
        "bundleMaxSeconds": BA_SECONDS if four else None,
    }


def register_pycolmap(pc: Any, parsed: Any) -> dict[str, object]:
    """4.2 equivalent of image_registrator + point_triangulator, with one bounded BA.

    Direct mapper calls deliberately avoid incremental_mapping's growing schedule of
    repeated global adjustment. Existing backbone poses and calibration remain fixed.
    """
    if pc.__version__ != "4.2.0":
        raise ValueError("backbone registration requires pycolmap==4.2.0")
    pc.set_random_seed(0)
    matching = pc.FeatureMatchingOptions()
    matching.use_gpu = False
    matching.num_threads = parsed.threads
    pairing = pc.ImportedPairingOptions()
    pairing.match_list_path = str(parsed.pairs)
    began = time.monotonic()
    pc.match_image_pairs(
        parsed.database, matching_options=matching, pairing_options=pairing, device=pc.Device.cpu
    )
    timings = {"registrationMatch": time.monotonic() - began}
    model = pc.Reconstruction(parsed.model)
    db = pc.Database.open(parsed.database)
    try:
        cache_options = pc.DatabaseCacheOptions()
        cache_options.min_num_matches = 15
        # A later rematch may leave an earlier model's anchor without verified
        # edges. Keep its database entry so beginning reconstruction preserves it.
        cache_options.load_all_images = True
        cache = pc.DatabaseCache.create(db, cache_options)
    finally:
        db.close()
    mapper = pc.IncrementalMapper(cache)
    options = pc.IncrementalMapperOptions()
    options.num_threads = parsed.threads
    options.fix_existing_frames = True
    options.constant_cameras = set(model.cameras)
    options.abs_pose_refine_focal_length = False
    options.abs_pose_refine_extra_params = False
    triangulation = pc.IncrementalTriangulatorOptions()
    mapper.begin_reconstruction(model)
    began = time.monotonic()
    # Two passes allow newly triangulated observations to rescue earlier failures;
    # no seed search or mapping of a second disconnected reconstruction.
    for _ in range(2):
        candidates = mapper.find_next_images(options, False)
        added = 0
        for image_id in candidates:
            if mapper.register_next_image(options, image_id):
                mapper.triangulate_image(triangulation, image_id)
                added += 1
        if not added:
            break
    mapper.complete_and_merge_tracks(triangulation)
    timings["registerAndTriangulate"] = time.monotonic() - began
    ba = pc.BundleAdjustmentOptions()
    ba.refine_focal_length = False
    ba.refine_extra_params = False
    ba.refine_principal_point = False
    ba.refine_sensor_from_rig = False
    ba.ceres.use_gpu = False
    ba.ceres.solver_options.num_threads = parsed.threads
    ba.ceres.solver_options.max_num_iterations = BA_ITERATIONS
    ba.ceres.solver_options.max_solver_time_in_seconds = BA_SECONDS
    began = time.monotonic()
    if not mapper.adjust_global_bundle(options, ba):
        raise ValueError("backbone bundle adjustment failed")
    mapper.filter_points(options)
    timings["bundleAdjust"] = time.monotonic() - began
    mapper.end_reconstruction(False)
    model.update_point_3d_errors()
    parsed.output.mkdir(parents=True, exist_ok=True)
    model.write(parsed.output)
    for path in parsed.output.iterdir():
        if path.name not in global_sfm.KEEP:
            path.unlink()
    return {"timings": timings}


def main() -> None:
    import argparse

    import pycolmap  # type: ignore[import-not-found,unused-ignore]

    parser = argparse.ArgumentParser()
    for flag in ("database", "model", "output", "pairs", "report"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=-1)
    parsed = parser.parse_args()
    report = register_pycolmap(pycolmap, parsed)
    parsed.report.write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()

"""The `optimise_lod` stage: the tileset's parent gaussians optimised against the photos.

`package` builds a level-of-detail tree whose parents are merged by Hierarchical 3DGS's
moment matching (tools/captures/splat_tiles.py). Merged parents are geometrically right and
visibly blurry: Phase 1 measured them ahead of the old thinned parents on PSNR, SSIM and
holes but not on LPIPS at the distance a viewer switches to them. H3DGS's answer is its
Sec. 5.1: optimise the interior nodes against the training images, leaves frozen (SmallCity
at tau = 15 px: 23.04 -> 25.68 dB). This stage does that, on the GPU, with
`lod_optimise.py` under the trainer's interpreter, and hands `package` the result.

**Where it runs, and why there.** Optimising parents needs the GPU, the photos and their
cameras, and the exact tree `package` will write -- and that tree is a function of
`canonical.ply`, which exists only after `quality` has cropped the splat (Modal's CPU box)
and `place` has moved it into east/north/up (the worker); the packer's octree is cut on
that frame's axes. So it cannot run inside `train`'s container: the gaussians it would
see are not the ones packaged. It runs as its own GPU stage between `place` and
`package`, on the very `canonical.ply` `package` reads, building the tree with the
packer's own code (`splat_tiles.hierarchy`) and `package`'s own parameters. The optimised
parents come back keyed by tile and cell and fingerprinted by the PLY's sha256
(`splat_tiles.ParentOverrides`); `package` draws them in place of the merged ones and
refuses them for any other PLY or parameters. The cameras are COLMAP's, moved by the same
similarity `place` applied to the splat (`placement.json`), so nothing is optimised in a
frame other than the one published.

**It cannot cost the capture its tileset.** Anything that goes wrong -- no trainer on this
machine, a crash, a watchdog -- is logged and recorded in `summary.json` with status
`failed`; `package` then packs the merged parents, as before this stage existed. So is a
run whose held-out loss did not fall (`rejected`). `enabled: false` turns it off.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import training
from artifacts import ArtifactDecl
from captures_bridge import TILE_GAUSSIANS
from contracts import MetricValue, StageContext, StageOutcome
from registry import stage_impl

__all__ = ["LOD_PARENTS", "PARENTS_FILE", "PLACEMENT", "SCRIPT", "SUMMARY_FILE", "placement"]

#: The GPU half, run with the trainer's interpreter; beside this file in the image too.
SCRIPT = Path(__file__).resolve().parent / "lod_optimise.py"
SUMMARY_FILE = "summary.json"
PARENTS_FILE = "lod_parents.npz"

LOD_PARENTS = ArtifactDecl(
    "lod_parents",
    kind="dir",
    content_type="inode/directory",
    summary="the tileset's parent gaussians optimised against the photos (H3DGS Sec. 5.1), "
    "keyed by tile and cell, for package to draw in place of the merged ones; summary.json "
    "says whether they were kept and what they measured",
    required_members=(SUMMARY_FILE,),
    stub_members=(SUMMARY_FILE,),
)
PLACEMENT = ArtifactDecl(
    "placement.json",
    content_type="application/json",
    summary="the similarity place applied, x' = scale * rotation @ x + translation, from "
    "the trained splat's (COLMAP) frame to canonical.ply's",
)

#: Seconds the optimisation loop may run before it stops and evaluates what it has; the
#: script's own watchdog ends it 15 minutes after that whatever it is doing. An hour of
#: L4 is $0.80 (providers.py); the planned run is a fraction of it (lod_maths).
DEFAULT_BUDGET_S = 3600.0


def placement(rotation: Any, translation: Any, scale: float, then: Any | None) -> dict[str, object]:
    """`place`'s two steps as one similarity: `scale * R x + t`, then (when it recentred)
    `s2 * R2 y + t2` -- `scale * s2`, `R2 R`, `s2 R2 t + t2`."""
    import numpy as np

    r = np.asarray(rotation, dtype=np.float64)
    t = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
    s = float(scale)
    if then is not None:
        r2 = np.asarray(then.rotation, dtype=np.float64)
        s2 = float(getattr(then, "scale", 1.0))
        t2 = np.asarray(then.translation, dtype=np.float64)
        r, t, s = r2 @ r, s2 * (r2 @ t) + t2, s * s2
    return {
        "scale": s,
        "rotation": [[float(v) for v in row] for row in r],
        "translation": [float(v) for v in t],
        "from": "the trained splat's frame (COLMAP)",
        "to": "canonical.ply (east/north/up about the placed origin)",
    }


@stage_impl(
    "h3dgs_parents",
    consumes=("canonical.ply", PLACEMENT.name, "frames", "poses"),
    produces=(LOD_PARENTS,),
    summary="optimise the level-of-detail parents against the photos (H3DGS Sec. 5.1)",
)
def h3dgs_parents(ctx: StageContext) -> StageOutcome:
    """Real: run `lod_optimise.py` on the GPU and keep what it wrote in `lod_parents/`.

    Params: `enabled` (true); `tile_gaussians` and `opacity_min`, which must be
    `package`'s (the recipe shares them by YAML anchor; the packer refuses a mismatch);
    `tau_min`/`tau_max` (3 / 64 px); `iterations` (0: planned from the tree, at most
    `max_iterations`, 15,000); `budget_s`; `trainer` and `python` as `train` takes them.
    """
    out = ctx.output(LOD_PARENTS.name)
    if not _flag(ctx.param("enabled", True)):
        return _outcome(ctx, out, {"status": "off", "reason": "enabled is false"})
    scratch = ctx.work_dir / "lod"
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        dataset = training.build_dataset(
            ctx.input("frames"), ctx.input("poses"), ctx.work_dir / "dataset"
        )
        # `script` is for tests: a stand-in with this argv and none of the GPU.
        argv = [
            training.trainer_python(ctx.param("python")),
            str(ctx.param("script") or SCRIPT),
            "--ply",
            str(ctx.input("canonical.ply")),
            "--data_dir",
            str(dataset),
            "--placement",
            str(ctx.input(PLACEMENT.name)),
            "--out",
            str(scratch),
            "--trainer",
            str(training.trainer_script(ctx.param("trainer"))),
            "--opacity_min",
            f"{float(ctx.param('opacity_min', 0.02)):g}",
            "--tile_gaussians",
            str(int(ctx.param("tile_gaussians", TILE_GAUSSIANS))),
            "--budget-s",
            f"{float(ctx.param('budget_s', DEFAULT_BUDGET_S)):g}",
        ]
        for name in ("tau_min", "tau_max", "iterations", "max_iterations", "seed"):
            value = ctx.param(name)
            if value is not None:
                argv += [f"--{name}", f"{value:g}" if isinstance(value, float) else str(value)]
        ctx.run(argv)
        summary = json.loads((scratch / SUMMARY_FILE).read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError(f"{SUMMARY_FILE} is not a JSON object")
        parents = scratch / PARENTS_FILE
        if summary.get("status") == "ok":
            if not parents.is_file():
                raise FileNotFoundError(f"status ok and no {PARENTS_FILE}")
            shutil.copyfile(parents, out / PARENTS_FILE)
    # Anything at all: nothing in here may cost the capture its tileset.
    except Exception as problem:
        reason = f"{type(problem).__name__}: {problem}"[:600]
        ctx.log(
            f"WARNING: the parents were not optimised ({reason}). package will draw the "
            f"merged parents, as it did before this stage"
        )
        (out / PARENTS_FILE).unlink(missing_ok=True)
        summary = {"status": "failed", "reason": reason}
    return _outcome(ctx, out, summary)


def _outcome(ctx: StageContext, out: Path, summary: dict[str, Any]) -> StageOutcome:
    (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    status = str(summary.get("status"))
    metrics: dict[str, MetricValue] = {"status": status}
    for key in ("iterationsRun", "parentGaussians", "parentTiles", "loopSeconds", "itPerSecond"):
        if isinstance(summary.get(key), int | float):
            metrics[key] = summary[key]
    for when in ("before", "after"):
        block = summary.get(when)
        if isinstance(block, dict):
            for key in ("loss", "psnr", "lpips"):
                if isinstance(block.get(key), int | float):
                    metrics[f"{when}{key.capitalize()}"] = block[key]
    switch = summary.get("switchDistance")
    if isinstance(switch, dict) and isinstance(switch.get("mean"), dict):
        for name in ("merged", "optimised"):
            value = switch["mean"].get(name, {}).get("lpips")
            if isinstance(value, int | float):
                metrics[f"switchLpips{name.capitalize()}"] = value
    ctx.log(f"parents: {status}: {summary.get('reason', '')}")
    return StageOutcome(metrics=metrics, summary=f"parents {status}")


def _flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"enabled must be true or false, not {value!r}")

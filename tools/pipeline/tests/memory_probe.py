"""Peak memory of the stages after training, measured in a process of their own.

Run by `tests/test_bounded_memory.py` (and by hand, to measure) as

    python tests/memory_probe.py make     ROOT N CAMERAS [--held] [--sh3]
    python tests/memory_probe.py chunked  ROOT STAGE... [--chunk ROWS] [--small] [--limit-mb MB]
    python tests/memory_probe.py whole    ROOT
    python tests/memory_probe.py whole-lane1 ROOT

`make` writes a synthetic orbit capture of N gaussians under ROOT/inputs, a million rows at
a time (so making an 8M-gaussian splat does not itself need gigabytes): `trained.ply` in the
canonical fourteen properties, or with `--sh3` an `upload/capture.ply` in gsplat's own
degree-3 export layout (59 floats a row), which is what a phone app's Lane 1 upload is.
`chunked` runs the named stages -- quality, place, thumbnail, ground, normalize -- through
the executor, as the worker does. `whole` runs the frozen whole-splat quality stage
(`quality_in_memory.py`) and the whole-splat place; `whole-lane1` the whole-splat ingest,
thumbnail and ground samples (`gaussians.read_splat`, `orient`, `render_thumbnail`,
`ground_samples`) -- the before of the before/after. Each prints one JSON line: the peak
resident set (`VmHWM`) of the process, which is what a machine's memory limit bounds,
and the seconds taken.

`--small` shrinks the fixed-size buffers -- `outofcore`'s partition and collection sizes
(1M rows each by default), quality's occluder block (2^18) and its in-memory depth samples
(32 MB) -- to 64k, 32k and 1 MB, so that a small splat already exercises the regime a
large one runs in: what the always-on scaling test needs to show that memory does not
grow with the splat on sizes a test can afford.
`--limit-mb` caps the address space (RLIMIT_AS), so exceeding it is an allocation failure
rather than a number.
"""

from __future__ import annotations

import json
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

STAGES: dict[str, dict[str, Any]] = {
    "quality": {"id": "quality", "impl": "support_gate", "params": {"bar": "balanced"}},
    "place": {"id": "place", "impl": "place_splat", "params": {}},
    "normalize": {"id": "normalize", "impl": "ingest_splat", "params": {"up_axis": "z"}},
    "thumbnail": {"id": "thumbnail", "impl": "splat_thumbnail", "params": {}},
    "ground": {"id": "ground", "impl": "splat_ground", "params": {"cell_m": 0.5}},
}


def peak_mb() -> float:
    """This process's own peak resident set, in MB.

    `VmHWM` from /proc, not `ru_maxrss`: Linux carries `ru_maxrss` across `exec` from the
    process that forked it, so a probe started by a 160 MB pytest reported 160 MB however
    little it used itself. `VmHWM` belongs to the address space `exec` made.
    """
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB on Linux


def make(root: Path, count: int, cameras: int, *, held: bool, sh3: bool) -> None:
    import numpy as np

    import holdout_maths
    import splat_io
    from gaussians import CANONICAL_PROPERTIES
    from synthetic_scene import ring, scene, write_model

    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    step = 1_000_000
    rest = [f"f_rest_{i}" for i in range(45)]
    if sh3:
        (inputs / "upload").mkdir(exist_ok=True)
        path = inputs / "upload" / "capture.ply"
        names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", *rest, "opacity"]
        names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    else:
        path, names = inputs / "trained.ply", list(CANONICAL_PROPERTIES)
    rng = np.random.default_rng(0)
    with splat_io.PlyWriter(path, names, count=count) as out:
        for index, start in enumerate(range(0, count, step)):
            columns, _ = scene(min(step, count - start), seed=index)
            size = int(columns["x"].shape[0])
            noise = {name: rng.normal(0, 0.1, size=(size,)).astype(np.float32) for name in rest}
            columns.update(noise)
            out.append(columns)
    write_model(inputs / "poses", ring(cameras, radius=1.5, height=1.0))
    (inputs / "train_metrics.json").write_text(json.dumps({"psnr": 22.0}))
    frame = {"source": "camera-up", "scale": 1.0, "rotation": np.eye(3).tolist(), "recentre": True}
    (inputs / "georef.json").write_text(json.dumps({"lat": 1.0, "lon": 2.0, "frame": frame}))
    if held:
        error = rng.gamma(2.0, 0.03, count).astype(np.float32)
        weight = rng.exponential(4.0, count).astype(np.float32)
        holdout_maths.write(inputs / "holdout", error, weight, None, {"status": "ok"})


def chunked(root: Path, stages: list[str], chunk: int | None, small: bool) -> dict[str, Any]:
    import outofcore
    import quality
    from conftest import make_recipe
    from executor import execute
    from runners import RunnerSet
    from workdir import Workdir

    if small:
        outofcore.BUDGET = 1 << 16
        outofcore.COLLECT = 1 << 16
        quality.OCCLUDER_BLOCK = 1 << 15
        quality.SAMPLE_BYTES = 1 << 20
    inputs = root / "inputs"
    workdir = Workdir.create(Path(tempfile.mkdtemp(prefix=f"run-{'-'.join(stages)}-", dir=root)))
    names = []
    for path in sorted(inputs.iterdir()):
        target = workdir.input_path(path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path)
        names.append(path.name)
    if "normalize" in stages:
        # Lane 1's thumbnail and ground samples read what `normalize` wrote.
        names = [name for name in names if name in ("upload", "georef.json")]
    recipe = []
    for stage in stages:
        entry = json.loads(json.dumps(STAGES[stage]))
        if chunk is not None:
            entry["params"]["chunk_gaussians"] = chunk
        recipe.append(entry)
    started = time.perf_counter()
    execute(make_recipe(recipe, inputs=names), workdir, RunnerSet.local())
    return {"seconds": round(time.perf_counter() - started, 1)}


def whole(root: Path) -> dict[str, Any]:
    import numpy as np

    import gaussians
    import quality_in_memory

    inputs = root / "inputs"
    out = root / "whole"
    out.mkdir(parents=True, exist_ok=True)

    class Context:
        params: dict[str, Any] = {"bar": "balanced"}

        def param(self, name: str, default: Any = None) -> Any:
            return self.params.get(name, default)

        def input(self, name: str) -> Path:
            return inputs / name

        def has_input(self, name: str) -> bool:
            return (inputs / name).exists()

        def output(self, name: str) -> Path:
            return out / name

        def log(self, message: str) -> None:
            pass

    started = time.perf_counter()
    quality_in_memory.support_gate(cast(Any, Context()))
    quality_peak = peak_mb()
    trained = gaussians.read_splat(out / "gated.ply")
    placed, _ = gaussians.orient(
        gaussians.Splat(
            columns=gaussians.transform(trained.columns, np.eye(3)),
            source_format="ply",
            source_name="gated.ply",
            source_bytes=0,
            source_checksum="",
            properties_in=(),
            dropped=(),
            non_finite=0,
        ),
        up_axis="z",
    )
    gaussians.write_ply(out / "canonical.ply", placed.columns)
    return {"seconds": round(time.perf_counter() - started, 1), "qualityPeakMb": quality_peak}


def whole_lane1(root: Path) -> dict[str, Any]:
    """Lane 1's `normalize`, `thumbnail` and `ground_samples` as they were: whole."""
    import gaussians

    out = root / "whole-lane1"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    read = gaussians.read_splat(root / "inputs" / "upload" / "capture.ply")
    splat, _ = gaussians.orient(read, up_axis="z")
    gaussians.write_ply(out / "canonical.ply", splat.columns)
    del read, splat
    canonical = gaussians.read_splat(out / "canonical.ply")
    gaussians.render_thumbnail(canonical, out / "thumbnail.jpg")
    gaussians.ground_samples(canonical, lat=1.0, lon=2.0, cell_m=0.5)
    return {"seconds": round(time.perf_counter() - started, 1)}


def main(argv: list[str]) -> int:
    command, root = argv[0], Path(argv[1])
    if command == "make":
        make(root, int(argv[2]), int(argv[3]), held="--held" in argv, sh3="--sh3" in argv)
        return 0
    limit = next((int(argv[i + 1]) for i, a in enumerate(argv) if a == "--limit-mb"), None)
    if limit is not None:
        cap = limit * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    if command == "chunked":
        chunk = next((int(argv[i + 1]) for i, a in enumerate(argv) if a == "--chunk"), None)
        flags = {"--chunk", "--limit-mb", "--small"}
        values = {argv[i + 1] for i, a in enumerate(argv) if a in ("--chunk", "--limit-mb")}
        stages = [a for a in argv[2:] if a not in flags and a not in values]
        result = chunked(root, stages, chunk, "--small" in argv)
    elif command == "whole":
        result = whole(root)
    elif command == "whole-lane1":
        result = whole_lane1(root)
    else:
        raise SystemExit(f"unknown command {command!r}")
    result["peakMb"] = round(peak_mb(), 1)
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

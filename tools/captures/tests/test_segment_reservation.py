"""How a segmentation run is sized from its scan, and how it says what it used.

`infra/modal/segment.py` estimates each run's cores and memory from the scan's tileset.json
(`sizing`) and spawns it on `with_options` with (request, limit) pairs; `segment_scene`
derives its render processes from the request it is told (`--cpus`, `--memory-gb`;
`default_workers`), because inside a Modal container the host's cores and memory are what
show, not what was paid for; and each run logs its peaks (`peak_usage`, `call_usage`) so
the estimate can be tuned. Both halves are checked here without Modal: segment_scene
directly, the app file by importing it against a stand-in `modal` module that records what
the decorators and `with_options` were given.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import segment_scene as ss
from splat_render import Splats

GIB = float(1 << 30)
SEGMENT_APP = Path(__file__).resolve().parents[3] / "infra" / "modal" / "segment.py"
#: The published scans' leaf gaussians and tiles (their tileset.json, 2026-10-03).
SPOOL, PUMPKIN, CAMP = 153_566, 387_813, 22_577_243


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A 64-core host with no cgroup memory limit, this process holding 6 GB."""
    state: dict[str, Any] = {"cores": 64, "room": None, "held": 6e9}
    monkeypatch.setattr(ss.os, "sched_getaffinity", lambda _pid: set(range(state["cores"])))
    monkeypatch.setattr(ss, "_memory_room", lambda: state["room"])
    monkeypatch.setattr(ss, "_resident_bytes", lambda: state["held"])
    return state


def test_without_a_reservation_the_host_is_what_shows(host: dict[str, Any]) -> None:
    """The case the reservation exists for: 64 renders on a container that paid for 6."""
    assert ss.default_workers() == 64


def test_the_reserved_cores_less_the_main_process_bound_the_renders(
    host: dict[str, Any],
) -> None:
    # 32 GiB less the 6 GB held is room for 11 renders at RENDER_WORKER_BYTES; of 8 cores,
    # MAIN_PROCESS_CORES are the masking process's.
    assert ss.MAIN_PROCESS_CORES == 2
    assert ss.default_workers(8, 32 * GIB) == 6
    assert ss.default_workers(6, 32 * GIB) == 4
    # A reservation of fewer cores than that still renders, one view at a time.
    assert ss.default_workers(1, 32 * GIB) == 1


def test_the_memory_left_after_the_scan_bounds_them_too(host: dict[str, Any]) -> None:
    # 16 GiB less 6 GB held: 11.2 GB, four renders of 2.5 GB.
    assert ss.default_workers(8, 16 * GIB) == 4
    # A scan that fills the reservation still renders, one view at a time.
    host["held"] = 40e9
    assert ss.default_workers(8, 32 * GIB) == 1


def test_a_tighter_cgroup_wins_over_the_reservation(host: dict[str, Any]) -> None:
    host["room"] = 5.5e9
    assert ss.default_workers(8, 32 * GIB) == 2


def test_a_render_of_a_huge_scan_takes_more_room(host: dict[str, Any]) -> None:
    """Up to ~62M gaussians a render holds RENDER_WORKER_BYTES (its samples are capped);
    past that a whole-scan view's culling, RENDER_BYTES_PER_GAUSSIAN a gaussian, is more."""
    assert ss.render_worker_bytes(CAMP) == ss.RENDER_WORKER_BYTES
    assert ss.render_worker_bytes(100_000_000) == 100_000_000 * ss.RENDER_BYTES_PER_GAUSSIAN
    # 16 GiB less 6 GB: 11.2 GB, two renders of 4 GB (100M gaussians), not four of 2.5.
    assert ss.default_workers(8, 16 * GIB, ss.render_worker_bytes(100_000_000)) == 2


def test_the_command_line_carries_the_reservation(
    host: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def stop(*_args: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(ss, "segment", stop)
    monkeypatch.setattr(ss, "load_source", lambda *_a: (None, None, 0))
    monkeypatch.setattr(ss, "load_embedder", lambda _name: None)
    monkeypatch.setattr(ss, "load_masks", lambda _name: None)
    argv = ["segment_scene.py", str(tmp_path / "s.ply"), str(tmp_path), "--masks", "m:M"]
    monkeypatch.setattr(sys, "argv", [*argv, "--cpus", "8", "--memory-gb", "32"])
    with pytest.raises(SystemExit):
        ss.main()
    assert seen["cpus"] == 8 and seen["memory_bytes"] == 32 * GIB and seen["workers"] is None


# --- what a run used ---------------------------------------------------------------------


def test_peak_usage_reports_this_process_its_workers_and_the_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Peak resident sets (this process; the largest child once it was waited for; what a
    forked render worker held of its own), CPU seconds, wall time and cores busy; the
    container's peak from the first cgroup file that has it."""
    peak = tmp_path / "memory.peak"
    peak.write_text(f"{3 * (1 << 30)}\n", encoding="utf-8")
    monkeypatch.setattr(ss, "CGROUP_PEAK_FILES", (str(tmp_path / "missing"), str(peak)))
    monkeypatch.setitem(ss._WORKER_PEAK, "bytes", 1.5 * GIB)
    # A child that holds ~200 MB.
    code = "import numpy; a = numpy.ones(25_000_000); print(a.sum())"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
    usage = ss.peak_usage(started=100.0, now=150.0)
    assert set(usage) == {
        "wallS",
        "mainPeakGiB",
        "workerPeakGiB",
        "workerResidentGiB",
        "containerPeakGiB",
        "mainCpuS",
        "workerCpuS",
        "coresUsed",
    }
    assert usage["wallS"] == 50.0 and usage["containerPeakGiB"] == 3.0
    assert usage["mainPeakGiB"] > 0.0 and usage["workerPeakGiB"] == 1.5
    if sys.platform.startswith("linux"):
        assert usage["workerResidentGiB"] is not None and usage["workerResidentGiB"] >= 0.18
    assert usage["workerCpuS"] > 0.0
    cpu = usage["mainCpuS"] + usage["workerCpuS"]
    assert usage["coresUsed"] == pytest.approx(cpu / 50.0, abs=0.01)
    monkeypatch.setattr(ss, "CGROUP_PEAK_FILES", (str(tmp_path / "missing"),))
    monkeypatch.setitem(ss._WORKER_PEAK, "bytes", 0.0)
    unforked = ss.peak_usage(started=100.0, now=150.0)
    assert unforked["containerPeakGiB"] is None and unforked["workerPeakGiB"] is None
    line = ss.usage_line({**usage, "workerPeakGiB": None})
    assert line.startswith("usage: peak ") and "3 GiB container" in line
    assert "? a render worker" in line and "over 50.0 s" in line


#: A forked render of a small scan, while the process holds ~0.4 GB more than the scan.
FORKED_RENDERS = """
import json
import numpy as np
import segment_scene as ss
from splat_render import Camera, Splats

rng = np.random.default_rng(0)
n = 20_000
splats = Splats(
    rng.uniform(-1, 1, (n, 3)), np.tile([1.0, 0, 0, 0], (n, 1)), np.full((n, 3), 0.03),
    rng.uniform(0, 1, (n, 3)), np.full(n, 0.8),
)
held = np.ones(50_000_000)
cameras = [
    Camera.look_at([3 * np.cos(a), 3 * np.sin(a), 1.0], [0, 0, 0], width=96, height=64)
    for a in (0.0, 2.0, 4.0)
]
views = list(ss.render_views(splats, cameras, np.zeros(n, np.int64), workers=2))
print(json.dumps({
    "views": len(views), "held": float(held[-1]), "own": ss._WORKER_PEAK["bytes"],
    "workerPeakGiB": ss.peak_usage(0.0, 1.0)["workerPeakGiB"],
}))
"""


def test_forked_renders_report_what_each_holds_of_its_own(
    fresh_process: Callable[[str], dict],
) -> None:
    """A forked worker's resident set starts with every page it shares with this process;
    what it holds of its own is its peak over that, kept for `peak_usage`. (The workers fork
    from a new interpreter, not this one: tests/conftest.py.)"""
    out = fresh_process(FORKED_RENDERS)
    assert out["views"] == 3 and out["held"] == 1.0
    # The ~0.4 GB the process holds is shared, not counted.
    assert 0 < out["own"] < 0.3 * GIB
    assert out["workerPeakGiB"] == round(out["own"] / GIB, 2)


def test_the_cli_writes_its_usage_into_the_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--summary` gets the printed summary, with the run's `usage` (and the scan's size);
    one `usage:` line is printed before it."""
    splats = Splats(
        np.zeros((2, 3)), np.tile([1.0, 0, 0, 0], (2, 1)), np.full((2, 3), 0.01),
        np.full((2, 3), 0.5), np.ones(2),
    )  # fmt: skip
    lifted = ss.Lifted(np.zeros(1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64))
    lifted.stats.update(views=3, renderWorkers=[4, 4])
    result = types.SimpleNamespace(
        instances=[], splat_id=np.array([1, 0]), lifted=lifted, views=[],
        timings={"masksS": 1.234},
    )  # fmt: skip
    monkeypatch.setattr(ss, "segment", lambda *_a, **_k: result)
    monkeypatch.setattr("splat_render.load_tileset", lambda _path: splats)
    monkeypatch.setattr(ss, "load_embedder", lambda _name: ss.FakeEmbedder())
    monkeypatch.setattr(ss, "load_masks", lambda _name: None)
    monkeypatch.setattr(ss, "make_renderer", lambda _name: None)
    monkeypatch.setattr(ss, "tile_binding_by_position", lambda *_a: {})
    monkeypatch.setattr(ss.rebind_instances, "rebind", lambda _tiles, tiles: tiles)
    monkeypatch.setattr(ss, "instances_document", lambda *_a, **_k: {})
    monkeypatch.setattr(ss, "write_instances", lambda *_a: None)
    summary = tmp_path / "out" / "run.json"
    argv = ["segment_scene.py", str(tmp_path / "tileset.json"), str(tmp_path), "--masks", "m:M"]
    out = ["--out", str(tmp_path / "out"), "--summary", str(summary)]
    monkeypatch.setattr(sys, "argv", [*argv, *out])
    ss.main()
    written = json.loads(summary.read_text(encoding="utf-8"))
    assert written["renderWorkers"] == [4, 4] and written["timingsS"] == {"masksS": 1.23}
    usage = written["usage"]
    assert usage["gaussians"] == 2
    assert {"wallS", "mainPeakGiB", "workerPeakGiB", "coresUsed"} <= set(usage)
    printed = capsys.readouterr().out
    assert printed.count("usage: peak ") == 1
    assert json.loads(printed[printed.index("{") :]) == written


# --- infra/modal/segment.py, against a stand-in `modal` ----------------------------------


class _Chain:
    """`modal.Image`'s builder: every call returns the builder."""

    def __getattr__(self, _name: str) -> Any:
        return lambda *_a, **_k: self


class _Call:
    def __init__(self, result: dict | Exception) -> None:
        self.result = result

    def get(self) -> dict:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _Function:
    def __init__(self, fn: Any, options: dict[str, Any], spawned: list | None = None) -> None:
        self.fn, self.options = fn, options
        self.spawned: list = [] if spawned is None else spawned

    def with_options(self, **options: Any) -> _Function:
        return _Function(self.fn, {**self.options, **options}, self.spawned)

    def spawn(self, *args: Any) -> _Call:
        """Records the options and arguments; the call 'returns' a summary that used 80%
        of its request -- or, for a scan named "lost", raises as Modal does for a container
        killed at its memory limit."""
        self.spawned.append((self.options, args))
        if args[0] == "lost":
            return _Call(RuntimeError("container killed: out of memory"))
        plan = args[6]
        usage = {"mainPeakGiB": 0.8 * plan["memoryMiB"] / 1024, "coresUsed": 4.5}
        result = {"name": args[0], "ok": True, "sizing": plan, "usage": usage, "totalS": 9.0}
        return _Call({**result, "files": {"instances.json": b"{}"}, "log": "ran"})


class _App:
    def __init__(self, name: str) -> None:
        self.name = name
        self.functions: dict[str, _Function] = {}

    def function(self, **options: Any) -> Any:
        def register(fn: Any) -> _Function:
            self.functions[fn.__name__] = _Function(fn, options)
            return self.functions[fn.__name__]

        return register

    def local_entrypoint(self) -> Any:
        return lambda fn: fn


def _segment_app(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    modal = types.ModuleType("modal")
    modal.App = _App  # type: ignore[attr-defined]
    modal.Image = _Chain()  # type: ignore[attr-defined]
    modal.Volume = types.SimpleNamespace(from_name=lambda *_a, **_k: object())  # type: ignore[attr-defined]
    modal.is_local = lambda: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal", modal)
    spec = importlib.util.spec_from_file_location("segment_app", SEGMENT_APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tileset(leaves: list[int | None], parent: int = 500) -> dict:
    """A root parent over `leaves` (their `extras.gaussians`; None: no extras)."""
    children = [
        {"content": {"uri": f"t{k}.glb"}, **({} if g is None else {"extras": {"gaussians": g}})}
        for k, g in enumerate(leaves)
    ]
    root = {"content": {"uri": "root.glb"}, "extras": {"gaussians": parent}}
    return {"root": {**root, "children": [{"children": children}]}}


def test_the_sizing_constants_match_segment_scenes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app cannot import segment_scene where `modal run` starts (no numpy there), so it
    keeps its own copies of what the estimate needs; they must agree."""
    app = _segment_app(monkeypatch)
    assert app.RENDER_WORKER_BYTES == ss.RENDER_WORKER_BYTES
    assert app.RENDER_BYTES_PER_GAUSSIAN == ss.RENDER_BYTES_PER_GAUSSIAN
    assert app.MAIN_PROCESS_CORES == ss.MAIN_PROCESS_CORES
    assert app.MAX_VIEWS == ss.MAX_VIEWS and app.COVERAGE_VIEWS == ss.COVERAGE_VIEWS
    assert (app.VIEW_WIDTH, app.VIEW_HEIGHT) == (ss.VIEW_WIDTH, ss.VIEW_HEIGHT)
    assert app.CGROUP_PEAK_FILES == ss.CGROUP_PEAK_FILES
    for gaussians in (0, SPOOL, CAMP, 62_500_001, 400_000_000):
        assert app.render_worker_bytes(gaussians) == ss.render_worker_bytes(gaussians)


def test_a_tileset_says_its_tiles_and_gaussians(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _segment_app(monkeypatch)
    assert app.scan_size(_tileset([1000, 2500])) == {
        "tiles": 3,
        "gaussians": 3500,
        "parentGaussians": 500,
    }
    # A leaf without a count is taken at the package stage's most.
    assert app.scan_size(_tileset([1000, None]))["gaussians"] == 1000 + app.TILE_GAUSSIANS


def test_the_estimate_grows_with_the_scan_and_the_views(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _segment_app(monkeypatch)
    sizes = [0, SPOOL, PUMPKIN, 5_000_000, CAMP, 45_000_000, 100_000_000, 300_000_000]
    plans = [app.sizing(g) for g in sizes]
    memory = [p["memoryMiB"] for p in plans]
    assert memory == sorted(memory) and memory[-1] > memory[4] > memory[0]
    terms = [p["termsGiB"]["scan"] for p in plans]
    assert terms == sorted(terms)
    # The cores follow the GPU's pace, not the scan.
    assert {p["cores"] for p in plans} == {app.CPU_CORES} == {6.0}
    # More views kept, more memory.
    few = app.sizing(CAMP, views=24, coverage_rounds=0)
    many = app.sizing(CAMP, views=24, coverage_rounds=4)
    assert few["views"] == app.MAX_VIEWS and many["views"] == app.MAX_VIEWS + 4 * 96
    assert few["memoryMiB"] < many["memoryMiB"]
    assert app.planned_views(600, 2) == 600 + 2 * app.COVERAGE_VIEWS
    # The camp: base, scan, views and four renders, rounded up to a GiB.
    camp = app.sizing(CAMP)
    terms = camp["termsGiB"]
    assert set(terms) == {"base", "scan", "views", "workers"}
    assert camp["memoryMiB"] / 1024 >= sum(terms.values()) > camp["memoryMiB"] / 1024 - 1.01
    assert (camp["cores"], camp["memoryMiB"], camp["workers"]) == (6.0, 22 * 1024, 4)


def test_the_estimate_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _segment_app(monkeypatch)
    # The largest request, and its limit, with the renders it still holds.
    huge = app.sizing(10_000_000_000)
    assert huge["memoryMiB"] == app.MAX_MEMORY_MIB
    assert huge["memoryLimitMiB"] == app.MAX_MEMORY_LIMIT_MIB
    assert huge["workers"] == 1
    # The smallest, once less is held than it.
    monkeypatch.setattr(app, "BASE_BYTES", 0)
    tiny = app.sizing(0, views=0, coverage_rounds=0)
    assert sum(tiny["termsGiB"].values()) < 12
    assert tiny["memoryMiB"] == app.MIN_MEMORY_MIB == 16 * 1024
    assert tiny["memoryLimitMiB"] == 2 * app.MIN_MEMORY_MIB
    monkeypatch.setattr(app, "CPU_CORES", 1.0)
    assert app.sizing(CAMP)["cores"] == app.MIN_CPU_CORES == 4.0
    monkeypatch.setattr(app, "CPU_CORES", 64.0)
    wide = app.sizing(CAMP)
    assert wide["cores"] == app.MAX_CPU_CORES == 16.0
    assert wide["coresLimit"] == 16.0 * app.CPU_HEADROOM


def test_the_request_and_its_ceiling_go_to_with_options(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _segment_app(monkeypatch)
    plan = app.sizing(CAMP)
    assert app.options(plan) == {
        "gpu": "L4",
        "cpu": (6.0, 9.0),
        "memory": (22 * 1024, 44 * 1024),
    }
    for gaussians in (SPOOL, PUMPKIN, CAMP, 100_000_000):
        options = app.options(app.sizing(gaussians))
        cpu, memory = options["cpu"], options["memory"]
        assert cpu[1] >= cpu[0] and memory[1] >= memory[0]
        assert all(isinstance(m, int) for m in memory)  # modal wants MiB as ints
        assert memory[1] == min(2 * memory[0], app.MAX_MEMORY_LIMIT_MIB)


def test_the_function_reserves_the_camps_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call spawned without `with_options` is sized as the camp, as (request, limit)."""
    app = _segment_app(monkeypatch)
    options = app.app.functions["segment_scan"].options
    assert options["gpu"] == "L4"
    assert app.DEFAULT_SIZING == app.sizing(app.FALLBACK_GAUSSIANS) == app.sizing(CAMP)
    assert options["cpu"] == app.options(app.DEFAULT_SIZING)["cpu"] == (6.0, 9.0)
    assert options["memory"] == app.options(app.DEFAULT_SIZING)["memory"]


def test_the_workers_come_from_the_request_not_the_limit(
    host: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The command line carries the request; segment_scene's `default_workers`, holding
    what the estimate says it holds, forks the estimate's `workers` from it -- never what
    the limit would make room for."""
    app = _segment_app(monkeypatch)
    for gaussians in (SPOOL, CAMP, 100_000_000):
        plan = app.sizing(gaussians)
        argv = app.segment_argv(
            Path("/w/tiles"), Path("/w"), views=24, cpus=plan["cores"], memory_mib=plan["memoryMiB"]
        )
        cpus = int(argv[argv.index("--cpus") + 1])
        memory = float(argv[argv.index("--memory-gb") + 1]) * GIB
        assert (cpus, memory) == (plan["cores"], plan["memoryMiB"] * 1024 * 1024)
        terms = plan["termsGiB"]
        host["held"] = (terms["base"] + terms["scan"] + terms["views"]) * GIB
        worker = ss.render_worker_bytes(gaussians)
        assert ss.default_workers(cpus, memory, worker) == plan["workers"] == 4
        # The limit's cores and memory would make room for more.
        limit = plan["memoryLimitMiB"] * 1024 * 1024
        assert ss.default_workers(int(plan["coresLimit"]), limit, worker) > plan["workers"]


def test_the_command_passes_the_reservation_not_a_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _segment_app(monkeypatch)
    argv = app.segment_argv(Path("/w/tiles"), Path("/w"), views=24, cpus=16.0, memory_mib=65536)
    assert "--workers" not in argv
    assert argv[argv.index("--cpus") + 1] == "16"
    assert argv[argv.index("--memory-gb") + 1] == "64"
    assert argv[argv.index("--summary") + 1] == "/w/run.json"
    assert "--cache" not in argv
    cached = app.segment_argv(
        Path("/w/tiles"), Path("/w"), views=24, cpus=8.0, memory_mib=32768, cache=Path("/c")
    )
    assert cached[-2:] == ["--cache", "/c"]


def test_the_command_carries_segmentation_v2s_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """The merged command line: v2's renderer, floater cut, coverage rounds and debug sheets
    beside the reservation, and no `--render-instances` PNG (v2 hands back report.json and
    compare.png instead)."""
    app = _segment_app(monkeypatch)
    argv = app.segment_argv(
        Path("/w/tiles"), Path("/w"), views=24, cpus=8.0, memory_mib=32768, coverage_rounds=3
    )
    assert "--render-instances" not in argv
    assert argv[argv.index("--renderer") + 1] == "gsplat"
    assert argv[argv.index("--max-scale-m") + 1] == str(app.MAX_SCALE_M)
    assert argv[argv.index("--coverage-rounds") + 1] == "3"
    assert argv[argv.index("--coverage-views") + 1] == str(app.COVERAGE_VIEWS)
    assert argv[argv.index("--debug-dir") + 1] == "/w/debug"
    assert ("--variants" in argv) == app.VARIANTS
    cpu = app.segment_argv(
        Path("/w/tiles"), Path("/w"), views=24, cpus=8.0, memory_mib=32768, renderer="cpu"
    )
    assert cpu[cpu.index("--renderer") + 1] == "cpu"
    assert cpu[cpu.index("--coverage-rounds") + 1] == str(app.COVERAGE_ROUNDS)


def test_segment_scan_takes_v2s_options_and_the_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main` spawns `segment_scan(name, url, views, keep_masks, renderer, coverage_rounds,
    plan, variant)` positionally: the function's parameters must stay in that order."""
    import inspect

    app = _segment_app(monkeypatch)
    names = list(inspect.signature(app.app.functions["segment_scan"].fn).parameters)
    assert names == [
        "name", "url", "views", "keep_masks", "renderer", "coverage_rounds", "plan", "variant",
    ]  # fmt: skip


def test_a_variant_runs_its_own_script_with_a_cache_and_a_known_worst_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _segment_app(monkeypatch)
    argv = app.segment_argv(
        Path("/t"), Path("/w"), views=24, cpus=6, memory_mib=16384, cache=Path("/c"),
        variant="ground-first",
    )  # fmt: skip
    assert argv[1] == "segment_ground_first.py"
    assert argv[argv.index("--out") + 1] == "/w/out"
    assert argv[argv.index("--masks") + 1] == "segment_models:Sam2LargeMasks"
    assert argv[argv.index("--boxes") + 1] == "segment_models:Sam2BoxMasks"
    assert argv[argv.index("--namer") + 1] == "segment_models:QwenNamer"
    assert argv[argv.index("--cache") + 1] == "/c"
    assert "--debug-dir" not in argv
    plain = app.segment_argv(Path("/t"), Path("/w"), views=24, cpus=6, memory_mib=16384)
    assert plain[1] == "segment_scene.py" and "--out" not in plain
    plan = app.sizing(153_566, extra_bytes=app.VARIANT_BYTES)
    assert plan["termsGiB"]["variant"] == 4.0
    # The worst a variant's call can cost: its rate for its whole timeout, under a dollar.
    assert plan["dollarsPerHour"] * app.VARIANT_TIMEOUT_S / 3600 < 1.2


def test_main_spawns_each_scan_on_its_own_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each scan's tileset.json is read first; it is spawned on `with_options` with its
    (request, limit) pairs and its plan, said in one line before and one after; one that
    cannot be read is sized as the camp."""
    app = _segment_app(monkeypatch)
    sizes = {"spool": [100_000, 53_566], "camp": [100_000] * 225 + [77_243]}

    def get(url: str, _timeout: float) -> bytes:
        for name, leaves in sizes.items():
            if url == app.SCANS[name]:
                return json.dumps(_tileset(leaves)).encode()
        raise RuntimeError(f"GET {url}: HTTP 404 Not Found")

    monkeypatch.setattr(app, "_get", get)
    monkeypatch.setitem(app.SCANS, "lost", "https://example.invalid/lost/tileset.json")
    with pytest.raises(SystemExit, match="segmentation failed for lost"):
        app.main(names="spool,lost,camp,pumpkin", out=str(tmp_path))
    spawned = app.app.functions["segment_scan"].spawned
    assert [args[0] for _, args in spawned] == ["spool", "lost", "camp", "pumpkin"]
    del spawned[1]
    for (options, args), gaussians in zip(spawned, (SPOOL, CAMP, CAMP), strict=True):
        plan = app.sizing(gaussians)
        assert args[6] == plan
        assert options["cpu"] == (plan["cores"], plan["coresLimit"])
        assert options["memory"] == (plan["memoryMiB"], plan["memoryLimitMiB"])
        assert options["gpu"] == "L4"
    printed = capsys.readouterr().out
    assert "spool: 3 tiles, 153,566 gaussians, up to 672 views -> 6 cores (limit 9)" in printed
    assert "16 GiB (limit 32) on an L4, 4 render workers" in printed
    assert "pumpkin: could not read its size" in printed and "sized as the camp" in printed
    assert "camp: peak 17.6 GiB main, ? a render worker, ? container; requested 22 GiB" in printed
    summary = json.loads((tmp_path / "camp" / "summary.json").read_text(encoding="utf-8"))
    assert summary["sizing"] == app.sizing(CAMP) and summary["usage"]["coresUsed"] == 4.5
    # A call that raised loses only its own result.
    lost = json.loads((tmp_path / "lost" / "summary.json").read_text(encoding="utf-8"))
    assert lost == {"name": "lost", "ok": False, "sizing": app.sizing(CAMP)}
    assert "out of memory" in (tmp_path / "lost" / "log.txt").read_text(encoding="utf-8")


def test_gpu_samples_and_peaks_become_the_calls_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _segment_app(monkeypatch)
    samples = "57, 10240\n[N/A], [N/A]\n100, 12288\n0, 9000\n"
    assert app.gpu_usage(samples) == {
        "gpuBusyShare": round(157 / 3 / 100, 3),
        "gpuPeakGiB": 12.0,
        "gpuSamples": 3,
    }
    assert app.gpu_usage("") == {}
    peak = tmp_path / "memory.peak"
    peak.write_text(str(20 * (1 << 30)), encoding="utf-8")
    assert app.container_peak_gib((str(tmp_path / "none"), str(peak))) == 20.0
    assert app.container_peak_gib((str(tmp_path / "none"),)) is None
    monkeypatch.setattr(app, "container_peak_gib", lambda: 20.0)
    run = {"instances": 3, "usage": {"mainPeakGiB": 14.5, "workerPeakGiB": 1.8, "coresUsed": 4.2}}
    usage = app.call_usage(run, samples)
    assert usage["mainPeakGiB"] == 14.5 and usage["containerPeakGiB"] == 20.0
    assert usage["gpuBusyShare"] == round(157 / 3 / 100, 3)
    assert usage["largestProcessGiB"] >= 0.0
    # Without segment_scene's summary (it failed) there is still the call's own.
    assert {"largestProcessGiB", "containerPeakGiB"} <= set(app.call_usage({}, ""))
    line = app.usage_line("camp", {"sizing": app.sizing(CAMP), "usage": usage, "totalS": 2700.0})
    assert line == (
        "camp: peak 14.5 GiB main, 1.8 GiB a render worker, 20 GiB container; requested 22 GiB"
        " (limit 44); 4.2 of 6 cores busy; GPU busy 52%; 2700.0 s"
    )


def test_segment_scan_runs_on_the_request_and_returns_its_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The call passes its plan's request (not the limit) to segment_scene and returns the
    plan and the run's usage beside its files -- failed or not."""
    app = _segment_app(monkeypatch)
    plan = app.sizing(CAMP)
    ran: list[list[str]] = []

    def fetch(_url: str, out: Path) -> int:
        out.mkdir(parents=True)
        (out / "tileset.json").write_text("{}", encoding="utf-8")
        return 514

    def run(argv: list[str], **_kwargs: Any) -> Any:
        ran.append(argv)
        if argv[1] == "segment_scene.py":
            work = Path(argv[argv.index("--summary") + 1]).parent
            (work / "tiles" / "instances.json").write_text("{}", encoding="utf-8")
            usage = {"mainPeakGiB": 15.0, "coresUsed": 4.0}
            summary = {"instances": 7, "views": 444, "usage": usage}
            (work / "run.json").write_text(json.dumps(summary), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    def unpublished(url: str, _timeout: float) -> bytes:
        raise RuntimeError(f"GET {url}: HTTP 404 Not Found")

    monkeypatch.setattr(app, "_fetch", fetch)
    monkeypatch.setattr(app, "_get", unpublished)
    monkeypatch.setattr(app, "WEIGHTS", types.SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(app.subprocess, "run", run)
    monkeypatch.setattr(app, "GPU_QUERY", ("no-such-nvidia-smi",))
    monkeypatch.setattr(app.tempfile, "tempdir", str(tmp_path))
    result = app.app.functions["segment_scan"].fn("camp", "https://x/tileset.json", plan=plan)
    argv = ran[0]
    assert argv[argv.index("--cpus") + 1] == "6"
    assert argv[argv.index("--memory-gb") + 1] == "22"
    assert ran[1][1] == "instances_report.py"
    assert result["ok"] and result["sizing"] == plan
    assert result["usage"]["mainPeakGiB"] == 15.0 and "largestProcessGiB" in result["usage"]
    assert result["run"] == {"instances": 7, "views": 444}

    def fail(argv: list[str], **_kwargs: Any) -> Any:
        return types.SimpleNamespace(returncode=-9, stdout="", stderr="Killed")

    monkeypatch.setattr(app.subprocess, "run", fail)
    failed = app.app.functions["segment_scan"].fn("camp", "https://x/tileset.json")
    assert not failed["ok"] and failed["sizing"] == app.DEFAULT_SIZING
    assert "largestProcessGiB" in failed["usage"] and failed["log"].endswith("Killed")


def test_the_cli_takes_v2s_flags_with_the_reservation(
    host: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`segment_scene.py` parses the merged command line (one `--workers`, v2's flags, the
    reservation) and hands all of it to `segment`."""
    seen: dict[str, Any] = {}

    def stop(*_args: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(ss, "segment", stop)
    monkeypatch.setattr(ss, "load_source", lambda *_a: (None, None, 0))
    monkeypatch.setattr(ss, "load_embedder", lambda _name: None)
    monkeypatch.setattr(ss, "load_masks", lambda _name: None)
    monkeypatch.setattr(ss, "make_renderer", lambda name: name)
    argv = ["segment_scene.py", str(tmp_path / "s.ply"), str(tmp_path), "--masks", "m:M"]
    flags = ["--renderer", "cpu", "--max-scale-m", "0.5", "--coverage-rounds", "2"]
    reservation = ["--cpus", "8", "--memory-gb", "32"]
    monkeypatch.setattr(sys, "argv", [*argv, *flags, *reservation])
    with pytest.raises(SystemExit):
        ss.main()
    assert seen["renderer"] == "cpu" and seen["max_scale_m"] == 0.5
    assert seen["coverage_rounds"] == 2
    assert seen["cpus"] == 8 and seen["memory_bytes"] == 32 * GIB and seen["workers"] is None

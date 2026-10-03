"""How many views render at once, from what a Modal call reserved -- not a fixed 24.

`segment_scene.default_workers` is told the container's reservation (`--cpus`,
`--memory-gb`), because inside a Modal container the host's cores and memory are what
show, not what was paid for. `infra/modal/segment.py` chooses the reservation from the
scan's tile count and passes it down. Both halves are checked here without Modal: the
first directly, the second by importing the app file against a stand-in `modal` module
that records what the decorators were given.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import segment_scene as ss

GIB = float(1 << 30)
SEGMENT_APP = Path(__file__).resolve().parents[3] / "infra" / "modal" / "segment.py"


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A 64-core host with no cgroup memory limit, this process holding 6 GB."""
    state: dict[str, Any] = {"cores": 64, "room": None, "held": 6e9}
    monkeypatch.setattr(ss.os, "sched_getaffinity", lambda _pid: set(range(state["cores"])))
    monkeypatch.setattr(ss, "_memory_room", lambda: state["room"])
    monkeypatch.setattr(ss, "_resident_bytes", lambda: state["held"])
    return state


def test_without_a_reservation_the_host_is_what_shows(host: dict[str, Any]) -> None:
    """The case the reservation exists for: 64 renders on a container that paid for 8."""
    assert ss.default_workers() == 64


def test_the_reserved_cores_bound_the_renders(host: dict[str, Any]) -> None:
    # 32 GiB less the 6 GB held is room for 11 renders at RENDER_WORKER_BYTES; 8 cores.
    assert ss.default_workers(8, 32 * GIB) == 8


def test_the_memory_left_after_the_scan_bounds_them_too(host: dict[str, Any]) -> None:
    # 16 GiB less 6 GB held: 11.2 GB, four renders of 2.5 GB.
    assert ss.default_workers(8, 16 * GIB) == 4
    # A scan that fills the reservation still renders, one view at a time.
    host["held"] = 40e9
    assert ss.default_workers(8, 32 * GIB) == 1


def test_a_tighter_cgroup_wins_over_the_reservation(host: dict[str, Any]) -> None:
    host["room"] = 5.5e9
    assert ss.default_workers(8, 32 * GIB) == 2


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


# --- infra/modal/segment.py, against a stand-in `modal` ----------------------------------


class _Chain:
    """`modal.Image`'s builder: every call returns the builder."""

    def __getattr__(self, _name: str) -> Any:
        return lambda *_a, **_k: self


class _Function:
    def __init__(self, fn: Any, options: dict[str, Any]) -> None:
        self.fn, self.options = fn, options

    def with_options(self, **options: Any) -> _Function:
        return _Function(self.fn, {**self.options, **options})


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


def test_the_function_reserves_8_cores_and_32_gib(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _segment_app(monkeypatch)
    options = app.app.functions["segment_scan"].options
    assert options["gpu"] == "L4"
    assert options["cpu"] == 8.0 and options["memory"] == 32 * 1024


def test_a_scan_past_the_threshold_gets_the_large_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _segment_app(monkeypatch)
    assert app.reservation(514) == (8.0, 32 * 1024)  # the camp
    assert app.reservation(app.LARGE_TILES) == (8.0, 32 * 1024)
    assert app.reservation(app.LARGE_TILES + 1) == (16.0, 64 * 1024)
    tileset = {
        "root": {
            "content": {"uri": "splat.glb"},
            "children": [{"content": {"uri": "a.glb"}}, {"children": [{"content": {"uri": "b"}}]}],
        }
    }
    assert app.tiles_in(tileset) == 3


def test_the_command_passes_the_reservation_not_a_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _segment_app(monkeypatch)
    argv = app.segment_argv(
        Path("/w/tiles"), Path("/w/i.png"), views=24, cpus=16.0, memory_mib=65536
    )
    assert "--workers" not in argv
    assert argv[argv.index("--cpus") + 1] == "16"
    assert argv[argv.index("--memory-gb") + 1] == "64"
    assert "--cache" not in argv
    cached = app.segment_argv(
        Path("/w/tiles"), Path("/w/i.png"), views=24, cpus=8.0, memory_mib=32768, cache=Path("/c")
    )
    assert cached[-2:] == ["--cache", "/c"]

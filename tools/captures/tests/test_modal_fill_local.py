"""infra/modal/fill.py's local side runs where fill.yml runs it: `uv run --with modal`
from the repository root, an environment with modal and the standard library and nothing
of tools/captures' dependencies. Its fill-gate pre-check must work there -- a numpy
import would crash every fill.yml run, drop-only and --parity-test included.

Run in a fresh interpreter with numpy (and anything else not stdlib) made unimportable and
modal replaced by a stub that accepts whatever fill.py declares at import.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FILL = REPO / "infra" / "modal" / "fill.py"
TARGETS = REPO / "data" / "tiles" / "synthetic-yard-epochs" / "fill-targets.json"

PROBE = r"""
import importlib.abc, importlib.util, json, sys, types

class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"numpy", "scipy", "cv2", "PIL", "cell_grid", "rescan_gate"}:
            raise ModuleNotFoundError(f"No module named {name!r} (blocked: not in fill.yml's env)")
        return None

sys.meta_path.insert(0, Blocked())

class Anything:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) and not k else Anything()
    def __getattr__(self, name): return Anything()
    def __bool__(self): return True

modal = types.ModuleType("modal")
modal.__getattr__ = lambda name: Anything()
modal.is_local = lambda: True
sys.modules["modal"] = modal

spec = importlib.util.spec_from_file_location("fillapp", sys.argv[1])
fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill)
out = {}
for name, still, jobs in json.loads(sys.argv[2]):
    try:
        out[name] = ["ok", sorted(fill._fill_targets(still, jobs))]
    except SystemExit as error:
        out[name] = ["refused", str(error)]
print(json.dumps(out))
"""


def _probe(cases: list[tuple[str, str, str]]) -> dict[str, list[object]]:
    done = subprocess.run(
        [sys.executable, "-I", "-c", PROBE, str(FILL), json.dumps(cases)],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    result: dict[str, list[object]] = json.loads(done.stdout.strip().splitlines()[-1])
    return result


def test_the_local_pre_check_needs_nothing_but_the_standard_library(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**json.loads(TARGETS.read_text()), "policy": "weak-v1"}))
    capped = tmp_path / "capped.json"
    capped.write_text(json.dumps({**json.loads(TARGETS.read_text()), "capped": True}))
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    result = _probe(
        [
            ("drop only", "", "drop:yard,drop:spool"),
            ("gated", f"camp={TARGETS}", "fill:camp,drop:yard"),
            ("capped", f"camp={capped}", "fill:camp"),
            ("no targets", "", "fill:camp"),
            ("wrong policy", f"camp={bad}", "fill:camp"),
            ("not json", f"camp={broken}", "fill:camp"),
            ("unknown scan", f"nope={TARGETS}", "fill:camp"),
        ]
    )
    assert result["drop only"] == ["ok", []]
    assert result["gated"] == ["ok", ["camp"]]
    assert result["capped"] == ["ok", ["camp"]]
    for name, words in [
        ("no targets", "a fill runs only where a re-scan left cells weak"),
        ("wrong policy", "rescan-gate-v1"),
        ("not json", "not JSON"),
        ("unknown scan", "<scan>=<file>"),
    ]:
        assert result[name][0] == "refused" and words in str(result[name][1]), (name, result)


def test_a_push_runs_no_fill_job_and_says_so() -> None:
    """fill.yml's jobs have no default: a push to a `wm-*` branch (no inputs) runs nothing
    -- its old default, fill:camp, can no longer pass the gate -- and every step but the
    notice that says so is skipped."""
    text = (REPO / ".github" / "workflows" / "fill.yml").read_text(encoding="utf-8")
    assert "JOBS: ${{ inputs.jobs || '' }}" in text
    defaults = [line for line in text.splitlines() if line.strip().startswith("default:")]
    assert not any("fill:" in line for line in defaults), defaults
    steps = text.split("    steps:\n", 1)[1]
    names = [line for line in steps.splitlines() if line.startswith("      - ")]
    guards = [line for line in steps.splitlines() if line.startswith("        if:")]
    assert len(guards) == len(names)
    assert guards[0].strip() == "if: env.JOBS == ''" and "Nothing to run" in names[0]
    assert all("env.JOBS != ''" in guard for guard in guards[1:])


# --- the remote job's fetch: everything the gate reads comes down with the tiles ---------

BASE = "https://scans.example.com/runs/yard/package/splat"
GRIDS = REPO / "data" / "tiles" / "synthetic-yard-epochs"


class _Anything:
    """Whatever fill.py declares at import, accepted (a decorator returns what it wraps)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __call__(self, *args: object, **kwargs: object) -> object:
        return args[0] if len(args) == 1 and callable(args[0]) and not kwargs else _Anything()

    def __getattr__(self, name: str) -> object:
        return _Anything()


def _fill_app(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    modal = types.ModuleType("modal")
    modal.__getattr__ = lambda name: _Anything()  # type: ignore[method-assign]
    modal.is_local = lambda: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal", modal)
    spec = importlib.util.spec_from_file_location("fill_fetch_under_test", FILL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _served(grid_uri: str = "confidence.bin") -> dict[str, bytes]:
    descriptor = json.loads((GRIDS / "grids.json").read_text())["epochs"]["e2"]["descriptor"]
    document = {
        "asset": {"version": "1.1"},
        "geometricError": 10,
        "root": {
            "boundingVolume": {"box": [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5]},
            "geometricError": 1,
            "content": {"uri": "0/0.glb"},
            "children": [{"geometricError": 0, "content": {"uri": "0/1.glb"}}],
            "extras": {
                "viewCones": {"uri": "viewcones.bin"},
                "confidence": {**descriptor, "uri": grid_uri},
            },
        },
    }
    return {
        f"{BASE}/tileset.json": json.dumps(document).encode(),
        f"{BASE}/0/1.glb": b"leaf",
        f"{BASE}/viewcones.bin": b"cones",
        f"{BASE}/confidence.bin": (GRIDS / "e2" / "confidence.bin").read_bytes(),
    }


def test_a_remote_fill_fetches_the_confidence_grid_its_gate_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex R3 #60: `_fetch` brought the tiles, view cones and instances, never the
    confidence grid, so `teacher_fill fill`'s gate (`RescanGate.check_tileset`) could not
    read it and refused every remote fill. Fetched with a fake fetcher, the scan the job
    sees passes the gate for the targets its epoch has."""
    import rescan_gate

    fill = _fill_app(monkeypatch)
    served = _served()
    asked: list[str] = []

    def get(url: str, timeout: float) -> bytes:
        asked.append(url)
        return served[url]

    monkeypatch.setattr(fill, "_get", get)
    tileset = fill._fetch(f"{BASE}/tileset.json", tmp_path / "scan")

    assert f"{BASE}/confidence.bin" in asked
    gate = rescan_gate.RescanGate.from_json(TARGETS)
    descriptor = json.loads(served[f"{BASE}/tileset.json"])["root"]["extras"]["confidence"]
    assert gate.check_tileset(tileset) == descriptor["sha256"]


@pytest.mark.parametrize("uri", ["../confidence.bin", "/etc/confidence.bin", "a/../../x.bin"])
def test_a_grid_uri_that_leaves_the_scan_is_not_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uri: str
) -> None:
    fill = _fill_app(monkeypatch)
    served = _served(uri)
    monkeypatch.setattr(fill, "_get", lambda url, timeout: served.get(url, b""))
    with pytest.raises(RuntimeError, match="beside the tileset"):
        fill._fetch(f"{BASE}/tileset.json", tmp_path / "scan")
    assert not (tmp_path / "confidence.bin").exists()

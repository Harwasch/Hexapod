"""infra/modal/fill.py's round-2 side without Modal: the classes and methods `anchor_fill`
calls exist, the options a run may set are checked and become `anchor_fill` flags, and the
cost guard's estimate and worst case grow with what a run asks for."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

FILL = Path(__file__).resolve().parents[3] / "infra" / "modal" / "fill.py"


class _Image:
    """Every builder method returns the image (`modal.Image` is a chain)."""

    def __getattr__(self, name: str):
        return lambda *a, **k: self


def _decorator(*a, **k):
    return lambda thing: thing


@pytest.fixture(scope="module")
def app():
    stub = types.ModuleType("modal")
    stub.App = lambda name: SimpleNamespace(
        cls=_decorator, function=_decorator, local_entrypoint=_decorator
    )
    stub.Volume = SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(commit=lambda: None))
    stub.Secret = SimpleNamespace(from_name=lambda *a, **k: None)
    stub.Image = SimpleNamespace(
        from_registry=lambda *a, **k: _Image(), debian_slim=lambda *a, **k: _Image()
    )
    stub.is_local = lambda: True
    stub.enter = _decorator
    stub.method = _decorator
    saved = sys.modules.get("modal")
    sys.modules["modal"] = stub
    try:
        spec = importlib.util.spec_from_file_location("fill_app_under_test", FILL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        if saved is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = saved


def test_the_classes_anchor_fill_calls_are_the_apps(app) -> None:
    import ast

    import anchor_fill as af

    tree = ast.parse(FILL.read_text(encoding="utf-8"))
    methods = {
        node.name: {f.name for f in node.body if isinstance(f, ast.FunctionDef)}
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert "edit" in methods[af.RemoteEditor.cls]
    assert "fill_set" in methods["FillVace14"]
    assert "inpaint" in methods["InpaintQwen"]
    assert set(app.ANCHOR_CLASSES) == {"EditQwen", "FillVace14", "InpaintQwen"}
    assert set(app.ANCHOR_SCANS) == {"spool", "pumpkin"}  # the camp has no poses
    assert (
        app.ANCHOR_SCANS["spool"]["leave"] == "high"
        and app.ANCHOR_SCANS["pumpkin"]["leave"] == "low"
    )


def test_options_are_checked_and_become_flags(app) -> None:
    import anchor_fill as af

    opts = app.parse_anchor_options(
        "arms=refs+norefs,seeds=4,update_strengths=0.5+0.3,fallback=true"
    )
    argv = app.anchor_argv(opts)
    assert argv == [
        "--arms",
        "refs,norefs",
        "--seeds",
        "4",
        "--update-strengths",
        "0.5,0.3",
        "--fallback",
    ]
    parsed = af.parser().parse_args(["run", "t", "o", "--caption", "c", "--poses", "p", *argv])
    o = af.options_from(parsed)
    assert o.arms == ("refs", "norefs") and o.seeds == 4 and o.update_strengths == (0.5, 0.3)
    assert o.fallback
    for bad in ("arms=refs;rm", "seeds=99", "nope=1", "vae_area=1"):
        with pytest.raises(SystemExit):
            app.parse_anchor_options(bad)


def test_the_estimate_and_its_worst_case_grow_with_the_run(app) -> None:
    base = {"arms": "refs+norefs+vace"}
    one = app.estimate_anchor_cost([("leaveout", "spool")], base)
    two = app.estimate_anchor_cost([("leaveout", "spool"), ("anchor", "spool")], base)
    assert 0 < one["totalUsd"] < one["worstUsd"]
    assert two["totalUsd"] > one["totalUsd"] and two["worstUsd"] > one["worstUsd"]
    lean = app.estimate_anchor_cost([("leaveout", "spool")], {"arms": "refs+norefs"})
    assert lean["worstUsd"] < one["worstUsd"]
    calls = app.anchor_counts(base)
    assert calls["anchor"] == calls["anchorNoRefs"] == 6 * 4 and calls["set"] == 2
    worst = app.anchor_counts(base, worst=True)
    assert worst["anchor"] == 8 * 4 and worst["prop"] == (24 + 17) * 2  # + the hemisphere views


def test_solidity_clones_and_the_shape_cost_their_updates_and_minutes(app) -> None:
    import anchor_fill as af

    plain = {"arms": "refs+norefs"}
    both = {**plain, "solidity": "alpha+full", "shape": "true"}
    opts = app.parse_anchor_options("arms=refs+norefs,solidity=alpha+full,shape=true")
    argv = app.anchor_argv(opts)
    parsed = af.parser().parse_args(["run", "t", "o", "--caption", "c", "--poses", "p", *argv])
    o = af.options_from(parsed)
    assert o.solidity == ("alpha", "full") and o.shape is True
    with pytest.raises(SystemExit):
        app.parse_anchor_options("solidity=everything")
    # Two presets clone both sequential arms: three times the update calls.
    assert app.anchor_counts(both)["update"] == 3 * app.anchor_counts(plain)["update"]
    one, more = (app.estimate_anchor_cost([("anchor", "spool")], x) for x in (plain, both))
    assert more["usd"]["anchor:spool L4"] > one["usd"]["anchor:spool L4"]
    assert more["totalUsd"] > one["totalUsd"] and more["worstUsd"] > one["worstUsd"]


def test_the_actual_cost_counts_every_call_and_start(app) -> None:
    results = [
        {
            "kind": "leaveout",
            "scan": "spool",
            "timings": {"totalS": 3600.0},
            "result": {
                "calls": [
                    {"key": "refs/a0-s17", "seconds": 10.0, "loadSeconds": 200.0},
                    {"key": "refs/a0-s1017", "seconds": 10.0, "loadSeconds": 200.0},
                    {"key": "set-s41", "seconds": 100.0, "loadSeconds": 300.0},
                    {"key": "norefs/p0-s29", "error": "boom", "seconds": 4.0, "loadSeconds": 180.0},
                ]
            },
        }
    ]
    cost = app.actual_anchor_cost(results)
    assert cost["gpuSeconds"] == {"EditQwen": 24.0, "FillVace14": 100.0}
    assert cost["starts"]["EditQwen"] == [180.0, 200.0]
    assert cost["usd"]["leaveout:spool L4"] == pytest.approx(0.80, abs=1e-3)
    assert cost["totalUsd"] > 0.8

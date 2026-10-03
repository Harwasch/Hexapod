"""`infra/modal/dream.py`'s run against a fake `modal`: what it cancels, and when.

The fake stands in for exactly what dream.py touches -- `App` decorators, the image builder,
`Function.from_name(...).remote()`, `Cls.from_name(...)().clip` -- and every `.spawn` returns
a call whose `get` answers from `answer` below. A call that is "running" blocks in `get`,
and a SIGTERM sent from inside that wait is a GitHub cancel landing mid-run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path

import pytest

DREAM = Path(__file__).resolve().parents[3] / "infra" / "modal" / "dream.py"
TERMINATE = [{"terminate_containers": True}]
RUNNING = object()  # an answer: the call has not finished; `get` waits
ACCESS = {"Wan-AI/Wan2.2-TI2V-5B-Diffusers": "ok"}  # Cosmos unreadable, so skipped

Answer = Callable[[str, tuple], object]


class FakeCall:
    def __init__(self, name: str, args: tuple, answer: Answer) -> None:
        self.name, self.args, self.answer = name, args, answer
        self.cancels: list[dict] = []

    def get(self, timeout: float | None = None) -> object:
        result = self.answer(self.name, self.args)
        if result is RUNNING:
            # A GitHub cancel, while the run waits on this call.
            threading.Timer(0.1, os.kill, (os.getpid(), signal.SIGTERM)).start()
            threading.Event().wait(10)
            raise AssertionError("waited on and never interrupted")
        return result

    def cancel(self, terminate_containers: bool = False) -> None:
        self.cancels.append({"terminate_containers": terminate_containers})


class FakeFunction:
    def __init__(self, name: str, answer: Answer, spawned: list[FakeCall]) -> None:
        self.name, self.answer, self.spawned = name, answer, spawned

    def spawn(self, *args: object) -> FakeCall:
        call = FakeCall(self.name, args, self.answer)
        self.spawned.append(call)
        return call

    def remote(self, *args: object) -> object:
        raise AssertionError(f"{self.name}.remote: dream.py spawns its calls, tracked")


def _fake_modal(answer: Answer, spawned: list[FakeCall]) -> types.ModuleType:
    modal = types.ModuleType("modal")

    class Image:  # every builder method hands the image back
        def __getattr__(self, name: str) -> Callable[..., Image]:
            return lambda *args, **kwargs: self

    class App:
        def __init__(self, name: str) -> None:
            self.name = name

        def function(self, **options: object) -> Callable:
            return lambda fn: FakeFunction(fn.__name__, answer, spawned)

        def local_entrypoint(self) -> Callable:
            return lambda fn: fn

    def cls(app: str, name: str) -> Callable[[], types.SimpleNamespace]:
        return lambda: types.SimpleNamespace(clip=FakeFunction(f"{name}.clip", answer, spawned))

    access = types.SimpleNamespace(remote=lambda: dict(ACCESS))
    modal.is_local = lambda: True
    modal.Image = Image()
    modal.App = App
    modal.Function = types.SimpleNamespace(from_name=lambda app, name: access)
    modal.Cls = types.SimpleNamespace(from_name=cls)
    return modal


def ok(name: str, args: tuple) -> object:
    """What each function answers on a good run."""
    if name == "starts":
        (scan,) = args
        view = f"{scan}-eye0-000"
        return {"scan": scan, "views": [{"name": view}], "files": {f"{view}.png": b"png"}}
    if name == "Wan.clip":
        return {"mp4": b"mp4", "model": "fake-wan", "fps": 24.0}
    if name == "sheet":
        return b"sheet"
    if name == "materials":
        scan, instance, _options = args
        materials = {"materials": [{"instance": instance, "stiffness": 1.0}]}
        files = {"report.json": b"[]", "materials.json": json.dumps(materials).encode()}
        return {"scan": scan, "instance": instance, "ok": True, "files": files, "log": ""}
    raise AssertionError(f"unexpected call {name}{args}")


def _run(
    monkeypatch: pytest.MonkeyPatch,
    out: Path,
    answer: Answer,
    spawned: list[FakeCall],
    instances: str = "225",
) -> None:
    """dream.py's `main` on the camp, Wan and Cosmos (skipped), two prompts; every call it
    spawns lands in `spawned`."""
    monkeypatch.setitem(sys.modules, "modal", _fake_modal(answer, spawned))
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("dream_under_test", DREAM)
    assert spec and spec.loader
    dream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dream)
    dream.main(scans="camp", prompts="gentle,gusty", instances=instances, out=str(out))


def _by_label(out: Path) -> dict[str, dict]:
    summary = json.loads((out / "summary.json").read_text())
    return {row["call"]: row for row in summary["calls"]}


GENTLE, GUSTY = "clip camp-eye0-000-gentle-wan-s1", "clip camp-eye0-000-gusty-wan-s1"


def test_a_good_run_cancels_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spawned: list[FakeCall] = []
    _run(monkeypatch, tmp_path, ok, spawned)
    assert [c.cancels for c in spawned] == [[]] * len(spawned)
    states = {label: row["state"] for label, row in _by_label(tmp_path).items()}
    assert states == {
        "materials 225": "succeeded",
        "starts camp": "succeeded",
        GENTLE: "succeeded",
        GUSTY: "succeeded",
        "sheet camp-eye0-000-gentle-wan-s1": "succeeded",
        "sheet camp-eye0-000-gusty-wan-s1": "succeeded",
    }
    assert (tmp_path / "clips" / "camp-eye0-000-gusty-wan-s1.mp4").read_bytes() == b"mp4"
    assert json.loads((tmp_path / "materials.json").read_text())["materials"][0]["instance"] == 225


def test_sigterm_mid_run_cancels_every_call_still_out_there(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def answer(name: str, args: tuple) -> object:
        if name == "Wan.clip" and "gusty" in args[0]["prompt"].lower():
            return RUNNING
        return ok(name, args)

    spawned: list[FakeCall] = []
    with pytest.raises(RuntimeError, match="stopped by SIGTERM; 3 of 3") as raised:
        _run(monkeypatch, tmp_path, answer, spawned)
    assert type(raised.value).__name__ == "RunStopped"  # `modal run` exits 1 on it
    cancelled = sorted(c.name for c in spawned if c.cancels == TERMINATE)
    assert cancelled == ["Wan.clip", "materials", "sheet"]  # gusty, the camp's fit, a sheet
    assert all(c.cancels in ([], TERMINATE) for c in spawned)
    rows = _by_label(tmp_path)
    assert rows[GUSTY]["state"] == rows["materials 225"]["state"] == "cancelled"
    assert rows[GENTLE]["state"] == "succeeded"


def test_a_clips_own_failure_is_reported_and_fails_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def answer(name: str, args: tuple) -> object:
        if name == "Wan.clip" and "gusty" in args[0]["prompt"].lower():
            raise RuntimeError("CUDA out of memory")
        return ok(name, args)

    message = re.escape(f"1 problem(s), everything else collected: {GUSTY}: RuntimeError")
    spawned: list[FakeCall] = []
    with pytest.raises(RuntimeError, match=message):
        _run(monkeypatch, tmp_path, answer, spawned)
    assert all(c.cancels == [] for c in spawned)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["clips"][1]["error"] == "RuntimeError('CUDA out of memory')"
    assert summary["materials"][0]["ok"] is True  # the rest still came back
    assert _by_label(tmp_path)[GUSTY]["state"] == "failed"


def test_a_dream_phase_that_raises_cancels_its_clips_not_the_materials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def answer(name: str, args: tuple) -> object:
        if name == "Wan.clip" and "gentle" in args[0]["prompt"].lower():
            return {"model": "fake-wan"}  # no "mp4": the dream phase raises a KeyError
        return ok(name, args)

    with pytest.raises(RuntimeError, match="the dream phase raised"):
        _run(monkeypatch, tmp_path, answer, [])
    rows = _by_label(tmp_path)
    assert rows[GUSTY]["state"] == "cancelled"
    assert rows["materials 225"]["state"] == "succeeded"
    assert "KeyError: 'mp4'" in json.loads((tmp_path / "summary.json").read_text())["dreamError"]


def test_an_exception_out_of_the_run_cancels_and_is_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def answer(name: str, args: tuple) -> object:
        result = ok(name, args)
        if name == "materials" and args[1] == 225:
            del result["instance"]  # the materials loop does not catch this one
        return result

    with pytest.raises(KeyError, match="instance"):
        _run(monkeypatch, tmp_path, answer, [], instances="225,244")
    rows = _by_label(tmp_path)
    assert rows["materials 244"]["state"] == "cancelled"
    assert rows["materials 225"]["state"] == "succeeded"


def test_dream_spawns_every_call_through_spawned_calls() -> None:
    """A bare `.spawn(` (or `.map(` / `.remote(` of a Modal function) would be a call
    nothing cancels."""
    source = DREAM.read_text(encoding="utf-8")
    assert re.findall(r"(?<!calls)\.spawn\(", source) == []
    assert re.findall(r"\b(?:starts|sheet|materials|clip)\.(?:map|remote)\(", source) == []

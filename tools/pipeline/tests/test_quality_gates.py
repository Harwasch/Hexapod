"""`QualityGateError`, `parse_quality_gate` and `check_gate`: a gate fails a run by name.

Defined in M3a with the signature M3b's pose gates use (DECISIONS D6, shared contracts),
so M3b extends these tests rather than redefining the error.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

import colmap4
import tree_frames
from artifacts import ArtifactDecl
from conftest import FIXTURE_PLY, make_recipe
from contracts import StageContext, StageOutcome
from errors import (
    BAD_INPUT_ERRORS,
    PipelineError,
    QualityGateError,
    check_gate,
    evaluate_gate,
    parse_quality_gate,
)
from executor import execute
from registry import stage_impl
from runners import LocalRunner, RunnerSet
from workdir import Workdir


def test_the_message_round_trips() -> None:
    error = QualityGateError(
        "registered_fraction", 0.62, 0.8, unit="fraction", reason="too few frames\nregistered"
    )

    assert str(error) == (
        "quality gate registered_fraction: measured 0.62 fraction, limit 0.8 fraction "
        "-- too few frames registered"
    )
    assert parse_quality_gate(str(error)) == {
        "gate": "registered_fraction",
        "measured": 0.62,
        "limit": 0.8,
        "unit": "fraction",
        "reason": "too few frames registered",
    }
    assert error.to_dict() == parse_quality_gate(str(error))
    wrapped = (
        "recipe 'photo-reconstruct', stage 'normalize' (impl 'ffmpeg_frames') failed on "
        f"'modal': QualityGateError: {error}\nTraceback (most recent call last): ..."
    )
    assert parse_quality_gate(wrapped) == error.to_dict()
    unitless = QualityGateError("clipped_frames_fraction", 1e-05, 0.2, reason="r")
    assert parse_quality_gate(str(unitless)) == {
        "gate": "clipped_frames_fraction",
        "measured": 1e-05,
        "limit": 0.2,
        "unit": "",
        "reason": "r",
    }
    assert parse_quality_gate("StageFailedError: boom") is None


def test_quality_gate_error_is_bad_input() -> None:
    assert issubclass(QualityGateError, PipelineError)
    assert "QualityGateError" in BAD_INPUT_ERRORS


def test_gates_are_off_unless_given() -> None:
    assert check_gate("g", 10.0, None, fail_when="above", reason="r") == {
        "name": "g",
        "enabled": False,
        "limit": None,
        "measured": 10.0,
        "passed": None,
    }


def test_an_enabled_gate_passes_fails_or_cannot_say() -> None:
    assert check_gate("g", 0.1, 0.2, fail_when="above", reason="r")["passed"] is True
    assert check_gate("g", 0.9, 0.8, fail_when="below", reason="r")["passed"] is True
    # At the limit is not past it.
    assert check_gate("g", 0.2, 0.2, fail_when="above", reason="r")["passed"] is True
    with pytest.raises(QualityGateError, match=r"quality gate g: measured 0\.3 px, limit 0\.2 px"):
        check_gate("g", 0.3, 0.2, fail_when="above", unit="px", reason="r")
    with pytest.raises(QualityGateError, match=r"measured 0\.5, limit 0\.8 -- why"):
        check_gate("g", 0.5, 0.8, fail_when="below", reason="why")
    # Nothing measured is not a failure: the gate cannot be applied, and says so.
    unknown = check_gate("g", float("nan"), 0.2, fail_when="above", reason="r")
    assert (unknown["enabled"], unknown["measured"], unknown["passed"]) == (True, None, None)


def test_gate_names_and_units_are_one_token() -> None:
    with pytest.raises(ValueError, match="one word"):
        QualityGateError("two words", 1.0, 2.0, reason="r")
    with pytest.raises(ValueError, match="one token"):
        QualityGateError("g", 1.0, 2.0, unit="per cent, roughly", reason="r")


def test_a_gate_error_survives_crossing_a_process() -> None:
    error = QualityGateError("g", 0.5, 0.2, unit="px", reason="why")

    copy = pickle.loads(pickle.dumps(error))

    assert isinstance(copy, QualityGateError)
    assert (str(copy), copy.to_dict()) == (str(error), error.to_dict())


# --- M3b: the gates a stage applies, and how a failure leaves the runner -------------------


def test_evaluate_gate_records_a_failure_without_raising() -> None:
    record, error = evaluate_gate("g", 0.4, 0.8, fail_when="below", reason="why")
    assert record == {"name": "g", "enabled": True, "limit": 0.8, "measured": 0.4, "passed": False}
    assert isinstance(error, QualityGateError) and error.gate == "g"
    passed, none = evaluate_gate("g", 0.9, 0.8, fail_when="below", reason="why")
    assert passed["passed"] is True and none is None
    off, nothing = evaluate_gate("g", 0.1, None, fail_when="below", reason="why")
    assert off["enabled"] is False and nothing is None


GATED = ArtifactDecl("gated.json", content_type="application/json")


@stage_impl("t_quality_gate_raises", produces=(GATED,), summary="fails its one gate")
def t_quality_gate_raises(ctx: StageContext) -> StageOutcome:
    check_gate(
        "t_fraction",
        0.25,
        _float(ctx.param("floor")),
        fail_when="below",
        reason="the test asked for more",
    )
    ctx.output(GATED.name).write_text("{}")
    return StageOutcome(metrics={}, summary="passed")


def _float(value: object) -> float | None:
    return None if value is None else float(str(value))


def test_a_gate_error_passes_the_runner_unwrapped(tmp_path: Path) -> None:
    recipe = make_recipe(
        [{"id": "g", "impl": "t_quality_gate_raises", "params": {"floor": 0.5}}], inputs=[]
    )
    with pytest.raises(QualityGateError) as raised:
        execute(recipe, Workdir.create(tmp_path / "w"), RunnerSet(cpu=LocalRunner()))
    assert raised.value.to_dict() == {
        "gate": "t_fraction",
        "measured": 0.25,
        "limit": 0.5,
        "unit": "",
        "reason": "the test asked for more",
    }
    # Off by default: the same stage with no limit passes.
    execute(
        make_recipe([{"id": "g", "impl": "t_quality_gate_raises"}], inputs=[]),
        Workdir.create(tmp_path / "off"),
        RunnerSet(cpu=LocalRunner()),
    )


@pytest.mark.skipif(
    colmap4.version(colmap4.python()) is None, reason="needs pycolmap==4.2.0 for the pose stage"
)
def test_the_registered_fraction_gate_fails_the_pose_stage(tmp_path: Path) -> None:
    """A limit no capture can meet (1.01 of its frames) fails the pose stage by name,
    after `poses.json` recorded what was measured; with no limit the gate is off."""
    workdir = Workdir.create(tmp_path / "pose")
    tree_frames.render_orbit(FIXTURE_PLY, workdir.input_path("frames"), count=24)
    params = {"matcher": "exhaustive", "colmap": "4.2", "first_octave": 0}
    recipe = make_recipe(
        [
            {
                "id": "pose",
                "impl": "colmap",
                "params": {**params, "fail_below_registered_fraction": 1.01},
            }
        ],
        inputs=["frames"],
    )
    with pytest.raises(QualityGateError) as raised:
        execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    assert raised.value.gate == "registered_fraction"
    assert raised.value.limit == 1.01 and raised.value.measured <= 1.0
    poses = json.loads((workdir.artifact_path("pose", "poses") / "poses.json").read_text())
    gates = {gate["name"]: gate for gate in poses["gates"]}
    assert gates["registered_fraction"]["passed"] is False
    assert gates["mean_reprojection_error_px"]["enabled"] is False

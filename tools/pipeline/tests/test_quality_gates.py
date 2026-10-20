"""`QualityGateError`, `parse_quality_gate` and `check_gate`: a gate fails a run by name.

Defined in M3a with the signature M3b's pose gates use (DECISIONS D6, shared contracts),
so M3b extends these tests rather than redefining the error.
"""

from __future__ import annotations

import pytest

from errors import BAD_INPUT_ERRORS, PipelineError, QualityGateError, check_gate, parse_quality_gate


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

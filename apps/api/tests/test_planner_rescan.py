"""A scout capture can be a re-scan of an epoch's weak areas, flown nadir, oblique or round.

The planner sizes a re-scan zone at its own capture numbers, by its pattern: nadir as a
fog scout, oblique as two crossing sets of the same lines, orbit as two rings at the
stand-off where the pitched camera meets the ground. The patterns' rules are UX rules of
thumb (ADR 0016 § API), held here to their arithmetic, not to a measurement.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.schemas.agent import PlannerZone
from app.schemas.scout import ScoutCapture
from app.services import coverage
from app.services.machine_classes import MACHINE_CLASSES
from app.services.planner import (
    ORBIT_MIN_STANDOFF_M,
    ORBIT_RINGS,
    RulesPlanner,
    _orbit_standoff_m,
)
from tests.test_agent import PADDOCK, land_request

GOAL = "Survey the weak areas of the yard scan again from the air"
REF = {
    "siteId": str(uuid.uuid4()),
    "epochId": str(uuid.uuid4()),
    "policy": "weak-v1",
    "cells": 640,
    "reasons": ["views", "spread"],
}


def rescan(pattern: str, **fields: Any) -> ScoutCapture:
    values: dict[str, Any] = {
        "purpose": "re-scan-weak",
        "pattern": pattern,
        "gimbalPitchDeg": {"nadir": -90, "oblique": -45, "orbit": -30}[pattern],
        "rescanOf": REF,
        "altitudeM": 30,
        "forwardOverlap": 0.8,
        "sideOverlap": 0.7,
        "speedMs": 5,
        **fields,
    }
    return ScoutCapture.model_validate(values)


def sized(capture: ScoutCapture | None) -> Any:
    zone = PlannerZone(id="W-01", name="Weak area 1", acres=1, geometry=PADDOCK, capture=capture)
    draft = RulesPlanner().draft(land_request(GOAL, [zone], preferred_zone_ids=["W-01"]))
    [found] = draft.coverage
    return found


def test_a_nadir_re_scan_is_flown_like_a_fog_scout() -> None:
    fog = ScoutCapture(altitude_m=30, forward_overlap=0.8, side_overlap=0.7, speed_ms=5)
    nadir, scout = sized(rescan("nadir")), sized(fog)
    assert (nadir.length_m, nadir.passes, nadir.hours) == (
        scout.length_m,
        scout.passes,
        scout.hours,
    )
    assert nadir.spacing_m == pytest.approx(MACHINE_CLASSES.drone.footprint_width_m(30) * 0.3)


def test_an_oblique_re_scan_flies_two_crossing_sets_of_lines() -> None:
    nadir, oblique = sized(rescan("nadir")), sized(rescan("oblique"))
    assert oblique.length_m == pytest.approx(2 * nadir.length_m)
    assert (oblique.passes, oblique.spans) == (2 * nadir.passes, 2 * nadir.spans)
    assert oblique.transit_m == pytest.approx(2 * nadir.transit_m)
    assert oblique.hours == pytest.approx(2 * nadir.hours)
    assert "two crossing passes" in oblique.basis and "45° down" in oblique.basis
    assert "oblique re-scan" in oblique.basis


def test_an_orbit_re_scan_flies_two_rings_at_its_stand_off() -> None:
    capture = rescan("orbit")
    orbit = sized(capture)
    standoff = 30 / math.tan(math.radians(30))
    assert _orbit_standoff_m(capture) == pytest.approx(standoff)
    ring = coverage.orbit_length_m(PADDOCK, standoff)
    assert ring is not None
    assert orbit.length_m == pytest.approx(ORBIT_RINGS * ring)
    assert (orbit.passes, orbit.spans, orbit.transit_m) == (2, 2, 0.0)
    assert orbit.hours == pytest.approx(orbit.length_m / 5 / 3600)
    assert "2 rings" in orbit.basis and "orbit re-scan" in orbit.basis
    assert f"{standoff:.0f} m out" in orbit.basis


def test_the_stand_off_is_where_the_camera_meets_the_ground_but_never_too_near() -> None:
    assert _orbit_standoff_m(rescan("orbit", gimbalPitchDeg=-45)) == pytest.approx(30.0)
    assert _orbit_standoff_m(rescan("orbit", gimbalPitchDeg=-90)) == ORBIT_MIN_STANDOFF_M
    assert _orbit_standoff_m(rescan("orbit", gimbalPitchDeg=-85, altitudeM=20)) == pytest.approx(
        ORBIT_MIN_STANDOFF_M
    )


def test_one_ring_is_the_hull_grown_by_the_stand_off() -> None:
    """A 100 m square paddock-sized outline: four sides and a circle of the stand-off."""
    side_deg_lat = 100 / 111_132.0
    side_deg_lon = 100 / (111_320.0 * math.cos(math.radians(47.644)))
    square = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.0, 47.644],
                [-122.0 + side_deg_lon, 47.644],
                [-122.0 + side_deg_lon, 47.644 + side_deg_lat],
                [-122.0, 47.644 + side_deg_lat],
                [-122.0, 47.644],
            ]
        ],
    }
    ring = coverage.orbit_length_m(square, 10.0)
    assert ring == pytest.approx(400 + 2 * math.pi * 10, rel=5e-3)
    assert coverage.orbit_length_m(square, 0.0) == pytest.approx(400, rel=5e-3)
    assert coverage.orbit_length_m(square, -1.0) is None


@pytest.mark.parametrize(
    "fields",
    [
        {"purpose": "lift-knowledge-fog", "pattern": "nadir"},  # rescanOf without a re-scan
        {"purpose": "lift-knowledge-fog", "pattern": "oblique", "rescanOf": None},
        {"gimbalPitchDeg": -20},
        {"gimbalPitchDeg": -95},
        {"pattern": "spiral"},
        {"rescanOf": {**REF, "reasons": ["vibes"]}},
        {"rescanOf": {**REF, "cells": -1}},
    ],
)
def test_a_capture_that_contradicts_itself_is_refused(fields: dict[str, Any]) -> None:
    valid = rescan("nadir").model_dump(mode="json", by_alias=True)
    with pytest.raises(ValidationError):
        ScoutCapture.model_validate({**valid, **fields})


def test_a_plan_keeps_a_re_scan_capture(client: TestClient) -> None:
    from tests.test_plans import BODY

    capture = rescan("orbit").model_dump(mode="json", by_alias=True)
    area: dict[str, Any] = dict(BODY["areas"][0])  # type: ignore[index, arg-type]
    created = client.post("/api/v1/plans", json={**BODY, "areas": [{**area, "capture": capture}]})
    assert created.status_code == 201, created.text
    stored = client.get(f"/api/v1/plans/{created.json()['id']}").json()
    assert stored["areas"][0]["capture"] == capture
    assert stored["areas"][0]["capture"]["rescanOf"]["epochId"] == REF["epochId"]

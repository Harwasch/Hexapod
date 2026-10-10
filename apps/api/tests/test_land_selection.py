from __future__ import annotations

import httpx
import pytest
from sqlalchemy.orm import Session

from app.config import Settings
from app.schemas.geojson import LineString, Point, Polygon
from app.schemas.land import BoundarySource
from app.schemas.land_selection import CandidateRequest, LandCandidate, SelectionInstruction
from app.services.errors import InvalidInputError
from app.services.land_selection import candidates, interpret
from tests.test_land import BODY


def line(identifier: str = "line-1") -> LandCandidate:
    return LandCandidate(
        id=identifier,
        label="Transmission line",
        geometry=LineString(coordinates=[[-122.14, 47.64], [-122.14, 47.65]]),
        source=BoundarySource(method="mapped-feature", label="OSM line"),
        distance_m=0,
    )


def test_grounded_corridor_width_and_ambiguity() -> None:
    settings = Settings(_env_file=None)
    first = line()
    result = interpret(
        SelectionInstruction(
            instruction="A 100 feet wide corridor along this line",
            candidates=[first],
            selected_ids=[first.id],
        ),
        settings,
    )
    assert result.operation == "corridor" and result.width_m == pytest.approx(30.48)
    assert result.candidate_ids == [first.id]
    ambiguous = interpret(
        SelectionInstruction(
            instruction="A 100 foot corridor along this line", candidates=[first, line("line-2")]
        ),
        settings,
    )
    assert ambiguous.operation == "clarify" and ambiguous.question
    assert (
        interpret(
            SelectionInstruction(instruction="A corridor along this line", candidates=[first]),
            settings,
        ).operation
        == "clarify"
    )
    with pytest.raises(InvalidInputError):
        interpret(
            SelectionInstruction(
                instruction="Select this", candidates=[first], selected_ids=["invented"]
            ),
            settings,
        )


def test_parcel_candidates_keep_holes_and_provenance(db: Session) -> None:
    geometry = BODY["boundary"]
    request = CandidateRequest(point=Point(coordinates=[-122.135, 47.645]), kind="parcel")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "type": "FeatureCollection",
                    "features": [
                        {"id": 7, "geometry": geometry, "properties": {"PARCEL_ID_NR": "12345"}}
                    ],
                },
            )
        )
    ) as client:
        result = candidates(db, request, client)
    assert result.status == "available"
    assert result.candidates[0].source.meaning == "recorded-parcel"
    assert result.candidates[0].source.record_id == "12345"
    assert result.candidates[0].geometry.model_dump() == geometry
    assert result.candidates[0].distance_m == 0


def test_no_parcel_coverage_does_not_call_provider(db: Session) -> None:
    def reject(request: httpx.Request) -> httpx.Response:
        pytest.fail("Uncovered region must not query the Washington provider")

    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        result = candidates(
            db, CandidateRequest(point=Point(coordinates=[2, 48]), kind="parcel"), client
        )
    assert result.status == "uncovered" and not result.candidates


def test_osm_uses_bounded_post_and_retains_mapped_feature_meaning(db: Session) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert b"timeout%3A20" in request.content
        return httpx.Response(
            200,
            json={
                "elements": [
                    {
                        "id": 9,
                        "type": "way",
                        "tags": {"power": "line", "voltage": "230000"},
                        "geometry": [
                            {"lon": -122.14, "lat": 47.64},
                            {"lon": -122.14, "lat": 47.65},
                        ],
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        result = candidates(
            db, CandidateRequest(point=Point(coordinates=[-122.14, 47.64]), kind="line"), client
        )
    assert result.status == "available", result.message
    assert result.candidates[0].geometry.type == "LineString"
    assert result.candidates[0].source.meaning == "physical-feature"
    assert result.candidates[0].properties["voltage"] == "230000"


def test_unsupported_local_exclusion_is_not_silently_ignored() -> None:
    candidate = LandCandidate(
        id="parcel",
        label="Parcel",
        geometry=Polygon.model_validate(BODY["boundary"]),
        source=BoundarySource(method="parcel", label="Parcel"),
        distance_m=0,
    )
    result = interpret(
        SelectionInstruction(instruction="Exclude the north half", candidates=[candidate]),
        Settings(_env_file=None),
    )
    assert result.operation == "clarify"

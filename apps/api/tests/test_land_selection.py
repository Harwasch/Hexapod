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


def square(lon: float, lat: float, half: float = 0.001) -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon - half, lat - half],
                [lon + half, lat - half],
                [lon + half, lat + half],
                [lon - half, lat + half],
                [lon - half, lat - half],
            ]
        ],
    }


ALAMEDA = "services5.arcgis.com/ROBnTHSNjoZ2Wm1P/arcgis/rest/services/Parcels/FeatureServer/0"


def test_california_parcels_come_from_the_county_service_with_provenance(db: Session) -> None:
    lon, lat = -122.2711, 37.8044
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "type": "FeatureCollection",
                "features": [
                    {"id": 7, "geometry": square(lon, lat), "properties": {"APN": "2-51-1"}},
                    # A parcel number stored as a number reads as the number it is.
                    {
                        "id": 8,
                        "geometry": square(lon + 0.003, lat),
                        "properties": {"APN": 4815.0},
                    },
                ],
            },
        )

    request = CandidateRequest(point=Point(coordinates=[lon, lat]), kind="parcel")
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        result = candidates(db, request, client)
    assert len(seen) == 1
    assert ALAMEDA + "/query" in str(seen[0].url)
    assert seen[0].url.params["outFields"] == "APN"
    assert seen[0].url.params["outSR"] == "4326"
    assert result.status == "available"
    assert "Alameda County" in result.message
    first, second = result.candidates
    assert first.id == "ca-parcel/06001/2-51-1"
    assert first.source.meaning == "recorded-parcel"
    assert first.source.record_id == "2-51-1"
    assert first.source.label == "Alameda County assessor parcel"
    assert first.distance_m == 0
    assert second.source.record_id == "4815"


def test_a_search_on_a_county_line_asks_both_counties(db: Session) -> None:
    hosts: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, json={"type": "FeatureCollection", "features": []})

    request = CandidateRequest(point=Point(coordinates=[-122.30, 37.8995]), kind="parcel")
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        result = candidates(db, request, client)
    assert sorted(hosts) == ["gis.cccounty.us", "services5.arcgis.com"]
    assert result.status == "empty"


def test_a_county_without_a_parcel_service_says_so_without_a_request(db: Session) -> None:
    def reject(request: httpx.Request) -> httpx.Response:
        pytest.fail("A county with no parcel service must not be queried")

    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        result = candidates(
            db, CandidateRequest(point=Point(coordinates=[-120.54, 41.49]), kind="parcel"), client
        )
    assert result.status == "uncovered" and not result.candidates
    assert "Modoc County does not publish parcel records" in result.message


def test_no_parcel_coverage_outside_california_does_not_call_a_provider(db: Session) -> None:
    def reject(request: httpx.Request) -> httpx.Response:
        pytest.fail("A point outside the registry must not query any parcel provider")

    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        for coordinates in ([2, 48], [-122.135, 47.645]):
            result = candidates(
                db, CandidateRequest(point=Point(coordinates=coordinates), kind="parcel"), client
            )
            assert result.status == "uncovered" and not result.candidates
            assert "California" in result.message


def test_an_unresponsive_county_service_is_reported_plainly(db: Session) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(503))) as client:
        result = candidates(
            db,
            CandidateRequest(point=Point(coordinates=[-122.2711, 37.8044]), kind="parcel"),
            client,
        )
    assert result.status == "unavailable"
    assert "Alameda County parcel service is not responding" in result.message


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

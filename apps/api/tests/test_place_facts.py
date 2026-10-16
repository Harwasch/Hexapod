"""Point facts for the inspector's "At this point" (app/research/place.py, /research/place).

Every answer names its source, basis and limits; a failure is "unavailable (why)" and never
reads as "nothing here".

The USGS, USDA and EPA responses for -120.72, 35.53 (Central Coast oak savanna) are trimmed
copies of what those services returned on 2026-10-16. FEMA's NFHL did not answer from this
machine when they were recorded, so the flood responses are only shaped like NFHL's: zone A,
SFHA 'T' and map 06079C are what it returned for this point while the lookup was built, and
the other zones are made up to exercise each case. Responses for other points, and the
land-cover tile (test_worldcover's synthetic COG), are fixtures, not recordings.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.analysis import worldcover
from app.research import place
from app.schemas.place import PlaceFact
from tests.test_worldcover import fixture_client

LON, LAT = -120.72, 35.53
Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def empty_cache() -> Iterator[None]:
    place.clear_cache()
    yield
    place.clear_cache()


class Recorder:
    """A MockTransport that keeps every request it answered."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.Client:
        def serve(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self.handler(request)

        return httpx.Client(transport=httpx.MockTransport(serve))


def answer(payload: object) -> Recorder:
    return Recorder(lambda _request: httpx.Response(200, json=payload))


def lookup(kind: str, recorder: Recorder, lon: float = LON, lat: float = LAT) -> PlaceFact:
    with recorder.client() as client:
        return place.lookup(kind, lon, lat, client)  # type: ignore[arg-type]


def test_elevation_is_a_cited_model_value_not_a_surveyed_height() -> None:
    epqs = answer(
        {
            "location": {"x": LON, "y": LAT, "spatialReference": {"wkid": 4326}},
            "value": "236.841323853",
            "rasterId": 36813,
            "resolution": 1,
            "attributes": {"AcquisitionDate": "1/0/2019"},
        }
    )
    fact = lookup("elevation", epqs)
    assert fact.status == "available" and fact.label == "236.8 m"
    assert fact.basis == "modelled-surface" and fact.vintage == "DEM acquired 2019"
    assert "not a surveyed height" in fact.caveat
    assert fact.data["elevationM"] == pytest.approx(236.841323853)
    assert fact.source is not None and fact.source.provider == "usgs-elevation"
    assert str(fact.source.url).startswith("https://epqs.nationalmap.gov/v1/json?")
    # A GET that returns readable JSON, so the citation opens the query itself.
    assert fact.source.link_url == fact.source.url
    assert fact.source.license == "U.S. public domain"
    assert epqs.requests[0].url.params["x"] == str(LON)
    off = lookup("elevation", answer({"value": -1000000}), lon=-120.0)
    assert off.status == "uncovered" and off.source is not None


MAP_UNIT = "Corducci and Typic Xerofluvents, 0 to 5 percent slopes, occasionally flooded, MLRA 14"


def test_soil_names_the_map_unit_and_its_components() -> None:
    sda = answer(
        {
            "Table": [
                ["mukey", "muname", "compname", "comppct_r", "drainagecl", "hydgrp", "slope_r"],
                [
                    "3027895",
                    MAP_UNIT,
                    "Corducci",
                    "50",
                    "Somewhat excessively drained",
                    "A",
                    "2",
                ],
                [
                    "3027895",
                    MAP_UNIT,
                    "Typic Xerofluvents",
                    "30",
                    "Somewhat excessively drained",
                    "A",
                    "2",
                ],
            ]
        }
    )
    fact = lookup("soil", sda)
    assert fact.status == "available" and fact.basis == "mapped-survey"
    assert fact.label.startswith("Corducci and Typic Xerofluvents")
    assert fact.detail is not None
    assert "Corducci 50%" in fact.detail and "Typic Xerofluvents 30%" in fact.detail
    assert "hydrologic group A" in fact.detail
    assert "not a soil test" in fact.caveat
    assert "POINT (-120.72" in sda.requests[0].content.decode()
    assert fact.data["components"][0] == {
        "name": "Corducci",
        "percent": 50.0,
        "drainage": "Somewhat excessively drained",
        "hydrologicGroup": "A",
        "slopePct": 2.0,
    }
    # Soil Data Access takes its SQL only as a POST: the citation opens Web Soil Survey, and
    # the exact query is kept with the answer.
    assert fact.source is not None
    assert str(fact.source.url) == "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
    assert str(fact.source.link_url) == "https://websoilsurvey.nrcs.usda.gov/app/"
    assert "SDA_Get_Mukey_from_intersection_with_WktWgs84" in fact.data["query"]
    assert json.loads(sda.requests[0].content)["query"] == fact.data["query"]
    empty = lookup("soil", answer({"Table": []}), lon=-120.0)
    assert empty.status == "empty" and empty.label == "No mapped soil components at this point."
    assert "POINT (-120.0000000000" in empty.data["query"]


def test_us_only_sources_answer_uncovered_abroad_without_asking() -> None:
    nobody = Recorder(lambda _request: httpx.Response(500))
    for kind in ("soil", "flood-zone", "ecoregion", "elevation"):
        fact = lookup(kind, nobody, lon=2.35, lat=48.85)
        assert fact.status == "uncovered" and fact.source is None, kind
    assert nobody.requests == []


def test_flood_zone_is_a_point_query_without_geometry() -> None:
    nfhl = answer(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "FLD_ZONE": "A",
                        "ZONE_SUBTY": None,
                        "SFHA_TF": "T",
                        "DFIRM_ID": "06079C",
                    },
                    "geometry": None,
                }
            ],
        }
    )
    fact = lookup("flood-zone", nfhl)
    params = nfhl.requests[0].url.params
    assert params["geometryType"] == "esriGeometryPoint"
    assert params["returnGeometry"] == "false"
    assert params["geometry"] == f"{LON},{LAT}"
    assert fact.status == "available" and fact.label == "Zone A"
    assert fact.detail == "Special Flood Hazard Area · map 06079C"
    assert fact.basis == "regulatory-map" and fact.data["sfha"] is True
    minimal = lookup(
        "flood-zone",
        answer(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "properties": {
                            "FLD_ZONE": "X",
                            "ZONE_SUBTY": "AREA OF MINIMAL FLOOD HAZARD",
                            "SFHA_TF": "F",
                            "DFIRM_ID": "53033C",
                        }
                    }
                ],
            }
        ),
        lon=-122.13,
        lat=47.64,
    )
    assert minimal.detail == "Area of minimal flood hazard · map 53033C"
    empty = lookup("flood-zone", answer({"type": "FeatureCollection", "features": []}), lon=-120.0)
    assert empty.status == "empty" and empty.label == "No mapped zone"
    assert "does not establish low flood risk" in empty.caveat


def nfhl(*zones: tuple[str, str | None]) -> Recorder:
    """An NFHL point answer with one record per (FLD_ZONE, ZONE_SUBTY), none in the SFHA."""
    return answer(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "FLD_ZONE": zone,
                        "ZONE_SUBTY": subtype,
                        "SFHA_TF": "F",
                        "DFIRM_ID": "06079C",
                    },
                    "geometry": None,
                }
                for zone, subtype in zones
            ],
        }
    )


def test_unstudied_and_unmapped_flood_zones_never_read_as_low_risk() -> None:
    # Zone D: possible but undetermined hazard, never studied. Not "outside the SFHA".
    unstudied = lookup("flood-zone", nfhl(("D", None)), lon=-120.1)
    assert unstudied.status == "available" and unstudied.label == "Zone D"
    assert unstudied.detail == "Undetermined flood hazard: not studied · map 06079C"
    assert "Outside" not in (unstudied.detail or "")
    assert "not low" in unstudied.caveat
    # Not included in the map at all: nothing mapped, said as plainly as "No mapped zone".
    excluded = lookup("flood-zone", nfhl(("AREA NOT INCLUDED", None)), lon=-120.2)
    assert excluded.status == "empty"
    assert excluded.label == "Not included in FEMA flood mapping"
    assert "does not establish low flood risk" in excluded.caveat
    assert excluded.source is not None and excluded.data["zone"] == "AREA NOT INCLUDED"
    # Where it touches a mapped zone, the mapped zone is the answer.
    edge = lookup(
        "flood-zone",
        nfhl(("AREA NOT INCLUDED", None), ("X", "AREA OF MINIMAL FLOOD HAZARD")),
        lon=-120.3,
    )
    assert edge.status == "available" and edge.label == "Zone X"
    assert edge.data["zones"] == ["X"]
    water = lookup("flood-zone", nfhl(("OPEN WATER", None)), lon=-120.4)
    assert water.label == "Open water" and water.detail == "Mapped as open water · map 06079C"


def test_ecoregion_is_the_level_iv_name_with_its_level_iii() -> None:
    epa = answer(
        {
            "features": [
                {
                    "attributes": {
                        "US_L4CODE": "6ak",
                        "US_L4NAME": "Paso Robles Hills and Valleys",
                        "US_L3NAME": "Central California Foothills and Coastal Mountains",
                        "NA_L2NAME": "MEDITERRANEAN CALIFORNIA",
                    }
                }
            ]
        }
    )
    fact = lookup("ecoregion", epa)
    assert fact.label == "Paso Robles Hills and Valleys" and fact.basis == "regional-framework"
    assert fact.detail == "Level III: Central California Foothills and Coastal Mountains"
    assert fact.data["level4Code"] == "6ak"
    assert fact.vintage == "2011 edition" and "1:250,000" in fact.caveat
    assert epa.requests[0].url.params["geometryType"] == "esriGeometryPoint"
    nothing = lookup("ecoregion", answer({"features": []}), lon=-120.0)
    assert nothing.status == "empty"


def test_land_cover_reads_one_worldcover_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worldcover, "tile_for", lambda lon, lat: "N45W123")
    with fixture_client() as client:
        fact = place.lookup("land-cover", -122.14, 47.65, client)
    assert fact.status == "available" and fact.label == "Tree cover (class 10)"
    assert fact.basis == "classified-imagery" and fact.vintage == "2021"
    assert fact.source is not None and "N45W123" in str(fact.source.url)
    assert "CC BY 4.0" in fact.source.license
    # The tile is a 10-20 MB image: the citation opens the WorldCover viewer instead, and the
    # tile read is kept with the answer.
    assert str(fact.source.link_url) == "https://viewer.esa-worldcover.org/worldcover/"
    assert fact.data["tileUrl"] == str(fact.source.url)
    missing = lookup("land-cover", Recorder(lambda _request: httpx.Response(404)), lon=-150.0)
    assert missing.status == "uncovered" and missing.label == "WorldCover has no tile here."


def test_tile_names_follow_the_worldcover_grid() -> None:
    assert worldcover.tile_for(LON, LAT) == "N33W123"
    assert worldcover.tile_for(-150.0, 0.5) == "N00W150"
    assert worldcover.tile_for(151.2, -33.9) == "S36E150"


def test_failures_say_unavailable_and_why_without_the_response_body() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret upstream detail", request=request)

    fact = lookup("elevation", Recorder(refuse))
    assert fact.status == "unavailable"
    assert fact.label == "USGS elevation could not be retrieved (ConnectError)"
    assert "secret" not in fact.model_dump_json()
    outage = lookup(
        "flood-zone", Recorder(lambda _r: httpx.Response(503, text="internal stack")), lon=-120.0
    )
    assert outage.label == "FEMA mapped flood zones could not be retrieved (HTTP 503)"
    assert "internal stack" not in outage.model_dump_json()
    oversize = lookup(
        "ecoregion",
        Recorder(lambda _r: httpx.Response(200, content=json.dumps({"x": "y" * 2_100_000}))),
        lon=-121.0,
    )
    assert oversize.status == "unavailable" and "(SourceError)" in oversize.label


def test_lookups_past_the_concurrency_bound_answer_busy_without_asking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(place, "_slots", slots)
    monkeypatch.setattr(place, "SLOT_WAIT_S", 0.01)
    epqs = answer({"value": 12.5, "resolution": 10})
    assert slots.acquire(blocking=False)  # A slow agency holds the only slot.
    try:
        busy = lookup("elevation", epqs)
    finally:
        slots.release()
    assert busy.status == "unavailable" and epqs.requests == []
    assert busy.label == (
        "USGS elevation was not asked because the server is busy with other lookups"
    )
    assert busy.source is None and busy.cached is False
    # Not cached: the next lookup asks, and a finished lookup gives its slot back.
    again = lookup("elevation", epqs)
    assert again.status == "available" and again.cached is False
    assert len(epqs.requests) == 1
    assert slots.acquire(blocking=False)
    slots.release()


def test_a_failed_lookup_gives_its_slot_back(monkeypatch: pytest.MonkeyPatch) -> None:
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(place, "_slots", slots)

    def boom(_client: httpx.Client, _lon: float, _lat: float) -> place._Answer:
        raise RuntimeError("not one of the source errors")

    monkeypatch.setitem(place.LOOKUPS, "elevation", boom)
    with pytest.raises(RuntimeError):
        lookup("elevation", answer({}))
    assert slots.acquire(blocking=False)
    slots.release()


def test_answers_are_cached_and_say_so() -> None:
    epqs = answer({"value": 12.5, "resolution": 10})
    first = lookup("elevation", epqs)
    second = lookup("elevation", epqs, lon=LON + 0.000001)
    assert first.cached is False and second.cached is True
    assert second.label == first.label and len(epqs.requests) == 1
    with epqs.client() as client:
        # Retry only skips a cached failure; a cached answer is still the answer.
        assert place.lookup("elevation", LON, LAT, client, retry=True).cached is True
    assert len(epqs.requests) == 1


def test_a_cached_failure_is_asked_again_only_on_retry() -> None:
    state = {"up": False}

    def flaky(_request: httpx.Request) -> httpx.Response:
        if state["up"]:
            return httpx.Response(200, json={"value": 12.5})
        return httpx.Response(503)

    source = Recorder(flaky)
    assert lookup("elevation", source).status == "unavailable"
    state["up"] = True
    again = lookup("elevation", source)
    assert again.status == "unavailable" and again.cached is True
    with source.client() as client:
        fresh = place.lookup("elevation", LON, LAT, client, retry=True)
    assert fresh.status == "available" and fresh.cached is False
    assert len(source.requests) == 2


def test_place_route_validates_and_answers_under_pilot_auth(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, float, float]] = []
    real = place.lookup

    def fake(kind: str, lon: float, lat: float, *, retry: bool) -> PlaceFact:
        calls.append((kind, lon, lat))
        with answer({"Table": []}).client() as http:
            return real(kind, lon, lat, http, retry=retry)  # type: ignore[arg-type]

    monkeypatch.setattr(place, "lookup", fake)
    response = client.get("/api/v1/research/place/soil?longitude=-120.72&latitude=35.53")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "soil" and body["status"] == "empty"
    assert body["source"]["provider"] == "usda-soils" and "retrievedAt" in body
    assert calls == [("soil", -120.72, 35.53)]
    assert client.get("/api/v1/research/place/foo?longitude=0&latitude=0").status_code == 422
    assert client.get("/api/v1/research/place/soil?longitude=0&latitude=91").status_code == 422
    assert client.get("/api/v1/research/place/soil?latitude=0").status_code == 422


def test_place_lookups_are_rate_limited(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        place,
        "lookup",
        lambda kind, lon, lat, retry: PlaceFact(
            kind=kind,
            status="unavailable",
            label="fixture",
            basis="modelled-surface",
            caveat="fixture",
            longitude=lon,
            latitude=lat,
            retrieved_at="2026-10-16T00:00:00Z",
            source=None,
        ),
    )
    url = "/api/v1/research/place/elevation?longitude=-120.72&latitude=35.53"
    for _ in range(25):
        assert client.get(url).status_code == 200
    refused = client.get(url)
    assert refused.status_code == 429 and int(refused.headers["Retry-After"]) >= 1

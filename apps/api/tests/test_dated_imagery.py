"""The Time Lantern's catalog of dated public imagery (app/research/providers/dated_imagery.py,
GET /api/v1/imagery/dated). Every outside service is faked with httpx.MockTransport: what is
asked, and what each answer becomes, are checked."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from datetime import date
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.v1 import imagery
from app.research.providers import dated_imagery
from app.research.providers.base import SourceContext
from app.research.providers.open_data import ADAPTERS, OVERVIEW_SOURCES, SOURCES, retrieve
from app.schemas.geojson import Polygon

BBOX = (-120.06, 34.68, -120.02, 34.72)
SEARCH_ID = "0123456789abcdef0123456789abcdef"
METADATA = (
    "https://naipeuwest.blob.core.windows.net/naip/v002/ca/2016/ca_fgdc_2016/34120/"
    "m_3412024_sw_10_h_20160615.txt"
)
FGDC = """Data_Quality_Information:
    Positional_Accuracy:
      Horizontal_Positional_Accuracy:
        Horizontal_Positional_Accuracy_Report:
          NAIP horizontal accuracy specifications have evolved over
          the life of the program.  From 2009 to present all NAIP imagery acquisitions used
          the +/- 6 meters to ground specification.
    Lineage:
"""
TOPO = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/HistoricalTopo/GeoTIFF/CA/"


def square(west: float, south: float, size: float = 0.07) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [west + size, south],
                [west + size, south + size],
                [west, south + size],
                [west, south],
            ]
        ],
    }


def naip_item(identifier: str, year: int, day: str, *, metadata: bool) -> dict[str, Any]:
    assets: dict[str, Any] = {"image": {"href": "https://naipeuwest.blob.core.windows.net/x.tif"}}
    if metadata:
        assets["metadata"] = {"href": METADATA, "type": "text/plain"}
    return {
        "id": identifier,
        "geometry": square(-120.1, 34.65),
        "properties": {
            "naip:year": str(year),
            "naip:state": "ca",
            "datetime": f"{day}T00:00:00Z",
            "gsd": 0.6 if year >= 2016 else 1.0,
        },
        "assets": assets,
        "links": [
            {
                "rel": "self",
                "href": f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/{identifier}",
            }
        ],
    }


def sentinel_item(identifier: str, when: str, cloud: float) -> dict[str, Any]:
    return {
        "id": identifier,
        "geometry": square(-121, 34, 1.2),
        "properties": {"datetime": when, "eo:cloud_cover": cloud},
        "links": [
            {
                "rel": "self",
                "href": f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/{identifier}",
            }
        ],
    }


def topo_record(
    stem: str, box: tuple[float, float, float, float], url: str = TOPO
) -> dict[str, Any]:
    return {
        "title": stem,
        "metaUrl": f"https://www.sciencebase.gov/catalog/item/{stem[-20:].replace(' ', '')}",
        "urls": {"GeoTIFF": url + stem.replace(" ", "%20") + ".tif"},
        "boundingBox": {"minX": box[0], "minY": box[1], "maxX": box[2], "maxY": box[3]},
    }


TOPO_RECORDS = [
    topo_record("CA_Los Olivos_292579_1948_24000_geo", (-120.125, 34.625, -120.0, 34.75)),
    # Another scan of the same edition: counted once.
    topo_record("CA_Los Olivos_292580_1948_24000_geo", (-120.125, 34.625, -120.0, 34.75)),
    topo_record("CA_Los Olivos_298063_1919_62500_geo", (-120.25, 34.5, -120.0, 34.75)),
    # Off the allow-listed prefix: never offered.
    topo_record(
        "CA_Los Olivos_1_1930_24000_geo",
        (-120.125, 34.625, -120.0, 34.75),
        url="https://example.com/CA/",
    ),
    # Coarse, and finer sheets exist.
    topo_record("CA_Santa Maria_707594_1948_250000_geo", (-121.0, 33.0, -120.0, 35.0)),
]


class Fake:
    """Answers each outside service as the test sets it, and records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.lock = threading.Lock()
        self.naip: Callable[[httpx.Request], httpx.Response] = lambda _: httpx.Response(
            200,
            json={
                "features": [
                    naip_item("ca_2016_a", 2016, "2016-06-15", metadata=True),
                    naip_item("ca_2016_b", 2016, "2016-06-20", metadata=True),
                    naip_item("ca_2012_a", 2012, "2012-05-18", metadata=False),
                ],
                "links": [],
            },
        )
        self.sentinel: dict[int, list[dict[str, Any]]] = {
            2017: [sentinel_item("S2A_MSIL2A_20170711", "2017-07-11T18:40:00Z", 0.97)]
        }
        self.topo: Callable[[httpx.Request], httpx.Response] = lambda _: httpx.Response(
            200, json={"items": TOPO_RECORDS, "total": len(TOPO_RECORDS)}
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            self.requests.append(request)
        url = str(request.url)
        if url.startswith(dated_imagery.PC_MOSAIC_REGISTER):
            return httpx.Response(200, json={"searchid": SEARCH_ID})
        if url.startswith(dated_imagery.PC_SEARCH) and request.method == "GET":
            return self.naip(request)
        if url.startswith(dated_imagery.PC_SEARCH):
            body = json.loads(request.content)
            year = int(body["datetime"][:4])
            return httpx.Response(200, json={"features": self.sentinel.get(year, [])})
        if url == METADATA:
            return httpx.Response(200, text=FGDC)
        if url.startswith(dated_imagery.TNM_PRODUCTS):
            return self.topo(request)
        return httpx.Response(404)

    def to(self, prefix: str, method: str | None = None) -> list[httpx.Request]:
        return [
            r
            for r in self.requests
            if str(r.url).startswith(prefix) and (method is None or r.method == method)
        ]


@pytest.fixture
def fake() -> Iterator[Fake]:
    dated_imagery.clear_caches()
    yield Fake()
    dated_imagery.clear_caches()


def client_for(fake: Fake) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(fake))


def test_naip_years_group_items_and_register_one_mosaic_per_year(fake: Fake) -> None:
    sources = dated_imagery.naip_years(BBOX, client_for(fake))
    assert [s.id for s in sources] == ["naip-2016", "naip-2012"]
    newest = sources[0]
    assert newest.date_label == "Flown 15 to 20 June 2016"
    assert newest.sort_date == date(2016, 6, 20)
    assert [p.observed_on for p in newest.parts] == [date(2016, 6, 20), date(2016, 6, 15)]
    assert newest.resolution_m == 0.6
    assert f"/mosaic/{SEARCH_ID}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x?" in (
        newest.tiles.url_template
    )
    assert "asset_bidx=image%7C1%2C2%2C3" in newest.tiles.url_template
    assert newest.license == "U.S. public domain (USDA Farm Service Agency)"
    registers = fake.to(dated_imagery.PC_MOSAIC_REGISTER, "POST")
    assert len(registers) == 2
    bodies = [json.loads(r.content) for r in registers]
    years = sorted(
        arg["args"][1]
        for body in bodies
        for arg in body["filter"]["args"]
        if arg["args"][0] == {"property": "naip:year"}
    )
    assert years == ["2012", "2016"]
    # The NAIP search itself is one GET with the area.
    (search,) = fake.to(dated_imagery.PC_SEARCH, "GET")
    query = parse_qs(urlsplit(str(search.url)).query)
    assert query["collections"] == ["naip"]
    assert query["bbox"] == ["-120.06,34.68,-120.02,34.72"]


def test_naip_accuracy_quotes_the_record_or_says_not_stated(fake: Fake) -> None:
    sources = dated_imagery.naip_years(BBOX, client_for(fake))
    stated = sources[0].parts[0].accuracy
    assert stated.kind == "stated-by-source"
    assert stated.value_m == 6
    assert stated.basis == (
        "From 2009 to present all NAIP imagery acquisitions used the +/- 6 meters to ground "
        "specification."
    )
    assert str(stated.citation_url) == METADATA
    # Fetched once for the year, not once per quarter quad.
    assert len(fake.to(METADATA)) == 1
    unstated = sources[1].parts[0].accuracy
    assert unstated.kind == "not-stated"
    assert unstated.value_m is None
    assert unstated.basis == "This item's record states no positional accuracy."


def test_naip_follows_the_search_pages(fake: Fake) -> None:
    pages = iter(
        [
            {
                "features": [naip_item("a", 2018, "2018-07-22", metadata=False)],
                "links": [
                    {
                        "rel": "next",
                        "href": f"{dated_imagery.PC_SEARCH}?collections=naip&token=next%3Aab",
                    }
                ],
            },
            {"features": [naip_item("b", 2014, "2014-06-01", metadata=False)], "links": []},
        ]
    )
    fake.naip = lambda _: httpx.Response(200, json=next(pages))
    sources = dated_imagery.naip_years(BBOX, client_for(fake))
    assert [s.id for s in sources] == ["naip-2018", "naip-2014"]
    second = fake.to(dated_imagery.PC_SEARCH, "GET")[1]
    assert parse_qs(urlsplit(str(second.url)).query)["token"] == ["next:ab"]


def test_sentinel_takes_the_clearest_dry_season_scene_of_each_year(fake: Fake) -> None:
    sources = dated_imagery.sentinel2_years(BBOX, client_for(fake), today=date(2019, 10, 1))
    assert [s.id for s in sources] == ["sentinel-2-S2A_MSIL2A_20170711"]
    (scene,) = sources
    assert scene.date_label == "Seen 11 July 2017"
    assert scene.resolution_m == 10
    assert scene.attribution.startswith("Contains modified Copernicus Sentinel data 2017")
    assert "item=S2A_MSIL2A_20170711" in scene.tiles.url_template
    assert scene.tiles.maximum_level == 15
    assert scene.parts[0].accuracy.kind == "not-stated"
    bodies = sorted(
        (json.loads(r.content) for r in fake.to(dated_imagery.PC_SEARCH, "POST")),
        key=lambda b: b["datetime"],
    )
    assert [b["datetime"][:4] for b in bodies] == ["2016", "2017", "2018", "2019"]
    assert bodies[0]["datetime"] == "2016-06-01T00:00:00Z/2016-09-30T23:59:59Z"
    assert bodies[0]["query"] == {"eo:cloud_cover": {"lt": 10}}
    assert bodies[0]["sortby"] == [{"field": "eo:cloud_cover", "direction": "asc"}]
    assert bodies[0]["limit"] == 1
    assert bodies[0]["collections"] == ["sentinel-2-l2a"]


def test_topo_sheets_keep_allow_listed_editions_with_their_standard(fake: Fake) -> None:
    sources = dated_imagery.usgs_topo_sheets(BBOX, client_for(fake))
    assert [s.title for s in sources] == ["USGS 1:24,000 Los Olivos", "USGS 1:62,500 Los Olivos"]
    modern, old = sources
    assert modern.date_label == "Map dated 1948"
    assert modern.map_scale == 24_000
    assert modern.parts[0].accuracy.kind == "standard"
    assert modern.parts[0].accuracy.value_m == 12.19
    assert str(modern.parts[0].accuracy.citation_url) == dated_imagery.NMAS_URL
    assert old.parts[0].accuracy.kind == "not-stated"
    assert old.parts[0].accuracy.basis.startswith("Predates the National Map Accuracy Standards")
    assert modern.tiles.scheme == "geographic"
    assert modern.tiles.url_template.startswith(
        "/api/v1/imagery/usgs-topo/tiles/{z}/{x}/{y}.png?sheet=CA%2FCA_Los%20Olivos_292579_1948"
    )
    assert "&w=-120.125&s=34.625&e=-120.0&n=34.75" in modern.tiles.url_template
    # The datum shift is named, with its stated accuracy.
    assert modern.warnings[0].startswith("Shifted from NAD27 to WGS 84 with")
    # The neatline is shifted to WGS 84: tens of metres west of the NAD27 graticule.
    assert -120.0015 < modern.bounds[2] < -120.0
    assert modern.date_meaning.startswith("Catalog publication year")


def test_topo_offers_a_coarse_sheet_only_when_nothing_finer_covers(fake: Fake) -> None:
    fake.topo = lambda _: httpx.Response(200, json={"items": TOPO_RECORDS[-1:]})
    sources = dated_imagery.usgs_topo_sheets(BBOX, client_for(fake))
    assert [s.map_scale for s in sources] == [250_000]


def test_a_failing_provider_is_unavailable_and_the_others_answer(fake: Fake) -> None:
    fake.topo = lambda _: httpx.Response(503)
    catalog = dated_imagery.discover(BBOX, client_for(fake))
    statuses = {p.provider: p for p in catalog.providers}
    assert statuses["usgs-historical-topo-sheets"].status == "unavailable"
    assert "HTTP 503" in statuses["usgs-historical-topo-sheets"].summary
    assert statuses["naip-aerial"].status == "available"
    assert any(s.kind == "aerial-photo" for s in catalog.sources)
    # Newest first, across providers.
    dates = [s.sort_date for s in catalog.sources]
    assert dates == sorted(dates, reverse=True)


def test_a_catalog_is_answered_from_the_cache_the_second_time(fake: Fake) -> None:
    http = client_for(fake)
    first = dated_imagery.discover(BBOX, http)
    asked = len(fake.requests)
    second = dated_imagery.discover((-120.059, 34.681, -120.021, 34.719), http)
    assert len(fake.requests) == asked
    assert second.cached and not first.cached
    assert second.bounds == [-120.06, 34.68, -120.02, 34.72]


def test_round_bbox_goes_outward() -> None:
    assert dated_imagery.round_bbox((-120.051, 34.681, -120.021, 34.719)) == (
        -120.06,
        34.68,
        -120.02,
        34.72,
    )


def test_the_three_sources_are_registered_but_not_run_by_the_overview() -> None:
    for provider in ("naip-aerial", "sentinel-2-true-colour", "usgs-historical-topo-sheets"):
        assert provider in SOURCES
        assert provider in ADAPTERS
        assert provider not in OVERVIEW_SOURCES
    assert OVERVIEW_SOURCES == (
        "usgs-elevation",
        "nasa-power",
        "gbif-occurrences",
        "usda-soils",
        "fema-flood-zones",
    )


def test_as_a_research_source_it_lists_without_storing_evidence(fake: Fake) -> None:
    boundary = Polygon.model_validate(square(-120.05, 34.69, 0.02))
    result = retrieve("naip-aerial", SourceContext(boundary=boundary), client_for(fake))
    assert result.status == "available"
    assert result.evidence == []
    assert "not stored as evidence" in result.summary
    assert result.data["sources"][0]["title"] == "NAIP 2016 aerial photograph"


# ---- The route ------------------------------------------------------------------------------


@pytest.fixture
def routed(fake: Fake, monkeypatch: pytest.MonkeyPatch) -> Fake:
    monkeypatch.setattr(imagery, "_http", lambda: client_for(fake))
    return fake


def test_route_lists_dated_imagery_in_camel_case(client: TestClient, routed: Fake) -> None:
    response = client.get(
        "/api/v1/imagery/dated",
        params={"west": BBOX[0], "south": BBOX[1], "east": BBOX[2], "north": BBOX[3]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["bounds"] == [-120.06, 34.68, -120.02, 34.72]
    first = body["sources"][0]
    assert {"dateLabel", "dateMeaning", "sortDate", "licenseUrl", "parts", "tiles"} <= set(first)
    assert first["tiles"]["urlTemplate"]
    assert response.headers["cache-control"] == "public, max-age=600"


def test_route_refuses_an_area_larger_than_a_ranch(client: TestClient, routed: Fake) -> None:
    response = client.get(
        "/api/v1/imagery/dated",
        params={"west": -120.5, "south": 34.5, "east": -120.0, "north": 34.6},
    )
    assert response.status_code == 422
    assert routed.requests == []


def test_route_spends_no_token_on_a_cached_catalog(client: TestClient, routed: Fake) -> None:
    params = {"west": BBOX[0], "south": BBOX[1], "east": BBOX[2], "north": BBOX[3]}
    limit = dated_imagery_limit(client)
    for _ in range(limit + 5):
        assert client.get("/api/v1/imagery/dated", params=params).status_code == 200
    # A new area past the burst is limited.
    statuses = [
        client.get(
            "/api/v1/imagery/dated",
            params={**params, "west": BBOX[0] - 0.01 * (i + 1)},
        ).status_code
        for i in range(limit + 1)
    ]
    assert statuses[-1] == 429


def dated_imagery_limit(client: TestClient) -> int:
    from app.services.ratelimit import IMAGERY_CATALOG

    return IMAGERY_CATALOG.burst

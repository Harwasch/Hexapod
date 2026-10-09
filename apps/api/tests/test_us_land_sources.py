from __future__ import annotations

import json
import uuid

import httpx
from shapely.geometry import box, mapping, shape

from app.research.outputs import overview_outputs
from app.research.providers.base import SourceContext
from app.research.providers.open_data import retrieve
from app.schemas.geojson import Polygon
from tests.test_land import BODY

CONTEXT = SourceContext(Polygon.model_validate(BODY["boundary"]))


def test_soil_point_components_are_not_parcel_coverage() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.method == "POST"
        assert "WktWgs84('POINT (" in body["query"]
        assert body["format"] == "JSON+COLUMNNAME"
        return httpx.Response(
            200,
            json={
                "Table": [
                    ["mukey", "muname", "compname", "comppct_r", "drainagecl", "hydgrp", "slope_r"],
                    ["1", "Survey unit", "Loam", "65", "Well drained", "B", "4"],
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = retrieve("usda-soils", CONTEXT, client)
    assert result.status == "available"
    assert result.data["records"][0]["comppct_r"] == 65
    assert "not this land" in result.evidence[0][1].relevance_note
    finding, artifacts = overview_outputs(result, [uuid.uuid4()])
    assert finding is not None
    assert finding.category == "physical"
    assert artifacts[0].output.kind == "table"
    assert "map-unit" in artifacts[0].method


def test_flood_zones_are_clipped_to_land_and_preserve_holes() -> None:
    footprint = {
        **BODY["boundary"],
        "coordinates": [
            *BODY["boundary"]["coordinates"],
            [[-122.138, 47.642], [-122.136, 47.642], [-122.136, 47.644], [-122.138, 47.642]],
        ],
    }
    context = SourceContext(Polygon.model_validate(footprint))
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "geometry": mapping(box(-123, 47, -122, 48)),
                            "properties": {"FLD_ZONE": "AE", "FLD_AR_ID": "9", "SFHA_TF": "T"},
                        }
                    ],
                },
            )
        )
    ) as client:
        result = retrieve("fema-flood-zones", context, client)
    assert result.status == "available"
    record = result.data["records"][0]
    assert shape(record["geometry"]).equals(shape(footprint))
    assert record["intersectedAreaM2"] > 0
    assert result.evidence[0][1].spatial_relevance == "intersects"
    finding, artifacts = overview_outputs(result, [uuid.uuid4()])
    assert finding is not None
    assert finding.category == "hazards"
    assert [artifact.output.kind for artifact in artifacts] == ["map", "table"]


def test_outage_empty_and_outside_coverage_remain_distinct() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="<html>Maintenance</html>")
        )
    ) as client:
        assert retrieve("usda-soils", CONTEXT, client).status == "unavailable"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"type": "FeatureCollection", "features": []})
        )
    ) as client:
        result = retrieve("fema-flood-zones", CONTEXT, client)
        assert result.status == "empty" and "does not establish" in result.summary
    outside = SourceContext(Polygon(coordinates=[[[2, 48], [2.01, 48], [2.01, 48.01], [2, 48]]]))
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("must not fetch"))
        )
    ) as client:
        assert retrieve("usda-soils", outside, client).status == "uncovered"
        assert retrieve("fema-flood-zones", outside, client).status == "uncovered"

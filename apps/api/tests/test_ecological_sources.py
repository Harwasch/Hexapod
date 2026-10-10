from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import ValidationError
from shapely.geometry import Polygon

from app.research.providers.base import SourceContext
from app.research.providers.ecology import ECOLOGY_SOURCES, ecological_sites, regions
from app.research.providers.taxonomy import CHECKLIST, match
from app.schemas.land_ecology import EcologyRequest, TaxonQuery
from tests.test_solar import BOUNDARY


def taxon_client(raw: dict[str, Any], index: str = CHECKLIST) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["checklistKey"] == CHECKLIST
        if request.url.path.endswith("/metadata"):
            return httpx.Response(
                200,
                json={
                    "created": "2026-07-18",
                    "mainIndex": {
                        "datasetKey": index,
                        "clbDatasetKey": "315557",
                        "datasetAlias": "COL26.6 XR",
                    },
                },
            )
        assert request.url.params["scientificName"] == "Quercus alba"
        assert request.url.params["kingdom"] == "Plantae"
        return httpx.Response(200, json=raw)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("match_type", ["EXACT", "FUZZY", "HIGHERRANK", "NONE"])
def test_taxonomic_matches_keep_uncertainty_and_exclude_separately_licensed_status(
    match_type: str,
) -> None:
    raw: dict[str, Any] = {
        "usage": {
            "key": "4R47Z",
            "name": "Quercus alba L.",
            "rank": "SPECIES",
            "status": "ACCEPTED",
        },
        "diagnostics": {"matchType": match_type, "confidence": 99},
        "additionalStatus": [{"datasetAlias": "IUCN", "status": "Endangered"}],
    }
    if match_type == "NONE":
        raw.pop("usage")
    with taxon_client(raw) as client:
        result = match(TaxonQuery(scientific_name="Quercus alba", kingdom="Plantae"), client)
    assert result.status == ("empty" if match_type == "NONE" else "available")
    assert result.data["matchType"] == match_type
    assert (
        result.data["nativeStatus"] == "not-assessed"
        and result.data["fieldIdentification"] == "not-verified"
    )
    assert result.data["indexObserved"]["clbDatasetKey"] == "315557"
    assert "Endangered" not in result.evidence[0][1].excerpt
    assert json.loads(result.evidence[0][1].excerpt) == result.data


def test_taxonomy_index_mismatch_and_long_details_remain_bounded() -> None:
    query = TaxonQuery(scientific_name="Quercus alba", kingdom="Plantae")
    with taxon_client({}, index="other-taxonomy") as client:
        assert match(query, client).status == "unavailable"
    raw = {
        "usage": {"key": "4R47Z", "name": "Quercus alba L."},
        "diagnostics": {"matchType": "EXACT"},
        "classification": [
            {"key": "a" * 300, "name": "界" * 300, "rank": "r" * 300} for _ in range(40)
        ],
        "alternatives": [
            {
                "usage": {
                    "key": "a" * 300,
                    "name": "界" * 300,
                    "canonicalName": "x" * 300,
                    "authorship": "z" * 300,
                    "rank": "r" * 300,
                    "status": "s" * 300,
                }
            }
            for _ in range(20)
        ],
    }
    with taxon_client(raw) as client:
        result = match(query, client)
    assert result.status == "available"
    assert len(result.evidence[0][1].excerpt) <= 29000
    assert json.loads(result.evidence[0][1].excerpt) == result.data
    assert result.data["classificationTruncated"] and result.data["alternativesTruncated"]


def test_regional_query_uses_actual_polygon_and_hole_orientation() -> None:
    footprint = BOUNDARY.model_copy(deep=True)
    assert footprint.type == "Polygon"
    footprint.coordinates.append(
        [[-77.052, 38.888], [-77.05, 38.888], [-77.05, 38.89], [-77.052, 38.89], [-77.052, 38.888]]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert (
            request.method == "POST"
            and str(request.url) == ECOLOGY_SOURCES["epa-ecoregions"].endpoint
        )
        params = parse_qs(request.content.decode())
        geometry = json.loads(params["geometry"][0])
        assert not Polygon(geometry["rings"][0]).exterior.is_ccw
        assert Polygon(geometry["rings"][1]).exterior.is_ccw
        assert params["geometryType"] == ["esriGeometryPolygon"] and params["returnGeometry"] == [
            "false"
        ]
        return httpx.Response(
            200,
            json={
                "features": [
                    {
                        "attributes": {
                            "OBJECTID": 674,
                            "US_L4CODE": "65n",
                            "US_L4NAME": "Chesapeake Rolling Coastal Plain",
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = regions(SourceContext(footprint), client)
    assert result.status == "available" and result.data["edition"] == "December 2011"
    assert "site habitat" in result.summary and "1:250,000" in result.evidence[0][1].relevance_note
    assert "intersectedAreaM2" not in result.data["records"][0]


def test_soil_ecological_links_remain_point_associations_not_land_fractions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        query = json.loads(request.content)["query"]
        assert "POINT (-77.0500000000 38.8885000000)" in query
        return httpx.Response(
            200,
            json={
                "Table": [
                    [
                        "mukey",
                        "muname",
                        "cokey",
                        "compname",
                        "comppct_r",
                        "ecoclassid",
                        "ecoclassname",
                        "ecoclasstypename",
                    ],
                    [
                        "1",
                        "Mapped unit",
                        "2",
                        "Component",
                        "60",
                        "R015XF008CA",
                        "Reference name",
                        "Ecological site",
                    ],
                    [
                        "1",
                        "Mapped unit",
                        "3",
                        "Other",
                        "40",
                        "unclassified",
                        "Other classification",
                        "Other",
                    ],
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = ecological_sites(SourceContext(BOUNDARY), client)
    assert result.data["records"][0]["comppct_r"] == 60
    assert (
        result.data["records"][0]["descriptionUrl"]
        == "https://edit.sc.egov.usda.gov/catalogs/esd/015X/R015XF008CA"
    )
    assert result.data["records"][1]["descriptionUrl"] is None
    assert (
        "not verified" in result.summary and "not verified" in result.evidence[0][1].relevance_note
    )


def test_empty_associations_and_unsupported_locations_are_distinct() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"Table": []}))
    ) as client:
        result = ecological_sites(SourceContext(BOUNDARY), client)
    assert result.status == "empty" and result.evidence
    foreign = BOUNDARY.model_copy(
        update={"coordinates": [[[10, 50], [11, 50], [11, 51], [10, 50]]]}
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: pytest.fail("No network outside coverage"))
    ) as client:
        assert ecological_sites(SourceContext(foreign), client).status == "uncovered"
        assert regions(SourceContext(foreign), client).status == "uncovered"
    with pytest.raises(ValidationError):
        TaxonQuery(scientific_name="   ")
    with pytest.raises(ValidationError):
        EcologyRequest(
            taxa=[
                TaxonQuery(scientific_name="Quercus alba"),
                TaxonQuery(scientific_name="quercus alba"),
            ]
        )


def test_unresolved_synonym_does_not_invent_an_accepted_usage() -> None:
    import uuid

    from app.research.ecology_outputs import ecology_outputs

    with taxon_client(
        {
            "usage": {"key": "fixture", "name": "Historical synonym", "status": "SYNONYM"},
            "synonym": True,
            "diagnostics": {"matchType": "EXACT"},
        }
    ) as client:
        result = match(TaxonQuery(scientific_name="Quercus alba", kingdom="Plantae"), client)
    _finding, artifacts = ecology_outputs(result, [uuid.uuid4()])
    output = artifacts[0].output
    assert output.kind == "table"
    assert output.rows[0]["matched"] == "Historical synonym"
    assert output.rows[0]["accepted"] is None

"""Research events say where on the map they are looking and what they found."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models.research import ResearchRun
from app.research.locations import (
    focus_location,
    geometry_location,
    media_location,
    provider_location,
    union_location,
)
from app.research.model import (
    ArchiveImageAction,
    ArtifactAction,
    CompleteAction,
    DecisionResult,
    FindingAction,
    MappedAssetSearchAction,
    ResearchDecision,
    RetrieveAction,
    SearchAction,
    TaxonAction,
)
from app.research.providers.base import SourceContext
from app.research.worker import ResearchWorker, action_location
from app.schemas.geojson import Point, Polygon
from app.schemas.land_ecology import TaxonQuery
from app.schemas.research import (
    ArtifactContent,
    EventLocation,
    EventRead,
    FindingContent,
    MapFeature,
    MapOutput,
)
from tests.test_land import BODY, polygon
from tests.test_research import start
from tests.test_research_focus import mapped_source
from tests.test_research_worker import transport

CONTEXT = SourceContext(Polygon.model_validate(polygon()))
LAND_BOX = [-122.14, 47.64, -122.13, 47.65]


def square(west: float, south: float, size: float = 0.001) -> dict[str, Any]:
    east, north = west + size, south + size
    return {
        "type": "Polygon",
        "coordinates": [
            [[west, south], [east, south], [east, north], [west, north], [west, south]]
        ],
    }


# --- Pure helpers -------------------------------------------------------------------------


def test_point_sources_report_their_sample_point() -> None:
    for provider in ("usgs-elevation", "nasa-power", "usda-soils", "usda-ecological-sites"):
        location = provider_location(provider, CONTEXT)
        assert location is not None and location.kind == "point"
        assert location.meaning == "sample-point"
        lon, lat = location.point or (0, 0)
        assert -122.14 <= lon <= -122.13 and 47.64 <= lat <= 47.65


def test_box_sources_report_the_land_box() -> None:
    for provider in ("gbif-occurrences", "fema-flood-zones", "usgs-historical-maps"):
        location = provider_location(provider, CONTEXT)
        assert location is not None and location.kind == "bbox"
        assert location.meaning == "query-bbox"
        assert list(location.bbox or ()) == pytest.approx(LAND_BOX)
    regional = provider_location("epa-ecoregions", CONTEXT)
    assert regional is not None and regional.meaning == "analysis-extent"


def test_photo_search_reports_its_search_circle() -> None:
    location = provider_location("commons-place-images", CONTEXT)
    assert location is not None and location.kind == "point"
    assert location.meaning == "query-radius"
    # The circle reaches every corner of a ~0.8 km square land, and stays within the cap.
    assert location.radius_m is not None and 500 <= location.radius_m <= 10_000
    assert provider_location("unknown-source", CONTEXT) is None


def test_geometry_becomes_a_point_or_its_box() -> None:
    marker = geometry_location({"type": "Point", "coordinates": [-122.1, 47.6]}, "result")
    assert marker is not None and marker.kind == "point" and marker.point == (-122.1, 47.6)
    area = geometry_location(Polygon.model_validate(square(-122.1, 47.6)), "result")
    assert area is not None and area.kind == "bbox"
    assert list(area.bbox or ()) == pytest.approx([-122.1, 47.6, -122.099, 47.601])
    assert geometry_location(None, "result") is None
    assert geometry_location({"type": "Polygon"}, "result") is None


def test_union_boxes_several_shapes_and_keeps_one_point() -> None:
    both = union_location(
        [square(-122.1, 47.6), {"type": "Point", "coordinates": [-122.2, 47.7]}], "result"
    )
    assert both is not None and both.kind == "bbox"
    assert list(both.bbox or ()) == pytest.approx([-122.2, 47.6, -122.099, 47.7])
    single = union_location([{"type": "Point", "coordinates": [1, 2]}], "catalog-coordinate")
    assert single is not None and single.kind == "point" and single.point == (1, 2)
    assert union_location([], "result") is None


def test_focus_and_media_locations_carry_their_reference() -> None:
    artifact_id = uuid.uuid4()
    snapshot = {
        "artifactId": str(artifact_id),
        "featureIndex": 3,
        "feature": {"label": "Pond", "geometry": square(-122.1, 47.6)},
    }
    focus = focus_location(snapshot)
    assert focus is not None and focus.meaning == "focus" and focus.kind == "bbox"
    assert focus.ref is not None and focus.ref.artifact_id == artifact_id
    assert focus.ref.feature_id == "3"
    assert focus_location(None) is None

    evidence_id = uuid.uuid4()
    sheet = media_location(
        {"location": square(-122.1, 47.6), "location_meaning": "catalog-footprint"}, evidence_id
    )
    assert sheet is not None and sheet.meaning == "catalog-footprint"
    assert sheet.ref is not None and sheet.ref.evidence_id == evidence_id
    photo = media_location(
        {"location": {"type": "Point", "coordinates": [-122.1, 47.6]}, "location_meaning": "x"}
    )
    assert photo is not None and photo.meaning == "catalog-coordinate" and photo.kind == "point"
    assert media_location({"location": None}) is None


def test_location_shape_is_validated() -> None:
    with pytest.raises(ValidationError):
        EventLocation(kind="point", meaning="result")
    with pytest.raises(ValidationError):
        EventLocation(kind="bbox", bbox=(10, 0, 0, 1), meaning="result")
    with pytest.raises(ValidationError):
        EventLocation(kind="point", point=(200, 0), meaning="result")
    with pytest.raises(ValidationError):
        EventLocation(kind="geometry-ref", ref={}, meaning="result")
    wire = EventLocation(kind="point", point=(1, 2), radius_m=50, meaning="query-radius")
    assert wire.model_dump(mode="json", by_alias=True, exclude_none=True) == {
        "kind": "point",
        "point": [1.0, 2.0],
        "radiusM": 50.0,
        "meaning": "query-radius",
    }


def test_model_actions_say_where_they_will_look() -> None:
    retrieve = action_location(
        RetrieveAction(kind="retrieve_source", provider="usgs-elevation"), CONTEXT
    )
    assert retrieve is not None and retrieve.meaning == "sample-point"
    nearby = action_location(
        MappedAssetSearchAction(
            kind="search_mapped_assets",
            source_kind="building",
            point=Point(coordinates=[-122.135, 47.645]),
            radius_m=300,
        ),
        CONTEXT,
    )
    assert nearby is not None and nearby.point == (-122.135, 47.645) and nearby.radius_m == 300
    search = action_location(SearchAction(kind="search_public_sources", query="wells"), CONTEXT)
    assert search is not None and search.meaning == "query-bbox"
    image = action_location(
        ArchiveImageAction(kind="inspect_archive_image", evidence_id=uuid.uuid4()), CONTEXT
    )
    assert image is not None and image.kind == "geometry-ref"
    evidence_id = uuid.uuid4()
    finding = action_location(
        FindingAction(
            kind="publish_finding",
            finding=FindingContent(
                title="Wet corner",
                summary="Water collects here.",
                category="physical",
                evidence_ids=[evidence_id],
                confidence="inferred",
                uncertainty="From terrain only.",
                boundary=Polygon.model_validate(square(-122.139, 47.641)),
            ),
        ),
        CONTEXT,
    )
    assert finding is not None and finding.meaning == "result" and finding.kind == "bbox"
    mapped = action_location(
        ArtifactAction(
            kind="publish_artifact",
            artifact=ArtifactContent(
                title="Wells",
                method="Test",
                evidence_ids=[evidence_id],
                output=MapOutput(
                    kind="map",
                    legend="Wells",
                    features=[
                        MapFeature(label="Well", geometry=Point(coordinates=[-122.135, 47.645]))
                    ],
                ),
            ),
        ),
        CONTEXT,
    )
    assert mapped is not None and mapped.point == (-122.135, 47.645)
    taxon = TaxonAction(kind="match_taxon", query=TaxonQuery(scientific_name="Quercus alba"))
    assert action_location(taxon, CONTEXT) is None


# --- Through the worker and the API --------------------------------------------------------


def compact(value: Any) -> Any:
    """The wire form without the empty fields FastAPI writes as nulls."""
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items() if item is not None}
    return value


def events(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    response = client.get(f"/api/v1/research/runs/{run_id}/events")
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


def test_overview_points_at_each_source_before_and_after_it_answers(
    client: TestClient, sessions: sessionmaker[Session]
) -> None:
    _, _inv, run, _ = start(client)
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        assert ResearchWorker(sessions, Settings(_env_file=None), client=http).run_once()
    items = events(client, run["id"])
    looking = [item for item in items if item["kind"] == "looking"]
    sources = [item for item in items if item["kind"] == "source"]
    assert [item["payload"]["provider"] for item in looking] == [
        item["payload"]["provider"] for item in sources
    ]
    assert len(looking) == 5
    for before, after in zip(looking, sources, strict=True):
        assert before["sequence"] < after["sequence"]
        assert before["location"] == after["location"]
        assert before["location"]["kind"] in {"point", "bbox"}
    elevation = next(item for item in looking if item["payload"]["provider"] == "usgs-elevation")
    assert elevation["location"]["meaning"] == "sample-point"
    flood = next(item for item in looking if item["payload"]["provider"] == "fema-flood-zones")
    assert flood["location"]["bbox"] == pytest.approx(LAND_BOX)
    # A finding with no boundary of its own is pinned where its source looked.
    findings = [item for item in items if item["kind"] == "finding"]
    assert findings and all(item["location"] for item in findings)
    assert findings[0]["location"]["meaning"] == "sample-point"
    # Charts from one sample point carry it too; the plain progress steps stay unplaced.
    charts = [item for item in items if item["kind"] == "artifact"]
    assert charts and charts[0]["location"]["meaning"] == "sample-point"
    assert all(item["location"] is None for item in items if item["kind"] == "progress")


def test_stream_frames_carry_the_location(
    client: TestClient, sessions: sessionmaker[Session]
) -> None:
    _, _inv, run, _ = start(client)
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        assert ResearchWorker(sessions, Settings(_env_file=None), client=http).run_once()
    body = client.get(f"/api/v1/research/runs/{run['id']}/stream").text
    frames = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: {}"
    ]
    located = [frame for frame in frames if frame["kind"] == "looking"]
    assert located and all(frame["location"]["kind"] in {"point", "bbox"} for frame in located)
    for frame in frames:
        EventRead.model_validate(frame)


class LocatingModel:
    """Retrieves one point source, then publishes a bounded finding and a map."""

    __test__ = False

    def decide(self, context: str, max_tokens: int) -> DecisionResult:
        state = json.loads(context)
        done = [item["action"] for item in state["previousActions"]]
        action: RetrieveAction | FindingAction | ArtifactAction | CompleteAction
        if not state["retrieved"]:
            action = RetrieveAction(kind="retrieve_source", provider="usgs-elevation")
        else:
            ids = [
                uuid.UUID(value) for value in state["retrieved"]["usgs-elevation"]["evidenceIds"]
            ]
            if "publish_finding" not in done:
                action = FindingAction(
                    kind="publish_finding",
                    finding=FindingContent(
                        title="Low corner",
                        summary="The southwest corner sits lowest.",
                        category="physical",
                        evidence_ids=ids,
                        confidence="inferred",
                        uncertainty="One sample point.",
                        boundary=Polygon.model_validate(square(-122.139, 47.641)),
                    ),
                )
            elif "publish_artifact" not in done:
                action = ArtifactAction(
                    kind="publish_artifact",
                    artifact=ArtifactContent(
                        title="Low ground",
                        method="Test",
                        evidence_ids=ids,
                        output=MapOutput(
                            kind="map",
                            legend="Low ground",
                            features=[
                                MapFeature(
                                    label="A",
                                    geometry=Point(coordinates=[-122.138, 47.642]),
                                ),
                                MapFeature(
                                    label="B",
                                    geometry=Point(coordinates=[-122.136, 47.644]),
                                ),
                            ],
                        ),
                    ),
                )
            else:
                action = CompleteAction(kind="complete", summary="Done.")
        return DecisionResult(ResearchDecision(progress="Working on it", action=action), 120)


def test_agent_steps_and_results_are_placed(
    client: TestClient, db: Session, sessions: sessionmaker[Session]
) -> None:
    _, _inv, run, _ = start(client)
    row = db.scalar(select(ResearchRun).where(ResearchRun.id == run["id"]))
    assert row is not None
    row.kind = "investigation"
    db.commit()
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        assert ResearchWorker(
            sessions, Settings(_env_file=None), model=LocatingModel(), client=http
        ).run_once()
    items = events(client, run["id"])
    steps = [item for item in items if item["kind"] == "progress" and item["payload"]]
    assert steps[0]["payload"]["action"] == "retrieve_source"
    assert steps[0]["location"]["meaning"] == "sample-point"
    finding = next(
        item
        for item in items
        if item["kind"] == "finding" and item["payload"]["title"] == "Low corner"
    )
    assert finding["location"]["meaning"] == "result"
    assert finding["location"]["bbox"] == pytest.approx([-122.139, 47.641, -122.138, 47.642])
    artifact = next(item for item in items if item["kind"] == "artifact")
    assert artifact["location"]["bbox"] == pytest.approx([-122.138, 47.642, -122.136, 47.644])
    assert artifact["location"]["ref"]["artifactId"] == artifact["payload"]["id"]
    results = [item for item in items if item["kind"] == "tool-result"]
    assert results[0]["location"]["meaning"] == "sample-point"


def test_a_focused_question_starts_at_its_feature(client: TestClient, db: Session) -> None:
    land, _, artifact_id, _ = mapped_source(client, db)
    other = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Follow-up", "boundaryRevision": 1},
    ).json()
    run = client.post(
        f"/api/v1/research/investigations/{other['id']}/runs",
        json={
            "requestKey": str(uuid.uuid4()),
            "question": "What about this?",
            "focus": {"artifactId": artifact_id, "featureIndex": 0},
        },
    ).json()
    queued = events(client, run["id"])[0]
    assert queued["kind"] == "queued"
    assert compact(queued["location"]) == {
        "kind": "point",
        "point": [-77.05, 38.89],
        "ref": {"artifactId": artifact_id, "featureId": "0"},
        "meaning": "focus",
    }


def test_unfocused_runs_start_unplaced(client: TestClient) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    inv = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Plain", "boundaryRevision": 1},
    ).json()
    run = client.post(
        f"/api/v1/research/investigations/{inv['id']}/runs",
        json={"requestKey": str(uuid.uuid4()), "kind": "overview", "question": "What?"},
    ).json()
    assert events(client, run["id"])[0]["location"] is None

"""Oak Bench Ranch as a saved Land (`python -m app.seed.demo_land`)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

import app.db
from app.models import (
    Evidence,
    FeatureInspection,
    Finding,
    LandAction,
    LandArea,
    LandBoundaryRevision,
    LandDocument,
    LandFeature,
    LandFeatureRevision,
    LandRaster,
    LandSurvey,
    ResearchArtifact,
    ResearchRun,
)
from app.models.workspace import PILOT_WORKSPACE_ID
from app.seed.demo_land import NAMESPACE, NOTE, feature_id, land_id, main, seed_demo_land
from app.seed.demo_ranch import load

FEATURES = {
    "fence": ("Perimeter fence", "other", "LineString"),
    "creek": ("Figueroa Creek", "water", "LineString"),
    "oakWoodland": ("Oak woodland", "vegetation", "Polygon"),
}


@pytest.fixture(scope="module")
def ranch() -> dict[str, Any]:
    return load()


def _counts(db: Session) -> dict[str, int]:
    return {
        model.__name__: db.scalar(select(func.count()).select_from(model)) or 0
        for model in (LandArea, LandBoundaryRevision, LandFeature, LandFeatureRevision)
    }


def test_the_seed_saves_the_ranch_as_a_land_with_its_three_features(
    client: TestClient, db: Session, ranch: dict[str, Any]
) -> None:
    seeded = seed_demo_land(db)
    assert seeded.land == "created"
    assert seeded.features == dict.fromkeys(FEATURES, "created")
    assert seeded.land_id == land_id(PILOT_WORKSPACE_ID, "demo-ranch")

    # Listed where the Land panel looks: the pilot workspace's saved land.
    listed = client.get("/api/v1/land").json()
    assert [land["name"] for land in listed] == ["Oak Bench Ranch"]
    land = listed[0]
    assert land["id"] == str(seeded.land_id)
    assert land["revision"] == 1
    # The outline as the file has it: PostGIS measures the same geodesic area.
    assert land["areaM2"] == pytest.approx(ranch["derived"]["siteAreaM2"], abs=1.0)
    assert land["source"]["meaning"] == "study-area"
    assert land["source"]["recordId"] == "demo-ranch"
    assert "not a property line" in land["description"]
    assert "simulated" in land["description"]

    features = client.get(f"/api/v1/land/{seeded.land_id}/features").json()
    by_key = {feature["externalRef"]["recordId"]: feature for feature in features}
    assert sorted(by_key) == sorted(FEATURES)
    for key, (name, category, geometry_type) in FEATURES.items():
        feature = by_key[key]
        assert feature["id"] == str(feature_id(seeded.land_id, key))
        assert (feature["name"], feature["category"]) == (name, category)
        assert feature["geometry"]["type"] == geometry_type
        assert feature["externalRef"]["namespace"] == NAMESPACE
        # Nobody has checked them on the ground.
        assert feature["status"] == "candidate"
        assert feature["intersectsLand"] is True and feature["distanceM"] == 0


def test_every_feature_names_where_it_came_from(
    client: TestClient, db: Session, ranch: dict[str, Any]
) -> None:
    seeded = seed_demo_land(db)
    features = client.get(f"/api/v1/land/{seeded.land_id}/features").json()
    by_key = {feature["externalRef"]["recordId"]: feature for feature in features}
    sources = ranch["sources"]

    fence = by_key["fence"]
    assert fence["source"]["label"] == "Demo outline fence, simulated"
    assert fence["attributes"] == {"simulated": True}
    assert "Simulated" in fence["description"]

    creek = by_key["creek"]
    assert creek["source"]["method"] == "mapped-feature"
    assert creek["source"]["url"] == sources["nhd"]["url"]
    assert sources["nhd"]["text"] in creek["source"]["attribution"]
    assert "Public domain" in creek["source"]["attribution"]

    oak = by_key["oakWoodland"]
    assert oak["source"]["method"] == "imagery"
    assert oak["source"]["url"] == sources["worldcover"]["url"]
    # CC BY 4.0: the credit travels with the feature.
    assert sources["worldcover"]["text"] in oak["source"]["attribution"]
    assert "CC BY 4.0" in oak["source"]["attribution"]
    assert oak["attributes"] == {"species": "oak (assumed)"}
    assert "assumed" in oak["description"]

    # The mission ground says the same, on every zone it draws.
    ground = client.get(f"/api/v1/land/{seeded.land_id}/ground").json()
    assert ground["skipped"] == 0
    notes = {item["name"]: item["source"] for item in ground["items"]}
    assert set(notes) == {"Oak Bench Ranch", "Perimeter fence", "Figueroa Creek", "Oak woodland"}
    assert "Demo outline fence, simulated" in notes["Perimeter fence"]
    assert "study area, not a verified property boundary" in notes["Oak Bench Ranch"]


def test_every_feature_lies_inside_the_outline(db: Session, ranch: dict[str, Any]) -> None:
    seeded = seed_demo_land(db)
    to_m = Transformer.from_crs(4326, 32610, always_xy=True).transform
    boundary = transform(to_m, shape(ranch["boundary"]["geometry"]))
    rows = db.scalars(select(LandFeature).where(LandFeature.land_id == seeded.land_id)).all()
    assert len(rows) == 3
    for row in rows:
        geometry = transform(to_m, shape(row.content["geometry"]))
        assert geometry.is_valid, row.content["name"]
        # Clipped to the outline; rounding to 1 cm may move a vertex on the edge by that much.
        outside = geometry.difference(boundary.buffer(0.05))
        assert outside.is_empty, (row.content["name"], outside.length, outside.area)
    creek = next(row for row in rows if row.content["name"] == "Figueroa Creek")
    length = transform(to_m, shape(creek.content["geometry"])).length
    # UTM length within 0.1% of the geodesic length the ranch build measured inside the outline.
    assert length == pytest.approx(ranch["derived"]["creekLengthM"], rel=1e-3)


def test_a_rerun_changes_nothing(db: Session) -> None:
    first = seed_demo_land(db)
    before = _counts(db)
    again = seed_demo_land(db)
    assert again.land == "unchanged"
    assert again.features == dict.fromkeys(FEATURES, "unchanged")
    assert (again.land_id, again.revision) == (first.land_id, first.revision)
    assert (
        _counts(db)
        == before
        == {
            "LandArea": 1,
            "LandBoundaryRevision": 1,
            "LandFeature": 3,
            "LandFeatureRevision": 3,
        }
    )


def test_nothing_observed_is_made_up(client: TestClient, db: Session) -> None:
    seeded = seed_demo_land(db, queue_research=True)
    for model in (
        FeatureInspection,
        Evidence,
        Finding,
        ResearchArtifact,
        LandRaster,
        LandSurvey,
        LandDocument,
        LandAction,
    ):
        assert db.scalar(select(func.count()).select_from(model)) == 0, model.__name__
    features = client.get(f"/api/v1/land/{seeded.land_id}/features").json()
    assert all(feature["evidenceIds"] == [] for feature in features)
    for feature in features:
        inspections = client.get(
            f"/api/v1/land/{seeded.land_id}/features/{feature['id']}/inspections"
        ).json()
        assert inspections == []


def test_research_queues_the_real_overview_once(db: Session) -> None:
    assert seed_demo_land(db).overview is None
    assert db.scalar(select(func.count()).select_from(ResearchRun)) == 0

    seeded = seed_demo_land(db, queue_research=True)
    assert seeded.overview is not None
    assert (seeded.overview.kind, seeded.overview.status) == ("overview", "queued")
    # No worker has checked in on this database, so nothing will run it until one starts.
    assert seeded.worker == "never"
    again = seed_demo_land(db, queue_research=True)
    assert again.overview is not None and again.overview.id == seeded.overview.id
    assert db.scalar(select(func.count()).select_from(ResearchRun)) == 1


def test_a_rerun_puts_the_file_back_and_keeps_what_the_user_decided(
    client: TestClient, db: Session, ranch: dict[str, Any]
) -> None:
    seeded = seed_demo_land(db)
    land = client.get(f"/api/v1/land/{seeded.land_id}").json()
    # A user renames the Land and drags its outline smaller...
    smaller = {
        "type": "Polygon",
        "coordinates": [
            [
                [-120.05, 34.69],
                [-120.03, 34.69],
                [-120.03, 34.705],
                [-120.05, 34.705],
                [-120.05, 34.69],
            ]
        ],
    }
    response = client.put(
        f"/api/v1/land/{seeded.land_id}",
        json={
            "name": "Our ranch",
            "description": land["description"],
            "boundary": smaller,
            "source": {"method": "drawn", "label": "Drawn on map"},
            "expectedRevision": 1,
            "note": "Tried a smaller outline",
        },
    )
    assert response.status_code == 200, response.text
    # ...confirms the fence, renames it and moves one end.
    fence_id = feature_id(seeded.land_id, "fence")
    path = f"/api/v1/land/{seeded.land_id}/features/{fence_id}"
    fence = client.get(path).json()
    moved = deepcopy(fence["geometry"])
    moved["coordinates"][0] = [-120.027, 34.698]
    edit = {
        key: fence[key]
        for key in (
            "name",
            "category",
            "source",
            "description",
            "attributes",
            "evidenceIds",
            "externalRef",
        )
    }
    edit.update(
        requestKey="6b0f8f5e-0000-4000-8000-000000000001",
        name="Boundary fence",
        status="confirmed",
        geometry=moved,
        expectedRevision=1,
        note="Walked it",
    )
    response = client.put(path, json=edit)
    assert response.status_code == 200, response.text

    again = seed_demo_land(db)
    assert again.land == "updated"
    assert again.features == {"fence": "updated", "creek": "unchanged", "oakWoodland": "unchanged"}
    land = client.get(f"/api/v1/land/{seeded.land_id}").json()
    assert land["revision"] == 3
    assert land["name"] == "Our ranch"
    assert land["source"]["recordId"] == "demo-ranch"
    assert land["areaM2"] == pytest.approx(ranch["derived"]["siteAreaM2"], abs=1.0)
    history = client.get(f"/api/v1/land/{seeded.land_id}/revisions").json()
    assert [revision["note"] for revision in history] == [
        NOTE,
        "Tried a smaller outline",
        "Area established",
    ]
    fence = client.get(path).json()
    assert (fence["revision"], fence["name"], fence["status"]) == (3, "Boundary fence", "confirmed")
    assert (
        fence["geometry"]["coordinates"][0]
        == ranch["features"]["fence"]["geometry"]["coordinates"][0]
    )
    assert seed_demo_land(db).features == dict.fromkeys(FEATURES, "unchanged")


def test_a_changed_file_brings_the_land_along(db: Session, ranch: dict[str, Any]) -> None:
    seed_demo_land(db)
    changed = deepcopy(ranch)
    changed["description"] = "A demo outline, moved."
    changed["features"]["fence"]["geometry"]["coordinates"][-1] = [-120.0533, 34.697]
    seeded = seed_demo_land(db, document=changed)
    assert seeded.land == "updated" and seeded.revision == 2
    assert seeded.features["fence"] == "updated"
    land = db.get(LandArea, seeded.land_id)
    assert land is not None and land.description.startswith("A demo outline, moved.")


def test_the_land_is_only_in_its_workspace(db: Session, client: TestClient) -> None:
    pilot = seed_demo_land(db).land_id
    # A second workspace gets its own Land, never the pilot's.
    from app.models import Workspace

    workspace = Workspace(name="Another workspace")
    db.add(workspace)
    db.commit()
    elsewhere = seed_demo_land(db, workspace_id=workspace.id)
    assert elsewhere.land == "created" and elsewhere.land_id != pilot
    assert [land["id"] for land in client.get("/api/v1/land").json()] == [str(pilot)]


def test_the_command_says_what_it_did(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ranch: dict[str, Any],
) -> None:
    monkeypatch.setattr(app.db, "get_session_factory", lambda: sessionmaker(bind=db.get_bind()))
    main([])
    out = capsys.readouterr().out
    assert "Oak Bench Ranch: created" in out
    assert "fence created, creek created, oakWoodland created" in out
    assert f"{ranch['derived']['siteAreaM2'] / 4046.8564224:,.0f} ac" in out
    main(["--research"])
    out = capsys.readouterr().out
    assert "Oak Bench Ranch: unchanged" in out
    assert "overview: queued" in out
    assert "pnpm dev:research" in out

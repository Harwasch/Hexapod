from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.seed import seed
from app.seed.captures import capture_sites


def test_seed_is_idempotent_and_lists_layers(client: TestClient, db: Session) -> None:
    first = seed(db)
    second = seed(db)
    # demo + 5 comparison sites + one per capture in data/tiles/captures.json
    captures = len(capture_sites(get_settings()))
    assert first["layers"] == 10 and first["sites"] == 6 + captures
    assert second == {"sites": 0, "layers": 0}
    layers = client.get("/api/v1/layers").json()
    assert len(layers) == 10
    terrain = next(layer for layer in layers if layer["slug"] == "cesium-world-terrain")
    assert terrain["source"] == {
        "type": "cesium-ion-terrain",
        "assetId": 1,
        "requestVertexNormals": True,
        "requestWaterMask": False,
    }
    assert terrain["builtin"] is True
    assert terrain["attribution"][0]["text"].startswith("Cesium World Terrain")
    assert terrain["spatialExtent"] == {"west": -180, "south": -90, "east": 180, "north": 90}
    sites = client.get("/api/v1/sites").json()
    assert sites[0]["slug"] == "cesium-splat-demo"
    assert sites[0]["representations"] == ["gaussian-splat"]


def test_create_xyz_layer_and_update(client: TestClient) -> None:
    payload = {
        "name": "My tiles",
        "category": "my-data",
        "source": {
            "type": "xyz",
            "urlTemplate": "https://tiles.example.com/{z}/{x}/{y}.png",
            "maximumLevel": 18,
        },
        "spatialExtent": {"west": -10, "south": 40, "east": 10, "north": 50},
        "temporalExtent": {"start": "2024-01-01T00:00:00Z", "end": "2024-12-31T00:00:00Z"},
        "attribution": [{"text": "© Me"}],
        "license": {"name": "CC0", "requiresAttribution": False},
        "render": {"opacity": 0.5},
    }
    created = client.post("/api/v1/layers", json=payload)
    assert created.status_code == 201, created.text
    layer = created.json()
    assert layer["slug"] == "my-tiles"
    assert layer["sourceType"] == "xyz"
    assert layer["render"]["opacity"] == 0.5
    updated = client.patch(
        f"/api/v1/layers/{layer['id']}", json={"render": {"opacity": 0.2}, "name": "Renamed"}
    )
    assert updated.json()["render"]["opacity"] == 0.2
    assert updated.json()["name"] == "Renamed"
    assert client.delete(f"/api/v1/layers/{layer['id']}").status_code == 204


def test_builtin_layers_cannot_be_deleted(client: TestClient, db: Session) -> None:
    seed(db)
    layer = client.get("/api/v1/layers").json()[0]
    assert client.delete(f"/api/v1/layers/{layer['id']}").status_code == 409


def test_layer_source_validation(client: TestClient) -> None:
    base = {"name": "Bad", "category": "my-data"}
    missing_xyz = {
        **base,
        "source": {"type": "xyz", "urlTemplate": "https://tiles.example.com/{z}/{x}.png"},
    }
    assert client.post("/api/v1/layers", json=missing_xyz).status_code == 422
    bad_scheme = {**base, "source": {"type": "geojson", "url": "javascript:alert(1)"}}
    assert client.post("/api/v1/layers", json=bad_scheme).status_code == 422
    unknown_type = {**base, "source": {"type": "carrier-pigeon"}}
    assert client.post("/api/v1/layers", json=unknown_type).status_code == 422
    bad_extent = {
        **base,
        "source": {"type": "czml", "url": "https://example.com/a.czml"},
        "spatialExtent": {"west": 0, "south": 10, "east": 1, "north": 0},
    }
    assert client.post("/api/v1/layers", json=bad_extent).status_code == 422
    bad_temporal = {
        **base,
        "source": {"type": "czml", "url": "https://example.com/a.czml"},
        "temporalExtent": {"start": "2025-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
    }
    assert client.post("/api/v1/layers", json=bad_temporal).status_code == 422


def test_stac_and_wms_sources_round_trip(client: TestClient) -> None:
    stac = {
        "name": "STAC item",
        "category": "imagery",
        "source": {
            "type": "stac",
            "url": "https://stac.example.com/items/abc",
            "kind": "item",
            "assetKey": "visual",
        },
    }
    assert client.post("/api/v1/layers", json=stac).json()["source"]["assetKey"] == "visual"
    wms = {
        "name": "WMS",
        "category": "land-cover",
        "source": {
            "type": "wms",
            "url": "https://wms.example.com/wms",
            "layers": "cover",
            "parameters": {"transparent": "true"},
        },
    }
    assert client.post("/api/v1/layers", json=wms).json()["source"]["parameters"] == {
        "transparent": "true"
    }


def test_seed_refreshes_bookmarks_of_seeded_sites(client: TestClient, db: Session) -> None:
    seed(db)
    demo = next(s for s in client.get("/api/v1/sites").json() if s["slug"] == "cesium-splat-demo")
    detail = client.get(f"/api/v1/sites/{demo['id']}").json()
    original = detail["cameraBookmarks"][0]
    # Simulate an older seed by tilting the stored bookmark towards the horizon.
    from app.models import CameraBookmark

    stored = db.get(CameraBookmark, original["id"])
    assert stored is not None
    stored.pitch = -25.0
    db.commit()
    assert seed(db) == {"sites": 0, "layers": 0}
    refreshed = client.get(f"/api/v1/sites/{demo['id']}").json()["cameraBookmarks"]
    assert len(refreshed) == 1
    assert refreshed[0]["pitch"] == original["pitch"] == -45.0


def test_seed_refreshes_boundary_and_footprints_of_seeded_sites(
    client: TestClient, db: Session
) -> None:
    seed(db)
    demo = next(s for s in client.get("/api/v1/sites").json() if s["slug"] == "cesium-splat-demo")
    wanted = client.get(f"/api/v1/sites/{demo['id']}").json()
    small = {
        "type": "Polygon",
        "coordinates": [[[-122.14, 47.64], [-122.13, 47.64], [-122.13, 47.65], [-122.14, 47.64]]],
    }
    assert client.patch(f"/api/v1/sites/{demo['id']}", json={"boundary": small}).status_code == 200
    from app.models import Asset

    for asset in db.scalars(select(Asset).where(Asset.site_id == demo["id"])):
        asset.footprint = None
    db.commit()
    assert seed(db) == {"sites": 0, "layers": 0}
    refreshed = client.get(f"/api/v1/sites/{demo['id']}").json()
    assert refreshed["boundary"] == wanted["boundary"]
    assert [a["footprint"] for a in refreshed["assets"]] == [
        a["footprint"] for a in wanted["assets"]
    ]


def test_seed_refreshes_render_config_of_seeded_assets(client: TestClient, db: Session) -> None:
    seed(db)
    sf = next(
        s for s in client.get("/api/v1/sites").json() if s["slug"].startswith("san-francisco")
    )
    from app.models import Asset

    asset = db.scalars(select(Asset).where(Asset.site_id == sf["id"])).first()
    assert asset is not None
    assert asset.render_config["screenSpaceErrorScale"] == 0.25
    asset.render_config = {**asset.render_config, "screenSpaceErrorScale": 1.0}
    db.commit()
    assert seed(db) == {"sites": 0, "layers": 0}
    detail = client.get(f"/api/v1/sites/{sf['id']}").json()
    assert detail["assets"][0]["renderConfig"]["screenSpaceErrorScale"] == 0.25

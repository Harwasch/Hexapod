from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.seed import seed


def test_seed_is_idempotent_and_lists_layers(client: TestClient, db: Session) -> None:
    first = seed(db)
    second = seed(db)
    assert first["layers"] == 10 and first["sites"] == 4  # demo + 3 comparison sites
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

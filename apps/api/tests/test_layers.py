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
    # the demo site + one per capture (the archive, and any site.json under data/tiles)
    captures = len(capture_sites(get_settings()))
    assert first["layers"] == 9 and first["sites"] == 1 + captures
    assert second == {"sites": 0, "layers": 0}
    layers = client.get("/api/v1/layers").json()
    assert len(layers) == 9
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
    demo = next(site for site in sites if site["slug"] == "cesium-splat-demo")
    # The place the simulated fleet runs in, with no reality model of its own.
    assert demo["representations"] == []


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
    tree = next(s for s in client.get("/api/v1/sites").json() if s["slug"] == "synthetic-tree")
    wanted = client.get(f"/api/v1/sites/{tree['id']}").json()
    assert wanted["assets"]
    small = {
        "type": "Polygon",
        "coordinates": [[[-122.14, 47.64], [-122.13, 47.64], [-122.13, 47.65], [-122.14, 47.64]]],
    }
    assert client.patch(f"/api/v1/sites/{tree['id']}", json={"boundary": small}).status_code == 200
    from app.models import Asset

    for asset in db.scalars(select(Asset).where(Asset.site_id == tree["id"])):
        asset.footprint = None
    db.commit()
    assert seed(db) == {"sites": 0, "layers": 0}
    refreshed = client.get(f"/api/v1/sites/{tree['id']}").json()
    assert refreshed["boundary"] == wanted["boundary"]
    assert [a["footprint"] for a in refreshed["assets"]] == [
        a["footprint"] for a in wanted["assets"]
    ]


def test_seed_refreshes_render_config_of_seeded_assets(client: TestClient, db: Session) -> None:
    seed(db)
    tree = next(s for s in client.get("/api/v1/sites").json() if s["slug"] == "synthetic-tree")
    from app.models import Asset

    asset = db.scalars(select(Asset).where(Asset.site_id == tree["id"])).first()
    assert asset is not None
    wanted = asset.render_config["screenSpaceErrorScale"]
    asset.render_config = {**asset.render_config, "screenSpaceErrorScale": wanted * 4}
    db.commit()
    assert seed(db) == {"sites": 0, "layers": 0}
    detail = client.get(f"/api/v1/sites/{tree['id']}").json()
    assert detail["assets"][0]["renderConfig"]["screenSpaceErrorScale"] == wanted


def test_seed_removes_what_was_withdrawn_from_an_older_database(
    client: TestClient, db: Session
) -> None:
    """A database seeded before 2026-10 still has the sites, the splat and the layer this
    deployment cannot serve; the next seed deletes them, and only them."""
    from app.models import Layer
    from app.schemas.asset import AssetBase, CesiumIonSource
    from app.schemas.layer import CesiumIonImagerySource, LayerCreate
    from app.seed.data import DEMO_SITE
    from app.services import layers as layer_service
    from app.services import sites as site_service

    old_demo = DEMO_SITE.model_copy(
        update={
            "description": "Public 3D Gaussian splat tileset",
            "assets": [
                AssetBase(
                    name="Gaussian splat (LOD)",
                    representation="gaussian-splat",
                    source=CesiumIonSource(asset_id=4547222),
                ),
                AssetBase(
                    name="Someone's mesh",
                    representation="mesh",
                    source=CesiumIonSource(asset_id=96188),
                ),
            ],
        }
    )
    # Held, as a long-lived session holds what it loaded: the seed must leave the site's own
    # list of assets right, not only the rows. Unheld, whether the session still had the
    # site at the next read was up to the garbage collector, and so was this test.
    held = site_service.create_site(db, old_demo)
    assert len(held.assets) == 2
    site_service.create_site(db, DEMO_SITE.model_copy(update={"slug": "mygla", "name": "Mygla"}))
    sentinel = LayerCreate(
        slug="sentinel-2",
        name="Sentinel-2 cloudless",
        category="imagery",
        source=CesiumIonImagerySource(asset_id=3954),
    )
    layer_service.create_layer(db, sentinel, builtin=True)

    seed(db)
    slugs = {site["slug"] for site in client.get("/api/v1/sites").json()}
    assert "mygla" not in slugs and "cesium-splat-demo" in slugs
    demo = client.get("/api/v1/sites/by-slug/cesium-splat-demo").json()
    # The splat goes; an asset someone added to the site stays.
    assert [asset["name"] for asset in demo["assets"]] == ["Someone's mesh"]
    assert demo["description"] == DEMO_SITE.description
    assert demo["license"] is None and demo["attribution"] == []
    assert db.scalar(select(Layer).where(Layer.slug == "sentinel-2")) is None
    assert seed(db) == {"sites": 0, "layers": 0}

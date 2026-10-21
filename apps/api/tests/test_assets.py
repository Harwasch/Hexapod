from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from tests.conftest import SQUARE, site_payload


def test_asset_crud_and_site_relationship(client: TestClient) -> None:
    site = client.post("/api/v1/sites", json=site_payload(assets=[])).json()
    payload = {
        "siteId": site["id"],
        "name": "Mesh",
        "representation": "mesh",
        "source": {"type": "3d-tiles-url", "url": "https://example.com/tiles/tileset.json"},
        "footprint": SQUARE,
        "observedAt": "2025-05-01T12:00:00Z",
        "validFrom": "2025-05-01T00:00:00Z",
        "validTo": "2026-05-01T00:00:00Z",
        "resolution": {"groundSampleDistanceM": 0.01, "description": "1 cm GSD"},
        "crs": {"horizontal": "EPSG:4978"},
        "attribution": [{"text": "Captured by drone"}],
        "renderConfig": {"maximumScreenSpaceError": 8, "clipsWorld": True},
        "provenance": {"sourceOrganization": "Acme", "sourceUrl": "https://example.com"},
    }
    created = client.post("/api/v1/assets", json=payload)
    assert created.status_code == 201, created.text
    asset = created.json()
    assert asset["provider"] == "3d-tiles-url"
    assert asset["footprint"]["type"] == "MultiPolygon"
    assert asset["renderConfig"]["maximumScreenSpaceError"] == 8
    assert asset["provenance"]["sourceOrganization"] == "Acme"

    assert client.get(f"/api/v1/sites/{site['id']}/assets").json()[0]["id"] == asset["id"]
    assert (
        client.get("/api/v1/assets", params={"siteId": site["id"]}).json()[0]["id"] == asset["id"]
    )

    updated = client.patch(
        f"/api/v1/assets/{asset['id']}",
        json={"name": "Mesh v2", "renderConfig": {"maximumScreenSpaceError": 4}},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "Mesh v2"
    assert updated.json()["renderConfig"]["maximumScreenSpaceError"] == 4
    assert updated.json()["provenance"]["sourceOrganization"] == "Acme"

    assert client.delete(f"/api/v1/assets/{asset['id']}").status_code == 204
    assert client.get(f"/api/v1/assets/{asset['id']}").status_code == 404


def test_asset_rejects_unknown_site(client: TestClient) -> None:
    payload = {
        "siteId": "00000000-0000-0000-0000-000000000000",
        "name": "Mesh",
        "representation": "mesh",
        "source": {"type": "cesium-ion", "assetId": 1},
    }
    assert client.post("/api/v1/assets", json=payload).status_code == 404


def test_asset_validation_errors(client: TestClient) -> None:
    base = {"name": "x", "representation": "mesh", "source": {"type": "cesium-ion", "assetId": 1}}
    assert (
        client.post("/api/v1/assets", json={**base, "representation": "hologram"}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/assets", json={**base, "source": {"type": "cesium-ion", "assetId": -1}}
        ).status_code
        == 422
    )
    bad_window = {**base, "validFrom": "2026-01-01T00:00:00Z", "validTo": "2025-01-01T00:00:00Z"}
    assert client.post("/api/v1/assets", json=bad_window).status_code == 422
    assert client.post("/api/v1/assets", json={**base, "unknownField": 1}).status_code == 422


# --- epoch assets (decision D3) ------------------------------------------------------------


def test_an_epoch_assets_dates_and_footprint_belong_to_its_epoch(
    client: TestClient, db: Session
) -> None:
    """Which day an epoch shows, and where, is the epoch's: an edit to its asset's dates or
    footprint would make the two disagree. What the asset is called and whether it shows
    by default stay editable."""
    from tests.test_epochs import T0, make_asset, make_epoch, make_site

    site = make_site(db)
    epoch = make_epoch(db, site, 1, observed=T0)
    asset = make_asset(db, site, epoch, "https://cdn.example.com/a/tileset.json", created=T0)
    db.commit()

    for body in (
        {"observedAt": "2026-01-01T00:00:00Z"},
        {"validFrom": "2026-01-01T00:00:00Z"},
        {"validTo": "2027-01-01T00:00:00Z"},
        {"footprint": SQUARE},
    ):
        refused = client.patch(f"/api/v1/assets/{asset.id}", json=body)
        assert refused.status_code == 409, (body, refused.text)
        assert refused.json()["code"] == "epoch_managed"
    renamed = client.patch(
        f"/api/v1/assets/{asset.id}", json={"name": "September", "defaultVisible": False}
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "September" and renamed.json()["defaultVisible"] is False


def test_a_superseded_revision_cannot_be_patched(client: TestClient, db: Session) -> None:
    from datetime import timedelta

    from tests.test_epochs import T0, make_asset, make_epoch, make_site

    site = make_site(db)
    epoch = make_epoch(db, site, 1, observed=T0)
    old = make_asset(
        db,
        site,
        epoch,
        "https://cdn.example.com/a/tileset.json",
        created=T0,
        valid_to=T0 + timedelta(days=1),
    )
    db.commit()

    refused = client.patch(f"/api/v1/assets/{old.id}", json={"name": "Renamed"})

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "epoch_managed"
    assert "superseded" in refused.json()["detail"]


def test_a_patch_cannot_change_an_assets_source(client: TestClient) -> None:
    """Pinned: `AssetUpdate` has no `source`, and unknown fields are refused, so no PATCH
    can repoint an asset at other tiles (a re-run adds a revision instead)."""
    site = client.post("/api/v1/sites", json=site_payload()).json()
    asset = site["assets"][0]

    response = client.patch(
        f"/api/v1/assets/{asset['id']}",
        json={"source": {"type": "3d-tiles-url", "url": "https://example.com/other.json"}},
    )

    assert response.status_code == 422, response.text


def test_a_patch_waits_for_a_re_run_superseding_the_asset_and_then_refuses(
    db: Session, engine: Engine
) -> None:
    """A re-run supersedes the epoch's asset holding its row lock (`registration.register`).
    A PATCH read the asset unlocked, so it saw a current revision, waited only at its own
    UPDATE, and then wrote its edit onto what had become history. It locks the asset
    first, as a scale and an attach do, and then sees the revision is superseded."""
    from datetime import UTC, datetime

    from app.models import Asset
    from app.schemas.asset import AssetUpdate
    from app.services import assets as asset_service
    from app.services.errors import ConflictError
    from tests.test_asset_scale import run_beside
    from tests.test_epochs import T0, make_asset, make_epoch, make_site

    site = make_site(db)
    epoch = make_epoch(db, site, 1, observed=T0)
    asset = make_asset(db, site, epoch, "https://cdn.example.com/a/tileset.json", created=T0)
    db.commit()
    holder = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        held = holder.get(Asset, asset.id, with_for_update=True)
        assert held is not None
        held.valid_to = datetime.now(tz=UTC)
        held.default_visible = False
        holder.flush()
        thread, result, waited = run_beside(
            engine,
            lambda session: asset_service.update_asset(
                session, asset.id, AssetUpdate(name="Renamed")
            ),
        )
        holder.commit()
    finally:
        holder.close()
    thread.join(timeout=30)

    assert waited, "the PATCH did not wait for the asset's lock"
    refused = result["value"]
    assert isinstance(refused, ConflictError), refused
    assert refused.code == "epoch_managed"
    db.expire_all()
    stored = db.get(Asset, asset.id)
    assert stored is not None and stored.name == "splat"

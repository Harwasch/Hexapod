"""A site's frame and epochs on the API, and a capture naming its site (decision D3)."""

from __future__ import annotations

import uuid
from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.conftest import site_payload
from tests.test_epochs import T0, make_asset, make_epoch, make_site


def test_a_site_lists_its_epochs_in_order_and_404s_when_unknown(
    client: TestClient, db: Session
) -> None:
    site = make_site(db)
    second = make_epoch(db, site, 2, observed=T0 + timedelta(days=30))
    first = make_epoch(db, site, 1, observed=T0)
    db.commit()

    listed = client.get(f"/api/v1/sites/{site.id}/epochs")

    assert listed.status_code == 200, listed.text
    assert [item["id"] for item in listed.json()] == [str(first.id), str(second.id)]
    assert [item["ordinal"] for item in listed.json()] == [1, 2]
    assert listed.json()[0]["joinMethod"] == "new-site"
    assert listed.json()[0]["observedAtSource"] == "captured-at"
    assert listed.json()[0]["footprint"]["type"] == "MultiPolygon"
    missing = client.get(f"/api/v1/sites/{uuid.uuid4()}/epochs")
    assert missing.status_code == 404


def test_a_site_record_carries_its_frame_and_epochs_and_hides_superseded_revisions(
    client: TestClient, db: Session
) -> None:
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
    new = make_asset(
        db, site, epoch, "https://cdn.example.com/b/tileset.json", created=T0 + timedelta(days=1)
    )
    db.commit()

    read = client.get(f"/api/v1/sites/{site.id}").json()

    assert [asset["id"] for asset in read["assets"]] == [str(new.id)]
    assert read["assets"][0]["epochId"] == str(epoch.id)
    assert read["assets"][0]["sourceKind"] == "observed"
    assert read["frame"]["siteId"] == str(site.id)
    assert read["frame"]["heightDatum"] == "msl-unverified"
    assert read["frame"]["epsg"] == 32610
    assert read["frame"]["source"] == "registration"
    assert [item["currentAssetId"] for item in read["epochs"]] == [str(new.id)]
    by_slug = client.get(f"/api/v1/sites/by-slug/{site.slug}").json()
    assert [asset["id"] for asset in by_slug["assets"]] == [str(new.id)]
    # History is still there for whoever asks for every asset.
    listed = client.get("/api/v1/assets", params={"siteId": str(site.id)}).json()
    assert {asset["id"] for asset in listed} == {str(old.id), str(new.id)}


def test_a_legacy_site_has_no_frame_and_no_epochs(client: TestClient) -> None:
    created = client.post("/api/v1/sites", json=site_payload())
    assert created.status_code == 201, created.text

    read = client.get(f"/api/v1/sites/{created.json()['id']}").json()

    assert read["frame"] is None and read["epochs"] == []
    assert len(read["assets"]) == 1
    assert read["assets"][0]["epochId"] is None
    assert read["assets"][0]["sourceKind"] == "observed"
    assert client.get(f"/api/v1/sites/{read['id']}/epochs").json() == []


def test_an_unparseable_stored_quality_reads_as_null(client: TestClient, db: Session) -> None:
    site = make_site(db)
    make_epoch(db, site, 1, observed=T0, quality={"verdict": 12, "nonsense": True})
    db.commit()

    [epoch] = client.get(f"/api/v1/sites/{site.id}/epochs").json()

    assert epoch["quality"] is None

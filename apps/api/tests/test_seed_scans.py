"""The hosted catalog's phone scans, registered locally (`python -m app.seed.scans`).

Everything here reads the committed snapshot or a stand-in: no test reaches the hosted API.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

import app.db
from app.models import Asset, Site
from app.models.enums import Representation
from app.schemas.site import SiteCreate
from app.seed.scans import (
    ORIGIN,
    SCAN_SLUGS,
    SNAPSHOT,
    load_snapshot,
    main,
    refresh_snapshot,
    seed_scans,
    snapshot_site,
)
from app.services import sites as site_service

BUCKET = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/"

#: A hosted `SiteRead`, as the API answers it: the run's metadata included.
HOSTED: dict[str, Any] = {
    "id": "793e0c17-aaf8-4f7c-9dab-3a60634ae056",
    "slug": "phone-capture-2026-09-29-16-34-utc",
    "name": "Camp",
    "description": None,
    "boundary": {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [-123.8815, 46.1328],
                    [-123.8796, 46.1328],
                    [-123.8796, 46.1341],
                    [-123.8815, 46.1341],
                    [-123.8815, 46.1328],
                ]
            ]
        ],
    },
    "centroid": {"longitude": -123.8805, "latitude": 46.1335, "height": None},
    "areaM2": 22047.8,
    "thumbnailUrl": f"{BUCKET}runs/50c2/thumbnail/thumbnail.jpg",
    "metadata": {
        "jobId": "50c25673-0940-4574-9b96-0b21362f83ca",
        "captureId": "c2f5ce7e-b9a8-4580-8cdb-d17a719bf9ec",
        "registration": {"georef": {"lat": 46.1335, "lon": -123.8805, "height": 0.0}},
    },
    "attribution": [],
    "license": None,
    "assets": [
        {
            "id": "f548189f-6089-4554-b573-d44872df3cfe",
            "siteId": "793e0c17-aaf8-4f7c-9dab-3a60634ae056",
            "provider": "3d-tiles-url",
            "name": "Phone capture 2026-09-29 16:34 UTC splat",
            "representation": "gaussian-splat",
            "source": {"type": "3d-tiles-url", "url": f"{BUCKET}runs/50c2/splat/tileset.json"},
            "footprint": None,
            "observedAt": None,
            "validFrom": None,
            "validTo": None,
            "resolution": None,
            "crs": None,
            "license": None,
            "attribution": [],
            "provenance": {
                "georefMethod": "manual",
                "scaleSource": "unresolved",
                "uncertaintyM": 10.0,
            },
            "renderConfig": {
                "clampToGround": True,
                "groundSamples": [{"lon": -123.88068, "lat": 46.13344, "height": 0.075}],
            },
            "defaultVisible": True,
            "sidecarFlags": [],
            "createdAt": "2026-09-29T18:12:49.233350Z",
            "updatedAt": "2026-10-06T02:29:48.707066Z",
        }
    ],
    "cameraBookmarks": [],
    "createdAt": "2026-09-29T18:12:49.203749Z",
    "updatedAt": "2026-10-05T19:47:16.135135Z",
}


def _sites(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(Site)) or 0


def test_the_committed_snapshot_registers_the_three_scans_on_the_bucket() -> None:
    sites = load_snapshot()
    assert [site.slug for site in sites] == list(SCAN_SLUGS)
    assert [site.name for site in sites] == ["Camp", "Pumpkin", "Spool"]
    for site in sites:
        assert site.metadata["origin"] == ORIGIN
        assert "captureId" not in site.metadata
        assert "hosted catalog" in (site.description or "")
        [splat] = site.assets
        assert splat.representation == Representation.GAUSSIAN_SPLAT
        assert splat.source.type == "3d-tiles-url"
        url = str(splat.source.url)
        assert url.startswith(BUCKET)
        assert url.endswith("/package/splat/tileset.json")
        # Where the scan rests on the map, as the hosted catalog measured it.
        assert splat.render_config.ground_samples
        assert splat.provenance is not None and splat.provenance.uncertainty_m is not None


def test_a_hosted_record_is_trimmed_to_what_post_sites_takes() -> None:
    made = snapshot_site(HOSTED, taken="2026-10-16")
    SiteCreate.model_validate(made)
    assert made["slug"] == HOSTED["slug"]
    assert made["metadata"] == {
        "origin": ORIGIN,
        "hostedApi": "https://twin-api.fly.dev",
        "hostedSiteId": HOSTED["id"],
        "snapshotTaken": "2026-10-16",
    }
    # The run's own fields stay hosted; the tiles, ground and provenance travel.
    [asset] = made["assets"]
    assert set(asset) <= {
        "name",
        "representation",
        "source",
        "footprint",
        "observedAt",
        "validFrom",
        "validTo",
        "resolution",
        "crs",
        "license",
        "attribution",
        "provenance",
        "renderConfig",
        "defaultVisible",
    }
    assert asset["source"]["url"].endswith("/splat/tileset.json")
    assert asset["renderConfig"]["groundSamples"][0]["height"] == 0.075
    assert "set by hand, not surveyed, good to about 10 m" in made["description"]


def test_seeding_is_idempotent(db: Session) -> None:
    assert seed_scans(db) == {"created": 3, "updated": 0, "unchanged": 0, "skipped": 0}
    assert seed_scans(db) == {"created": 0, "updated": 0, "unchanged": 3, "skipped": 0}
    assert _sites(db) == 3
    camp = site_service.get_site_by_slug(db, SCAN_SLUGS[0])
    assert camp.name == "Camp"
    assert len(camp.assets) == 1


def test_the_seeded_scans_are_served_like_any_site(client: TestClient, db: Session) -> None:
    seed_scans(db)
    listed = {site["slug"]: site for site in client.get("/api/v1/sites").json()}
    assert set(SCAN_SLUGS) <= set(listed)
    camp = client.get(f"/api/v1/sites/by-slug/{SCAN_SLUGS[0]}").json()
    [asset] = camp["assets"]
    assert asset["source"]["url"].startswith(BUCKET)
    assert asset["renderConfig"]["clampToGround"] is True
    assert asset["provenance"]["georefMethod"] == "manual"


def test_a_newer_snapshot_brings_a_seeded_scan_in_step(db: Session) -> None:
    seed_scans(db)
    rerun = f"{BUCKET}runs/rerun/pnew/package/splat/tileset.json"
    documents = [site.model_dump(mode="json", by_alias=True) for site in load_snapshot()]
    documents[0]["assets"][0]["source"]["url"] = rerun
    newer = [SiteCreate.model_validate(document) for document in documents]
    assert seed_scans(db, newer) == {"created": 0, "updated": 1, "unchanged": 2, "skipped": 0}
    db.expire_all()
    camp = site_service.get_site_by_slug(db, SCAN_SLUGS[0])
    assert camp.assets[0].source["url"] == rerun
    assert db.scalar(select(func.count()).select_from(Asset)) == 3


def test_a_site_this_seed_did_not_make_is_left_alone(db: Session) -> None:
    [camp, *_] = load_snapshot()
    theirs = camp.model_copy(update={"name": "Someone's camp", "metadata": {"origin": "upload"}})
    site_service.create_site(db, theirs)
    assert seed_scans(db)["skipped"] == 1
    db.expire_all()
    assert site_service.get_site_by_slug(db, camp.slug or "").name == "Someone's camp"


def test_refreshing_the_snapshot_reads_each_scan_by_slug(tmp_path: Path) -> None:
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        record = copy.deepcopy(HOSTED)
        record["slug"] = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json=record)

    out = tmp_path / "scans.json"
    committed = SNAPSHOT.read_bytes()
    client = httpx.Client(transport=httpx.MockTransport(answer))
    assert refresh_snapshot(out, api="https://hosted.test", client=client) == 3
    assert asked == [f"/api/v1/sites/by-slug/{slug}" for slug in SCAN_SLUGS]
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["source"] == "https://hosted.test/api/v1/sites/by-slug/<slug>"
    assert [site["slug"] for site in document["sites"]] == list(SCAN_SLUGS)
    assert all(site["metadata"]["hostedApi"] == "https://hosted.test" for site in document["sites"])
    # And it is a snapshot this seed reads.
    assert [site.slug for site in load_snapshot(out)] == list(SCAN_SLUGS)
    # The committed one was not touched.
    assert SNAPSHOT.read_bytes() == committed


def test_a_failed_refresh_leaves_the_snapshot_as_it_was(tmp_path: Path) -> None:
    out = tmp_path / "scans.json"
    out.write_text('{"sites": []}\n', encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        refresh_snapshot(out, api="https://hosted.test", client=client)
    assert out.read_text(encoding="utf-8") == '{"sites": []}\n'


def test_the_command_says_what_it_did(
    db: Session, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Its own session on the test database, as `python -m app.seed.scans` opens one.
    monkeypatch.setattr(app.db, "get_session_factory", lambda: sessionmaker(bind=db.get_bind()))
    main([])
    assert "'created': 3" in capsys.readouterr().out
    assert _sites(db) == 3

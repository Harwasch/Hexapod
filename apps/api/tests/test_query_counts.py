"""Reads that cost the same number of queries however many rows there are.

`GET /sites` asked PostGIS for each site's area in a query of its own, and the artifacts
list loaded each referencing asset's site lazily, one query per asset. Both are counted
here at two sizes: the number of statements must not grow with the rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import Engine, event
from sqlalchemy.orm import Session

from app.services import artifacts as artifact_service
from tests.conftest import site_payload


@contextmanager
def counting(engine: Engine) -> Iterator[list[str]]:
    statements: list[str] = []

    def record(*args: Any) -> None:
        statements.append(str(args[2]))

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", record)


def add_sites(client: TestClient, start: int, count: int) -> None:
    for index in range(start, start + count):
        payload = site_payload(name=f"Site {index}")
        assets: Any = payload["assets"]
        assets[0]["source"] = {
            "type": "3d-tiles-url",
            "url": f"https://tiles.example.com/sites/{index}/tileset.json",
        }
        response = client.post("/api/v1/sites", json=payload)
        assert response.status_code == 201, response.text


def test_the_site_list_does_not_query_per_site(
    client: TestClient, db: Session, engine: Engine
) -> None:
    add_sites(client, 0, 2)
    db.expunge_all()
    with counting(engine) as few:
        listed = client.get("/api/v1/sites").json()
    assert len(listed) == 2 and all(site["areaM2"] > 10_000 for site in listed)

    add_sites(client, 2, 5)
    db.expunge_all()
    with counting(engine) as many:
        listed = client.get("/api/v1/sites").json()
    assert len(listed) == 7 and all(site["areaM2"] > 10_000 for site in listed)
    assert len(many) == len(few), many


def test_the_artifact_references_do_not_query_per_asset(
    client: TestClient, db: Session, engine: Engine
) -> None:
    add_sites(client, 0, 2)
    db.expunge_all()
    with counting(engine) as few:
        found = artifact_service._reference_urls(db)
    assert len(found) == 2

    add_sites(client, 2, 5)
    db.expunge_all()
    with counting(engine) as many:
        found = artifact_service._reference_urls(db)
    assert len(found) == 7
    assert {reference.site_slug for _, reference in found} == {
        f"site-{index}" for index in range(7)
    }
    assert len(many) == len(few), many


def test_the_knowledge_fog_does_not_query_per_record(
    client: TestClient, db: Session, engine: Engine
) -> None:
    """The fog reads each kind of record in one statement, however many there are."""
    from datetime import UTC, datetime, timedelta

    from tests.test_land_features import FEATURE
    from tests.test_land_knowledge import add_evidence, inspect, make_land, research_run

    land = make_land(client)
    run = research_run(db, land["id"])
    observed = datetime.now(UTC) - timedelta(days=1)

    read_counts: list[int] = []

    def grow(count: int) -> None:
        read_counts.append(count)
        for index in range(count):
            source = {**FEATURE["source"], "recordId": f"pole-{index}-{len(read_counts)}"}
            inspect(client, land["id"], observed, name=f"Pole {index}", source=source)
            add_evidence(
                db, run, title=f"Soil record {index}", record_id=f"soil-{index}-{len(read_counts)}"
            )

    grow(1)
    db.expunge_all()
    with counting(engine) as few:
        read = client.get(f"/api/v1/land/{land['id']}/knowledge").json()
    assert len(read["records"]) == 2

    grow(9)
    db.expunge_all()
    with counting(engine) as many:
        read = client.get(f"/api/v1/land/{land['id']}/knowledge").json()
    assert len(read["records"]) == 20
    assert len(many) == len(few), many


def test_a_site_record_does_not_query_per_epoch(
    client: TestClient, db: Session, engine: Engine
) -> None:
    from datetime import timedelta

    from tests.test_epochs import T0, make_asset, make_epoch, make_site

    def site_with(slug: str, count: int) -> str:
        site = make_site(db, slug)
        for ordinal in range(1, count + 1):
            epoch = make_epoch(db, site, ordinal, observed=T0 + timedelta(days=ordinal))
            make_asset(
                db,
                site,
                epoch,
                f"https://cdn.example.com/{slug}/{ordinal}/tileset.json",
                created=T0 + timedelta(days=ordinal),
            )
        db.commit()
        return str(site.id)

    one, four = site_with("one", 1), site_with("four", 4)
    db.expunge_all()
    with counting(engine) as few:
        read = client.get(f"/api/v1/sites/{one}").json()
    assert len(read["epochs"]) == 1
    db.expunge_all()
    with counting(engine) as many:
        read = client.get(f"/api/v1/sites/{four}").json()
    assert len(read["epochs"]) == 4 and len(read["assets"]) == 4
    assert len(many) == len(few), many

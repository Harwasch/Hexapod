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

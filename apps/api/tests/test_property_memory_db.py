"""Property queries against the migrated Ask tables and views on PostGIS."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.models import AskRun, ScanEntity
from app.schemas.ask import Entity
from app.schemas.capture import CaptureCreate
from app.schemas.site import SiteCreate
from app.services import ask_entities as ledger
from app.services.captures import create_capture
from app.services.property_memory import MemoryQuery, PostgresMemory
from app.services.sites import create_site
from tests.conftest import site_payload
from tests.test_ask import ENTITY


@pytest.fixture
def memory_db(db: Session) -> Session:
    return db


def test_property_queries_postgis_scope_distance_polygon_attributes_time(
    memory_db: Session,
) -> None:
    db = memory_db
    site = create_site(db, SiteCreate.model_validate(site_payload())).id
    other = create_site(db, SiteCreate.model_validate(site_payload(slug="other"))).id
    for i, (owner, lon, label, year) in enumerate(
        [
            (site, 0, "oak", 2025),
            (site, 0.0001, "oak", 2026),
            (other, 0, "oak", 2026),
            (site, 0.1, "fence", 2026),
        ],
        start=1,
    ):
        scan = create_capture(
            db, CaptureCreate(name="Simulated memory", kind="video", site_id=owner)
        )
        scan.captured_at = datetime(year, 1, 1, tzinfo=UTC)
        row = ScanEntity(id=uuid.UUID(int=i), capture_id=scan.id, site_id=owner)
        ledger.assign(
            row,
            Entity.model_validate(
                {
                    **ENTITY,
                    "label": label,
                    "attrs": {"healthy": True},
                    "anchor": {
                        "world": ENTITY["anchor"]["world"],
                        "geo": {"lat": 0, "lon": lon, "alt": 0},
                    },
                }
            ),
        )
        db.add(row)
    db.flush()
    repo = PostgresMemory(db)
    assert len(repo.entities(site, MemoryQuery())) == 3
    assert [
        r["id"] for r in repo.entities(site, MemoryQuery(lon=0, lat=0, radius_m=20, nearest=True))
    ] == [str(uuid.UUID(int=1)), str(uuid.UUID(int=2))]
    assert (
        len(
            repo.entities(
                site,
                MemoryQuery(
                    label="oak", attrs={"healthy": True}, since=datetime(2026, 1, 1, tzinfo=UTC)
                ),
            )
        )
        == 1
    )
    assert repo.entities(site, MemoryQuery(attrs={"healthy": False})) == []
    polygon = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]]],
    }
    assert len(repo.entities(site, MemoryQuery.model_validate({"polygon": polygon}))) == 2
    assert len(repo.entities(site, MemoryQuery(limit=1, offset=1))) == 1
    for owner in (site, other):
        db.add(
            AskRun(
                id=str(uuid.uuid4()),
                site_id=owner,
                state="complete",
                request_hash="0" * 64,
                request={"question": "How many oak trees?"},
                events=[],
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
    db.flush()
    assert len(repo.answers(site, MemoryQuery(label="oak"))) == 1
    assert repo.answers(site, MemoryQuery(label="%")) == []

"""Run on PostGIS; temporary adapter tables do not depend on B's migration numbering."""

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.services.property_memory import MemoryQuery, PostgresMemory


@pytest.fixture
def memory_db() -> Iterator[Session]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("set TEST_DATABASE_URL to a local PostGIS test database")
    engine = create_engine(url, connect_args={"connect_timeout": 3})
    with engine.connect() as connection, connection.begin():
        connection.execute(
            text(
                "CREATE TEMP TABLE property_entity_memory (id uuid, site_id uuid, scan_id uuid, "
                "label text, attrs jsonb, entity jsonb, geo geometry(Point,4326), "
                "uncertainty_m float8, observed_at timestamptz) ON COMMIT DROP"
            )
        )
        connection.execute(
            text(
                "CREATE TEMP TABLE property_answer_memory (id uuid, site_id uuid, "
                "scan_id uuid, question text, answer jsonb, observed_at timestamptz) ON COMMIT DROP"
            )
        )
        with Session(bind=connection) as db:
            yield db
    engine.dispose()


def test_property_queries_postgis_scope_distance_polygon_attributes_time(
    memory_db: Session,
) -> None:
    db = memory_db
    site, other = uuid.UUID(int=1), uuid.UUID(int=2)
    for i, (owner, lon, label, year) in enumerate(
        [
            (site, 0, "oak", 2025),
            (site, 0.0001, "oak", 2026),
            (other, 0, "oak", 2026),
            (site, 0.1, "fence", 2026),
        ],
        start=1,
    ):
        db.execute(
            text(
                "INSERT INTO property_entity_memory VALUES (:id, :site, :scan, :label, '{\"healthy\"\\:true}', "
                "'{}', ST_SetSRID(ST_MakePoint(:lon,0),4326), 1, :time)"
            ),
            {
                "id": uuid.UUID(int=i),
                "site": owner,
                "scan": uuid.UUID(int=10 + i),
                "label": label,
                "lon": lon,
                "time": datetime(year, 1, 1, tzinfo=UTC),
            },
        )
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
        db.execute(
            text(
                "INSERT INTO property_answer_memory VALUES (:id,:site,NULL,'How many oak trees?', '{}', :time)"
            ),
            {"id": uuid.uuid4(), "site": owner, "time": datetime(2026, 1, 1, tzinfo=UTC)},
        )
    assert len(repo.answers(site, MemoryQuery(label="oak"))) == 1
    assert repo.answers(site, MemoryQuery(label="%")) == []

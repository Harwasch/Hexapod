"""Timestamps come back from the database in UTC, whatever the session's time zone.

psycopg hands a `timestamptz` back at the offset of the session's `TimeZone`, which comes
from the server's configuration or the client's `PGTZ`. The API serialises datetimes as
they are, so one instant read on two machines printed as two strings ("...T17:00:00-07:00"
for "...T00:00:00Z"), and a date taken from one moved to the day before.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import text

from app.db import make_engine
from tests.conftest import TEST_DATABASE_URL


def test_a_timestamptz_reads_back_in_utc_in_any_session_time_zone() -> None:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        with engine.connect() as connection:
            connection.execute(text("SET TIME ZONE 'America/Los_Angeles'"))
            value = connection.execute(
                text("SELECT TIMESTAMPTZ '2026-09-12 00:00:00+00'")
            ).scalar_one()
    finally:
        engine.dispose()

    assert value.utcoffset() == timedelta(0)
    assert value.isoformat() == "2026-09-12T00:00:00+00:00"

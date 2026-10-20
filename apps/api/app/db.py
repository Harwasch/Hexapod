"""SQLAlchemy engine and session management."""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from psycopg.types.datetime import TimestamptzBinaryLoader, TimestamptzLoader
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


class _UtcTimestamptzLoader(TimestamptzLoader):
    def load(self, data: Any) -> datetime:
        return super().load(data).astimezone(UTC)


class _UtcTimestamptzBinaryLoader(TimestamptzBinaryLoader):
    def load(self, data: Any) -> datetime:
        return super().load(data).astimezone(UTC)


def _read_timestamps_in_utc(dbapi_connection: Any, _record: object) -> None:
    adapters = getattr(dbapi_connection, "adapters", None)
    if adapters is None:  # pragma: no cover - not psycopg 3
        return
    adapters.register_loader("timestamptz", _UtcTimestamptzLoader)
    adapters.register_loader("timestamptz", _UtcTimestamptzBinaryLoader)


def make_engine(url: str, **kwargs: Any) -> Engine:
    """An engine whose `timestamptz` values come back in UTC, whatever the session's
    `TimeZone` (the server's setting, or the client's `PGTZ`).

    psycopg returns a `timestamptz` at the session's offset, and the API serialises a
    datetime as it is: the same instant printed as "...T17:00:00-07:00" on one machine and
    "...T00:00:00Z" on another, and a date read off it a day early. Converting each value
    on the client, rather than `SET TIME ZONE` per session, also holds behind a pooler that
    does not keep session settings. SQL that casts a `timestamptz` to a date or a local
    time still uses the session's zone (no query here does); such SQL must say
    `AT TIME ZONE 'UTC'`.
    """
    engine = create_engine(url, **kwargs)
    event.listen(engine, "connect", _read_timestamps_in_utc)
    return engine


@lru_cache
def get_engine(database_url: str | None = None, pool_size: int = 5) -> Engine:
    """One engine per URL and pool size. 5 is SQLAlchemy's own default, which is what the
    API uses; the worker asks for more when it runs more slots (`loop.pool_size_for`)."""
    url = database_url or get_settings().database_url
    return make_engine(url, pool_pre_ping=True, future=True, pool_size=pool_size)


def get_session_factory(
    database_url: str | None = None, pool_size: int = 5
) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url, pool_size), expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()

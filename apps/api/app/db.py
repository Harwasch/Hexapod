"""SQLAlchemy engine and session management."""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def get_engine(database_url: str | None = None, pool_size: int = 5) -> Engine:
    """One engine per URL and pool size. 5 is SQLAlchemy's own default, which is what the
    API uses; the worker asks for more when it runs more slots (`loop.pool_size_for`)."""
    url = database_url or get_settings().database_url
    return create_engine(url, pool_pre_ping=True, future=True, pool_size=pool_size)


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

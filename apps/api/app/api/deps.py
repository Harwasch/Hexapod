from __future__ import annotations

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.services.ion import IonClient
from app.storage import ObjectStorage, get_storage


def _db() -> Generator[Session, None, None]:
    yield from get_db()


def _ion() -> IonClient:
    return IonClient()


DbSession = Annotated[Session, Depends(_db)]
Ion = Annotated[IonClient, Depends(_ion)]
Storage = Annotated[ObjectStorage, Depends(get_storage)]

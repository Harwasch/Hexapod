"""Storage as the console sees it: reconciled against the database, in both directions.

The two answers this gives are different questions, and neither one implies the other:

* **orphans** — objects in the bucket that no row claims. They happen: a run that
  uploaded three artifacts and died before committing its step, a capture whose row was
  deleted, a key written by a version of the code that no longer exists. They cost money
  every month and nothing points at them.
* **missing** — rows whose objects are gone. A site pointing at a deleted tileset is a
  dead link on the globe, and there is no other place that would notice.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from app.schemas.base import CamelModel


class OrphanObject(CamelModel):
    """An object in the bucket that no row accounts for."""

    key: str
    bytes: int
    last_modified: datetime
    etag: str
    #: Why nothing claims it, in one line — the console shows this beside the key.
    reason: str


class MissingObject(CamelModel):
    """A row whose object is not in the bucket."""

    #: `capture-file`, `artifact` or `step-log` — which table the row is in.
    kind: str
    id: uuid.UUID
    key: str
    #: The filename, artifact kind or stage id, whichever names this row to a person.
    label: str
    capture_id: uuid.UUID | None
    job_id: uuid.UUID | None


class StorageReconciliation(CamelModel):
    """One pass over the prefixes this system owns, against the rows that claim them."""

    prefixes: list[str]
    #: Objects walked. `truncated` means the walk hit `maxObjects` and stopped — the
    #: orphan list is then a sample, and the counts below are of what was walked.
    scanned: int
    truncated: bool
    matched: int
    bytes_scanned: int
    bytes_orphaned: int
    orphans: list[OrphanObject]
    #: Rows checked against storage directly, so this half is exact even when the walk
    #: was truncated.
    rows_checked: int
    missing: list[MissingObject]
    checked_at: datetime

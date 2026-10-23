"""Re-date the epochs still dated by their upload, from their own run's evidence.

A one-time correction, kept as a command so it can be read, tested and run again. Before
the re-run fix in `registration.register`, a re-run never re-dated an existing epoch, so an
epoch whose first runs read no capture time kept its upload time (`observed_at_source`
`created`) after a later run had read the files' own clock. Production, 2026-10-22: Pumpkin
and Spool, re-processed on recipe 15, kept 2026-09-28 and 2026-09-26 (their uploads) while
their videos say 2026-09-27 and 2026-09-24.

It reads only what the database holds: the registration document of the epoch's current
run, which `register` mirrors onto a pipeline-made site's metadata for the site's first
epoch. The document must be that run's own: `registrationJobId` names the run that wrote
it. A mirror from before that key names only the latest run (`jobId`), which may have had
no georeference to write a document; it is trusted only when that run's `georeference`
step completed, since such a run always writes one. The date comes out of the document by
the rule `register` uses (`registration.observed_at_of`). An epoch it cannot vouch for is
listed and left alone: one whose site does not hold its current run's document, or one
whose run read no date either. The epoch's current asset revisions that carry the old date
move with it (`registration.redate_current_assets`). Nothing reads object storage, and
nothing else changes.

    uv run python -m app.scripts.redate_epochs                 # dry run: lists, writes nothing
    uv run python -m app.scripts.redate_epochs --apply         # writes, in one transaction
    uv run python -m app.scripts.redate_epochs --epoch <uuid>  # only that epoch (repeatable)

In production, on the API machine (`fly ssh console -a twin-api --user api`), with the
image's interpreter, `/app/.venv/bin/python -m app.scripts.redate_epochs`.

Idempotent: a re-dated epoch is no longer `created`, so a second run finds nothing to do.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Capture, Epoch, JobStep, Site
from app.models.enums import ObservedAtSource, RunStatus
from app.worker.registration import observed_at_of, redate_current_assets

#: The last line of a run that finished, for a caller over SSH.
DONE = "REDATE-EPOCHS-OK"

NO_DOCUMENT = "its site does not hold its current run's registration document"
NO_DATE = "its current run read no capture time either"


@dataclass(frozen=True)
class Finding:
    """One epoch dated by its upload, and what this command does about it."""

    epoch_id: uuid.UUID
    site_name: str
    old: datetime
    #: The date and its source from the run's evidence; None when the epoch is left alone.
    new: datetime | None = None
    source: ObservedAtSource | None = None
    #: Why it is left alone, when it is.
    skipped: str | None = None

    def line(self) -> str:
        head = f"epoch {self.epoch_id} ({self.site_name}): {_utc(self.old)} (created)"
        if self.new is None or self.source is None:
            return f"{head} left as it is: {self.skipped}"
        return f"{head} -> {_utc(self.new)} ({self.source.value})"


def _utc(when: datetime) -> str:
    return when.astimezone(UTC).isoformat()


def _current_document(db: Session, epoch: Epoch, site: Site | None) -> dict[str, Any] | None:
    """The registration document of the epoch's current run, when its site holds it."""
    metadata = (site.metadata_ if site is not None else None) or {}
    document = metadata.get("registration")
    run = str(epoch.job_id) if epoch.job_id is not None else None
    if run is None or not isinstance(document, dict):
        return None
    if metadata.get("captureId") != str(epoch.capture_id):
        return None
    if "registrationJobId" in metadata:
        return document if metadata["registrationJobId"] == run else None
    # A mirror from before `registrationJobId`: only the latest run's id, trusted when that
    # run georeferenced (and so wrote the document beside it).
    georeferenced = db.scalar(
        select(JobStep.id).where(
            JobStep.job_id == epoch.job_id,
            JobStep.stage_id == "georeference",
            JobStep.status == RunStatus.COMPLETE,
        )
    )
    return document if metadata.get("jobId") == run and georeferenced is not None else None


def _judge(db: Session, epoch: Epoch) -> Finding:
    site = db.get(Site, epoch.site_id)
    name = site.name if site is not None else "no site"
    document = _current_document(db, epoch, site)
    capture = db.get(Capture, epoch.capture_id) if document is not None else None
    if capture is None or document is None:
        return Finding(epoch.id, name, epoch.observed_at, skipped=NO_DOCUMENT)
    when, source = observed_at_of(capture, document)
    if source is ObservedAtSource.CREATED:
        return Finding(epoch.id, name, epoch.observed_at, skipped=NO_DATE)
    return Finding(epoch.id, name, epoch.observed_at, when, source)


def _created(only: Sequence[uuid.UUID]) -> Any:
    query = select(Epoch).where(Epoch.observed_at_source == ObservedAtSource.CREATED.value)
    if only:
        query = query.where(Epoch.id.in_(only))
    return query.order_by(Epoch.site_id, Epoch.ordinal)


def find(db: Session, only: Sequence[uuid.UUID] = ()) -> list[Finding]:
    """Every epoch dated by its upload (of `only`, when given), with its run's date."""
    return [_judge(db, epoch) for epoch in db.scalars(_created(only))]


def apply(db: Session, only: Sequence[uuid.UUID] = ()) -> list[Finding]:
    """Re-dates them, in the caller's transaction, each judged again under its row's lock
    (a run registered since a dry run may have dated it, or superseded its asset)."""
    findings = []
    for epoch in db.scalars(_created(only).with_for_update(of=Epoch)):
        finding = _judge(db, epoch)
        findings.append(finding)
        if finding.new is None or finding.source is None:
            continue
        epoch.observed_at = finding.new
        epoch.observed_at_source = finding.source.value
        redate_current_assets(db, epoch, finding.old)
    return findings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.scripts.redate_epochs",
        description=(
            "Re-date the epochs still dated by their upload (`created`) from their own run's "
            "evidence. A dry run unless --apply."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="write the new dates")
    parser.add_argument(
        "--epoch",
        action="append",
        type=uuid.UUID,
        default=[],
        help="only this epoch (repeatable); every `created` epoch when left out",
    )
    args = parser.parse_args(argv)
    from app.db import get_session_factory

    with get_session_factory()() as db:
        findings = apply(db, args.epoch) if args.apply else find(db, args.epoch)
        for finding in findings:
            sys.stdout.write(finding.line() + "\n")
        moved = sum(1 for finding in findings if finding.new is not None)
        if args.apply:
            db.commit()
            sys.stdout.write(f"re-dated {moved} epoch(s)\n")
        else:
            sys.stdout.write(f"dry run: {moved} epoch(s) would be re-dated; nothing written\n")
    sys.stdout.write(DONE + "\n")


if __name__ == "__main__":
    main()

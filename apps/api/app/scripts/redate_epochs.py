"""Re-date the epochs still dated by their upload, from their own run's evidence.

A one-time correction, kept as a command so it can be read, tested and run again. Before
the re-run fix in `registration.register`, a re-run never re-dated an existing epoch, so an
epoch whose first runs read no capture time kept its upload time (`observed_at_source`
`created`) after a later run had read the files' own clock. Production, 2026-10-22: Pumpkin
and Spool, re-processed on recipe 15, kept 2026-09-28 and 2026-09-26 (their uploads) while
their videos say 2026-09-27 and 2026-09-24.

It reads only what the database holds: the registration document of the epoch's current
run, which `register` mirrors onto a pipeline-made site's metadata for the site's first
epoch (`registration`, beside `captureId` and `jobId` naming that run). The date comes out
of the document by the rule `register` uses (`registration.observed_at_of`). An epoch it
cannot vouch for is listed and left alone: one whose site does not mirror its current run,
or one whose run read no date either. The epoch's current asset revisions that carry the
old date move with it. Nothing reads object storage, and nothing else changes.

    uv run python -m app.scripts.redate_epochs                 # dry run: lists, writes nothing
    uv run python -m app.scripts.redate_epochs --apply         # writes, in one transaction
    uv run python -m app.scripts.redate_epochs --epoch <uuid>  # only that epoch (repeatable)

Idempotent: a re-dated epoch is no longer `created`, so a second run finds nothing to do.
`.github/workflows/redate-epochs.yml` runs it on the API machine.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Asset, Capture, Epoch, Site
from app.models.enums import ObservedAtSource
from app.worker.registration import observed_at_of

#: The last line of a run that finished, for a caller over SSH (`redate-epochs.yml`).
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
    #: The epoch's current asset revisions that carry the old date.
    assets: tuple[uuid.UUID, ...] = ()

    def line(self) -> str:
        head = f"epoch {self.epoch_id} ({self.site_name}): {_utc(self.old)} (created)"
        if self.new is None or self.source is None:
            return f"{head} left as it is: {self.skipped}"
        return (
            f"{head} -> {_utc(self.new)} ({self.source.value}),"
            f" {len(self.assets)} current asset(s) with it"
        )


def _utc(when: datetime) -> str:
    return when.astimezone(UTC).isoformat()


def find(db: Session, only: Sequence[uuid.UUID] = ()) -> list[Finding]:
    """Every epoch dated by its upload (of `only`, when given), with its run's date."""
    query = select(Epoch).where(Epoch.observed_at_source == ObservedAtSource.CREATED.value)
    if only:
        query = query.where(Epoch.id.in_(only))
    findings = []
    for epoch in db.scalars(query.order_by(Epoch.site_id, Epoch.ordinal)):
        site = db.get(Site, epoch.site_id)
        name = site.name if site is not None else "no site"
        metadata = (site.metadata_ if site is not None else None) or {}
        document = metadata.get("registration")
        mirrored = (
            isinstance(document, dict)
            and epoch.job_id is not None
            and metadata.get("captureId") == str(epoch.capture_id)
            and metadata.get("jobId") == str(epoch.job_id)
        )
        capture = db.get(Capture, epoch.capture_id) if mirrored else None
        if capture is None or not isinstance(document, dict):
            findings.append(Finding(epoch.id, name, epoch.observed_at, skipped=NO_DOCUMENT))
            continue
        when, source = observed_at_of(capture, document)
        if source is ObservedAtSource.CREATED:
            findings.append(Finding(epoch.id, name, epoch.observed_at, skipped=NO_DATE))
            continue
        assets = db.scalars(
            select(Asset.id).where(
                Asset.epoch_id == epoch.id,
                Asset.valid_to.is_(None),
                Asset.observed_at == epoch.observed_at,
            )
        )
        findings.append(
            Finding(epoch.id, name, epoch.observed_at, when, source, assets=tuple(assets))
        )
    return findings


def apply(db: Session, findings: Sequence[Finding]) -> int:
    """Writes what `find` found, in the caller's transaction; how many epochs moved."""
    moved = 0
    for finding in findings:
        if finding.new is None or finding.source is None:
            continue
        epoch = db.get(Epoch, finding.epoch_id, with_for_update=True, populate_existing=True)
        # Checked again under the lock: a run registered since `find` may have dated it.
        if epoch is None or epoch.observed_at_source != ObservedAtSource.CREATED.value:
            continue
        epoch.observed_at = finding.new
        epoch.observed_at_source = finding.source.value
        for asset_id in finding.assets:
            asset = db.get(Asset, asset_id)
            if asset is not None and asset.observed_at == finding.old:
                asset.observed_at = finding.new
        moved += 1
    return moved


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
        findings = find(db, args.epoch)
        for finding in findings:
            sys.stdout.write(finding.line() + "\n")
        movable = sum(1 for finding in findings if finding.new is not None)
        if args.apply:
            moved = apply(db, findings)
            db.commit()
            sys.stdout.write(f"re-dated {moved} epoch(s)\n")
        else:
            sys.stdout.write(f"dry run: {movable} epoch(s) would be re-dated; nothing written\n")
    sys.stdout.write(DONE + "\n")


if __name__ == "__main__":
    main()

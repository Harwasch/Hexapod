"""Every output the pipeline has ever written, with its provenance and its references.

The `artifacts` table exists so outputs are explicit rather than implicit in the bucket
(A2). This is the read side of that: one flat list, because the question the Outputs view
answers — "what is in here, what made it, and does anything still need it?" — is not a
question about one run.

*Referenced* means something points at the object. `register` (A8) writes the packaged
tileset's public URL onto a site asset and the thumbnail's onto the site, and those URLs
are `public_url(storage_key)`, so a reference is found by looking for the key inside them.
That is a substring test rather than a join, and deliberately: the URL is what a browser
actually fetches, and a reference that only exists in a column the renderer never reads
would be a reference that does not stop the object being deleted.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models import Artifact, Asset, Capture, Job, JobStep, Site
from app.models.enums import ArtifactKind
from app.schemas.job import ArtifactReference, ArtifactRow


def _reference_urls(db: Session) -> list[tuple[str, ArtifactReference]]:
    """Every URL a site points at, paired with the reference it would become."""
    found: list[tuple[str, ArtifactReference]] = []
    for site in db.scalars(select(Site)).all():
        if site.thumbnail_url:
            found.append(
                (
                    site.thumbnail_url,
                    ArtifactReference(
                        kind="site-thumbnail",
                        site_id=site.id,
                        site_slug=site.slug,
                        label=f"{site.name} thumbnail",
                    ),
                )
            )
    for asset in db.scalars(select(Asset).where(Asset.site_id.isnot(None))).all():
        url = asset.source.get("url") if isinstance(asset.source, dict) else None
        owner = asset.site
        if not isinstance(url, str) or owner is None:
            continue
        found.append(
            (
                url,
                ArtifactReference(
                    kind="site-asset",
                    site_id=owner.id,
                    site_slug=owner.slug,
                    label=asset.name,
                ),
            )
        )
    return found


def _rows() -> Select[tuple[Artifact, JobStep, Job, Capture]]:
    return (
        select(Artifact, JobStep, Job, Capture)
        .join(JobStep, Artifact.job_step_id == JobStep.id)
        .join(Job, JobStep.job_id == Job.id)
        .join(Capture, Job.capture_id == Capture.id)
        .order_by(Artifact.created_at.desc(), Artifact.id.desc())
    )


def list_artifacts(
    db: Session,
    *,
    kind: ArtifactKind | None = None,
    job_id: uuid.UUID | None = None,
    capture_id: uuid.UUID | None = None,
    unreferenced: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[ArtifactRow]:
    """Artifacts newest first, each carrying its run and what points at it.

    `unreferenced` filters *after* the reference lookup rather than in SQL, because a
    reference is a substring of a URL and not a foreign key. That makes `limit` a limit on
    rows examined as well as returned, which is honest for a console page and would not be
    for an API meant to be paged through to the end.
    """
    stmt = _rows().limit(limit).offset(offset)
    if kind is not None:
        stmt = stmt.where(Artifact.kind == kind)
    if job_id is not None:
        stmt = stmt.where(Job.id == job_id)
    if capture_id is not None:
        stmt = stmt.where(Job.capture_id == capture_id)

    urls = _reference_urls(db)
    rows: list[ArtifactRow] = []
    for artifact, step, job, capture in db.execute(stmt).all():
        references = [
            reference
            for url, reference in urls
            if artifact.storage_key and artifact.storage_key in url
        ]
        if unreferenced is True and references:
            continue
        if unreferenced is False and not references:
            continue
        rows.append(
            ArtifactRow(
                id=artifact.id,
                job_step_id=artifact.job_step_id,
                kind=artifact.kind,
                storage_key=artifact.storage_key,
                bytes=artifact.bytes,
                checksum=artifact.checksum,
                content_type=artifact.content_type,
                created_at=artifact.created_at,
                updated_at=artifact.updated_at,
                stage_id=step.stage_id,
                impl=step.impl,
                job_id=job.id,
                recipe=job.recipe,
                capture_id=capture.id,
                capture_name=capture.name,
                references=references,
            )
        )
    return rows

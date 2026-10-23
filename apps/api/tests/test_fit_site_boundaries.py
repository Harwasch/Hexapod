"""The one-time fit of the boundaries registration only grew (`app.scripts.fit_site_boundaries`).

Each test starts from the state Spool was in on 2026-10-22 (`_left_by_grow_only`): its
epoch's footprint the small box its re-run measured, the floaters' box in the epoch's
history, and the site still on the floaters' boundary.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from geoalchemy2.shape import to_shape
from sqlalchemy.orm import Session

from app.models import Site
from app.scripts import fit_site_boundaries
from app.services import geometry
from app.storage import S3Storage
from tests.test_registration_epochs import _epochs, _left_by_grow_only


def _site(db: Session, site_id: uuid.UUID) -> Site:
    db.expire_all()
    site = db.get(Site, site_id)
    assert site is not None
    return site


def test_a_dry_run_lists_the_fit_with_its_areas_and_writes_nothing(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, site_id, floaters = _left_by_grow_only(db, storage, tmp_path)

    [fitted] = fit_site_boundaries.fit(db)
    db.rollback()

    assert fitted.site_id == site_id
    assert fitted.after_m2 < fitted.before_m2 / 40
    assert f"{fitted.before_m2:,.1f} m2 -> {fitted.after_m2:,.1f} m2" in fitted.line()
    assert to_shape(_site(db, site_id).boundary).equals(to_shape(floaters))


def test_apply_fits_the_boundary_to_the_epoch_keeps_the_centroid_inside_and_is_idempotent(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, site_id, _floaters = _left_by_grow_only(db, storage, tmp_path)

    assert len(fit_site_boundaries.fit(db)) == 1
    db.commit()

    site = _site(db, site_id)
    [epoch] = _epochs(db, site_id)
    assert geometry.contains(site.boundary, epoch.footprint)
    assert geometry.contains(epoch.footprint, site.boundary)
    assert geometry.contains(site.boundary, site.centroid)
    assert fit_site_boundaries.fit(db) == []


def test_a_boundary_somebody_drew_is_left_as_it_is(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, site_id, floaters = _left_by_grow_only(db, storage, tmp_path)
    site = _site(db, site_id)
    site.boundary_source = "manual"
    db.commit()

    assert fit_site_boundaries.fit(db) == []
    assert to_shape(_site(db, site_id).boundary).equals(to_shape(floaters))


def test_only_the_sites_asked_for_are_looked_at(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, site_id, _floaters = _left_by_grow_only(db, storage, tmp_path)

    assert fit_site_boundaries.fit(db, [uuid.uuid4()]) == []
    assert [f.site_id for f in fit_site_boundaries.fit(db, [site_id])] == [site_id]

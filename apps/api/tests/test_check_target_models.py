"""`site_check_targets`: what the database itself refuses, whatever the API lets through."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Site, SiteCheckTarget
from app.schemas.site import SiteCreate
from app.services import sites as site_service
from tests.conftest import site_payload


def _site(db: Session) -> Site:
    return site_service.create_site(db, SiteCreate.model_validate(site_payload(assets=[])))


def _target(site: Site, **overrides: Any) -> SiteCheckTarget:
    fields: dict[str, Any] = {
        "site_id": site.id,
        "label": "T1",
        "role": "check",
        "lat": 47.6443,
        "lon": -122.138,
        "height": 10.8,
        "height_datum": "ellipsoid-wgs84",
        "sigma_h_m": 0.005,
        "sigma_v_m": 0.01,
        "source": "rtk-rover",
    }
    fields.update(overrides)
    return SiteCheckTarget(**fields)


@pytest.mark.parametrize(
    "bad",
    [
        {"role": "gcp"},
        {"height_datum": "egm96"},
        {"sigma_h_m": 0.0},
        {"sigma_v_m": -0.01},
        {"lat": 91.0},
    ],
    ids=["role", "datum", "sigma-h", "sigma-v", "lat"],
)
def test_the_table_refuses_what_is_not_a_survey(db: Session, bad: dict[str, Any]) -> None:
    site = _site(db)
    db.add(_target(site, **bad))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_one_active_row_per_label_and_any_number_retired(db: Session) -> None:
    site = _site(db)
    now = datetime.now(tz=UTC)
    db.add_all([_target(site, retired_at=now), _target(site, retired_at=now), _target(site)])
    db.commit()
    db.add(_target(site, lat=47.6444))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    # Another site's T1 is another mark.
    other = _site(db)
    db.add(_target(other))
    db.commit()


def test_deleting_the_site_deletes_its_targets(db: Session) -> None:
    site = _site(db)
    db.add(_target(site))
    db.commit()
    site_service.delete_site(db, site.id)
    db.expire_all()
    assert list(db.scalars(select(SiteCheckTarget))) == []

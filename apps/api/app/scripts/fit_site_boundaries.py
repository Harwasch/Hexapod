"""Fit the site boundaries registration only ever grew to their epochs' current footprints.

A one-time correction, kept as a command so it can be read, tested and run again, beside
`redate_epochs`. Before `registration.fit_site_boundary`, a re-run replaced its epoch's
footprint but only grew the site's boundary, so a site whose first run drew a large box
kept it after a later run measured less. Production, 2026-10-22: Spool's v15 re-run measured
3.7 x 3.9 m and its site kept the 44 x 53 m (2,361 m^2) its first run's background floaters
drew; Pumpkin's kept 7.6 x 6.9 m around a 2.3 x 2.1 m epoch.

Each site goes through `fit_site_boundary`, the rule `register` now applies after every
run: only a boundary registration drew (`boundary_source` `registration`), with a published
epoch and no runtime scale in effect; it becomes the union of the current footprints, and
a centroid left outside moves inside. Anything else is left as it is. It reads only the
database.

**Name the sites (`--site`).** Migration 0038 backfilled `registration` where a boundary
was exactly its epochs' footprints, current and replaced; a boundary somebody drew by hand
before 0038 that happened to equal them cannot be told from the shapes. So in production
this runs on the sites it was written for (Pumpkin, Spool), never on all of them.

    uv run python -m app.scripts.fit_site_boundaries                # dry run: lists, writes nothing
    uv run python -m app.scripts.fit_site_boundaries --apply        # writes, in one transaction
    uv run python -m app.scripts.fit_site_boundaries --site <uuid>  # only that site (repeatable)

In production, on the API machine (`fly ssh console -a twin-api --user api`), with the
image's interpreter, `/app/.venv/bin/python -m app.scripts.fit_site_boundaries`.

Each site is fitted under its row lock, a dry run too (rolled back), so the areas it prints
are the ones an apply writes. Each fitted site's old boundary and centroid are printed in
full, as WKT at full precision, so an apply's output is what a rollback restores from.
Idempotent: a fitted boundary is its epochs' current union, which a second run leaves as
it is.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from geoalchemy2 import WKBElement
from geoalchemy2.shape import to_shape
from shapely import wkt
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Site
from app.models.enums import SiteBoundarySource
from app.services import geometry
from app.services.sites import site_area_m2
from app.worker.registration import fit_site_boundary

#: The last line of a run that finished, for a caller over SSH.
DONE = "FIT-SITE-BOUNDARIES-OK"


@dataclass(frozen=True)
class Fitted:
    """One site whose boundary was fitted: its area and centroid before and after, and what
    it replaced in full (WKT, EPSG:4326) for a rollback."""

    site_id: uuid.UUID
    name: str
    before_m2: float
    after_m2: float
    before: tuple[float, float]
    after: tuple[float, float]
    old_boundary_wkt: str
    old_centroid_wkt: str

    def line(self) -> str:
        return (
            f"site {self.site_id} ({self.name}): {self.before_m2:,.1f} m2 -> "
            f"{self.after_m2:,.1f} m2, centroid {_point(self.before)} -> {_point(self.after)}\n"
            f"  was boundary {self.old_boundary_wkt}\n"
            f"  was centroid {self.old_centroid_wkt}"
        )


def _wkt(value: WKBElement) -> str:
    """A stored geometry as WKT at full precision, the form a rollback writes back."""
    return wkt.dumps(to_shape(value), rounding_precision=-1)


def _point(lon_lat: tuple[float, float]) -> str:
    return f"({lon_lat[1]:.6f}, {lon_lat[0]:.6f})"


def _centroid(site: Site) -> tuple[float, float]:
    position = geometry.wkb_to_position(site.centroid, site.centroid_height)
    return position.longitude, position.latitude


def fit(db: Session, only: Sequence[uuid.UUID] = ()) -> list[Fitted]:
    """Fits every registration-drawn boundary that needs it (of `only`, when given), in the
    caller's transaction; the caller commits or rolls back."""
    query = select(Site).where(Site.boundary_source == SiteBoundarySource.REGISTRATION.value)
    if only:
        query = query.where(Site.id.in_(only))
    fitted = []
    for site in db.scalars(query.order_by(Site.created_at).with_for_update()):
        before_m2, before = site_area_m2(db, site), _centroid(site)
        old_boundary, old_centroid = _wkt(site.boundary), _wkt(site.centroid)
        if not fit_site_boundary(db, site):
            continue
        db.flush()
        fitted.append(
            Fitted(
                site.id,
                site.name,
                before_m2,
                site_area_m2(db, site),
                before,
                _centroid(site),
                old_boundary,
                old_centroid,
            )
        )
    return fitted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.scripts.fit_site_boundaries",
        description=(
            "Fit the site boundaries registration only ever grew to their epochs' current "
            "footprints. A dry run unless --apply."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="write the fitted boundaries")
    parser.add_argument(
        "--site",
        action="append",
        type=uuid.UUID,
        default=[],
        help="only this site (repeatable); every registration-drawn site when left out",
    )
    args = parser.parse_args(argv)
    from app.db import get_session_factory

    with get_session_factory()() as db:
        fitted = fit(db, args.site)
        for site in fitted:
            sys.stdout.write(site.line() + "\n")
        if args.apply:
            db.commit()
            sys.stdout.write(f"fitted {len(fitted)} site boundary(ies)\n")
        else:
            db.rollback()
            sys.stdout.write(
                f"dry run: {len(fitted)} site boundary(ies) would be fitted; nothing written\n"
            )
    sys.stdout.write(DONE + "\n")


if __name__ == "__main__":
    main()

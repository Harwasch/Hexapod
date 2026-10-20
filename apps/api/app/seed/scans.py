"""The hosted catalog's phone scans, registered in a local catalog: an opt-in seed.

The scans that carry objects (`instances.json`) with image embeddings (`instances.emb`) --
the camp, the pumpkin and the spool -- were captured on a phone and run by the hosted
deployment's pipeline. Their tiles, objects and embeddings live in the public bucket, and
only the hosted database has rows for them, so a fresh local catalog has no scan that "More
like this" (apps/web/src/features/sites/MoreLikeThis.tsx) can be tried on.

This registers them locally the way `POST /api/v1/sites` would: one site per scan, its splat
asset pointing at the same public-bucket tileset the hosted catalog serves, nothing copied.
What it registers is `hosted_scans.json`, beside this file: a snapshot of the hosted
records, trimmed to what `SiteCreate` takes and committed, so seeding reads no network and
the hosted API is never a dependency of a run or a test. The tiles themselves are of course
fetched from the bucket by the browser.

    uv run python -m app.seed.scans                     # register them (idempotent)
    uv run python -m app.seed.scans --refresh-snapshot  # re-read the hosted records

A hosted re-run publishes a scan's tiles under a new prefix and repoints its site; the old
prefix stays readable, so a stale snapshot still draws, just not the newest run. Refreshing
the snapshot is the one step that reads the hosted API, and only when asked.

Seeding is idempotent by slug. A site this created is brought in step with the snapshot (its
asset's tiles, render config and provenance, its words and boundary); a site with the same
slug that this did not create (`metadata.origin`) is left alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Site
from app.schemas.site import SiteCreate
from app.services import geometry
from app.services import sites as site_service
from app.services.assets import build_asset

logger = logging.getLogger("twin.seed.scans")

#: The hosted catalog the snapshot is read from (only by `--refresh-snapshot`).
HOSTED_API = "https://twin-api.fly.dev"

#: The snapshot this seeds from.
SNAPSHOT = Path(__file__).with_name("hosted_scans.json")

#: What `metadata.origin` says on a site this created.
ORIGIN = "hosted-catalog"

#: The scans with objects and SigLIP 2 embeddings, by their hosted slug: the camp (28,400
#: objects, 4,292 described, a 43.6 MB embedding file), the pumpkin and the spool.
SCAN_SLUGS = (
    "phone-capture-2026-09-29-16-34-utc",
    "phone-capture-2026-09-28-00-30-utc",
    "phone-capture-2026-09-26-20-41-utc",
)

#: Asset fields `AssetBase` takes, as the hosted `AssetRead` names them.
ASSET_FIELDS = (
    "name",
    "representation",
    "source",
    "footprint",
    "observedAt",
    "validFrom",
    "validTo",
    "resolution",
    "crs",
    "license",
    "attribution",
    "provenance",
    "renderConfig",
    "defaultVisible",
)

#: How each georeference method placed a scan, for its description.
PLACED = {
    "manual": "set by hand, not surveyed",
    "exif-gps": "from the photos' GPS",
}


def _description(site: dict[str, Any]) -> str:
    """Says where the scan comes from and how well it is placed, from its own provenance."""
    provenance = next(
        (a.get("provenance") for a in site.get("assets", []) if a.get("provenance")), None
    )
    placed = ""
    if isinstance(provenance, dict):
        method = PLACED.get(str(provenance.get("georefMethod")), "from its pipeline run")
        uncertainty = provenance.get("uncertaintyM")
        placed = f" Its position is {method}" + (
            f", good to about {uncertainty:g} m." if isinstance(uncertainty, int | float) else "."
        )
    return (
        "A phone scan from the hosted catalog, registered in this catalog by "
        "`python -m app.seed.scans`. Its tiles, objects and object embeddings are read from "
        f"the public bucket the hosted catalog uses.{placed}"
    )


def snapshot_site(site: dict[str, Any], *, taken: str, api: str = HOSTED_API) -> dict[str, Any]:
    """A hosted `SiteRead` as the `SiteCreate` this seeds: the same slug, name, boundary,
    thumbnail and assets (their tileset URLs, render configs and provenance), with words and
    metadata that say where it came from. The hosted run's metadata (its manifest and
    registration) is left out: nothing local reads it, and without `captureId` a local
    resize is refused rather than written against a run this catalog does not have."""
    created = {
        "slug": site["slug"],
        "name": site["name"],
        "description": _description(site),
        "boundary": site["boundary"],
        "centroid": site.get("centroid"),
        "thumbnailUrl": site.get("thumbnailUrl"),
        "metadata": {
            "origin": ORIGIN,
            "hostedApi": api,
            "hostedSiteId": site["id"],
            "snapshotTaken": taken,
        },
        "attribution": site.get("attribution", []),
        "license": site.get("license"),
        "assets": [
            {key: asset[key] for key in ASSET_FIELDS if key in asset}
            for asset in site.get("assets", [])
        ],
        "cameraBookmarks": [
            {
                key: bookmark[key]
                for key in (
                    "name",
                    "longitude",
                    "latitude",
                    "height",
                    "heading",
                    "pitch",
                    "roll",
                    "isDefault",
                )
                if key in bookmark
            }
            for bookmark in site.get("cameraBookmarks", [])
        ],
    }
    # What it registers must be what `POST /sites` would take.
    SiteCreate.model_validate(created)
    return created


def load_snapshot(path: Path = SNAPSHOT) -> list[SiteCreate]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return [SiteCreate.model_validate(site) for site in document["sites"]]


def refresh_snapshot(
    path: Path = SNAPSHOT,
    *,
    api: str = HOSTED_API,
    slugs: tuple[str, ...] = SCAN_SLUGS,
    client: httpx.Client | None = None,
) -> int:
    """Re-reads the scans' records from the hosted catalog and rewrites the snapshot."""
    taken = datetime.now(UTC).date().isoformat()
    owned = client is None
    http = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        sites = []
        for slug in slugs:
            response = http.get(f"{api.rstrip('/')}/api/v1/sites/by-slug/{slug}")
            response.raise_for_status()
            sites.append(snapshot_site(response.json(), taken=taken, api=api))
    finally:
        if owned:
            http.close()
    document = {
        "about": (
            "The hosted catalog's phone scans with objects and image embeddings, as "
            "`python -m app.seed.scans` registers them. Rewritten by "
            "`python -m app.seed.scans --refresh-snapshot`; not hand-edited."
        ),
        "source": f"{api.rstrip('/')}/api/v1/sites/by-slug/<slug>",
        "taken": taken,
        "sites": sites,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(sites)


def _bring_in_step(db: Session, existing: Site, wanted: SiteCreate) -> list[str]:
    """What changed on a site this seeded, to match the snapshot."""
    changed: list[str] = []
    for field, value in (
        ("name", wanted.name),
        ("description", wanted.description),
        ("thumbnail_url", str(wanted.thumbnail_url) if wanted.thumbnail_url else None),
        ("metadata_", dict(wanted.metadata)),
    ):
        if getattr(existing, field) != value:
            setattr(existing, field, value)
            changed.append(field.rstrip("_"))
    boundary = geometry.footprint_to_wkb(wanted.boundary)
    if geometry.wkb_to_footprint(existing.boundary) != geometry.wkb_to_footprint(boundary):
        existing.boundary = boundary
        centroid = wanted.centroid or geometry.centroid_of(wanted.boundary)
        existing.centroid = geometry.position_to_wkb(centroid)
        existing.centroid_height = centroid.height
        changed.append("boundary")
    # The seed's own assets only: an epoch's asset is a capture's, whatever it is called.
    by_name = {asset.name: asset for asset in existing.assets if asset.epoch_id is None}
    for payload in wanted.assets:
        built = build_asset(payload, site_id=existing.id)
        asset = by_name.get(payload.name)
        if asset is None:
            existing.assets.append(built)
            changed.append(f"added {payload.name}")
            continue
        for field in ("source", "render_config", "resolution", "attribution", "license"):
            if getattr(asset, field) != getattr(built, field):
                setattr(asset, field, getattr(built, field))
                changed.append(f"{field} of {payload.name}")
    if changed:
        db.commit()
    return changed


def seed_scans(db: Session, sites: list[SiteCreate] | None = None) -> dict[str, int]:
    """Registers the snapshot's scans: creates the missing, brings this seed's own in step,
    and leaves a site with the same slug that something else made alone."""
    counts = {"created": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    for wanted in load_snapshot() if sites is None else sites:
        existing = db.scalar(select(Site).where(Site.slug == wanted.slug))
        if existing is None:
            site_service.create_site(db, wanted)
            counts["created"] += 1
            logger.info("registered %s (%s)", wanted.name, wanted.slug)
        elif (existing.metadata_ or {}).get("origin") != ORIGIN:
            counts["skipped"] += 1
            logger.warning(
                "left %s alone: a site with that slug exists and this seed did not make it",
                wanted.slug,
            )
        elif changed := _bring_in_step(db, existing, wanted):
            counts["updated"] += 1
            logger.info("brought %s in step: %s", wanted.slug, ", ".join(changed))
        else:
            counts["unchanged"] += 1
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.seed.scans",
        description="Register the hosted catalog's phone scans (camp, pumpkin, spool) here.",
    )
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help=f"re-read the scans' records from {HOSTED_API} into {SNAPSHOT.name}; no DB writes",
    )
    parser.add_argument("--api", default=HOSTED_API, help="the hosted API to read from")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.refresh_snapshot:
        count = refresh_snapshot(api=args.api)
        sys.stdout.write(f"snapshot rewritten: {count} scans in {SNAPSHOT}\n")
        return
    from app.db import get_session_factory

    with get_session_factory()() as session:
        result = seed_scans(session)
    sys.stdout.write(f"scans seeded: {result}\n")


if __name__ == "__main__":
    main()

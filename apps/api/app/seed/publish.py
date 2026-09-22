"""Move a capture's tiles into object storage, and publish the catalog the console
falls back to when this API is unreachable.

Two jobs, one command, because they are two halves of the same fact: **the bytes a
browser needs are not served by this API any more.**

*Tiles.* `publish_tiles` copies `data/tiles/<slug>/**` to `sites/<slug>/**` in the
bucket. This is the one-shot migration A9 exists for -- the three drone captures were
104 MB of committed 3D Tiles, and they are gone from the working tree. It is also what a
developer runs after `build_site.py`, and what a deployment runs for the one capture that
*stays* in git: `synthetic-tree` is a CI fixture with a byte-identity gate on it, so it
keeps its place in the repository, but the container image has no `data/tiles` and serves
it from the bucket like everything else.

*Catalog.* `publish_catalog` writes `catalog.json` beside them: the sites, exactly as
`GET /api/v1/sites/{id}` would return them. Before A9 an unreachable API meant unreachable
tiles as well -- they were a `StaticFiles` mount *on that API* -- so a local capture in the
web app's offline catalog would have been a dead link, which is why `fallback.ts` only ever
held Cesium ion asset ids. Now the tiles are on a different host that stays up when this
one does not, so an offline capture is reachable for the first time; this file is how the
console learns which ones there are without a table compiled into its bundle.

    uv run python -m app.seed.publish                  # every capture on disk, + catalog
    uv run python -m app.seed.publish --slug mygla     # one capture
    uv run python -m app.seed.publish --catalog-only   # after registering a pipeline run
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_session_factory
from app.seed.captures import STORAGE_PREFIX, capture_descriptions, tiles_root
from app.services import sites as site_service
from app.storage import ObjectStorage
from app.storage.factory import build_storage

logger = logging.getLogger("twin.seed.publish")

#: The offline catalog's key, at the bucket root beside `sites/`.
CATALOG_KEY = "catalog.json"

#: Content types the browser and CesiumJS care about that `mimetypes` does not know.
_TILE_TYPES = {
    ".b3dm": "application/octet-stream",
    ".pnts": "application/octet-stream",
    ".glb": "model/gltf-binary",
    ".ply": "application/octet-stream",
    ".f32": "application/octet-stream",
    ".spz": "application/octet-stream",
}


@dataclass(frozen=True)
class PublishReport:
    slug: str
    files: int
    bytes: int


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _TILE_TYPES:
        return _TILE_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def tile_key(slug: str, relative: str) -> str:
    return f"{STORAGE_PREFIX}/{slug}/{relative}"


def publish_tiles(storage: ObjectStorage, source: Path, slug: str) -> PublishReport:
    """Upload every file under `source` to `sites/<slug>/`, keeping the layout.

    The layout is load-bearing rather than incidental: a `tileset.json` refers to its
    tiles by relative URI, and a rig is `../source/rig.json` from the tileset. Flattening
    or renaming anything here would break both without any error to read.
    """
    files = 0
    total = 0
    for member in sorted(p for p in source.rglob("*") if p.is_file()):
        data = member.read_bytes()
        storage.put_object(
            tile_key(slug, member.relative_to(source).as_posix()),
            data,
            content_type_for(member),
        )
        files += 1
        total += len(data)
    return PublishReport(slug=slug, files=files, bytes=total)


def catalog_document(db: Session) -> dict[str, object]:
    """Every site, in the shape `GET /api/v1/sites/{id}` returns.

    Same schema as the live API on purpose: the console parses it with the generated
    contract types and nothing has to agree with anything by hand.
    """
    sites = [site_service.site_to_read(db, site) for site in site_service.list_sites(db)]
    return {
        "version": 1,
        "sites": [site.model_dump(mode="json", by_alias=True) for site in sites],
    }


def publish_catalog(db: Session, storage: ObjectStorage) -> int:
    document = catalog_document(db)
    payload = json.dumps(document, indent=1, sort_keys=True).encode("utf-8")
    storage.put_object(CATALOG_KEY, payload, "application/json")
    sites = document["sites"]
    return len(sites) if isinstance(sites, list) else 0


def publishable_slugs(settings: Settings, only: str | None) -> list[str]:
    """Captures this checkout can publish: a description *and* tiles on disk.

    A capture whose bytes were migrated out of the working tree is not publishable from
    here any more, and saying so is better than uploading an empty prefix over a good one.
    """
    root = tiles_root(settings)
    described = [capture.slug for capture in capture_descriptions(settings)]
    if only is not None:
        described = [slug for slug in described if slug == only]
        if not described:
            raise SystemExit(f"no capture called {only!r} is described by this checkout")
    return [slug for slug in described if (root / slug).is_dir()]


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", help="publish one capture rather than every one on disk")
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="skip the tiles and refresh catalog.json from the database",
    )
    parser.add_argument(
        "--no-catalog", action="store_true", help="upload tiles without touching catalog.json"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    storage = build_storage(settings)
    if not storage.available:
        parser.error(
            "object storage is not configured: set OBJECT_STORAGE_* in .env (see .env.example). "
            "Tiles cannot be published to a bucket that does not exist."
        )

    if not args.catalog_only:
        root = tiles_root(settings)
        slugs = publishable_slugs(settings, args.slug)
        if not slugs:
            logger.warning("no capture tiles found under %s", root)
        for slug in slugs:
            report = publish_tiles(storage, root / slug, slug)
            logger.info(
                "published %s: %d files, %.1f MiB -> %s",
                report.slug,
                report.files,
                report.bytes / 1048576,
                tile_key(slug, ""),
            )

    if not args.no_catalog:
        with get_session_factory()() as session:
            count = publish_catalog(session, storage)
        logger.info("published %s with %d sites", CATALOG_KEY, count)
    return 0


if __name__ == "__main__":
    sys.exit(_main())

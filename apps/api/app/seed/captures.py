"""Capture sites that predate the pipeline, and where their tiles are served from.

**`captures.json` is no longer the interface.** Until A9 one mutable manifest at
`data/tiles/captures.json` was written by `tools/captures/build_site.py`, hand-edited
afterwards, committed beside 104 MB of tiles, and read by the seeder. Since A8 a capture
becomes a site by being *registered* -- the pipeline's `catalog` stage writes
`registration.json` and the worker turns it into rows (`app/worker/registration.py`) --
so nothing writes a shared manifest any more.

What is left is two things, and they are both one-shot:

* **`legacy_captures.json`**, beside this file. The four captures that existed before the
  pipeline, frozen exactly as the old manifest described them, because their provenance
  (attribution, licence, capture date, image count, estimated GSD) is real and was not
  reconstructible from the tiles. It is an archive: nothing writes it, and a fifth
  capture does not go in it.
* **`site.json`**, which `build_site.py` now writes into the site folder it builds. One
  document per capture, never shared, never hand-merged -- so a locally built capture
  still seeds on a developer's machine without a bucket. `publish.py` is what moves it
  and its tiles into object storage.

Both are the same `Capture` shape, which is why the models below are still the schema of
record for a capture *description*. The schema of record for a capture *site* is the
database.

Where the tiles are actually served from is `tiles_base_url()`: **a capture is served
from the checkout when the checkout has it, and from the bucket otherwise.** The
`StaticFiles` mount is development only -- the container image does not contain
`data/tiles` and never did, which is why every capture 404'd in it until A9.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import REPO_ROOT, Settings
from app.models.enums import Representation
from app.schemas.asset import AssetBase, RenderConfig, ResolutionMetadata, TilesUrlSource
from app.schemas.bookmark import CameraBookmarkCreate
from app.schemas.common import Attribution, GeoPosition, LicenseMetadata, Provenance
from app.schemas.geojson import Polygon
from app.schemas.site import SiteCreate
from app.storage.factory import build_public_storage

logger = logging.getLogger("twin.seed.captures")

#: The frozen pre-pipeline archive, beside this module rather than under `data/tiles`:
#: the tiles it describes are not in the repository any more, so a path into the checkout
#: would have been a path to nothing.
ARCHIVE = Path(__file__).with_name("legacy_captures.json")

#: What `build_site.py` writes into each site folder it builds.
SITE_DOCUMENT_NAME = "site.json"

#: Key prefix for published capture tiles in the bucket: `sites/<slug>/splat/tileset.json`.
#: Deliberately not `runs/<job id>/...`, which is where a pipeline run's own artifacts go
#: (`app/worker/outputs.py`) -- a capture migrated out of git never had a run.
STORAGE_PREFIX = "sites"


class CaptureAsset(BaseModel):
    representation: Representation
    name: str
    path: str = Field(description="Relative to the site folder, e.g. mesh/tileset.json")
    ground_sample_distance_m: float | None = None
    point_spacing_m: float | None = None
    description: str | None = None
    maximum_screen_space_error: float | None = None
    screen_space_error_scale: float = 1.0
    # A capture whose own ellipsoid height is not trustworthy (a procedural fixture, a phone
    # scan) rests on whatever ground the viewer has instead.
    clamp_to_ground: bool = False
    height_offset_m: float = 0.0
    default: bool = False
    # A Living Survey motion rig for this asset, relative to its tileset URL. This is the
    # claim that used to live in LIVING_RIGS in the web bundle; see RenderConfig.rig_url.
    rig: str | None = None


class Capture(BaseModel):
    slug: str
    name: str
    description: str = ""
    boundary: list[list[float]] = Field(description="Closed ring of [lon, lat]")
    center: list[float] = Field(description="[lon, lat, height]")
    attribution: str
    attribution_url: str | None = None
    license_name: str
    license_url: str | None = None
    source_url: str | None = None
    captured: str | None = None
    pipeline: str = ""
    images: int | None = None
    assets: list[CaptureAsset]
    bookmark: dict[str, Any] | None = None


def tiles_root(settings: Settings) -> Path:
    path = Path(settings.tiles_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def tiles_base_url(settings: Settings, slug: str) -> str:
    """The URL prefix a capture's tiles are fetched from, ending in a slash.

    One rule: **a capture is served from the checkout when the checkout has it, and from
    the bucket otherwise.** Production never reads the checkout, because it does not have
    one -- infra/api.Dockerfile copies `apps/api/` and nothing else, so `data/tiles` is
    absent there and the `StaticFiles` mount is disabled outright (app/main.py). That is
    why every local capture 404'd in the documented deployment path before A9, and why it
    cannot again: production's answer is now the bucket, not a directory.

    In order:

    1. `TILES_BASE_URL` set wins everywhere. It is how you put a CDN in front of the
       bucket -- R2 presigns only on its S3 API domain while public reads come from the
       custom domain, so those are two different URLs and this is the one a browser uses.
    2. Development with the tiles on disk -> this API's `StaticFiles` mount over
       `tiles_dir`. A clone needs no bucket and no publish step to see `synthetic-tree`
       move, and a capture just built by `build_site.py` is visible before it is uploaded.
    3. Object storage configured -> its public URL under `sites/`, where
       `python -m app.seed.publish` puts them. Always the answer in production, and the
       answer in development for the three drone captures whose 104 MB left git in A9.
    4. Nothing else available -> the static mount anyway, which is a 404 waiting to
       happen. In production that is a misconfiguration and it is logged as one.
    """
    if settings.tiles_base_url:
        return f"{settings.tiles_base_url.rstrip('/')}/{slug}/"
    local = f"{settings.public_api_base.rstrip('/')}/api/v1/tiles/{slug}/"
    if not settings.is_production and (tiles_root(settings) / slug).is_dir():
        return local
    # The public bucket where there are two: `sites/` is what a browser fetches.
    storage = build_public_storage(settings)
    if storage.available:
        return storage.public_url(f"{STORAGE_PREFIX}/{slug}/")
    if settings.is_production:
        logger.error(
            "production has neither TILES_BASE_URL nor object storage configured: capture "
            "%s will be seeded against a static mount this image does not have",
            slug,
        )
    return local


def load_archive() -> list[Capture]:
    """The four pre-pipeline captures, frozen. Nothing writes this file."""
    return _load_document(ARCHIVE)


def load_site_documents(settings: Settings) -> list[Capture]:
    """`data/tiles/<slug>/site.json` for every capture built locally by `build_site.py`.

    A per-capture document rather than a shared manifest: two builds cannot race on it,
    and there is nothing to merge by hand.
    """
    root = tiles_root(settings)
    if not root.is_dir():
        return []
    captures: list[Capture] = []
    for document in sorted(root.glob(f"*/{SITE_DOCUMENT_NAME}")):
        captures.extend(_load_document(document))
    return captures


def _load_document(path: Path) -> list[Capture]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("capture document %s could not be read", path)
        return []
    entries = raw.get("captures", [raw]) if isinstance(raw, dict) else []
    captures: list[Capture] = []
    for entry in entries:
        try:
            captures.append(Capture.model_validate(entry))
        except ValueError:
            logger.exception("capture document %s has an entry this API cannot read", path)
    return captures


def site_from_capture(capture: Capture, settings: Settings) -> SiteCreate:
    base = tiles_base_url(settings, capture.slug)
    ring = [[float(x), float(y)] for x, y in capture.boundary]
    boundary = Polygon(coordinates=[ring])
    attribution = Attribution(
        text=capture.attribution, organization=capture.attribution, url=capture.attribution_url
    )
    license_ = LicenseMetadata(
        name=capture.license_name, url=capture.license_url, requires_attribution=True
    )
    assets: list[AssetBase] = []
    for item in capture.assets:
        assets.append(
            AssetBase(
                name=item.name,
                representation=item.representation,
                source=TilesUrlSource(url=base + item.path),
                footprint=boundary,
                resolution=ResolutionMetadata(
                    ground_sample_distance_m=item.ground_sample_distance_m,
                    point_spacing_m=item.point_spacing_m,
                    description=item.description,
                ),
                attribution=[attribution],
                license=license_,
                provenance=Provenance(
                    source_organization=capture.attribution,
                    source_url=capture.source_url,
                    notes=capture.pipeline,
                ),
                render_config=RenderConfig(
                    maximum_screen_space_error=item.maximum_screen_space_error,
                    screen_space_error_scale=item.screen_space_error_scale,
                    clips_world=True,
                    clip_footprint="catalog",
                    clamp_to_ground=item.clamp_to_ground,
                    height_offset_m=item.height_offset_m,
                    rig_url=item.rig,
                ),
                default_visible=item.default,
            )
        )
    lon, lat, height = capture.center[0], capture.center[1], capture.center[2]
    bookmarks: list[CameraBookmarkCreate] = []
    if capture.bookmark:
        bookmarks.append(CameraBookmarkCreate.model_validate(capture.bookmark))
    return SiteCreate(
        slug=capture.slug,
        name=capture.name,
        description=capture.description,
        boundary=boundary,
        centroid=GeoPosition(longitude=lon, latitude=lat, height=height),
        metadata={
            # Load-bearing: `app/seed/_refresh` only rebuilds a site whose origin is this
            # string, because a capture is rebuilt whole by its tooling while a user's own
            # site is not.
            "origin": "capture-pipeline",
            "pipeline": capture.pipeline,
            "captured": capture.captured,
            "images": capture.images,
            "quality": {
                "resolutionDescription": next(
                    (a.description for a in capture.assets if a.description), None
                )
            },
        },
        attribution=[attribution],
        license=license_,
        assets=assets,
        camera_bookmarks=bookmarks,
    )


def capture_descriptions(settings: Settings) -> list[Capture]:
    """The archive, plus anything built locally. A local build of a slug wins."""
    by_slug: dict[str, Capture] = {capture.slug: capture for capture in load_archive()}
    for capture in load_site_documents(settings):
        by_slug[capture.slug] = capture
    return [by_slug[slug] for slug in sorted(by_slug)]


def capture_sites(settings: Settings) -> list[SiteCreate]:
    sites: list[SiteCreate] = []
    for capture in capture_descriptions(settings):
        try:
            sites.append(site_from_capture(capture, settings))
        except Exception:
            logger.exception("capture %s skipped", capture.slug)
    return sites

"""Sites built from processed captures on disk.

A pipeline (tools/captures) writes `data/tiles/<slug>/` with one 3D Tiles tileset per
representation and a `captures.json` manifest beside them. Each manifest entry becomes a
site with a mesh, a point cloud and a Gaussian splat the console can switch between. The
tilesets are served by the API itself (see `create_app`), so the URLs point back at it.
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

logger = logging.getLogger("twin.seed.captures")

MANIFEST_NAME = "captures.json"


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


def load_manifest(settings: Settings) -> list[Capture]:
    manifest = tiles_root(settings) / MANIFEST_NAME
    if not manifest.is_file():
        return []
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    return [Capture.model_validate(entry) for entry in raw.get("captures", [])]


def site_from_capture(capture: Capture, settings: Settings) -> SiteCreate:
    base = settings.public_api_base.rstrip("/") + "/api/v1/tiles/" + capture.slug + "/"
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


def capture_sites(settings: Settings) -> list[SiteCreate]:
    sites: list[SiteCreate] = []
    for capture in load_manifest(settings):
        try:
            sites.append(site_from_capture(capture, settings))
        except Exception:
            logger.exception("capture %s skipped", capture.slug)
    return sites

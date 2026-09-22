"""Capture sites: the frozen archive, locally built `site.json`, and where tiles come from."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
import pytest
from moto import mock_aws
from sqlalchemy.orm import Session

from app.config import Settings, repo_root_for
from app.main import create_app
from app.schemas.asset import TilesUrlSource
from app.seed import _refresh as refresh_site
from app.seed.captures import (
    SITE_DOCUMENT_NAME,
    capture_descriptions,
    capture_sites,
    load_archive,
    tiles_base_url,
)
from app.seed.publish import CATALOG_KEY, publish_catalog, publish_tiles, tile_key
from app.services import sites as site_service
from app.storage import S3Storage

SLUG = "test-field"
CAPTURE: dict[str, Any] = {
    "slug": SLUG,
    "name": "Test field",
    "boundary": [[-91.0, 46.0], [-90.999, 46.0], [-90.999, 46.001], [-91.0, 46.0]],
    "center": [-90.9995, 46.0005, 152.6],
    "attribution": "Someone",
    "license_name": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "source_url": "https://example.org/dataset",
    "captured": "2016",
    "pipeline": "OpenDroneMap and OpenSplat",
    "images": 18,
    "assets": [
        {
            "representation": "mesh",
            "name": "Textured mesh",
            "path": "mesh/tileset.json",
            "ground_sample_distance_m": 0.02,
            "description": "about 2.0 cm per pixel, estimated",
            "maximum_screen_space_error": 2,
            "default": True,
        },
        {
            "representation": "point-cloud",
            "name": "Point cloud",
            "path": "pointcloud/tileset.json",
            "point_spacing_m": 0.13,
        },
        {
            "representation": "gaussian-splat",
            "name": "Splat",
            "path": "splat/tileset.json",
            "rig": "../source/rig.json",
        },
    ],
}

BUCKET = "twin-assets"


#: Storage left unconfigured unless a test asks for it. The repository's own .env points at
#: MinIO, and a test that inherited it would assert on a developer's machine rather than on
#: this code -- and would then pass here and fail in CI, which has no .env.
NO_STORAGE: dict[str, object] = {
    "object_storage_endpoint_url": None,
    "object_storage_bucket": None,
    "object_storage_access_key": None,
    "object_storage_secret_key": None,
    "object_storage_public_url": None,
    "tiles_base_url": None,
}


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    site_dir = tmp_path / SLUG
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / SITE_DOCUMENT_NAME).write_text(json.dumps(CAPTURE), encoding="utf-8")
    return Settings(
        tiles_dir=str(tmp_path),
        public_api_base="http://api.test:8000/",
        **{**NO_STORAGE, **overrides},  # type: ignore[arg-type]
    )


def bucket_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Settings with a bucket, every field explicit (see NO_STORAGE)."""
    return settings_for(
        tmp_path,
        object_storage_endpoint_url="https://s3.example.com",
        object_storage_bucket=BUCKET,
        object_storage_access_key="key",
        object_storage_secret_key="secret",
        object_storage_region="us-east-1",
        **overrides,
    )


def production_settings(tmp_path: Path, **overrides: object) -> Settings:
    return bucket_settings(
        tmp_path,
        # APP_ENV, not `environment`: the field is aliased and pydantic-settings takes the
        # alias on init, so `environment="production"` is silently dropped by extra="ignore".
        APP_ENV="production",
        api_write_token="token",
        **overrides,
    )


# --- the archive ------------------------------------------------------------------


def test_the_four_pre_pipeline_captures_survive_the_move_with_their_provenance() -> None:
    """The point of keeping an archive at all: attribution, licence and capture date."""
    archive = {capture.slug: capture for capture in load_archive()}
    assert sorted(archive) == ["brighton-beach", "mygla", "sheffield-park", "synthetic-tree"]
    sheffield = archive["sheffield-park"]
    assert sheffield.attribution == "Piero Toffanin / OpenDroneMap"
    assert sheffield.license_name == "BSD-2-Clause"
    assert sheffield.captured == "2016"
    assert sheffield.images == 32
    assert [asset.representation.value for asset in sheffield.assets] == [
        "mesh",
        "point-cloud",
        "gaussian-splat",
    ]
    # The fixture that stays in git is still a catalog entry, and it is the one that moves.
    tree = archive["synthetic-tree"]
    assert [asset.rig for asset in tree.assets] == ["../source/rig.json"]


def test_a_local_build_wins_over_the_archive(tmp_path: Path) -> None:
    """`build_site.py` rebuilt a slug the archive also names: the rebuild is the truth."""
    site_dir = tmp_path / "mygla"
    site_dir.mkdir()
    rebuilt = dict(CAPTURE, slug="mygla", name="Rebuilt mygla")
    (site_dir / SITE_DOCUMENT_NAME).write_text(json.dumps(rebuilt), encoding="utf-8")
    by_slug = {
        capture.slug: capture for capture in capture_descriptions(Settings(tiles_dir=str(tmp_path)))
    }
    assert by_slug["mygla"].name == "Rebuilt mygla"
    assert "sheffield-park" in by_slug


# --- where the tiles come from ----------------------------------------------------


def test_development_serves_a_capture_it_has_on_disk(tmp_path: Path) -> None:
    """A bucket being configured does not move the tiles out from under a local clone."""
    assert tiles_base_url(bucket_settings(tmp_path), SLUG) == (
        f"http://api.test:8000/api/v1/tiles/{SLUG}/"
    )


def test_development_falls_back_to_the_bucket_for_a_capture_it_does_not_have(
    tmp_path: Path,
) -> None:
    """The three drone captures after A9: described here, but 104 MB of tiles are not.

    Without this a fresh clone would seed three sites whose every asset 404s, which is a
    worse answer than fetching them from the bucket they were migrated to.
    """
    assert tiles_base_url(bucket_settings(tmp_path), "mygla") == (
        "https://s3.example.com/twin-assets/sites/mygla/"
    )


def test_with_no_disk_and_no_bucket_it_says_where_it_would_have_looked(tmp_path: Path) -> None:
    """A clone with no bucket at all. `mygla` 404s, and that is the honest answer: its
    bytes are in git history and nowhere else this deployment can reach."""
    assert tiles_base_url(settings_for(tmp_path), "mygla") == (
        "http://api.test:8000/api/v1/tiles/mygla/"
    )


def test_production_never_reads_the_checkout_even_when_there_is_one(tmp_path: Path) -> None:
    """`settings_for` puts SLUG's tiles on disk; production must ignore them.

    An image that happened to carry a stale `data/tiles` would otherwise serve it, which is
    the failure this step exists to remove rather than to move around.
    """
    settings = production_settings(tmp_path)
    assert tiles_base_url(settings, SLUG) == f"https://s3.example.com/twin-assets/sites/{SLUG}/"
    assert tiles_base_url(settings, "mygla") == "https://s3.example.com/twin-assets/sites/mygla/"


def test_an_explicit_base_wins_over_both(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, tiles_base_url="https://tiles.example.com/v1/")
    assert tiles_base_url(settings, "mygla") == "https://tiles.example.com/v1/mygla/"


# --- seeding ----------------------------------------------------------------------


def test_a_site_document_becomes_a_site_served_by_the_api(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    site = next(s for s in capture_sites(settings) if s.slug == "test-field")
    assert site.centroid is not None and site.centroid.height == 152.6
    assert [a.representation.value for a in site.assets] == [
        "mesh",
        "point-cloud",
        "gaussian-splat",
    ]
    mesh, points, splat = site.assets
    assert isinstance(mesh.source, TilesUrlSource)
    assert str(mesh.source.url) == "http://api.test:8000/api/v1/tiles/test-field/mesh/tileset.json"
    assert mesh.default_visible is True and points.default_visible is False
    assert mesh.render_config.clips_world is True
    assert mesh.resolution is not None and mesh.resolution.ground_sample_distance_m == 0.02
    assert points.resolution is not None and points.resolution.point_spacing_m == 0.13
    assert site.license is not None and site.license.name == "CC-BY-4.0"
    assert site.attribution[0].text == "Someone"
    assert site.metadata["origin"] == "capture-pipeline" and site.metadata["images"] == 18
    # The rig is catalog data now, not a table in the web bundle.
    assert splat.render_config.rig_url == "../source/rig.json"
    assert mesh.render_config.rig_url is None


def test_a_production_capture_is_seeded_pointing_at_the_bucket(tmp_path: Path) -> None:
    """The Dockerfile defect, as a test.

    infra/api.Dockerfile copies `apps/api/` and nothing else, so `data/tiles` is absent in
    the container and `/api/v1/tiles/...` 404s there. This asserts that no seeded capture
    URL points at it in production -- which is what makes that failure mode gone rather
    than patched.
    """
    site = next(s for s in capture_sites(production_settings(tmp_path)) if s.slug == "test-field")
    urls = [str(a.source.url) for a in site.assets if isinstance(a.source, TilesUrlSource)]
    assert urls and all(
        url.startswith("https://s3.example.com/twin-assets/sites/test-field/") for url in urls
    )
    assert not any("/api/v1/tiles/" in url for url in urls)


def test_every_archived_capture_resolves_in_production(tmp_path: Path) -> None:
    """Including `synthetic-tree`, which stays in git but is not in the image."""
    archived = {"brighton-beach", "mygla", "sheffield-park", "synthetic-tree"}
    sites = [s for s in capture_sites(production_settings(tmp_path)) if s.slug in archived]
    urls = [
        str(a.source.url)
        for site in sites
        for a in site.assets
        if isinstance(a.source, TilesUrlSource)
    ]
    assert len(urls) == 10
    assert all(url.startswith("https://s3.example.com/twin-assets/sites/") for url in urls)


def test_an_unreadable_site_document_is_skipped_not_fatal(tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / SITE_DOCUMENT_NAME).write_text("{not json", encoding="utf-8")
    slugs = [capture.slug for capture in capture_descriptions(Settings(tiles_dir=str(tmp_path)))]
    assert "broken" not in slugs
    assert "sheffield-park" in slugs


def test_missing_tiles_dir_still_seeds_the_archive(tmp_path: Path) -> None:
    slugs = [s.slug for s in capture_sites(Settings(tiles_dir=str(tmp_path / "nothing")))]
    assert slugs == ["brighton-beach", "mygla", "sheffield-park", "synthetic-tree"]


def test_rebuilt_capture_refreshes_the_seeded_site(tmp_path: Path, db: Session) -> None:
    settings = settings_for(tmp_path)
    first = next(s for s in capture_sites(settings) if s.slug == "test-field")
    first.assets = first.assets[:1]
    created = site_service.create_site(db, first)
    assert len(created.assets) == 1

    grown = json.loads(json.dumps(CAPTURE))
    grown["assets"][0]["ground_sample_distance_m"] = 0.027
    grown["images"] = 32
    (tmp_path / "test-field" / SITE_DOCUMENT_NAME).write_text(json.dumps(grown), encoding="utf-8")
    wanted = next(s for s in capture_sites(settings) if s.slug == "test-field")
    refresh_site(db, created, wanted)

    db.refresh(created)
    assert [a.representation.value for a in created.assets] == [
        "mesh",
        "point-cloud",
        "gaussian-splat",
    ]
    assert created.assets[0].resolution is not None
    assert created.assets[0].resolution["groundSampleDistanceM"] == 0.027
    assert created.metadata_["images"] == 32


# --- the static mount -------------------------------------------------------------


def test_production_does_not_mount_local_tiles(tmp_path: Path) -> None:
    mounts = {
        getattr(route, "name", None) for route in create_app(production_settings(tmp_path)).routes
    }
    assert "tiles" not in mounts


def test_development_mounts_local_tiles(tmp_path: Path) -> None:
    mounts = {getattr(route, "name", None) for route in create_app(settings_for(tmp_path)).routes}
    assert "tiles" in mounts


# --- publishing -------------------------------------------------------------------


@pytest.fixture
def moto_storage() -> Iterator[S3Storage]:
    """A bucket that exists, with the API's own S3Storage in front of it.

    moto does not enforce auth (A1 measured it returning 200 for an unsigned PUT), so this
    proves layout and content types, never authorisation. CI's MinIO service container is
    where signed requests meet a real server.

    `endpoint_url=None` for the same reason tests/test_storage_objects.py gives: moto
    intercepts the AWS hostnames, and a custom endpoint host makes botocore open a real
    connection instead.
    """
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield S3Storage(
            bucket=BUCKET,
            endpoint_url=None,
            access_key="key",
            secret_key="secret",
            region="us-east-1",
            public_base_url="https://s3.example.com/twin-assets",
        )


def test_publishing_a_capture_keeps_its_layout(tmp_path: Path, moto_storage: S3Storage) -> None:
    source = tmp_path / "source-tiles"
    (source / "splat").mkdir(parents=True)
    (source / "source").mkdir(parents=True)
    (source / "splat" / "tileset.json").write_text('{"asset":{"version":"1.1"}}', encoding="utf-8")
    (source / "splat" / "splat.glb").write_bytes(b"glb")
    (source / "source" / "rig.json").write_text("{}", encoding="utf-8")

    report = publish_tiles(moto_storage, source, "test-field")
    assert report.files == 3

    # The layout is what a tileset.json's relative URIs and a `../source/rig.json` rig both
    # depend on: flattening it would break both with nothing to read in a log.
    assert moto_storage.get_object(tile_key("test-field", "splat/tileset.json")).startswith(b"{")
    assert moto_storage.get_object(tile_key("test-field", "source/rig.json")) == b"{}"
    head = moto_storage.head_object(tile_key("test-field", "splat/tileset.json"))
    assert head is not None and head.content_type == "application/json"
    glb = moto_storage.head_object(tile_key("test-field", "splat/splat.glb"))
    assert glb is not None and glb.content_type == "model/gltf-binary"


def test_the_published_catalog_is_what_the_console_falls_back_to(
    tmp_path: Path, db: Session, moto_storage: S3Storage
) -> None:
    site = next(s for s in capture_sites(production_settings(tmp_path)) if s.slug == "test-field")
    site_service.create_site(db, site)

    assert publish_catalog(db, moto_storage) == 1
    document = json.loads(moto_storage.get_object(CATALOG_KEY))
    [published] = document["sites"]
    assert published["slug"] == "test-field"
    # Camel case, exactly as GET /api/v1/sites/{id} returns it, so the console parses it
    # with the generated contract types rather than a second hand-written shape.
    assert published["assets"][0]["renderConfig"]["clipsWorld"] is True
    rigs = [asset["renderConfig"]["rigUrl"] for asset in published["assets"]]
    assert rigs == [None, None, "../source/rig.json"]
    # And it points at the bucket, which is what makes it reachable while the API is not.
    assert published["assets"][0]["source"]["url"].startswith("https://s3.example.com/")


# --- the container layout ---------------------------------------------------------


def test_the_container_layout_does_not_crash_on_import() -> None:
    """infra/api.Dockerfile is `WORKDIR /app` + `COPY apps/api/ ./`.

    That puts this package at `/app/app/config.py`, which has three parents, so the
    original `parents[3]` raised `IndexError: 3` while importing `app.config` -- the image
    could never start at all. Reproduced against the real layout before this was changed.
    """
    assert repo_root_for(PurePosixPath("/app/app/config.py")) == Path("/app")
    assert repo_root_for(PurePosixPath("/src/twin/apps/api/app/config.py")) == Path("/src/twin")

"""Sites seeded from the capture manifest (tools/captures output)."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.schemas.asset import TilesUrlSource
from app.seed.captures import capture_sites, load_manifest

MANIFEST = {
    "captures": [
        {
            "slug": "test-field",
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
                },
            ],
        }
    ]
}


def settings_for(tmp_path: Path) -> Settings:
    (tmp_path / "captures.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    return Settings(tiles_dir=str(tmp_path), public_api_base="http://api.test:8000/")


def test_manifest_becomes_a_site_served_by_the_api(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    assert [c.slug for c in load_manifest(settings)] == ["test-field"]
    [site] = capture_sites(settings)
    assert site.slug == "test-field"
    assert site.centroid is not None and site.centroid.height == 152.6
    assert [a.representation.value for a in site.assets] == [
        "mesh",
        "point-cloud",
        "gaussian-splat",
    ]
    mesh, points = site.assets[0], site.assets[1]
    assert isinstance(mesh.source, TilesUrlSource)
    assert str(mesh.source.url) == "http://api.test:8000/api/v1/tiles/test-field/mesh/tileset.json"
    assert mesh.default_visible is True and points.default_visible is False
    assert mesh.render_config.clips_world is True
    assert mesh.resolution is not None and mesh.resolution.ground_sample_distance_m == 0.02
    assert points.resolution is not None and points.resolution.point_spacing_m == 0.13
    assert site.license is not None and site.license.name == "CC-BY-4.0"
    assert site.attribution[0].text == "Someone"
    assert site.metadata["origin"] == "capture-pipeline" and site.metadata["images"] == 18


def test_missing_manifest_seeds_nothing(tmp_path: Path) -> None:
    assert capture_sites(Settings(tiles_dir=str(tmp_path))) == []

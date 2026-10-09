from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyogrio.raw import write
from pyproj import Transformer
from shapely.geometry import LinearRing, Polygon, mapping, shape
from shapely.ops import transform

from app.services.errors import InvalidInputError
from app.services.land_import import import_boundary

POLYGON = Polygon(
    [(2, 48), (2.01, 48), (2.01, 48.01), (2, 48.01), (2, 48)],
    [[(2.002, 48.002), (2.004, 48.002), (2.004, 48.004), (2.002, 48.002)]],
)


def archive(files: Mapping[str, str | bytes]) -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as target:
        for name, content in files.items():
            target.writestr(name, content)
    return result.getvalue()


def test_projected_geojson_preserves_exclusion() -> None:
    projected = transform(Transformer.from_crs(4326, 3857, always_xy=True).transform, POLYGON)
    result = import_boundary(json.dumps(mapping(projected)).encode(), "land.geojson", "EPSG:3857")
    assert result.status == "ready" and result.boundary is not None
    assert shape(result.boundary.model_dump()).symmetric_difference(POLYGON).area < 1e-12
    assert result.warnings and result.source_crs == "EPSG:3857"


def test_kml_kmz_preserve_holes_and_reject_external_entities() -> None:
    def coordinates(ring: LinearRing) -> str:
        return " ".join(f"{x},{y},0" for x, y in ring.coords)

    kml = f"""<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><Polygon>
    <outerBoundaryIs><LinearRing><coordinates>{coordinates(POLYGON.exterior)}</coordinates></LinearRing></outerBoundaryIs>
    <innerBoundaryIs><LinearRing><coordinates>{coordinates(POLYGON.interiors[0])}</coordinates></LinearRing></innerBoundaryIs>
    </Polygon></Placemark></kml>""".encode()
    for data, name in [(kml, "land.kml"), (archive({"doc.kml": kml}), "land.kmz")]:
        result = import_boundary(data, name)
        assert result.status == "ready" and result.boundary is not None
        assert shape(result.boundary.model_dump()).equals(POLYGON)
    with pytest.raises(InvalidInputError):
        import_boundary(
            b'<!DOCTYPE a [<!ENTITY x SYSTEM "file:///etc/passwd">]><a>&x;</a>', "a.kml"
        )
    with pytest.raises(InvalidInputError, match="paths"):
        import_boundary(archive({"../doc.kml": kml}), "land.kmz")


def test_repair_requires_explicit_preview() -> None:
    invalid = b'{"type":"Polygon","coordinates":[[[0,0],[1,1],[0,1],[1,0],[0,0]]]}'
    assert import_boundary(invalid, "land.json").status == "needs-repair"
    repaired = import_boundary(invalid, "land.json", repair=True)
    assert repaired.status == "ready" and repaired.warnings and repaired.boundary is not None
    assert shape(repaired.boundary.model_dump()).is_valid


def test_shapefile_requires_crs_and_reprojects(tmp_path: Path) -> None:
    projected = transform(Transformer.from_crs(4326, 3857, always_xy=True).transform, POLYGON)
    path = tmp_path / "land.shp"
    write(
        path,
        np.array([projected.wkb], dtype=object),
        [],
        [],
        driver="ESRI Shapefile",
        geometry_type="Polygon",
        crs="EPSG:3857",
    )
    files = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = import_boundary(archive(files), "land.zip")
    assert result.status == "ready" and result.boundary is not None
    assert shape(result.boundary.model_dump()).symmetric_difference(POLYGON).area < 1e-12
    files.pop("land.prj")
    assert import_boundary(archive(files), "land.zip").status == "needs-crs"
    assert import_boundary(archive(files), "land.zip", "EPSG:3857").status == "ready"


def test_geopackage_layer_choice(tmp_path: Path) -> None:
    path = tmp_path / "land.gpkg"
    for layer in ("east", "west"):
        write(
            path,
            np.array([POLYGON.wkb], dtype=object),
            [],
            [],
            driver="GPKG",
            layer=layer,
            geometry_type="Polygon",
            crs="EPSG:4326",
        )
    result = import_boundary(path.read_bytes(), "land.gpkg")
    assert result.status == "choose-layer" and set(result.layers) == {"east", "west"}
    result = import_boundary(path.read_bytes(), "land.gpkg", layer="west")
    assert (
        result.status == "ready"
        and result.boundary is not None
        and shape(result.boundary.model_dump()).equals(POLYGON)
    )


def test_import_endpoint_enforces_format_and_returns_review(client: TestClient) -> None:
    response = client.post(
        "/api/v1/land/import", files={"file": ("land.json", json.dumps(mapping(POLYGON)))}
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ready"
    response = client.post("/api/v1/land/import", files={"file": ("land.gpkg", b"<VRTDataset/>")})
    assert response.status_code == 422

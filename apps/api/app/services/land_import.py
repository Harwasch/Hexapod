"""Bounded geospatial imports. Uploaded names never become filesystem paths."""

from __future__ import annotations

import io
import json
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pyogrio
from defusedxml import ElementTree
from pyogrio.raw import read
from pyproj import CRS, Transformer
from shapely import from_wkb, get_num_coordinates, make_valid
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.validation import explain_validity

from app.schemas.land import LandCreate
from app.schemas.land_import import BoundaryImportRead
from app.services.errors import InvalidInputError
from app.services.land import FOOTPRINT

MAX_UPLOAD = 20 * 1024 * 1024
MAX_EXPANDED = 100 * 1024 * 1024


def _archive(data: bytes, suffixes: set[str]) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            files = [info for info in archive.infolist() if not info.is_dir()]
            if len(files) > 100 or sum(info.file_size for info in files) > MAX_EXPANDED:
                raise InvalidInputError("The archive expands beyond the import limit.")
            result = {}
            for info in files:
                path = PurePosixPath(info.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)
                ):
                    raise InvalidInputError("Archive paths and links must stay within the import.")
                if path.suffix.lower() not in suffixes:
                    continue
                if path.name in result:
                    raise InvalidInputError("Archive file names must be unique.")
                result[path.name] = archive.read(info)
            return result
    except zipfile.BadZipFile as error:
        raise InvalidInputError("The archive is not a valid ZIP file.") from error


def _kml(data: bytes) -> list[BaseGeometry]:
    root = ElementTree.fromstring(data)
    polygons: list[BaseGeometry] = []
    namespace = "{http://www.opengis.net/kml/2.2}"
    for element in root.iter(namespace + "Polygon"):
        rings = []
        for kind in ("outerBoundaryIs", "innerBoundaryIs"):
            for boundary in element.findall(namespace + kind):
                coordinates = boundary.find(f"{namespace}LinearRing/{namespace}coordinates")
                if coordinates is None or not coordinates.text:
                    raise InvalidInputError("A KML polygon has no ring coordinates.")
                ring = []
                for token in coordinates.text.split():
                    values = token.split(",")
                    if len(values) < 2:
                        raise InvalidInputError("Invalid KML coordinate.")
                    ring.append([float(values[0]), float(values[1])])
                rings.append(ring)
        if rings:
            polygons.append(Polygon(rings[0], rings[1:]))
    if not polygons:
        raise InvalidInputError("The KML file contains no polygon land boundaries.")
    return polygons


def _geojson(value: dict[str, Any]) -> list[BaseGeometry]:
    if value.get("type") == "FeatureCollection":
        features = value.get("features", [])
        if len(features) > 1000:
            raise InvalidInputError("Import at most 1,000 polygon features at once.")
        return [shape(feature["geometry"]) for feature in features]
    if value.get("type") == "Feature":
        return [shape(value["geometry"])]
    return [shape(value)]


def import_boundary(
    data: bytes,
    filename: str,
    source_crs: str | None = None,
    layer: str | None = None,
    repair: bool = False,
) -> BoundaryImportRead:
    if len(data) > MAX_UPLOAD:
        raise InvalidInputError("Boundary imports are limited to 20 MB.")
    extension = Path(filename).suffix.lower()
    warnings: list[str] = []
    layers: list[str] = []
    crs = source_crs
    try:
        if extension in {".geojson", ".json"}:
            document = json.loads(data)
            geometries = _geojson(document)
            declared = document.get("crs", {}).get("properties", {}).get("name")
            crs = crs or declared or "EPSG:4326"
        elif extension in {".kml", ".kmz"}:
            if extension == ".kmz":
                files = _archive(data, {".kml"})
                if len(files) != 1:
                    raise InvalidInputError("Choose a KMZ containing one KML document.")
                data = next(iter(files.values()))
            geometries = _kml(data)
            crs = "EPSG:4326"
        elif extension in {".zip", ".gpkg"}:
            with tempfile.TemporaryDirectory(prefix="land-import-") as temporary:
                directory = Path(temporary)
                if extension == ".zip":
                    files = _archive(data, {".shp", ".shx", ".dbf", ".prj", ".cpg"})
                    shapes = [name for name in files if name.lower().endswith(".shp")]
                    if not shapes:
                        raise InvalidInputError("The ZIP file contains no shapefile.")
                    layers = [Path(name).stem for name in shapes]
                    if len(layers) > 1 and layer not in layers:
                        return BoundaryImportRead(status="choose-layer", layers=layers)
                    for name, content in files.items():
                        (directory / name).write_bytes(content)
                    path = directory / next(
                        name for name in shapes if layer is None or Path(name).stem == layer
                    )
                    if path.read_bytes()[:4] != b"\x00\x00\x27\x0a":
                        raise InvalidInputError("The file does not have a shapefile header.")
                    driver = "ESRI Shapefile"
                    selected_layer = None
                else:
                    path = directory / "upload.gpkg"
                    if not data.startswith(b"SQLite format 3\x00"):
                        raise InvalidInputError(
                            "The file does not have a GeoPackage database header."
                        )
                    path.write_bytes(data)
                    driver = "GPKG"
                    layers = [
                        str(row[0]) for row in pyogrio.list_layers(path) if row[1] is not None
                    ]
                    if not layers:
                        raise InvalidInputError("The GeoPackage contains no vector layers.")
                    if len(layers) > 1 and layer not in layers:
                        return BoundaryImportRead(status="choose-layer", layers=layers)
                    selected_layer = layer or layers[0]
                info = pyogrio.read_info(path, layer=selected_layer)
                if info.get("driver") != driver:
                    raise InvalidInputError(
                        "The uploaded file format does not match its extension."
                    )
                metadata, _, wkb, _ = read(
                    path,
                    layer=selected_layer,
                    columns=[],
                    max_features=1001,
                    read_geometry=True,
                )
                if wkb is None or len(wkb) > 1000:
                    raise InvalidInputError("Import at most 1,000 polygon features at once.")
                crs = crs or metadata.get("crs")
                if not crs:
                    return BoundaryImportRead(
                        status="needs-crs",
                        layers=layers,
                        warnings=[
                            "This file has no coordinate reference system. Choose its source EPSG code before import."
                        ],
                    )
                geometries = [from_wkb(value) for value in wkb if value is not None]
        else:
            raise InvalidInputError("Use GeoJSON, KML/KMZ, a zipped shapefile, or GeoPackage.")
        if not geometries:
            raise InvalidInputError("The file contains no polygon features.")
        if sum(get_num_coordinates(geometry) for geometry in geometries) > 20_000:
            raise InvalidInputError("Import at most 20,000 boundary vertices.")
        resolved_crs = CRS.from_user_input(crs)
        if resolved_crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs(resolved_crs, 4326, always_xy=True)
            geometries = [transform(transformer.transform, geometry) for geometry in geometries]
            warnings.append(
                f"Reprojected from {resolved_crs.to_string()} to WGS 84; review the placement."
            )
        polygons: list[Polygon] = []
        for geometry in geometries:
            if not isinstance(geometry, Polygon | MultiPolygon):
                raise InvalidInputError("A land boundary import must contain polygons only.")
            if not geometry.is_valid:
                if not repair:
                    return BoundaryImportRead(
                        status="needs-repair",
                        source_crs=resolved_crs.to_string(),
                        layers=layers,
                        warnings=[
                            explain_validity(geometry),
                            "Preview a repair before accepting this boundary.",
                        ],
                    )
                geometry = make_valid(geometry)
                warnings.append(
                    "Repaired invalid geometry. Inspect all outlines and exclusions before saving."
                )
            if isinstance(geometry, Polygon):
                polygons.append(geometry)
            elif isinstance(geometry, MultiPolygon):
                polygons.extend(geometry.geoms)
            else:
                raise InvalidInputError(
                    "Repair produced non-polygon fragments. Repair the source file explicitly."
                )
        combined = MultiPolygon(polygons)
        if not combined.is_valid:
            raise InvalidInputError(
                "Imported polygon features overlap. Merge or repair them before importing."
            )
        boundary = FOOTPRINT.validate_python(mapping(combined))
        LandCreate.bounded_boundary(boundary)
        return BoundaryImportRead(
            status="ready",
            boundary=boundary,
            source_crs=resolved_crs.to_string(),
            layers=layers,
            warnings=warnings,
        )
    except InvalidInputError:
        raise
    except Exception as error:
        raise InvalidInputError(
            "The file could not be read as a valid land boundary. Check its format, geometry and coordinate system."
        ) from error

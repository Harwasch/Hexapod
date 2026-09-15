"""Conversions between GeoJSON schemas, Shapely and PostGIS geometries."""

from __future__ import annotations

from typing import Any

from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from app.schemas.common import BoundingBox, GeoPosition
from app.schemas.geojson import MultiPolygon, Point, Polygon

SRID = 4326


def footprint_to_wkb(footprint: Polygon | MultiPolygon) -> WKBElement:
    """Store every footprint as a MultiPolygon so the column type is uniform."""
    geom = shape(footprint.model_dump())
    if isinstance(geom, ShapelyPolygon):
        geom = ShapelyMultiPolygon([geom])
    return from_shape(geom, srid=SRID)


def wkb_to_footprint(value: WKBElement | None) -> Polygon | MultiPolygon | None:
    if value is None:
        return None
    geom = to_shape(value)
    data: dict[str, Any] = dict(mapping(geom))
    data["coordinates"] = _to_lists(data["coordinates"])
    if data["type"] == "Polygon":
        return Polygon.model_validate(data)
    return MultiPolygon.model_validate(data)


def bbox_to_wkb(bbox: BoundingBox) -> WKBElement:
    poly = ShapelyPolygon(
        [
            (bbox.west, bbox.south),
            (bbox.east, bbox.south),
            (bbox.east, bbox.north),
            (bbox.west, bbox.north),
            (bbox.west, bbox.south),
        ]
    )
    return from_shape(poly, srid=SRID)


def wkb_to_bbox(value: WKBElement | None) -> BoundingBox | None:
    if value is None:
        return None
    west, south, east, north = to_shape(value).bounds
    return BoundingBox(west=west, south=south, east=east, north=north)


def position_to_wkb(position: GeoPosition) -> WKBElement:
    return from_shape(
        shape(Point(coordinates=[position.longitude, position.latitude]).model_dump()), srid=SRID
    )


def wkb_to_position(value: WKBElement, height: float | None) -> GeoPosition:
    geom = to_shape(value)
    if not isinstance(geom, ShapelyPoint):
        raise ValueError("centroid column must contain a Point")
    return GeoPosition(longitude=float(geom.x), latitude=float(geom.y), height=height)


def centroid_of(footprint: Polygon | MultiPolygon) -> GeoPosition:
    geom: BaseGeometry = shape(footprint.model_dump())
    centroid: ShapelyPoint = (
        geom.representative_point() if not geom.centroid.within(geom) else geom.centroid
    )
    return GeoPosition(longitude=float(centroid.x), latitude=float(centroid.y), height=None)


def _to_lists(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_to_lists(v) for v in value]
    return value

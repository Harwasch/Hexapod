"""Strict GeoJSON geometry schemas (RFC 7946) with topological validation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from shapely.geometry import shape
from shapely.validation import explain_validity

from app.schemas.base import CamelModel

Position = Annotated[list[float], Field(min_length=2, max_length=3)]
LinearRing = Annotated[list[Position], Field(min_length=4)]
PolygonCoordinates = Annotated[list[LinearRing], Field(min_length=1)]


def _check_position(position: list[float]) -> None:
    lon, lat = position[0], position[1]
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"longitude {lon} out of range [-180, 180]")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude {lat} out of range [-90, 90]")


def _check_ring(ring: list[list[float]]) -> None:
    for position in ring:
        _check_position(position)
    if ring[0][:2] != ring[-1][:2]:
        raise ValueError("linear ring must be closed (first and last positions equal)")


class Point(CamelModel):
    type: Literal["Point"] = "Point"
    coordinates: Position

    @field_validator("coordinates")
    @classmethod
    def _validate(cls, value: list[float]) -> list[float]:
        _check_position(value)
        return value


class Polygon(CamelModel):
    type: Literal["Polygon"] = "Polygon"
    coordinates: PolygonCoordinates

    @field_validator("coordinates")
    @classmethod
    def _validate_rings(cls, value: list[list[list[float]]]) -> list[list[list[float]]]:
        for ring in value:
            _check_ring(ring)
        return value

    @model_validator(mode="after")
    def _validate_topology(self) -> Polygon:
        geom = shape(self.model_dump())
        if not geom.is_valid:
            raise ValueError(f"invalid polygon: {explain_validity(geom)}")
        if geom.area == 0:
            raise ValueError("polygon has zero area")
        return self


class MultiPolygon(CamelModel):
    type: Literal["MultiPolygon"] = "MultiPolygon"
    coordinates: Annotated[list[PolygonCoordinates], Field(min_length=1)]

    @field_validator("coordinates")
    @classmethod
    def _validate_rings(cls, value: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
        for polygon in value:
            for ring in polygon:
                _check_ring(ring)
        return value

    @model_validator(mode="after")
    def _validate_topology(self) -> MultiPolygon:
        geom = shape(self.model_dump())
        if not geom.is_valid:
            raise ValueError(f"invalid multipolygon: {explain_validity(geom)}")
        if geom.area == 0:
            raise ValueError("multipolygon has zero area")
        return self


Footprint = Annotated[Polygon | MultiPolygon, Field(discriminator="type")]

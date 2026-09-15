from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.geojson import Footprint, Polygon

adapter: TypeAdapter[object] = TypeAdapter(Footprint)


def test_valid_polygon() -> None:
    poly = Polygon.model_validate(
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    )
    assert poly.type == "Polygon"


@pytest.mark.parametrize(
    "coordinates,message",
    [
        ([[[0, 0], [1, 0], [1, 1], [0, 1]]], "closed"),
        ([[[0, 0], [1, 0], [1, 1]]], "at least 4"),
        ([[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]], "Self-intersection"),
        ([[[0, 0], [1, 0], [2, 0], [3, 0], [0, 0]]], "invalid polygon"),
        ([[[200, 0], [1, 0], [1, 1], [0, 1], [200, 0]]], "longitude"),
        ([[[0, 95], [1, 0], [1, 1], [0, 1], [0, 95]]], "latitude"),
    ],
)
def test_invalid_polygons(coordinates: list[list[list[float]]], message: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Polygon.model_validate({"type": "Polygon", "coordinates": coordinates})
    assert message.lower() in str(excinfo.value).lower()


def test_discriminator_rejects_other_geometries() -> None:
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "Point", "coordinates": [0, 0]})
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "LineString", "coordinates": [[0, 0], [1, 1]]})

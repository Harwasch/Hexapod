"""Bounded model access to the exact map feature pinned when a run starts."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from app.services.errors import InvalidInputError


def vertices(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def visit(coordinates: list[Any], path: list[int]) -> None:
        if coordinates and isinstance(coordinates[0], (float, int)):
            result.append({"path": path, "coordinates": coordinates})
        else:
            for index, child in enumerate(coordinates):
                visit(child, [*path, index])

    visit(geometry["coordinates"], [])
    return result


def model_context(snapshot: dict[str, Any] | None, boundary_revision: int) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    result = deepcopy(snapshot)
    result["boundaryDiffersFromInvestigation"] = snapshot["boundaryRevision"] != boundary_revision
    geometry = result["feature"]["geometry"]
    points = vertices(geometry)
    result["geometryType"] = geometry["type"]
    result["vertexCount"] = len(points)
    if len(json.dumps(geometry)) > 12_000:
        result["feature"].pop("geometry")
        result["geometryReadTool"] = "read_focused_geometry"
        result["geometryOmitted"] = True
    return result


def read_geometry(snapshot: dict[str, Any] | None, offset: int, count: int) -> dict[str, Any]:
    if snapshot is None:
        raise InvalidInputError("This research request has no focused map feature.")
    geometry = snapshot["feature"]["geometry"]
    points = vertices(geometry)
    if offset >= len(points):
        raise InvalidInputError("The requested vertex page is outside the focused geometry.")
    return {
        "type": geometry["type"],
        "vertices": points[offset : offset + count],
        "total": len(points),
        "nextOffset": offset + count if offset + count < len(points) else None,
        "pathMeaning": "Indices into GeoJSON coordinates; polygon rings include holes and closure vertices.",
    }

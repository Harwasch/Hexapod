from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Layer
from app.schemas.common import Attribution, LicenseMetadata, Provenance, TemporalExtent
from app.schemas.layer import (
    LayerCreate,
    LayerRead,
    LayerUpdate,
    LegendMetadata,
    RenderMetadata,
    source_type_for,
)
from app.services import geometry
from app.services.errors import ConflictError, NotFoundError
from app.services.slugs import slugify
from app.services.urls import validate_dataset_url

URL_FIELDS = ("url", "url_template")


def _validate_source_urls(source: Any) -> None:
    for field in URL_FIELDS:
        value = getattr(source, field, None)
        if value is not None:
            validate_dataset_url(str(value))


def _unique_slug(db: Session, base: str) -> str:
    slug = base
    counter = 2
    while db.scalar(select(Layer.id).where(Layer.slug == slug)) is not None:
        slug = f"{base}-{counter}"
        counter += 1
    return slug


def create_layer(db: Session, payload: LayerCreate, *, builtin: bool = False) -> Layer:
    _validate_source_urls(payload.source)
    if payload.slug and db.scalar(select(Layer.id).where(Layer.slug == payload.slug)) is not None:
        raise ConflictError(f"layer slug '{payload.slug}' already exists")
    render = payload.render.model_dump(mode="json", by_alias=True)
    if payload.provenance is not None:
        render["provenance"] = payload.provenance.model_dump(mode="json", by_alias=True)
    layer = Layer(
        slug=payload.slug or _unique_slug(db, slugify(payload.name)),
        name=payload.name,
        description=payload.description,
        category=payload.category,
        source_type=source_type_for(payload.source),
        source=payload.source.model_dump(mode="json", by_alias=True),
        spatial_extent=geometry.bbox_to_wkb(payload.spatial_extent)
        if payload.spatial_extent
        else None,
        temporal_extent=(
            payload.temporal_extent.model_dump(mode="json", by_alias=True)
            if payload.temporal_extent
            else None
        ),
        observed_at=payload.observed_at,
        resolution=payload.resolution,
        coverage=payload.coverage,
        render=render,
        legend=payload.legend.model_dump(mode="json", by_alias=True) if payload.legend else None,
        attribution=[a.model_dump(mode="json", by_alias=True) for a in payload.attribution],
        license=payload.license.model_dump(mode="json", by_alias=True) if payload.license else None,
        default_visible=payload.default_visible,
        builtin=builtin,
    )
    db.add(layer)
    db.commit()
    db.refresh(layer)
    return layer


def list_layers(db: Session) -> list[Layer]:
    return list(db.scalars(select(Layer).order_by(Layer.category, Layer.created_at)).all())


def get_layer(db: Session, layer_id: uuid.UUID) -> Layer:
    layer = db.get(Layer, layer_id)
    if layer is None:
        raise NotFoundError("layer", layer_id)
    return layer


def update_layer(db: Session, layer_id: uuid.UUID, payload: LayerUpdate) -> Layer:
    layer = get_layer(db, layer_id)
    data = payload.model_dump(exclude_unset=True)
    for field in ("name", "description", "category", "default_visible", "observed_at"):
        if field in data:
            setattr(layer, field, data[field])
    if "render" in data and payload.render is not None:
        provenance = layer.render.get("provenance")
        layer.render = payload.render.model_dump(mode="json", by_alias=True)
        if provenance:
            layer.render["provenance"] = provenance
    if "attribution" in data and payload.attribution is not None:
        layer.attribution = [a.model_dump(mode="json", by_alias=True) for a in payload.attribution]
    if "license" in data:
        layer.license = (
            payload.license.model_dump(mode="json", by_alias=True) if payload.license else None
        )
    if "temporal_extent" in data:
        layer.temporal_extent = (
            payload.temporal_extent.model_dump(mode="json", by_alias=True)
            if payload.temporal_extent
            else None
        )
    db.commit()
    db.refresh(layer)
    return layer


def delete_layer(db: Session, layer_id: uuid.UUID) -> None:
    layer = get_layer(db, layer_id)
    if layer.builtin:
        raise ConflictError("built-in catalog layers cannot be deleted")
    db.delete(layer)
    db.commit()


def layer_to_read(layer: Layer) -> LayerRead:
    render_raw = dict(layer.render)
    provenance_raw = render_raw.pop("provenance", None)
    return LayerRead.model_validate(
        {
            "id": layer.id,
            "slug": layer.slug,
            "name": layer.name,
            "description": layer.description,
            "category": layer.category,
            "sourceType": layer.source_type,
            "source": layer.source,
            "spatialExtent": geometry.wkb_to_bbox(layer.spatial_extent),
            "temporalExtent": (
                TemporalExtent.model_validate(layer.temporal_extent)
                if layer.temporal_extent
                else None
            ),
            "observedAt": layer.observed_at,
            "resolution": layer.resolution,
            "coverage": layer.coverage,
            "render": RenderMetadata.model_validate(render_raw),
            "legend": LegendMetadata.model_validate(layer.legend) if layer.legend else None,
            "attribution": [Attribution.model_validate(a) for a in layer.attribution],
            "license": LicenseMetadata.model_validate(layer.license) if layer.license else None,
            "provenance": Provenance.model_validate(provenance_raw) if provenance_raw else None,
            "defaultVisible": layer.default_visible,
            "builtin": layer.builtin,
            "createdAt": layer.created_at,
            "updatedAt": layer.updated_at,
        }
    )

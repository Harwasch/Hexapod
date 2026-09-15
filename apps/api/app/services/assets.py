from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Asset, Site
from app.schemas.asset import (
    AssetBase,
    AssetCreate,
    AssetRead,
    AssetUpdate,
    CesiumIonSource,
    CrsMetadata,
    RenderConfig,
    ResolutionMetadata,
    TilesUrlSource,
    provider_for_source,
)
from app.schemas.common import Attribution, LicenseMetadata, Provenance
from app.services import geometry
from app.services.errors import NotFoundError
from app.services.urls import validate_dataset_url


def build_asset(payload: AssetBase, site_id: uuid.UUID | None) -> Asset:
    if payload.source.type == "3d-tiles-url":
        validate_dataset_url(str(payload.source.url))
    render = payload.render_config.model_dump(mode="json", by_alias=True)
    if payload.provenance is not None:
        render["provenance"] = payload.provenance.model_dump(mode="json", by_alias=True)
    return Asset(
        site_id=site_id,
        name=payload.name,
        representation=payload.representation,
        provider=provider_for_source(payload.source),
        source=payload.source.model_dump(mode="json", by_alias=True),
        footprint=geometry.footprint_to_wkb(payload.footprint) if payload.footprint else None,
        observed_at=payload.observed_at,
        valid_from=payload.valid_from,
        valid_to=payload.valid_to,
        resolution=payload.resolution.model_dump(mode="json", by_alias=True)
        if payload.resolution
        else None,
        crs=payload.crs.model_dump(mode="json", by_alias=True) if payload.crs else None,
        license=payload.license.model_dump(mode="json", by_alias=True) if payload.license else None,
        attribution=[a.model_dump(mode="json", by_alias=True) for a in payload.attribution],
        render_config=render,
        default_visible=payload.default_visible,
    )


def create_asset(db: Session, payload: AssetCreate) -> Asset:
    if payload.site_id is not None and db.get(Site, payload.site_id) is None:
        raise NotFoundError("site", payload.site_id)
    asset = build_asset(payload, site_id=payload.site_id)
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def list_assets(db: Session, site_id: uuid.UUID | None = None) -> list[Asset]:
    stmt = select(Asset).order_by(Asset.created_at.asc())
    if site_id is not None:
        stmt = stmt.where(Asset.site_id == site_id)
    return list(db.scalars(stmt).all())


def get_asset(db: Session, asset_id: uuid.UUID) -> Asset:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise NotFoundError("asset", asset_id)
    return asset


def update_asset(db: Session, asset_id: uuid.UUID, payload: AssetUpdate) -> Asset:
    asset = get_asset(db, asset_id)
    data = payload.model_dump(exclude_unset=True)
    for field in ("name", "observed_at", "valid_from", "valid_to", "default_visible"):
        if field in data:
            setattr(asset, field, data[field])
    if "resolution" in data:
        asset.resolution = (
            payload.resolution.model_dump(mode="json", by_alias=True)
            if payload.resolution
            else None
        )
    if "license" in data:
        asset.license = (
            payload.license.model_dump(mode="json", by_alias=True) if payload.license else None
        )
    if "attribution" in data and payload.attribution is not None:
        asset.attribution = [a.model_dump(mode="json", by_alias=True) for a in payload.attribution]
    if "render_config" in data and payload.render_config is not None:
        provenance = asset.render_config.get("provenance")
        asset.render_config = payload.render_config.model_dump(mode="json", by_alias=True)
        if provenance:
            asset.render_config["provenance"] = provenance
    if "footprint" in data:
        asset.footprint = (
            geometry.footprint_to_wkb(payload.footprint) if payload.footprint else None
        )
    if asset.valid_from and asset.valid_to and asset.valid_to < asset.valid_from:
        raise ValueError("validTo must not precede validFrom")
    db.commit()
    db.refresh(asset)
    return asset


def delete_asset(db: Session, asset_id: uuid.UUID) -> None:
    db.delete(get_asset(db, asset_id))
    db.commit()


def asset_to_read(asset: Asset) -> AssetRead:
    source_raw = asset.source
    source: CesiumIonSource | TilesUrlSource = (
        CesiumIonSource.model_validate(source_raw)
        if source_raw.get("type") == "cesium-ion"
        else TilesUrlSource.model_validate(source_raw)
    )
    render_raw = dict(asset.render_config)
    provenance_raw = render_raw.pop("provenance", None)
    return AssetRead(
        id=asset.id,
        site_id=asset.site_id,
        provider=asset.provider,
        name=asset.name,
        representation=asset.representation,
        source=source,
        footprint=geometry.wkb_to_footprint(asset.footprint),
        observed_at=asset.observed_at,
        valid_from=asset.valid_from,
        valid_to=asset.valid_to,
        resolution=ResolutionMetadata.model_validate(asset.resolution)
        if asset.resolution
        else None,
        crs=CrsMetadata.model_validate(asset.crs) if asset.crs else None,
        license=LicenseMetadata.model_validate(asset.license) if asset.license else None,
        attribution=[Attribution.model_validate(a) for a in asset.attribution],
        provenance=Provenance.model_validate(provenance_raw) if provenance_raw else None,
        render_config=RenderConfig.model_validate(render_raw),
        default_visible=asset.default_visible,
        created_at=asset.created_at,
        updated_at=asset.updated_at,
    )

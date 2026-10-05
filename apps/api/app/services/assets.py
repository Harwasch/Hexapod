from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Asset, Site
from app.models.enums import AssetProvider, Representation, ScaleSource
from app.schemas.asset import (
    AssetBase,
    AssetCreate,
    AssetRead,
    AssetScaleUpdate,
    AssetUpdate,
    CesiumIonSource,
    CrsMetadata,
    RenderConfig,
    ResolutionMetadata,
    ScaleEvidence,
    ScaleEvidenceInput,
    SidecarFlag,
    TilesUrlSource,
    provider_for_source,
)
from app.schemas.common import Attribution, LicenseMetadata, Provenance
from app.services import geometry, placement
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.urls import validate_dataset_url

#: The Problem `code` of a scale refused because nothing says where the asset's tiles are
#: placed: not a pipeline-registered splat, so there is no origin to resize it about.
NOT_SCALABLE = "not_scalable"

#: The render config keys `set_scale` owns. A PATCH of the render config keeps them.
SCALE_KEYS = ("scale", "scaleEvidence")


def render_config_document(config: RenderConfig) -> dict[str, Any]:
    """A render config as stored: camel case, with the runtime scale only when it has one.

    `scale` at 1 and no `scaleEvidence` is every asset nobody resized, and is what reading
    a stored render config without them gives back. Leaving them out keeps those rows
    exactly as they were before the keys existed -- so the code before them can still read
    every asset that was never scaled.
    """
    render = config.model_dump(mode="json", by_alias=True)
    if render.get("scale") == 1.0:
        render.pop("scale")
    if render.get("scaleEvidence") is None:
        render.pop("scaleEvidence", None)
    return render


def build_asset(payload: AssetBase, site_id: uuid.UUID | None) -> Asset:
    if payload.source.type == "3d-tiles-url":
        validate_dataset_url(str(payload.source.url))
    render = render_config_document(payload.render_config)
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
        # The provenance and the runtime scale are not the render config's to replace: the
        # scale moves the site's boundary with it, so only `set_scale` changes it.
        stored = asset.render_config
        render = render_config_document(payload.render_config)
        for key in SCALE_KEYS:
            render.pop(key, None)
            if key in stored:
                render[key] = stored[key]
        if stored.get("provenance"):
            render["provenance"] = stored["provenance"]
        asset.render_config = render
    if "footprint" in data:
        asset.footprint = (
            geometry.footprint_to_wkb(payload.footprint) if payload.footprint else None
        )
    if asset.valid_from and asset.valid_to and asset.valid_to < asset.valid_from:
        raise InvalidInputError("validTo must not precede validFrom")
    db.commit()
    db.refresh(asset)
    return asset


def set_scale(
    db: Session, asset_id: uuid.UUID, payload: AssetScaleUpdate, *, now: datetime | None = None
) -> Asset:
    """Resize a pipeline-placed splat at runtime, or put it back as registered.

    `renderConfig.scale` is the factor the viewer draws the tileset at, relative to the
    model as registered -- absolute, so 0.8 and then 0.5 is 0.5, and a reset is 1 -- with
    `scaleEvidence` saying how it was found. Everything the catalog places on the globe
    moves with it, about the origin the tiles are placed at, by the new scale over the old:
    the site's boundary and centroid, the asset's footprint and its measured ground. The
    viewer then has only the tiles to scale, and what it clips, clamps and outlines already
    fits them.

    The provenance says what the asset is now drawn at: `manual` for a measured length or a
    typed factor, `camera-height-estimate` with its ±% for an estimate. A reset restores the
    provenance it was registered with, which the evidence carries from the first scale on.
    The capture row's `scaleSource` is left alone: it describes the capture's files
    (`canonical.ply`), which a runtime scale does not change.

    Refused (409, `not_scalable`) for anything but the splat a pipeline run registered: a
    seeded or hand-made asset has no recorded origin to resize it about.
    """
    asset = _lock(db, asset_id)
    try:
        site, origin = _placed(db, asset)
    except ConflictError:
        db.rollback()
        raise
    render = dict(asset.render_config or {})
    config = RenderConfig.model_validate({k: v for k, v in render.items() if k != "provenance"})
    provenance = Provenance.model_validate(render.get("provenance") or {})
    before = config.scale_evidence
    registered = (
        (before.registered_scale_source, before.registered_scale_uncertainty_pct)
        if before is not None
        else (provenance.scale_source, provenance.scale_uncertainty_pct)
    )
    if payload.reset or payload.scale is None:
        scale = 1.0
        for key in SCALE_KEYS:
            render.pop(key, None)
        if before is not None:
            provenance = provenance.model_copy(
                update={"scale_source": registered[0], "scale_uncertainty_pct": registered[1]}
            )
    else:
        scale = payload.scale
        given = payload.evidence or ScaleEvidenceInput(method="direct")
        evidence = ScaleEvidence(
            **given.model_dump(),
            set_at=now or datetime.now(tz=UTC),
            registered_scale_source=registered[0],
            registered_scale_uncertainty_pct=registered[1],
        )
        render["scale"] = scale
        render["scaleEvidence"] = evidence.model_dump(mode="json", by_alias=True)
        provenance = provenance.model_copy(update=_scale_provenance(evidence))
    render["provenance"] = provenance.model_dump(mode="json", by_alias=True)
    factor = scale / config.scale
    if factor != 1.0:
        try:
            placement.rescale_site(site, origin, factor)
            placement.rescale_footprint(asset, origin, factor)
            if render.get("groundSamples"):
                render["groundSamples"] = placement.rescale_ground_samples(
                    render["groundSamples"], origin, factor
                )
        except InvalidInputError:
            # Nothing half-resized stays in the session for a later commit to write.
            db.rollback()
            raise
    asset.render_config = render
    db.commit()
    db.refresh(asset)
    return asset


def _scale_provenance(evidence: ScaleEvidence) -> dict[str, Any]:
    """What a scale found this way makes the asset's scale source, and its ±%."""
    if evidence.method == "camera-height-estimate":
        return {
            "scale_source": ScaleSource.CAMERA_HEIGHT_ESTIMATE,
            "scale_uncertainty_pct": evidence.uncertainty_pct,
        }
    return {"scale_source": ScaleSource.MANUAL, "scale_uncertainty_pct": None}


def _lock(db: Session, asset_id: uuid.UUID) -> Asset:
    """The asset, locked for this transaction: the row lock a re-run takes before it
    repoints the asset (`app/worker/registration.py`), so a scale and a re-run's reset of
    it happen one after the other and the boundary is resized from the right scale."""
    asset = db.execute(
        select(Asset)
        .where(Asset.id == asset_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if asset is None:
        db.rollback()
        raise NotFoundError("asset", asset_id)
    return asset


def _placed(db: Session, asset: Asset) -> tuple[Site, placement.Origin]:
    """The site of a pipeline-placed splat and the origin its tiles are placed at, or 409.

    The splat a run registered and repoints -- the site's first gaussian-splat, a 3D Tiles
    URL -- on a site whose metadata carries the run's registration and so its origin.
    """
    why = None
    site = db.get(Site, asset.site_id) if asset.site_id is not None else None
    origin = placement.registered_origin(site.metadata_) if site is not None else None
    if asset.representation != Representation.GAUSSIAN_SPLAT:
        why = f"it is a {asset.representation.value}, and only a splat's tiles are resized"
    elif asset.provider != AssetProvider.TILES_3D_URL:
        why = "its tiles are not a 3D Tiles URL a pipeline run published"
    elif site is None:
        why = "it is on no site"
    elif origin is None or not site.metadata_.get("captureId"):
        why = (
            "its site was not registered by a pipeline run, so nothing records the origin "
            "its tiles are placed at"
        )
    elif _registered_splat(db, site) != asset.id:
        why = "it is not the splat the site's pipeline run registered"
    if why is not None or site is None or origin is None:
        raise ConflictError(
            f"asset {asset.id} cannot be rescaled: {why or 'it has no placement'}",
            code=NOT_SCALABLE,
        )
    return site, origin


def _registered_splat(db: Session, site: Site) -> uuid.UUID | None:
    """The asset a run repoints: the site's first splat by creation, as the worker's
    `registration.splat_asset` picks it."""
    return db.scalar(
        select(Asset.id)
        .where(Asset.site_id == site.id, Asset.representation == Representation.GAUSSIAN_SPLAT)
        .order_by(Asset.created_at, Asset.id)
        .limit(1)
    )


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
        sidecar_flags=[SidecarFlag.model_validate(entry) for entry in asset.sidecar_flags or []],
        created_at=asset.created_at,
        updated_at=asset.updated_at,
    )

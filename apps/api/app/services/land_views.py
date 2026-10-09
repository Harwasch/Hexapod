from __future__ import annotations

import hashlib
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.land import LandArea, LandBoundaryRevision
from app.models.land_feature import LandFeature
from app.models.land_image_registration import LandImageRegistration
from app.models.land_raster import LandRaster
from app.models.land_solar import LandSolar
from app.models.land_survey import LandSurvey
from app.models.land_view import LandView
from app.models.research import Evidence, Investigation, ResearchArtifact, ResearchRun
from app.schemas.land_image_registrations import ImageRegistrationResult
from app.schemas.land_rasters import RasterMetadata
from app.schemas.land_views import (
    LandViewCreate,
    LandViewMap,
    LandViewOpen,
    LandViewRasterRead,
    LandViewRead,
)
from app.schemas.research import ArtifactContent
from app.services.errors import ConflictError, InvalidInputError, NotFoundError


def area(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, *, lock: bool = False
) -> LandArea:
    query = select(LandArea).where(LandArea.id == land_id, LandArea.workspace_id == workspace_id)
    row = db.scalar(query.with_for_update() if lock else query)
    if row is None:
        raise NotFoundError("land area", land_id)
    return row


def scoped(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    view_id: uuid.UUID,
    *,
    lock: bool = False,
) -> LandView:
    area(db, workspace_id, land_id)
    query = select(LandView).where(LandView.id == view_id, LandView.land_id == land_id)
    row = db.scalar(query.with_for_update() if lock else query)
    if row is None:
        raise NotFoundError("saved land view", view_id)
    return row


def resolve(
    db: Session, land: LandArea, view: LandViewRead, *, strict: bool = False
) -> LandViewOpen:
    state = view.state.model_copy(deep=True)
    warnings: list[str] = []
    maps: list[LandViewMap] = []
    rasters: list[LandViewRasterRead] = []

    def unavailable(message: str) -> None:
        if strict:
            raise InvalidInputError(message)
        warnings.append(message)

    if state.boundary_revision != land.revision:
        warnings.append(
            f"Saved against boundary revision {state.boundary_revision}; "
            f"the current boundary is revision {land.revision}. The current boundary remains on the map."
        )
    if not db.scalar(
        select(LandBoundaryRevision.id).where(
            LandBoundaryRevision.land_id == land.id,
            LandBoundaryRevision.revision == state.boundary_revision,
        )
    ):
        unavailable("The saved boundary revision is unavailable.")
    if state.investigation_id and not db.scalar(
        select(Investigation.id).where(
            Investigation.id == state.investigation_id, Investigation.land_id == land.id
        )
    ):
        unavailable("The saved investigation is unavailable for this land.")
        state.investigation_id = None
    for attribute, model, label in (
        ("inventory_id", LandFeature, "asset"),
        ("survey_id", LandSurvey, "survey"),
        ("solar_id", LandSolar, "solar assessment"),
    ):
        identifier = getattr(state, attribute)
        if identifier and not db.scalar(
            select(model.id).where(model.id == identifier, model.land_id == land.id)
        ):
            unavailable(f"The saved {label} is unavailable for this land.")
            setattr(state, attribute, None)
    for identifier in dict.fromkeys(state.artifact_ids):
        row = db.execute(
            select(ResearchArtifact, Investigation.boundary_revision)
            .join(ResearchRun, ResearchRun.id == ResearchArtifact.run_id)
            .join(Investigation, Investigation.id == ResearchRun.investigation_id)
            .where(ResearchArtifact.id == identifier, Investigation.land_id == land.id)
        ).first()
        if row is None:
            unavailable(f"Saved research map {identifier} is unavailable for this land.")
            continue
        artifact, revision = row
        content = ArtifactContent.model_validate(artifact.content)
        if content.output.kind != "map":
            unavailable(f"Saved output {identifier} is not a research map.")
            continue
        maps.append(
            LandViewMap(id=identifier, title=content.title, features=content.output.features)
        )
        if revision != land.revision:
            warnings.append(f"{content.title} was researched against boundary revision {revision}.")
    for reference in state.rasters:
        if reference.kind == "raster":
            raster = db.scalar(
                select(LandRaster).where(
                    LandRaster.id == reference.id, LandRaster.land_id == land.id
                )
            )
            if raster is None:
                unavailable(f"Saved raster {reference.id} is unavailable for this land.")
                continue
            metadata = RasterMetadata.model_validate(raster.metadata_json)
            band = next((item for item in metadata.bands if item.index == reference.band), None)
            if band is None:
                unavailable(f"Saved band {reference.band} of raster {reference.id} is unavailable.")
                continue
            rasters.append(
                LandViewRasterRead(
                    **reference.model_dump(),
                    bounds=metadata.bounds,
                    categorical=band.palette == "categorical",
                    attribution=" · ".join(
                        dict.fromkeys(source.attribution for source in metadata.sources)
                    ),
                )
            )
            if raster.boundary_revision != land.revision:
                warnings.append(
                    f"Raster {reference.id} was analyzed against boundary revision {raster.boundary_revision}."
                )
        else:
            result = db.execute(
                select(LandImageRegistration, Evidence.content)
                .join(Evidence, Evidence.id == LandImageRegistration.evidence_id)
                .join(ResearchRun, ResearchRun.id == Evidence.run_id)
                .join(Investigation, Investigation.id == ResearchRun.investigation_id)
                .where(LandImageRegistration.id == reference.id, Investigation.land_id == land.id)
            ).first()
            if result is None or reference.band != 1:
                unavailable(
                    f"Saved historical map alignment {reference.id} is unavailable for this land."
                )
                continue
            registration, evidence = result
            aligned = ImageRegistrationResult.model_validate(registration.result)
            rasters.append(
                LandViewRasterRead(
                    **reference.model_dump(),
                    bounds=aligned.bounds,
                    attribution=str(evidence.get("attribution", "Historical map alignment")),
                )
            )
    state.artifact_ids = [item.id for item in maps]
    state.rasters = [
        reference
        for reference in state.rasters
        if any(item.id == reference.id and item.kind == reference.kind for item in rasters)
    ]
    opened = LandViewOpen(view=view, state=state, maps=maps, rasters=rasters, warnings=warnings)
    if len(opened.model_dump_json()) > 4_000_000:
        raise InvalidInputError("This view exceeds the 4 MB map limit. Save fewer research layers.")
    return opened


def create(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, payload: LandViewCreate
) -> LandViewRead:
    land = area(db, workspace_id, land_id, lock=True)
    digest = hashlib.sha256(payload.model_dump_json().encode()).hexdigest()
    existing = db.scalar(
        select(LandView).where(
            LandView.land_id == land_id, LandView.request_key == payload.request_key
        )
    )
    if existing:
        if existing.request_sha256 != digest:
            raise ConflictError(
                "This save request already stored a different view. Reload the saved views before trying again."
            )
        return LandViewRead.model_validate(existing)
    row = LandView(
        land_id=land_id,
        name=payload.name,
        request_key=payload.request_key,
        request_sha256=digest,
        state=payload.state.model_dump(mode="json"),
    )
    db.add(row)
    db.flush()
    saved = LandViewRead.model_validate(row)
    resolve(db, land, saved, strict=True)
    db.commit()
    return saved

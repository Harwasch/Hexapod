from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.raster_runner import RasterCancelledError
from app.analysis.raster_runner import run as run_raster
from app.analysis.solar_runner import SolarCancelledError
from app.analysis.solar_runner import run as run_solar
from app.analysis.terrain import SPEC as TERRAIN_SPEC
from app.analysis.vegetation import SPEC as VEGETATION_SPEC
from app.analysis.worldcover import SPEC as COVER_SPEC
from app.config import Settings
from app.models.land import LandArea, LandBoundaryRevision
from app.models.land_archive_image import LandArchiveImage, LandArchiveImageBlob
from app.models.land_document import LandDocument, LandDocumentLink
from app.models.land_feature import LandFeature
from app.models.land_raster import LandRaster
from app.models.land_solar import LandSolar
from app.models.land_survey import LandSurvey
from app.models.research import Evidence, Investigation, ResearchMessage, ResearchRun
from app.models.scenario import LandScenario, LandScenarioRevision
from app.research import inventory as inventory_research
from app.research import queue
from app.research.documents import retrieve as retrieve_documents
from app.research.ecology_outputs import ecology_outputs
from app.research.model import (
    ActionDraftAction,
    ArchiveImageAction,
    ArtifactAction,
    ClaudeResearchModel,
    CompleteAction,
    DocumentOcrAction,
    DocumentReadAction,
    DocumentSearchAction,
    EvidenceReadAction,
    FindingAction,
    InventoryReadAction,
    MappedAssetProposeAction,
    MappedAssetSearchAction,
    MultimodalResearchModel,
    RasterAction,
    ResearchDecision,
    ResearchImage,
    ResearchModel,
    RetrieveAction,
    ScenarioAction,
    ScenarioReadAction,
    SearchAction,
    SolarAction,
    SolarReadAction,
    SurveyReadAction,
    TaxonAction,
)
from app.research.outputs import overview_outputs
from app.research.providers.archives import ARCHIVE_SOURCES
from app.research.providers.base import SourceContext, SourceResult
from app.research.providers.open_data import OVERVIEW_SOURCES, SOURCES, retrieve
from app.research.providers.taxonomy import match as match_taxon
from app.research.scenarios import read as read_scenario
from app.research.search import ClaudeResearchSearch, ResearchSearch
from app.schemas.geojson import Footprint, Point
from app.schemas.land_ecology import EcologyRequest, TaxonQuery
from app.schemas.land_features import InventoryLocator
from app.schemas.land_rasters import RasterMetadata, RasterRequest
from app.schemas.land_selection import CandidateRequest, LandCandidate
from app.schemas.land_solar import SolarMetadata, SolarRequest
from app.schemas.land_surveys import SurveyLocator
from app.schemas.research import (
    ArtifactContent,
    EvidenceContent,
    FindingContent,
    MapFeature,
    MapOutput,
    RasterOutput,
    ResearchBudget,
    SolarOutput,
)
from app.schemas.scenarios import ScenarioCreate
from app.services import (
    land_actions,
    land_archive_images,
    land_features,
    land_rasters,
    land_solar,
    land_surveys,
    scenarios,
)
from app.services.errors import ConflictError, InvalidInputError, NotFoundError

log = logging.getLogger("twin.research")
FOOTPRINT: TypeAdapter[Footprint] = TypeAdapter(Footprint)


def pack(result: SourceResult) -> dict[str, Any]:
    return {
        "provider": result.provider,
        "status": result.status,
        "summary": result.summary,
        "data": result.data,
        "evidence": [[key, item.model_dump(mode="json")] for key, item in result.evidence],
    }


def unpack(value: dict[str, Any]) -> SourceResult:
    return SourceResult(
        value["provider"],
        value["status"],
        value["summary"],
        [(key, EvidenceContent.model_validate(item)) for key, item in value["evidence"]],
        value["data"],
    )


class ResearchWorker:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        settings: Settings,
        *,
        model: ResearchModel | None = None,
        client: httpx.Client | None = None,
        searcher: ResearchSearch | None = None,
    ) -> None:
        self.sessions = sessions
        self.settings = settings
        self.model = model
        self.client = client
        self.searcher = searcher

    def _keep_alive(self, run_id: uuid.UUID, token: uuid.UUID, stopped: threading.Event) -> None:
        while not stopped.wait(5):
            try:
                with self.sessions() as db:
                    run = queue.locked(db, run_id, token)
                    budget = ResearchBudget.model_validate(run.budget)
                    if (
                        run.started_at
                        and (datetime.now(UTC) - run.started_at).total_seconds()
                        >= budget.max_seconds
                    ):
                        queue.finish(
                            db,
                            run_id,
                            token,
                            "partial",
                            "The research time budget was reached. Completed evidence and outputs have been retained.",
                        )
                        stopped.set()
                        return
                    queue.heartbeat(db, run_id, token)
            except queue.LeaseLostError:
                stopped.set()
                return
            except Exception as error:
                log.warning("Research heartbeat failed: %s", type(error).__name__)
                # The lease expires naturally if the database remains unreachable.

    def run_once(self) -> bool:
        with self.sessions() as db:
            claimed = queue.claim(db)
        if claimed is None:
            return False
        run_id, token = claimed
        stopped = threading.Event()
        keeper = threading.Thread(
            target=self._keep_alive, args=(run_id, token, stopped), daemon=True
        )
        keeper.start()
        client = self.client or httpx.Client(headers={"User-Agent": "LivingWorld-LandResearch/1.0"})
        try:
            self._execute(run_id, token, client, stopped)
        except queue.LeaseLostError:
            pass  # Cancelled, timed out or superseded; never publish late results.
        except Exception as error:
            log.warning("Research run %s failed: %s", run_id, type(error).__name__)
            with self.sessions() as db, suppress(queue.LeaseLostError):
                queue.finish(
                    db,
                    run_id,
                    token,
                    "failed",
                    "Research could not finish. Completed evidence is preserved; start a new run to retry.",
                )
        finally:
            stopped.set()
            keeper.join(timeout=6)
            if self.client is None:
                client.close()
        return True

    def _solar(
        self,
        db: Session,
        run_id: uuid.UUID,
        token: uuid.UUID,
        workspace_id: uuid.UUID,
        context: SourceContext,
        request: SolarRequest,
        state: dict[str, Any],
        cancelled: threading.Event | None = None,
    ) -> dict[str, Any]:
        key = "solar/" + hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        if key in state["sources"]:
            return dict(state["sources"][key])
        identifier = uuid.uuid5(run_id, key)
        row = db.get(LandSolar, identifier)
        if row is None:
            queue.checkpoint(
                db,
                run_id,
                token,
                state,
                kind="analysis",
                payload={"message": "Reading hourly weather and modeling the mapped solar array."},
            )
            db.rollback()
            try:
                calculated = run_solar(context.boundary, request, cancelled)
            except SolarCancelledError as error:
                raise queue.LeaseLostError() from error
            run = queue.locked(db, run_id, token)
            row = land_solar.save(
                db,
                workspace_id,
                run,
                identifier,
                request,
                calculated,
                self.settings.land_solar_workspace_quota_bytes,
            )
            queue.checkpoint(
                db,
                run_id,
                token,
                state,
                kind="analysis",
                payload={
                    "message": "Hourly calculation and source archive saved.",
                    "assessmentId": str(identifier),
                },
            )
        metadata = SolarMetadata.model_validate(row.metadata_json)
        evidence_id = queue.save_evidence(
            db,
            run_id,
            token,
            key,
            EvidenceContent(
                provider=request.dataset,
                title=f"NASA POWER hourly solar · {request.year}",
                url=metadata.source_url,
                license=metadata.license,
                attribution=metadata.attribution,
                record_id=str(identifier),
                retrieved_at=metadata.retrieved_at,
                excerpt=metadata.model_dump_json(),
                snapshot_hash=metadata.source_sha256,
                spatial_relevance="regional",
                relevance_note="Regional hourly weather at the array zone; equipment, orientation and "
                "horizon are recorded assumptions. Original weather and modeled hours are preserved.",
            ),
        )
        summary = (
            f"Modeled {metadata.modeled_generation_kwh:,.0f} kWh in "
            f"{metadata.valid_hours:,} of {metadata.expected_hours:,} source hours for {request.year}. "
        )
        summary += (
            "Complete historical year; not a future performance guarantee."
            if metadata.complete_year
            else "Incomplete weather coverage; this sum is not annual generation."
        )
        queue.save_finding(
            db,
            run_id,
            token,
            key,
            FindingContent(
                title=f"Solar generation · {request.year}",
                summary=summary,
                category="physical",
                evidence_ids=[evidence_id],
                confidence="supported" if metadata.valid_hours else "uncertain",
                uncertainty=" ".join(metadata.limitations)[:3000],
                suggested_questions=[
                    "How do equipment, shading and electricity tariffs change this option?"
                ],
            ),
        )
        queue.save_artifact(
            db,
            run_id,
            token,
            key,
            ArtifactContent(
                title=f"Hourly solar assessment · {request.year}",
                method=metadata.algorithm,
                evidence_ids=[evidence_id],
                output=SolarOutput(kind="solar", assessment_id=identifier),
            ),
        )
        value = {
            "provider": request.dataset,
            "status": "available" if metadata.valid_hours else "empty",
            "summary": summary,
            "evidenceIds": [str(evidence_id)],
            "data": {"assessmentId": str(identifier), "metadata": metadata.model_dump(mode="json")},
        }
        state["sources"][key] = value
        queue.checkpoint(db, run_id, token, state)
        return value

    def _raster(
        self,
        db: Session,
        run_id: uuid.UUID,
        token: uuid.UUID,
        workspace_id: uuid.UUID,
        context: SourceContext,
        request: RasterRequest,
        state: dict[str, Any],
        cancelled: threading.Event | None = None,
    ) -> dict[str, Any]:
        categorical = request.dataset == "esa-worldcover-2021"
        vegetation = request.dataset == "sentinel-2-ndvi"
        spec = VEGETATION_SPEC if vegetation else COVER_SPEC if categorical else TERRAIN_SPEC
        # Preserve keys for pre-period terrain/cover runs resumed after this upgrade.
        canonical = request.model_dump_json(exclude=None if vegetation else {"periods"})
        key = "raster/" + hashlib.sha256(canonical.encode()).hexdigest()
        if key in state["sources"]:
            return dict(state["sources"][key])
        identifier = uuid.uuid5(run_id, key)
        row = db.get(LandRaster, identifier)
        if row is None:
            queue.checkpoint(
                db,
                run_id,
                token,
                state,
                kind="analysis",
                payload={
                    "message": "Reading dated satellite reflectance and checking local cloud/quality masks."
                    if vegetation
                    else "Reading mapped land-cover classes inside the selected boundary."
                    if categorical
                    else "Reading surface elevation and calculating slope inside the selected boundary."
                },
            )
            db.rollback()
            try:
                calculated = run_raster(context.boundary, request, cancelled)
            except RasterCancelledError as error:
                raise queue.LeaseLostError() from error
            run = queue.locked(db, run_id, token)
            row = land_rasters.save(
                db,
                workspace_id,
                run,
                identifier,
                request,
                calculated,
                self.settings.land_raster_workspace_quota_bytes,
            )
            # The immutable bytes and metadata commit with the same live lease checkpoint.
            queue.checkpoint(
                db,
                run_id,
                token,
                state,
                kind="analysis",
                payload={
                    "message": "Raster calculation saved; preparing the map and source evidence.",
                    "rasterId": str(identifier),
                },
            )
        metadata = RasterMetadata.model_validate(row.metadata_json)
        ids = []
        for source in metadata.sources:
            item = EvidenceContent(
                provider=request.dataset,
                title=f"{spec.name} · {source.id}",
                url=source.catalog_url,
                license=source.license,
                attribution=source.attribution,
                record_id=source.id,
                retrieved_at=row.created_at,
                observed_at=datetime.fromisoformat(source.catalog_datetime.replace("Z", "+00:00"))
                if vegetation and source.catalog_datetime
                else None,
                excerpt=json.dumps(
                    {
                        "source": source.model_dump(mode="json"),
                        "analysisMethod": metadata.method,
                        "resolutionM": metadata.resolution_m,
                        "outputSha256": row.sha256,
                    }
                ),
                snapshot_hash=row.sha256,
                spatial_relevance="regional",
                relevance_note="Public raster source tile used for the selected land and analysis context. "
                "The clipped output and source version are preserved; these are not surveyed site measurements.",
            )
            ids.append(queue.save_evidence(db, run_id, token, f"{key}/{source.id}", item))
        if vegetation:
            series = metadata.vegetation
            if series is None:
                raise ValueError("The vegetation processor omitted its temporal coverage metadata.")
            observed = sum(item.valid_cells > 0 for item in series.observations)
            summary = (
                f"{observed} of {len(series.observations)} requested windows have clear land NDVI samples. "
                f"{series.common_cells} grid cells are valid in every observation "
                f"({(series.common_coverage_fraction or 0) * 100:.1f}% of boundary cells)."
            )
            if series.mean_change is not None:
                summary += (
                    f" First-to-last mean NDVI difference: {series.mean_change:+.3f}, "
                    f"using {series.change_cells} cells valid at both endpoints."
                )
            summary += " This is a vegetation signal, not species cover or proof of ecological improvement."
        elif categorical:
            if not metadata.valid_cells:
                summary = "No valid 2021 land-cover samples fall inside this boundary at the analysis resolution."
            else:
                leading = sorted(
                    metadata.bands[0].classes, key=lambda item: item.cells, reverse=True
                )[:3]
                composition = ", ".join(
                    f"{item.label}: {(item.fraction or 0) * 100:.1f}%"
                    for item in leading
                    if item.cells
                )
                summary = (
                    f"Sampled 2021 land cover: {composition}. "
                    "These are broad mapped classes, not a species survey."
                )
        else:
            elevation, slope = metadata.bands[:2]
            if elevation.mean is None:
                summary = (
                    "The surface model has no valid analysis samples inside this boundary "
                    "at the selected resolution."
                )
            else:
                summary = (
                    f"Mean sampled surface elevation is {elevation.mean:.1f} m "
                    f"on a {metadata.resolution_m:.1f} m grid."
                )
                if slope.mean is not None:
                    summary += f" Mean sampled surface slope is {slope.mean:.1f} degrees."
                summary += (
                    f" Valid elevation coverage: {(metadata.coverage_fraction or 0) * 100:.1f}% "
                    "of boundary grid cells."
                )
        queue.save_finding(
            db,
            run_id,
            token,
            key,
            FindingContent(
                title=spec.name,
                summary=summary,
                category="ecology" if categorical or vegetation else "physical",
                evidence_ids=ids,
                confidence="supported" if metadata.valid_cells else "uncertain",
                uncertainty=" ".join(metadata.warnings)[:3000],
                suggested_questions=[
                    "How do season, cloud coverage and field observations affect this vegetation comparison?"
                    if vegetation
                    else "What field observations would help assess restoration opportunities in these mapped classes?"
                    if categorical
                    else "Where are the steepest sampled areas, and what would field measurements need to verify?"
                ],
            ),
        )
        queue.save_artifact(
            db,
            run_id,
            token,
            key,
            ArtifactContent(
                title="Vegetation through time"
                if vegetation
                else "Land cover in 2021"
                if categorical
                else "Surface elevation and slope",
                method=metadata.method,
                evidence_ids=ids,
                output=RasterOutput(kind="raster", raster_id=identifier),
            ),
        )
        value = {
            "provider": request.dataset,
            "status": "available" if metadata.valid_cells else "empty",
            "summary": summary,
            "evidenceIds": [str(value) for value in ids],
            "data": {"rasterId": str(identifier), "metadata": metadata.model_dump(mode="json")},
        }
        state["sources"][key] = value
        queue.checkpoint(db, run_id, token, state)
        return value

    def _source(
        self,
        db: Session,
        run_id: uuid.UUID,
        token: uuid.UUID,
        provider: str,
        context: SourceContext,
        client: httpx.Client,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        if provider in state["sources"]:
            return state["sources"][provider]  # type: ignore[no-any-return]
        if provider not in SOURCES:
            raise InvalidInputError("Choose a source from the available registry.")
        cached = state.get("pending_source")
        if cached and cached["provider"] == provider:
            result = unpack(cached)
        else:
            # Persist the exact retrieved snapshot before any derived outputs, so a
            # crash cannot associate newly fetched values with an older evidence ID.
            db.rollback()
            result = retrieve(provider, context, client)
            state["pending_source"] = pack(result)
            queue.checkpoint(
                db,
                run_id,
                token,
                state,
                kind="source",
                payload={"provider": provider, "status": result.status, "message": result.summary},
            )
        ids = [
            queue.save_evidence(db, run_id, token, f"{provider}/{key}", item)
            for key, item in result.evidence
        ]
        finding, artifacts = overview_outputs(result, ids)
        if finding:
            queue.save_finding(db, run_id, token, f"overview/{provider}", finding)
        for index, artifact in enumerate(artifacts):
            queue.save_artifact(db, run_id, token, f"overview/{provider}/{index}", artifact)
        value = {
            "provider": provider,
            "status": result.status,
            "summary": result.summary,
            "evidenceIds": [str(identifier) for identifier in ids],
            "data": result.data,
        }
        state["sources"][provider] = value
        state.pop("pending_source", None)
        queue.checkpoint(db, run_id, token, state)
        return value

    def _mapped_assets(
        self,
        db: Session,
        run_id: uuid.UUID,
        token: uuid.UUID,
        request: CandidateRequest,
        context: SourceContext,
        client: httpx.Client,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        key = "mapped-assets/" + hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        if key in state["sources"]:
            return state["sources"][key]  # type: ignore[no-any-return]
        cached = state.get("pending_mapped_assets")
        if cached and cached["key"] == key:
            result = unpack(cached["result"])
        else:
            db.rollback()
            with self.sessions() as source_db:
                result = inventory_research.mapped_search(source_db, context, request, client)
            state["pending_mapped_assets"] = {"key": key, "result": pack(result)}
            queue.checkpoint(
                db,
                run_id,
                token,
                state,
                kind="source",
                payload={
                    "provider": "mapped-assets",
                    "status": result.status,
                    "message": result.summary,
                },
            )
        ids = [
            queue.save_evidence(db, run_id, token, f"{key}/{identifier}", item)
            for identifier, item in result.evidence
        ]
        candidates = result.data["candidates"]
        for candidate, evidence_id in zip(candidates, ids, strict=True):
            candidate["evidenceId"] = str(evidence_id)
        artifact_id = None
        if ids:
            features = [
                LandCandidate.model_validate(json.loads(item.excerpt)["candidate"])
                for _, item in result.evidence
            ]
            artifact_id = queue.save_artifact(
                db,
                run_id,
                token,
                key + "/map",
                ArtifactContent(
                    title="Mapped asset candidates",
                    method=result.data["limitations"],
                    evidence_ids=ids,
                    output=MapOutput(
                        kind="map",
                        features=[
                            MapFeature(label=candidate.label, geometry=candidate.geometry)
                            for candidate in features
                        ],
                        legend=(
                            "Community-mapped physical features; review identity, location and completeness "
                            "before recording them."
                        ),
                    ),
                ),
            )
        value = {
            "provider": "mapped-assets",
            "status": result.status,
            "summary": result.summary,
            "data": result.data,
            "evidenceIds": [str(identifier) for identifier in ids],
            "artifactId": str(artifact_id) if artifact_id else None,
        }
        state["sources"][key] = value
        state.pop("pending_mapped_assets", None)
        queue.checkpoint(db, run_id, token, state)
        return value

    @staticmethod
    def _taxon_key(query: TaxonQuery) -> str:
        return "taxonomy/" + hashlib.sha256(query.model_dump_json().encode()).hexdigest()

    def _taxonomy(
        self,
        db: Session,
        run_id: uuid.UUID,
        token: uuid.UUID,
        query: TaxonQuery,
        client: httpx.Client,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        key = self._taxon_key(query)
        if key in state["sources"]:
            return state["sources"][key]  # type: ignore[no-any-return]
        cached = state.get("pending_taxonomy")
        if cached and cached["key"] == key:
            result = unpack(cached["result"])
        else:
            db.rollback()
            result = match_taxon(query, client)
            state["pending_taxonomy"] = {"key": key, "result": pack(result)}
            queue.checkpoint(
                db,
                run_id,
                token,
                state,
                kind="source",
                payload={
                    "provider": result.provider,
                    "status": result.status,
                    "message": result.summary,
                },
            )
        ids = [
            queue.save_evidence(db, run_id, token, f"{key}/{suffix}", item)
            for suffix, item in result.evidence
        ]
        finding, artifacts = ecology_outputs(result, ids)
        if finding:
            queue.save_finding(db, run_id, token, key, finding)
        for index, artifact in enumerate(artifacts):
            queue.save_artifact(db, run_id, token, f"{key}/{index}", artifact)
        value = {
            "provider": result.provider,
            "status": result.status,
            "summary": result.summary,
            "evidenceIds": [str(identifier) for identifier in ids],
            "data": result.data,
        }
        state["sources"][key] = value
        state.pop("pending_taxonomy", None)
        queue.checkpoint(db, run_id, token, state)
        return value

    def _ecology(
        self,
        db: Session,
        run_id: uuid.UUID,
        token: uuid.UUID,
        context: SourceContext,
        request: EcologyRequest,
        client: httpx.Client,
        state: dict[str, Any],
        budget: ResearchBudget,
    ) -> None:
        tasks: list[tuple[str, TaxonQuery | None]] = [
            (provider, None)
            for provider, enabled in [
                ("epa-ecoregions", request.include_ecoregions),
                ("usda-ecological-sites", request.include_ecological_sites),
                ("gbif-occurrences", request.include_occurrences),
            ]
            if enabled
        ]
        tasks.extend((self._taxon_key(query), query) for query in request.taxa)
        for key, query in tasks:
            if key in state["sources"]:
                continue
            # Persist the active task with its charged step, so crash recovery doesn't
            # consume another step or associate a new lookup with existing citations.
            if state.get("ecology_task") != key:
                if state["steps"] >= budget.max_steps:
                    queue.finish(
                        db,
                        run_id,
                        token,
                        "partial",
                        "Ecological research reached its step budget. Completed evidence is retained.",
                    )
                    return
                state["steps"] += 1
                state["ecology_task"] = key
                queue.checkpoint(db, run_id, token, state)
            if query is None:
                self._source(db, run_id, token, key, context, client, state)
            else:
                self._taxonomy(db, run_id, token, query, client, state)
            state.pop("ecology_task", None)
            queue.checkpoint(db, run_id, token, state)
        missing = any(value["status"] == "unavailable" for value in state["sources"].values())
        queue.finish(
            db,
            run_id,
            token,
            "partial" if missing else "succeeded",
            "Ecological context is ready. Review source coverage, dates and name matches "
            "before setting restoration targets."
            + (
                " Some sources were unavailable; completed results are retained." if missing else ""
            ),
        )

    def _documents(
        self,
        db: Session,
        run_id: uuid.UUID,
        token: uuid.UUID,
        workspace_id: uuid.UUID,
        land_id: uuid.UUID,
        action: DocumentSearchAction | DocumentReadAction,
        state: dict[str, Any],
    ) -> str:
        key = "documents/" + hashlib.sha256(action.model_dump_json().encode()).hexdigest()
        if key in state["sources"]:
            return "These document passages are already available in retrieved evidence."
        cached = state.get("pending_documents")
        if cached and cached["key"] == key:
            result = unpack(cached["result"])
        else:
            if isinstance(action, DocumentSearchAction):
                result = retrieve_documents(db, workspace_id, land_id, query=action.query)
            else:
                result = retrieve_documents(
                    db,
                    workspace_id,
                    land_id,
                    document_id=action.document_id,
                    first_page=action.first_page,
                    count=action.count,
                    ocr_id=action.ocr_id,
                )
            state["pending_documents"] = {"key": key, "result": pack(result)}
            queue.checkpoint(db, run_id, token, state)
        ids = [
            queue.save_evidence(db, run_id, token, f"{key}/{item_key}", item)
            for item_key, item in result.evidence
        ]
        for page, identifier in zip(result.data["pages"], ids, strict=True):
            page["evidenceId"] = str(identifier)
        state["sources"][key] = {
            "provider": result.provider,
            "status": result.status,
            "summary": result.summary,
            "evidenceIds": [str(identifier) for identifier in ids],
            "data": result.data,
        }
        state.pop("pending_documents", None)
        queue.checkpoint(
            db,
            run_id,
            token,
            state,
            kind="source",
            payload={"provider": "land-documents", "message": result.summary},
        )
        return result.summary

    def _search(
        self,
        db: Session,
        run_id: uuid.UUID,
        token: uuid.UUID,
        action: SearchAction,
        context: SourceContext,
        state: dict[str, Any],
        budget: ResearchBudget,
    ) -> str:
        key = (
            "search/"
            + hashlib.sha256(
                json.dumps([action.query, sorted(action.domains)]).encode()
            ).hexdigest()
        )
        if key in state["sources"]:
            return "This query is already available in retrieved sources."
        cached = state.get("pending_search")
        if cached and cached["key"] == key:
            result = unpack(cached["result"])
        else:
            searcher = self.searcher
            if searcher is None and self.settings.anthropic_api_key:
                searcher = ClaudeResearchSearch(self.settings)
            if searcher is None:
                raise InvalidInputError("Public-source search is not configured.")
            searches = min(2, budget.max_web_searches - state.get("web_searches", 0))
            allocation = min(2000, budget.max_output_tokens - state["output_tokens"])
            if searches <= 0 or allocation < 500:
                raise InvalidInputError("The public-source search budget is exhausted.")
            previous_tokens = state["output_tokens"]
            previous_searches = state.get("web_searches", 0)
            state["output_tokens"] += allocation
            state["web_searches"] = previous_searches + searches
            queue.checkpoint(
                db, run_id, token, state, kind="search", payload={"query": action.query}
            )
            db.rollback()
            try:
                response = searcher.search(
                    action.query, context.geometry.bounds, allocation, searches, action.domains
                )
            except Exception as error:
                raise InvalidInputError(
                    "Public-source search was unavailable. "
                    "Try a registered source or finish with existing evidence."
                ) from error
            state["output_tokens"] = previous_tokens + response.output_tokens
            state["web_searches"] = previous_searches + response.searches
            result = response.result
            state["pending_search"] = {"key": key, "result": pack(result)}
            queue.checkpoint(db, run_id, token, state)
        ids = [
            queue.save_evidence(db, run_id, token, f"{key}/{item_key}", item)
            for item_key, item in result.evidence
        ]
        state["sources"][key] = {
            "provider": result.provider,
            "status": result.status,
            "summary": result.summary,
            "evidenceIds": [str(identifier) for identifier in ids],
            "data": result.data,
        }
        state.pop("pending_search", None)
        queue.checkpoint(db, run_id, token, state)
        return result.summary

    def _image(
        self,
        db: Session,
        run_id: uuid.UUID,
        token: uuid.UUID,
        workspace_id: uuid.UUID,
        evidence_id: uuid.UUID,
        client: httpx.Client,
        state: dict[str, Any],
    ) -> str:
        run = queue.locked(db, run_id, token)
        queue.validate_citations(db, run, [evidence_id])
        evidence = land_archive_images.scoped_evidence(db, workspace_id, evidence_id)
        media = land_archive_images.media(evidence)
        row = db.get(LandArchiveImage, evidence_id)
        if row is None:
            db.rollback()
            snapshot = land_archive_images.retrieve(media, client)
            # Image persistence and checkpoint publication require the still-live run lease.
            queue.locked(db, run_id, token)
            row = land_archive_images.save(
                db,
                workspace_id,
                evidence_id,
                snapshot,
                self.settings.land_archive_workspace_quota_bytes,
            )
            queue.checkpoint(db, run_id, token, state)
        blob = db.get(LandArchiveImageBlob, evidence_id)
        if blob is None:
            raise InvalidInputError("The saved archive image is unavailable.")
        metadata = land_archive_images.read(row)
        preview = blob.preview
        db.rollback()
        _data, vision = land_archive_images.vision_input(preview)
        image = {
            "evidenceId": str(evidence_id),
            "snapshotSha256": metadata.sha256,
            "vision": vision,
        }
        previous_images = [
            item for item in state.get("images", []) if item["evidenceId"] != str(evidence_id)
        ]
        state["images"] = [*previous_images[-1:], image]
        state["sources"][f"image/{evidence_id}"] = {
            "provider": "saved-archive-image",
            "status": "available",
            "summary": "Saved image pixels are attached to subsequent model decisions; inspect only legible details.",
            "evidenceIds": [str(evidence_id)],
            "data": {
                "media": media.model_dump(mode="json"),
                "snapshot": metadata.model_dump(mode="json"),
                "vision": vision,
            },
        }
        queue.checkpoint(db, run_id, token, state, kind="image", payload=image)
        return "The saved archive image is ready for visual inspection; its evidence ID and checksums are recorded."

    def _model_images(
        self,
        db: Session,
        run_id: uuid.UUID,
        token: uuid.UUID,
        workspace_id: uuid.UUID,
        state: dict[str, Any],
    ) -> list[ResearchImage]:
        references = state.get("images", [])
        if len(references) > 2:
            raise InvalidInputError("At most two archive images can be inspected at once.")
        result = []
        for reference in references:
            evidence_id = uuid.UUID(reference["evidenceId"])
            queue.validate_citations(db, queue.locked(db, run_id, token), [evidence_id])
            land_archive_images.scoped_evidence(db, workspace_id, evidence_id)
            row = db.get(LandArchiveImage, evidence_id)
            blob = db.get(LandArchiveImageBlob, evidence_id)
            if (
                row is None
                or blob is None
                or row.metadata_json["sha256"] != reference["snapshotSha256"]
            ):
                raise InvalidInputError(
                    "The inspected image snapshot no longer matches this research run."
                )
            preview = blob.preview
            db.rollback()
            data, vision = land_archive_images.vision_input(preview)
            if vision != reference["vision"]:
                raise InvalidInputError(
                    "The image processor version changed. Start a new investigation run."
                )
            result.append(
                ResearchImage(evidence_id, reference["snapshotSha256"], str(vision["sha256"]), data)
            )
        if result:
            queue.locked(db, run_id, token)
            db.rollback()
        return result

    def _execute(
        self,
        run_id: uuid.UUID,
        token: uuid.UUID,
        client: httpx.Client,
        cancelled: threading.Event | None = None,
    ) -> None:
        with self.sessions() as db:
            run = queue.locked(db, run_id, token)
            investigation = db.get(Investigation, run.investigation_id)
            if investigation is None:
                raise queue.LeaseLostError()
            boundary = db.scalar(
                select(LandBoundaryRevision).where(
                    LandBoundaryRevision.land_id == investigation.land_id,
                    LandBoundaryRevision.revision == investigation.boundary_revision,
                )
            )
            if boundary is None:
                raise queue.LeaseLostError()
            context = SourceContext(FOOTPRINT.validate_python(boundary.boundary))
            budget = ResearchBudget.model_validate(run.budget)
            kind, question = run.kind, run.question
            analysis_request = (
                TypeAdapter(RasterRequest | SolarRequest | EcologyRequest).validate_python(
                    run.analysis
                )
                if run.analysis
                else None
            )
            boundary_revision = investigation.boundary_revision
            land_id, principal_id = investigation.land_id, investigation.created_by
            workspace_id = db.execute(
                select(LandArea.workspace_id).where(LandArea.id == land_id)
            ).scalar_one()
            history = list(
                db.scalars(
                    select(ResearchMessage)
                    .where(ResearchMessage.investigation_id == investigation.id)
                    .order_by(ResearchMessage.created_at.desc(), ResearchMessage.id)
                    .limit(20)
                )
            )
            saved_scenarios = [
                {
                    "id": str(row.id),
                    "revision": row.revision,
                    "boundaryRevision": snapshot.boundary_revision,
                    "inputs": snapshot.payload["inputs"]
                    if len(json.dumps(snapshot.payload["inputs"])) <= 12_000
                    else None,
                    "inputReadTool": "read_scenario",
                    "name": row.name,
                    "summary": snapshot.result["summary"],
                }
                for row, snapshot in db.execute(
                    select(LandScenario, LandScenarioRevision)
                    .join(
                        LandScenarioRevision,
                        (LandScenarioRevision.scenario_id == LandScenario.id)
                        & (LandScenarioRevision.revision == LandScenario.revision),
                    )
                    .where(LandScenario.land_id == land_id)
                    .order_by(LandScenario.updated_at.desc())
                    .limit(10)
                )
            ]
            saved_solar = [
                {
                    "id": str(assessment.id),
                    "year": assessment.request["year"],
                    "capacityKwDc": assessment.metadata_json["capacity_kw_dc"],
                    "completeYear": assessment.metadata_json["complete_year"],
                    "boundaryRevision": assessment.boundary_revision,
                }
                for assessment in db.scalars(
                    select(LandSolar)
                    .where(LandSolar.land_id == land_id)
                    .order_by(LandSolar.created_at.desc())
                    .limit(30)
                )
            ]
            field_surveys = [
                {
                    "id": str(survey.id),
                    "name": survey.content["name"],
                    "observedOn": survey.content["observed_on"],
                    "method": survey.content["method"],
                    "boundaryRevision": survey.boundary_revision,
                    "sha256": survey.sha256,
                    "supersedesId": survey.content.get("supersedes_id"),
                }
                for survey in db.scalars(
                    select(LandSurvey)
                    .where(LandSurvey.land_id == land_id)
                    .order_by(LandSurvey.created_at.desc())
                    .limit(30)
                )
            ]
            saved_documents = [
                {
                    "id": str(document.id),
                    "title": document.metadata_json["title"],
                    "kind": document.metadata_json["kind"],
                    "pages": document.page_count,
                    "status": document.status,
                    "documentDate": document.metadata_json.get("document_date"),
                    "recordedDate": document.metadata_json.get("recorded_date"),
                }
                for document in db.scalars(
                    select(LandDocument)
                    .where(LandDocument.land_id == land_id)
                    .order_by(LandDocument.created_at.desc())
                    .limit(50)
                )
            ]
            document_relationships = [
                relationship.content
                for relationship in db.scalars(
                    select(LandDocumentLink)
                    .where(LandDocumentLink.land_id == land_id)
                    .order_by(LandDocumentLink.created_at.desc())
                    .limit(30)
                )
            ]
            saved_features = [
                {
                    "id": str(feature.id),
                    "revision": feature.revision,
                    "name": feature.content["name"],
                    "category": feature.content["category"],
                    "status": feature.content["status"],
                    "source": feature.content["source"],
                }
                for feature in db.scalars(
                    select(LandFeature)
                    .where(LandFeature.land_id == land_id)
                    .order_by(LandFeature.updated_at.desc())
                    .limit(20)
                )
            ]
            conversation = [
                {"role": message.role, "content": message.content} for message in reversed(history)
            ]
            archive_sources = [
                {"evidenceId": str(item.id), "media": item.content["media"]}
                for item in db.scalars(
                    select(Evidence)
                    .join(ResearchRun, ResearchRun.id == Evidence.run_id)
                    .where(
                        ResearchRun.investigation_id == investigation.id,
                        Evidence.content["media"].astext.is_not(None),
                    )
                    .order_by(Evidence.created_at.desc())
                    .limit(30)
                )
            ]
            saved_research_evidence = [
                {
                    "id": str(item.id),
                    "title": item.content["title"],
                    "provider": item.content["provider"],
                    "retrievedAt": item.content["retrieved_at"],
                    "spatialRelevance": item.content["spatial_relevance"],
                }
                for item in db.scalars(
                    select(Evidence)
                    .join(ResearchRun, ResearchRun.id == Evidence.run_id)
                    .where(ResearchRun.investigation_id == investigation.id)
                    .order_by(Evidence.created_at.desc(), Evidence.id)
                    .limit(150)
                )
            ]
            state = dict(run.checkpoint) or {
                "sources": {},
                "steps": 0,
                "output_tokens": 0,
                "actions": [],
            }
            db.rollback()
            if kind == "ecology":
                if not isinstance(analysis_request, EcologyRequest):
                    raise ValueError("This ecology run has no context parameters.")
                self._ecology(db, run_id, token, context, analysis_request, client, state, budget)
                return
            if kind == "solar":
                try:
                    if not isinstance(analysis_request, SolarRequest):
                        raise ValueError("This solar run has no analysis parameters.")
                    result = self._solar(
                        db, run_id, token, workspace_id, context, analysis_request, state, cancelled
                    )
                    queue.finish(db, run_id, token, "succeeded", result["summary"])
                except ValueError as error:
                    queue.finish(db, run_id, token, "failed", str(error)[:1000])
                return
            if kind == "raster":
                try:
                    if not isinstance(analysis_request, RasterRequest):
                        raise ValueError("This raster run has no analysis parameters.")
                    raster_run_result = self._raster(
                        db, run_id, token, workspace_id, context, analysis_request, state, cancelled
                    )
                    queue.finish(db, run_id, token, "succeeded", raster_run_result["summary"])
                except ValueError as error:
                    queue.finish(db, run_id, token, "failed", str(error)[:1000])
                return
            if kind in {"overview", "archive"}:
                providers = tuple(ARCHIVE_SOURCES) if kind == "archive" else OVERVIEW_SOURCES
                for provider in providers:
                    if provider in state["sources"]:
                        continue
                    if state["steps"] >= budget.max_steps:
                        queue.finish(
                            db,
                            run_id,
                            token,
                            "partial",
                            "The overview reached its step budget. Completed findings are available.",
                        )
                        return
                    state["steps"] += 1
                    queue.checkpoint(db, run_id, token, state)
                    self._source(db, run_id, token, provider, context, client, state)
                missing = [
                    item for item in state["sources"].values() if item["status"] == "unavailable"
                ]
                queue.finish(
                    db,
                    run_id,
                    token,
                    "partial" if missing else "succeeded",
                    (
                        "The archive discovery is ready. Inspect source dates, rights and land relevance."
                        if kind == "archive"
                        else "The open-data overview is ready. Each finding links to its source and limitations."
                    )
                    + (
                        " Some providers were unavailable; their coverage status is retained."
                        if missing
                        else ""
                    ),
                )
                return
            model = self.model
            if model is None and self.settings.anthropic_api_key:
                model = ClaudeResearchModel(self.settings)
            if model is None:
                queue.finish(
                    db,
                    run_id,
                    token,
                    "failed",
                    "AI research is not configured on this server. The open-data overview remains available.",
                )
                return
            while state["steps"] < budget.max_steps or state.get("pending_action"):
                if state.get("pending_action"):
                    decision = ResearchDecision.model_validate(state["pending_action"])
                else:
                    remaining = budget.max_output_tokens - state["output_tokens"]
                    if remaining < 500:
                        break
                    allocation = min(4096, remaining)
                    # Reserve spend before the request. A process crash cannot reset
                    # the consumed step/token budget and make retries unbounded.
                    state["steps"] += 1
                    previous_tokens = state["output_tokens"]
                    state["output_tokens"] += allocation
                    queue.checkpoint(
                        db, run_id, token, state, kind="model", payload={"step": state["steps"]}
                    )
                    prompt = json.dumps(
                        {
                            "question": question,
                            "currentDate": datetime.now(UTC).date().isoformat(),
                            "conversation": conversation,
                            "savedResearchEvidence": saved_research_evidence,
                            "savedScenarios": saved_scenarios,
                            "fieldSurveys": field_surveys,
                            "savedSolarAssessments": saved_solar,
                            "inventoryFeatures": saved_features,
                            "landDocuments": saved_documents,
                            "archiveSources": archive_sources,
                            "attachedArchiveImages": state.get("images", []),
                            "userRecordedDocumentRelationships": document_relationships,
                            "boundaryRevision": boundary_revision,
                            "bounds": context.geometry.bounds,
                            "sourcesAvailable": [asdict(source) for source in SOURCES.values()],
                            "retrieved": state["sources"],
                            "previousActions": state["actions"][-20:],
                            "remainingSteps": budget.max_steps - state["steps"],
                            "remainingWebSearches": budget.max_web_searches
                            - state.get("web_searches", 0),
                        },
                        default=str,
                    )
                    image_inputs = self._model_images(db, run_id, token, workspace_id, state)
                    if image_inputs:
                        if not isinstance(model, MultimodalResearchModel):
                            raise InvalidInputError(
                                "The configured research model cannot inspect images."
                            )
                        response = model.decide_with_images(prompt, allocation, image_inputs)
                    else:
                        response = model.decide(prompt, allocation)
                    state["output_tokens"] = previous_tokens + response.output_tokens
                    decision = response.decision
                    state["pending_action"] = decision.model_dump(mode="json")
                    queue.checkpoint(
                        db,
                        run_id,
                        token,
                        state,
                        kind="progress",
                        payload={"message": decision.progress},
                    )
                action = decision.action
                output_key = f"agent/{state['steps']}"
                try:
                    if isinstance(action, RetrieveAction):
                        self._source(db, run_id, token, action.provider, context, client, state)
                        result = f"Retrieved {action.provider}; see retrieved source data."
                    elif isinstance(action, EvidenceReadAction):
                        current = queue.locked(db, run_id, token)
                        queue.validate_citations(db, current, [action.evidence_id])
                        evidence_row = db.get(Evidence, action.evidence_id)
                        if evidence_row is None:
                            raise InvalidInputError("The source evidence is unavailable.")
                        item = EvidenceContent.model_validate(evidence_row.content)
                        state["sources"][f"saved-evidence/{action.evidence_id}"] = {
                            "provider": item.provider,
                            "status": "available",
                            "evidenceIds": [str(action.evidence_id)],
                            "data": item.model_dump(mode="json"),
                        }
                        queue.checkpoint(db, run_id, token, state)
                        result = "Saved evidence retrieved. Review its content and limitations."

                    elif isinstance(action, TaxonAction):
                        self._taxonomy(db, run_id, token, action.query, client, state)
                        result = (
                            "Taxonomic name lookup complete; see source diagnostics and evidence."
                        )
                    elif isinstance(action, SearchAction):
                        result = self._search(db, run_id, token, action, context, state, budget)
                    elif isinstance(action, ArchiveImageAction):
                        if not isinstance(model, MultimodalResearchModel):
                            raise InvalidInputError(
                                "The configured research model cannot inspect images."
                            )
                        result = self._image(
                            db, run_id, token, workspace_id, action.evidence_id, client, state
                        )
                    elif isinstance(action, SolarReadAction):
                        assessment = land_solar.read(
                            db, land_solar.scoped(db, workspace_id, action.assessment_id)
                        )
                        if assessment.land_id != land_id:
                            raise InvalidInputError("Read solar assessments from this land only.")
                        key = f"saved-solar/{assessment.id}"
                        metadata = assessment.metadata
                        solar_data = {
                            "assessmentId": str(assessment.id),
                            "boundaryRevision": assessment.boundary_revision,
                            "stale": assessment.boundary_revision != boundary_revision,
                            "sha256": assessment.sha256,
                            "request": assessment.request.model_dump(
                                mode="json", exclude={"array_zone", "horizon"}
                            ),
                            "horizonPointCount": len(assessment.request.horizon),
                            "metadata": metadata.model_dump(mode="json"),
                            "geometryNote": "Full array geometry and horizon are in the private assessment archive.",
                        }
                        excerpt = json.dumps(solar_data, ensure_ascii=False)
                        if len(excerpt) > 29_000:
                            raise InvalidInputError(
                                "Assessment metadata exceeds the evidence limit."
                            )
                        identifier = queue.save_evidence(
                            db,
                            run_id,
                            token,
                            key,
                            EvidenceContent(
                                provider="saved-solar-assessment",
                                title=f"Saved hourly solar · {assessment.request.year}",
                                url=metadata.source_url,
                                license=metadata.license,
                                attribution=metadata.attribution,
                                record_id=str(assessment.id),
                                retrieved_at=metadata.retrieved_at,
                                excerpt=excerpt,
                                snapshot_hash=assessment.sha256,
                                spatial_relevance="regional",
                                relevance_note="Regional weather and private entered array assumptions from the "
                                "saved pinned-boundary assessment, not measured roof output.",
                            ),
                        )
                        state["sources"][key] = {
                            "provider": "saved-solar-assessment",
                            "status": "available",
                            "data": solar_data,
                            "evidenceIds": [str(identifier)],
                        }
                        result = "Read saved hourly solar assumptions, output, coverage and source evidence."
                    elif isinstance(action, SolarAction):
                        solar_result = self._solar(
                            db,
                            run_id,
                            token,
                            workspace_id,
                            context,
                            action.analysis,
                            state,
                            cancelled,
                        )
                        result = solar_result["summary"]
                    elif isinstance(action, RasterAction):
                        raster_result = self._raster(
                            db,
                            run_id,
                            token,
                            workspace_id,
                            context,
                            action.analysis,
                            state,
                            cancelled,
                        )
                        result = raster_result["summary"]
                    elif isinstance(action, (DocumentSearchAction, DocumentReadAction)):
                        result = self._documents(
                            db, run_id, token, workspace_id, land_id, action, state
                        )
                    elif isinstance(action, DocumentOcrAction):
                        from app.services import document_ocr

                        ocr = document_ocr.extract(
                            db,
                            workspace_id,
                            land_id,
                            action.document_id,
                            action.page,
                            action.language,
                            commit=False,
                        )
                        result = self._documents(
                            db,
                            run_id,
                            token,
                            workspace_id,
                            land_id,
                            DocumentReadAction(
                                kind="read_document_pages",
                                document_id=action.document_id,
                                first_page=action.page,
                                ocr_id=ocr.id,
                            ),
                            state,
                        )
                    elif isinstance(action, SurveyReadAction):
                        survey = land_surveys.read(
                            db, land_surveys.scoped(db, workspace_id, land_id, action.survey_id)
                        )
                        key = f"field-survey/{survey.id}/{action.offset}/{action.count}"
                        data: dict[str, Any] = {
                            "surveyId": str(survey.id),
                            "sha256": survey.sha256,
                            "boundaryRevision": survey.boundary_revision,
                            "observedOn": str(survey.observed_on),
                            "method": survey.method,
                            "design": survey.design,
                            "methodNotes": survey.method_notes,
                            "assessedStrata": survey.assessed_strata,
                            "observer": survey.observer,
                            "supersedesId": str(survey.supersedes_id)
                            if survey.supersedes_id
                            else None,
                            "sampledFraction": survey.summary.sampled_fraction,
                            "totalSpeciesRows": len(survey.summary.species),
                            "offset": action.offset,
                            "species": [
                                v.model_dump(mode="json")
                                for v in survey.summary.species[
                                    action.offset : action.offset + action.count
                                ]
                            ],
                            "limitations": survey.summary.limitations,
                        }
                        # Bound complete JSON records, not a truncated JSON string. Long Unicode
                        # names/notes can make fewer than the requested rows fit one evidence page.
                        while True:
                            data["nextOffset"] = (
                                action.offset + len(data["species"])
                                if action.offset + len(data["species"])
                                < len(survey.summary.species)
                                else None
                            )
                            excerpt = json.dumps(data, ensure_ascii=False)
                            if len(excerpt) <= 29_000:
                                break
                            if not data["species"]:
                                raise InvalidInputError(
                                    "The survey metadata exceeds the evidence page limit."
                                )
                            data["species"].pop()
                        item = EvidenceContent(
                            provider="field-survey",
                            title=survey.name,
                            survey=SurveyLocator(
                                land_id=land_id, survey_id=survey.id, sha256=survey.sha256
                            ),
                            license="Private workspace field record; no public redistribution permission inferred.",
                            attribution=survey.observer,
                            record_id=str(survey.id),
                            retrieved_at=datetime.now(UTC),
                            observed_at=datetime.combine(
                                survey.observed_on, datetime.min.time(), tzinfo=UTC
                            ),
                            excerpt=excerpt,
                            spatial_relevance="within",
                            relevance_note=(
                                "User-recorded observations within the cited boundary revision. Sample means "
                                "are not whole-land estimates."
                            ),
                            snapshot_hash=survey.sha256,
                        )
                        evidence_id = queue.save_evidence(db, run_id, token, key, item)
                        state["sources"][key] = {
                            "provider": "field-survey",
                            "status": "available",
                            "data": data,
                            "evidenceIds": [str(evidence_id)],
                        }
                        queue.checkpoint(db, run_id, token, state)
                        result = "Field survey page retrieved with immutable source citation."
                    elif isinstance(action, MappedAssetSearchAction):
                        mapped_request = CandidateRequest(
                            kind=action.source_kind,
                            point=action.point or Point(coordinates=list(context.point)),
                            radius_m=action.radius_m,
                        )
                        self._mapped_assets(
                            db, run_id, token, mapped_request, context, client, state
                        )
                        result = (
                            "Mapped asset search completed; inspect retained candidate sources and coverage "
                            "limits."
                        )
                    elif isinstance(action, MappedAssetProposeAction):
                        current = queue.locked(db, run_id, token)
                        queue.validate_citations(db, current, [action.evidence_id])
                        mapped_evidence = db.get(Evidence, action.evidence_id)
                        if mapped_evidence is None:
                            raise InvalidInputError("The mapped candidate is unavailable.")
                        feature_request = inventory_research.mapped_feature(
                            mapped_evidence.content,
                            uuid.uuid5(run_id, output_key),
                            action.evidence_id,
                            action.name,
                            action.category,
                            action.description,
                        )
                        db.execute(
                            select(LandArea.id).where(LandArea.id == land_id).with_for_update()
                        )
                        feature_identifier = uuid.uuid5(
                            land_id, f"feature/{feature_request.request_key}"
                        )
                        duplicate = land_features.duplicate_source(
                            db, land_id, feature_request, feature_identifier
                        )
                        if duplicate:
                            saved_feature = land_features.read(
                                db, land_features.scoped(db, workspace_id, land_id, duplicate)
                            )
                        else:
                            saved_feature = land_features.create(
                                db, workspace_id, land_id, feature_request, commit=False
                            )
                        result = json.dumps(
                            {
                                "featureId": str(saved_feature.id),
                                "revision": saved_feature.revision,
                                "status": saved_feature.status,
                                "existingUnchanged": bool(duplicate),
                                "intersectsLand": saved_feature.intersects_land,
                                "distanceM": saved_feature.distance_m,
                                "note": (
                                    "Review this candidate in Assets. No existing identity or revision was changed."
                                ),
                            }
                        )
                    elif isinstance(action, InventoryReadAction):
                        key = (
                            "inventory/"
                            + hashlib.sha256(action.model_dump_json().encode()).hexdigest()
                        )
                        if key not in state["sources"]:
                            pending = state.get("pending_inventory")
                            if pending and pending["key"] == key:
                                data = pending["data"]
                            else:
                                data = inventory_research.read(
                                    db,
                                    workspace_id,
                                    land_id,
                                    action.feature_id,
                                    action.revision,
                                    action.section,
                                    action.offset,
                                    action.count,
                                    action.as_of,
                                )
                                # Freeze the page before save_evidence commits. A retry must
                                # never pair an older citation with a newly edited asset.
                                state["pending_inventory"] = {"key": key, "data": data}
                                queue.checkpoint(db, run_id, token, state)
                            inventory_ids = []
                            if action.feature_id:
                                excerpt = json.dumps(data, ensure_ascii=False)
                                digest = hashlib.sha256(excerpt.encode()).hexdigest()
                                inventory_item = EvidenceContent(
                                    provider="private-inventory",
                                    title=f"{data['name']} · revision {data['revision']} · {action.section}",
                                    inventory=InventoryLocator(
                                        land_id=land_id,
                                        feature_id=action.feature_id,
                                        revision=data["revision"],
                                        section=action.section,
                                        sha256=digest,
                                    ),
                                    license=(
                                        "Private workspace asset record; no public redistribution permission inferred."
                                    ),
                                    attribution="Workspace inventory and recorded inspections",
                                    retrieved_at=datetime.now(UTC),
                                    excerpt=excerpt,
                                    spatial_relevance="unresolved",
                                    snapshot_hash=digest,
                                    relevance_note=(
                                        "A saved asset revision and bounded data page. Boundary relationship metrics "
                                        "refer to the stated current land revision; records do not independently "
                                        "verify "
                                        "condition, surveyed accuracy or ownership."
                                    ),
                                )
                                inventory_ids.append(
                                    str(queue.save_evidence(db, run_id, token, key, inventory_item))
                                )
                            state["sources"][key] = {
                                "provider": "private-inventory",
                                "status": "available",
                                "data": data,
                                "evidenceIds": inventory_ids,
                            }
                            state.pop("pending_inventory", None)
                            queue.checkpoint(db, run_id, token, state)
                        result = (
                            "Inventory page read. Preserve revision and asOf when following nextOffset; cite "
                            "the returned evidence for specific asset records."
                        )
                    elif isinstance(action, ScenarioReadAction):
                        key = (
                            "saved-scenario/"
                            + hashlib.sha256(action.model_dump_json().encode()).hexdigest()
                        )
                        if key not in state["sources"]:
                            scenario_data = read_scenario(
                                db,
                                workspace_id,
                                land_id,
                                action.scenario_id,
                                action.revision,
                                action.section,
                                action.offset,
                                action.count,
                            )
                            state["sources"][key] = {
                                "provider": "saved-scenario",
                                "status": "available",
                                "data": scenario_data,
                            }
                            queue.checkpoint(db, run_id, token, state)
                        result = "Saved scenario section retrieved; follow nextOffset using its pinned revision."
                    elif isinstance(action, ScenarioAction):
                        current = queue.locked(db, run_id, token)
                        scenario_citations = {uuid.UUID(value) for value in action.evidence_ids}
                        if action.inputs.kind == "restoration" and action.inputs.ecology:
                            scenario_citations.update(action.inputs.ecology.evidence_ids())
                        queue.validate_citations(db, current, list(scenario_citations))
                        scenario = scenarios.create(
                            db,
                            workspace_id,
                            land_id,
                            principal_id,
                            ScenarioCreate(
                                request_key=uuid.uuid5(run_id, output_key),
                                name=action.name,
                                boundary_revision=boundary_revision,
                                inputs=action.inputs,
                                field_survey_ids=action.field_survey_ids,
                                solar_assessment_id=action.solar_assessment_id,
                                evidence_ids=[uuid.UUID(value) for value in action.evidence_ids],
                            ),
                            identifier=uuid.uuid5(run_id, output_key),
                            commit=False,
                        )
                        scenario_reply = {
                            "scenarioId": str(scenario.id),
                            "revision": scenario.revision,
                            "inputs": scenario.inputs.model_dump(mode="json"),
                            "result": scenario.result.model_dump(mode="json"),
                        }
                        if len(json.dumps(scenario_reply)) > 29_000:
                            scenario_reply = {
                                "scenarioId": str(scenario.id),
                                "revision": scenario.revision,
                                "summary": scenario.result.summary,
                                "readTool": "read_scenario",
                                "note": "Use the paged read tool for full saved assumptions and results.",
                            }
                        result = json.dumps(scenario_reply)
                    elif isinstance(action, ActionDraftAction):
                        current = queue.locked(db, run_id, token)
                        action_citations = [
                            *action.draft.evidence_ids,
                            *[
                                identifier
                                for constraint in action.draft.constraints
                                for identifier in constraint.evidence_ids
                            ],
                        ]
                        queue.validate_citations(db, current, action_citations)
                        draft = action.draft.model_copy(
                            update={
                                "request_key": uuid.uuid5(run_id, output_key),
                                "boundary_revision": boundary_revision,
                            }
                        )
                        saved_action = land_actions.create(
                            db, workspace_id, land_id, principal_id, draft, commit=False
                        )
                        result = json.dumps(
                            {
                                "actionId": str(saved_action.id),
                                "revision": saved_action.revision,
                                "status": saved_action.status,
                                "staleReasons": saved_action.stale_reasons,
                                "knownCost": saved_action.total_known_cost,
                                "uncostedSteps": saved_action.uncosted_steps,
                            }
                        )
                    elif isinstance(action, FindingAction):
                        identifier = queue.save_finding(
                            db, run_id, token, output_key, action.finding
                        )
                        result = f"Finding published: {identifier}"
                    elif isinstance(action, ArtifactAction):
                        identifier = queue.save_artifact(
                            db, run_id, token, output_key, action.artifact
                        )
                        result = f"Artifact published: {identifier}"
                    elif isinstance(action, CompleteAction):
                        run = queue.locked(db, run_id, token)
                        queue.validate_citations(
                            db, run, [uuid.UUID(value) for value in action.evidence_ids]
                        )
                        # References are part of the persisted answer, not discarded.
                        citations = (
                            "\n\nEvidence: " + ", ".join(action.evidence_ids)
                            if action.evidence_ids
                            else ""
                        )
                        queue.finish(db, run_id, token, "succeeded", action.summary + citations)
                        return
                except (ValueError, NotFoundError, ConflictError) as error:
                    db.rollback()
                    result = str(error)[:1000]
                state["actions"].append({"action": action.kind, "result": result})
                state.pop("pending_action", None)
                queue.checkpoint(
                    db,
                    run_id,
                    token,
                    state,
                    kind="tool-result",
                    payload={"action": action.kind, "result": result},
                )
            queue.finish(
                db,
                run_id,
                token,
                "partial",
                "The research budget was reached. Completed findings and visual outputs have been retained.",
            )

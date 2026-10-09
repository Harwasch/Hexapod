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
from app.analysis.terrain import SPEC as TERRAIN_SPEC
from app.analysis.worldcover import SPEC as COVER_SPEC
from app.config import Settings
from app.models.land import LandArea, LandBoundaryRevision
from app.models.land_document import LandDocument, LandDocumentLink
from app.models.land_feature import LandFeature
from app.models.land_raster import LandRaster
from app.models.research import Investigation, ResearchMessage
from app.models.scenario import LandScenario, LandScenarioRevision
from app.research import queue
from app.research.documents import retrieve as retrieve_documents
from app.research.model import (
    ActionDraftAction,
    ArtifactAction,
    ClaudeResearchModel,
    CompleteAction,
    DocumentOcrAction,
    DocumentReadAction,
    DocumentSearchAction,
    FindingAction,
    RasterAction,
    ResearchDecision,
    ResearchModel,
    RetrieveAction,
    ScenarioAction,
    SearchAction,
)
from app.research.outputs import overview_outputs
from app.research.providers.archives import ARCHIVE_SOURCES
from app.research.providers.base import SourceContext, SourceResult
from app.research.providers.open_data import OVERVIEW_SOURCES, SOURCES, retrieve
from app.research.search import ClaudeResearchSearch, ResearchSearch
from app.schemas.geojson import Footprint
from app.schemas.land_rasters import RasterMetadata, RasterRequest
from app.schemas.research import (
    ArtifactContent,
    EvidenceContent,
    FindingContent,
    RasterOutput,
    ResearchBudget,
)
from app.schemas.scenarios import ScenarioCreate
from app.services import land_actions, land_rasters, scenarios
from app.services.errors import InvalidInputError, NotFoundError

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
        spec = COVER_SPEC if categorical else TERRAIN_SPEC
        key = "raster/" + hashlib.sha256(request.model_dump_json().encode()).hexdigest()
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
                    "message": "Reading mapped land-cover classes inside the selected boundary."
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
        if categorical:
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
                category="ecology" if categorical else "physical",
                evidence_ids=ids,
                confidence="supported" if metadata.valid_cells else "uncertain",
                uncertainty=" ".join(metadata.warnings)[:3000],
                suggested_questions=[
                    "What field observations would help assess restoration opportunities in these mapped classes?"
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
                title="Land cover in 2021" if categorical else "Surface elevation and slope",
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
            raster_request = RasterRequest.model_validate(run.analysis) if run.analysis else None
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
                    "inputs": snapshot.payload["inputs"],
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
            state = dict(run.checkpoint) or {
                "sources": {},
                "steps": 0,
                "output_tokens": 0,
                "actions": [],
            }
            db.rollback()
            if kind == "raster":
                try:
                    if raster_request is None:
                        raise ValueError("This raster run has no analysis parameters.")
                    raster_run_result = self._raster(
                        db, run_id, token, workspace_id, context, raster_request, state, cancelled
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
                            "conversation": conversation,
                            "savedScenarios": saved_scenarios,
                            "inventoryFeatures": saved_features,
                            "landDocuments": saved_documents,
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
                    elif isinstance(action, SearchAction):
                        result = self._search(db, run_id, token, action, context, state, budget)
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
                    elif isinstance(action, ScenarioAction):
                        current = queue.locked(db, run_id, token)
                        queue.validate_citations(
                            db, current, [uuid.UUID(value) for value in action.evidence_ids]
                        )
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
                                evidence_ids=[uuid.UUID(value) for value in action.evidence_ids],
                            ),
                            identifier=uuid.uuid5(run_id, output_key),
                            commit=False,
                        )
                        result = json.dumps(
                            {
                                "scenarioId": str(scenario.id),
                                "revision": scenario.revision,
                                "inputs": scenario.inputs.model_dump(mode="json"),
                                "result": scenario.result.model_dump(mode="json"),
                            }
                        )
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
                except (ValueError, NotFoundError) as error:
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

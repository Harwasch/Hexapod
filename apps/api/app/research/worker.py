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

from app.config import Settings
from app.models.land import LandArea, LandBoundaryRevision
from app.models.research import Investigation, ResearchMessage
from app.models.scenario import LandScenario, LandScenarioRevision
from app.research import queue
from app.research.model import (
    ArtifactAction,
    ClaudeResearchModel,
    CompleteAction,
    FindingAction,
    ResearchDecision,
    ResearchModel,
    RetrieveAction,
    ScenarioAction,
    SearchAction,
)
from app.research.outputs import overview_outputs
from app.research.providers.base import SourceContext, SourceResult
from app.research.providers.open_data import SOURCES, retrieve
from app.research.search import ClaudeResearchSearch, ResearchSearch
from app.schemas.geojson import Footprint
from app.schemas.research import EvidenceContent, ResearchBudget
from app.schemas.scenarios import ScenarioCreate
from app.services import scenarios
from app.services.errors import InvalidInputError

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
                        return
                    queue.heartbeat(db, run_id, token)
            except queue.LeaseLostError:
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
            self._execute(run_id, token, client)
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

    def _execute(self, run_id: uuid.UUID, token: uuid.UUID, client: httpx.Client) -> None:
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
            if kind == "overview":
                for provider in SOURCES:
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
                    "The open-data overview is ready. Each finding links to its source and limitations."
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
                except (InvalidInputError, ValueError) as error:
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

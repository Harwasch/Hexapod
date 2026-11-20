"""Write-authenticated question stream and scan-scoped observation CRUD."""

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from app.api.deps import DbSession, RequireWriteOrPhoneKey, RequireWriteToken, SettingsDep, Storage
from app.models import Capture, ScanEntity
from app.models.enums import UploadStatus
from app.schemas.ask import AskEvent, AskRequest, DoneEvent, Entity, StatusEvent
from app.schemas.base import CamelModel
from app.services import ask_entities as ledger
from app.services import phone_access
from app.services.ask_analysis import ModalAnalysis
from app.services.ask_reasoner import Answering, Reasoner, ScriptedAnswerModel, ToolModel
from app.services.ask_replay import ReplayAnswerer, recorded_events
from app.services.errors import InvalidInputError, UnauthorizedError
from app.services.property_memory import PostgresMemory

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ask"])


def reasoner(settings: SettingsDep, storage: Storage, db: DbSession) -> Answering:
    if settings.ask_mode == "replay":
        return ReplayAnswerer(PostgresMemory(db, release_reads=True))
    model = ScriptedAnswerModel() if settings.ask_reasoner == "mock" else ToolModel(settings)
    return Reasoner(model, ModalAnalysis(settings, storage), PostgresMemory(db, release_reads=True))


ReasonerDep = Annotated[Answering, Depends(reasoner)]


class EntityRecord(CamelModel):
    id: uuid.UUID
    scan_id: uuid.UUID
    site_id: uuid.UUID | None = None
    entity: Entity


def envelope(row: ScanEntity, capture: Capture) -> EntityRecord:
    # Captures may acquire their site after bundle processing; read the current site.
    return EntityRecord(
        id=row.id,
        scan_id=capture.id,
        site_id=capture.site_id,
        entity=Entity.model_validate(row.entity),
    )


def required_capture(db: DbSession, capture_id: uuid.UUID, site_id: uuid.UUID | None) -> Capture:
    capture = ledger.capture_scope(db, str(capture_id))
    assert capture is not None
    ledger.check_site(capture, site_id)
    return capture


@router.get(
    "/captures/{capture_id}/entities",
    response_model=list[EntityRecord],
    response_model_exclude_none=True,
    dependencies=[RequireWriteToken],
)
def list_entities(
    db: DbSession,
    capture_id: uuid.UUID,
    site_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    after: uuid.UUID | None = None,
) -> list[EntityRecord]:
    capture = required_capture(db, capture_id, site_id)
    return [envelope(row, capture) for row in ledger.rows(db, capture, limit, after)]


@router.post(
    "/captures/{capture_id}/entities",
    response_model=EntityRecord,
    response_model_exclude_none=True,
    dependencies=[RequireWriteToken],
    status_code=201,
)
def create_entity(
    db: DbSession, capture_id: uuid.UUID, payload: Entity, site_id: uuid.UUID | None = None
) -> EntityRecord:
    capture = required_capture(db, capture_id, site_id)
    row = ledger.add(db, capture, payload)
    db.commit()
    return envelope(row, capture)


@router.get(
    "/captures/{capture_id}/entities/{entity_id}",
    response_model=EntityRecord,
    response_model_exclude_none=True,
    dependencies=[RequireWriteToken],
)
def read_entity(
    db: DbSession, capture_id: uuid.UUID, entity_id: uuid.UUID, site_id: uuid.UUID | None = None
) -> EntityRecord:
    capture = required_capture(db, capture_id, site_id)
    return envelope(ledger.get_row(db, capture, entity_id), capture)


@router.put(
    "/captures/{capture_id}/entities/{entity_id}",
    response_model=EntityRecord,
    response_model_exclude_none=True,
    dependencies=[RequireWriteToken],
)
def update_entity(
    db: DbSession,
    capture_id: uuid.UUID,
    entity_id: uuid.UUID,
    payload: Entity,
    site_id: uuid.UUID | None = None,
) -> EntityRecord:
    capture = required_capture(db, capture_id, site_id)
    row = ledger.get_row(db, capture, entity_id)
    ledger.assign(row, ledger.prepare(capture, payload))
    ledger.link_identity(db, capture, row)
    row.site_id = capture.site_id
    db.commit()
    return envelope(row, capture)


@router.delete(
    "/captures/{capture_id}/entities/{entity_id}", status_code=204, dependencies=[RequireWriteToken]
)
def delete_entity(
    db: DbSession, capture_id: uuid.UUID, entity_id: uuid.UUID, site_id: uuid.UUID | None = None
) -> Response:
    capture = required_capture(db, capture_id, site_id)
    db.delete(ledger.get_row(db, capture, entity_id))
    db.commit()
    return Response(status_code=204)


class EventStreamResponse(StreamingResponse):
    media_type = "text/event-stream"


@router.post(
    "/ask",
    dependencies=[RequireWriteOrPhoneKey],
    response_class=EventStreamResponse,
    responses={
        200: {"model": AskEvent, "description": "One AskEvent JSON per SSE data line"},
        503: {"description": "Reasoner is not configured"},
    },
)
async def ask(
    payload: AskRequest,
    request: Request,
    db: DbSession,
    settings: SettingsDep,
    storage: Storage,
    engine: ReasonerDep,
) -> StreamingResponse:
    # Limits constrain work, not the shared wire shape.
    if (
        len(payload.keyframes) > 8
        or len(payload.device_entities) > 256
        or len(payload.question) > 8192
        or len(payload.id) > 256
    ):
        raise InvalidInputError("question exceeds analysis resource limits")
    is_phone = getattr(request.state, "api_credential", None) == "phone"
    capture: Capture | None
    if is_phone and payload.scan_id is not None:
        try:
            capture_id = uuid.UUID(payload.scan_id)
        except ValueError as exc:
            raise InvalidInputError("scanId must be a capture UUID") from exc
        capture = phone_access.capture(db, capture_id)
    else:
        capture = ledger.capture_scope(db, payload.scan_id)
    if is_phone:
        if payload.scope != "property" and capture is None:
            raise UnauthorizedError("That phone key is not right.")
        if payload.site_id is not None:
            phone_access.site(db, uuid.UUID(payload.site_id))
    ledger.request_site(db, payload, capture)
    if capture is None and (payload.keyframes or payload.device_entities):
        raise InvalidInputError("scene evidence requires scanId")
    run, replay = ledger.claim(db, payload, capture)
    bundle_url = None
    previous: list[Entity] = []
    if not replay:
        try:
            if settings.ask_mode == "replay":
                if payload.scope != "property":
                    recorded_events(payload.question)
                if capture is None and payload.scope != "property":
                    raise InvalidInputError("synthetic replay entities require scanId")
            if (
                settings.ask_mode == "live"
                and settings.ask_reasoner != "mock"
                and not settings.anthropic_api_key
            ):
                raise HTTPException(503, "Reasoner is not configured")
            if settings.ask_mode == "live" and capture is not None and payload.keyframes:
                files = [
                    f
                    for f in capture.files
                    if f.status == UploadStatus.COMPLETE and f.filename == "capture.hexcap"
                ]
                if capture.metadata_.get("bundleFormat") != "hex-capture-v1" or len(files) != 1:
                    raise InvalidInputError("keyframes require one completed capture bundle")
                bundle_url = storage.presign_get(files[0].storage_key, expires_in=900)
            previous = (
                [Entity.model_validate(r.entity) for r in ledger.rows(db, capture, 200)]
                if capture
                else []
            )
            # Never keep an idle read transaction open during model/remote calls.
            db.commit()
        except BaseException:
            ledger.fail(db, run)
            raise

    async def stream() -> AsyncIterator[str]:
        if replay:
            for saved_event in run.events:
                value = ledger.EVENT.validate_python(saved_event)
                yield f"data: {value.model_dump_json(by_alias=True, exclude_none=True)}\n\n"
            return
        events = []
        completed = False
        try:
            async for event in engine.answer(payload, previous, bundle_url):
                if await request.is_disconnected():
                    return
                if event.type == "entity" and capture is None:
                    raise InvalidInputError("analysis entities require a capture")
                if event.type == "entity" and capture is not None:
                    event = event.model_copy(
                        update={"entity": ledger.prepare(capture, event.entity)}
                    )
                events.append(event.model_dump(mode="json", by_alias=True, exclude_none=True))
                yield f"data: {event.model_dump_json(by_alias=True, exclude_none=True)}\n\n"
            done = DoneEvent()
            events.append(done.model_dump())
            ledger.finish(db, run, events, capture)
            completed = True
            yield f"data: {done.model_dump_json()}\n\n"
        except Exception as exc:
            # Provider messages can include signed URLs; log only the exception class.
            logger.warning("ask interrupted (%s)", type(exc).__name__)
            yield f"data: {StatusEvent(text='Answer interrupted').model_dump_json()}\n\n"
        finally:
            if not completed:
                ledger.fail(db, run)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )

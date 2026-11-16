"""Offline contract, tool-loop and stream tests; provider clients never reach the network."""

import asyncio
import copy
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.deps import _db
from app.api.v1.ask import reasoner
from app.config import Settings
from app.main import create_app
from app.models import AskRun
from app.schemas.ask import AskRequest, DoneEvent, Entity, EntityEvent, SpeechEvent, StatusEvent
from app.services import ask_entities as ledger
from app.services.ask_analysis import AnalysisUnavailableError, ModalAnalysis
from app.services.ask_reasoner import Reasoner, ToolModel

ENTITY: dict[str, Any] = {
    "label": "plant",
    "anchor": {"world": {"x": 1, "y": 2, "z": -3}},
    "extent": {"x": 1, "y": 2, "z": 0},
    "masks": [],
    "confidence": 0.8,
    "attrs": {"nested": [True, None, "observation", 2.5]},
    "source": {"module": "test", "model": "fixture"},
}
REQUEST: dict[str, Any] = {
    "id": "question-1",
    "question": "What is this?",
    "keyframes": [],
    "deviceEntities": [],
}


def collect(iterator: Any) -> Any:

    async def run() -> Any:
        return [event async for event in iterator]

    return asyncio.run(run())


def test_wire_shapes_roundtrip_and_validation() -> None:
    entity = Entity.model_validate(ENTITY)
    assert entity.model_dump(mode="json", exclude_none=True) == ENTITY
    assert (
        AskRequest.model_validate(REQUEST).model_dump(mode="json", by_alias=True, exclude_none=True)
        == REQUEST
    )
    for value in [
        EntityEvent(entity=entity),
        SpeechEvent(text="One plant."),
        StatusEvent(text="Looking"),
        DoneEvent(),
    ]:
        assert ledger.EVENT.validate_json(value.model_dump_json(exclude_none=True)) == value
    for changes in (
        {"confidence": 1.1},
        {"extent": {"x": -1, "y": 0, "z": 0}},
        {"extra": 1},
        {"anchor": {"world": {"x": float("nan"), "y": 0, "z": 0}}},
    ):
        with pytest.raises(ValidationError):
            Entity.model_validate({**ENTITY, **changes})
    with pytest.raises(ValidationError):
        AskRequest.model_validate(
            {**REQUEST, "keyframes": [{"index": 0, "image": "a", "pose": "wrong"}]}
        )


class Model:
    def __init__(self, responses: Any) -> None:
        self.responses = iter(responses)
        self.messages: list[Any] = []

    async def turn(self, messages: Any) -> Any:
        self.messages.append(copy.deepcopy(messages))
        return next(self.responses)


class Analysis:
    async def context(self, request: Any, bundle_url: Any) -> Any:
        return [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "fixture"},
            }
        ]

    async def run(self, name: Any, arguments: Any, request: Any, bundle_url: Any) -> Any:
        assert name == "count_measure"
        assert arguments == {"prompt": "plant"}
        return [Entity.model_validate(ENTITY)]


TOOL_TURN = {
    "stop_reason": "tool_use",
    "content": [
        {"type": "tool_use", "id": "call-1", "name": "count_measure", "input": {"prompt": "plant"}}
    ],
}
FINAL_TURN = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "One plant."}]}


def test_tool_loop_sees_pixels_masks_depth_context_and_previous_entities() -> None:
    model = Model([TOOL_TURN, FINAL_TURN])
    events = collect(
        Reasoner(model, Analysis()).answer(
            AskRequest.model_validate(REQUEST), [Entity.model_validate(ENTITY)], "private-url"
        )
    )
    assert [event.type for event in events] == ["status", "status", "entity", "speech"]
    assert events[-1].text == "One plant."
    context = model.messages[0][0]["content"]
    assert context[1]["type"] == "image"
    assert json.loads(context[0]["text"])["earlierEntities"] == [ENTITY]
    assert "private-url" not in json.dumps(model.messages)
    result = model.messages[1][-1]["content"][0]
    assert result["tool_use_id"] == "call-1"
    assert json.loads(result["content"])["entities"] == [ENTITY]


def test_tool_failure_is_reported_to_model_without_fake_entities() -> None:
    analysis = MagicMock(spec=Analysis)
    analysis.run = AsyncMock(side_effect=AnalysisUnavailableError("private details"))
    analysis.context = AsyncMock(side_effect=AnalysisUnavailableError("unconfigured"))
    model = Model([TOOL_TURN, FINAL_TURN])
    events = collect(Reasoner(model, analysis).answer(AskRequest.model_validate(REQUEST), [], None))
    assert all(event.type != "entity" for event in events)
    assert model.messages[-1][-1]["content"][0]["is_error"] is True
    assert "private details" not in json.dumps(model.messages)


def test_tool_loop_is_bounded_and_truncation_is_not_spoken() -> None:
    with pytest.raises(RuntimeError, match="turn limit"):
        collect(
            Reasoner(Model([TOOL_TURN] * 8), Analysis()).answer(
                AskRequest.model_validate(REQUEST), [], None
            )
        )
    with pytest.raises(RuntimeError, match="did not finish"):
        collect(
            Reasoner(Model([{**FINAL_TURN, "stop_reason": "max_tokens"}]), Analysis()).answer(
                AskRequest.model_validate(REQUEST), [], None
            )
        )


def test_sdk_client_is_fully_mocked(monkeypatch: Any) -> None:
    response = MagicMock()
    response.model_dump.return_value = FINAL_TURN
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr("anthropic.AsyncAnthropic", constructor)
    settings = Settings(_env_file=None, anthropic_api_key="fixture-key", ask_model="fixture-model")
    assert asyncio.run(ToolModel(settings).turn([])) == FINAL_TURN
    assert client.messages.create.call_args.kwargs["model"] == "fixture-model"
    assert constructor.call_args.kwargs["max_retries"] == 0


def offline_client(
    monkeypatch: Any, *, replay: Any = None, failure: Any = False, model_configured: Any = True
) -> Any:
    settings = Settings(
        _env_file=None,
        api_write_token="fixture-write",
        anthropic_api_key="fixture-key" if model_configured else None,
    )
    app = create_app(settings)
    db = MagicMock()
    app.dependency_overrides[_db] = lambda: db
    app.dependency_overrides[reasoner] = lambda: Reasoner(Model([FINAL_TURN]), Analysis())
    run = AskRun(id="question-1", events=replay or [], state="complete" if replay else "running")
    monkeypatch.setattr(ledger, "claim", lambda *_: (run, bool(replay)))
    finish = MagicMock(side_effect=RuntimeError("private") if failure else None)
    monkeypatch.setattr(ledger, "finish", finish)
    return (TestClient(app), db, run, finish)


def events(response: Any) -> Any:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.mark.parametrize("path", ["/api/v1/ask", "/v1/ask"])
def test_stream_auth_framing_and_done_after_commit(monkeypatch: Any, path: Any) -> None:
    client, _, _, finish = offline_client(monkeypatch)
    assert client.post(path, json=REQUEST).status_code == 401
    response = client.post(path, json=REQUEST, headers={"Authorization": "Bearer fixture-write"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert events(response) == [
        {"type": "status", "text": "Looking"},
        {"type": "speech", "text": "One plant."},
        {"type": "done"},
    ]
    assert finish.call_count == 1
    assert finish.call_args.args[2][-1] == {"type": "done"}


def test_failed_commit_has_no_done_and_marks_failed(monkeypatch: Any) -> None:
    client, db, run, _ = offline_client(monkeypatch, failure=True)
    response = client.post(
        "/v1/ask", json=REQUEST, headers={"Authorization": "Bearer fixture-write"}
    )
    assert events(response)[-1] == {"type": "status", "text": "Answer interrupted"}
    assert all(event["type"] != "done" for event in events(response))
    assert run.state == "failed"
    db.rollback.assert_called()


def test_completed_answer_replays_without_reasoning(monkeypatch: Any) -> None:
    replay = [{"type": "speech", "text": "Saved."}, {"type": "done"}]
    client, _, _, finish = offline_client(monkeypatch, replay=replay)
    response = client.post(
        "/v1/ask", json=REQUEST, headers={"Authorization": "Bearer fixture-write"}
    )
    assert events(response) == replay
    finish.assert_not_called()


def test_unconfigured_reasoner_fails_before_streaming(monkeypatch: Any) -> None:
    client, _, _, finish = offline_client(monkeypatch, model_configured=False)
    assert (
        client.post(
            "/v1/ask", json=REQUEST, headers={"Authorization": "Bearer fixture-write"}
        ).status_code
        == 503
    )
    finish.assert_not_called()


def test_analysis_disabled_never_initializes_provider() -> None:
    adapter = ModalAnalysis(Settings(_env_file=None, ask_analysis_enabled=False), MagicMock())
    with pytest.raises(AnalysisUnavailableError, match="disabled"):
        asyncio.run(adapter.call({}))


def test_analysis_rejects_unknown_tools_before_remote_call() -> None:
    adapter = ModalAnalysis(Settings(_env_file=None), MagicMock())
    with pytest.raises(ValueError, match="unknown"):
        asyncio.run(adapter.run("shell", {}, AskRequest.model_validate(REQUEST), None))


def test_openapi_documents_event_union_under_sse_content_type() -> None:
    document = create_app(Settings(_env_file=None)).openapi()
    schema = document["paths"]["/api/v1/ask"]["post"]["responses"]["200"]["content"][
        "text/event-stream"
    ]["schema"]
    assert "oneOf" in schema
    assert schema["discriminator"]["propertyName"] == "type"
    for name in ("EntityEvent", "SpeechEvent", "StatusEvent", "DoneEvent"):
        assert name in document["components"]["schemas"]


def test_analysis_adapter_stores_masks_and_never_exposes_bundle_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64

    storage = MagicMock()
    adapter = ModalAnalysis(Settings(_env_file=None), storage)
    entity = {**ENTITY, "masks": [{"keyframe": "frames.jsonl#0", "ref": "analysis/mask-0-0.png"}]}
    remote = AsyncMock(
        return_value={
            "entities": [entity],
            "artifacts": [
                {"ref": "analysis/mask-0-0.png", "data": base64.b64encode(b"png").decode()}
            ],
        }
    )
    monkeypatch.setattr(adapter, "call", remote)
    request = AskRequest.model_validate(
        {**REQUEST, "scanId": "00000000-0000-0000-0000-000000000001"}
    )
    result = asyncio.run(adapter.run("segment", {"prompt": "plant"}, request, "private-url"))
    assert result[0].masks[0].ref.startswith("analysis/")
    assert result[0].masks[0].ref.endswith("mask-0-0.png")
    assert "private-url" not in result[0].model_dump_json()
    assert storage.put_object.call_args.args[1:] == (b"png", "image/png")
    assert remote.call_args.args[0]["bundleUrl"] == "private-url"


def test_analysis_adapter_rejects_unsafe_or_missing_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    for ref, artifacts in [
        ("../../secret", [{"ref": "../../secret", "data": ""}]),
        ("analysis/mask-0-0.png", []),
    ]:
        adapter = ModalAnalysis(Settings(_env_file=None), MagicMock())
        remote = AsyncMock(
            return_value={
                "entities": [{**ENTITY, "masks": [{"keyframe": "frames.jsonl#0", "ref": ref}]}],
                "artifacts": artifacts,
            }
        )
        monkeypatch.setattr(adapter, "call", remote)
        with pytest.raises(ValueError):
            asyncio.run(
                adapter.run(
                    "segment",
                    {"prompt": "plant"},
                    AskRequest.model_validate(REQUEST),
                    "private-url",
                )
            )


def test_analysis_cancellation_terminates_remote_call(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    call = MagicMock()
    call.get.aio = AsyncMock(side_effect=asyncio.CancelledError())
    call.cancel.aio = AsyncMock()
    function = MagicMock()
    function.spawn.aio = AsyncMock(return_value=call)
    client.Function.from_name.return_value = function
    monkeypatch.setattr("app.services.ask_analysis.importlib.import_module", lambda _: client)
    adapter = ModalAnalysis(Settings(_env_file=None, ask_analysis_enabled=True), MagicMock())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(adapter.call({"action": "context"}))
    call.cancel.aio.assert_awaited_once_with(terminate_containers=True)


def test_api_and_remote_tool_descriptors_agree() -> None:
    import runpy
    import sys

    from app.config import REPO_ROOT
    from app.services.ask_analysis import TOOLS

    root = str(REPO_ROOT / "tools" / "analysis")
    sys.path.insert(0, root)
    try:
        assert runpy.run_module("hex_analysis.registry")["TOOLS"] == TOOLS
    finally:
        sys.path.remove(root)


def test_wire_omission_and_numeric_types_match_shared_schema() -> None:
    for change in ({"scanId": None}, {"location": None}):
        with pytest.raises(ValidationError):
            AskRequest.model_validate({**REQUEST, **change})
    for confidence in (True, "0.5"):
        with pytest.raises(ValidationError):
            Entity.model_validate({**ENTITY, "confidence": confidence})


def test_saved_answers_replay_without_model_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = [{"type": "speech", "text": "Saved."}, {"type": "done"}]
    client, _, _, finish = offline_client(monkeypatch, replay=replay, model_configured=False)
    response = client.post(
        "/v1/ask", json=REQUEST, headers={"Authorization": "Bearer fixture-write"}
    )
    assert events(response) == replay
    finish.assert_not_called()

"""Frame-grounded guidance. Endpoints propose events; only the player applies them."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.api.deps import RequireWriteToken
from app.worlds import vision
from app.worlds.request_limits import AuthenticatedBodyRoute

router = APIRouter(
    prefix="/intelligence", dependencies=[RequireWriteToken], route_class=AuthenticatedBodyRoute
)
ShortText = Annotated[str, Field(max_length=100)]


class SceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=8000)
    modelId: str = Field(min_length=1, max_length=100)  # noqa: N815
    capabilities: dict[str, Any] = Field(default_factory=dict)
    nativeActions: list[ShortText] = Field(default_factory=list, max_length=64)  # noqa: N815
    frame: str = Field(min_length=1, max_length=2_800_000)
    objective: str = Field(default="", max_length=1000)
    elapsedSeconds: float = Field(default=0, ge=0, le=86400)  # noqa: N815
    recentEvents: list[Annotated[str, Field(max_length=1000)]] = Field(  # noqa: N815
        default_factory=list, max_length=20
    )

    @model_validator(mode="after")
    def bound_capabilities(self) -> SceneRequest:
        if len(json.dumps(self.capabilities)) > 16_000:
            raise ValueError("Capabilities exceed the request limit")
        return self


class CommandRequest(SceneRequest):
    command: str = Field(min_length=1, max_length=1000)


class DirectorRequest(SceneRequest):
    mode: Literal["relaxed", "cinematic", "challenging", "chaotic"] = "cinematic"


class ProposedAction(BaseModel):
    type: Literal["native", "prompt", "semantic"]
    action: str | None = Field(default=None, max_length=100)
    prompt: str | None = Field(default=None, max_length=1500)


class CommandResult(BaseModel):
    observation: str = Field(max_length=2000)
    explanation: str = Field(max_length=1500)
    action: ProposedAction | None = None
    source: Literal["vision-llm"] = "vision-llm"
    experimental: bool = True


class ObjectiveAssessment(BaseModel):
    progress: Literal["unknown", "not-started", "in-progress", "appears-complete"]
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(max_length=2000)


class DirectorResult(BaseModel):
    observation: str = Field(max_length=2000)
    objective: ObjectiveAssessment
    event: ProposedAction | None = None
    reason: str = Field(max_length=1500)
    source: Literal["vision-llm"] = "vision-llm"


def supported(body: SceneRequest, action: ProposedAction | None) -> ProposedAction | None:
    if action is None:
        return None
    control = body.capabilities.get("control", {})
    if not isinstance(control, dict):
        return None
    if action.type == "native":
        if action.action not in body.nativeActions:
            return None
        action.prompt = None
    else:
        if not action.prompt:
            return None
        if action.type == "semantic" and control.get("semanticActions") is not True:
            return None
        if action.type == "prompt" and not (
            control.get("promptDuringRollout") is True or control.get("promptSwitching") is True
        ):
            return None
        action.action = None
    return action


def context(body: SceneRequest) -> dict[str, Any]:
    return body.model_dump(exclude={"frame"})


@router.post("/command")
async def interpret_command(body: CommandRequest) -> CommandResult:
    result = await vision.complete(
        context(body),
        (
            "Interpret the user command using the current image and supplied action vocabulary. "
            "Prefer an exact supplied native action when appropriate, otherwise a supported "
            'semantic action, otherwise supported live prompting. Return {"observation":"visible '
            'context", "explanation":"why this is the best supported mechanism", '
            '"action":{"type":"native|prompt|semantic","action":"supplied native ID or '
            'null","prompt":"contextual instruction or null"}}. Return action:null when '
            "unsupported or ambiguous. Never invent native action IDs."
        ),
        [body.frame],
    )
    try:
        parsed = CommandResult.model_validate(
            {**result, "source": "vision-llm", "experimental": True}
        )
    except ValidationError as exc:
        raise HTTPException(502, "The vision model returned an invalid command proposal.") from exc
    proposed = parsed.action
    parsed.action = supported(body, proposed)
    if proposed is not None and parsed.action is None:
        parsed.explanation = (
            "The proposed action is not supported by this model. No action was applied."
        )
    return parsed


@router.post("/director")
async def direct_scene(body: DirectorRequest) -> DirectorResult:
    result = await vision.complete(
        context(body),
        (
            "Observe the current frame, recent events, elapsed time, objective and requested "
            "pacing. Propose at most ONE modest event; no repetitive rapid changes. Return "
            '{"observation":"visible scene", "objective":{"progress":"unknown|not-started|in-'
            'progress|appears-complete","confidence":0.0,"evidence":"visible evidence, and '
            'uncertainty"}, "event":{"type":"prompt|semantic|native","prompt":"supported event '
            'instruction or null","action":"supplied native action or null"}, "reason":"why now, '
            'or why no event"}. event may be null. Objective progress is uncertain visual '
            "interpretation, never authoritative scoring or a verified win. If objective is empty "
            "use unknown. Do not suggest native movement or combat for the player; use world "
            "events only."
        ),
        [body.frame],
    )
    try:
        parsed = DirectorResult.model_validate({**result, "source": "vision-llm"})
    except ValidationError as exc:
        raise HTTPException(502, "The vision model returned invalid director guidance.") from exc
    parsed.event = supported(body, parsed.event)
    if parsed.event and parsed.event.type == "native":
        parsed.event = None
    if not body.objective.strip():
        parsed.objective = ObjectiveAssessment(
            progress="unknown", confidence=0, evidence="No objective was set."
        )
    return parsed


class HighlightSample(BaseModel):
    atSeconds: float = Field(ge=0, le=86400)  # noqa: N815
    image: str = Field(max_length=2_800_000)


class HighlightsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: list[HighlightSample] = Field(min_length=2, max_length=8)
    durationSeconds: float = Field(gt=0, le=86400)  # noqa: N815
    prompt: str = Field(default="", max_length=8000)


class Highlight(BaseModel):
    startSeconds: float = Field(ge=0)  # noqa: N815
    endSeconds: float = Field(gt=0)  # noqa: N815
    title: str = Field(max_length=100)
    evidence: str = Field(max_length=1000)


class HighlightsResult(BaseModel):
    highlights: list[Highlight] = Field(max_length=8)
    source: Literal["vision-llm"] = "vision-llm"
    note: str = (
        "Suggestions based on sampled frames, not a full video analysis. Review before sharing."
    )


@router.post("/highlights")
async def highlights(body: HighlightsRequest) -> HighlightsResult:
    times = [sample.atSeconds for sample in body.samples]
    if times != sorted(times) or any(at > body.durationSeconds for at in times):
        raise HTTPException(
            422, "Sample timestamps must be ordered and within the replay duration."
        )
    result = await vision.complete(
        {"sampleSeconds": times, "durationSeconds": body.durationSeconds, "prompt": body.prompt},
        (
            "Find visually notable changes or moments across the ordered sampled images. Return "
            '{"highlights":[{"startSeconds":0,"endSeconds":5,"title":"short '
            'label","evidence":"what is visible"}]}. Use only times within the supplied duration, '
            "do not invent unseen action or audio; maximum four clips. Return an empty list if no "
            "distinct moment is evident."
        ),
        [sample.image for sample in body.samples],
    )
    try:
        parsed = HighlightsResult.model_validate({**result, "source": "vision-llm"})
    except ValidationError as exc:
        raise HTTPException(
            502, "The vision model returned invalid highlight suggestions."
        ) from exc
    parsed.highlights = [
        item
        for item in parsed.highlights
        if item.startSeconds < item.endSeconds <= body.durationSeconds
    ]
    return parsed

"""Outlining ground from a picture of the map.

When the operator clicks a spot and no map data knows the feature there, the console sends
one view of the map with the click marked and asks the model to trace the field, pond or lot
around it. The outline comes back in normalized image coordinates; the console projects it
onto the ground. Only available when the API is configured with a key: a rule-based fallback
would be a guess dressed as an answer, so there is none here.
"""

from __future__ import annotations

import base64
import binascii
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.schemas.agent import Outline, OutlinePoint, OutlineRequest
from app.services.planner import PlannerError

MIN_POINTS = 3
MAX_POINTS = 32


class ModelOutlinePoint(BaseModel):
    x: float = Field(description="0 = left edge, 1 = right edge")
    y: float = Field(description="0 = top edge, 1 = bottom edge")


class ModelOutline(BaseModel):
    points: list[ModelOutlinePoint] = Field(
        description="4 to 24 vertices in order around the feature, normalized image coordinates"
    )
    label: str = Field(description="Short name of the feature, e.g. 'orchard block', 'pond'")
    confidence: float = Field(description="0..1, how sure you are the outline follows real edges")
    note: str = Field(description="One sentence on what you outlined and any caveat")


ModelOutline.model_rebuild()

OUTLINE_PROMPT = """You outline land features on aerial and oblique 3D map imagery for a \
robot-fleet mission planner. You are given one image of the map view and the location of a \
point the operator clicked, drawn as a magenta ring. Trace the single contiguous feature that \
contains that point: a field, orchard block, paddock, pond or lake, lot, lawn, building \
footprint or similar working area. Follow its visible edges (fence lines, roads, tree lines, \
shoreline, crop boundaries) with 4 to 24 vertices in order, in normalized image coordinates \
(x from 0 at the left to 1 at the right, y from 0 at the top to 1 at the bottom). The view may \
be oblique; outline the feature as it appears in the image. If the point sits on a road, \
water edge or somewhere without a clear boundary, outline the nearest sensible working area \
around it and say so in the note. Never outline the whole image."""


def _media_type(data: bytes) -> Literal["image/jpeg", "image/png"]:
    return "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"


def _decode(image: str) -> tuple[str, Literal["image/jpeg", "image/png"]]:
    raw = image.split(",", 1)[1] if image.startswith("data:") else image
    try:
        data = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PlannerError(422, "The image is not valid base64.") from exc
    return raw, _media_type(data)


class Outliner:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        self._settings = settings
        self._client = client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=60.0
        )

    def outline(self, request: OutlineRequest) -> Outline:
        data, media_type = _decode(request.image)
        text = (
            f"The image is {request.width}x{request.height} pixels. The operator clicked at "
            f"x={request.point.x:.3f}, y={request.point.y:.3f} (normalized), marked with a "
            "magenta ring."
            + (f" Their goal: {request.hint.strip()}" if request.hint.strip() else "")
        )
        try:
            response = self._client.messages.parse(
                model=self._settings.anthropic_model,
                max_tokens=2000,
                system=OUTLINE_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            },
                            {"type": "text", "text": text},
                        ],
                    }
                ],
                output_format=ModelOutline,
            )
        except anthropic.AuthenticationError as exc:
            raise PlannerError(503, "The planning model rejected the configured API key.") from exc
        except anthropic.RateLimitError as exc:
            raise PlannerError(
                429, "The planning model is rate-limited; try again shortly."
            ) from exc
        except anthropic.APIStatusError as exc:
            raise PlannerError(
                502, f"The planning model returned an error ({exc.status_code})."
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise PlannerError(502, "The planning model could not be reached.") from exc
        if response.stop_reason == "refusal":
            raise PlannerError(422, "The planning model declined to outline this view.")
        parsed = response.parsed_output
        if parsed is None:
            raise PlannerError(502, "The planning model returned no outline.")
        points = [
            OutlinePoint(x=min(1.0, max(0.0, p.x)), y=min(1.0, max(0.0, p.y)))
            for p in parsed.points[:MAX_POINTS]
        ]
        if len(points) < MIN_POINTS:
            raise PlannerError(422, "The planning model could not find a feature there.")
        return Outline(
            points=points,
            label=parsed.label.strip()[:80] or "feature",
            confidence=min(1.0, max(0.0, parsed.confidence)),
            note=parsed.note.strip()[:300],
        )


def build_outliner(settings: Settings | None = None) -> Outliner | None:
    settings = settings or get_settings()
    return Outliner(settings) if settings.anthropic_api_key else None

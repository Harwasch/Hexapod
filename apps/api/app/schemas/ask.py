"""Wire models mirror the shared phone contract; ledger envelopes are separate."""

import uuid
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator
from pydantic.json_schema import SkipJsonSchema

from app.schemas.base import CamelModel


class WireModel(CamelModel):
    model_config = ConfigDict(allow_inf_nan=False, str_strip_whitespace=False, strict=True)

    @model_validator(mode="before")
    @classmethod
    def omitted_optionals(cls, value: Any) -> Any:
        # The shared schema allows omission, not JSON null, for optional references.
        if isinstance(value, dict):
            for name, field in cls.model_fields.items():
                if not field.is_required() and field.default is None:
                    for key in (name, field.alias):
                        if key in value and value[key] is None:
                            raise ValueError("optional references must be omitted rather than null")
        return value


class Vector3(WireModel):
    x: float
    y: float
    z: float


class Extent(WireModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    z: float = Field(ge=0)


class GeoCoordinate(WireModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt: float


class Anchor(WireModel):
    world: Vector3
    geo: GeoCoordinate | SkipJsonSchema[None] = None


class MaskReference(WireModel):
    keyframe: str
    ref: str


class EntitySource(WireModel):
    module: str
    model: str


class Entity(WireModel):
    label: str
    anchor: Anchor
    extent: Extent
    masks: list[MaskReference]
    confidence: float = Field(ge=0, le=1)
    attrs: dict[str, JsonValue]
    source: EntitySource


class KeyframeRef(WireModel):
    index: int = Field(ge=0)
    image: str = Field(min_length=1)
    depth: str | SkipJsonSchema[None] = Field(default=None, min_length=1)
    pose: str = Field(pattern=r"^frames\.jsonl#[0-9]+$")


class AskRequest(WireModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    scan_id: str | SkipJsonSchema[None] = Field(default=None, min_length=1)
    scope: Literal["scan", "property"] | SkipJsonSchema[None] = None
    site_id: str | SkipJsonSchema[None] = Field(default=None, min_length=1)
    location: GeoCoordinate | SkipJsonSchema[None] = None
    keyframes: list[KeyframeRef]
    device_entities: list[Entity]

    @model_validator(mode="after")
    def property_site(self) -> "AskRequest":
        if self.site_id is not None:
            uuid.UUID(self.site_id)
        if self.scope == "property" and self.site_id is None:
            raise ValueError("property scope requires siteId")
        return self


class EntityEvent(WireModel):
    type: Literal["entity"] = "entity"
    entity: Entity


class SpeechEvent(WireModel):
    type: Literal["speech"] = "speech"
    text: str


class StatusEvent(WireModel):
    type: Literal["status"] = "status"
    text: str


class DoneEvent(WireModel):
    type: Literal["done"] = "done"


AskEvent = Annotated[
    EntityEvent | SpeechEvent | StatusEvent | DoneEvent, Field(discriminator="type")
]

"""Mission-planning agent schemas: the context the operator's console sends and the draft
that comes back. Every draft names its source so the UI never presents a rule-based draft as
a model's work."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import (
    BeforeValidator,
    Field,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
)
from shapely import make_valid
from shapely.errors import ShapelyError
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union

from app.schemas.base import CamelModel
from app.schemas.geojson import MapGeometry
from app.schemas.scout import ScoutCapture

Cadence = Literal["once", "daily", "weekly", "monthly", "seasonal"]
PlannerSource = Literal["claude", "rules"]
ZONE_NAME_MAX = 120


def _clip_zone_name(value: object) -> object:
    """A zone name longer than ZONE_NAME_MAX is clipped, not refused. Zone names come from a
    Land's names (up to 200 characters) and from names the console derives from them
    ("<feature>, workable"), and one long name must not refuse a whole draft or approval."""
    if isinstance(value, str) and len(text := value.strip()) > ZONE_NAME_MAX:
        return text[: ZONE_NAME_MAX - 1].rstrip() + "…"
    return value


#: A zone's or a plan area's name, as the planner reads it and an approved plan keeps it.
ZoneName = Annotated[
    str, BeforeValidator(_clip_zone_name), Field(min_length=1, max_length=ZONE_NAME_MAX)
]


class PlannerMachine(CamelModel):
    id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=120)
    model: str = Field(default="", max_length=120)
    status: str = Field(default="idle", max_length=40)
    task: str = Field(default="", max_length=200)
    battery_pct: int = Field(default=100, ge=0, le=100)


class PlannerZone(CamelModel):
    id: str = Field(min_length=1, max_length=40)
    name: ZoneName
    acres: float = Field(default=0, ge=0)
    task: str = Field(default="", max_length=200)
    progress_pct: int = Field(default=0, ge=0, le=100)
    treated: bool = False
    note: str = Field(default="", max_length=500)
    # The zone's outline (or, for a mapped line such as a fence, the line itself), so the
    # planner sizes work from the ground rather than from stated acres. Never sent to a model.
    geometry: MapGeometry | None = None
    # A scout area's capture parameters: the planner flies it at these, not its detail table.
    capture: ScoutCapture | None = None

    @field_validator("geometry", mode="wrap")
    @classmethod
    def _tolerate_invalid_outline(
        cls, value: object, handler: ValidatorFunctionWrapHandler
    ) -> object:
        """An outline is sizing context, not a stored shape: one the console let through
        invalid (a corner dragged across an edge, OSM outers that touch) is repaired, or
        dropped so the zone falls back to its acres, rather than failing the whole draft."""
        try:
            return handler(value)
        except ValidationError:
            repaired = _repaired_outline(value)
            if repaired is None:
                return None
            try:
                return handler(repaired)
            except ValidationError:
                return None


def _repaired_outline(value: object) -> dict[str, object] | None:
    """The polygonal part of `shapely.make_valid` for an invalid (Multi)Polygon, as GeoJSON;
    None when nothing with area is left or the value is not a polygon at all."""
    if not isinstance(value, dict) or value.get("type") not in {"Polygon", "MultiPolygon"}:
        return None
    try:
        fixed = make_valid(shape(value))
    except (ShapelyError, ValueError, TypeError, AttributeError, IndexError, KeyError):
        return None
    if not isinstance(fixed, Polygon | MultiPolygon):
        # A collection (a bow-tie with a stray edge): keep the parts with area.
        parts = [g for g in getattr(fixed, "geoms", []) if isinstance(g, Polygon | MultiPolygon)]
        if not parts:
            return None
        fixed = unary_union(parts)
    if not isinstance(fixed, Polygon | MultiPolygon) or fixed.is_empty or fixed.area == 0:
        return None
    return dict(mapping(fixed))


class BusyWindow(CamelModel):
    """Days a machine is already booked by another plan (inclusive start, exclusive end)."""

    machine_id: str = Field(min_length=1, max_length=40)
    start_date: date
    end_date: date


class PlannerExistingPlan(CamelModel):
    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    zone_ids: list[str] = Field(default_factory=list)
    status: str = Field(default="scheduled", max_length=20)
    busy: list[BusyWindow] = Field(default_factory=list, max_length=500)


class PlannerRate(CamelModel):
    """A treatment rate learned from the work log, per task family."""

    task: str = Field(min_length=1, max_length=80)
    acres_per_machine_hour: float = Field(gt=0)
    samples: int = Field(ge=1)
    machine_ids: list[str] = Field(default_factory=list)


ClarificationKind = Literal["choice", "range", "area"]
AnswerValue = str | float | bool


class ClarificationOption(CamelModel):
    value: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=80)


class Clarification(CamelModel):
    """A question the planner asks with a structured answer: a choice (chips), a range
    (slider) or an area choice (how the ground is picked). Never free text."""

    id: str = Field(min_length=1, max_length=40)
    question: str = Field(min_length=1, max_length=200)
    kind: ClarificationKind
    options: list[ClarificationOption] = Field(default_factory=list, max_length=8)
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = Field(default=None, max_length=20)
    default: AnswerValue | None = None
    # Why the answer matters, shown under the question.
    why: str = Field(default="", max_length=200)


class PlanStep(CamelModel):
    """Every field explicit (no defaults) so the contract marks them required on read."""

    title: str = Field(min_length=1, max_length=200)
    detail: str = Field(max_length=1000)
    machine_ids: list[str]
    zone_id: str | None
    when: str = Field(max_length=120)
    # Schedule: offset from the plan start in days and duration in days (>= 1 for a real step).
    start_day: int = Field(ge=0)
    days: int = Field(ge=0)


class PlanEstimates(CamelModel):
    acres: float = Field(ge=0)
    machine_hours: float = Field(ge=0)
    calendar_days: int = Field(ge=0)
    # Metres of path one run covers, when the work was sized from outlines.
    path_m: float | None = Field(default=None, ge=0)


MachineClassId = Literal["mower", "seeder", "survey", "drone"]


class ZoneCoverage(CamelModel):
    """How one zone's work was sized from its outline: the path, the assumed machine class
    (`app/services/machine_classes.json`) and the hours that follow. `basis` is the sentence
    the console shows beside the number."""

    zone_id: str = Field(min_length=1, max_length=40)
    kind: Literal["area", "line"]
    machine_class: MachineClassId
    area_m2: float | None
    length_m: float = Field(ge=0)
    transit_m: float = Field(ge=0)
    passes: int = Field(ge=0)
    spans: int = Field(ge=0)
    spacing_m: float | None
    # Flight height above ground the drone class flew at (set by the detail asked); None for
    # ground classes, whose spacing is their swath.
    altitude_m: float | None = Field(default=None, ge=0)
    speed_mps: float = Field(gt=0)
    turn_s: float = Field(ge=0)
    hours: float = Field(ge=0)
    capped: bool
    basis: str = Field(max_length=600)


class PlanDraftBody(CamelModel):
    """The plan itself, as drafted: what the operator reviews, edits and approves."""

    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=2000)
    zone_ids: list[str] = Field(default_factory=list)
    machine_ids: list[str] = Field(default_factory=list)
    cadence: Cadence = "once"
    start_date: date
    end_date: date | None = None
    estimates: PlanEstimates
    steps: list[PlanStep] = Field(default_factory=list)
    # What the estimate rests on (rates, hours per day, weather windows); shown with the numbers.
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)
    coverage: list[ZoneCoverage] = Field(default_factory=list, max_length=500)


class PlanDraftRequest(CamelModel):
    goal: str = Field(min_length=3, max_length=2000)
    project_name: str = Field(min_length=1, max_length=200)
    site_name: str | None = Field(default=None, max_length=200)
    machines: list[PlannerMachine] = Field(default_factory=list, max_length=200)
    zones: list[PlannerZone] = Field(default_factory=list, max_length=500)
    existing_plans: list[PlannerExistingPlan] = Field(default_factory=list, max_length=200)
    # Rates learned from the work log; the planner prefers them over its defaults.
    rates: list[PlannerRate] = Field(default_factory=list, max_length=100)
    # Operator pre-selections from the console (a selected zone, machines ticked in Fleet).
    preferred_zone_ids: list[str] = Field(default_factory=list)
    preferred_machine_ids: list[str] = Field(default_factory=list)
    # A follow-up instruction on a previous draft ("use three machines", "skip Z-21").
    refinement: str | None = Field(default=None, max_length=2000)
    # Answers to the planner's clarifications, by clarification id.
    answers: dict[str, AnswerValue] = Field(default_factory=dict)
    previous_draft: PlanDraftBody | None = None
    today: date | None = None


class PlanDraft(CamelModel):
    """Read model: every field is explicit (no defaults) so the contract marks it required."""

    title: str
    objective: str
    zone_ids: list[str]
    machine_ids: list[str]
    cadence: Cadence
    start_date: date
    end_date: date | None
    estimates: PlanEstimates
    steps: list[PlanStep]
    assumptions: list[str]
    risks: list[str]
    questions: list[str]
    clarifications: list[Clarification]
    source: PlannerSource
    model: str | None
    # Shown next to the draft: who drafted it and what it assumed.
    note: str
    # Per zone, how the hours were sized from its outline (empty when no zone had one). Not
    # persisted with an approved plan; its sentence is, under assumptions.
    coverage: list[ZoneCoverage] = Field(default_factory=list)


class PlannerStatus(CamelModel):
    configured: bool
    provider: PlannerSource
    model: str | None


class OutlinePoint(CamelModel):
    """A polygon vertex in normalized image coordinates: x right, y down, both 0..1."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class OutlineRequest(CamelModel):
    """One view of the map with a marked point; the model outlines the feature under it."""

    image: str = Field(min_length=64, max_length=4_000_000, description="JPEG or PNG, base64")
    width: int = Field(ge=64, le=4096)
    height: int = Field(ge=64, le=4096)
    point: OutlinePoint
    hint: str = Field(default="", max_length=300, description="The operator's goal, for context")


class Outline(CamelModel):
    points: list[OutlinePoint]
    label: str = Field(max_length=80)
    confidence: float = Field(ge=0, le=1)
    note: str = Field(default="", max_length=300)
    source: Literal["claude"] = "claude"

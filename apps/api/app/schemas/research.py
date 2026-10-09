from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, HttpUrl

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint, MapGeometry


class ResearchBudget(CamelModel):
    max_web_searches: int = Field(default=6, ge=0, le=30)
    max_steps: int = Field(default=16, ge=1, le=100)
    max_seconds: int = Field(default=180, ge=10, le=1800)
    max_output_tokens: int = Field(default=12_000, ge=500, le=100_000)


class InvestigationCreate(CamelModel):
    title: str = Field(min_length=1, max_length=300)
    question: str = Field(
        default="What is useful and interesting about this land?", min_length=1, max_length=10_000
    )
    boundary_revision: int = Field(ge=1)


class InvestigationRead(InvestigationCreate):
    id: uuid.UUID
    land_id: uuid.UUID
    created_at: datetime
    stale: bool


class RunCreate(CamelModel):
    request_key: uuid.UUID
    kind: Literal["overview", "investigation"] = "investigation"
    question: str = Field(min_length=1, max_length=10_000)
    budget: ResearchBudget = Field(default_factory=ResearchBudget)


class RunRead(CamelModel):
    id: uuid.UUID
    investigation_id: uuid.UUID
    question: str
    kind: Literal["overview", "investigation"]
    status: Literal["queued", "running", "succeeded", "partial", "failed", "cancelled"]
    budget: ResearchBudget
    attempt: int
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class EventRead(CamelModel):
    sequence: int
    kind: str
    payload: dict[str, object]
    created_at: datetime


class EvidenceContent(CamelModel):
    provider: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    license: str = Field(min_length=1, max_length=1000)
    attribution: str = Field(min_length=1, max_length=2000)
    record_id: str | None = Field(default=None, max_length=500)
    retrieved_at: datetime
    observed_at: datetime | None = None
    published_at: datetime | None = None
    excerpt: str = Field(max_length=30_000)
    spatial_relevance: Literal["intersects", "within", "nearby", "regional", "unresolved"]
    relevance_note: str = Field(min_length=1, max_length=2000)
    snapshot_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class EvidenceRead(EvidenceContent):
    id: uuid.UUID
    run_id: uuid.UUID


class FindingContent(CamelModel):
    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=10_000)
    category: Literal[
        "physical",
        "ecology",
        "history",
        "rights",
        "hazards",
        "infrastructure",
        "energy",
        "coverage",
    ]
    evidence_ids: list[uuid.UUID] = Field(min_length=1, max_length=100)
    confidence: Literal["measured", "supported", "inferred", "uncertain"]
    uncertainty: str = Field(min_length=1, max_length=3000)
    boundary: Footprint | None = None
    suggested_questions: list[str] = Field(default_factory=list, max_length=8)


class FindingRead(FindingContent):
    id: uuid.UUID
    run_id: uuid.UUID
    disposition: Literal["visible", "pinned", "dismissed"]


class FindingDisposition(CamelModel):
    disposition: Literal["visible", "pinned", "dismissed"]


class TableColumn(CamelModel):
    key: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=200)
    unit: str | None = Field(default=None, max_length=100)


class TableOutput(CamelModel):
    kind: Literal["table"]
    columns: list[TableColumn] = Field(min_length=1, max_length=50)
    rows: list[dict[str, str | float | bool | None]] = Field(max_length=5000)


class ChartSeries(CamelModel):
    label: str = Field(max_length=200)
    values: list[float | None] = Field(max_length=5000)


class ChartOutput(CamelModel):
    kind: Literal["chart"]
    chart_type: Literal["line", "bar", "scatter"]
    x_label: str
    y_label: str
    unit: str
    labels: list[str] = Field(max_length=5000)
    series: list[ChartSeries] = Field(min_length=1, max_length=20)


class MapFeature(CamelModel):
    label: str = Field(max_length=300)
    geometry: MapGeometry
    value: float | None = None


class MapOutput(CamelModel):
    kind: Literal["map"]
    features: list[MapFeature] = Field(max_length=2000)
    unit: str | None = None
    legend: str = Field(max_length=2000)


class DocumentOutput(CamelModel):
    kind: Literal["document"]
    markdown: str = Field(max_length=100_000)


class TimelineEntry(CamelModel):
    date: str = Field(max_length=100)
    title: str = Field(max_length=300)
    description: str = Field(max_length=5000)
    evidence_ids: list[uuid.UUID] = Field(min_length=1, max_length=100)


class TimelineOutput(CamelModel):
    kind: Literal["timeline"]
    entries: list[TimelineEntry] = Field(max_length=1000)


Output = Annotated[
    TableOutput | ChartOutput | MapOutput | DocumentOutput | TimelineOutput,
    Field(discriminator="kind"),
]


class ArtifactContent(CamelModel):
    title: str = Field(min_length=1, max_length=300)
    method: str = Field(min_length=1, max_length=5000)
    evidence_ids: list[uuid.UUID] = Field(min_length=1, max_length=100)
    output: Output


class ResearchArtifactRead(ArtifactContent):
    id: uuid.UUID
    run_id: uuid.UUID


class MessageRead(CamelModel):
    id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ResearchPage(CamelModel):
    offset: int
    limit: int
    totals: dict[str, int]


class InvestigationDetail(CamelModel):
    page: ResearchPage
    investigation: InvestigationRead
    messages: list[MessageRead]
    runs: list[RunRead]
    findings: list[FindingRead]
    evidence: list[EvidenceRead]
    artifacts: list[ResearchArtifactRead]


class ResearchStatus(CamelModel):
    model_configured: bool
    model: str | None
    worker_required: bool = True


class SourceRead(CamelModel):
    id: str
    name: str
    domain: str
    coverage: str
    resolution: str
    license: str
    attribution: str
    documentation_url: HttpUrl


class OverviewRequest(CamelModel):
    boundary_revision: int = Field(ge=1)


class OverviewRead(CamelModel):
    investigation: InvestigationRead
    run: RunRead

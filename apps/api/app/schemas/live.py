"""What the live viewer polls: a run's cameras as they are solved and its splat as it trains.

The cameras and points arrive quantised and base64-packed exactly as the pipeline logged
them (`tools/pipeline/live.py`, `encode_model`), so this is a pass-through rather than a
second encoding: 12 bytes per camera (int16 centre, int8 view direction, int8 up) and 9
bytes per point (int16 position, uint8 colour), positions as `origin + q / 32767 * scale`.
Everything is in the pose solve's own frame -- COLMAP's, not yet east/north/up -- with
`up` the mean camera-up estimate in that frame.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from app.models.enums import RunStatus
from app.schemas.base import CamelModel


class LiveProgress(CamelModel):
    """The running stage's tool progress (`metrics.progress`), when it prints one."""

    done: int
    total: int
    elapsed_s: int | None
    remaining_s: int | None


class LiveStage(CamelModel):
    """The step the run is on: running now, or the last one to have started."""

    stage_id: str
    impl: str
    status: RunStatus
    ordinal: int
    started_at: datetime | None
    progress: LiveProgress | None


class LiveCameras(CamelModel):
    """Registered cameras and a sample of the sparse points, from the newest snapshot."""

    stage_id: str
    seq: int
    final: bool
    registered: int
    frames: int | None
    origin: list[float]
    scale: float
    up: list[float] | None
    fov_y: float | None = Field(default=None, gt=0, lt=180, alias="fovY")
    aspect: float | None
    camera_count: int
    #: base64: 12 bytes per camera, little-endian int16 x, y, z; int8 forward; int8 up.
    cameras: str
    point_count: int
    points_total: int
    #: base64: 9 bytes per point, little-endian int16 x, y, z; uint8 r, g, b.
    points: str


class LiveSplat(CamelModel):
    """The newest intermediate splat, and where to fetch it."""

    stage_id: str
    step: int
    total: int | None
    count: int
    of: int | None
    bytes: int | None
    up: list[float] | None
    #: The snapshot's file name: changes with every new snapshot, unlike `url`, whose
    #: signature changes on every request.
    name: str
    #: A short-lived signed GET for the SPZ, or null until the checkpoint syncer has
    #: uploaded it (it syncs every minute) or when there is no bucket.
    url: str | None


class TrainModel(CamelModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="ignore")


class LiveTrainRule(TrainModel):
    every: int = Field(gt=0)
    window: int = Field(gt=0)
    min_gain_db: float = Field(ge=0)


class LiveTrainPlan(TrainModel):
    antialiased: bool | None = None
    cap: int | None = Field(default=None, ge=0)
    cap_max: int | None = Field(default=None, ge=0)
    mode: str | None = None
    clamp: str | None = None
    steps_max: int | None = Field(default=None, gt=0)
    schedule_factor: float | None = Field(default=None, gt=0)
    refine_start: int | None = Field(default=None, ge=0)
    refine_stop: int | None = Field(default=None, ge=0)
    refine_every: int | None = Field(default=None, ge=0)
    sh_degree: int | None = Field(default=None, ge=0)
    sh_interval: int | None = Field(default=None, gt=0)
    rule: LiveTrainRule | None = None
    blocks: int | None = Field(default=None, ge=1)
    gpu: str | None = None
    card_gb: float | None = Field(default=None, gt=0)
    usd_per_hour: float | None = Field(default=None, ge=0)
    metres_per_unit: float | None = Field(default=None, gt=0)
    frames: int | None = Field(default=None, ge=0)
    sparse: int | None = Field(default=None, ge=0)


class LiveTrainSeries(TrainModel):
    step: list[int] = Field(default_factory=list, max_length=300)
    t: list[float | None] = Field(default_factory=list)
    gs: list[int | None] = Field(default_factory=list)
    lr: list[float | None] = Field(default_factory=list)
    sh: list[int | None] = Field(default_factory=list)
    its: list[float | None] = Field(default_factory=list)
    mem: list[float | None] = Field(default_factory=list)
    loss: dict[str, list[float | None]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def aligned(self) -> LiveTrainSeries:
        columns: list[Sequence[float | int | None]] = [
            self.t,
            self.gs,
            self.lr,
            self.sh,
            self.its,
            self.mem,
            *self.loss.values(),
        ]
        if any(len(column) != len(self.step) for column in columns):
            raise ValueError("training columns must align")
        if any(step < 0 for step in self.step) or self.step != sorted(set(self.step)):
            raise ValueError("training steps must increase")
        return self


class LiveTrainEval(TrainModel):
    step: int = Field(ge=0)
    psnr: float | None = None
    gain: float | None = None
    decision: str | None = None


class LiveTrainStop(TrainModel):
    step: int = Field(ge=0)
    max_steps: int | None = None
    gain: float | None = None


class LiveTrainEnded(TrainModel):
    reason: Literal["preempted", "oom", "failed"]
    step: int = Field(ge=0)


class LiveTrainEvent(TrainModel):
    step: int = Field(ge=0)
    kind: str
    text: str
    steps_saved: int | None = None
    usd_saved: float | None = None


class LiveQualitySeries(TrainModel):
    step: list[int] = Field(default_factory=list, max_length=300)
    far_field_share: list[float | None] = Field(default_factory=list)
    elongation_share: list[float | None] = Field(default_factory=list)
    peak_allocated_gb: list[float | None] = Field(default_factory=list)
    loss: dict[str, list[float | None]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def aligned(self) -> LiveQualitySeries:
        columns = [
            self.far_field_share,
            self.elongation_share,
            self.peak_allocated_gb,
            *self.loss.values(),
        ]
        if (
            any(len(column) != len(self.step) for column in columns)
            or self.step != sorted(set(self.step))
            or any(s < 0 for s in self.step)
        ):
            raise ValueError("quality columns must align and steps must increase")
        return self


class LiveTrainQuality(TrainModel):
    step: int | None = Field(default=None, ge=0)
    far_field_share: float | None = Field(default=None, ge=0, le=1)
    elongation_share: float | None = Field(default=None, ge=0, le=1)
    series: LiveQualitySeries | None = None
    dense_init_points: int | None = Field(default=None, ge=0)
    dense_init_added_points: int | None = Field(default=None, ge=0)
    dense_init_seconds: float | None = Field(default=None, ge=0)


class LiveTrainSeeds(TrainModel):
    key: str
    count: int = Field(ge=0, le=200000)
    of: int = Field(ge=0)
    sources: dict[str, int] = Field(default_factory=dict)
    origin: list[float] = Field(min_length=3, max_length=3)
    scale: float = Field(gt=0)
    bytes: int = Field(ge=0)
    url: str | None = None

    @model_validator(mode="after")
    def packed(self) -> LiveTrainSeeds:
        if (
            self.bytes != 10 * self.count
            or self.of < self.count
            or any(v < 0 for v in self.sources.values())
        ):
            raise ValueError("invalid seed counts")
        return self


class LiveTrainFinal(TrainModel):
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    cc_psnr: float | None = None
    cc_ssim: float | None = None
    cc_lpips: float | None = None


class LiveTrainSnapshot(TrainModel):
    step: int = Field(ge=0)
    key: str
    count: int = Field(ge=0)
    of: int | None = None
    bytes: int | None = None
    url: str | None = None
    far_field_share: float | None = Field(default=None, ge=0, le=1)
    elongation_share: float | None = Field(default=None, ge=0, le=1)


class LiveTrainTiles(TrainModel):
    """A full LOD tileset. Omitted frame means the producer's solve-local coordinates."""

    url: str
    step: int = Field(ge=0)
    final: bool
    frame: Literal["solve", "enu"] = "solve"


class LiveTrain(TrainModel):
    tiles: LiveTrainTiles | None = None
    series: LiveTrainSeries = Field(default_factory=LiveTrainSeries)
    evals: list[LiveTrainEval] = Field(default_factory=list, max_length=256)
    plan: LiveTrainPlan | None = None
    events: list[LiveTrainEvent] = Field(default_factory=list)
    snapshots: list[LiveTrainSnapshot] = Field(default_factory=list, max_length=64)
    stop: LiveTrainStop | None = None
    ended: LiveTrainEnded | None = None
    replaced_by: str | None = None
    block: str | None = None
    run: str | None = None
    plan_id: str | None = None
    quality: LiveTrainQuality | None = None
    seeds: LiveTrainSeeds | None = None
    final: LiveTrainFinal | None = None


class LiveState(CamelModel):
    job_id: uuid.UUID
    capture_id: uuid.UUID
    capture_name: str
    recipe: str
    status: RunStatus
    error: str | None
    #: The site the run registered, once it has: where the final scan is.
    site_id: uuid.UUID | None
    stage: LiveStage | None
    steps_done: int
    steps_started: int
    cameras: LiveCameras | None
    splat: LiveSplat | None
    train: LiveTrain | None = None

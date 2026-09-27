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
from datetime import datetime

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

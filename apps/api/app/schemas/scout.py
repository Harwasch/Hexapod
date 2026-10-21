"""Capture parameters a scout area carries: how a mapping drone would fly it.

Set on areas the Fog of Knowing proposes ("Lift the fog"), and on the weak areas of an
epoch's confidence map (`re-scan-weak`, M4: flown nadir, in crossing obliques or round in
an orbit, with the camera pitched as given); stored with the plan (PlanArea,
in the plans' areas JSONB) so an approved scout plan can still be exported as a route, and
sent with the draft request (PlannerZone) so the planner sizes the flying from these numbers
rather than from its own detail table. The defaults live in one place, the drone class of
`app/services/machine_classes.json`; these are the values chosen for this area.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.base import CamelModel

#: The confidence map's reasons a re-scan may name (ADR 0016).
RescanReason = Literal["views", "spread", "gsd", "accuracy", "unseen", "alignment"]


class ScoutRescanRef(CamelModel):
    """Which epoch's weak cells a re-scan area is for. Informational only: plans are a
    workspace's and sites are public, so this is a reference, never a foreign key (the
    re-scan of a cell is any later epoch of the site that covers it, ADR 0016)."""

    site_id: uuid.UUID
    epoch_id: uuid.UUID
    policy: Literal["weak-v1"] = "weak-v1"
    #: Weak cells the area was drawn round, at the map's finest level.
    cells: int = Field(ge=0)
    reasons: list[RescanReason] = Field(default_factory=list, max_length=6)


class ScoutCapture(CamelModel):
    purpose: Literal["lift-knowledge-fog", "re-scan-weak"] = "lift-knowledge-fog"
    #: How it is flown: parallel nadir lines, two crossing passes with the camera tilted,
    #: or two rings round it (`app/services/planner._size_zone`).
    pattern: Literal["nadir", "oblique", "orbit"] = "nadir"
    #: The camera's pitch, degrees below the horizon: -90 is straight down.
    gimbal_pitch_deg: float = Field(default=-90, ge=-90, le=-30, allow_inf_nan=False)
    rescan_of: ScoutRescanRef | None = None
    #: The camera the exported route (DJI WPML KMZ) targets.
    camera: Literal["dji-m3e-wide"] = "dji-m3e-wide"
    #: Above the take-off point, metres (the export's height mode), within Part 107's 120 m.
    altitude_m: float = Field(ge=20, le=120, allow_inf_nan=False)
    forward_overlap: float = Field(ge=0.5, le=0.9, allow_inf_nan=False)
    side_overlap: float = Field(ge=0.5, le=0.9, allow_inf_nan=False)
    speed_ms: float = Field(ge=1, le=15, allow_inf_nan=False)
    heading_deg: float | None = Field(default=None, ge=0, lt=360, allow_inf_nan=False)

    @model_validator(mode="after")
    def _a_rescan_says_what_of(self) -> ScoutCapture:
        if self.rescan_of is not None and self.purpose != "re-scan-weak":
            raise ValueError("rescanOf belongs to a re-scan-weak capture")
        if self.purpose == "lift-knowledge-fog" and self.pattern != "nadir":
            raise ValueError("lifting the fog is flown nadir")
        return self

"""Capture parameters a scout area carries: how a mapping drone would fly it to lift the fog.

Set on areas the Fog of Knowing proposes ("Lift the fog"); stored with the plan (PlanArea,
in the plans' areas JSONB) so an approved scout plan can still be exported as a route, and
sent with the draft request (PlannerZone) so the planner sizes the flying from these numbers
rather than from its own detail table. The defaults live in one place, the drone class of
`app/services/machine_classes.json`; these are the values chosen for this area.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel


class ScoutCapture(CamelModel):
    purpose: Literal["lift-knowledge-fog"] = "lift-knowledge-fog"
    #: The camera the exported route (DJI WPML KMZ) targets.
    camera: Literal["dji-m3e-wide"] = "dji-m3e-wide"
    #: Above the take-off point, metres (the export's height mode), within Part 107's 120 m.
    altitude_m: float = Field(ge=20, le=120, allow_inf_nan=False)
    forward_overlap: float = Field(ge=0.5, le=0.9, allow_inf_nan=False)
    side_overlap: float = Field(ge=0.5, le=0.9, allow_inf_nan=False)
    speed_ms: float = Field(ge=1, le=15, allow_inf_nan=False)
    heading_deg: float | None = Field(default=None, ge=0, lt=360, allow_inf_nan=False)

"""Assumed machine classes: the one table both the planner and the console read.

`machine_classes.json` beside this module holds every speed, swath and camera number the
mission planner sizes work with; the web app imports the same file (`missions/machineClasses.ts`)
to label what it shows. None of it is measured. Each class's label says "(assumed)", and every
estimate built on it names the class, so a number is never read as a fleet's real performance.
A fleet backend that reports its own machines replaces these per project later.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel

MachineClassId = Literal["mower", "seeder", "survey", "drone"]

SOURCE = Path(__file__).with_name("machine_classes.json")


class GroundClass(CamelModel):
    """A machine that works the ground in passes one swath apart. The last two numbers are
    the limits the console's dress rehearsal checks a plan's lanes against
    (`missions/rehearsal.ts`): the steepest slope along or across a lane, and how close a lane
    may come to a mapped creek."""

    label: str = Field(min_length=1)
    swath_m: float = Field(gt=0)
    speed_mps: float = Field(gt=0)
    turn_s: float = Field(ge=0)
    max_slope_pct: float = Field(gt=0)
    creek_buffer_m: float = Field(ge=0)


class DroneClass(CamelModel):
    """A mapping drone: flight lines one image footprint apart, less the side overlap."""

    label: str = Field(min_length=1)
    altitude_agl_m: float = Field(gt=0)
    speed_mps: float = Field(gt=0)
    sidelap_pct: float = Field(ge=0, lt=100)
    frontlap_pct: float = Field(ge=0, lt=100)
    camera_hfov_deg: float = Field(gt=0, lt=180)
    image_aspect: float = Field(gt=0)
    max_wind_mps: float = Field(gt=0)

    def footprint_width_m(self, altitude_m: float | None = None) -> float:
        """Ground width one image covers across the flight line, at `altitude_m` above ground."""
        height = self.altitude_agl_m if altitude_m is None else altitude_m
        return 2 * height * math.tan(math.radians(self.camera_hfov_deg) / 2)

    def line_spacing_m(self, altitude_m: float | None = None) -> float:
        """Distance between neighbouring flight lines: the footprint less the side overlap."""
        return self.footprint_width_m(altitude_m) * (1 - self.sidelap_pct / 100)


class MachineClasses(CamelModel):
    mower: GroundClass
    seeder: GroundClass
    survey: GroundClass
    drone: DroneClass

    def ground(self, identifier: MachineClassId) -> GroundClass:
        found = getattr(self, identifier)
        if not isinstance(found, GroundClass):
            raise KeyError(f"{identifier} is not a ground machine class")
        return found


def load(path: Path = SOURCE) -> MachineClasses:
    return MachineClasses.model_validate(json.loads(path.read_text(encoding="utf-8")))


MACHINE_CLASSES = load()

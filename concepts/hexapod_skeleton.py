"""Shared build123d construction for the hexapod skeleton concepts.

Not a concept itself (defines no PART).  Both concept modules call
`build(stance)` and differ only in the Stance they pass in, so the two renders
are the same robot with two leg proportions — which is exactly the choice on
the table.

`build()` returns (robot, figure): the robot alone, and a 6 ft reference
figure standing beside it on the ground plane.  The concept modules combine
them into PART but report the robot-only envelope in NOTES, because the
vision renderer measures whatever PART contains.
"""
from __future__ import annotations

import math
import os
import sys

from build123d import (Axis, Box, Compound, Cylinder, Pos, Rot, Sphere, fillet)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from hexapod_model import ACT, BODY, HUMAN_HEIGHT, Stance, ik  # noqa: E402

COXA_SECTION = (56.0, 34.0)    # width, thickness of the coxa arm (carries the overturning moment)
FEMUR_SECTION = (40.0, 26.0)
TIBIA_SECTION = (30.0, 20.0)
FOOT_D = 45.0
PLATE_T = 6.0
BATTERY = (140.0, 200.0, 110.0)
COMPUTE = (120.0, 140.0, 60.0)
HIP_DROP = 18.0                # yaw output / coxa centreline below the floor plate


def _link(length: float, section: tuple[float, float]):
    """A link along +x from the origin, rounded ends."""
    w, t = section
    body = Pos(length / 2, 0, 0) * Box(length, w, t)
    return fillet(body.edges().filter_by(Axis.Y), radius=min(w, t) * 0.45)


def _place(shape, at, yaw_deg: float, pitch_deg: float):
    """Pitch about local y (positive tips +x up), then yaw about z, then move."""
    return Pos(*at) * Rot(0, 0, yaw_deg) * Rot(0, -pitch_deg, 0) * shape


def leg(hip, side: int, stance: Stance, yaw_deg: float):
    """One leg.  hip = (x, y, z) of the yaw axis at the coxa centreline."""
    lg = stance.leg
    hx, hy, hz = hip
    a1, k = ik(lg, stance.foot_reach, -stance.hip_height)
    a2 = a1 + k - math.pi                       # tibia direction from the knee
    if stance.leg_plane == "radial":
        # leg plane direction from +y (outboard) rotated `yaw_deg` toward +x
        d = (math.sin(math.radians(yaw_deg)), side * math.cos(math.radians(yaw_deg)))
    else:
        d = (1.0 if yaw_deg >= 0 else -1.0, 0.0)
    phi = math.degrees(math.atan2(d[1], d[0]))
    femur_axis = (hx + lg.coxa * d[0], hy + lg.coxa * d[1], hz)
    coxa = _place(_link(lg.coxa, COXA_SECTION), (hx, hy, hz), phi, 0)
    femur = _place(_link(lg.femur, FEMUR_SECTION), femur_axis, phi, math.degrees(a1))
    knee = (femur_axis[0] + lg.femur * math.cos(a1) * d[0],
            femur_axis[1] + lg.femur * math.cos(a1) * d[1],
            femur_axis[2] + lg.femur * math.sin(a1))
    tibia = _place(_link(lg.tibia, TIBIA_SECTION), knee, phi, math.degrees(a2))
    foot_c = (knee[0] + lg.tibia * math.cos(a2) * d[0],
              knee[1] + lg.tibia * math.cos(a2) * d[1],
              knee[2] + lg.tibia * math.sin(a2))
    foot = Pos(*foot_c) * Sphere(FOOT_D / 2)
    # joint hubs so the axes read in the render
    hub = Cylinder(FEMUR_SECTION[1] * 1.1, FEMUR_SECTION[0] + 10)
    j1 = Pos(*femur_axis) * Rot(0, 0, phi) * Rot(90, 0, 0) * hub
    j2 = Pos(*knee) * Rot(0, 0, phi) * Rot(90, 0, 0) * hub
    return [coxa, femur, tibia, foot, j1, j2]


def human(x: float, ground_z: float):
    """A 6 ft (1829 mm) reference figure standing at (x, 0) on the ground."""
    s = HUMAN_HEIGHT / 1829.0
    parts = []
    parts.append(Pos(x, 0, ground_z + 1719 * s) * Sphere(110 * s))                       # head
    parts.append(Pos(x, 0, ground_z + 1585 * s) * Cylinder(55 * s, 70 * s))              # neck
    torso = Pos(x, 0, ground_z + 1330 * s) * Box(220 * s, 400 * s, 600 * s)
    parts.append(fillet(torso.edges().filter_by(Axis.Z), radius=60 * s))
    for sy in (1, -1):
        parts.append(Pos(x, sy * 250 * s, ground_z + 1300 * s) * Cylinder(45 * s, 640 * s))   # arms
        parts.append(Pos(x, sy * 100 * s, ground_z + 515 * s) * Cylinder(70 * s, 1030 * s))   # legs
        parts.append(Pos(x + 40 * s, sy * 100 * s, ground_z + 25 * s) * Box(260 * s, 100 * s, 50 * s))  # feet
    return Compound(children=parts)


def build(stance: Stance):
    W = BODY.slab_width(ACT)                      # hip stacks + a rail each side
    L, H = BODY.length, BODY.height
    RAIL = BODY.side_rail
    parts = []

    # Skeleton body: top deck, floor plate, two side rails
    for z in (PLATE_T / 2, H - PLATE_T / 2):
        plate = Pos(0, 0, z) * Box(L, W, PLATE_T)
        parts.append(fillet(plate.edges().filter_by(Axis.Z), radius=30))
    for s in (1, -1):
        parts.append(Pos(0, s * (W / 2 - RAIL / 2), H / 2) * Box(L - 120, RAIL, H))

    # Six hip stacks: three pancake actuators on each yaw axis
    z0 = PLATE_T + 4
    for hx in BODY.hip_x:
        for s in (1, -1):
            for i in range(3):
                zc = z0 + ACT.thickness / 2 + i * (ACT.thickness + ACT.stack_gap)
                parts.append(Pos(hx, s * BODY.width / 2, zc) * Cylinder(ACT.od / 2, ACT.thickness))
            # yaw output hub down through the floor to the coxa
            parts.append(Pos(hx, s * BODY.width / 2, -HIP_DROP / 2) * Cylinder(40, HIP_DROP + 6))

    # Two hot-swap batteries in the gaps between hip stacks, compute in the centre
    for s in (1, -1):
        parts.append(Pos(s * 165, 0, PLATE_T + BATTERY[2] / 2 + 2) * Box(*BATTERY))
    parts.append(Pos(0, 0, PLATE_T + COMPUTE[2] / 2 + 2) * Box(*COMPUTE))

    # Legs: the coxa centreline sits just below the floor plate
    for hx, yaw in zip(BODY.hip_x, stance.yaw_deg):
        for s in (1, -1):
            parts += leg((hx, s * BODY.width / 2, -HIP_DROP), s, stance, yaw)

    robot = Compound(children=parts)
    ground_z = -HIP_DROP - stance.hip_height
    figure = human(-L / 2 - 650, ground_z)
    return robot, figure


def describe(robot) -> str:
    bb = robot.bounding_box()
    return f"{bb.size.X:.0f} × {bb.size.Y:.0f} × {bb.size.Z:.0f} mm"

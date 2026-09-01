"""Shared build123d construction for the hexapod skeleton concepts.

Not a concept itself (defines no PART).  Both concept modules call
`build(stance)` and differ only in the Stance they pass in, so the two renders
are the same robot in two postures — which is exactly the choice on the table.
"""
from __future__ import annotations

import math
import os
import sys

from build123d import (Axis, Box, Compound, Cylinder, Location, Pos, Rot, Sphere, fillet)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from hexapod_model import ACT, BODY, LEG, Stance, ik  # noqa: E402

FEMUR_SECTION = (40.0, 26.0)   # width, thickness of the femur link
TIBIA_SECTION = (30.0, 20.0)
FOOT_D = 45.0
PLATE_T = 6.0
BATTERY = (140.0, 200.0, 110.0)
COMPUTE = (120.0, 140.0, 60.0)


def _link(length: float, section: tuple[float, float]):
    """A link along +x from the origin, rounded ends."""
    w, t = section
    body = Pos(length / 2, 0, 0) * Box(length, w, t)
    return fillet(body.edges().filter_by(Axis.Y), radius=min(w, t) * 0.45)


def _place(shape, at, yaw_deg: float, pitch_deg: float):
    """Pitch about local y (positive tips +x up), then yaw about z, then move."""
    return Pos(*at) * Rot(0, 0, yaw_deg) * Rot(0, -pitch_deg, 0) * shape


def leg(hip, side: int, stance: Stance, yaw_deg: float):
    """One leg.  hip = (x, y, z) of the yaw axis at the underside of the slab."""
    hx, hy, hz = hip
    a1, k = ik(LEG, stance.foot_reach, -stance.hip_height)
    a2 = a1 + k - math.pi                       # tibia direction from the knee
    if stance.leg_plane == "radial":
        # leg plane direction from +y (outboard) rotated `yaw_deg` toward +x
        d = (math.sin(math.radians(yaw_deg)), side * math.cos(math.radians(yaw_deg)))
        femur_axis = (hx + LEG.coxa * d[0], hy + LEG.coxa * d[1], hz)
        coxa = _place(_link(LEG.coxa, (50.0, 30.0)), (hx, hy, hz), math.degrees(math.atan2(d[1], d[0])), 0)
    else:
        # sagittal: coxa steps outboard, leg plane runs along ±x
        d = (1.0 if yaw_deg >= 0 else -1.0, 0.0)
        femur_axis = (hx, hy + side * LEG.coxa, hz)
        coxa = _place(_link(LEG.coxa, (50.0, 30.0)), (hx, hy, hz), 90 * side, 0)
    phi = math.degrees(math.atan2(d[1], d[0]))
    femur = _place(_link(LEG.femur, FEMUR_SECTION), femur_axis, phi, math.degrees(a1))
    knee = (femur_axis[0] + LEG.femur * math.cos(a1) * d[0],
            femur_axis[1] + LEG.femur * math.cos(a1) * d[1],
            femur_axis[2] + LEG.femur * math.sin(a1))
    tibia = _place(_link(LEG.tibia, TIBIA_SECTION), knee, phi, math.degrees(a2))
    foot_c = (knee[0] + LEG.tibia * math.cos(a2) * d[0],
              knee[1] + LEG.tibia * math.cos(a2) * d[1],
              knee[2] + LEG.tibia * math.sin(a2))
    foot = Pos(*foot_c) * Sphere(FOOT_D / 2)
    # joint hubs so the axes read in the render
    hub = Cylinder(FEMUR_SECTION[1] * 1.1, FEMUR_SECTION[0] + 10)
    j1 = Pos(*femur_axis) * Rot(0, 0, phi) * Rot(90, 0, 0) * hub
    j2 = Pos(*knee) * Rot(0, 0, phi) * Rot(90, 0, 0) * hub
    return [coxa, femur, tibia, foot, j1, j2]


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
            # yaw output shaft down through the floor
            parts.append(Pos(hx, s * BODY.width / 2, -6) * Cylinder(28, 24))

    # Two hot-swap batteries in the gaps between hip stacks, compute in the centre
    for s in (1, -1):
        parts.append(Pos(s * 165, 0, PLATE_T + BATTERY[2] / 2 + 2) * Box(*BATTERY))
    parts.append(Pos(0, 0, PLATE_T + COMPUTE[2] / 2 + 2) * Box(*COMPUTE))

    # Legs: yaw axis emerges below the floor plate at z = 0 (the femur-axis height)
    for hx, yaw in zip(BODY.hip_x, stance.yaw_deg):
        for s in (1, -1):
            parts += leg((hx, s * BODY.width / 2, -18), s, stance, yaw)

    return Compound(children=parts)

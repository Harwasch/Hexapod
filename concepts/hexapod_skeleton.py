"""Shared build123d construction for the hexapod skeleton concepts.

Not a concept itself (defines no PART).  Each concept module calls
`build(topology)` with one of the leg topologies from analysis/leg3d.py, so
the renders are the same body with different legs — which is the choice on
the table.  Joint positions and axes come from the 3-D leg model's forward
kinematics, so the picture and the torque numbers cannot disagree.

`build_groups()` returns named groups of solids (body, actuators, coxa,
femur, tibia, figure) for the review page's 3-D viewer; `build()` flattens
them to (robot, figure) for the vision renderer.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from build123d import Axis, Box, Compound, Cylinder, Pos, Rot, Sphere, fillet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from hexapod_model import ACT, BODY, HUMAN_HEIGHT  # noqa: E402
from leg3d import Topology  # noqa: E402

COXA_SECTION = (56.0, 34.0)    # width, thickness of the coxa arm (carries the overturning moment)
FEMUR_SECTION = (40.0, 26.0)
TIBIA_SECTION = (30.0, 20.0)
FOOT_D = 45.0
PLATE_T = 6.0
BATTERY = (140.0, 200.0, 110.0)
COMPUTE = (120.0, 140.0, 60.0)
HIP_DROP = 18.0                # first joint / coxa centreline below the floor plate
LEG_YAW = (30.0, 0.0, -30.0)   # front, mid, rear leg-plane yaw for radial legs


def _link_between(p, q, section):
    """A rounded bar from point p to point q."""
    d = np.asarray(q, float) - np.asarray(p, float)
    L = float(np.linalg.norm(d))
    w, t = section
    bar = Pos(L / 2, 0, 0) * Box(L, w, t)
    bar = fillet(bar.edges().filter_by(Axis.Y), radius=min(w, t) * 0.45)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return Pos(*p) * Rot(0, 0, yaw) * Rot(0, -pitch, 0) * bar


def _hub(p, axis, radius, length):
    """A cylinder centred on p along a unit axis."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    yaw = math.degrees(math.atan2(a[1], a[0]))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, a[2]))))
    return Pos(*p) * Rot(0, 0, yaw) * Rot(0, tilt, 0) * Cylinder(radius, length)


def leg_points(topo: Topology, hip, side: int, base_yaw_deg: float, mirror_x: bool):
    """World joint points and axes for one leg at the neutral pose."""
    pts, axes = topo.fk((0.0, 0.0, 0.0))
    Rz = np.array([[math.cos(math.radians(base_yaw_deg)), -math.sin(math.radians(base_yaw_deg)), 0],
                   [math.sin(math.radians(base_yaw_deg)), math.cos(math.radians(base_yaw_deg)), 0], [0, 0, 1]])
    M = np.diag([-1.0 if mirror_x else 1.0, float(side), 1.0])
    hip = np.asarray(hip, float)
    P = [hip + M @ (Rz @ p) for p in pts]
    A = [M @ (Rz @ a) for a in axes]
    return P, A


def leg(topo: Topology, hip, side: int, base_yaw_deg: float, mirror_x: bool):
    P, A = leg_points(topo, hip, side, base_yaw_deg, mirror_x)
    coxa = _link_between(P[0], P[1], COXA_SECTION)
    femur = _link_between(P[1], P[2], FEMUR_SECTION)
    tibia = _link_between(P[2], P[3], TIBIA_SECTION)
    foot = Pos(*P[3]) * Sphere(FOOT_D / 2)
    j1 = _hub(P[1], A[1], FEMUR_SECTION[1] * 1.1, FEMUR_SECTION[0] + 10)
    j2 = _hub(P[2], A[2], FEMUR_SECTION[1] * 1.1, FEMUR_SECTION[0] + 10)
    return {"coxa": [coxa], "femur": [femur, j1, j2], "tibia": [tibia, foot]}


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


def ground_level(topo: Topology) -> float:
    """z of the ground plane in the body frame."""
    return -HIP_DROP - topo.hip_height


def build_groups(topo: Topology) -> dict:
    W = BODY.slab_width(ACT)                      # hip stacks + a rail each side
    L, H = BODY.length, BODY.height
    RAIL = BODY.side_rail
    parts, actuators = [], []
    legs = {"coxa": [], "femur": [], "tibia": []}
    radial = topo.stride_joint == "yaw"

    # Skeleton body: top deck, floor plate, two side rails
    for z in (PLATE_T / 2, H - PLATE_T / 2):
        plate = Pos(0, 0, z) * Box(L, W, PLATE_T)
        parts.append(fillet(plate.edges().filter_by(Axis.Z), radius=30))
    for s in (1, -1):
        parts.append(Pos(0, s * (W / 2 - RAIL / 2), H / 2) * Box(L - 120, RAIL, H))

    # Six hip stacks: three pancake actuators on each hip
    z0 = PLATE_T + 4
    for hx in BODY.hip_x:
        for s in (1, -1):
            for i in range(3):
                zc = z0 + ACT.thickness / 2 + i * (ACT.thickness + ACT.stack_gap)
                actuators.append(Pos(hx, s * BODY.width / 2, zc) * Cylinder(ACT.od / 2, ACT.thickness))
            actuators.append(Pos(hx, s * BODY.width / 2, -HIP_DROP / 2) * Cylinder(40, HIP_DROP + 6))

    # Two hot-swap batteries in the gaps between hip stacks, compute in the centre
    for s in (1, -1):
        parts.append(Pos(s * 165, 0, PLATE_T + BATTERY[2] / 2 + 2) * Box(*BATTERY))
    parts.append(Pos(0, 0, PLATE_T + COMPUTE[2] / 2 + 2) * Box(*COMPUTE))

    # Legs: the first joint sits just below the floor plate
    for i, hx in enumerate(BODY.hip_x):
        for s in (1, -1):
            hip = (hx, s * BODY.width / 2, -HIP_DROP)
            base_yaw = LEG_YAW[i] if radial else 0.0
            mirror_x = (not radial) and hx < 0          # rear mammal legs: knee back
            for k, v in leg(topo, hip, s, base_yaw, mirror_x).items():
                legs[k] += v

    groups = {"body": parts, "actuators": actuators, **legs,
              "figure": list(human(-L / 2 - 650, ground_level(topo)).children)}
    return {k: Compound(children=v) for k, v in groups.items()}


def build(topo: Topology):
    """(robot, figure) as two compounds — what the concept modules use."""
    g = build_groups(topo)
    robot = Compound(children=[s for k, c in g.items() if k != "figure" for s in c.solids()])
    return robot, g["figure"]


def describe(robot) -> str:
    bb = robot.bounding_box()
    return f"{bb.size.X:.0f} × {bb.size.Y:.0f} × {bb.size.Z:.0f} mm"

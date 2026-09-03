#!/opt/hw-py/bin/python
"""General arrangement of the robot as a physical object.

    /opt/hw-py/bin/python analysis/arrangement.py

Writes docs/design/arrangement/{ga-side,ga-front,ga-top,ga-body-plan,
ga-leg-envelope,ga-scale}.png, hw/arrangement.json and
docs/design/10-general-arrangement.md.

Every dimension is read from a model, and the model it came from is named in
hw/arrangement.json:

  * the leg's real pivots, link lengths and joint windows from the round-14
    CAD, cad/leg/leg.json;
  * the actuator from hw/stator/frameless_motor.json (round 14b, the Wheemo
    frameless motor scaled to Ø160 x 25 with the cycloid inside its bore);
  * the body slab, the hip grid and the mass budget from
    analysis/hexapod_model.py;
  * the battery and electronics envelopes from concepts/hexapod_skeleton.py.

Anything that is an assumption rather than a model output is drawn with the
word "assumed" or "estimate" on the figure itself.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Polygon, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "concepts"))
from hexapod_model import BODY, MASS, ENERGY, HUMAN_HEIGHT, YAW_RANGE_DEG, G  # noqa: E402
from drawing import dim, label, frame, INK, DIM, FILL, ACTC, LEGC              # noqa: E402
from hexapod_skeleton import BATTERY, COMPUTE, LEG_YAW                         # noqa: E402

OUT = os.path.join(ROOT, "docs", "design", "arrangement")
DOC = os.path.join(ROOT, "docs", "design", "10-general-arrangement.md")
JSN = os.path.join(ROOT, "hw", "arrangement.json")
os.makedirs(OUT, exist_ok=True)

FM = json.load(open(os.path.join(ROOT, "hw", "stator", "frameless_motor.json")))
LG = json.load(open(os.path.join(ROOT, "cad", "leg", "leg.json")))
BOMT = json.load(open(os.path.join(ROOT, "hw", "leg", "bom_totals.json")))

GREY, HID, NOTE, WARN = "#7f8c8d", "#8d8d8d", "#555555", "#b03a2e"
OKC = "#1e8449"


def mm(v):
    """Millimetres, without a pointless trailing .0."""
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"

# ---------------------------------------------------------------------------
# 1. The actuator can, from hw/stator/frameless_motor.json
# ---------------------------------------------------------------------------
# The half-section in analysis/frameless_motor.py draws the housing as the
# motor plus 6 mm radially and 6 mm each end; that is where Ø172 x 37 comes
# from.  The same file's sweep comment sizes the motor against a Ø192 can
# carried over from the PCB unit -- 12 mm more on the diameter.  Both are
# recorded; the drawings use the half-section's Ø172.
HOUSING_R, HOUSING_Z = 6.0, 6.0
CAN_OD_ALT = 192.0                     # the can the sweep's OD limit assumed
MOT_FK, MOT_YAW = FM["pick"]["motor"], FM["yaw"]["motor"]
# What §9.17 quotes: the motor plus 6 mm of case radially and each end.
CAN_FK_917 = (MOT_FK["od_mm"] + 2 * HOUSING_R, MOT_FK["len_mm"] + 2 * HOUSING_Z)     # 172 x 37
CAN_YAW_917 = (MOT_YAW["od_mm"] + 2 * HOUSING_R, MOT_YAW["len_mm"] + 2 * HOUSING_Z)  # 142 x 32
M_FK_917 = FM["pick"]["m_fk"]                                   # 2.84 kg
M_YAW_917 = M_FK_917 - FM["yaw"]["saving_g"] / 1000.0           # 2.46 kg


def _cad_unit(name):
    p = os.path.join(ROOT, "cad", "actuator", f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


CAD_FK, CAD_YAW = _cad_unit("frameless"), _cad_unit("frameless-yaw")
# The built unit wins over the section sketch when the CAD exists.  It is the
# same diameter and 12.7 mm taller, because §9.17's height left out the
# reducer's own axial stack.
CAN_FK = ((CAD_FK["envelope"]["od_mm"], CAD_FK["envelope"]["height_mm"]) if CAD_FK else CAN_FK_917)
CAN_YAW = ((CAD_YAW["envelope"]["od_mm"], CAD_YAW["envelope"]["height_mm"]) if CAD_YAW else CAN_YAW_917)
M_FK = (CAD_FK["total_g"] / 1000.0) if CAD_FK else M_FK_917
M_YAW = (CAD_YAW["total_g"] / 1000.0) if CAD_YAW else M_YAW_917
CAN_SOURCE = ("cad/actuator/frameless{,-yaw}.json (the built unit)" if CAD_FK
              else "hw/stator/frameless_motor.json + the §9.17 half-section's housing allowance")

# ---------------------------------------------------------------------------
# 2. Geometry, from cad/leg/leg.json and analysis/hexapod_model.py
# ---------------------------------------------------------------------------
COXA = LG["links"]["coxa"]
FEMUR = LG["links"]["femur"]
TIBIA = LG["links"]["tibia"]
COXA_DROP = -LG["pivots"]["femur_pivot"][2]          # 100 mm, femur axis below the floor plate
HIP_H = LG["pivots"]["hip_height_mm"]                # 323.2 mm, femur axis above ground
PLATE_H = LG["pivots"]["floor_plate_above_ground_mm"]  # 423.2 mm
SLAB_H = LG["body"]["slab_height"]                   # 220
SLAB_W = LG["body"]["slab_width"]                    # 390
EDGE = LG["body"]["slab_edge_from_yaw_axis"]         # 75
SLAB_L = BODY.length                                 # 900
HIPX = tuple(BODY.hip_x)                             # 330 / 0 / -330
HIPY = BODY.width / 2                                # 120
DECK = PLATE_H + SLAB_H
POD_BOT = LG["pivots"]["pod_bottom_z"]               # -96
COXA_PLATE_BOT = LG["pivots"]["coxa_plate_bottom_z"]  # -142
CLEARANCE = PLATE_H + COXA_PLATE_BOT                 # lowest fixed structure above ground
FEMUR_DEG = LG["stance"]["femur_deg"]
TAU_DEG = LG["stance"]["tau_deg"]
KNEE_DEG = LG["stance"]["knee_deg"]
FEM_RANGE = tuple(LG["femur_range_deg"])             # (-60, 85)
KNEE_LIM = tuple(LG["knee_limits_deg"])              # (15, 145)
TAU_RANGE = tuple(LG["tau_range_deg"])               # (200, 330)
FEM_UP_CLEAR = LG["clearances"]["summary"]["femur_up_limit_deg"]   # by yaw, deg
FEM_UP_0 = FEM_UP_CLEAR["+0"]                        # 75.1 deg at yaw 0

PLATE_T = 6.0                                        # floor / deck plate, concepts


# ---------------------------------------------------------------------------
# 3. Kinematics
# ---------------------------------------------------------------------------
def leg_plane(femur_deg=FEMUR_DEG, tau_deg=TAU_DEG):
    """(r, z) of yaw axis, femur pivot, knee and foot in the leg plane,
    relative to the yaw axis at the floor-plate underside.  r is outboard."""
    a, t = math.radians(femur_deg), math.radians(tau_deg)
    p0 = np.array([0.0, 0.0])
    p1 = np.array([COXA, -COXA_DROP])
    p2 = p1 + FEMUR * np.array([math.cos(a), math.sin(a)])
    p3 = p2 + TIBIA * np.array([math.cos(t), math.sin(t)])
    return np.array([p0, p1, p2, p3])


def leg_world(i_hip, side, femur_deg=FEMUR_DEG, tau_deg=TAU_DEG):
    """The same four points in the body frame, for hip i and side +1 (left)."""
    P = leg_plane(femur_deg, tau_deg)
    yaw = math.radians(LEG_YAW[i_hip])
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for r, z in P:
        x, y = r * s, r * c           # leg plane fans out from +y, +ve yaw toward +x
        out.append(np.array([HIPX[i_hip] + x, side * (HIPY + y), z]))
    return np.array(out)


ALL_LEGS = [leg_world(i, s) for i in range(3) for s in (1, -1)]
FEET = np.array([P[3] for P in ALL_LEGS])
KNEES = np.array([P[2] for P in ALL_LEGS])
NEUTRAL = leg_plane()
FOOT_RADIUS = float(NEUTRAL[3][0])                  # 326.8 mm from the yaw axis

STANCE_WIDTH = float(FEET[:, 1].max() - FEET[:, 1].min())
STANCE_LENGTH = float(FEET[:, 0].max() - FEET[:, 0].min())


def can_half_extent(axis, R, t):
    """Half-extents in x, y, z of a disc of radius R and half-thickness t whose
    axis is the unit vector `axis`."""
    out = []
    for e in np.eye(3):
        d = abs(float(np.dot(axis, e)))
        out.append(d * t + R * math.sqrt(max(0.0, 1.0 - d * d)))
    return np.array(out)


def cans():
    """(centre, axis, radius, half-thickness, name) for all eighteen units."""
    out = []
    for i in range(3):
        for s in (1, -1):
            P = leg_world(i, s)
            yaw = math.radians(LEG_YAW[i])
            u = np.array([math.sin(yaw), s * math.cos(yaw), 0.0])   # leg-plane radial, outboard
            a = np.array([u[1], -u[0], 0.0])                        # pitch axis = u x z
            out.append((np.array([HIPX[i], s * HIPY, -CAN_YAW[1] / 2 - 4.0]), np.array([0.0, 0.0, 1.0]),
                        CAN_YAW[0] / 2, CAN_YAW[1] / 2, f"yaw{i}{s}"))
            out.append((P[1], a, CAN_FK[0] / 2, CAN_FK[1] / 2, f"femur{i}{s}"))
            out.append((P[2], a, CAN_FK[0] / 2, CAN_FK[1] / 2, f"knee{i}{s}"))
    return out


CANS = cans()
_lo = np.min([c - can_half_extent(a, R, t) for c, a, R, t, _ in CANS], axis=0)
_hi = np.max([c + can_half_extent(a, R, t) for c, a, R, t, _ in CANS], axis=0)
CAN_ENV = (_lo, _hi)
OVER_CANS_L = float(_hi[0] - _lo[0])
OVER_CANS_W = float(_hi[1] - _lo[1])
KNEE_TOP = float(_hi[2]) + PLATE_H                  # highest can, above ground
OVERALL_W = max(STANCE_WIDTH, SLAB_W, OVER_CANS_W)
OVERALL_L = max(STANCE_LENGTH, SLAB_L, OVER_CANS_L)
OVERALL_H = max(DECK, KNEE_TOP)
WIDEST_IS_CANS = OVER_CANS_W > STANCE_WIDTH
LONGEST_IS_CANS = OVER_CANS_L > STANCE_LENGTH


# ---------------------------------------------------------------------------
# 4. The leg's reachable workspace in its own plane
# ---------------------------------------------------------------------------
def workspace(n=421):
    """Foot (r, z) relative to the yaw axis over the femur and knee windows,
    with the tibia's absolute-angle window applied.  Returns the feasible
    (femur, knee, r, z) arrays."""
    f = np.linspace(FEM_RANGE[0], FEM_RANGE[1], n)
    k = np.linspace(KNEE_LIM[0], KNEE_LIM[1], n)
    F, K = np.meshgrid(f, k)
    T = F + K + 180.0                       # tibia absolute angle tau
    ok = (T >= TAU_RANGE[0]) & (T <= TAU_RANGE[1])
    fa, ta = np.radians(F[ok]), np.radians(T[ok])
    r = COXA + FEMUR * np.cos(fa) + TIBIA * np.cos(ta)
    z = -COXA_DROP + FEMUR * np.sin(fa) + TIBIA * np.sin(ta)
    return F[ok], K[ok], T[ok], r, z


WF, WK, WT, WR, WZ = workspace()
GROUND_Z = -PLATE_H                          # ground in the body frame
_on_ground = np.abs(WZ - GROUND_Z) < 4.0
REACH_GROUND = (float(WR[_on_ground].min()), float(WR[_on_ground].max())) if _on_ground.any() else (float("nan"),) * 2
REACH_MAX = float(WR.max())
STEP_UP = float(WZ.max() - GROUND_Z)         # highest the foot goes above the ground line
DROP_MAX = float(GROUND_Z - WZ.min())        # deepest below the ground line

_i_reach = int(np.argmax(WR))
POSE_EXT = (float(WF[_i_reach]), float(WT[_i_reach]))
POSE_FOLD = (FEM_UP_0, FEM_UP_0 + KNEE_LIM[0] + 180.0)      # knee shut, femur at its clearance limit
FOLD = leg_plane(*POSE_FOLD)
FOLD_R = float(FOLD[:, 0].max() - FOLD[:, 0].min())
FOLD_Z = float(FOLD[:, 1].max() - FOLD[:, 1].min())
FOLD_FOOT_R = float(FOLD[3][0])


# ---------------------------------------------------------------------------
# 5. Where the eighteen units go, and whether they fit
# ---------------------------------------------------------------------------
BATT_XY = [(165.0, 0.0), (-165.0, 0.0)]        # concepts/hexapod_skeleton placement
CLEAR = 3.0                                     # mm between cans


def _rects():
    r = [(x - BATTERY[0] / 2, y - BATTERY[1] / 2, BATTERY[0], BATTERY[1], "battery") for x, y in BATT_XY]
    r.append((-COMPUTE[0] / 2, -COMPUTE[1] / 2, COMPUTE[0], COMPUTE[1], "electronics"))
    return r


def pack_inboard(od=CAN_FK[0], want=12, with_boxes=True):
    """Greedy bottom-left packing of `want` Ø`od` cans into the slab planform,
    outside the six yaw stacks and the battery / electronics boxes.  One level."""
    R = od / 2 + CLEAR
    keep = [(HIPX[i], s * HIPY, CAN_YAW[0] / 2 + CLEAR) for i in range(3) for s in (1, -1)]
    rects = _rects() if with_boxes else []
    placed = []
    step = 4.0
    xs = np.arange(-SLAB_L / 2 + R, SLAB_L / 2 - R + step, step)
    ys = np.arange(-SLAB_W / 2 + R, SLAB_W / 2 - R + step, step)
    for y in ys:
        for x in xs:
            if len(placed) >= want:
                break
            if any(math.hypot(x - kx, y - ky) < R + kr - CLEAR for kx, ky, kr in keep):
                continue
            if any(_circle_rect(x, y, R, rr) for rr in rects):
                continue
            if any(math.hypot(x - px, y - py) < 2 * R for px, py in placed):
                continue
            placed.append((x, y))
    return placed


def _circle_rect(cx, cy, r, rect):
    x0, y0, w, h, _ = rect
    dx = max(x0 - cx, 0, cx - (x0 + w))
    dy = max(y0 - cy, 0, cy - (y0 + h))
    return math.hypot(dx, dy) < r


PACKED = pack_inboard()
PACKED_BARE = pack_inboard(with_boxes=False)     # the slab's raw capacity, boxes removed
A_SLAB = SLAB_L * SLAB_W
A_CAN_FK = math.pi * (CAN_FK[0] / 2) ** 2
A_CAN_YAW = math.pi * (CAN_YAW[0] / 2) ** 2
A_YAW_ALL = 6 * A_CAN_YAW
A_BOX = 2 * BATTERY[0] * BATTERY[1] + COMPUTE[0] * COMPUTE[1]
A_FREE = A_SLAB - A_YAW_ALL - A_BOX
A_NEED_12 = 12 * A_CAN_FK
AREA_SHORT = A_NEED_12 / A_FREE

YAW_OVERHANG = CAN_YAW[0] / 2 - EDGE                 # -ve = inside the slab edge
YAW_OVERHANG_FK = CAN_FK[0] / 2 - EDGE               # if the yaw shared the femur/knee unit
STACK_H = 3 * CAN_FK[1] + 2 * 8.0                    # three cans coaxial, 8 mm gaps


# ---------------------------------------------------------------------------
# 6. Mass
# ---------------------------------------------------------------------------
M_FIXED = FM["requirement"]["m_fixed"]                       # 29.2 kg, everything but the actuators
M_ACT = 12 * M_FK + 6 * M_YAW
M_ROBOT_CLOSURE = FM["pick"]["m_robot"]                      # 80.3 kg, all 18 units identical
M_ACT_917 = 12 * M_FK_917 + 6 * M_YAW_917
M_ROBOT_SIZED_YAW = M_FIXED + M_ACT
LEG_STRUCT_CAD = LG["leg_structure_g"] / 1000.0              # 8.00 kg incl. the capstan transmission
LEG_TRANS_CAD = LG["group_mass_g"]["transmission"] / 1000.0  # 3.07 kg of rope drive, deleted in 14b
LEG_EXTRAS_KEPT = (LG["unit_extras_g"]["hollow_eccentric_and_bearing_upgrade_x3"]
                   + LG["unit_extras_g"]["yaw_output_bearing_RB7013_vs_RB5013"]) / 1000.0
LEG_STRUCT_EST = LEG_STRUCT_CAD - LEG_TRANS_CAD + LEG_EXTRAS_KEPT     # estimate
M_ROBOT_CAD = M_FIXED - MASS.legs + 6 * LEG_STRUCT_EST + M_ACT
C_PER_KG = FM["requirement"]["c_per_kg"]
T_FK = FM["pick"]["T_joint_fk"]
T_YAW_SIZED = MOT_YAW["T_cont"] * FM["pick"]["ratio_yaw"] * 0.90
MARGIN_AT = {d: (T_YAW_SIZED if d == "yaw" else T_FK) / (C_PER_KG[d] * M_ROBOT_CAD) for d in C_PER_KG}
MARGIN_CLOSURE = dict(FM["pick"]["margin"])

WALK_W = ENERGY.walking_power(M_ROBOT_CLOSURE, 1.0)
PACK_WH = ENERGY.pack_wh(WALK_W)
PACK_KG = ENERGY.pack_kg(WALK_W)
WALK_W_CAD = ENERGY.walking_power(M_ROBOT_CAD, 1.0)
PACK_WH_CAD = ENERGY.pack_wh(WALK_W_CAD)
PACK_KG_CAD = ENERGY.pack_kg(WALK_W_CAD)


# ---------------------------------------------------------------------------
# 7. Silhouettes for scale
# ---------------------------------------------------------------------------
_HUMAN = [(0.040, 0.900), (0.100, 0.865), (0.118, 0.640), (0.108, 0.470), (0.078, 0.470),
          (0.080, 0.320), (0.070, 0.020), (0.078, 0.000), (0.014, 0.000), (0.022, 0.320),
          (0.000, 0.470)]
_DOG = [(1.02, 0.74), (1.03, 0.82), (0.90, 0.86), (0.84, 0.96), (0.78, 1.01), (0.70, 1.02),
        (0.66, 1.09), (0.60, 0.99), (0.45, 0.97), (0.25, 1.00), (0.00, 0.97), (-0.25, 0.94),
        (-0.42, 0.96), (-0.55, 1.02), (-0.80, 1.10), (-0.78, 1.00), (-0.52, 0.88), (-0.46, 0.60),
        (-0.54, 0.34), (-0.48, 0.03), (-0.40, 0.00), (-0.36, 0.03), (-0.34, 0.35), (-0.28, 0.58),
        (-0.10, 0.52), (0.14, 0.50), (0.26, 0.55), (0.24, 0.30), (0.22, 0.00), (0.32, 0.00),
        (0.34, 0.32), (0.36, 0.62), (0.40, 0.80), (0.52, 0.84), (0.72, 0.80), (0.88, 0.76)]


def human(ax, x0, y0, h=HUMAN_HEIGHT, fc="#dfe3e6", ec="#a9b0b6", z=0):
    pts = [(x0 + fx * h, y0 + fy * h) for fx, fy in _HUMAN]
    pts += [(x0 - fx * h, y0 + fy * h) for fx, fy in reversed(_HUMAN)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, lw=0.8, zorder=z))
    ax.add_patch(Circle((x0, y0 + 0.935 * h), 0.062 * h, facecolor=fc, edgecolor=ec, lw=0.8, zorder=z))


def dog(ax, x0, y0, h, fc="#e2ded6", ec="#b3aca0", z=0, flip=False):
    s = -1 if flip else 1
    ax.add_patch(Polygon([(x0 + s * fx * h, y0 + fy * h) for fx, fy in _DOG], closed=True,
                         facecolor=fc, edgecolor=ec, lw=0.8, zorder=z))


def draw_leg2(ax, pts, lw=5, alpha=1.0):
    for (p, q), name in zip(zip(pts[:-1], pts[1:]), ("coxa", "femur", "tibia")):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=LEGC[name], lw=lw, solid_capstyle="round",
                zorder=3, alpha=alpha)
    for p in pts[:3]:
        ax.plot(p[0], p[1], "o", color=INK, ms=3.5, zorder=4, alpha=alpha)
    ax.plot(pts[3][0], pts[3][1], "o", color=LEGC["tibia"], ms=5.5, mec=INK, zorder=4, alpha=alpha)


def ground(ax, x0, x1, z=-PLATE_H):
    ax.plot([x0, x1], [z, z], color=GREY, lw=1.5, zorder=0)
    ax.add_patch(Rectangle((x0, z - 28), x1 - x0, 28, facecolor="#ececec", edgecolor="none", zorder=0))


def can_rect(ax, cx, cz, od, h, axis="h", fc="#cfe9e6", ec=ACTC, ls="-", z=2, alpha=0.95):
    """A can seen edge-on: `axis`='h' the can's axis is horizontal in this view
    (so it draws h wide x od tall), 'v' the axis is vertical (od wide x h tall)."""
    w, t = (h, od) if axis == "h" else (od, h)
    ax.add_patch(Rectangle((cx - w / 2, cz - t / 2), w, t, facecolor=fc, edgecolor=ec,
                           lw=1.1, ls=ls, zorder=z, alpha=alpha))


# ===========================================================================
# VIEWS
# ===========================================================================
TITLE_NOTE = "all dimensions mm — every number from analysis/arrangement.py; sources in hw/arrangement.json"


def notes(ax, x, y, lines, color=NOTE, fs=8.5, va="top"):
    ax.text(x, y, lines, fontsize=fs, color=color, va=va, ha="left", zorder=6)


def view_side():
    fig, ax = plt.subplots(figsize=(14, 12.9))
    HX = -830.0
    ground(ax, -1100, 1400)
    # human for scale, standing beside the robot
    human(ax, HX, -PLATE_H)
    ax.text(HX, -PLATE_H + HUMAN_HEIGHT + 55, f"{HUMAN_HEIGHT/1000:.2f} m person", ha="center", fontsize=8.5, color=NOTE)
    dim(ax, (HX - 200, -PLATE_H), (HX - 200, -PLATE_H + HUMAN_HEIGHT), 0, f"{HUMAN_HEIGHT:.0f}")
    # slab
    ax.add_patch(Rectangle((-SLAB_L / 2, 0), SLAB_L, SLAB_H, facecolor=FILL, edgecolor=INK, lw=1.3, zorder=1))
    for z0 in (0.0, SLAB_H - PLATE_T):
        ax.add_patch(Rectangle((-SLAB_L / 2, z0), SLAB_L, PLATE_T, facecolor="#b9bcbf", edgecolor="none", zorder=1))
    # legs, far pair light, near pair heavy
    for i in range(3):
        for s in (-1, 1):
            P = leg_world(i, s)
            draw_leg2(ax, [(p[0], p[2]) for p in P], lw=5 if s == 1 else 2.2, alpha=1.0 if s == 1 else 0.45)
    # actuator cans
    for i in range(3):
        P = leg_world(i, 1)
        can_rect(ax, HIPX[i], -CAN_YAW[1] / 2 - 4, CAN_YAW[0], CAN_YAW[1], axis="v")
        can_rect(ax, P[1][0], P[1][2], CAN_FK[0], CAN_FK[1], axis="h")
        can_rect(ax, P[2][0], P[2][2], CAN_FK[0], CAN_FK[1], axis="h")
        ax.plot([HIPX[i], HIPX[i]], [-COXA_DROP - 40, SLAB_H + 40], color=ACTC, lw=0.6, ls="-.", zorder=1)
    # dimensions
    dim(ax, (-SLAB_L / 2, SLAB_H), (SLAB_L / 2, SLAB_H), 190, f"body slab length {SLAB_L:.0f}")
    dim(ax, (FEET[:, 0].min(), -PLATE_H), (FEET[:, 0].max(), -PLATE_H), -110, f"stance length {STANCE_LENGTH:.0f}")
    dim(ax, (CAN_ENV[0][0], -PLATE_H), (CAN_ENV[1][0], -PLATE_H), -215,
        f"length over the knee cans {OVER_CANS_L:.0f}")
    x = SLAB_L / 2 + 60
    dim(ax, (x, -PLATE_H), (x, -COXA_DROP), -260, f"hip (femur axis) height {HIP_H:.0f}")
    dim(ax, (x, -PLATE_H), (x, 0), -430, f"floor plate {PLATE_H:.0f}")
    dim(ax, (x, -PLATE_H), (x, SLAB_H), -600, f"deck height {DECK:.0f}")
    dim(ax, (x, -PLATE_H), (x, COXA_PLATE_BOT), -770, f"ground clearance {CLEARANCE:.0f}")
    ax.plot([-SLAB_L / 2, x + 770], [COXA_PLATE_BOT, COXA_PLATE_BOT], color=DIM, lw=0.6, ls=":", zorder=0)
    dim(ax, (HIPX[2], SLAB_H + 40), (HIPX[1], SLAB_H + 40), 60, f"{HIPX[1]-HIPX[2]:.0f}", ext=False)
    dim(ax, (HIPX[1], SLAB_H + 40), (HIPX[0], SLAB_H + 40), 60, f"{HIPX[0]-HIPX[1]:.0f}", ext=False)
    # callouts, in the empty space above the robot
    ax.annotate("", (HIPX[2], -CAN_YAW[1] / 2 - 4), (-560, 620), color=NOTE,
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
    notes(ax, -700, 780,
          f"yaw unit Ø{CAN_YAW[0]:.0f} × {mm(CAN_YAW[1])}, {M_YAW:.2f} kg, on the yaw axis\n"
          f"under the floor plate.  Placement assumed inside\nthe {-POD_BOT:.0f} mm hip pod from the leg CAD.")
    ax.annotate("", (leg_world(0, 1)[2][0], leg_world(0, 1)[2][2]), (620, 620), color=WARN,
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
    notes(ax, 340, 1030,
          f"femur and knee units Ø{CAN_FK[0]:.0f} × {mm(CAN_FK[1])}, {M_FK:.2f} kg each, from\n"
          f"cad/actuator/frameless.json — {CAN_FK[1]-CAN_FK_917[1]:.1f} mm taller and {M_FK-M_FK_917:.2f} kg\n"
          f"heavier than 08 §9.17's Ø{CAN_FK_917[0]:.0f} × {mm(CAN_FK_917[1])}, {M_FK_917:.2f} kg.\n"
          f"Drawn ON their joints: round 14b deletes the capstan,\n"
          f"so the reduction has nowhere else to be, and no CAD\nexists for that leg.", color=WARN)
    ax.text(-1080, -PLATE_H - 250, "+x forward →", fontsize=8.5, ha="left", color=NOTE)
    frame(ax, f"GENERAL ARRANGEMENT — SIDE (looking along −y), sprawl stance\n{TITLE_NOTE}",
          (-1120, 1420), (-PLATE_H - 330, -PLATE_H + HUMAN_HEIGHT + 200))
    fig.tight_layout()
    p = os.path.join(OUT, "ga-side.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def view_front():
    fig, ax = plt.subplots(figsize=(14, 12.9))
    HX = -880.0
    ground(ax, -1150, 1350)
    human(ax, HX, -PLATE_H)
    ax.text(HX, -PLATE_H + HUMAN_HEIGHT + 55, f"{HUMAN_HEIGHT/1000:.2f} m person", ha="center", fontsize=8.5, color=NOTE)
    dim(ax, (HX - 200, -PLATE_H), (HX - 200, -PLATE_H + HUMAN_HEIGHT), 0, f"{HUMAN_HEIGHT:.0f}")
    ax.add_patch(Rectangle((-SLAB_W / 2, 0), SLAB_W, SLAB_H, facecolor=FILL, edgecolor=INK, lw=1.3, zorder=1))
    for z0 in (0.0, SLAB_H - PLATE_T):
        ax.add_patch(Rectangle((-SLAB_W / 2, z0), SLAB_W, PLATE_T, facecolor="#b9bcbf", edgecolor="none", zorder=1))
    # front and rear legs behind, mid legs in the plane of the section
    for i in (0, 2):
        for s in (1, -1):
            P = leg_world(i, s)
            draw_leg2(ax, [(p[1], p[2]) for p in P], lw=2.2, alpha=0.35)
    for s in (1, -1):
        P = leg_world(1, s)
        draw_leg2(ax, [(p[1], p[2]) for p in P], lw=5)
        can_rect(ax, s * HIPY, -CAN_YAW[1] / 2 - 4, CAN_YAW[0], CAN_YAW[1], axis="v")
        for j in (1, 2):
            ax.add_patch(Circle((P[j][1], P[j][2]), CAN_FK[0] / 2, facecolor="#cfe9e6", edgecolor=ACTC, lw=1.1, zorder=2, alpha=0.95))
            ax.plot(P[j][1], P[j][2], "+", color=ACTC, ms=7, mew=1.1, zorder=3)
        ax.plot([s * HIPY] * 2, [COXA_PLATE_BOT - 20, SLAB_H + 40], color=ACTC, lw=0.6, ls="-.", zorder=1)
    # dimensions
    dim(ax, (-SLAB_W / 2, SLAB_H), (SLAB_W / 2, SLAB_H), 300, f"slab width {SLAB_W:.0f}")
    dim(ax, (-HIPY, SLAB_H), (HIPY, SLAB_H), 130, f"yaw axes {2*HIPY:.0f}", ext=False)
    fy = [P[3][1] for P in (leg_world(1, 1), leg_world(1, -1))]
    dim(ax, (min(fy), -PLATE_H), (max(fy), -PLATE_H), -110, f"stance width (mid legs) {STANCE_WIDTH:.0f}")
    dim(ax, (CAN_ENV[0][1], -PLATE_H), (CAN_ENV[1][1], -PLATE_H), -215, f"width over the knee cans {OVER_CANS_W:.0f}")
    x = max(fy) + 130
    dim(ax, (x, -PLATE_H), (x, -COXA_DROP), -170, f"hip height {HIP_H:.0f}")
    dim(ax, (x, -PLATE_H), (x, SLAB_H), -340, f"deck height {DECK:.0f}")
    dim(ax, (x, -PLATE_H), (x, COXA_PLATE_BOT), -510, f"ground clearance {CLEARANCE:.0f}")
    dim(ax, (SLAB_W / 2 + 40, 0), (SLAB_W / 2 + 40, SLAB_H), -30, f"{SLAB_H:.0f}")
    # link lengths, as labels rather than dimension lines — the cans leave no room
    Pl = leg_world(1, -1)
    for (p, q), name, val, off in zip(zip(Pl[:-1], Pl[1:]), ("coxa", "femur", "tibia"),
                                      (COXA, FEMUR, TIBIA), (150, 130, -130)):
        d = np.array([q[1] - p[1], q[2] - p[2]])
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        m = (np.array([p[1], p[2]]) + np.array([q[1], q[2]])) / 2 + n * off
        ax.text(m[0], m[1], f"{name} {val:.0f}", fontsize=8.5, color=LEGC[name], ha="center", va="center", zorder=6)
    notes(ax, -1060, 940,
          f"femur {FEMUR_DEG:.0f}° above horizontal, tibia vertical (τ = {TAU_DEG:.0f}°).\n"
          f"The coxa drops {COXA_DROP:.0f} mm from the floor plate to the\nfemur axis (cad/leg/leg.json).")
    ax.annotate("", (P[2][1] + CAN_FK[0] / 2, P[2][2] + 40), (620, 620), arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
    notes(ax, 340, 1030,
          f"Ø{CAN_FK[0]:.0f} × {mm(CAN_FK[1])} femur and knee cans, true size, {M_FK:.2f} kg each\n"
          f"(cad/actuator/frameless.json).  {CAN_FK[0]:.0f} mm of can hung on a\n"
          f"{LG['links']['femur_beam_section'][1]:.0f} mm femur beam, and {M_FK:.2f} kg put at the knee —\n"
          f"neither is in any leg CAD.", color=WARN)
    frame(ax, f"GENERAL ARRANGEMENT — FRONT (looking along −x), sprawl stance\n{TITLE_NOTE}",
          (-1170, 1370), (-PLATE_H - 260, -PLATE_H + HUMAN_HEIGHT + 200))
    fig.tight_layout()
    p = os.path.join(OUT, "ga-front.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def view_top():
    fig, ax = plt.subplots(figsize=(14, 10.8))
    ax.add_patch(Rectangle((-SLAB_L / 2, -SLAB_W / 2), SLAB_L, SLAB_W, facecolor=FILL, edgecolor=INK, lw=1.3, zorder=1))
    # foot circle about the mid-left yaw axis, over the yaw range
    ax.add_patch(Arc((HIPX[1], HIPY), 2 * FOOT_RADIUS, 2 * FOOT_RADIUS, angle=0,
                     theta1=90 - YAW_RANGE_DEG, theta2=90 + YAW_RANGE_DEG, edgecolor=LEGC["tibia"], ls=":", lw=1.0))
    for i in range(3):
        for s in (1, -1):
            P = leg_world(i, s)
            draw_leg2(ax, [(p[0], p[1]) for p in P], lw=5)
            ax.add_patch(Circle((HIPX[i], s * HIPY), CAN_YAW[0] / 2, facecolor="#cfe9e6", edgecolor=ACTC, lw=1.1, zorder=2))
            ax.plot(HIPX[i], s * HIPY, "+", color=ACTC, ms=8, mew=1.2, zorder=3)
            for j in (1, 2):
                w, h = CAN_FK[1], CAN_FK[0]      # axis horizontal, in the leg plane
                ang = LEG_YAW[i] * (1 if s > 0 else -1)
                ax.add_patch(Rectangle((P[j][0] - w / 2, P[j][1] - h / 2), w, h, angle=ang,
                                       rotation_point="center", facecolor="#cfe9e6", edgecolor=ACTC,
                                       lw=1.0, zorder=2, alpha=0.9))
    for x, y in BATT_XY:
        ax.add_patch(Rectangle((x - BATTERY[0] / 2, y - BATTERY[1] / 2), BATTERY[0], BATTERY[1],
                               fill=False, edgecolor=HID, ls="--", lw=0.9, zorder=2))
        ax.text(x, y, f"battery\n{BATTERY[0]:.0f}×{BATTERY[1]:.0f}", fontsize=7, ha="center", va="center", color=NOTE)
    ax.add_patch(Rectangle((-COMPUTE[0] / 2, -COMPUTE[1] / 2), COMPUTE[0], COMPUTE[1],
                           fill=False, edgecolor=HID, ls="--", lw=0.9, zorder=2))
    ax.text(0, 0, f"electronics\n{COMPUTE[0]:.0f}×{COMPUTE[1]:.0f}", fontsize=7, ha="center", va="center", color=NOTE)
    # dimensions
    dim(ax, (-SLAB_L / 2, -SLAB_W / 2), (SLAB_L / 2, -SLAB_W / 2), -280, f"body length {SLAB_L:.0f}")
    dim(ax, (SLAB_L / 2, -SLAB_W / 2), (SLAB_L / 2, SLAB_W / 2), -120, f"slab width {SLAB_W:.0f}")
    dim(ax, (HIPX[2], -HIPY), (HIPX[2], HIPY), 250, f"yaw axis spacing {2*HIPY:.0f}")
    dim(ax, (HIPX[2], SLAB_W / 2), (HIPX[1], SLAB_W / 2), 260, f"{HIPX[1]-HIPX[2]:.0f}")
    dim(ax, (HIPX[1], SLAB_W / 2), (HIPX[0], SLAB_W / 2), 260, f"{HIPX[0]-HIPX[1]:.0f}")
    dim(ax, (FEET[:, 0].min(), FEET[:, 1].min()), (FEET[:, 0].max(), FEET[:, 1].min()), -190, f"stance length {STANCE_LENGTH:.0f}")
    dim(ax, (CAN_ENV[0][0], FEET[:, 1].min()), (CAN_ENV[1][0], FEET[:, 1].min()), -330,
        f"length over the cans {OVER_CANS_L:.0f}")
    dim(ax, (FEET[:, 0].max() + 90, FEET[:, 1].min()), (FEET[:, 0].max() + 90, FEET[:, 1].max()), -80, f"stance width {STANCE_WIDTH:.0f}")
    dim(ax, (FEET[:, 0].max() + 90, CAN_ENV[0][1]), (FEET[:, 0].max() + 90, CAN_ENV[1][1]), -230,
        f"width over the knee cans {OVER_CANS_W:.0f}")
    dim(ax, (HIPX[1], HIPY), (HIPX[1], HIPY + FOOT_RADIUS), -70, f"foot circle R{FOOT_RADIUS:.0f}")
    ax.annotate("", (HIPX[0], HIPY + 90), (HIPX[0] - 100, SLAB_W / 2 + 300),
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
    notes(ax, HIPX[0] - 250, SLAB_W / 2 + 400,
          f"front / rear leg planes yawed ±{LEG_YAW[0]:.0f}°; yaw window ±{YAW_RANGE_DEG:.0f}°.\n"
          f"The foot sweeps a R{FOOT_RADIUS:.0f} circle about each yaw axis;\nthe leg can reach R{REACH_MAX:.0f} at full stretch.")
    notes(ax, -920, SLAB_W / 2 + 430,
          f"Femur and knee cans Ø{CAN_FK[0]:.0f} × {mm(CAN_FK[1])} shown on their joints, edge-on\n"
          f"in this view; yaw cans Ø{CAN_YAW[0]:.0f} on the yaw axes.  The knee cans,\n"
          f"not the feet, are the widest and longest points of the machine:\n"
          f"{OVER_CANS_L:.0f} × {OVER_CANS_W:.0f} against {STANCE_LENGTH:.0f} × {STANCE_WIDTH:.0f} over the feet.", color=WARN)
    ax.text(-830, -640, "+x forward →", fontsize=8.5, ha="left", color=NOTE)
    frame(ax, f"GENERAL ARRANGEMENT — TOP, sprawl stance\n{TITLE_NOTE}",
          (-950, 950), (-780, 760))
    fig.tight_layout()
    p = os.path.join(OUT, "ga-top.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def view_body_plan():
    fig = plt.figure(figsize=(14, 11.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05), left=0.045, right=0.975, top=0.93,
                          bottom=0.045, hspace=0.20, wspace=0.16)

    # --- A: the slab in plan, with what actually lives in it -------------
    ax = fig.add_subplot(gs[0, :])
    ax.add_patch(Rectangle((-SLAB_L / 2, -SLAB_W / 2), SLAB_L, SLAB_W, facecolor=FILL, edgecolor=INK, lw=1.3))
    for i in range(3):
        for s in (1, -1):
            ax.add_patch(Circle((HIPX[i], s * HIPY), CAN_YAW[0] / 2, facecolor="#cfe9e6", edgecolor=ACTC, lw=1.2, zorder=2))
            ax.add_patch(Circle((HIPX[i], s * HIPY), CAN_FK[0] / 2, fill=False, edgecolor=WARN, ls=":", lw=0.9, zorder=2))
            ax.plot(HIPX[i], s * HIPY, "+", color=ACTC, ms=7, mew=1.1, zorder=3)
            ax.text(HIPX[i], s * HIPY, f"yaw\nØ{CAN_YAW[0]:.0f}", fontsize=7, ha="center", va="center", color="#0a6b62", zorder=4)
    for x, y in BATT_XY:
        ax.add_patch(Rectangle((x - BATTERY[0] / 2, y - BATTERY[1] / 2), BATTERY[0], BATTERY[1],
                               facecolor="#f6e6c8", edgecolor="#a9852f", lw=1.0, zorder=2))
        ax.text(x, y, f"battery\n{BATTERY[0]:.0f}×{BATTERY[1]:.0f}×{BATTERY[2]:.0f}\n{MASS.batteries/2:.1f} kg",
                fontsize=7.5, ha="center", va="center", color="#6b5417", zorder=3)
    ax.add_patch(Rectangle((-COMPUTE[0] / 2, -COMPUTE[1] / 2), COMPUTE[0], COMPUTE[1],
                           facecolor="#dfe8f6", edgecolor="#2a5d9e", lw=1.0, zorder=2))
    ax.text(0, 0, f"electronics\n{COMPUTE[0]:.0f}×{COMPUTE[1]:.0f}×{COMPUTE[2]:.0f}\n{MASS.electronics:.1f} kg",
            fontsize=7.5, ha="center", va="center", color="#1d3f6b", zorder=3)
    dim(ax, (-SLAB_L / 2, -SLAB_W / 2), (SLAB_L / 2, -SLAB_W / 2), -60, f"{SLAB_L:.0f}")
    dim(ax, (SLAB_L / 2, -SLAB_W / 2), (SLAB_L / 2, SLAB_W / 2), -90, f"{SLAB_W:.0f}")
    dim(ax, (HIPX[1] - CAN_YAW[0] / 2, HIPY + CAN_YAW[0] / 2), (HIPX[1] + CAN_YAW[0] / 2, HIPY + CAN_YAW[0] / 2),
        60, f"Ø{CAN_YAW[0]:.0f} yaw can", ext=False)
    ax.annotate("", (HIPX[0], HIPY + CAN_FK[0] / 2), (HIPX[0] - 40, SLAB_W / 2 + 60),
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
    notes(ax, HIPX[0] - 130, SLAB_W / 2 + 175,
          f"dotted: the Ø{CAN_FK[0]:.0f} femur/knee can for comparison.\n"
          f"The Ø{CAN_YAW[0]:.0f} yaw can clears the slab edge by {-YAW_OVERHANG:.0f} mm; if the yaw\n"
          f"shared the Ø{CAN_FK[0]:.0f} unit — which is what the closure's own mass\n"
          f"fixed point assumes — it would overhang by {YAW_OVERHANG_FK:.0f} mm a side.", color=WARN, fs=8.2)
    ax.text(0, -SLAB_W / 2 - 150,
            f"slab planform {A_SLAB/1e6:.3f} m²   −  six yaw cans {A_YAW_ALL/1e6:.3f}  −  battery + electronics boxes {A_BOX/1e6:.3f} "
            f"  =  {A_FREE/1e6:.3f} m² free ({100*A_FREE/A_SLAB:.0f} % of the slab)",
            fontsize=9.5, color=INK, ha="center")
    frame(ax, "A — the 900 mm slab in plan: what fits without argument (mm)", (-580, 580), (-330, 390))

    # --- B: transverse section at a hip station ---------------------------
    ax = fig.add_subplot(gs[1, 0])
    ground(ax, -700, 700)
    ax.add_patch(Rectangle((-SLAB_W / 2, 0), SLAB_W, SLAB_H, facecolor=FILL, edgecolor=INK, lw=1.3))
    for z0 in (0.0, SLAB_H - PLATE_T):
        ax.add_patch(Rectangle((-SLAB_W / 2, z0), SLAB_W, PLATE_T, facecolor="#b9bcbf", edgecolor="none"))
    ax.add_patch(Rectangle((-BATTERY[0] / 2, PLATE_T), BATTERY[0], BATTERY[2], facecolor="#f6e6c8", edgecolor="#a9852f", lw=1.0))
    ax.text(0, PLATE_T + BATTERY[2] / 2, f"battery\n{BATTERY[2]:.0f} tall", fontsize=7.5, ha="center", va="center", color="#6b5417")
    for s in (1, -1):
        P = leg_world(1, s)
        can_rect(ax, s * HIPY, -CAN_YAW[1] / 2 - 4, CAN_YAW[0], CAN_YAW[1], axis="v")
        ax.add_patch(Rectangle((s * HIPY - 55, POD_BOT), 110, -POD_BOT, fill=False, edgecolor=HID, ls="--", lw=0.8))
        draw_leg2(ax, [(p[1], p[2]) for p in P], lw=4)
        ax.add_patch(Circle((P[1][1], P[1][2]), CAN_FK[0] / 2, facecolor="#cfe9e6", edgecolor=ACTC, lw=1.1, zorder=2))
        ax.text(P[1][1], P[1][2] - CAN_FK[0] / 2 - 30, f"femur unit Ø{CAN_FK[0]:.0f}", fontsize=7.5,
                ha="center", va="center", color="#0a6b62", zorder=3)
    dim(ax, (-SLAB_W / 2 - 20, 0), (-SLAB_W / 2 - 20, SLAB_H), 80, f"slab {SLAB_H:.0f}")
    ax.text(0, -255, f"yaw can {mm(CAN_YAW[1])} deep, in the {-POD_BOT:.0f} mm hip pod\n"
                    f"(coxa plate reaches {-COXA_PLATE_BOT:.0f} below the plate)",
            fontsize=7.8, ha="center", va="top", color=DIM)
    dim(ax, (-SLAB_W / 2, SLAB_H + 20), (SLAB_W / 2, SLAB_H + 20), 60, f"{SLAB_W:.0f}")
    notes(ax, -690, SLAB_H + 330,
          f"three Ø{CAN_FK[0]:.0f} × {mm(CAN_FK[1])} cans stacked coaxially on a yaw axis would be\n"
          f"{STACK_H:.0f} mm tall inside the {SLAB_H:.0f} mm slab — height is not the problem.\n"
          f"The problem is that with the capstan deleted there is no drive\nfrom the yaw axis out to the femur pivot.",
          color=WARN, fs=8.2)
    frame(ax, "B — section at the mid hip station (mm)", (-700, 700), (-PLATE_H - 40, SLAB_H + 400))

    # --- C: the packing test ---------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.add_patch(Rectangle((-SLAB_L / 2, -SLAB_W / 2), SLAB_L, SLAB_W, facecolor="#f4f4f4", edgecolor=INK, lw=1.2))
    for i in range(3):
        for s in (1, -1):
            ax.add_patch(Circle((HIPX[i], s * HIPY), CAN_YAW[0] / 2, facecolor="#cfe9e6", edgecolor=ACTC, lw=1.0))
    for x, y in BATT_XY:
        ax.add_patch(Rectangle((x - BATTERY[0] / 2, y - BATTERY[1] / 2), BATTERY[0], BATTERY[1],
                               facecolor="#f6e6c8", edgecolor="#a9852f", lw=0.9))
    ax.add_patch(Rectangle((-COMPUTE[0] / 2, -COMPUTE[1] / 2), COMPUTE[0], COMPUTE[1],
                           facecolor="#dfe8f6", edgecolor="#2a5d9e", lw=0.9))
    for k, (x, y) in enumerate(PACKED_BARE):
        ax.add_patch(Circle((x, y), CAN_FK[0] / 2, facecolor="none", edgecolor="#8e5fa8", ls="--", lw=1.2))
    ax.text(0, SLAB_W / 2 + 30, f"dashed: the {len(PACKED_BARE)} that would fit if the batteries and electronics came out",
            fontsize=8.2, ha="center", va="bottom", color="#8e5fa8")
    for k, (x, y) in enumerate(PACKED):
        ax.add_patch(Circle((x, y), CAN_FK[0] / 2, facecolor="#f6d9d4", edgecolor=WARN, lw=1.1))
        ax.text(x, y, f"{k+1}", fontsize=8, ha="center", va="center", color=WARN)
    ax.text(0, -SLAB_W / 2 - 60,
            f"{len(PACKED)} of 12 placed with the batteries and electronics in.\n"
            f"With those boxes taken out entirely the slab still takes only {len(PACKED_BARE)}.\n"
            f"Twelve cans need {A_NEED_12/1e6:.3f} m² against {A_FREE/1e6:.3f} m² free — {AREA_SHORT:.1f}× short.\n"
            f"Stacking helps the height, not the plan: two levels of {len(PACKED_BARE)} is still {12 - 2*len(PACKED_BARE)} outside.",
            fontsize=8.8, ha="center", va="top", color=INK)
    ax.text(0, SLAB_W / 2 + 100, "THE TWELVE FEMUR/KNEE UNITS DO NOT FIT INBOARD",
            fontsize=11.5, ha="center", color=WARN, fontweight="bold")
    frame(ax, "C — packing test: twelve Ø172 cans inboard in the slab (mm)", (-580, 580), (-450, 340))

    fig.suptitle(f"BODY PLAN — where the eighteen actuator units go.  Cans Ø{CAN_FK[0]:.0f} × {mm(CAN_FK[1])} "
                 f"(femur/knee, {M_FK:.2f} kg) and Ø{CAN_YAW[0]:.0f} × {mm(CAN_YAW[1])} (yaw, {M_YAW:.2f} kg), "
                 f"true size, from cad/actuator/frameless{{,-yaw}}.json", fontsize=11, y=0.985)
    p = os.path.join(OUT, "ga-body-plan.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def view_leg_envelope():
    fig = plt.figure(figsize=(14, 9.4))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.42), left=0.05, right=0.985, top=0.90, bottom=0.06, wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    # reachable region, rasterised from the joint windows
    hb = 6.0
    xe = np.arange(min(WR.min(), -60) - 40, WR.max() + 40 + hb, hb)
    ze = np.arange(WZ.min() - 40, max(WZ.max(), 60) + 40 + hb, hb)
    Hh, _, _ = np.histogram2d(WR, WZ, bins=(xe, ze))
    ax.imshow((Hh.T > 0).astype(float), origin="lower", extent=(xe[0], xe[-1], ze[0], ze[-1]),
              cmap=matplotlib.colors.ListedColormap([(1, 1, 1, 0), (0.16, 0.61, 0.84, 0.22)]),
              interpolation="nearest", aspect="equal", zorder=0)
    ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ze[:-1] + ze[1:]), (Hh.T > 0).astype(float),
               levels=[0.5], colors=["#2980b9"], linewidths=1.2, zorder=1)
    ground(ax, xe[0], xe[-1], z=GROUND_Z)
    # body edge and floor plate (the slab runs from the far side of the machine out to EDGE)
    ax.add_patch(Rectangle((xe[0], 0), -xe[0] + EDGE, SLAB_H, facecolor=FILL, edgecolor=INK, lw=1.1, zorder=2))
    ax.text(xe[0] + 90, SLAB_H / 2, "body slab", fontsize=8.5, ha="left", va="center", color=NOTE, zorder=3)
    # poses
    for pose, col, lab, lw, dxy, ha in (((FEMUR_DEG, TAU_DEG), INK, "neutral stance", 4.5, (0, -46), "center"),
                                        (POSE_FOLD, "#8e44ad", f"fully folded\nfemur {POSE_FOLD[0]:.0f}°, knee {KNEE_LIM[0]:.0f}°",
                                         2.6, (-14, -70), "right"),
                                        (POSE_EXT, "#e67e22", f"fully extended\nfemur {POSE_EXT[0]:.0f}°, τ {POSE_EXT[1]:.0f}°",
                                         2.6, (16, 10), "left")):
        P = leg_plane(*pose)
        ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=lw, solid_capstyle="round", zorder=4,
                alpha=1.0 if col == INK else 0.9, label=None)
        ax.plot(P[:, 0], P[:, 1], "o", color=col, ms=4, zorder=5)
        ax.text(P[3][0] + dxy[0], P[3][1] + dxy[1], lab, fontsize=8.5, color=col, ha=ha, va="top", zorder=5)
    P = leg_plane()
    for (p, q), name in zip(zip(P[:-1], P[1:]), ("coxa", "femur", "tibia")):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=LEGC[name], lw=6, solid_capstyle="round", zorder=4)
    ax.plot(0, 0, "+", color=ACTC, ms=11, mew=1.4, zorder=6)
    ax.text(6, 26, "yaw axis", fontsize=8, color="#0a6b62")
    # the cans the sweep does not know about
    for p in (P[1], P[2]):
        ax.add_patch(Circle((p[0], p[1]), CAN_FK[0] / 2, fill=False, edgecolor=WARN, ls="--", lw=1.1, zorder=5))
    ax.annotate(f"Ø{CAN_FK[0]:.0f} femur and knee cans — {COXA_DROP - CAN_FK[0]/2:.0f} mm\nto the floor plate, and not in the sweep",
                (P[1][0], P[1][1] + CAN_FK[0] / 2), (P[1][0] + 30, SLAB_H + 40), fontsize=8.2, color=WARN,
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.6), zorder=6)
    # dimensions
    dim(ax, (0, WZ.min() - 80), (REACH_MAX, WZ.min() - 80), 0, f"max reach {REACH_MAX:.0f} (at any height)")
    dim(ax, (REACH_GROUND[0], GROUND_Z), (REACH_GROUND[1], GROUND_Z), -(DROP_MAX + 190),
        f"reach along the ground line {REACH_GROUND[0]:.0f} → {REACH_GROUND[1]:.0f}")
    dim(ax, (REACH_MAX + 90, GROUND_Z), (REACH_MAX + 90, GROUND_Z + STEP_UP), -50, f"step-up {STEP_UP:.0f}")
    dim(ax, (REACH_MAX + 90, GROUND_Z), (REACH_MAX + 90, WZ.min()), 50, f"reach below ground {DROP_MAX:.0f}")
    dim(ax, (-140, GROUND_Z), (-140, 0), 0, f"floor plate {PLATE_H:.0f}")
    ax.text(-25, -COXA_DROP / 2, f"coxa drop {COXA_DROP:.0f}", fontsize=8, color=DIM, va="center", ha="right")
    dim(ax, (FOLD[:, 0].min(), FOLD[:, 1].min() - 40), (FOLD[:, 0].max(), FOLD[:, 1].min() - 40), 0,
        f"folded envelope {FOLD_R:.0f} wide × {FOLD_Z:.0f} tall")
    ax.axhline(GROUND_Z, color=GREY, lw=1.0, zorder=0)
    ax.text(xe[-1] - 20, GROUND_Z + 14, "ground", fontsize=8, color=NOTE, ha="right")
    frame(ax, "LEG WORKSPACE IN ITS OWN PLANE — foot positions over the femur, knee and tibia windows (mm)",
          (xe[0] - 30, xe[-1] + 130), (ze[0] - 300, max(ze[-1], SLAB_H) + 90))

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    rows = [
        ("SOURCE", "cad/leg/leg.json (round-14 leg CAD)"),
        ("links coxa / femur / tibia", f"{COXA:.0f} / {FEMUR:.0f} / {TIBIA:.0f} mm"),
        ("femur window φ", f"{FEM_RANGE[0]:+.0f}° … {FEM_RANGE[1]:+.0f}°"),
        ("knee window θ", f"{KNEE_LIM[0]:.0f}° … {KNEE_LIM[1]:.0f}°"),
        ("tibia absolute window τ", f"{TAU_RANGE[0]:.0f}° … {TAU_RANGE[1]:.0f}°"),
        ("neutral stance", f"φ {FEMUR_DEG:.0f}°, θ {KNEE_DEG:.0f}°, τ {TAU_DEG:.0f}°"),
        ("", ""),
        ("REACH", ""),
        ("neutral foot radius", f"{FOOT_RADIUS:.0f} mm from the yaw axis"),
        ("max reach, any height", f"{REACH_MAX:.0f} mm  (φ {POSE_EXT[0]:.0f}°, τ {POSE_EXT[1]:.0f}°)"),
        ("reach on the ground line", f"{REACH_GROUND[0]:.0f} → {REACH_GROUND[1]:.0f} mm"),
        ("usable stride at that height", f"{REACH_GROUND[1]-REACH_GROUND[0]:.0f} mm radially"),
        ("step-up above the ground line", f"{STEP_UP:.0f} mm"),
        ("reach below the ground line", f"{DROP_MAX:.0f} mm"),
        ("", ""),
        ("FOLD", ""),
        ("folded pose", f"φ {POSE_FOLD[0]:.1f}°, θ {KNEE_LIM[0]:.0f}°"),
        ("folded foot radius", f"{FOLD_FOOT_R:.0f} mm"),
        ("folded envelope", f"{FOLD_R:.0f} × {FOLD_Z:.0f} mm"),
        ("", ""),
        ("WHAT LIMITS THE FOLD", ""),
    ]
    y = 0.985
    for k, v in rows:
        if k in ("SOURCE", "REACH", "FOLD", "WHAT LIMITS THE FOLD"):
            ax.text(0.0, y, k, fontsize=9, fontweight="bold", color=INK, va="top")
        elif k:
            ax.text(0.0, y, k, fontsize=8.5, color=NOTE, va="top")
            ax.text(1.0, y, v, fontsize=8.5, color=INK, va="top", ha="right")
        y -= 0.0355 if k else 0.018
    ax.text(0.0, y + 0.005,
            f"The femur window asks for {FEM_RANGE[1]:.0f}° up. The round-14 ROM sweep\n"
            f"(cad/leg/leg_rom.py) finds the femur root hits the body's floor\n"
            f"plate first: {FEM_UP_CLEAR['+0']:.0f}° at yaw 0°, {FEM_UP_CLEAR['+45']:.0f}° at ±45°, {FEM_UP_CLEAR['+90']:.0f}° at ±90°.\n"
            f"The folded pose above uses {FEM_UP_0:.1f}°, the yaw-0 limit — so the\n"
            f"leg cannot fold as tight as the joint window claims, and it\n"
            f"folds worse the more it is yawed.\n\n"
            f"The Ø{CAN_FK[0]:.0f} femur and knee cans are NOT in this sweep: it was\n"
            f"run on the round-14 leg, whose motors were in the body. A can\n"
            f"of radius {CAN_FK[0]/2:.0f} mm on the femur pivot ({COXA_DROP:.0f} mm below the plate)\n"
            f"leaves {COXA_DROP - CAN_FK[0]/2:.0f} mm to the floor plate at neutral and will\n"
            f"re-limit the femur. That check has not been run.",
            fontsize=8.2, color=WARN, va="top")
    fig.suptitle("LEG ENVELOPE — one leg, its own plane, from the CAD joint windows", fontsize=11.5, y=0.965)
    p = os.path.join(OUT, "ga-leg-envelope.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def view_scale():
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.34), left=0.03, right=0.985, top=0.90, bottom=0.05, wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    g = 0.0
    ax.plot([-800, 3050], [g, g], color=GREY, lw=1.5)
    ax.add_patch(Rectangle((-800, g - 40), 3850, 40, facecolor="#ececec", edgecolor="none"))

    # the robot, side view, feet on the ground
    ox, oz = 0.0, PLATE_H
    ax.add_patch(Rectangle((ox - SLAB_L / 2, oz), SLAB_L, SLAB_H, facecolor=FILL, edgecolor=INK, lw=1.2, zorder=2))
    for i in range(3):
        for s in (1, -1):
            P = leg_world(i, s)
            draw_leg2(ax, [(p[0] + ox, p[2] + oz) for p in P], lw=4.5 if s == 1 else 2.0,
                      alpha=1.0 if s == 1 else 0.4)
        can_rect(ax, ox + HIPX[i], oz - CAN_YAW[1] / 2 - 4, CAN_YAW[0], CAN_YAW[1], axis="v")
        P = leg_world(i, 1)
        can_rect(ax, ox + P[1][0], oz + P[1][2], CAN_FK[0], CAN_FK[1], axis="h")
        can_rect(ax, ox + P[2][0], oz + P[2][2], CAN_FK[0], CAN_FK[1], axis="h")
    ax.text(ox, oz + SLAB_H + 300, f"the robot — {STANCE_LENGTH:.0f} long over the feet,\n{DECK:.0f} to the deck, {M_ROBOT_CLOSURE:.0f}–{M_ROBOT_CAD:.0f} kg",
            fontsize=10, ha="center", color=INK)
    dim(ax, (ox - SLAB_L / 2, g - 150), (ox + SLAB_L / 2, g - 150), 0, f"body {SLAB_L:.0f}")
    dim(ax, (ox + OVERALL_L / 2 + 70, g), (ox + OVERALL_L / 2 + 70, oz + SLAB_H), 0, f"{DECK:.0f}")

    # a large dog
    DOG_H = 800.0
    dx = 1400.0
    dog(ax, dx, g, DOG_H)
    ax.text(dx, DOG_H * 1.30 + 90, f"a large dog — {DOG_H:.0f} mm at the withers\n(assumed scale reference, not a model number)",
            fontsize=8.5, ha="center", color=NOTE)
    dim(ax, (dx - 700, g), (dx - 700, g + DOG_H), 0, f"{DOG_H:.0f}")

    # a person
    px = 2500.0
    human(ax, px, g)
    ax.text(px, HUMAN_HEIGHT + 110, f"{HUMAN_HEIGHT/1000:.2f} m person", fontsize=8.5, ha="center", color=NOTE)
    dim(ax, (px + 240, g), (px + 240, g + HUMAN_HEIGHT), 0, f"{HUMAN_HEIGHT:.0f}")

    # a 1 m rule
    rx = 3020.0
    ax.add_patch(Rectangle((rx, g), 60, 1000, facecolor="#fdf3d3", edgecolor=INK, lw=1.0))
    for t in range(0, 1001, 100):
        ax.plot([rx, rx + (60 if t % 500 == 0 else 34)], [g + t, g + t], color=INK, lw=0.8)
    ax.text(rx + 30, 1060, "1 m\nrule", fontsize=8.5, va="bottom", ha="center", color=NOTE)

    frame(ax, "SCALE — the robot against a large dog, a person and a 1 m rule, all to scale (mm)",
          (-830, 3260), (-330, 2180))

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    rows = [
        ("HEADLINE NUMBERS", ""),
        ("length × width over the feet", f"{STANCE_LENGTH:.0f} × {STANCE_WIDTH:.0f} mm"),
        ("length × width over the cans", f"{OVER_CANS_L:.0f} × {OVER_CANS_W:.0f} mm"),
        ("body slab", f"{SLAB_L:.0f} × {SLAB_W:.0f} × {SLAB_H:.0f} mm"),
        ("deck height", f"{DECK:.0f} mm"),
        ("hip (femur axis) height", f"{HIP_H:.0f} mm"),
        ("ground clearance", f"{CLEARANCE:.0f} mm"),
        ("", ""),
        ("MASS", ""),
        ("femur / knee unit ×12", f"{M_FK:.2f} kg each"),
        ("yaw unit ×6", f"{M_YAW:.2f} kg each"),
        ("all eighteen units", f"{M_ACT:.1f} kg"),
        ("closure model (14b)", f"{M_ROBOT_CLOSURE:.1f} kg"),
        ("with the leg CAD's structure", f"~{M_ROBOT_CAD:.0f} kg (estimate)"),
        ("", ""),
        ("PAYLOAD", ""),
        ("mission payload", f"{MASS.mission_payload:.0f} kg (trash + tool)"),
        ("rider, stretch goal", f"{MASS.rider:.0f} kg"),
        ("", ""),
        ("ENERGY", ""),
        ("walking power at 1 m/s", f"{WALK_W:.0f} W at {M_ROBOT_CLOSURE:.0f} kg"),
        ("pack for a {:.0f} h mission".format(ENERGY.endurance_h), f"{PACK_WH:.0f} Wh, {PACK_KG:.1f} kg × {ENERGY.packs}"),
        ("the same at ~{:.0f} kg".format(M_ROBOT_CAD), f"{WALK_W_CAD:.0f} W, {PACK_KG_CAD:.1f} kg × {ENERGY.packs}"),
    ]
    y = 0.98
    for k, v in rows:
        if k in ("HEADLINE NUMBERS", "MASS", "PAYLOAD", "ENERGY"):
            ax.text(0.0, y, k, fontsize=9.5, fontweight="bold", color=INK, va="top")
        elif k:
            ax.text(0.0, y, k, fontsize=8.5, color=NOTE, va="top")
            ax.text(1.0, y, v, fontsize=8.5, color=INK, va="top", ha="right")
        y -= 0.040 if k else 0.020
    ax.text(0.0, y - 0.01,
            f"The two mass lines disagree by {M_ROBOT_CAD - M_ROBOT_CLOSURE:.0f} kg, from two corrections\n"
            f"the closure has not taken: the unit is {M_FK:.2f} kg in the CAD,\n"
            f"not the {M_FK_917:.2f} kg §9.17 costed ({M_ACT - M_ACT_917:+.1f} kg over eighteen),\n"
            f"and leg structure is {LEG_STRUCT_EST:.1f} kg a leg in the leg CAD with\n"
            f"the capstan already deleted, not the {MASS.legs/6:.1f} kg budgeted.\n"
            f"At {M_ROBOT_CAD:.0f} kg the margins go from {MARGIN_CLOSURE['femur']:.2f} / {MARGIN_CLOSURE['knee']:.2f} / {FM['yaw']['margin']:.2f} to\n"
            f"femur {MARGIN_AT['femur']:.2f}, knee {MARGIN_AT['knee']:.2f}, yaw {MARGIN_AT['yaw']:.2f}: neither pitch joint closes.",
            fontsize=8.2, color=WARN, va="top")
    fig.suptitle("HOW BIG IS IT", fontsize=12, y=0.965)
    p = os.path.join(OUT, "ga-scale.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


# ===========================================================================
# numbers file and design note
# ===========================================================================
def write_json():
    d = {
        "generated_by": "analysis/arrangement.py",
        "units": "mm, kg, deg unless stated",
        "sources": {
            "actuator_can": CAN_SOURCE,
            "actuator_electrical": "hw/stator/frameless_motor.json (round 14b, analysis/frameless_motor.py)",
            "leg": "cad/leg/leg.json (round 14 leg CAD)",
            "body_and_mass": "analysis/hexapod_model.py",
            "battery_electronics_envelopes": "concepts/hexapod_skeleton.py",
        },
        "actuator": {
            "femur_knee": {"motor_od": MOT_FK["od_mm"], "motor_len": MOT_FK["len_mm"],
                           "can_od": CAN_FK[0], "can_height": CAN_FK[1], "mass_kg": M_FK,
                           "torque_joint_Nm": FM["pick"]["T_joint_fk"], "ratio": FM["pick"]["ratio_fk"]},
            "yaw": {"motor_od": MOT_YAW["od_mm"], "motor_len": MOT_YAW["len_mm"],
                    "can_od": CAN_YAW[0], "can_height": CAN_YAW[1], "mass_kg": M_YAW,
                    "torque_joint_Nm": T_YAW_SIZED, "ratio": FM["pick"]["ratio_yaw"]},
            "as_quoted_in_08_9_17": {"can_od": CAN_FK_917[0], "can_height": CAN_FK_917[1], "mass_kg": M_FK_917,
                                     "yaw_can_od": CAN_YAW_917[0], "yaw_can_height": CAN_YAW_917[1], "yaw_mass_kg": M_YAW_917,
                                     "delta_height_mm": CAN_FK[1] - CAN_FK_917[1], "delta_mass_kg": M_FK - M_FK_917,
                                     "note": "the section sketch's motor + 6 mm of case each way; it omits the reducer's "
                                             "own axial stack, which the CAD includes"},
            "housing_allowance_mm": {"radial": HOUSING_R, "axial_each_end": HOUSING_Z,
                                     "note": "from the half-section in analysis/frameless_motor.py; the same file's "
                                             f"sweep comment assumes a Ø{CAN_OD_ALT:.0f} can, {CAN_OD_ALT-CAN_FK[0]:.0f} mm larger"},
        },
        "body": {"slab_length": SLAB_L, "slab_width": SLAB_W, "slab_height": SLAB_H,
                 "yaw_axis_spacing": 2 * HIPY, "hip_x": list(HIPX),
                 "slab_edge_from_yaw_axis": EDGE, "floor_plate_thickness": PLATE_T,
                 "hip_pod_depth": -POD_BOT, "coxa_plate_bottom": COXA_PLATE_BOT},
        "stance": {"name": "sprawl, tibia vertical", "femur_deg": FEMUR_DEG, "knee_deg": KNEE_DEG,
                   "tau_deg": TAU_DEG, "leg_plane_yaw_deg": list(LEG_YAW), "yaw_window_deg": YAW_RANGE_DEG},
        "overall": {"length_over_feet": STANCE_LENGTH, "width_over_feet": STANCE_WIDTH,
                    "length_over_cans": OVER_CANS_L, "width_over_cans": OVER_CANS_W,
                    "length_overall": OVERALL_L, "width_overall": OVERALL_W,
                    "widest_point_is_the_knee_cans": bool(WIDEST_IS_CANS),
                    "longest_point_is_the_knee_cans": bool(LONGEST_IS_CANS),
                    "height_to_deck": DECK, "height_over_knee_cans": KNEE_TOP, "height_overall": OVERALL_H,
                    "hip_height": HIP_H, "floor_plate_height": PLATE_H, "ground_clearance": CLEARANCE,
                    "stance_width": STANCE_WIDTH, "stance_length": STANCE_LENGTH,
                    "foot_circle_radius": FOOT_RADIUS, "human_reference_height": HUMAN_HEIGHT},
        "leg_envelope": {"links": {"coxa": COXA, "femur": FEMUR, "tibia": TIBIA},
                         "coxa_drop": COXA_DROP,
                         "femur_window_deg": list(FEM_RANGE), "knee_window_deg": list(KNEE_LIM),
                         "tau_window_deg": list(TAU_RANGE),
                         "femur_up_limit_by_yaw_deg": FEM_UP_CLEAR,
                         "neutral_foot_radius": FOOT_RADIUS,
                         "max_reach": REACH_MAX, "max_reach_pose_deg": {"femur": POSE_EXT[0], "tau": POSE_EXT[1]},
                         "reach_on_ground_line": list(REACH_GROUND),
                         "step_up_above_ground": STEP_UP, "reach_below_ground": DROP_MAX,
                         "folded_pose_deg": {"femur": POSE_FOLD[0], "knee": KNEE_LIM[0], "tau": POSE_FOLD[1]},
                         "folded_foot_radius": FOLD_FOOT_R, "folded_envelope": [FOLD_R, FOLD_Z]},
        "packing": {"slab_area_mm2": A_SLAB, "yaw_cans_area_mm2": A_YAW_ALL,
                    "battery_electronics_area_mm2": A_BOX, "free_area_mm2": A_FREE,
                    "free_fraction": A_FREE / A_SLAB,
                    "twelve_fk_cans_area_mm2": A_NEED_12, "area_shortfall_factor": AREA_SHORT,
                    "fk_cans_placed_one_level": len(PACKED), "fk_cans_wanted": 12,
                    "fits_one_level": len(PACKED) >= 12,
                    "yaw_can_edge_margin": -YAW_OVERHANG,
                    "yaw_can_overhang_if_shared_fk_unit": YAW_OVERHANG_FK,
                    "three_can_coaxial_stack_height": STACK_H,
                    "clearance_between_cans": CLEAR},
        "mass": {"unit_femur_knee": M_FK, "unit_yaw": M_YAW, "all_units": M_ACT,
                 "unit_femur_knee_in_9_17": M_FK_917, "unit_yaw_in_9_17": M_YAW_917, "all_units_in_9_17": M_ACT_917,
                 "fixed_non_actuator": M_FIXED,
                 "robot_closure_14b": M_ROBOT_CLOSURE,
                 "robot_with_sized_yaw": M_ROBOT_SIZED_YAW,
                 "robot_with_leg_cad_structure_estimate": M_ROBOT_CAD,
                 "leg_structure_cad": LEG_STRUCT_CAD, "leg_capstan_transmission_deleted": LEG_TRANS_CAD,
                 "leg_structure_estimate_after_delete": LEG_STRUCT_EST,
                 "leg_total_round14_cad": LG["leg_total_g"] / 1000.0,
                 "budget_leg_structure_each": MASS.legs / 6,
                 "mission_payload": MASS.mission_payload, "rider_stretch": MASS.rider,
                 "margins_at_closure_mass": MARGIN_CLOSURE,
                 "margins_at_leg_cad_mass": MARGIN_AT},
        "energy": {"walking_power_W_at_1ms": WALK_W, "pack_wh": PACK_WH, "pack_kg": PACK_KG,
                   "walking_power_W_at_leg_cad_mass": WALK_W_CAD, "pack_wh_at_leg_cad_mass": PACK_WH_CAD,
                   "pack_kg_at_leg_cad_mass": PACK_KG_CAD,
                   "packs": ENERGY.packs, "endurance_h": ENERGY.endurance_h,
                   "battery_envelope_mm": list(BATTERY), "electronics_envelope_mm": list(COMPUTE)},
        "gaps": [
            f"The built unit is Ø{CAN_FK[0]:.0f} × {mm(CAN_FK[1])} and {M_FK:.2f} kg against the Ø{CAN_FK_917[0]:.0f} × {mm(CAN_FK_917[1])}, {M_FK_917:.2f} kg that "
            f"08 §9.17 costed: {CAN_FK[1]-CAN_FK_917[1]:.1f} mm taller and {M_FK-M_FK_917:.2f} kg heavier a unit, {M_ACT-M_ACT_917:.1f} kg over eighteen.",
            f"Twelve Ø{CAN_FK[0]:.0f} femur/knee cans do not fit inboard in the {SLAB_L:.0f} × {SLAB_W:.0f} slab on one "
            f"level: {len(PACKED)} placed of 12, and the area alone is short by {AREA_SHORT:.1f}×.",
            "Round 14b deletes the capstan, so the femur and knee reductions have to sit on their own joints; "
            "the drawings show them there, and no CAD exists for that.",
            f"A Ø{CAN_FK[0]:.0f} can on the femur pivot leaves {COXA_DROP - CAN_FK[0]/2:.0f} mm to the floor plate and is not in the "
            "round-14 ROM sweep, which already limits the femur to "
            f"{FEM_UP_0:.0f}° at yaw 0 against the {FEM_RANGE[1]:.0f}° window.",
            f"Mass: the 14b closure stands at {M_ROBOT_CLOSURE:.1f} kg on {M_FK_917:.2f} kg units and {MASS.legs/6:.1f} kg of leg structure per leg; "
            f"the CAD says {M_FK:.2f} kg and {LEG_STRUCT_EST:.1f} kg, which puts the robot near {M_ROBOT_CAD:.0f} kg and the joint margins at "
            f"femur {MARGIN_AT['femur']:.2f} / knee {MARGIN_AT['knee']:.2f} / yaw {MARGIN_AT['yaw']:.2f} — it does not close.",
            f"The can diameter itself is soft: Ø{CAN_FK[0]:.0f} from the half-section, Ø{CAN_OD_ALT:.0f} from the sweep's own comment.",
        ],
    }
    with open(JSN, "w") as fh:
        json.dump(d, fh, indent=1)
    return d


def write_doc(figs):
    rel = lambda p: os.path.relpath(p, os.path.dirname(DOC))
    fits = "do NOT fit" if len(PACKED) < 12 else "fit"
    doc = f"""# 10 — General arrangement

*Generated by [`analysis/arrangement.py`](../../analysis/arrangement.py) from
the models. Numbers in [`hw/arrangement.json`](../../hw/arrangement.json).
All dimensions mm unless stated.*

The actuator drawn here is the built unit from
[`cad/actuator/frameless.json`](../../cad/actuator/frameless.json):
**Ø{CAN_FK[0]:.0f} × {mm(CAN_FK[1])} mm, {M_FK:.2f} kg** for a femur or knee, Ø{CAN_YAW[0]:.0f} × {mm(CAN_YAW[1])} mm,
{M_YAW:.2f} kg for a yaw. [08 §9.17](08-actuator-design.md) quotes Ø{CAN_FK_917[0]:.0f} × {mm(CAN_FK_917[1])} mm and
{M_FK_917:.2f} kg for the same unit — the same diameter, **{CAN_FK[1]-CAN_FK_917[1]:.1f} mm taller and
{M_FK-M_FK_917:.2f} kg heavier**, because §9.17's height and mass left out the reducer's
own axial stack. Every number below uses the CAD.

## 0. What the reviewer needs to know first

1. **The twelve femur/knee units {fits} inboard in the body.** Ø{CAN_FK[0]:.0f} cans need
   {A_NEED_12/1e6:.3f} m² of planform against the {A_FREE/1e6:.3f} m² the {SLAB_L:.0f} × {SLAB_W:.0f} mm slab has
   left once the six yaw cans, the two batteries and the electronics box are
   in — short by **{AREA_SHORT:.1f}×**. A greedy pack places **{len(PACKED)} of 12** on one level.
   Take the batteries and the electronics out of the body entirely and the
   slab still holds only **{len(PACKED_BARE)}**. Stacking does not rescue it: two levels of
   {len(PACKED_BARE)} is {12 - 2*len(PACKED_BARE)} units still outside. The constraint is planform, not height.
2. **They should not be in the body anyway.** Round 14b
   ([08 §9.17](08-actuator-design.md)) deletes the capstan because the rope
   drive cannot be wound, and the capstan was the only thing carrying the
   femur and knee drives from the body out to the joints. The drawings
   therefore put the femur and knee cans **on their own joints**, which is
   where a cycloid-inside-the-motor unit has to be. **No CAD exists for that
   leg**; [09](09-leg-assembly.md) is the body-mounted, capstan-driven leg.
3. **A Ø{CAN_FK[0]:.0f} can on the femur pivot leaves {COXA_DROP - CAN_FK[0]/2:.0f} mm to the floor plate** and is not in
   the round-14 range-of-motion sweep. That sweep already limits the femur to
   {FEM_UP_0:.0f}° at yaw 0 and {FEM_UP_CLEAR['+90']:.0f}° at yaw ±90 against an {FEM_RANGE[1]:.0f}° window; the can will
   make it worse, and the check has not been run.
4. **Mass, twice over.** The 14b closure stands at {M_ROBOT_CLOSURE:.1f} kg on two numbers the
   CAD has since contradicted: {M_FK_917:.2f} kg a unit (the CAD says {M_FK:.2f}) and {MASS.legs/6:.1f} kg of
   leg structure per leg (the leg CAD says {LEG_STRUCT_CAD:.1f}, {LEG_STRUCT_EST:.1f} once the deleted
   capstan comes out). Eighteen units are **{M_ACT:.1f} kg**, not {M_ACT_917:.1f}. Put both
   corrections in and the robot is near **{M_ROBOT_CAD:.0f} kg**, and the joint margins go
   from femur {MARGIN_CLOSURE['femur']:.2f} / knee {MARGIN_CLOSURE['knee']:.2f} / yaw {FM['yaw']['margin']:.2f} to **femur {MARGIN_AT['femur']:.2f}, knee {MARGIN_AT['knee']:.2f},
   yaw {MARGIN_AT['yaw']:.2f}**. **The design as drawn does not close.** That is arithmetic on
   `frameless_motor.json`'s own torque coefficients, not a re-solved fixed
   point — the study has to be re-run.
5. **The widest and longest points of the machine are the knee cans, not the
   feet** — {OVER_CANS_L:.0f} × {OVER_CANS_W:.0f} mm against {STANCE_LENGTH:.0f} × {STANCE_WIDTH:.0f} over the feet. A Ø{CAN_FK[0]:.0f} disc on
   a knee reaches {CAN_FK[0]/2:.0f} mm outboard of the foot under it, so the machine's
   footprint, its door width and its leg-to-leg collision limits are all set
   by an actuator housing rather than by the stance.

## 1. The machine, dimensioned

![side]({rel(figs['side'])})

![front]({rel(figs['front'])})

![top]({rel(figs['top'])})

A {STANCE_LENGTH:.0f} × {STANCE_WIDTH:.0f} mm footprint over the feet, {DECK:.0f} mm to the top deck,
standing on a {SLAB_L:.0f} × {SLAB_W:.0f} × {SLAB_H:.0f} mm slab {PLATE_H:.0f} mm off the ground. The
lowest fixed structure is the coxa plate at {CLEARANCE:.0f} mm. Feet sit on a
R{FOOT_RADIUS:.0f} circle about each yaw axis.

**The widest and longest points of the machine are the knee cans, not the
feet**: {OVER_CANS_L:.0f} × {OVER_CANS_W:.0f} mm over the cans against {STANCE_LENGTH:.0f} × {STANCE_WIDTH:.0f} over the feet. A
Ø{CAN_FK[0]:.0f} disc centred on a knee reaches {CAN_FK[0]/2:.0f} mm past the foot it stands on.

## 2. Principal dimensions

| Quantity | Value | Where it comes from |
|---|---|---|
| Length × width over the feet | {STANCE_LENGTH:.0f} × {STANCE_WIDTH:.0f} mm | stance, hip grid |
| Length × width over the cans | {OVER_CANS_L:.0f} × {OVER_CANS_W:.0f} mm | can envelope on the joints |
| Highest can (knee) above ground | {KNEE_TOP:.0f} mm | vs {DECK:.0f} mm deck |
| Body slab | {SLAB_L:.0f} × {SLAB_W:.0f} × {SLAB_H:.0f} mm | `hexapod_model.BODY`, `cad/leg/leg.json` |
| Deck height | {DECK:.0f} mm | floor plate {PLATE_H:.0f} + slab {SLAB_H:.0f} |
| Hip (femur axis) height | {HIP_H:.0f} mm | `cad/leg/leg.json` pivots |
| Ground clearance (coxa plate) | {CLEARANCE:.0f} mm | `cad/leg/leg.json` pivots |
| Stance width / length | {STANCE_WIDTH:.0f} / {STANCE_LENGTH:.0f} mm | computed |
| Foot circle about a yaw axis | R{FOOT_RADIUS:.0f} mm | computed |
| Yaw axis spacing / hip pitch | {2*HIPY:.0f} / {HIPX[0]-HIPX[1]:.0f} mm | `hexapod_model.BODY` |
| Coxa / femur / tibia | {COXA:.0f} / {FEMUR:.0f} / {TIBIA:.0f} mm | `cad/leg/leg.json` |
| Femur / knee actuator can | Ø{CAN_FK[0]:.0f} × {mm(CAN_FK[1])} mm, {M_FK:.2f} kg | `cad/actuator/frameless.json` |
| Yaw actuator can | Ø{CAN_YAW[0]:.0f} × {mm(CAN_YAW[1])} mm, {M_YAW:.2f} kg | `cad/actuator/frameless-yaw.json` |
| The same in 08 §9.17 | Ø{CAN_FK_917[0]:.0f} × {mm(CAN_FK_917[1])} mm, {M_FK_917:.2f} kg | motor + 6 mm case; no reducer stack |

## 2b. Masses

| Item | Mass | Where it comes from |
|---|---|---|
| Femur / knee unit × 12 | {M_FK:.2f} kg each, {12*M_FK:.1f} kg | `cad/actuator/frameless.json` `total_g` |
| Yaw unit × 6 | {M_YAW:.2f} kg each, {6*M_YAW:.1f} kg | `cad/actuator/frameless-yaw.json` `total_g` |
| All eighteen units | **{M_ACT:.1f} kg** | against {M_ACT_917:.1f} kg on §9.17's {M_FK_917:.2f} / {M_YAW_917:.2f} kg units |
| Everything else in the closure | {M_FIXED:.1f} kg | body {MASS.body_structure:.0f}, legs {MASS.legs:.1f}, batteries {MASS.batteries:.0f}, electronics {MASS.electronics:.0f}, deck {MASS.payload_interface:.0f}, margin {MASS.margin:.0f} |
| **Robot, 14b closure** | **{M_ROBOT_CLOSURE:.1f} kg** | fixed point in `frameless_motor.json` (all 18 units identical) |
| Robot, sized yaw unit | {M_ROBOT_SIZED_YAW:.1f} kg | the same with the Ø{CAN_YAW[0]-2*HOUSING_R:.0f} yaw motor |
| Leg structure, round-14 CAD | {LEG_STRUCT_CAD:.2f} kg per leg | `cad/leg/leg.json` `leg_structure_g` |
| less the deleted capstan drive | −{LEG_TRANS_CAD:.2f} kg per leg | `group_mass_g.transmission` |
| **Robot on the leg CAD's structure** | **~{M_ROBOT_CAD:.0f} kg (estimate)** | against {MASS.legs/6:.1f} kg per leg in the budget |
| Mission payload | {MASS.mission_payload:.0f} kg | `hexapod_model.MASS` |
| Rider (stretch goal) | {MASS.rider:.0f} kg | `hexapod_model.MASS` |

Joint margins move with that mass: femur {MARGIN_CLOSURE['femur']:.2f} / knee {MARGIN_CLOSURE['knee']:.2f} / yaw {FM['yaw']['margin']:.2f} at
{M_ROBOT_CLOSURE:.1f} kg become femur {MARGIN_AT['femur']:.2f} / knee {MARGIN_AT['knee']:.2f} / yaw {MARGIN_AT['yaw']:.2f} at {M_ROBOT_CAD:.0f} kg.

## 3. Where the eighteen units go

![body plan]({rel(figs['body'])})

The six yaw cans sit on the yaw axes under the floor plate, inside the
{-POD_BOT:.0f} mm hip pod, and clear the slab edge by {-YAW_OVERHANG:.0f} mm. If the yaw shared the
Ø{CAN_FK[0]:.0f} femur/knee unit — which is what the closure's own mass fixed point
assumes — it would overhang the slab by {YAW_OVERHANG_FK:.0f} mm a side.

The twelve femur/knee cans are the problem. Height is not it: three cans
coaxial on a yaw axis are {STACK_H:.0f} mm inside a {SLAB_H:.0f} mm slab. Planform is:

| | mm² |
|---|---|
| Slab planform | {A_SLAB:,.0f} |
| less six yaw cans | −{A_YAW_ALL:,.0f} |
| less two batteries and the electronics box | −{A_BOX:,.0f} |
| **free** | **{A_FREE:,.0f}** ({100*A_FREE/A_SLAB:.0f} %) |
| twelve Ø{CAN_FK[0]:.0f} cans need | {A_NEED_12:,.0f} |
| **shortfall** | **{AREA_SHORT:.1f}×** |

## 4. What one leg can reach

![leg envelope]({rel(figs['leg'])})

Swept over the femur window {FEM_RANGE[0]:+.0f}…{FEM_RANGE[1]:+.0f}°, the knee window
{KNEE_LIM[0]:.0f}…{KNEE_LIM[1]:.0f}° and the tibia's absolute window {TAU_RANGE[0]:.0f}…{TAU_RANGE[1]:.0f}°:

| | |
|---|---|
| Neutral foot radius | {FOOT_RADIUS:.0f} mm from the yaw axis |
| Max reach, any height | {REACH_MAX:.0f} mm (femur {POSE_EXT[0]:.0f}°, τ {POSE_EXT[1]:.0f}°) |
| Reach along the ground line | {REACH_GROUND[0]:.0f} → {REACH_GROUND[1]:.0f} mm |
| Step-up above the ground line | {STEP_UP:.0f} mm |
| Reach below the ground line | {DROP_MAX:.0f} mm |
| Folded pose | femur {POSE_FOLD[0]:.1f}°, knee {KNEE_LIM[0]:.0f}° |
| Folded envelope | {FOLD_R:.0f} × {FOLD_Z:.0f} mm, foot at R{FOLD_FOOT_R:.0f} |

The folded pose uses {FEM_UP_0:.1f}° of femur, not the {FEM_RANGE[1]:.0f}° the window allows,
because the round-14 sweep found the femur root hits the floor plate there.

## 5. How big it is

![scale]({rel(figs['scale'])})

## 6. Computed, assumed, and not known

**Computed from the models:** every length in §2 and §4; the packing areas
and the greedy pack; the unit can sizes from the motor in
`hw/stator/frameless_motor.json` plus the half-section's housing allowance;
the mass roll-ups; the walking power and pack size from `hexapod_model.ENERGY`.

**Assumed, and marked as such on the figures:** the yaw can's placement inside
the hip pod (the pod envelope is from the CAD, the position in it is not); the
femur and knee cans sitting on their joints (a consequence of deleting the
capstan, not a designed mounting); the battery and electronics envelopes,
which are the concept boxes from `concepts/hexapod_skeleton.py`; the large dog
at {800:.0f} mm at the withers, which is a scale reference and not a model number.

**Soft numbers.** The can diameter: Ø{CAN_FK[0]:.0f} is the half-section's motor + {HOUSING_R:.0f} mm
radial; the same file's sweep comment sizes the motor against a Ø{CAN_OD_ALT:.0f} can
carried over from the PCB unit — {CAN_OD_ALT - CAN_FK[0]:.0f} mm more on the diameter. If every unit
uses the Ø{CAN_OD_ALT:.0f} can, the units on the yaw axes overhang the slab by
{CAN_OD_ALT/2 - EDGE:.0f} mm a side, the machine is {CAN_OD_ALT - CAN_FK[0]:.0f} mm wider over the knees, and the packing
gets worse, not better. Which of the two is the real can has not been settled.

**Not known:** the mounting of a femur or knee unit on its own joint; whether
the femur can still lift with a Ø{CAN_FK[0]:.0f} can on its pivot; the mass of the leg
that results; the top-deck payload interface.

## 7. Open items

1. Decide where the femur and knee reductions live now the capstan is gone —
   on the joints (drawn here, unbuilt) or a new stage from the body.
2. Re-run the range-of-motion sweep with the cans on the joints.
3. Re-solve the closure on the CAD unit mass ({M_FK:.2f} kg, not {M_FK_917:.2f}) and the leg CAD's
   structure ({LEG_STRUCT_EST:.1f} kg a leg, not {MASS.legs/6:.1f}). At ~{M_ROBOT_CAD:.0f} kg the femur is {MARGIN_AT['femur']:.2f} and the
   knee {MARGIN_AT['knee']:.2f}; the battery grows from {2*PACK_KG:.1f} kg to {2*PACK_KG_CAD:.1f} kg to hold the same
   {ENERGY.endurance_h:.0f} h, which makes it worse again.
4. Carry the {mm(CAN_FK[1])} mm can height into 08 §9.17, which still says {mm(CAN_FK_917[1])} mm.
5. Fix the actuator can diameter at Ø{CAN_FK[0]:.0f} or Ø{CAN_OD_ALT:.0f} and re-cut the packing.
6. Nothing here is reviewed. This note has had no human review.
"""
    with open(DOC, "w") as fh:
        fh.write(doc)


if __name__ == "__main__":
    figs = {"side": view_side(), "front": view_front(), "top": view_top(),
            "body": view_body_plan(), "leg": view_leg_envelope(), "scale": view_scale()}
    d = write_json()
    write_doc(figs)
    print(f"can (femur/knee) Ø{CAN_FK[0]:.0f} x {mm(CAN_FK[1])}, {M_FK:.2f} kg;  yaw Ø{CAN_YAW[0]:.0f} x {mm(CAN_YAW[1])}, {M_YAW:.2f} kg")
    print(f"overall {OVERALL_L:.0f} L x {OVERALL_W:.0f} W, deck {DECK:.0f}, hip {HIP_H:.0f}, clearance {CLEARANCE:.0f}")
    print(f"packing: {len(PACKED)}/12 fk cans on one level; free {A_FREE/1e6:.3f} m2 vs {A_NEED_12/1e6:.3f} needed ({AREA_SHORT:.1f}x short)")
    print(f"reach {REACH_MAX:.0f} max, ground line {REACH_GROUND[0]:.0f}..{REACH_GROUND[1]:.0f}, step-up {STEP_UP:.0f}, folded {FOLD_R:.0f}x{FOLD_Z:.0f}")
    print(f"mass: units {M_ACT:.1f}, closure {M_ROBOT_CLOSURE:.1f}, leg-CAD estimate {M_ROBOT_CAD:.1f} kg")
    print("wrote", JSN, DOC, *figs.values(), sep="\n  ")

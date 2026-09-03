#!/opt/hw-py/bin/python
"""Leg assembly CAD: one of six legs on the off-the-shelf outrunner actuator
family (round 10-12 of docs/design/08-actuator-design.md), in build123d.

    /opt/hw-py/bin/python cad/leg/leg.py            # build, sweep, export, render
    /opt/hw-py/bin/python cad/leg/leg.py --quick    # skip the range-of-motion sweep

Frame (the LEG frame, = body frame translated to the left-mid hip): origin on
the yaw axis at the underside of the body floor plate; +x forward, +y outboard,
+z up.  The femur pitch axis is at (0, 150, Z_PIVOT) and is parallel to x.
Femur local frame: origin at the femur pivot, +y along the femur toward the
knee, +z the "exterior" side of the knee (the tibia folds under the -z side).
Tibia local frame: origin at the knee pin, +y along the tibia toward the foot.
Crank local frame: origin at the femur pivot, same orientation as the tibia.

Architecture (why it is shaped like this, see docs/design/09-leg-assembly.md):
  * three cycloid modules stacked coaxially on the yaw axis inside the body
    slab: yaw (bottom, hollow 40-lobe), knee (middle, hollow 40-lobe), femur
    (top, 25-lobe).  Each is driven by its 8318 outrunner beside it through a
    1:1 HTD 5M belt, so the eccentric sleeve can be hollow: the femur's output
    shaft and the knee's output tube nest down through the modules below them
    to the two capstan drums in the coxa hub.
  * the drums are journaled in the coxa hub, so the rope pull is a load
    internal to the coxa; the through-shafts carry torque only.
  * femur: drum r 20 -> sector r 80 (4:1) on the femur.  Knee: drum r 32 ->
    crank sector r 80 (2.5:1) free on the femur pivot, then a 1:1 rope loop
    (two r 70 pulleys, parallelogram) along the femur to the tibia; 40 x 2.5 =
    100:1 like the femur's 25 x 4.  Both runs of each capstan rope stay in one
    x-plane (x = +-r_drum) from drum to sector: zero fleet angle.
  * the round-1 routing (knee sector at the knee, knee rope over an idler on
    the femur pivot) is geometrically impossible over the femur range: one
    run lifts off the idler below femur -6 deg and the other needs a 285 deg
    self-crossing wrap.  The parallelogram replaces it.
Writes build/cad/leg/*.step + *.stl, cad/leg/leg.json, docs/design/leg/*.png.
"""
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from build123d import (Align, Box, Circle, Compound, Cylinder, Edge, Plane, Pos, Rot,  # noqa: E402
                       SlotCenterToCenter, Sphere, Wire, export_step, export_stl, extrude, make_face, sweep)

QUICK = "--quick" in sys.argv
OUT_DIR = os.path.join(ROOT, "build", "cad", "leg")
FIG_DIR = os.path.join(ROOT, "docs", "design", "leg")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
CS = json.load(open(os.path.join(ROOT, "hw", "stator", "cost_search.json")))
CAP = json.load(open(os.path.join(ROOT, "hw", "stator", "capstan.json")))

# ---- leg geometry (analysis/hexapod_model.py: coxa 150 / femur 250 / tibia 500) ----------------
L_COXA, L_FEMUR, L_TIBIA = 150.0, 250.0, 500.0
STANCE = dict(femur_deg=45.0, knee_deg=45.0, yaw_deg=0.0)       # sprawl stance: femur 45 deg up, tibia vertical
MAMMAL = dict(femur_deg=-45.0, knee_deg=114.3, yaw_deg=90.0)     # the reconfigured mammal stance (leg3d.sprawl_mammal_mode)
HIP_HEIGHT = L_TIBIA * math.cos(0.0) - L_FEMUR * math.sin(math.radians(45.0))   # 323 mm, femur axis above ground in the stance
FEMUR_RANGE = (-60.0, 85.0)      # deg from horizontal, 145 deg of femur capstan travel (the sector groove is cut for it)
# The knee capstan drives the crank, i.e. the tibia's ABSOLUTE angle tau (direction angle of the tibia in the leg
# plane, from +y toward +z; 270 = straight down).  The sizing workspaces (analysis/leg3d.py, ROUTINE and STUMBLE in
# both stances) need tau 209..314; the crank groove covers 130 deg like capstan.py.
TAU_RANGE = (200.0, 330.0)
KNEE_LIMITS = (15.0, 145.0)      # interior knee angle where the folded tibia hits the femur pivot parts / full extension
YAW_CHECK = (-90.0, -45.0, 0.0, 45.0, 90.0)   # +-45 asked; +-90 is the mammal reconfiguration (YAW_RANGE_DEG 95)


def tau_of(phi, theta):
    return phi + 180.0 + theta


def theta_of(phi, tau):
    return tau - phi - 180.0


# ---- body slab around the hip -------------------------------------------------------------------
T_PLATE = 6.0
SLAB_H = 220.0                   # hexapod_model.Body.height
Y_SLAB_EDGE = 75.0               # slab side face outboard of the yaw axis: module r 60 + 15 mm rail; was 232 with the pancakes
Y_RAIL_IN = 59.0
HIP_PITCH = 330.0
BODY_HALF_W = 120.0              # yaw axis from the body centreline
# the three cycloid modules stacked on the yaw axis (bottom -> top): yaw, knee, femur
LEVEL_PITCH = 69.0
MODULE_H = 40.0                  # housing: output bearing 13 + flange 4 + two 8 mm discs at 12 pitch + cage top
LEVEL_PLATE_T = 6.0              # the module cover = the motor mount / heat-sink plate
PULLEY_H, PULLEY_R = 17.0, 19.1  # HTD 5M 24T, 15 mm belt + flange
MOD_R = {"yaw": 60.0, "knee": 55.0, "femur": 55.0}          # yaw: RB7013 (OD 100) needs the bigger foot; hollow modules get a bigger pin circle
LOBES = {"yaw": 40, "knee": 40, "femur": 25}                 # cycloid lobes; knee 40 x 2.5 capstan = femur 25 x 4 = 100:1
CAPSTAN_RATIO = {"femur": 4.0, "knee": 2.5}
LEVEL = {"yaw": 0, "knee": 1, "femur": 2}
MOTOR_R, MOTOR_H = 46.0, 40.0                                # 8318: dia 92 x 40 (hw/stator/motor_market.json listing)
MOTOR_XY = {"yaw": (50.0, -115.0), "knee": (115.0, 0.0), "femur": (-115.0, 0.0)}
MOTOR_MASS = CS["outrunner"]["mass_kg"] * 1000               # 650 g, listing
M_RED, M_HSG = 1.25, 0.55                                    # kg, analysis/cost_search.py roll-up for the reducer and housing
UNIT_MASS = MOTOR_MASS + (M_RED + M_HSG) * 1000              # 2450 g, the cost-search unit
UNIT_PRICE = {20: 423.05, 100: 284.16}

# ---- hip pod: yaw output flange and the coxa hub with the two drums -----------------------------
Z_PIVOT = -100.0                 # femur pitch axis below the floor plate underside (was -80: the crank sector's top arc must clear the hub)
R_YAW_FLANGE = 50.0              # output flange on the RB7013 inner ring (bore 70, OD 100)
HUB_T = 6.0
Z_HUB_TOP = (-6.0, -12.0)
Z_HUB_MID = (-48.0, -54.0)
Z_HUB_BOT = (-90.0, -96.0)
HUB_X = 46.0                     # half-width inside the coxa side plates
HUB_Y_IN = -52.0                 # inboard extent of the hub plates
DRUM_H = 32.0                    # 5.8 turns of 5 mm rope at 5.5 mm pitch: 3 dead wraps + working turns
ROPE_D = CAP["geometry"]["rope"]["d_mm"]                     # 5 mm Marlow D12 Max 78
R_DRUM = {"femur": 20.0, "knee": 32.0}      # rope centreline radii; D/d = 8 (the capstan.py floor) and 12.8
R_SECTOR = {j: CAPSTAN_RATIO[j] * R_DRUM[j] for j in R_DRUM}  # 80 / 80
R_LINK = 70.0                    # the 1:1 knee link loop pulleys (drive pulley on the crank shaft, knee pulley on the tibia)
X_LINK = (54.0, 66.0)            # the link loop lives outside the +x cheeks; rope plane x = 60
# Round 14b: torque into the aluminium parts on the two Ø30 shafts goes through 42CrMo4 hubs, not keys in aluminium
# (the 12 mm key in the pulley hub was SF 0.9 in shear and 0.23 in hub bearing, analysis/leg_loads.py 'keys'):
HUB_L = 24.0                     # link pulley hubs: keyed sleeve x 54..78, two 10 x 8 keys; the pulley web is dowelled to the hub flange
HUB_R = 20.0                     # hub sleeve OD 40 under the pulley (r 15..20)
HUB_FLANGE_R, HUB_FLANGE_T = 38.0, 3.0   # the flange the aluminium part is dowelled to: 4 x Ø8 at r 36 + 6 x M6
STUB_D, STUB_BORE = 24.0, 10.0   # the crank shaft's -x end steps down to Ø24: crank plate B's hub (bore 24, two keys 26 mm) is the Ø30 journal there
STUB_X = (-56.0, -26.0)
SHAFT_END = X_LINK[0] + HUB_L + 2.0      # 80: both shafts run to the end of the pulley hub
Z_DRUM = {"knee": (-16.0, -48.0), "femur": (-58.0, -90.0)}   # knee drum on the tube (upper), femur drum on the inner shaft (lower)
N_DEAD_WRAPS = 3.0               # capstan.py: turns of rope on the drum beyond the working turns
ROPE_PITCH = 5.5                 # mm per turn on the drum
DRUM_FLANGE = 3.0
X_ROPE = {"femur": R_DRUM["femur"], "knee": R_DRUM["knee"]}  # the plane each rope's runs live in: x = +-r_drum
SHAFT_D = 18.0                   # femur output shaft, 42CrMo4
TUBE_OD, TUBE_ID = 30.0, 22.0    # knee output tube
# coxa side plates
X_CHEEK = (46.0, 52.0)           # 6 mm 6061 plates, both sides
X_BOSS = (40.0, 46.0)            # bearing bosses inside the cheeks (HK3012 needs 12 mm)
PIN_D, PIN_BORE = 30.0, 18.0     # hollow hardened shafts at the femur pivot (the knee crank shaft) and the knee (the tibia's pin)
# femur pivot stack along x (one side): carrier |x|<12.5 | femur sector 15..25 | crank sector 27..37 | boss 40..46 | cheek 46..52 | drive pulley 54..66 (+x only)
X_FSECT = (15.0, 25.0)
X_CRANK = (27.0, 37.0)
CARRIER_HALF = 12.5
PLATE_CORE, PLATE_CHEEK = 6.0, 2.0
# links
BEAM_SECTION = (30.0, 60.0, 3.0)    # femur: 30 x 60 x 3 6061-T6 rectangular tube, 30 across x
BEAM_V = 65.0                       # beam centreline above the pivot-knee line (the tibia folds under it)
BEAM_U = (45.0, 205.0)
TIBIA_SECTION = (30.0, 50.0, 3.0)   # 30 x 50 x 3, 30 across x
TIBIA_U = (45.0, 480.0)
FOOT_PAD_R, FOOT_PAD_H = 30.0, 15.0
AL, STEEL, RUBBER, BRONZE = 2.70e-3, 7.85e-3, 1.15e-3, 8.8e-3   # g/mm^3
DENS = {}                         # part -> density (or None for an assigned mass)
ASSIGNED = {}                     # part -> mass g

GROUPS = ("body_stub", "hip_pod", "coxa", "femur", "tibia", "transmission", "foot")
COLORS = {"body_stub": "#b8bfc6", "hip_pod": "#8a97a3", "coxa": "#d98c3a", "femur": "#0f9b8e", "tibia": "#2f6f9f",
          "transmission": "#c0392b", "foot": "#3a3a3a", "motor": "#555b61", "rope": "#e8c15a", "sector": "#a93226", "drum": "#6c8e3a",
          "crank": "#7d3c98", "pulley": "#5b8c5a"}


# ---- helpers ---------------------------------------------------------------------------------------
def cyl(r, z0, z1, x=0.0, y=0.0):
    return Pos(x, y, min(z0, z1)) * Cylinder(r, abs(z1 - z0), align=(Align.CENTER, Align.CENTER, Align.MIN))


def ring(r_in, r_out, z0, z1, x=0.0, y=0.0):
    return cyl(r_out, z0, z1, x, y) - cyl(r_in, min(z0, z1) - 1, max(z0, z1) + 1, x, y)


def xbox(x0, x1, y0, y1, z0, z1):
    return Pos(x0, y0, z0) * Box(x1 - x0, y1 - y0, z1 - z0, align=(Align.MIN, Align.MIN, Align.MIN))


def cyl_between(p, q, r):
    """Cylinder from p to q (3-vectors)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = float(np.linalg.norm(d))
    d = d / L
    pl = Plane(origin=tuple(p), z_dir=tuple(d))
    return pl * Cylinder(r, L, align=(Align.CENTER, Align.CENTER, Align.MIN))


def x_plate(x0, x1, profile_pts, holes=()):
    """A plate normal to x between x0..x1 with a (y, z) polygon profile and round holes ((y, z, r), ...)."""
    face = make_face(_yz_polygon(profile_pts))
    solid = extrude(face, amount=x1 - x0)
    bbx = solid.bounding_box().min.X
    solid = Pos(x0 - bbx, 0, 0) * solid
    for (y, z, r) in holes:
        solid = solid - Pos(x0 - 1, y, z) * (Plane.YZ * Cylinder(r, x1 - x0 + 2, align=(Align.CENTER, Align.CENTER, Align.MIN)))
    return solid


def _yz_polygon(pts):
    """Closed wire in the YZ plane (x = 0) through (y, z) points."""
    P = [(0.0, float(y), float(z)) for y, z in pts]
    return Wire.make_polygon([*P, P[0]])


def x_disc(x0, x1, r, y=0.0, z=0.0):
    return Pos(x0, y, z) * (Plane.YZ * Cylinder(r, x1 - x0, align=(Align.CENTER, Align.CENTER, Align.MIN)))


def x_arc_sector(x0, x1, r_in, r_out, a0, a1, y=0.0, z=0.0, n=24):
    """Annular sector normal to x between angles a0..a1 (deg, from +y toward +z)."""
    pts = [(y + r_out * math.cos(math.radians(a)), z + r_out * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]
    pts += [(y + r_in * math.cos(math.radians(a)), z + r_in * math.sin(math.radians(a))) for a in np.linspace(a1, a0, n)]
    return x_plate(x0, x1, pts)


def rect_tube(w, h, t, u0, u1, v_c):
    """Rectangular tube along local +y (u) from u0 to u1, w across x, h across z, centred at z = v_c."""
    outer = Pos(0, u0, v_c) * Box(w, u1 - u0, h, align=(Align.CENTER, Align.MIN, Align.CENTER))
    inner = Pos(0, u0 - 1, v_c) * Box(w - 2 * t, u1 - u0 + 2, h - 2 * t, align=(Align.CENTER, Align.MIN, Align.CENTER))
    return outer - inner


def yz_dir(a_deg):
    return np.array([0.0, math.cos(math.radians(a_deg)), math.sin(math.radians(a_deg))])


# ---- rope geometry in the plane of each run -------------------------------------------------------
def tangent_point_circle(P, C, r, upper=True):
    """Tangent point on the circle (centre C, radius r) of the tangent line from the point P; (y, z)."""
    P, C = np.asarray(P, float), np.asarray(C, float)
    d = np.linalg.norm(P - C)
    alpha = math.atan2(P[1] - C[1], P[0] - C[0])
    beta = math.acos(r / d)
    cands = [C + r * np.array([math.cos(alpha + s * beta), math.sin(alpha + s * beta)]) for s in (1, -1)]
    cands.sort(key=lambda t: t[1], reverse=upper)
    return cands[0]


def ang(C, T):
    return math.degrees(math.atan2(T[1] - C[1], T[0] - C[0]))


def unwrap_to(a, ref):
    while a - ref > 180:
        a -= 360
    while a - ref < -180:
        a += 360
    return a


PIV = np.array([L_COXA, Z_PIVOT])


def drum_band(j):
    """Where the rope leaves each drum, mm from the drum's mid-height: the wrapped band is N_DEAD_WRAPS + the working
    turns, it walks along the drum by one pitch per working turn, and run A leaves one end of it, run B the other.
    Returns (A range, B range, band height, groove height, walk)."""
    work = (FEMUR_RANGE[1] - FEMUR_RANGE[0] if j == "femur" else TAU_RANGE[1] - TAU_RANGE[0]) * CAPSTAN_RATIO[j] / 360
    band, walk = (N_DEAD_WRAPS + work) * ROPE_PITCH, work * ROPE_PITCH
    z0, z1 = Z_DRUM[j]
    groove = (z0 - z1) - 2 * DRUM_FLANGE
    lo = z1 + DRUM_FLANGE + ROPE_D / 2 - 0.5 * (z0 + z1)          # lowest turn centre, relative to mid-height
    hi = z0 - DRUM_FLANGE - ROPE_D / 2 - 0.5 * (z0 + z1)          # highest turn centre
    # run B (the lower run) leaves the bottom of the band, run A the top; the band starts at the bottom of the groove.
    # Where the band + walk does not fit the groove (leg.json drum_bands.fits) the departures are clamped to the groove.
    B = (lo, min(lo + walk, hi))
    A = (min(lo + band - ROPE_PITCH, hi), min(lo + band - ROPE_PITCH + walk, hi))
    return A, B, band, groove, walk


def rope_geometry(phi, tau, z_off=None):
    """All rope runs in the coxa frame for femur angle phi and tibia absolute angle tau (deg).
    Returns dict of run name -> (P, Q) endpoints as 3-vectors, plus contact angles (coxa frame).
    z_off: optional {run name: dz} moving a capstan run's drum departure from the drum's mid-height (see drum_band)."""
    out = {"runs": {}, "contact": {}}
    for j in ("femur", "knee"):
        zd = 0.5 * (Z_DRUM[j][0] + Z_DRUM[j][1])
        for run, sgn, upper in (("A", 1, True), ("B", -1, False)):
            x = sgn * X_ROPE[j]
            zr = zd + (z_off or {}).get(f"{j}_{run}", 0.0)
            T = tangent_point_circle((0.0, zr), PIV, R_SECTOR[j], upper)
            out["runs"][f"{j}_{run}"] = (np.array([x, 0.0, zr]), np.array([x, T[0], T[1]]))
            out["contact"][f"{j}_{run}"] = ang(PIV, T)
    # the 1:1 link loop along the femur: both runs parallel to the pivot-knee line at +-R_LINK, in the plane x = 60
    u = yz_dir(phi)
    v = yz_dir(phi + 90.0)          # exterior
    knee = PIV + L_FEMUR * u[1:]
    xl = 0.5 * (X_LINK[0] + X_LINK[1])
    for run, s in (("ext", 1), ("int", -1)):
        p = np.array([xl, PIV[0], PIV[1]]) + s * R_LINK * v
        q = np.array([xl, knee[0], knee[1]]) + s * R_LINK * v
        out["runs"][f"link_{run}"] = (p, q)
    out["knee"] = knee
    return out


# ---- groove arcs (in the link frames) from the contact geometry over the ranges ------------------
G0 = rope_geometry(STANCE["femur_deg"], tau_of(STANCE["femur_deg"], STANCE["knee_deg"]))
GROOVE_MARGIN = 6.0
# femur sector: contact angles are fixed in the coxa frame; in the femur frame they are (c - phi)
GROOVE_F = {run: (G0["contact"][f"femur_{run}"] - FEMUR_RANGE[1], G0["contact"][f"femur_{run}"] - FEMUR_RANGE[0]) for run in ("A", "B")}
# knee crank: fixed contact angles in the coxa frame; in the crank frame (rotating with tau) they are (c - tau)
GROOVE_C = {run: (G0["contact"][f"knee_{run}"] - TAU_RANGE[1], G0["contact"][f"knee_{run}"] - TAU_RANGE[0]) for run in ("A", "B")}
# link loop: the tangent points are fixed in the FEMUR frame (+90 exterior, -90 interior); the pulleys turn with the
# tibia, i.e. by (tau - phi) = 180 + theta relative to the femur.  Each run winds CCW on one pulley and CW on the other,
# so its total length is constant (parallelogram); the eye on each pulley sits where the wrap is >= EYE_MARGIN at the
# end of the knee range where that pulley has paid out.
EYE_MARGIN = 10.0
REL0, REL1 = 180.0 + KNEE_LIMITS[0], 180.0 + KNEE_LIMITS[1]       # tau - phi over the knee range
LINK_EYES = {  # eye angle in the pulley's own frame (drive pulley: crank frame; knee pulley: tibia frame)
    "ext": dict(drive=90.0 - REL0 + EYE_MARGIN, knee=90.0 - REL1 - EYE_MARGIN),     # drive winds CCW from +90, knee winds CW from +90
    "int": dict(drive=-90.0 - REL1 - EYE_MARGIN, knee=-90.0 - REL0 + EYE_MARGIN),   # drive winds CW from -90 (pays out as the knee opens), knee winds CCW from -90
}


def link_wound(run, pulley, phi, tau):
    """Wound angle (deg, >= 0 when the rope is on the pulley) of a link run on a pulley, and the arc (a0, a1) in the coxa frame."""
    rel = tau - phi
    if run == "ext":
        tangent = 90.0 + phi
        eye = LINK_EYES["ext"][pulley] + rel + phi
        wound = (eye - tangent) if pulley == "drive" else (tangent - eye)
    else:
        tangent = -90.0 + phi
        eye = LINK_EYES["int"][pulley] + rel + phi
        wound = (tangent - eye) if pulley == "drive" else (eye - tangent)
    return wound, tuple(sorted((tangent, eye)))


# ---- part builders -------------------------------------------------------------------------------
def build_body_stub():
    P = {}
    x0, x1 = -HIP_PITCH / 2, HIP_PITCH / 2
    y_in = -BODY_HALF_W - 60                          # to a little past the body centreline
    P["floor_plate"] = xbox(x0, x1, y_in, Y_SLAB_EDGE, 0, T_PLATE) - cyl(R_YAW_FLANGE + 2, -1, T_PLATE + 1)
    P["top_deck"] = xbox(x0, x1, y_in, Y_SLAB_EDGE, SLAB_H - T_PLATE, SLAB_H)
    P["side_rail"] = xbox(x0, x1, Y_RAIL_IN, Y_SLAB_EDGE, T_PLATE, SLAB_H - T_PLATE)
    DENS["floor_plate"] = DENS["top_deck"] = DENS["side_rail"] = AL
    for j, k in LEVEL.items():
        zb = T_PLATE + k * LEVEL_PITCH
        r = MOD_R[j]
        P[f"module_{j}"] = cyl(r, zb, zb + MODULE_H) - cyl(35.0 if j != "yaw" else 45.0, zb - 1, zb + MODULE_H + 1)
        ASSIGNED[f"module_{j}"] = (M_RED + M_HSG) * 1000 - 400      # the cost-search reducer + housing less the level plate (modelled below)
        mx, my = MOTOR_XY[j]
        zc = zb + MODULE_H
        plate = cyl(r + 12, zc, zc + LEVEL_PLATE_T) + cyl(MOTOR_R + 10, zc, zc + LEVEL_PLATE_T, mx, my)
        d = math.hypot(mx, my)
        bar = Pos(0, 0, zc) * (Rot(0, 0, math.degrees(math.atan2(my, mx))) * Box(d, 70, LEVEL_PLATE_T, align=(Align.MIN, Align.CENTER, Align.MIN)))
        P[f"level_plate_{j}"] = (plate + bar) - cyl(PULLEY_R + 3, zc - 1, zc + LEVEL_PLATE_T + 1) - cyl(6, zc - 1, zc + LEVEL_PLATE_T + 1, mx, my)
        DENS[f"level_plate_{j}"] = AL
        P[f"motor_{j}"] = cyl(MOTOR_R, zc - MOTOR_H, zc, mx, my)
        ASSIGNED[f"motor_{j}"] = MOTOR_MASS
        P[f"pulley_{j}"] = ring(16, PULLEY_R, zc + LEVEL_PLATE_T, zc + LEVEL_PLATE_T + PULLEY_H) + ring(4, PULLEY_R, zc + LEVEL_PLATE_T, zc + LEVEL_PLATE_T + PULLEY_H, mx, my)
        ASSIGNED[f"pulley_{j}"] = 2 * 55.0                       # two HTD 5M 24T aluminium pulleys, 15 mm, estimate
        slot_o = SlotCenterToCenter(d, 2 * PULLEY_R + 3.0)
        slot_i = SlotCenterToCenter(d, 2 * PULLEY_R)
        belt = extrude(slot_o - slot_i, amount=15.0)
        belt = Pos(0, 0, zc + LEVEL_PLATE_T + 1) * (Rot(0, 0, math.degrees(math.atan2(my, mx))) * (Pos(d / 2, 0, 0) * belt))
        P[f"belt_{j}"] = belt
        ASSIGNED[f"belt_{j}"] = 30.0                              # HTD 5M-375-15, estimate
    return P


def build_hip_pod():
    """Rotates with the coxa: yaw output flange, hub plates, drum bearings; plus the two through-shafts."""
    P = {}
    P["yaw_flange"] = ring(TUBE_OD / 2 + 4, R_YAW_FLANGE, -6.0, 0.0)
    DENS["yaw_flange"] = AL
    for name, (z0, z1), r_disc in (("hub_top", Z_HUB_TOP, 42.0), ("hub_mid", Z_HUB_MID, 26.0), ("hub_bot", Z_HUB_BOT, 22.0)):   # hub_bot r 22: the femur rope B passes 5 mm above it
        pl = xbox(-HUB_X, HUB_X, HUB_Y_IN, 0.0, z1, z0) + cyl(r_disc, z1, z0)
        bore = {"hub_top": 23.5, "hub_mid": 18.5, "hub_bot": 16.0}[name]     # HK4012 / HK3012 / HK2512 seats
        P[name] = pl - cyl(bore, z1 - 1, z0 + 1)
        DENS[name] = AL
    for i, (sx, sy) in enumerate(((-36, -44), (0, -44), (36, -44), (-36, -14), (36, -14))):
        P[f"hub_standoff_{i}"] = cyl(5.0, Z_HUB_BOT[1], Z_HUB_TOP[0], sx, sy)
        DENS[f"hub_standoff_{i}"] = AL
    # through-shafts from the modules above: knee tube from the middle module's flange, femur shaft from the top module's
    z_knee_flange = T_PLATE + LEVEL["knee"] * LEVEL_PITCH + 4.0
    z_femur_flange = T_PLATE + LEVEL["femur"] * LEVEL_PITCH + 4.0
    P["knee_tube"] = ring(TUBE_ID / 2, TUBE_OD / 2, Z_DRUM["knee"][1] + 4, z_knee_flange + 10)
    P["femur_shaft"] = ring(5.0, SHAFT_D / 2, Z_DRUM["femur"][1] + 4, z_femur_flange + 10)     # Ø18/Ø10 42CrMo4 tube: 91 N·m peak drum torque -> 88 MPa
    DENS["knee_tube"] = DENS["femur_shaft"] = STEEL
    # drum bearings (needle cups) as rings
    P["drum_bearings"] = (ring(20.0, 23.5, Z_HUB_TOP[1], Z_HUB_TOP[0]) + ring(15.0, 18.5, Z_HUB_MID[1], Z_HUB_MID[0])
                          + ring(12.5, 16.0, Z_HUB_BOT[1], Z_HUB_BOT[0]))
    ASSIGNED["drum_bearings"] = 31.1 + 24.0 + 20.0                # PTI HK4012 / HK3012 / HK2512 catalogue masses
    return P


def coxa_profile():
    zp = Z_PIVOT
    return [(-52, -6), (70, -6), (L_COXA + 30, zp + 42), (L_COXA + 44, zp), (L_COXA + 30, zp - 42), (70, zp - 40), (-52, -100)]


def build_coxa():
    """Coxa side plates with the crank-shaft bearing bosses; the crank shaft itself is in the transmission group."""
    P = {}
    holes = ((L_COXA, Z_PIVOT, 18.5),)                           # HK3012 cup bore (30 x 37 x 12) in the cheek + boss
    for side, (xa, xb) in (("L", (-X_CHEEK[1], -X_CHEEK[0])), ("R", X_CHEEK)):
        P[f"coxa_plate_{side}"] = x_plate(xa, xb, coxa_profile(), holes) - cyl(R_YAW_FLANGE + 3, -7, 1)
        DENS[f"coxa_plate_{side}"] = AL
        xb0, xb1 = (X_BOSS if side == "R" else (-X_BOSS[1], -X_BOSS[0]))
        P[f"coxa_boss_{side}"] = x_disc(xb0, xb1, 28.0, L_COXA, Z_PIVOT) - x_disc(xb0 - 1, xb1 + 1, 18.5, L_COXA, Z_PIVOT)
        DENS[f"coxa_boss_{side}"] = AL
    P["coxa_shaft_bearings"] = (x_disc(X_BOSS[0], X_CHEEK[1], 18.5, L_COXA, Z_PIVOT) + x_disc(-X_CHEEK[1], -X_BOSS[0], 18.5, L_COXA, Z_PIVOT)) - x_disc(-60, 70, 15.0, L_COXA, Z_PIVOT)
    ASSIGNED["coxa_shaft_bearings"] = 2 * 24.0                    # HK3012 x 2, PTI catalogue mass
    # femur pivot bearings: two HK3012 in the femur carrier, on the crank shaft
    P["femur_pivot_bearings"] = x_disc(-CARRIER_HALF, CARRIER_HALF, 18.5, L_COXA, Z_PIVOT) - x_disc(-20, 20, 15.0, L_COXA, Z_PIVOT)
    ASSIGNED["femur_pivot_bearings"] = 2 * 24.0
    return P


def sector_plate(x0, x1, r_rope, groove, hub_r, spoke_holes=3, bore=15.0):
    """Laser-cut 6061 laminate: 6 mm core with the rope on its edge, 2 mm cheeks either side over the groove arc.
    Local frame of the link, at the pivot."""
    a0, a1 = groove[0] - GROOVE_MARGIN, groove[1] + GROOVE_MARGIN
    xm0, xm1 = x0 + PLATE_CHEEK, x1 - PLATE_CHEEK
    core = x_arc_sector(xm0, xm1, hub_r - 8, r_rope - ROPE_D / 2, a0, a1) + x_disc(xm0, xm1, hub_r)
    cheeks = x_arc_sector(x0, xm0, r_rope - 12, r_rope + 3, a0, a1) + x_arc_sector(xm1, x1, r_rope - 12, r_rope + 3, a0, a1)
    plate = core + cheeks
    # lightening: holes in the spokes
    for k in range(spoke_holes):
        a = a0 + (k + 0.5) * (a1 - a0) / spoke_holes
        rr = 0.5 * (hub_r + r_rope - 10)
        plate = plate - x_disc(x0 - 1, x1 + 1, 0.22 * (r_rope - hub_r), rr * math.cos(math.radians(a)), rr * math.sin(math.radians(a)))
    plate = plate - x_disc(x0 - 1, x1 + 1, bore)
    return plate


def build_femur_local():
    """Femur parts in the femur local frame (origin at the pivot, +y to the knee, +z exterior)."""
    P = {}
    w, h, t = BEAM_SECTION
    # root carrier: 25 mm 6061 plate profile from the pin hub up to the beam
    prof = [(-22, 12), (8, BEAM_V + h / 2), (BEAM_U[0] + 15, BEAM_V + h / 2), (BEAM_U[0] + 15, BEAM_V - h / 2 - 8), (30, 8), (22, -14), (-10, -22)]
    P["femur_carrier"] = (x_plate(-CARRIER_HALF, CARRIER_HALF, prof) + x_disc(-CARRIER_HALF, CARRIER_HALF, 25.0)) - x_disc(-20, 20, 18.5)
    DENS["femur_carrier"] = AL
    P["femur_beam"] = rect_tube(w, h, t, BEAM_U[0], BEAM_U[1], BEAM_V)
    DENS["femur_beam"] = AL
    # knee end: two cheek plates from the beam end down to the knee pin, with spacer blocks clamping the beam; bearing bosses inside
    holes = ((L_FEMUR, 0.0, 18.5),)
    prof_k = [(BEAM_U[1] - 20, BEAM_V + h / 2), (L_FEMUR + 10, 30), (L_FEMUR + 30, 0), (L_FEMUR + 10, -30), (L_FEMUR - 30, -24), (BEAM_U[1] - 20, BEAM_V - h / 2)]
    for side, (xa, xb) in (("L", (-X_CHEEK[1], -X_CHEEK[0])), ("R", X_CHEEK)):
        P[f"knee_cheek_{side}"] = x_plate(xa, xb, prof_k, holes)
        DENS[f"knee_cheek_{side}"] = AL
        xb0, xb1 = (X_BOSS if side == "R" else (-X_BOSS[1], -X_BOSS[0]))
        P[f"knee_boss_{side}"] = x_disc(xb0, xb1, 28.0, L_FEMUR, 0.0) - x_disc(xb0 - 1, xb1 + 1, 18.5, L_FEMUR, 0.0)
        DENS[f"knee_boss_{side}"] = AL
        xs0, xs1 = (w / 2, X_CHEEK[0]) if side == "R" else (-X_CHEEK[0], -w / 2)
        P[f"knee_spacer_{side}"] = xbox(xs0, xs1, BEAM_U[1] - 20, BEAM_U[1], BEAM_V - h / 2, BEAM_V + h / 2)
        DENS[f"knee_spacer_{side}"] = AL
    P["knee_pivot_bearings"] = (x_disc(X_BOSS[0], X_CHEEK[1], 18.5, L_FEMUR, 0.0) + x_disc(-X_CHEEK[1], -X_BOSS[0], 18.5, L_FEMUR, 0.0)) - x_disc(-60, 70, 15.0, L_FEMUR, 0.0)
    ASSIGNED["knee_pivot_bearings"] = 2 * 24.0                    # HK3012 x 2 in the femur's knee cheeks
    return P


def build_femur_sectors_local():
    P = {}
    for run, (x0, x1) in (("A", X_FSECT), ("B", (-X_FSECT[1], -X_FSECT[0]))):
        P[f"femur_sector_{run}"] = sector_plate(x0, x1, R_SECTOR["femur"], GROOVE_F[run], 40.0, bore=16.0)
        DENS[f"femur_sector_{run}"] = AL
        # tensioner: M8 eye screw at the groove's eye end (the unwound end), tangential
        a_eye = (GROOVE_F[run][0] if run == "A" else GROOVE_F[run][1])
        P[f"femur_tensioner_{run}"] = x_disc(x0, x1, 5.0, (R_SECTOR["femur"] - 14) * math.cos(math.radians(a_eye)), (R_SECTOR["femur"] - 14) * math.sin(math.radians(a_eye)))
        ASSIGNED[f"femur_tensioner_{run}"] = 45.0
    return P


def build_crank_local():
    """The knee crank in its own frame (origin at the femur pivot, rotates with the tibia's absolute angle tau):
    two sector plates on the crank shaft, the shaft itself, and the drive pulley of the 1:1 link loop on the +x end."""
    P = {}
    for run, (x0, x1) in (("A", X_CRANK), ("B", (-X_CRANK[1], -X_CRANK[0]))):
        P[f"crank_sector_{run}"] = sector_plate(x0, x1, R_SECTOR["knee"], GROOVE_C[run], 40.0)
        DENS[f"crank_sector_{run}"] = AL
        a_eye = (GROOVE_C[run][0] if run == "A" else GROOVE_C[run][1])
        P[f"crank_tensioner_{run}"] = x_disc(x0, x1, 5.0, (R_SECTOR["knee"] - 14) * math.cos(math.radians(a_eye)), (R_SECTOR["knee"] - 14) * math.sin(math.radians(a_eye)))
        ASSIGNED[f"crank_tensioner_{run}"] = 45.0
    # Round 14b: the crank shaft is Ø30/18 with a flange integral with it at x 35..38 (turned from Ø80 bar) that crank
    # plate A is dowelled to; its -x end steps down to Ø24/10 and crank plate B's 42CrMo4 hub (bore 24, two 8 x 7 keys
    # over 26 mm, OD 30 = the left cheek bearing's journal) carries its flange at x -38..-35.  No key in aluminium.
    xf = X_CRANK[1]
    P["crank_shaft"] = ((x_disc(STUB_X[1], SHAFT_END, PIN_D / 2) - x_disc(STUB_X[1] - 1, SHAFT_END + 1, PIN_BORE / 2))
                        + (x_disc(xf - 2, xf + 1, HUB_FLANGE_R) - x_disc(xf - 3, xf + 2, PIN_D / 2))
                        + (x_disc(STUB_X[0], STUB_X[1], STUB_D / 2) - x_disc(STUB_X[0] - 1, STUB_X[1] + 1, STUB_BORE / 2)))
    DENS["crank_shaft"] = STEEL
    P["crank_hub_B"] = ((x_disc(-X_CHEEK[1], STUB_X[1], PIN_D / 2) - x_disc(-X_CHEEK[1] - 1, STUB_X[1] + 1, STUB_D / 2))
                        + (x_disc(-xf - 1, -xf + 2, HUB_FLANGE_R) - x_disc(-xf - 2, -xf + 3, PIN_D / 2)))
    DENS["crank_hub_B"] = STEEL
    P["drive_pulley"] = _link_pulley()
    DENS["drive_pulley"] = AL
    P["drive_pulley_hub"] = _pulley_hub()
    DENS["drive_pulley_hub"] = STEEL
    for run in ("ext", "int"):
        a = LINK_EYES[run]["drive"]
        P[f"link_tensioner_{run}"] = x_disc(X_LINK[0], X_LINK[1], 5.0, (R_LINK - 12) * math.cos(math.radians(a)), (R_LINK - 12) * math.sin(math.radians(a)))
        ASSIGNED[f"link_tensioner_{run}"] = 45.0
    return P


def _link_pulley():
    """Turned 6061 pulley: 12 mm rim with the groove, 6 mm web dowelled to the steel hub's flange, bore = the hub OD;
    two flanges.  (Round 14b: was a 4 mm web and a keyed aluminium hub.)"""
    x0, x1 = X_LINK
    rim = x_disc(x0, x1, R_LINK - ROPE_D / 2) - x_disc(x0 - 1, x1 + 1, R_LINK - 14.0)
    web = x_disc(x1 - 6, x1, R_LINK - 13.0)
    p = rim + web + x_disc(x0, x0 + 2, R_LINK + 3) + x_disc(x1 - 2, x1, R_LINK + 3)
    p = p - x_disc(x0 - 1, x1 + 1, HUB_R)
    for k in range(6):
        a = k * 60.0
        p = p - x_disc(x0 - 1, x1 + 1, 6.0, 48.0 * math.cos(math.radians(a)), 48.0 * math.sin(math.radians(a)))
    return p


def _pulley_hub():
    """42CrMo4 hub of a link pulley on its Ø30 shaft: keyed sleeve (two 10 x 8 keys over HUB_L) with a flange the
    pulley web is dowelled to (4 x Ø8 at r 36 + 6 x M6).  Sleeve x 54..78, flange x 66..69 against the web."""
    x0, x1 = X_LINK
    return ((x_disc(x0, x0 + HUB_L, HUB_R) - x_disc(x0 - 1, x0 + HUB_L + 1, PIN_D / 2))
            + (x_disc(x1, x1 + HUB_FLANGE_T, HUB_FLANGE_R) - x_disc(x1 - 1, x1 + HUB_FLANGE_T + 1, HUB_R)))


def build_tibia_local():
    P = {}
    w, h, t = TIBIA_SECTION
    hub = x_disc(-30, 30, 28.0) - x_disc(-31, 31, PIN_D / 2)                       # keyed on the knee pin
    web = Pos(0, 20, 0) * Box(w, TIBIA_U[0] - 20 + 1, h, align=(Align.CENTER, Align.MIN, Align.CENTER))
    plug = Pos(0, TIBIA_U[0] - 1, 0) * Box(w - 2 * t - 0.5, 40.0, h - 2 * t - 0.5, align=(Align.CENTER, Align.MIN, Align.CENTER))
    P["tibia_carrier"] = hub + web + plug
    DENS["tibia_carrier"] = AL
    P["knee_pin"] = x_disc(-X_CHEEK[1] - 4, SHAFT_END, PIN_D / 2) - x_disc(-60, 90, PIN_BORE / 2)     # keyed to the carrier: two 10 x 8 x 50
    DENS["knee_pin"] = STEEL
    P["tibia_tube"] = rect_tube(w, h, t, TIBIA_U[0], TIBIA_U[1], 0.0)
    DENS["tibia_tube"] = AL
    P["knee_pulley"] = _link_pulley()
    DENS["knee_pulley"] = AL
    P["knee_pulley_hub"] = _pulley_hub()
    DENS["knee_pulley_hub"] = STEEL
    return P


def build_foot_local():
    P = {}
    w, h, t = TIBIA_SECTION
    y0 = TIBIA_U[1]
    P["foot_plug"] = Pos(0, y0 - 30, 0) * Box(w - 2 * t - 0.5, 30 + 8, h - 2 * t - 0.5, align=(Align.CENTER, Align.MIN, Align.CENTER))
    DENS["foot_plug"] = AL
    P["foot_load_cell"] = Pos(0, y0 + 8, 0) * (Plane(origin=(0, 0, 0), z_dir=(0, 1, 0)) * Cylinder(12.5, 10.0, align=(Align.CENTER, Align.CENTER, Align.MIN)))
    ASSIGNED["foot_load_cell"] = 40.0
    pad = Plane(origin=(0, y0 + 18, 0), z_dir=(0, 1, 0)) * Cylinder(FOOT_PAD_R, FOOT_PAD_H - 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    dome = Pos(0, y0 + 18 + FOOT_PAD_H - 6, 0) * Sphere(FOOT_PAD_R)
    dome = dome - Pos(0, y0 + 18 + FOOT_PAD_H - 6 - FOOT_PAD_R, 0) * Box(80, FOOT_PAD_R, 80, align=(Align.CENTER, Align.MIN, Align.CENTER))
    dome = dome - Pos(0, y0 + 18 + FOOT_PAD_H - 6 + 8, 0) * Box(80, 60, 80, align=(Align.CENTER, Align.MIN, Align.CENTER))
    P["foot_pad"] = pad + dome
    DENS["foot_pad"] = RUBBER
    return P


def build_transmission_coxa():
    """Drums with their rope, in the coxa frame."""
    P = {}
    for j in ("knee", "femur"):
        z0, z1 = Z_DRUM[j]
        rc = R_DRUM[j] - ROPE_D / 2
        d = ring(TUBE_OD / 2 + 0.5 if j == "knee" else SHAFT_D / 2 + 0.5, rc, z1, z0)
        d = d + ring(rc - 1, R_DRUM[j] + DRUM_FLANGE, z0 - DRUM_FLANGE, z0) + ring(rc - 1, R_DRUM[j] + DRUM_FLANGE, z1, z1 + DRUM_FLANGE)
        # bosses into the hub bearings
        if j == "knee":
            d = d + ring(TUBE_OD / 2 + 0.5, 20.0, z0, Z_HUB_TOP[0]) + ring(TUBE_OD / 2 + 0.5, 15.0, Z_HUB_MID[1], z1)
        else:
            d = d + ring(SHAFT_D / 2 + 0.5, 12.5, z0, Z_HUB_MID[1]) + ring(SHAFT_D / 2 + 0.5, 12.5, Z_HUB_BOT[1], z1)
        P[f"drum_{j}"] = d
        DENS[f"drum_{j}"] = AL
        # the rope on the drum: 5 turns as rings
        for k in range(5):
            zc = z1 + 3 + k * 5.5
            P[f"rope_{j}_drum_{k}"] = sweep(Plane(origin=(R_DRUM[j], 0, zc), z_dir=(0, 1, 0)) * Circle(ROPE_D / 2), path=Edge.make_circle(R_DRUM[j], Plane.XY.offset(zc)))
            ASSIGNED[f"rope_{j}_drum_{k}"] = 15.6 * 2 * math.pi * R_DRUM[j] / 1000
    return P


def fleet_angles(g=None):
    """Angle (deg) between each capstan run and the drum's plane of rotation at the departure point.  The drum turns
    about z and the sector about x, so a straight run from drum to sector is inclined; a grooved capstan drum wants
    this under ~1.5 deg.  The runs are in the planes x = +-r_drum (no fleet in x); this is the other component."""
    g = g or rope_geometry(STANCE["femur_deg"], tau_of(STANCE["femur_deg"], STANCE["knee_deg"]))
    out = {}
    for n, (p, q) in g["runs"].items():
        if n.startswith("link"):
            continue
        d = q - p
        out[n] = round(math.degrees(math.atan2(abs(d[2]), abs(d[1]))), 1)
    return out


def rope_solids(phi, tau, z_off=None):
    """Rope runs and wound arcs for a pose, in the coxa frame."""
    g = rope_geometry(phi, tau, z_off)
    P = {}
    for name, (p, q) in g["runs"].items():
        P[f"rope_{name}"] = cyl_between(p, q, ROPE_D / 2)
        ASSIGNED[f"rope_{name}"] = 15.6 * float(np.linalg.norm(q - p)) / 1000
    piv = np.array([0.0, L_COXA, Z_PIVOT])
    knee = np.array([0.0, g["knee"][0], g["knee"][1]])
    # wound arcs on the femur sector and the crank: from the contact to the eye (A unwinds as it is pulled: the eye is
    # at the groove's low end for A, high end for B)
    for j, ang_link, groove in (("femur", phi, GROOVE_F), ("knee", tau, GROOVE_C)):
        for run, sgn in (("A", 1), ("B", -1)):
            x = sgn * X_ROPE[j]
            c = g["contact"][f"{j}_{run}"]
            eye = (groove[run][0] if run == "A" else groove[run][1]) + ang_link
            a0, a1 = sorted((c, unwrap_to(eye, c)))
            if a1 - a0 > 0.5:
                P[f"rope_{j}_{run}_arc"] = _arc_rope(piv + np.array([x, 0, 0]), R_SECTOR[j], a0, a1)
    # link loop arcs on the drive pulley (at the pivot) and the knee pulley
    xl = 0.5 * (X_LINK[0] + X_LINK[1])
    for run in ("ext", "int"):
        for pulley, centre in (("drive", piv), ("knee", knee)):
            wound, (a0, a1) = link_wound(run, pulley, phi, tau)
            if wound > 0.5:
                P[f"rope_link_{run}_{pulley}_arc"] = _arc_rope(centre + np.array([xl, 0, 0]), R_LINK, a0, a1)
    for n in P:
        if n not in ASSIGNED:
            ASSIGNED[n] = 15.6 * P[n].volume / (math.pi * (ROPE_D / 2) ** 2) / 1000
    return P, g


def _arc_rope(centre, r, a0, a1):
    e = Edge.make_circle(r, Plane(origin=tuple(centre), x_dir=(0, 1, 0), z_dir=(1, 0, 0)), start_angle=a0, end_angle=a1)
    return sweep(Plane(origin=e.position_at(0), z_dir=e.tangent_at(0)) * Circle(ROPE_D / 2), path=e)


# ---- assembly ------------------------------------------------------------------------------------
def femur_loc(phi, psi):
    return Rot(0, 0, psi) * Pos(0, L_COXA, Z_PIVOT) * Rot(phi, 0, 0)


def crank_loc(tau, psi):
    return Rot(0, 0, psi) * Pos(0, L_COXA, Z_PIVOT) * Rot(tau, 0, 0)


def tibia_loc(phi, tau, psi):
    ky = L_COXA + L_FEMUR * math.cos(math.radians(phi))
    kz = Z_PIVOT + L_FEMUR * math.sin(math.radians(phi))
    return Rot(0, 0, psi) * Pos(0, ky, kz) * Rot(tau, 0, 0)


class Leg:
    def __init__(self):
        t = time.time()
        self.body = build_body_stub()
        self.pod = build_hip_pod()
        self.coxa = build_coxa()
        self.femur = build_femur_local()
        self.fsect = build_femur_sectors_local()
        self.crank = build_crank_local()
        self.tibia = build_tibia_local()
        self.foot = build_foot_local()
        self.trans = build_transmission_coxa()
        print(f"parts built in {time.time() - t:.1f} s")

    def assemble(self, phi, tau, psi, ropes=True):
        """dict name -> (solid in the leg frame, group); phi = femur angle, tau = tibia absolute angle (deg)"""
        A = {}
        yaw = Rot(0, 0, psi)
        for n, s in self.body.items():
            A[n] = (s, "body_stub")
        for n, s in self.pod.items():
            A[n] = (s.moved(yaw) if n not in ("knee_tube", "femur_shaft") else s, "hip_pod")
        for n, s in self.coxa.items():
            A[n] = (s.moved(yaw), "coxa")
        fl = femur_loc(phi, psi)
        for n, s in self.femur.items():
            A[n] = (s.moved(fl), "femur")
        for n, s in self.fsect.items():
            A[n] = (s.moved(fl), "transmission")
        cl = crank_loc(tau, psi)
        for n, s in self.crank.items():
            A[n] = (s.moved(cl), "transmission")
        tl = tibia_loc(phi, tau, psi)
        for n, s in self.tibia.items():
            A[n] = (s.moved(tl), "transmission" if n.startswith("knee_pulley") else "tibia")
        for n, s in self.foot.items():
            A[n] = (s.moved(tl), "foot")
        for n, s in self.trans.items():
            A[n] = (s.moved(yaw), "transmission")
        if ropes:
            R, g = rope_solids(phi, tau)
            for n, s in R.items():
                A[n] = (s.moved(yaw), "transmission")
        return A


def part_mass(name, solid):
    if name in ASSIGNED:
        return ASSIGNED[name]
    return solid.volume * DENS[name]


# ---- clearance sweep ------------------------------------------------------------------------------
def compound(parts, names):
    return Compound(children=[parts[n][0] for n in names if n in parts])


def clearance_pairs(leg):
    """Moving-group pairs for the sweep.  Bearings (in their seats), a rope's own pulley, and the designed 2 mm axial
    gaps between the stacked plates at the pivot are left out; everything that can meet radially is in."""
    body_names = [n for n in leg.body]
    fixed_hub = ["yaw_flange", "hub_top", "hub_mid", "hub_bot"] + [f"hub_standoff_{i}" for i in range(5)] + ["knee_tube", "femur_shaft"]
    coxa_names = [n for n in leg.coxa if "bearing" not in n]
    drums = ["drum_knee", "drum_femur"] + [f"rope_{j}_drum_{k}" for j in ("knee", "femur") for k in range(5)]
    femur_struct = [n for n in leg.femur if "bearing" not in n]
    femur_all = femur_struct + list(leg.fsect)
    crank_plates = ["crank_sector_A", "crank_sector_B", "crank_tensioner_A", "crank_tensioner_B", "crank_hub_B"]
    drive = ["drive_pulley", "drive_pulley_hub"]
    knee_pulley = ["knee_pulley", "knee_pulley_hub"]
    tibia_struct = ["tibia_carrier", "tibia_tube"] + list(leg.foot)
    return {
        "femur+sectors vs body/coxa/hub/drums": (femur_all, body_names + fixed_hub + coxa_names + drums),
        "tibia+foot vs body/coxa/hub/drums/femur/sectors/crank/pulley": (tibia_struct, body_names + fixed_hub + coxa_names + drums + femur_all + crank_plates + drive),
        "knee pulley vs body/coxa/hub/drums/crank/drive pulley": (knee_pulley, body_names + fixed_hub + coxa_names + drums + crank_plates + drive),
        "femur rope runs vs everything but their drum/sector": (["rope_femur_A", "rope_femur_B"], body_names + fixed_hub + coxa_names + ["drum_knee"] + [f"rope_knee_drum_{k}" for k in range(5)] + femur_struct + crank_plates + tibia_struct + knee_pulley),
        "knee rope runs vs everything but their drum/crank": (["rope_knee_A", "rope_knee_B"], body_names + fixed_hub + coxa_names + ["drum_femur"] + [f"rope_femur_drum_{k}" for k in range(5)] + femur_all + tibia_struct + knee_pulley),
        "link loop runs vs everything but their pulleys": (["rope_link_ext", "rope_link_int"], body_names + coxa_names + femur_struct + list(leg.fsect) + tibia_struct + crank_plates),
    }


def clearance_sweep(leg, n_f=7, n_t=7):
    """Minimum distances between moving groups over the femur x tibia-angle x yaw grid."""
    pairs = clearance_pairs(leg)
    phis = np.linspace(FEMUR_RANGE[0], FEMUR_RANGE[1], n_f)
    taus = np.linspace(TAU_RANGE[0], TAU_RANGE[1], n_t)
    grid = {}
    t0 = time.time()
    for psi in YAW_CHECK:
        M = np.full((n_f, n_t), np.inf)
        worst = {}
        for i, phi in enumerate(phis):
            for k, tau in enumerate(taus):
                th = theta_of(phi, tau)
                if th < 0 or th > 180:                    # tibia through the femur line: not a pose
                    M[i, k] = 0.0
                    worst[(i, k)] = "knee interior angle outside 0..180"
                    continue
                A = leg.assemble(phi, tau, psi, ropes=True)
                dmin, who = np.inf, ""
                for pname, (a, b) in pairs.items():
                    d = compound(A, a).distance_to(compound(A, b))
                    if d < dmin:
                        dmin, who = d, pname
                M[i, k] = dmin
                worst[(i, k)] = who
        grid[psi] = (M, worst)
        print(f"  yaw {psi:+.0f}: min clearance {M.min():.1f} mm, {int((M < 3).sum())}/{M.size} poses blocked, {time.time() - t0:.0f} s")
    return phis, taus, grid, list(pairs)


# ---- rendering (numpy z-buffer, as cad/actuator/actuator.py) --------------------------------------
def _mesh(part, tol=1.0):
    v, t = part.tessellate(tol, 0.5)
    return np.array([(q.X, q.Y, q.Z) for q in v]), np.array(t, dtype=int)


def render(A, out, title, cam, right, size=(1100, 900), scale=None, centre=None, colour_of=None, annotate=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W, H = size
    cam = np.asarray(cam, float); cam /= np.linalg.norm(cam)
    right = np.asarray(right, float); right -= right.dot(cam) * cam; right /= np.linalg.norm(right)
    up = np.cross(cam, right)
    light = cam + 0.5 * up + 0.35 * right; light /= np.linalg.norm(light)
    pts_all = np.array([[p.X, p.Y, p.Z] for s, g in A.values() for p in (s.bounding_box().min, s.bounding_box().max)])
    if centre is None:
        centre = 0.5 * (pts_all.min(axis=0) + pts_all.max(axis=0))
    if scale is None:
        ext = pts_all - centre
        scale = 0.9 * min(W / (2 * np.abs(ext @ right).max() + 1), H / (2 * np.abs(ext @ up).max() + 1))
    zbuf = np.full((H, W), -1e9); img = np.ones((H, W, 3))
    for n, (s, g) in A.items():
        col = colour_of(n, g) if colour_of else COLORS[g]
        base = np.array(matplotlib.colors.to_rgb(col))
        v, t = _mesh(s)
        if len(t) == 0:
            continue
        tri = v[t] - centre
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        shade = 0.35 + 0.65 * np.abs(nrm @ light)
        sx = W / 2 + scale * (tri @ right); sy = H / 2 - scale * (tri @ up); sz = tri @ cam
        for k in range(len(tri)):
            x0, x1 = int(max(sx[k].min(), 0)), int(min(sx[k].max(), W - 1)) + 1
            y0, y1 = int(max(sy[k].min(), 0)), int(min(sy[k].max(), H - 1)) + 1
            if x1 <= x0 or y1 <= y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            (ax_, ay_), (bx_, by_), (cx_, cy_) = zip(sx[k], sy[k])
            det = (bx_ - ax_) * (cy_ - ay_) - (cx_ - ax_) * (by_ - ay_)
            if abs(det) < 1e-9:
                continue
            l1 = ((bx_ - gx) * (cy_ - gy) - (cx_ - gx) * (by_ - gy)) / det
            l2 = ((cx_ - gx) * (ay_ - gy) - (ax_ - gx) * (cy_ - gy)) / det
            l3 = 1 - l1 - l2
            inside = (l1 >= -1e-3) & (l2 >= -1e-3) & (l3 >= -1e-3)
            if not inside.any():
                continue
            depth = l1 * sz[k][0] + l2 * sz[k][1] + l3 * sz[k][2]
            sub = zbuf[y0:y1, x0:x1]
            upd = inside & (depth > sub)
            sub[upd] = depth[upd]
            img[y0:y1, x0:x1][upd] = np.clip(base * shade[k], 0, 1)
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.imshow(img); ax.set_axis_off(); ax.set_title(title, fontsize=10)

    def proj(p):
        p = np.asarray(p, float) - centre
        return W / 2 + scale * (p @ right), H / 2 - scale * (p @ up)
    if annotate:
        annotate(ax, proj)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)
    return proj


def colour_by_name(n, g):
    if n.startswith("rope"):
        return COLORS["rope"]
    if n.startswith("crank"):
        return COLORS["crank"]
    if "sector" in n:
        return COLORS["sector"]
    if n.endswith("pulley") and n not in ("pulley_yaw", "pulley_knee", "pulley_femur"):
        return COLORS["pulley"]
    if n.endswith("pulley_hub"):
        return "#6d6d6d"
    if n.startswith("drum"):
        return COLORS["drum"]
    if n.startswith("motor"):
        return COLORS["motor"]
    return COLORS[g]


def section_polys(part, x=0.0, tol=0.5):
    slab = Pos(x, 0, 0) * Box(0.02, 2000, 2000)
    try:
        sec = part & slab
    except Exception:
        return []
    polys = []
    for f in sec.faces():
        if abs(f.normal_at().X) < 0.9:
            continue
        for w in f.wires():
            pts = [w.position_at(t) for t in np.linspace(0, 1, 120, endpoint=False)]
            polys.append([(p.Y, p.Z) for p in pts])
    return polys


def draw_hip_section(A, g, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 9.5))
    for n, (s, grp) in A.items():
        if n.startswith("rope") or n.startswith("motor") or n.startswith("belt"):
            continue
        for poly in section_polys(s):
            ax.fill(*zip(*poly), color=colour_by_name(n, grp), lw=0.3, ec="k", alpha=0.9)
    # motors and belts as projected outlines (they are off the x = 0 plane)
    for j, (mx, my) in MOTOR_XY.items():
        zb = T_PLATE + LEVEL[j] * LEVEL_PITCH + MODULE_H
        ax.add_patch(plt.Rectangle((my - MOTOR_R, zb - MOTOR_H), 2 * MOTOR_R, MOTOR_H, fill=False, ls="--", lw=0.8, color=COLORS["motor"]))
        ax.text(my, zb - MOTOR_H / 2, f"8318\n{j}\nx = {mx:+.0f}", ha="center", va="center", fontsize=7, color=COLORS["motor"])
    # the off-plane sectors, crank and link pulleys as outlines
    for r, col, lab in ((R_SECTOR["femur"], COLORS["sector"], f"femur sector r {R_SECTOR['femur']:.0f} (x = ±{X_ROPE['femur']:.0f})"),
                        (R_SECTOR["knee"] + 0.5, COLORS["crank"], f"knee crank r {R_SECTOR['knee']:.0f} (x = ±{X_ROPE['knee']:.0f})"),
                        (R_LINK, COLORS["pulley"], f"link drive pulley r {R_LINK:.0f} (x = {0.5 * (X_LINK[0] + X_LINK[1]):.0f})")):
        ax.add_patch(plt.Circle((L_COXA, Z_PIVOT), r, fill=False, ls=":", lw=1.0, color=col))
    ax.add_patch(plt.Circle((g["knee"][0], g["knee"][1]), R_LINK, fill=False, ls=":", lw=1.0, color=COLORS["pulley"]))
    for name, (p, q) in g["runs"].items():
        ax.plot([p[1], q[1]], [p[2], q[2]], color=COLORS["rope"], lw=1.6, solid_capstyle="round", alpha=0.9)
        ax.text(0.5 * (p[1] + q[1]) + 6, 0.5 * (p[2] + q[2]), f"{name} (x = {p[0]:+.0f})", fontsize=6.5, color="#7a5c00")
    ax.axhline(0, color="#b03a2e", lw=0.6, ls="--")
    ax.text(-200, 3, "floor plate underside z = 0", fontsize=7, color="#b03a2e")
    labels = [(0, T_PLATE + 2 * LEVEL_PITCH + 20, "femur module (top): 25-lobe cycloid, output shaft Ø18 down the middle"),
              (0, T_PLATE + LEVEL_PITCH + 20, "knee module: 40-lobe, hollow eccentric (HK3512), output tube Ø30/22"),
              (0, T_PLATE + 20, "yaw module: 40-lobe, hollow eccentric (HK4012), RB7013 output carries the coxa"),
              (0, -32, f"knee drum r {R_DRUM['knee']:.0f} on the tube, HK4012 / HK3012 in the hub"),
              (0, -74, f"femur drum r {R_DRUM['femur']:.0f} on the shaft, HK2512 x 2"),
              (L_COXA, Z_PIVOT, f"femur pivot: Ø30 crank shaft in 2 x HK3012 (cheeks) + 2 x HK3012 (femur carrier);\nfemur sector r {R_SECTOR['femur']:.0f} at x = ±{X_ROPE['femur']:.0f}, knee crank r {R_SECTOR['knee']:.0f} at x = ±{X_ROPE['knee']:.0f}, drive pulley r {R_LINK:.0f} at x = +60"),
              (Y_SLAB_EDGE, SLAB_H / 2, "slab side face / rail"),
              (-52, -50, "coxa side plates 6 mm at x = ±46..52,\nhub plates 6 mm, standoffs inboard")]
    for k, (y, z, t) in enumerate(labels):
        ax.annotate(t, (y, z), (-300 if k % 2 else 330, 200 - 34 * k), fontsize=7, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#777", lw=0.5))
    ax.set_aspect("equal"); ax.set_xlim(-320, 560); ax.set_ylim(-300, 240)
    ax.set_xlabel("y, mm (outboard →)"); ax.set_ylabel("z, mm")
    ax.set_title("Hip section at x = 0 (through the yaw axis and the femur pivot): three stacked modules, nested shafts, two drums in the coxa hub;\n"
                 "off-plane sectors, crank and link pulleys dotted, rope runs projected", fontsize=9.5)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=105); plt.close(fig)


def draw_rom(leg, phis, taus, grid, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.7, 1])
    ax = fig.add_subplot(gs[0, :])
    # body slab and pod outline in the leg plane
    ax.add_patch(plt.Rectangle((-BODY_HALF_W - 250, 0), BODY_HALF_W + 250 + Y_SLAB_EDGE, SLAB_H, color="#dfe3e6", ec="#999"))
    ax.fill(*zip(*coxa_profile()), color=COLORS["coxa"], alpha=0.35)
    for j in ("knee", "femur"):
        z0, z1 = Z_DRUM[j]
        ax.add_patch(plt.Rectangle((-R_DRUM[j], z1), 2 * R_DRUM[j], z0 - z1, color=COLORS["drum"], alpha=0.5))
    gz = Z_PIVOT - HIP_HEIGHT
    ax.axhline(gz, color="#5a3e1b", lw=1.2); ax.text(-300, gz + 8, f"ground in the stance (femur axis {HIP_HEIGHT:.0f} mm up)", fontsize=8, color="#5a3e1b")
    for i, phi in enumerate(phis):
        for k, tau in enumerate(taus):
            th = theta_of(phi, tau)
            if th < 0 or th > 180:
                continue
            kn = PIV + L_FEMUR * yz_dir(phi)[1:]
            ft = kn + L_TIBIA * yz_dir(tau)[1:]
            ok = grid[0.0][0][i, k] >= 3.0
            ax.plot([PIV[0], kn[0]], [PIV[1], kn[1]], color="#0f9b8e" if ok else "#ccc", lw=0.6, alpha=0.6)
            ax.plot([kn[0], ft[0]], [kn[1], ft[1]], color="#2f6f9f" if ok else "#ddd", lw=0.5, alpha=0.5)
            ax.plot(ft[0], ft[1], ".", color="#2f6f9f" if ok else "#e0b0b0", ms=3)
    # the stance and mammal poses with ropes and sector arcs
    for pose, col, lab in ((STANCE, "#c0392b", "sprawl stance"), (MAMMAL, "#7d3c98", "mammal stance (yaw 90)")):
        phi, th = pose["femur_deg"], pose["knee_deg"]
        tau = tau_of(phi, th)
        g = rope_geometry(phi, tau)
        kn = g["knee"]
        ft = kn + L_TIBIA * yz_dir(tau)[1:]
        ax.plot([PIV[0], kn[0], ft[0]], [PIV[1], kn[1], ft[1]], color=col, lw=3, label=f"{lab}: femur {phi:.0f}°, knee {th:.0f}°, tibia {tau:.0f}°")
        for name, (p, q) in g["runs"].items():
            ax.plot([p[1], q[1]], [p[2], q[2]], color=COLORS["rope"], lw=1.6)
        for run in ("A", "B"):
            a0, a1 = GROOVE_F[run]
            th_ = np.radians(np.linspace(a0 + phi, a1 + phi, 40))
            ax.plot(PIV[0] + R_SECTOR["femur"] * np.cos(th_), PIV[1] + R_SECTOR["femur"] * np.sin(th_), color=COLORS["sector"], lw=2, alpha=0.8)
            a0, a1 = GROOVE_C[run]
            th_ = np.radians(np.linspace(a0 + tau, a1 + tau, 40))
            ax.plot(PIV[0] + (R_SECTOR["knee"] + 4) * np.cos(th_), PIV[1] + (R_SECTOR["knee"] + 4) * np.sin(th_), color=COLORS["crank"], lw=2, alpha=0.8)
        for c in (PIV, kn):
            ax.add_patch(plt.Circle((c[0], c[1]), R_LINK, fill=False, color=COLORS["pulley"], lw=1.0, ls=":"))
    ax.set_aspect("equal"); ax.set_xlim(-330, 900); ax.set_ylim(gz - 60, SLAB_H + 40)
    ax.set_xlabel("y, mm (outboard →); yaw 0"); ax.set_ylabel("z, mm")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Range of motion in the leg plane: femur {FEMUR_RANGE[0]:.0f}..{FEMUR_RANGE[1]:.0f}°, tibia absolute {TAU_RANGE[0]:.0f}..{TAU_RANGE[1]:.0f}° (the crank window) "
                 f"— grey = pose blocked by a self-collision at yaw 0; sector arcs (red femur, purple crank), link pulleys dotted, rope runs at the two named stances", fontsize=9)
    for c, psi in enumerate(YAW_CHECK):
        axm = fig.add_subplot(gs[1, c])
        M = grid[psi][0]
        axm.imshow(np.clip(M, 0, 40).T, origin="lower", extent=(phis[0], phis[-1], taus[0], taus[-1]), aspect="auto", cmap="RdYlGn", vmin=0, vmax=40)
        for i in range(len(phis)):
            for k in range(len(taus)):
                axm.text(phis[i], taus[k], f"{M[i, k]:.0f}", ha="center", va="center", fontsize=6, color="k" if M[i, k] >= 3 else "w")
        for th_lim in KNEE_LIMITS:
            axm.plot(phis, phis + 180 + th_lim, color="#555", lw=0.8, ls="--")
        axm.set_ylim(taus[0], taus[-1])
        axm.set_title(f"yaw {psi:+.0f}°: min clearance (mm)", fontsize=8)
        axm.set_xlabel("femur, deg")
        if c == 0:
            axm.set_ylabel("tibia absolute angle, deg (dashed: knee 15° / 145°)")
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


# ---- main ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    t_all = time.time()
    leg = Leg()
    TAU_STANCE = tau_of(STANCE["femur_deg"], STANCE["knee_deg"])
    A = leg.assemble(STANCE["femur_deg"], TAU_STANCE, STANCE["yaw_deg"])
    g_stance = rope_geometry(STANCE["femur_deg"], TAU_STANCE)

    # masses ----------------------------------------------------------------------------------------
    masses, by_group = {}, {gname: 0.0 for gname in GROUPS}
    for n, (s, grp) in A.items():
        m = part_mass(n, s)
        masses[n] = round(m, 1)
        by_group[grp] += m
    # one leg = everything below the floor plate + the three units; the body stub's plates/deck/rail are body structure
    unit_extra = {"belt_pulleys_x3": sum(masses[f"pulley_{j}"] + masses[f"belt_{j}"] for j in LEVEL),
                  "hollow_eccentric_and_bearing_upgrade_x3": 3 * 90.0,        # HK3512/HK4012 sleeves vs the BOM's solid Ø25 shaft on HK2512, estimate
                  "yaw_output_bearing_RB7013_vs_RB5013": 350.0 - 270.0}      # THK catalogue masses
    leg_structure = sum(m for n, m in masses.items() if A[n][1] not in ("body_stub",))
    leg_total = leg_structure + 3 * UNIT_MASS + sum(unit_extra.values())
    print("contacts (stance):", {k: round(v, 1) for k, v in g_stance["contact"].items()}, "grooves F", GROOVE_F, "C", GROOVE_C)
    print(f"leg structure (everything below the floor plate incl. shafts and ropes) {leg_structure / 1000:.2f} kg; "
          f"with three {UNIT_MASS / 1000:.2f} kg units and unit extras {sum(unit_extra.values()) / 1000:.2f}: {leg_total / 1000:.2f} kg per leg")

    # geometry record -------------------------------------------------------------------------------
    def L(p, q):
        return float(np.linalg.norm(np.asarray(q) - np.asarray(p)))
    run_len = {n: round(L(p, q), 1) for n, (p, q) in g_stance["runs"].items()}

    # rope lengths as functions of the joint angles (geometric, the drum end held): the coupling, by finite differences
    def sector_rope_len(j, run, phi, tau):
        g = rope_geometry(phi, tau)
        link_ang = phi if j == "femur" else tau
        groove = GROOVE_F if j == "femur" else GROOVE_C
        eye = (groove[run][0] if run == "A" else groove[run][1]) + link_ang
        wound = abs(unwrap_to(eye, g["contact"][f"{j}_{run}"]) - g["contact"][f"{j}_{run}"])
        return L(*g["runs"][f"{j}_{run}"]) + R_SECTOR[j] * math.radians(wound)

    def link_rope_len(run, phi, tau):
        g = rope_geometry(phi, tau)
        w = sum(link_wound(run, p, phi, tau)[0] for p in ("drive", "knee"))
        return L(*g["runs"][f"link_{run}"]) + R_LINK * math.radians(w)
    d = 1.0
    phi0, tau0 = STANCE["femur_deg"], TAU_STANCE
    coupling = {}
    for j in ("femur", "knee"):
        coupling[j] = dict(dL_dphi_mm_per_rad={r: round((sector_rope_len(j, r, phi0 + d, tau0) - sector_rope_len(j, r, phi0 - d, tau0)) / (2 * math.radians(d)), 2) for r in ("A", "B")},
                           dL_dtau_mm_per_rad={r: round((sector_rope_len(j, r, phi0, tau0 + d) - sector_rope_len(j, r, phi0, tau0 - d)) / (2 * math.radians(d)), 2) for r in ("A", "B")})
    coupling["femur"]["formula"] = "phi_femur = (r_drum/r_sector) * (delta_femur - psi) = (delta_femur - psi) / 4"
    coupling["knee"]["formula"] = "tau_tibia = (r_drum/r_crank) * (delta_knee - psi) = (delta_knee - psi) / 2.5; theta_knee = tau_tibia - phi_femur - 180"
    coupling["link"] = dict(dL_dphi_mm_per_rad={r: round((link_rope_len(r, phi0 + d, tau0) - link_rope_len(r, phi0 - d, tau0)) / (2 * math.radians(d)), 3) for r in ("ext", "int")},
                            dL_dtau_mm_per_rad={r: round((link_rope_len(r, phi0, tau0 + d) - link_rope_len(r, phi0, tau0 - d)) / (2 * math.radians(d)), 3) for r in ("ext", "int")})
    # total rope length over the whole reachable set (must be constant: capstan runs + wound arcs, both ends)
    var = {}
    for j in ("femur", "knee"):
        Ls = [sector_rope_len(j, "A", p, t) + sector_rope_len(j, "B", p, t) for p in np.linspace(*FEMUR_RANGE, 9) for t in np.linspace(*TAU_RANGE, 9)]
        var[j] = round(max(Ls) - min(Ls), 3)
    Ls = {r: [link_rope_len(r, p, t) for p in np.linspace(*FEMUR_RANGE, 9) for t in np.linspace(*TAU_RANGE, 9) if KNEE_LIMITS[0] <= theta_of(p, t) <= KNEE_LIMITS[1]] for r in ("ext", "int")}
    var["link_ext"], var["link_int"] = (round(max(Ls[r]) - min(Ls[r]), 3) for r in ("ext", "int"))
    coupling["loop_length_variation_over_range_mm"] = var
    link_wound_range = {r: {p: [round(min(link_wound(r, p, ph, t)[0] for ph in np.linspace(*FEMUR_RANGE, 9) for t in np.linspace(*TAU_RANGE, 9) if KNEE_LIMITS[0] <= theta_of(ph, t) <= KNEE_LIMITS[1]), 1),
                                round(max(link_wound(r, p, ph, t)[0] for ph in np.linspace(*FEMUR_RANGE, 9) for t in np.linspace(*TAU_RANGE, 9) if KNEE_LIMITS[0] <= theta_of(ph, t) <= KNEE_LIMITS[1]), 1)]
                            for p in ("drive", "knee")} for r in ("ext", "int")}
    drum_turns = dict(femur=(FEMUR_RANGE[1] - FEMUR_RANGE[0]) * CAPSTAN_RATIO["femur"] / 360, knee=(TAU_RANGE[1] - TAU_RANGE[0]) * CAPSTAN_RATIO["knee"] / 360)
    bands = {}
    for j in ("femur", "knee"):
        Ab, Bb, band, groove, walk = drum_band(j)
        bands[j] = dict(dead_wraps=N_DEAD_WRAPS, working_turns=round(drum_turns[j], 2), band_mm=round(band, 1), walk_mm=round(walk, 1), groove_mm=round(groove, 1),
                        needs_mm=round(band + walk, 1), fits=bool(band + walk <= groove + 0.05),
                        departure_from_mid_mm=dict(A=[round(v, 1) for v in Ab], B=[round(v, 1) for v in Bb]))
    rope_total = dict(femur=round(sum(run_len[n] for n in run_len if n.startswith("femur")) + math.radians(FEMUR_RANGE[1] - FEMUR_RANGE[0]) * R_SECTOR["femur"] + (3 + drum_turns["femur"]) * 2 * math.pi * R_DRUM["femur"], 0),
                      knee=round(sum(run_len[n] for n in run_len if n.startswith("knee")) + math.radians(TAU_RANGE[1] - TAU_RANGE[0]) * R_SECTOR["knee"] + (3 + drum_turns["knee"]) * 2 * math.pi * R_DRUM["knee"], 0),
                      link_each=round(L_FEMUR + math.radians(KNEE_LIMITS[1] - KNEE_LIMITS[0] + 2 * EYE_MARGIN) * R_LINK, 0))

    # clearance sweep -------------------------------------------------------------------------------
    if not QUICK:
        phis, taus, grid, pair_names = clearance_sweep(leg)
        clear = {}
        for psi, (M, worst) in grid.items():
            valid = np.array([[KNEE_LIMITS[0] <= theta_of(p, t) <= KNEE_LIMITS[1] for t in taus] for p in phis])
            ok = (M >= 3.0) & valid
            feas = M[ok]
            blocked = [dict(femur=float(phis[i]), tau=float(taus[k]), knee=round(theta_of(phis[i], taus[k]), 1), clearance=round(float(M[i, k]), 1), pair=worst[(i, k)])
                       for i in range(len(phis)) for k in range(len(taus)) if valid[i, k] and not ok[i, k]]
            clear[f"{psi:+.0f}"] = dict(min_over_feasible_poses_mm=round(float(feas.min()), 1) if feas.size else None, feasible=int(ok.sum()), within_knee_limits=int(valid.sum()), total=int(M.size),
                                        blocked=blocked, grid=[[round(float(v), 1) for v in row] for row in M])
        clear["axes"] = dict(femur_deg=[float(p) for p in phis], tau_deg=[float(t) for t in taus])
        draw_rom(leg, phis, taus, grid, os.path.join(FIG_DIR, "leg-rom.png"))
    else:
        clear, phis, taus = {"skipped": True}, None, None

    # static design gaps at the stance (parts that are adjacent by design) -------------------------
    def dist(a, b):
        return round(float(A[a][0].distance_to(A[b][0])), 2)
    static = {"femur_sector_A vs crank_sector_A (x gap)": dist("femur_sector_A", "crank_sector_A"),
              "femur_sector_A vs femur_carrier (x gap)": dist("femur_sector_A", "femur_carrier"),
              "crank_sector_A vs coxa_boss_R (x gap)": dist("crank_sector_A", "coxa_boss_R"),
              "crank_sector_A vs drum_knee": dist("crank_sector_A", "drum_knee"),
              "crank_sector_A vs hub_top": dist("crank_sector_A", "hub_top"),
              "drive_pulley vs coxa_plate_R (x gap)": dist("drive_pulley", "coxa_plate_R"),
              "drive_pulley_hub vs coxa_shaft_bearings (x gap)": dist("drive_pulley_hub", "coxa_shaft_bearings"),
              "crank_hub_B flange vs coxa_boss_L (x gap)": dist("crank_hub_B", "coxa_boss_L"),
              "crank_shaft flange vs coxa_boss_R (x gap)": dist("crank_shaft", "coxa_boss_R"),
              "knee_pulley vs knee_cheek_R (x gap)": dist("knee_pulley", "knee_cheek_R"),
              "rope_femur_A vs drum_knee": dist("rope_femur_A", "drum_knee"),
              "rope_femur_A vs hub_mid": dist("rope_femur_A", "hub_mid"),
              "rope_knee_A vs hub_top": dist("rope_knee_A", "hub_top"),
              "rope_knee_A vs floor_plate": dist("rope_knee_A", "floor_plate"),
              "rope_knee_B vs hub_bot": dist("rope_knee_B", "hub_bot"),
              "rope_link_ext vs femur_beam": dist("rope_link_ext", "femur_beam"),
              "rope_link_int vs tibia_tube (stance)": dist("rope_link_int", "tibia_tube"),
              "femur_sector_A vs hub_top (stance)": dist("femur_sector_A", "hub_top"),
              "femur_beam vs side_rail (stance)": dist("femur_beam", "side_rail"),
              "motor_yaw vs module_knee": dist("motor_yaw", "module_knee"),
              "motor_femur vs top_deck": dist("motor_femur", "top_deck")}
    stack_top = max(A[f"pulley_{j}"][0].bounding_box().max.Z for j in LEVEL)
    print("static gaps:", static, "stack top z", stack_top)

    # exports ----------------------------------------------------------------------------------------
    comp = Compound(children=[s for s, _ in A.values()])
    export_step(comp, os.path.join(OUT_DIR, "leg-stance.step"))
    for grp in GROUPS:
        names = [n for n, (s, gg) in A.items() if gg == grp]
        export_stl(Compound(children=[A[n][0] for n in names]), os.path.join(OUT_DIR, f"{grp}.stl"), tolerance=0.5, angular_tolerance=0.3)
    bb = comp.bounding_box()

    rec = dict(
        frame="leg frame: origin on the yaw axis at the floor plate underside; +x forward, +y outboard, +z up",
        stance=dict(**STANCE, tau_deg=TAU_STANCE), mammal=dict(**MAMMAL, tau_deg=tau_of(MAMMAL["femur_deg"], MAMMAL["knee_deg"])),
        femur_range_deg=FEMUR_RANGE, tau_range_deg=TAU_RANGE, knee_limits_deg=KNEE_LIMITS, yaw_checked_deg=YAW_CHECK,
        pivots=dict(yaw_axis=[0, 0, 0], femur_pivot=[0, L_COXA, Z_PIVOT], knee_stance=[0, float(g_stance["knee"][0]), float(g_stance["knee"][1])],
                    hip_height_mm=HIP_HEIGHT, floor_plate_above_ground_mm=HIP_HEIGHT - Z_PIVOT, pod_bottom_z=Z_HUB_BOT[1], coxa_plate_bottom_z=Z_PIVOT - 42),
        links=dict(coxa=L_COXA, femur=L_FEMUR, tibia=L_TIBIA, femur_beam=f"{BEAM_SECTION[0]:.0f}x{BEAM_SECTION[1]:.0f}x{BEAM_SECTION[2]:.0f} 6061-T6 rect tube, centreline {BEAM_V:.0f} mm exterior of the pivot-knee line",
                   femur_beam_section=BEAM_SECTION, tibia_tube=f"{TIBIA_SECTION[0]:.0f}x{TIBIA_SECTION[1]:.0f}x{TIBIA_SECTION[2]:.0f} 6061-T6 rect tube", tibia_section=TIBIA_SECTION,
                   coxa_structure="two 6 mm 6061 side plates at x = +-46..52 with HK3012 bosses + three 6 mm hub plates", pin_d=PIN_D, pin_bore=PIN_BORE),
        body=dict(slab_height=SLAB_H, slab_edge_from_yaw_axis=Y_SLAB_EDGE, slab_width=2 * (BODY_HALF_W + Y_SLAB_EDGE), level_pitch=LEVEL_PITCH,
                  module_stack_top_z=round(stack_top, 1), stack_fits=bool(stack_top <= SLAB_H - T_PLATE), motors_xy=MOTOR_XY, module_radius=MOD_R, lobes=LOBES),
        transmission=dict(r_drum=R_DRUM, r_sector=R_SECTOR, capstan_ratio=CAPSTAN_RATIO, total_ratio={j: LOBES[j] * CAPSTAN_RATIO.get(j, 1.0) for j in LOBES},
                          r_link=R_LINK, x_link=X_LINK, x_rope_planes=X_ROPE, drum_height=DRUM_H, rope=CAP["geometry"]["rope"],
                          contact_angles_stance_deg={k: round(v, 1) for k, v in g_stance["contact"].items()},
                          groove_femur_frame_deg={k: [round(a, 1) for a in v] for k, v in GROOVE_F.items()},
                          groove_crank_frame_deg={k: [round(a, 1) for a in v] for k, v in GROOVE_C.items()},
                          link_eyes_deg=LINK_EYES, link_wound_range_deg=link_wound_range,
                          run_lengths_stance_mm=run_len, rope_length_mm=rope_total, drum_working_turns=drum_turns, drum_bands=bands,
                          fleet_angle_deg=fleet_angles(g_stance),
                          fleet_angle_note="inclination of each straight run to its drum's plane of rotation (the drum turns about z, the sector about x); "
                                           "a grooved capstan drum wants < ~1.5 deg. The runs have no fleet in x (planes x = +-r_drum).",
                          coupling=coupling,
                          hubs="crank plate A on a flange integral with the crank shaft; crank plate B on a 42CrMo4 hub (bore 24, 2 x 8x7 keys x 26, OD 30 = the left journal); "
                               "link pulleys on 42CrMo4 hubs (2 x 10x8 keys x 24) with dowelled flanges; tibia carrier 2 x 10x8 keys x 50 in 6061",
                          belt="HTD 5M 1:1, 24T/24T, 15 mm"),
        masses_g=masses, group_mass_g={k: round(v, 1) for k, v in by_group.items()},
        unit_mass_g=UNIT_MASS, unit_extras_g={k: round(v, 1) for k, v in unit_extra.items()},
        leg_structure_g=round(leg_structure, 1), leg_total_g=round(leg_total, 1),
        assigned_masses=sorted(ASSIGNED),
        bearings=dict(crank_shaft_in_coxa="2 x HK3012 (PTI: Cr 11.5, C0r 17.3 kN) in bosses on the coxa cheeks", femur_carrier="2 x HK3012 on the crank shaft",
                      knee_pin_in_femur="2 x HK3012 in bosses on the femur's knee cheeks", femur_sectors="bronze bushes on the crank shaft (bore 16)",
                      knee_drum="HK4012 + HK3012 (PTI: C0r 23.1 / 17.3 kN)", femur_drum="2 x HK2512 (NTN: C0r 16.3 kN)",
                      yaw_output="RB7013 (THK 382-5E: C0 27.7 kN, dp 84 -> M0 1163 N·m)",
                      knee_eccentric="HK3512 (PTI: C0r 20.25 kN), hollow sleeve bore 22", yaw_eccentric="HK4012, hollow sleeve bore 32"),
        clearances=clear, static_gaps_mm=static, bbox=[round(bb.size.X), round(bb.size.Y), round(bb.size.Z)])
    json.dump(rec, open(os.path.join(ROOT, "cad", "leg", "leg.json"), "w"), indent=1)

    # renders ----------------------------------------------------------------------------------------
    t = time.time()
    render(A, os.path.join(FIG_DIR, "leg-iso.png"), "Leg assembly, sprawl stance (femur 45°, tibia vertical), from outboard-front-above: drums in the coxa hub, femur sector + knee crank at the pivot, 1:1 link loop along the femur",
           cam=(0.55, 0.6, 0.5), right=(-0.7, 0.7, 0), colour_of=colour_by_name)
    render(A, os.path.join(FIG_DIR, "leg-iso-hip.png"), "Hip from inboard-front-above: the three modules and their 8318 motors on the level plates inside the slab, the pod and the coxa below",
           cam=(0.6, -0.55, 0.45), right=(0.6, 0.8, 0), colour_of=colour_by_name)
    gz = Z_PIVOT - HIP_HEIGHT

    def side_notes(ax, proj):
        def dim(p, q, text, off=(0, 0), col="#b03a2e"):
            a, b = proj(p), proj(q)
            ax.annotate("", a, b, arrowprops=dict(arrowstyle="<->", color=col, lw=0.9))
            ax.text(0.5 * (a[0] + b[0]) + off[0], 0.5 * (a[1] + b[1]) + off[1], text, color=col, fontsize=8, ha="center", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
        dim((0, 0, -150), (0, L_COXA, -150), "coxa 150", off=(0, 12))
        kn = g_stance["knee"]
        dim((0, L_COXA, Z_PIVOT), (0, kn[0], kn[1]), "femur 250", off=(-40, -10))
        ft = (kn[0], kn[1] - L_TIBIA)
        dim((0, kn[0] + 110, kn[1]), (0, ft[0] + 110, ft[1]), "tibia 500", off=(30, 0))
        dim((0, -200, gz), (0, -200, Z_PIVOT), f"femur axis {HIP_HEIGHT:.0f} above ground", off=(-60, 0))
        dim((0, -260, gz), (0, -260, 0), f"floor plate {HIP_HEIGHT - Z_PIVOT:.0f}", off=(-40, 0))
        dim((0, -260, 0), (0, -260, SLAB_H), f"slab {SLAB_H:.0f}", off=(-30, 0))
        dim((0, L_COXA + 130, Z_PIVOT - R_SECTOR['femur']), (0, L_COXA + 130, Z_PIVOT + R_SECTOR['femur']), f"femur sector Ø{2 * R_SECTOR['femur']:.0f}\nknee crank Ø{2 * R_SECTOR['knee']:.0f}", off=(50, 0))
        dim((0, kn[0] + 90, kn[1] - R_LINK), (0, kn[0] + 90, kn[1] + R_LINK), f"link pulleys Ø{2 * R_LINK:.0f}", off=(45, 0))
        dim((0, -60, Z_HUB_BOT[1]), (0, -60, 0), f"pod {-Z_HUB_BOT[1]:.0f}", off=(-25, 0))
        dim((0, -120, Z_PIVOT), (0, -120, 0), f"pivot {-Z_PIVOT:.0f}", off=(-30, 0))
        a = proj((0, -400, gz)); b = proj((0, 700, gz))
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#5a3e1b", lw=1.5)
        ax.text(b[0], b[1] + 12, "ground (stance)", fontsize=8, color="#5a3e1b", ha="right")
    render(A, os.path.join(FIG_DIR, "leg-side.png"), "Side view along +x (leg plane), sprawl stance, dimensions in mm", cam=(1, 0, 0), right=(0, 1, 0), annotate=side_notes, colour_of=colour_by_name)
    A_nodeck = {n: v for n, v in A.items() if n not in ("top_deck", "level_plate_femur", "pulley_femur", "belt_femur")}
    render(A_nodeck, os.path.join(FIG_DIR, "leg-top.png"), "Top view (deck and the top level plate removed): three modules on the yaw axis, the 8318 motors around them; the leg below in the stance",
           cam=(0, 0, 1), right=(0, 1, 0), colour_of=colour_by_name)
    draw_hip_section(A, g_stance, os.path.join(FIG_DIR, "leg-section-hip.png"))
    print(f"renders {time.time() - t:.0f} s; total {time.time() - t_all:.0f} s")
    print(json.dumps({k: rec[k] for k in ("group_mass_g", "leg_structure_g", "leg_total_g", "static_gaps_mm")}, indent=1))
    print("coupling", json.dumps(coupling, indent=1))
    print("link wound ranges", json.dumps(link_wound_range))

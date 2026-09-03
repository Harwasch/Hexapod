#!/opt/hw-py/bin/python
"""Frameless-motor actuator CAD (round 14b, docs/design/08-actuator-design.md §9.17).

    /opt/hw-py/bin/python cad/actuator/frameless.py [--yaw]

The unit the frameless motor makes possible: a Wheemo-family kit motor as a
bare stator annulus bonded into the housing wall and a bare rotor ring on a
carrier, with the single-stage cycloid living *inside* the motor bore instead
of underneath it, and no capstan.  Numbers come from hw/stator/frameless_motor.json
("pick" for femur/knee, "yaw" for the yaw variant) and analysis/cycloid.py.

WHAT IS MODELLED AND WHAT IS ASSUMED
------------------------------------
* The motor is a *kit*: the datasheet gives terminal quantities and no
  geometry, so there are no teeth, slots, coils or magnet blocks in this model
  and there must not be — they cannot be known.  The stator is one plain
  annulus and the rotor one plain ring, each 7.1 mm of radial build about the
  Ø145.8 airgap, which is the 14.2 mm total build analysis/frameless_motor.py
  inferred from the 1039 g active mass.  Their densities are set so the pair
  still weighs that inferred active mass once a 0.5 mm mechanical airgap is
  carved out of the build (see ACTIVE_RHO below).
* INNER ROTOR is a *choice*.  The datasheet does not say inner- or outer-rotor.
  Inner-rotor is picked because it is what makes the architecture work: the
  stator, the loss-producing part, is then bonded straight to the housing wall
  and has the short conduction path to ambient that the whole thermal argument
  in §9.17 rests on, and the rotor ring is what runs on the cycloid's eccentric
  input shaft.  An outer-rotor kit of the same OD would put the heat inside a
  rotating ring and the architecture would have to change.
* NO MOTOR BEARINGS AND NO MOTOR CASE.  That is the point of a frameless kit:
  the rotor is carried by the reducer's own input shaft, on the 6905 in the
  output flange and the 6802 in the cover.
* The reducer is real geometry: the cycloid profile is generated from
  analysis/cycloid.py's parametrisation, the ring pins, output pins, discs,
  eccentric journals and clearances are all modelled and checked.

SIGN BUG IN analysis/cycloid.py::profile()  (do not silently inherit it)
-----------------------------------------------------------------------
cycloid.py offsets the epitrochoid OUTWARD by the pin radius, so its "disc"
comes out larger than its own ring-pin circle (radii 59.6-64.1 mm on a
R 59.3 pin circle) and cannot mesh — the discs in cad/actuator/actuator.py
interfere with the pin cage by ~3 mm as a result.  A cycloid disc is the
epitrochoid offset INWARD.  This script therefore uses its own `profile_in()`
with the sign corrected, and proves it by a meshing check
(check "cycloid_mesh_min_gap_mm" in the written JSON): the minimum distance
from every ring-pin centre to the disc profile, over a full input revolution,
is exactly the pin radius, i.e. contact with zero penetration.  Reported to the
parent session; logged in docs/design/friction-log.md.

Writes build/cad/frameless[-yaw]/ (STEP, per-group STL, per-group quarter-cut
STL), cad/actuator/frameless[-yaw].json and docs/design/actuator/frameless-cad-*.png.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from build123d import (Align, Box, Compound, Cylinder, Location, Part, Polygon, Pos,  # noqa: E402
                       export_step, export_stl, extrude)
import cycloid as cy  # noqa: E402

FM = json.load(open(os.path.join(ROOT, "hw", "stator", "frameless_motor.json")))
YAW = "--yaw" in sys.argv
TAG = "frameless-yaw" if YAW else "frameless"
FIG = os.path.join(ROOT, "docs", "design", "actuator")
OUT_DIR = os.path.join(ROOT, "build", "cad", TAG)

# ---------------------------------------------------------------- the motor --
if YAW:
    MOT = FM["yaw"]["motor"]
    N_LOBES = int(FM["pick"]["ratio_yaw"])                      # 30
    T_CYC_CONT = FM["pick"]["need"]["yaw"]                      # N·m at the joint, continuous
    T_CYC_PEAK = FM["requirement"]["c_peak_per_kg"]["yaw"] * FM["pick"]["m_robot"]
else:
    MOT = FM["pick"]["motor"]
    N_LOBES = int(FM["pick"]["ratio_fk"])                       # 25
    T_CYC_CONT = FM["pick"]["reducer"]["T_cyc_cont"]            # 214.3 N·m
    T_CYC_PEAK = FM["pick"]["reducer"]["T_cyc_peak"]            # 449.6 N·m

MOTOR_OD, MOTOR_L = MOT["od_mm"], MOT["len_mm"]
R_GAP = MOT["d_gap_mm"] / 2                                     # airgap radius
R_BORE = MOT["bore_mm"] / 2                                     # rotor inside radius
R_MOT_OUT = MOTOR_OD / 2                                        # stator outside radius
BUILD_TOT = 2 * (R_MOT_OUT - R_GAP)                             # 14.2 mm, split evenly
AIRGAP = 0.5                                                    # mechanical gap carved out of the build
R_STA_IN = R_GAP + AIRGAP / 2
R_ROT_OUT = R_GAP - AIRGAP / 2
ACTIVE_MASS_G = MOT["mass_kg"] * 1e3                            # 1039 g / 660 g, inferred in §9.17
RHO_EFF = FM["inferred"]["rho_eff"] / 1e6                       # g/mm^3, the inference's effective density
# split the inferred active mass by the nominal (gapless) annulus volumes, then
# raise each part's density so the modelled, gapped annuli still carry it
_V_STA_NOM = math.pi * (R_MOT_OUT**2 - R_GAP**2) * MOTOR_L
_V_ROT_NOM = math.pi * (R_GAP**2 - R_BORE**2) * MOTOR_L
M_STATOR_G = ACTIVE_MASS_G * _V_STA_NOM / (_V_STA_NOM + _V_ROT_NOM)
M_ROTOR_RING_G = ACTIVE_MASS_G - M_STATOR_G

# -------------------------------------------------------------- the reducer --
R_PINS = R_BORE - 4.0 - 2.5                     # the pin-circle rule §9.17 uses: 59.30 / 44.30
CYC = cy.design(N_LOBES, T_CYC_CONT, T_CYC_PEAK, R=R_PINS)
E_ECC, R_PIN = CYC["e"], CYC["r_pin"]
N_PINS = N_LOBES + 1
DISC_T, N_DISCS = cy.DISC_T, cy.N_DISCS         # 8 mm, two discs 180° apart


def profile_in(N, R, r, e, n=1440):
    """Cycloid disc profile: the epitrochoid of the N+1 pins on radius R,
    offset INWARD by the pin radius (analysis/cycloid.py offsets outward — see
    the module docstring)."""
    t = np.linspace(0, 2 * math.pi, n)
    x0 = R * np.cos(t) - e * np.cos((N + 1) * t)
    y0 = R * np.sin(t) - e * np.sin((N + 1) * t)
    dx = -R * np.sin(t) + e * (N + 1) * np.sin((N + 1) * t)
    dy = R * np.cos(t) - e * (N + 1) * np.cos((N + 1) * t)
    nrm = np.hypot(dx, dy)
    return x0 - r * dy / nrm, y0 + r * dx / nrm


_px, _py = profile_in(N_LOBES, R_PINS, R_PIN, E_ECC, n=2880)
_prad = np.hypot(_px, _py)
DISC_ROOT_R = float(_prad.min())                # smallest radius of the profile, disc frame
DISC_TIP_R = float(_prad.max())                 # largest radius of the profile, disc frame
DISC_SWEPT_R = DISC_TIP_R + E_ECC               # largest radius the disc reaches from the axis


def mesh_min_gap(n_theta=181):
    """Minimum distance from any ring-pin centre to the disc profile over one
    input revolution.  Equals the pin radius exactly when the profile is right."""
    P = np.stack(profile_in(N_LOBES, R_PINS, R_PIN, E_ECC, n=1440), 1)
    worst = 1e9
    for th in np.linspace(0, 2 * math.pi, n_theta):
        c, s = math.cos(-th / N_LOBES), math.sin(-th / N_LOBES)
        Q = P @ np.array([[c, s], [-s, c]]) + np.array([E_ECC * math.cos(th), E_ECC * math.sin(th)])
        for k in range(N_PINS):
            a = 2 * math.pi * k / N_PINS
            worst = min(worst, float(np.hypot(Q[:, 0] - R_PINS * math.cos(a), Q[:, 1] - R_PINS * math.sin(a)).min()))
    return worst


# ------------------------------------------------------------ the bearings ---
# Output crossed roller.  With the capstan deleted the output bearing carries the
# whole joint moment, so it is re-selected here against THK 382-5E (docs/reference).
# Static equivalent radial load P0 = 2M/dp (moment only); fs = C0/P0.
CRB_TABLE = {                       # model: (d, D, B, dp, C_kN, C0_kN, mass_kg)  -- THK 382-5E p.19
    "RB5013": (50, 80, 13, 64.0, 16.7, 20.9, 0.27),
    "RB6013": (60, 90, 13, 74.0, 18.0, 24.3, 0.30),
    "RB7013": (70, 100, 13, 84.0, 19.4, 27.7, 0.35),
    "RB8016": (80, 120, 16, 98.0, 30.1, 42.1, 0.70),
}
def _pick_crb(M_Nm, r_hub_min, fs_min=2.5):
    """Smallest crossed roller with room for the flange hub and a static safety
    factor of at least fs_min on the peak joint moment (THK 382-5E p.6:
    P0 = X0.Fr + Y0.Fa + 2M/dp; with Fr = Fa = 0 that is 2M/dp)."""
    for name, v in CRB_TABLE.items():
        if v[0] / 2 >= r_hub_min and v[5] * 1e3 / (2 * M_Nm * 1e3 / v[3]) >= fs_min:
            return name
    return list(CRB_TABLE)[-1]


CRB_NAME = _pick_crb(T_CYC_PEAK, 21.0 + 3.0)
CRB = CRB_TABLE[CRB_NAME]
R_OBRG_IN, R_OBRG_OUT, T_OBRG, CRB_DP = CRB[0] / 2, CRB[1] / 2, CRB[2], CRB[3]
M_OBRG_G = CRB[6] * 1e3

HK2512 = dict(name="HK2512", Fw=25.0, D=32.0, C=12.0, Cr_N=11800.0, C0r_N=16300.0, mass_g=20.0)  # NTN sheet
JOURNAL_R, ECC_BRG_OUT, ECC_BRG_W = HK2512["Fw"] / 2, HK2512["D"] / 2, HK2512["C"]
SHAFT_R = JOURNAL_R                                     # Ø25 straight stock, journals turned off-centre
SHAFT_BRG = dict(name="6905-2RS", d=25.0, D=42.0, B=9.0)        # no datasheet filed: mass modelled
TOP_BRG = dict(name="6802-2RS", d=15.0, D=24.0, B=5.0)          # no datasheet filed: mass modelled
SHAFT_BRG_OUT, SHAFT_BRG_W = SHAFT_BRG["D"] / 2, SHAFT_BRG["B"]
TOP_BRG_IN, TOP_BRG_OUT, TOP_BRG_W = TOP_BRG["d"] / 2, TOP_BRG["D"] / 2, TOP_BRG["B"]

# ------------------------------------------------------------ radial build ---
CLR = 0.5
CAGE_WALL = 1.2                                  # steel behind each ring pin
R_CAGE_IN = DISC_SWEPT_R + 0.3                   # the disc lobe tips sweep between the pins
R_CAGE_OUT = R_PINS + R_PIN + CAGE_WALL
SKIRT_T = 1.7                                    # rotor carrier skirt, between the cage and the rotor bore
R_SKIRT_OUT = R_BORE
R_SKIRT_IN = R_BORE - SKIRT_T
PIN_GROOVE_DEPTH = (R_PINS + R_PIN) - R_CAGE_IN  # how much of the Ø6 pin the cage actually holds
SKIRT_CLEARANCE = R_SKIRT_IN - R_CAGE_OUT

R_OD = R_MOT_OUT + 6.0                           # housing wall 6 mm on the stator OD
WALL_T = 6.0
N_BOLTS, BOLT_D, R_BOLTS = 12, 4.4, R_OD - 2.8

OUT_PIN_D, N_OUT = cy.OUT_PIN_D, cy.N_OUT        # Ø10, 8 pins
R_OUT_HOLE = OUT_PIN_D / 2 + E_ECC               # the disc's oversized hole
# analysis/cycloid.py fixes the output-pin circle at r 26, which was all the Ø100
# PCB-stator bore allowed.  This bore is Ø131.6 / Ø101.6, so the circle is moved
# out to the largest radius that leaves a 5 mm web to the disc root.  Both the
# cycloid.py value and this one are reported in the JSON.
R_OUT_PINS = math.floor(DISC_ROOT_R - E_ECC - R_OUT_HOLE - 5.0)
F_OUT_PIN = 4 * T_CYC_PEAK * 1000 / (N_OUT * R_OUT_PINS) / N_DISCS
F_OUT_PIN_CY = 4 * T_CYC_PEAK * 1000 / (N_OUT * cy.R_OUT) / N_DISCS
R_FLANGE = R_OUT_PINS + OUT_PIN_D / 2 + 3.0
R_FLANGE_BORE = ECC_BRG_OUT + E_ECC + 0.5      # the needle cup sweeps out to ECC_BRG_OUT + e
R_LIGHT1, LIGHT1_D = DISC_ROOT_R - E_ECC - 6.0 - 2.5, 12.0
_lo, _hi = ECC_BRG_OUT + E_ECC + 2.0, R_OUT_PINS - R_OUT_HOLE - 2.0    # room for an inner lightening ring
LIGHT2_D = min(10.0, _hi - _lo) if _hi - _lo >= 6.0 else 0.0
R_LIGHT2 = (_lo + _hi) / 2
N_OUT_BOLTS, R_OUT_BOLTS = 6, (R_OBRG_IN + ECC_BRG_OUT + 2.0) / 2

# -------------------------------------------------------------- axial build --
T_FLOOR = 6.0                                    # laser-cut 6061 floor plate
Z_OBRG0, Z_OBRG1 = 0.0, T_OBRG
Z_FLANGE0 = T_OBRG
T_FLANGE = 4.0
Z_FLANGE1 = Z_FLANGE0 + T_FLANGE
CUP_OVER = (ECC_BRG_W - DISC_T) / 2              # 2 mm of needle cup each side of its disc
DISC_PITCH = DISC_T + 2 * CUP_OVER               # 12 mm: neighbouring cups touch
Z_DISC0 = Z_FLANGE1 + 0.2
Z_DISCS = [Z_DISC0 + k * DISC_PITCH for k in range(N_DISCS)]
Z_CUP0 = Z_DISC0 - CUP_OVER
Z_CUP1 = Z_DISCS[-1] + DISC_T + CUP_OVER
Z_CAGE0, Z_CAGE1 = Z_CUP0 - 0.2, Z_CUP1
Z_MOT1 = Z_CAGE1                                 # motor top level with the reducer
Z_MOT0 = Z_MOT1 - MOTOR_L
Z_ROTCAR0 = Z_CAGE1 + CLR
T_ROTCAR = 4.5
Z_ROTCAR1 = Z_ROTCAR0 + T_ROTCAR
Z_COVER0 = Z_ROTCAR1 + CLR
T_COVER = 2.0
Z_BOSS1 = Z_COVER0 + TOP_BRG_W
H_TOTAL = Z_BOSS1
Z_SHAFT0 = Z_FLANGE0 - SHAFT_BRG_W               # shaft bottom, in the 6905
assert Z_MOT0 > T_FLOOR + CLR, (Z_MOT0, "stator hits the floor plate")

AL, STEEL = 2.70e-3, 7.85e-3                     # g/mm^3, 6061 / hardened steel
ECC = [(E_ECC * math.cos(math.pi * k), E_ECC * math.sin(math.pi * k)) for k in range(N_DISCS)]


def cyl(r, z0, z1, x=0.0, y=0.0):
    return Pos(x, y, z0) * Cylinder(r, z1 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN))


def ring(r_in, r_out, z0, z1):
    return cyl(r_out, z0, z1) - cyl(r_in, z0 - 1, z1 + 1)


# ---------------------------------------------------------------- the build --
def build():
    parts, mass = {}, {}

    # -- housing: laser-cut floor plate, a slice of tube for the wall, a turned
    #    bearing carrier, a stack of laser-cut pin-cage rings, a laser-cut cover
    floor = ring(R_OBRG_OUT, R_OD, 0, T_FLOOR)
    for k in range(N_BOLTS):
        a = 2 * math.pi * (k + 0.5) / N_BOLTS
        floor -= cyl(BOLT_D / 2, -1, T_FLOOR + 1, R_BOLTS * math.cos(a), R_BOLTS * math.sin(a))
    parts["floor_plate"] = floor; mass["floor_plate"] = floor.volume * AL

    wall = ring(R_OD - WALL_T, R_OD, T_FLOOR, Z_COVER0)
    for k in range(N_BOLTS):
        a = 2 * math.pi * (k + 0.5) / N_BOLTS
        wall -= cyl(BOLT_D / 2 - 0.5, T_FLOOR - 1, Z_COVER0 + 1, R_BOLTS * math.cos(a), R_BOLTS * math.sin(a))
    parts["wall_tube"] = wall; mass["wall_tube"] = wall.volume * AL

    carrier = ring(R_OBRG_OUT, R_CAGE_OUT, T_FLOOR, Z_CAGE0)      # turned 6061: crossed-roller seat + cage foot
    parts["bearing_carrier"] = carrier; mass["bearing_carrier"] = carrier.volume * AL

    cage = Part()                                                 # laser-cut 8 mm steel rings
    z0 = Z_CAGE0
    while z0 < Z_CAGE1 - 0.5:
        z1 = min(z0 + 8.0, Z_CAGE1)
        cage += ring(R_CAGE_IN, R_CAGE_OUT, z0, z1)
        z0 = z1
    for k in range(N_PINS):
        a = 2 * math.pi * k / N_PINS
        cage -= cyl(R_PIN + 0.03, Z_CAGE0 - 1, Z_CAGE1 + 1, R_PINS * math.cos(a), R_PINS * math.sin(a))
    parts["pin_cage"] = cage; mass["pin_cage"] = cage.volume * STEEL

    cover = cyl(R_OD, Z_COVER0, Z_COVER0 + T_COVER) - cyl(TOP_BRG_OUT, Z_COVER0 - 1, Z_BOSS1 + 1)
    cover += ring(TOP_BRG_OUT, TOP_BRG_OUT + 3.0, Z_COVER0 + T_COVER, Z_BOSS1)
    for k in range(N_BOLTS):
        a = 2 * math.pi * (k + 0.5) / N_BOLTS
        cover -= cyl(BOLT_D / 2, Z_COVER0 - 1, Z_BOSS1 + 1, R_BOLTS * math.cos(a), R_BOLTS * math.sin(a))
    parts["cover"] = cover; mass["cover"] = cover.volume * AL

    # -- the frameless kit: one plain annulus each, no teeth, no slots, no blocks
    sta = ring(R_STA_IN, R_MOT_OUT, Z_MOT0, Z_MOT1)
    parts["stator_annulus"] = sta; mass["stator_annulus"] = M_STATOR_G
    rot = ring(R_BORE, R_ROT_OUT, Z_MOT0, Z_MOT1)
    parts["rotor_ring"] = rot; mass["rotor_ring"] = M_ROTOR_RING_G

    # -- rotor carrier: a skirt through the annulus of space between the pin cage
    #    and the rotor bore, and a top disc with the hub on the shaft
    rc = ring(R_SKIRT_IN, R_SKIRT_OUT, Z_MOT0, Z_ROTCAR0)
    rc += ring(SHAFT_R, R_SKIRT_OUT, Z_ROTCAR0, Z_ROTCAR1)
    parts["rotor_carrier"] = rc; mass["rotor_carrier"] = rc.volume * AL

    # -- eccentric shaft: 6905 journal, one eccentric journal per disc, hub, stub
    shaft = cyl(SHAFT_R, Z_SHAFT0, Z_CUP0)
    for k, zd in enumerate(Z_DISCS):
        shaft += cyl(JOURNAL_R, zd - CUP_OVER, zd + DISC_T + CUP_OVER, *ECC[k])
    shaft += cyl(SHAFT_R, Z_CUP1, Z_ROTCAR1)
    shaft -= Pos(SHAFT_R - 2.0, -20, Z_ROTCAR0) * Box(10, 40, T_ROTCAR, align=(Align.MIN, Align.MIN, Align.MIN))
    shaft += cyl(TOP_BRG_IN, Z_ROTCAR1, Z_BOSS1)
    parts["shaft"] = shaft; mass["shaft"] = shaft.volume * STEEL

    # -- cycloid discs
    x, y = profile_in(N_LOBES, R_PINS, R_PIN, E_ECC, n=720)
    disc = Part()
    for k, zd in enumerate(Z_DISCS):
        ex, ey = ECC[k]
        pts = [(float(px) + ex, float(py) + ey) for px, py in zip(x[:-1], y[:-1])]
        d = Pos(0, 0, zd) * extrude(Polygon(*pts, align=None), amount=DISC_T)
        d -= cyl(ECC_BRG_OUT, zd - 1, zd + DISC_T + 1, ex, ey)
        for i in range(N_OUT):
            a = 2 * math.pi * i / N_OUT
            d -= cyl(R_OUT_HOLE, zd - 1, zd + DISC_T + 1, R_OUT_PINS * math.cos(a) + ex, R_OUT_PINS * math.sin(a) + ey)
            b = a + math.pi / N_OUT
            d -= cyl(LIGHT1_D / 2, zd - 1, zd + DISC_T + 1, R_LIGHT1 * math.cos(b) + ex, R_LIGHT1 * math.sin(b) + ey)
            if LIGHT2_D > 0:
                d -= cyl(LIGHT2_D / 2, zd - 1, zd + DISC_T + 1, R_LIGHT2 * math.cos(b) + ex, R_LIGHT2 * math.sin(b) + ey)
        disc += d
    parts["cycloid_discs"] = disc; mass["cycloid_discs"] = disc.volume * STEEL

    pins = Part()
    for k in range(N_PINS):
        a = 2 * math.pi * k / N_PINS
        pins += cyl(R_PIN, Z_CAGE0, Z_CAGE1, R_PINS * math.cos(a), R_PINS * math.sin(a))
    parts["ring_pins"] = pins; mass["ring_pins"] = pins.volume * STEEL

    # -- output flange: turned hub inside the crossed roller + laser-cut plate
    flange = ring(ECC_BRG_OUT + 1.5, R_OBRG_IN, 0, Z_FLANGE0)
    flange -= cyl(SHAFT_BRG_OUT, Z_SHAFT0, Z_FLANGE0 + 0.5)              # 6905 pocket, bored from above
    flange += ring(R_FLANGE_BORE, R_FLANGE, Z_FLANGE0, Z_FLANGE1)        # plate clears the swept needle cup
    for k in range(N_OUT_BOLTS):
        a = 2 * math.pi * k / N_OUT_BOLTS
        flange -= cyl(3.3 / 2, -1, 6, R_OUT_BOLTS * math.cos(a), R_OUT_BOLTS * math.sin(a))
    opins = Part()
    for k in range(N_OUT):
        a = 2 * math.pi * k / N_OUT
        opins += cyl(OUT_PIN_D / 2, Z_FLANGE0, Z_DISCS[-1] + DISC_T + 0.8,
                     R_OUT_PINS * math.cos(a), R_OUT_PINS * math.sin(a))
        flange -= cyl(OUT_PIN_D / 2, Z_FLANGE0 - 0.5, Z_FLANGE1 + 1, R_OUT_PINS * math.cos(a), R_OUT_PINS * math.sin(a))
    parts["output_flange"] = flange; mass["output_flange"] = flange.volume * AL
    parts["output_pins"] = opins; mass["output_pins"] = opins.volume * STEEL

    # -- bearings.  Catalogue mass where a datasheet is filed; modelled otherwise.
    ob = ring(R_OBRG_IN, R_OBRG_OUT, Z_OBRG0, Z_OBRG1)
    parts["output_bearing"] = ob; mass["output_bearing"] = M_OBRG_G
    eb = Part()
    for k, zd in enumerate(Z_DISCS):
        eb += ring(JOURNAL_R, ECC_BRG_OUT, zd - CUP_OVER, zd + DISC_T + CUP_OVER).moved(Location((ECC[k][0], ECC[k][1], 0)))
    parts["eccentric_bearings"] = eb; mass["eccentric_bearings"] = HK2512["mass_g"] * N_DISCS
    sb = ring(SHAFT_R, SHAFT_BRG_OUT, Z_SHAFT0, Z_FLANGE0)
    sb += ring(TOP_BRG_IN, TOP_BRG_OUT, Z_COVER0, Z_BOSS1)
    parts["shaft_bearings"] = sb; mass["shaft_bearings"] = sb.volume * STEEL * 0.8
    return parts, mass


GROUPS = {"housing": ["floor_plate", "wall_tube", "bearing_carrier", "pin_cage", "cover"],
          "stator": ["stator_annulus"],
          "magnets": ["rotor_ring"],                    # the kit's rotor ring: back iron + magnets, inferred as one annulus
          "rotor": ["rotor_carrier", "shaft"],
          "reducer": ["cycloid_discs", "ring_pins", "output_flange", "output_pins"],
          "bearings": ["output_bearing", "eccentric_bearings", "shaft_bearings"]}
COLORS = {"housing": "#9aa5ad", "stator": "#0f9b8e", "magnets": "#c0392b",
          "rotor": "#d98c3a", "reducer": "#3a3a3a", "bearings": "#e0e0e0"}
GROUP_OF = {n: g for g, ns in GROUPS.items() for n in ns}
PRETTY = {"floor_plate": "Floor plate — laser-cut 6 mm 6061 (BOM: Floor plate)",
          "wall_tube": "Wall tube — 6061 tube slice, stator bonded in its bore (BOM: Wall tube)",
          "bearing_carrier": "Bearing carrier — turned 6061 (BOM: Bearing carrier)",
          "pin_cage": "Pin cage — laser-cut 8 mm steel rings (BOM: Pin cage)",
          "cover": "Cover — laser-cut 2 mm 6061 + turned 6802 boss (BOM: Cover)",
          "stator_annulus": "Frameless STATOR annulus — kit part, bonded into the wall",
          "rotor_ring": "Frameless ROTOR ring — kit part, bonded on the carrier skirt",
          "rotor_carrier": "Rotor carrier — skirt + top disc, turned/laser-cut 6061 (new line)",
          "shaft": "Eccentric shaft — turned 42CrMo4 (BOM: Eccentric shaft)",
          "cycloid_discs": "Cycloid discs ×2 — laser-cut 8 mm 42CrMo4, hardened (BOM: Cycloid disc)",
          "ring_pins": "Ring pins — dowel Ø6 hardened (BOM: Ring pins)",
          "output_flange": "Output flange — turned hub + laser-cut plate (BOM: Output flange)",
          "output_pins": "Output pins — dowel Ø10 hardened (BOM: Output pins)",
          "output_bearing": "Output bearing — crossed roller (BOM: Output bearing)",
          "eccentric_bearings": "Eccentric bearings ×2 — HK2512 (BOM: Eccentric bearing)",
          "shaft_bearings": "Shaft bearings — 6905 + 6802 (BOM: Shaft bearing lower/top)"}


# ------------------------------------------------------------- the renderer --
def _mesh(part, tol=0.7):
    v, t = part.tessellate(tol, 0.35)
    return np.array([(q.X, q.Y, q.Z) for q in v]), np.array(t, dtype=int)


def raster(items, size, scale, centre, azim=-50, elev=28):
    """Orthographic z-buffer raster.  items: list of (part, rgb).  Returns the
    image and a projector so labels can be placed on real 3D points."""
    import matplotlib
    W, H = size
    az, el = math.radians(azim), math.radians(elev)
    cam = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.cross([0, 0, 1], cam); right /= np.linalg.norm(right)
    up = np.cross(cam, right)
    light = cam + 0.6 * up + 0.3 * right; light /= np.linalg.norm(light)
    c = np.asarray(centre, float)

    def project(p):
        q = np.asarray(p, float) - c
        return W / 2 + scale * (q @ right), H / 2 - scale * (q @ up)

    zbuf = np.full((H, W), -1e9)
    img = np.ones((H, W, 3))
    for part, col in items:
        base = np.array(matplotlib.colors.to_rgb(col))
        v, t = _mesh(part)
        if len(t) == 0:
            continue
        tri = v[t]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        shade = 0.32 + 0.68 * np.clip(np.abs(nrm @ light), 0, 1)
        q = tri - c
        sx = W / 2 + scale * (q @ right); sy = H / 2 - scale * (q @ up); sz = q @ cam
        for k in range(len(tri)):
            x0, x1 = int(max(sx[k].min(), 0)), int(min(sx[k].max(), W - 1)) + 1
            y0, y1 = int(max(sy[k].min(), 0)), int(min(sy[k].max(), H - 1)) + 1
            if x1 <= x0 or y1 <= y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            (ax_, ay_), (bx_, by_), (cx_, cy_) = list(zip(sx[k], sy[k]))
            det = (bx_ - ax_) * (cy_ - ay_) - (cx_ - ax_) * (by_ - ay_)
            if abs(det) < 1e-9:
                continue
            l1 = ((bx_ - gx) * (cy_ - gy) - (cx_ - gx) * (by_ - gy)) / det
            l2 = ((cx_ - gx) * (ay_ - gy) - (ax_ - gx) * (cy_ - gy)) / det
            l3 = 1 - l1 - l2
            inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            if not inside.any():
                continue
            depth = l1 * sz[k][0] + l2 * sz[k][1] + l3 * sz[k][2]
            sub = zbuf[y0:y1, x0:x1]
            upd = inside & (depth > sub)
            sub[upd] = depth[upd]
            img[y0:y1, x0:x1][upd] = np.clip(base * shade[k], 0, 1)
    return img, project


CUTTER = Box(400, 400, 400, align=(Align.MIN, Align.MIN, Align.MIN)).moved(Location((0, -400, -100)))


def cut_part(p):
    """Quarter cut.  OCCT occasionally leaves a stray lower-dimensional shape in a
    part after a boolean, and build123d then refuses the next subtraction with
    "Dimensions of objects to subtract from are inconsistent"; rebuilding from the
    solids alone clears it."""
    try:
        return p - CUTTER
    except ValueError:
        return Part(Compound(children=list(p.solids())).wrapped) - CUTTER


def group_items(parts, cut=False):
    out = []
    for g, names in GROUPS.items():
        for n in names:
            if n in parts:
                out.append((cut_part(parts[n]) if cut else parts[n], COLORS[g]))
    return out


def section_polys(part, half=False):
    """Outline polygons of the part's section by the XZ plane (y = 0)."""
    slab = Box(600, 0.02, 300, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(Location((0, 0, 60)))
    sec = part & slab
    polys = []
    for f in sec.faces():
        if abs(f.normal_at().Y) < 0.9:
            continue
        for w in f.wires():
            pts = [w.position_at(t) for t in np.linspace(0, 1, 200, endpoint=False)]
            poly = [(p.X, p.Z) for p in pts]
            if half and max(p[0] for p in poly) < 0:
                continue
            polys.append(poly)
    return polys


# ------------------------------------------------------------ the drawings --
def _dim_h(ax, x0, x1, y, text, col="#b03a2e", fs=8.5, tick=None, va="top", dy=-1.4):
    ax.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<|-|>", color=col, lw=0.8,
                                                      mutation_scale=8, shrinkA=0, shrinkB=0))
    if tick is not None:
        for x in (x0, x1):
            ax.plot([x, x], [tick, y], color=col, lw=0.5, ls=(0, (4, 3)))
    ax.text((x0 + x1) / 2, y + dy, text, ha="center", va=va, color=col, fontsize=fs,
            bbox=dict(fc="white", ec="none", pad=0.6))


def _dim_v(ax, y0, y1, x, text, col="#b03a2e", fs=8.5, tick=None, ha="left", dx=1.6):
    ax.annotate("", (x, y0), (x, y1), arrowprops=dict(arrowstyle="<|-|>", color=col, lw=0.8,
                                                      mutation_scale=8, shrinkA=0, shrinkB=0))
    if tick is not None:
        for y in (y0, y1):
            ax.plot([tick, x], [y, y], color=col, lw=0.5, ls=(0, (4, 3)))
    ax.text(x + dx, (y0 + y1) / 2, text, ha=ha, va="center", color=col, fontsize=fs, rotation=90,
            bbox=dict(fc="white", ec="none", pad=0.6))


def draw_section(parts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    XL, XR = -R_OD - 32, R_OD + 32
    dims = [(R_OD, f"\u00d8{2*R_OD:.0f}  housing OD"),
            (R_MOT_OUT, f"\u00d8{MOTOR_OD:.1f}  motor OD = stator OD"),
            (R_GAP, f"\u00d8{2*R_GAP:.1f}  airgap"),
            (R_BORE, f"\u00d8{2*R_BORE:.1f}  motor bore = rotor ID"),
            (R_PINS, f"\u00d8{2*R_PINS:.1f}  cycloid ring-pin circle ({N_PINS} \u00d7 \u00d8{2*R_PIN:.0f}, pitch {CYC['pitch']:.2f})"),
            (R_OUT_PINS, f"\u00d8{2*R_OUT_PINS:.0f}  output-pin circle ({N_OUT} \u00d7 \u00d8{OUT_PIN_D:.0f})"),
            (R_OBRG_IN, f"\u00d8{2*R_OBRG_IN:.0f}  {CRB_NAME} crossed-roller bore"),
            (ECC_BRG_OUT, f"\u00d8{2*ECC_BRG_OUT:.0f} / \u00d8{2*JOURNAL_R:.0f}  HK2512 eccentric bearing")]
    Y_BOT = -5.0 - 5.6 * len(dims) - 4
    Y_TOP = H_TOTAL + 14 + 11 * 5
    fig, ax = plt.subplots(figsize=(14.0, 14.0 * (Y_TOP - Y_BOT) / (XR - XL) + 0.55))
    for g, names in GROUPS.items():
        for n in names:
            if n not in parts:
                continue
            for poly in section_polys(parts[n]):
                ax.fill(*zip(*poly), color=COLORS[g], lw=0.45, ec="#1a1a1a",
                        alpha=1.0 if g != "bearings" else 0.9, zorder=3)
    ax.axvline(0, color="#555", lw=0.7, ls=(0, (8, 3, 1, 3)), zorder=1)
    ax.plot([XL + 2, XR - 2], [0, 0], color="#b03a2e", lw=1.0, ls=(0, (6, 3)), zorder=2)
    ax.text(XR - 2, -2.2, "MOUNTING / OUTPUT FACE   z = 0", color="#b03a2e", fontsize=9, va="top", ha="right")

    y = -5.0
    for r, txt in dims:
        _dim_h(ax, -r, r, y, txt, tick=0.0)
        y -= 5.6

    _dim_v(ax, 0, H_TOTAL, R_OD + 8, f"{H_TOTAL:.1f}  overall height", tick=R_OD)
    _dim_v(ax, Z_MOT0, Z_MOT1, R_OD + 18, f"{MOTOR_L:.0f}  motor stack", tick=R_MOT_OUT)
    _dim_v(ax, Z_CAGE0, Z_CAGE1, R_OD + 28, f"{Z_CAGE1-Z_CAGE0:.1f}  ring pins", tick=R_CAGE_OUT)
    _dim_v(ax, Z_DISCS[0], Z_DISCS[0] + DISC_T, -R_OD - 8, f"{DISC_T:.0f}  disc", tick=-R_PINS, ha="right", dx=-1.8)
    _dim_v(ax, Z_DISCS[0], Z_DISCS[1], -R_OD - 18, f"{DISC_PITCH:.0f}  disc pitch", tick=-R_PINS, ha="right", dx=-1.8)
    _dim_v(ax, Z_OBRG0, Z_OBRG1, -R_OD - 28, f"{T_OBRG:.0f}  {CRB_NAME}", tick=-R_OBRG_OUT, ha="right", dx=-1.8)

    right = [((R_MOT_OUT + R_STA_IN) / 2, (Z_MOT0 + Z_MOT1) / 2,
              f"FRAMELESS STATOR annulus, {(R_MOT_OUT-R_STA_IN):.2f} mm radial build, bonded/shrunk\n"
              f"into the wall bore. No motor case: the wall IS the stator carrier."),
             (R_OD - 3, (Z_MOT1 + Z_COVER0) / 2, f"wall tube \u00d8{2*R_OD:.0f} \u00d7 {WALL_T:.0f} mm 6061, {N_BOLTS} \u00d7 M4 through the cover"),
             (R_OBRG_OUT - 3, T_OBRG / 2,
              f"{CRB_NAME} crossed roller {CRB[0]:.0f}\u00d7{CRB[1]:.0f}\u00d7{CRB[2]:.0f}, C0 {CRB[5]:.1f} kN, dp {CRB_DP:.0f} \u2014 with the\n"
              f"capstan deleted it carries the whole joint moment (fs {CRB[5]*1e3/(2*T_CYC_PEAK*1e3/CRB_DP):.2f})"),
             (R_FLANGE - 5, (Z_FLANGE0 + Z_FLANGE1) / 2, "output flange: turned hub in the bearing bore +\nlaser-cut plate; output exits through z = 0"),
             (SHAFT_BRG_OUT, Z_FLANGE0 - SHAFT_BRG_W / 2, "6905 in the output-flange hub")]
    left = [(-(R_BORE + R_ROT_OUT) / 2, (Z_MOT0 + Z_MOT1) / 2,
             f"FRAMELESS ROTOR ring, {(R_ROT_OUT-R_BORE):.2f} mm build, bonded on the\ncarrier skirt. No motor bearings: it runs on the cycloid's own\neccentric input shaft."),
            (-(R_SKIRT_IN + R_SKIRT_OUT) / 2, Z_MOT0 + 6,
             f"rotor carrier skirt {SKIRT_T:.1f} mm \u2014 threads the {R_BORE-R_PINS-R_PIN:.1f} mm between the\nring pins and the motor bore, {SKIRT_CLEARANCE:.1f} mm clear of the cage"),
            (-R_PINS, Z_DISCS[1] + DISC_T - 2, f"{N_PINS} ring pins \u00d8{2*R_PIN:.0f} in laser-cut cage rings;\nonly {CAGE_WALL:.1f} mm of steel behind each pin"),
            (-R_OUT_PINS, Z_DISCS[0] + DISC_T / 2,
             f"{N_DISCS} cycloid discs, {N_LOBES}:1, {DISC_T:.0f} mm, 180\u00b0 apart on HK2512,\ne = {E_ECC:.2f} mm; profile offset INWARD (see the docstring)"),
            (-TOP_BRG_OUT, Z_BOSS1 - 2.5, "6802 in the cover boss")]
    zr = H_TOTAL + 11 * 5
    for x, z, t in sorted(right, key=lambda e: -e[1]):
        ax.annotate(t, (x, z), (XR - 2, zr), fontsize=8.0, ha="right", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#666", lw=0.6, connectionstyle="arc3,rad=0.06"))
        zr -= 11
    zl = H_TOTAL + 11 * 5
    for x, z, t in sorted(left, key=lambda e: -e[1]):
        ax.annotate(t, (x, z), (XL + 2, zl), fontsize=8.0, ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#666", lw=0.6, connectionstyle="arc3,rad=-0.06"))
        zl -= 11
    ax.set_aspect("equal")
    ax.set_xlim(XL, XR); ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_xlabel("mm"); ax.set_ylabel("mm above the mounting face")
    ax.grid(alpha=0.13, zorder=0)
    ttl = ("yaw" if YAW else "femur / knee")
    ax.set_title(f"Frameless-motor actuator, {ttl} variant \u2014 section through the axis, from the build123d model\n"
                 f"\u00d8{2*R_OD:.0f} \u00d7 {H_TOTAL:.1f} mm  \u00b7  motor \u00d8{MOTOR_OD:.0f} \u00d7 {MOTOR_L:.0f} frameless kit  \u00b7  {N_LOBES}:1 cycloid inside the bore  \u00b7  no capstan",
                 fontsize=11.5)
    hand = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[g], ec="#1a1a1a", lw=0.4) for g in GROUPS]
    ax.legend(hand, ["housing", "stator (kit)", "rotor ring (kit)", "rotor carrier + shaft", "reducer", "bearings"],
              fontsize=8.5, ncol=6, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(out, dpi=100, facecolor="white"); plt.close(fig)


def auto_fit(parts_list, size, azim, elev, fill=0.95):
    """Scale and camera centre that fit every part into `size` with `fill` margin.
    Every part here is a body of revolution about z, so its silhouette is exactly a
    radius and a z range; using bounding-*box* corners would over-estimate the width
    by up to sqrt(2) and shrink the render."""
    az, el = math.radians(azim), math.radians(elev)
    cam = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.cross([0, 0, 1], cam); right /= np.linalg.norm(right)
    up = np.cross(cam, right)
    h_up = math.hypot(up[0], up[1])
    us, vs = [], []
    for part in parts_list:
        bb = part.bounding_box()
        r = max(abs(bb.min.X), abs(bb.max.X), abs(bb.min.Y), abs(bb.max.Y))
        us += [-r, r]
        vs += [bb.min.Z * up[2] - r * h_up, bb.max.Z * up[2] + r * h_up]
    umin, umax, vmin, vmax = min(us), max(us), min(vs), max(vs)
    scale = min(fill * size[0] / max(umax - umin, 1e-6), fill * size[1] / max(vmax - vmin, 1e-6))
    return scale, (umin + umax) / 2 * right + (vmin + vmax) / 2 * up


def draw_iso(parts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    whole = group_items(parts)
    cutw = group_items(parts, cut=True)
    plist = [p for p, _ in whole]

    PW, PH, BW, BH = 690, 500, 1380, 700
    views = []
    for azim, elev in ((-52, 26), (-52, -32)):
        sc, c = auto_fit(plist, (PW, PH), azim, elev)
        views.append(raster(whole, (PW, PH), sc, c, azim=azim, elev=elev)[0])
    sc, c = auto_fit(plist, (int(BW * 0.70), BH), -46, 24)
    big, project = raster(cutw, (BW, BH), sc, c, azim=-46, elev=24)

    fig = plt.figure(figsize=(14.0, (PH + BH) / 100 + 1.1))
    gs = GridSpec(2, 2, height_ratios=(PH, BH), hspace=0.10, wspace=0.03,
                  left=0.01, right=0.99, top=0.93, bottom=0.01)
    for k, (im, t) in enumerate(zip(views, (f"from above \u2014 the whole unit, \u00d8{2*R_OD:.0f} \u00d7 {H_TOTAL:.1f} mm",
                                            "from below \u2014 the mounting face and the output flange at z = 0"))):
        a = fig.add_subplot(gs[0, k]); a.imshow(im); a.set_axis_off(); a.set_title(t, fontsize=9.5)
    a = fig.add_subplot(gs[1, :]); a.imshow(big); a.set_xlim(0, BW); a.set_ylim(BH, 0); a.set_axis_off()
    a.set_title("quarter cutaway \u2014 the reducer sits inside the motor bore, not underneath it", fontsize=9.5)
    # the quarter cut removes x > 0, y < 0, so the two visible cut faces are
    # {y = 0, x > 0} and {x = 0, y < 0}; anchoring on one or the other puts a
    # label on the right or the left of the render without crossing leaders.
    fa = lambda r, z: (r, 0.0, z)          # right-hand cut face  # noqa: E731
    fb = lambda r, z: (0.0, -r, z)         # left-hand cut face   # noqa: E731
    tips = [(fa((R_MOT_OUT + R_STA_IN) / 2, (Z_MOT0 + Z_MOT1) / 2), "frameless STATOR annulus \u2014 no motor case"),
            (fa((R_BORE + R_ROT_OUT) / 2, (Z_MOT0 + Z_MOT1) / 2), "frameless ROTOR ring \u2014 no motor bearings"),
            (fa((R_SKIRT_IN + R_SKIRT_OUT) / 2, Z_MOT0 + 6), "rotor carrier skirt, %.1f mm" % SKIRT_T),
            (fa(R_OD - 3, (Z_MOT1 + Z_COVER0) / 2), f"wall tube, {N_BOLTS} \u00d7 M4"),
            (fa(TOP_BRG_OUT, Z_BOSS1 - 2), "6802 in the cover boss"),
            (fb(R_PINS, Z_DISCS[1] + DISC_T / 2), f"{N_PINS} ring pins in the laser-cut cage"),
            (fb(R_OUT_PINS - 8, Z_DISCS[0] + DISC_T / 2), f"{N_DISCS} cycloid discs, {N_LOBES}:1, 180\u00b0 apart"),
            (fb(R_OUT_PINS, Z_FLANGE1 + 4), f"{N_OUT} output pins \u00d8{OUT_PIN_D:.0f}"),
            (fb((R_OBRG_IN + R_OBRG_OUT) / 2, T_OBRG / 2), f"{CRB_NAME} crossed roller"),
            (fb(R_OBRG_IN - 6, Z_FLANGE0 - SHAFT_BRG_W / 2), "6905 in the output-flange hub")]
    proj = [(project(pt), txt) for pt, txt in tips]
    for side in (0, 1):
        sel = [(xy, t) for xy, t in proj if (xy[0] < BW / 2) == (side == 0)]
        sel.sort(key=lambda e: e[0][1])
        for k, ((sx, sy), txt) in enumerate(sel):
            ty = 34 + (BH - 68) * (k + 0.5) / max(len(sel), 1)
            tx = 14 if side == 0 else BW - 14
            a.annotate(txt, (sx, sy), (tx, ty), fontsize=8.6, ha="left" if side == 0 else "right",
                       va="center", arrowprops=dict(arrowstyle="-", color="#555", lw=0.6),
                       bbox=dict(fc="white", ec="#ccc", lw=0.4, pad=1.8, alpha=0.94))
    fig.suptitle(f"Frameless-motor actuator ({'yaw' if YAW else 'femur / knee'}) \u2014 motor \u00d8{MOTOR_OD:.0f} \u00d7 {MOTOR_L:.0f} frameless kit, "
                 f"{N_LOBES}:1 cycloid inside the \u00d8{2*R_BORE:.0f} bore, no capstan, no motor case and no motor bearings",
                 fontsize=12)
    fig.savefig(out, dpi=100, facecolor="white"); plt.close(fig)


def draw_exploded(parts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["floor_plate", "output_bearing", "output_flange", "output_pins", "bearing_carrier",
             "shaft_bearings", "shaft", "eccentric_bearings", "cycloid_discs", "ring_pins",
             "pin_cage", "wall_tube", "stator_annulus", "rotor_ring", "rotor_carrier", "cover"]
    order = [n for n in order if n in parts]
    step = 30.0
    items, anno = [], []
    for i, n in enumerate(order):
        p = parts[n].moved(Location((0, 0, i * step)))
        items.append((p, COLORS[GROUP_OF[n]]))
        sol = max(p.solids(), key=lambda q: q.volume) if p.solids() else p   # avoid anchoring in the gap
        bb = sol.bounding_box()
        anno.append((n, (bb.max.X, 0.0, (bb.min.Z + bb.max.Z) / 2)))
    W, H = 1400, 1560
    azim, elev = -58, 16
    sc, c = auto_fit([p for p, _ in items], (int(W * 0.52), int(H * 0.965)), azim, elev)
    az, el = math.radians(azim), math.radians(elev)
    cam = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    rt = np.cross([0, 0, 1], cam); rt /= np.linalg.norm(rt)
    c = c + (0.24 * W / sc) * rt                       # push the object into the left half
    img, project = raster(items, (W, H), sc, c, azim=azim, elev=elev)
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.imshow(img); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    for n, p in anno:
        sx, sy = project(p)
        ax.annotate(f"{PRETTY[n]}   [{round(MASS[n]):d} g]", (sx, sy), (W - 8, sy), fontsize=8.4,
                    ha="right", va="center", color="#1a1a1a",
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.6),
                    bbox=dict(fc="white", ec="#ccc", lw=0.4, pad=1.8, alpha=0.94))
    ax.set_title(f"Frameless-motor actuator ({'yaw' if YAW else 'femur / knee'}), exploded along the axis \u2014 mounting face at the bottom.\n"
                 f"Names keyed to docs/design/bom-actuator.csv where they match; mass from this model.  Total {sum(MASS.values())/1e3:.2f} kg.",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=100, facecolor="white"); plt.close(fig)


# ------------------------------------------------- the three-unit comparison --
PCB_CACHE = os.path.join(ROOT, "build", "cad", "frameless", "pcb-section.json")


def pcb_section():
    """True section of the PCB two-stator femur unit, built from cad/actuator/actuator.py
    itself (not redrawn by hand).  Cached because it takes ~1 min to build."""
    if os.path.exists(PCB_CACHE):
        return json.load(open(PCB_CACHE))
    argv = sys.argv
    sys.argv = ["actuator.py", "femur"]                       # actuator.py reads sys.argv at import
    try:
        import actuator as act
        aparts, _amass, _ainfo = act.build("femur")
        out = {}
        for g, names in act.GROUPS.items():
            polys = []
            for n in names:
                if n not in aparts:
                    continue
                polys += [[[float(a), float(b)] for a, b in poly] for poly in act.section_polys(aparts[n])]
            if polys:
                out[g] = polys
        out["_meta"] = dict(od=act.R_OD * 2, h=act.H_TOTAL, drum_h=act.DRUM_H)
    finally:
        sys.argv = argv
    json.dump(out, open(PCB_CACHE, "w"))
    return out


def draw_compare(parts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fem = json.load(open(os.path.join(ROOT, "cad", "actuator", "femur.json")))
    fig, ax = plt.subplots(figsize=(14.0, 4.9))
    ACT = {"housing": "#9aa5ad", "rotor": "#d98c3a", "magnets": "#c0392b", "stator": "#0f9b8e",
           "reducer": "#3a3a3a", "bearings": "#e0e0e0", "transmission": "#6c8e3a"}

    def put(polys_by_group, x0, colours):
        for g, polys in polys_by_group.items():
            if g.startswith("_"):
                continue
            for poly in polys:
                ax.fill([p[0] + x0 for p in poly], [p[1] for p in poly], color=colours.get(g, "#ccc"),
                        lw=0.35, ec="#1a1a1a", zorder=3)

    # 1. this design, real geometry
    mine = {g: [[(p[0], p[1]) for p in poly] for n in ns if n in parts for poly in section_polys(parts[n])]
            for g, ns in GROUPS.items()}
    put(mine, 0, COLORS)
    # 2. the PCB two-stator unit, real geometry from cad/actuator/actuator.py
    pcb = pcb_section()
    X2 = 300.0
    put(pcb, X2, ACT)
    # 3. the 8318 outrunner unit — block model: no CAD of it exists anywhere in this repo
    X3 = 560.0
    OTS = [(0, 46, 0, 40, "#0f9b8e"), (0, 46, 42, 58, "#3a3a3a"), (0, 96, -6, 0, "#9aa5ad"),
           (0, 96, 58, 64, "#9aa5ad"), (0, 30, -32, -6, "#6c8e3a")]
    for (r0, r1, z0, z1, c) in OTS:
        for sgn in (+1, -1):
            lo = X3 + (r0 if sgn > 0 else -r1)
            ax.add_patch(plt.Rectangle((lo, z0), r1 - r0, z1 - z0, fc=c, ec="#1a1a1a", lw=0.5,
                                       alpha=0.55, hatch="//", zorder=3))

    tot_g = sum(MASS.values())
    fem_env = fem["height_mm"] + pcb.get("_meta", {}).get("drum_h", 26.0)
    cols = [(0.0, R_OD, 0.0, H_TOTAL, H_TOTAL,
             f"THIS DESIGN \u2014 frameless \u00d8{MOTOR_OD:.0f} \u00d7 {MOTOR_L:.0f} kit\n"
             f"\u00d8{2*R_OD:.0f} \u00d7 {H_TOTAL:.1f} mm   \u2022   {tot_g/1e3:.2f} kg from THIS CAD\n"
             f"(\u00a79.17 assumed {FM['pick']['m_fk']:.2f} kg and {MOTOR_L+12:.0f} mm)\n"
             f"{MOT['T_cont']:.1f} N\u00b7m motor, {N_LOBES}:1, {FM['pick']['T_joint_fk']:.0f} N\u00b7m at the joint, no capstan"),
            (X2, 95.9, -pcb.get("_meta", {}).get("drum_h", 26.0), fem["height_mm"], fem_env,
             f"PCB two-stator (round 9) \u2014 cad/actuator/actuator.py\n"
             f"\u00d8{fem['od_mm']:.0f} \u00d7 {fem['height_mm']:.1f} mm   \u2022   {fem['total_g']/1e3:.2f} kg from that CAD\n"
             f"envelope {fem_env:.1f} mm with the capstan drum\n"
             f"5.9 N\u00b7m motor, 20:1 cycloid \u00d7 4:1 capstan"),
            (X3, 96, -32.0, 64.0, 96.0,
             "8318 outrunner (round 10) \u2014 BLOCK MODEL, no CAD exists\n"
             "\u00d8192 \u00d7 64 mm   \u2022   2.45 kg as costed\n"
             "envelope 96 mm with the capstan drum\n"
             "2.6 N\u00b7m motor, 25:1 cycloid \u00d7 4:1 capstan  (hatched = envelope only)")]
    for x0, r, zlo, zhi, env, txt in cols:
        ax.annotate("", (x0 + r + 8, 0), (x0 + r + 8, zhi), arrowprops=dict(arrowstyle="<|-|>", color="#b03a2e",
                                                                           lw=0.9, mutation_scale=8))
        ax.text(x0 + r + 10.5, zhi / 2, f"{zhi:.1f} above the face", color="#b03a2e", fontsize=8, rotation=90, va="center")
        if zlo < 0:
            ax.annotate("", (x0 + r + 26, zlo), (x0 + r + 26, zhi), arrowprops=dict(arrowstyle="<|-|>", color="#7a4b8a",
                                                                                    lw=0.9, mutation_scale=8))
            ax.text(x0 + r + 28.5, (zlo + zhi) / 2, f"{env:.1f} full envelope", color="#7a4b8a", fontsize=8,
                    rotation=90, va="center")
        ax.text(x0, -48, txt, ha="center", va="top", fontsize=8.8)
    ax.plot([-140, X3 + 150], [0, 0], color="#b03a2e", lw=0.9, ls=(0, (6, 3)), zorder=2)
    ax.text(-138, 1.8, "mounting face  z = 0", color="#b03a2e", fontsize=8)
    ax.set_xlim(-145, X3 + 160); ax.set_ylim(-118, 82)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Three actuator units, same scale, sections through the axis.  The frameless motor is an annulus, so the cycloid lives inside its bore instead of\n"
                 "underneath it \u2014 but the reducer's own axial stack, not the motor, sets the height of all three.  Deleting the capstan is what actually shortens the unit.",
                 fontsize=11)
    hand = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[g], ec="#1a1a1a", lw=0.4) for g in GROUPS]
    hand += [plt.Rectangle((0, 0), 1, 1, fc="#6c8e3a", ec="#1a1a1a", lw=0.4)]
    ax.legend(hand, ["housing", "stator", "rotor ring / magnets", "rotor + shaft", "reducer", "bearings", "capstan drum"],
              fontsize=8.5, ncol=7, loc="lower center", frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(); fig.savefig(out, dpi=100, facecolor="white"); plt.close(fig)


# ------------------------------------------------------------------ checks ---
def checks(parts):
    """Every clearance and capacity number this model can actually settle."""
    c = {}
    c["cycloid_mesh_min_gap_mm"] = round(mesh_min_gap(), 6)
    c["cycloid_mesh_ok"] = abs(mesh_min_gap() - R_PIN) < 1e-6
    c["note_cycloid_profile"] = ("analysis/cycloid.py::profile() offsets the epitrochoid OUTWARD, which makes a "
                                 "disc larger than its own pin circle (it cannot mesh, and the discs in "
                                 "cad/actuator/actuator.py interfere with the pin cage by ~3 mm). This model "
                                 "offsets INWARD; the mesh gap above equals the pin radius exactly.")
    c["radial_budget_mm"] = {
        "motor_bore_radius": round(R_BORE, 3),
        "ring_pin_circle": round(R_PINS, 3),
        "ring_pin_outer_surface": round(R_PINS + R_PIN, 3),
        "cage_bore (must clear the disc lobe tips)": round(R_CAGE_IN, 3),
        "disc_swept_outer_radius": round(DISC_SWEPT_R, 3),
        "disc_to_cage_bore_clearance": round(R_CAGE_IN - DISC_SWEPT_R, 3),
        "cage_outer": round(R_CAGE_OUT, 3),
        "steel_behind_each_ring_pin": round(CAGE_WALL, 3),
        "pin_groove_depth (of Ø%.0f)" % (2 * R_PIN): round(PIN_GROOVE_DEPTH, 3),
        "pin_groove_fraction_of_diameter": round(PIN_GROOVE_DEPTH / (2 * R_PIN), 3),
        "cage_to_rotor_skirt_clearance": round(SKIRT_CLEARANCE, 3),
        "rotor_carrier_skirt_thickness": round(SKIRT_T, 3),
        "spare_to_motor_bore": round(R_BORE - (R_CAGE_OUT + SKIRT_CLEARANCE + SKIRT_T), 4)}
    c["airgap_mm"] = AIRGAP
    c["disc_root_to_output_hole_web_mm"] = round(DISC_ROOT_R - E_ECC - (R_OUT_PINS + R_OUT_HOLE), 3)
    c["disc_root_to_lightening_hole_web_mm"] = round(DISC_ROOT_R - E_ECC - (R_LIGHT1 + LIGHT1_D / 2), 3)
    c["disc_bore_to_lightening_hole_web_mm"] = (round((R_LIGHT2 - LIGHT2_D / 2) - (ECC_BRG_OUT + E_ECC), 3)
                                            if LIGHT2_D > 0 else "no inner lightening ring: no room between the needle cup and the output holes")
    c["flange_plate_to_disc_clearance_mm"] = round(Z_DISCS[0] - Z_FLANGE1, 3)
    c["flange_plate_bore_to_needle_cup_clearance_mm"] = round(R_FLANGE_BORE - (ECC_BRG_OUT + E_ECC), 3)
    c["flange_plate_to_bearing_carrier_clearance_mm"] = round(R_OBRG_OUT - R_FLANGE, 3)
    c["rotor_carrier_to_cage_top_clearance_mm"] = round(Z_ROTCAR0 - Z_CAGE1, 3)
    c["cover_to_rotor_carrier_clearance_mm"] = round(Z_COVER0 - Z_ROTCAR1, 3)
    # bearing capacity: the capstan is gone, so the output bearing carries the whole joint moment
    M = T_CYC_PEAK * 1000.0
    c["output_bearing"] = {"fitted": CRB_NAME, "d_D_B": [CRB[0], CRB[1], CRB[2]], "dp_mm": CRB_DP,
                           "C0_kN": CRB[5], "peak_joint_moment_Nm": round(T_CYC_PEAK, 1),
                           "P0_from_moment_N": round(2 * M / CRB_DP, 0),
                           "static_safety_factor_fs": round(CRB[5] * 1e3 / (2 * M / CRB_DP), 2),
                           "source": "THK 382-5E p.19, docs/reference/thk-cross-roller-ring-382-5E.pdf; "
                                     "P0 = 2M/dp with Fr = Fa = 0, fs = C0/P0 (THK p.6)",
                           "alternates_fs": {k: round(v[5] * 1e3 / (2 * M / v[3]), 2) for k, v in CRB_TABLE.items()},
                           "fs_target": 2.5,
                           "fs_limits_THK": "1-2 normal load, 2-3 impact load (THK 382-5E p.6)",
                           "note": ("The BOM's RB5013 was picked in round 8, when the capstan carried the joint "
                                    "moment and the bearing saw rope pull. With the capstan deleted the output "
                                    "bearing carries the whole joint moment: the RB5013 then gives fs %.2f, %s"
                                    % (CRB_TABLE["RB5013"][5] * 1e3 / (2 * M / CRB_TABLE["RB5013"][3]),
                                       "which is adequate, and it is kept." if CRB_NAME == "RB5013" else
                                       "below THK's 2-3 for impact loads, so %s is fitted here instead. This is a "
                                       "BOM change this CAD forces." % CRB_NAME))}
    c["eccentric_bearing"] = {"fitted": "HK2512", "F_ecc_peak_per_disc_N": round(CYC["F_ecc"], 0),
                              "C0r_N": HK2512["C0r_N"],
                              "static_margin": round(HK2512["C0r_N"] / CYC["F_ecc"], 2),
                              "source": "NTN sheet, docs/reference/ntn-hk2512.pdf"}
    c["cycloid_loads"] = {"ring_pin_peak_N_per_disc": round(CYC["F_pin"], 0),
                          "hertz_peak_MPa": round(CYC["sigma_peak"], 0),
                          "hertz_cont_MPa": round(CYC["sigma_cont"], 0),
                          "hertz_allowable_MPa": cy.SIGMA_H_ALLOW,
                          "output_pin_peak_N_per_disc_at_r%.0f" % R_OUT_PINS: round(F_OUT_PIN, 0),
                          "output_pin_peak_N_per_disc_at_cycloid_py_r%.0f" % cy.R_OUT: round(F_OUT_PIN_CY, 0),
                          "note": "analysis/cycloid.py fixes the output-pin circle at r %.0f, sized for the Ø100 "
                                  "PCB-stator bore. This bore is Ø%.1f, so the circle moves out to r %.0f and the "
                                  "peak output-pin force falls %.0f%%." % (cy.R_OUT, 2 * R_BORE, R_OUT_PINS,
                                                                          100 * (1 - F_OUT_PIN / F_OUT_PIN_CY))}
    # solid-body interference: every unordered pair of parts that should not touch
    inter = []
    names = [n for n in parts if n not in ("stator_annulus",)]
    skip = {("shaft", "eccentric_bearings"), ("shaft", "shaft_bearings"), ("cycloid_discs", "eccentric_bearings"),
            ("pin_cage", "ring_pins"), ("output_flange", "output_pins"), ("output_flange", "output_bearing"),
            ("output_flange", "shaft_bearings"), ("cover", "shaft_bearings"), ("rotor_carrier", "rotor_ring"),
            ("floor_plate", "output_bearing"), ("bearing_carrier", "output_bearing"),
            ("bearing_carrier", "pin_cage"), ("floor_plate", "wall_tube"), ("wall_tube", "cover"),
            ("rotor_carrier", "shaft"), ("cycloid_discs", "output_pins"), ("cycloid_discs", "ring_pins")}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (a, b) in skip or (b, a) in skip:
                continue
            v = (parts[a] & parts[b]).volume
            if v > 1.0:
                inter.append(dict(a=a, b=b, overlap_mm3=round(v, 1)))
    c["solid_interferences"] = inter
    c["solid_interference_pairs_skipped"] = sorted("+".join(s) for s in skip)
    return c


MASS: dict = {}

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    PARTS, MASS = build()

    comp = Compound(children=[PARTS[n] for n in PARTS])
    export_step(comp, os.path.join(OUT_DIR, f"actuator-{TAG}.step"))
    cut = {n: cut_part(p) for n, p in PARTS.items()}
    for g, names in GROUPS.items():
        names = [n for n in names if n in PARTS]
        if not names:
            continue
        export_stl(Compound(children=[PARTS[n] for n in names]), os.path.join(OUT_DIR, f"{g}.stl"),
                   tolerance=0.4, angular_tolerance=0.3)
        export_stl(Compound(children=[cut[n] for n in names]), os.path.join(OUT_DIR, f"{g}-cut.stl"),
                   tolerance=0.4, angular_tolerance=0.3)

    total_g = sum(MASS.values())
    group_g = {g: round(sum(MASS[n] for n in ns if n in MASS), 1) for g, ns in GROUPS.items()}
    # what §9.17 assumed this unit weighs: the active motor mass plus the fixed
    # 1.25 kg reducer + 0.55 kg housing of analysis/frameless_motor.py (M_REDUCER,
    # M_HOUSING, themselves lifted from the Ø100-bore / Ø192 CAD mass table).
    M_REDUCER_ASSUMED, M_HOUSING_ASSUMED = 1.25, 0.55
    assumed_kg = MOT["mass_kg"] + M_REDUCER_ASSUMED + M_HOUSING_ASSUMED
    # knock-on of the CAD mass on the closure fixed point (arithmetic only; analysis/ is not touched)
    m_fk_cad = total_g / 1e3 if not YAW else None
    CH = checks(PARTS)
    bb = comp.bounding_box()
    rec = dict(
        variant="yaw" if YAW else "femur/knee",
        source="hw/stator/frameless_motor.json " + ("['yaw']" if YAW else "['pick']") + "; analysis/cycloid.py; docs/design/08-actuator-design.md §9.17",
        modelled_vs_assumed=dict(
            modelled=["housing (floor plate, wall tube, bearing carrier, pin cage, cover)",
                      "reducer (cycloid discs from the corrected profile, ring pins, output pins, flange)",
                      "eccentric shaft with its journals", "rotor carrier", "bearing envelopes"],
            assumed=["frameless kit internal geometry: no teeth, slots, coils or magnet blocks — "
                     "the datasheet gives none, so the stator is one plain annulus and the rotor one plain ring",
                     "INNER ROTOR (the datasheet does not say; chosen so the loss-producing stator conducts "
                     "straight into the housing wall)",
                     "the 14.2 mm radial build and the 1039 g / 660 g active mass are analysis/frameless_motor.py's "
                     "inference from the datasheet mass, not measured",
                     "6905 and 6802 masses are modelled steel rings × 0.8 (no datasheet filed)"]),
        motor=dict(od_mm=MOTOR_OD, length_mm=MOTOR_L, bore_mm=2 * R_BORE, airgap_dia_mm=2 * R_GAP,
                   radial_build_mm=round(BUILD_TOT, 3), mechanical_airgap_mm=AIRGAP,
                   stator_radii_mm=[round(R_STA_IN, 3), R_MOT_OUT], rotor_radii_mm=[round(R_BORE, 3), round(R_ROT_OUT, 3)],
                   active_mass_g=round(ACTIVE_MASS_G, 1), T_cont_Nm=round(MOT["T_cont"], 3), T_peak_Nm=round(MOT["T_peak"], 3)),
        reducer=dict(lobes=N_LOBES, ring_pins=N_PINS, ring_pin_dia_mm=2 * R_PIN,
                     pin_circle_radius_mm=round(R_PINS, 3), pitch_mm=round(CYC["pitch"], 3),
                     eccentricity_mm=round(E_ECC, 4), discs=N_DISCS, disc_thickness_mm=DISC_T,
                     disc_pitch_mm=DISC_PITCH, output_pins=N_OUT, output_pin_dia_mm=OUT_PIN_D,
                     output_pin_circle_mm=R_OUT_PINS, disc_root_r_mm=round(DISC_ROOT_R, 3),
                     disc_tip_r_mm=round(DISC_TIP_R, 3), disc_swept_r_mm=round(DISC_SWEPT_R, 3),
                     T_cont_Nm=round(T_CYC_CONT, 1), T_peak_Nm=round(T_CYC_PEAK, 1)),
        envelope=dict(od_mm=2 * R_OD, bore_mm=2 * R_BORE, height_mm=H_TOTAL,
                      bbox_mm=[round(bb.size.X, 1), round(bb.size.Y, 1), round(bb.size.Z, 1)],
                      height_claimed_in_9_17_mm=MOTOR_L + 12,
                      height_note=("§9.17 quotes %d mm for this unit, which is the motor stack plus 6 mm of case "
                                   "each end. It does not include the reducer's own axial stack: the output "
                                   "crossed roller (%.0f), the output flange (%.0f), two %.0f mm discs on the %.0f mm "
                                   "HK2512 cup pitch (%.0f), and the rotor carrier plus cover (%.1f). The modelled "
                                   "unit is %.1f mm." % (MOTOR_L + 12, T_OBRG, T_FLANGE, DISC_T, DISC_PITCH,
                                                         Z_CAGE1 - Z_CAGE0, H_TOTAL - Z_CAGE1, H_TOTAL))),
        axial_stack_mm={
            "mounting/output face": 0.0,
            "floor plate": [0.0, T_FLOOR],
            "output crossed roller %s" % CRB_NAME: [Z_OBRG0, Z_OBRG1],
            "6905 shaft bearing": [Z_SHAFT0, Z_FLANGE0],
            "output flange plate": [Z_FLANGE0, Z_FLANGE1],
            "bearing carrier": [T_FLOOR, Z_CAGE0],
            "pin cage / ring pins": [Z_CAGE0, Z_CAGE1],
            "HK2512 cups": [[round(z - CUP_OVER, 2), round(z + DISC_T + CUP_OVER, 2)] for z in Z_DISCS],
            "cycloid discs": [[round(z, 2), round(z + DISC_T, 2)] for z in Z_DISCS],
            "output pins": [Z_FLANGE0, round(Z_DISCS[-1] + DISC_T + 0.8, 2)],
            "stator annulus": [round(Z_MOT0, 2), round(Z_MOT1, 2)],
            "rotor ring": [round(Z_MOT0, 2), round(Z_MOT1, 2)],
            "rotor carrier skirt": [round(Z_MOT0, 2), Z_ROTCAR0],
            "rotor carrier disc": [Z_ROTCAR0, Z_ROTCAR1],
            "wall tube": [T_FLOOR, Z_COVER0],
            "cover plate": [Z_COVER0, Z_COVER0 + T_COVER],
            "6802 / cover boss (top of unit)": [Z_COVER0, Z_BOSS1]},
        mass_g={k: round(v, 1) for k, v in MASS.items()},
        mass_by_group_g=group_g,
        total_g=round(total_g, 1),
        mass_assumed_in_closure_kg=round(assumed_kg, 3),
        mass_delta_g=round(total_g - assumed_kg * 1e3, 1),
        mass_note=("§9.17 costed the unit at %.2f kg = %.3f kg of active motor + %.2f kg reducer + %.2f kg housing "
                   "(analysis/frameless_motor.py M_REDUCER / M_HOUSING, themselves the CAD mass table of the "
                   "Ø100-bore / Ø192 design in cost_search.py). Neither fixed number survives this design: the "
                   "cycloid pin circle moves from r 43.5 out to r %.1f so the two discs alone weigh %.0f g, and the "
                   "housing is a Ø%.0f can. This CAD gives %.2f kg, %+.0f g."
                   % (assumed_kg, ACTIVE_MASS_G / 1e3, M_REDUCER_ASSUMED, M_HOUSING_ASSUMED, R_PINS,
                      MASS["cycloid_discs"], 2 * R_OD, total_g / 1e3, total_g - assumed_kg * 1e3)),
        mass_reducer_g=round(sum(MASS[n] for n in GROUPS["reducer"]) + MASS["eccentric_bearings"]
                             + MASS["output_bearing"] + MASS["shaft_bearings"], 1),
        mass_housing_g=round(sum(MASS[n] for n in GROUPS["housing"]), 1),
        checks=CH,
        densities_g_per_mm3=dict(aluminium_6061=AL, steel=STEEL,
                                 stator_effective=round(M_STATOR_G / PARTS["stator_annulus"].volume, 6),
                                 rotor_ring_effective=round(M_ROTOR_RING_G / PARTS["rotor_ring"].volume, 6),
                                 inference_rho_eff=round(RHO_EFF, 6)),
        files=dict(step=f"build/cad/{TAG}/actuator-{TAG}.step",
                   stl=[f"build/cad/{TAG}/{g}.stl" for g in GROUPS] + [f"build/cad/{TAG}/{g}-cut.stl" for g in GROUPS],
                   renders=[f"docs/design/actuator/{'frameless-cad-yaw' if YAW else 'frameless-cad'}-{k}.png"
                            for k in (["section", "iso", "exploded"] + ([] if YAW else ["compare"]))]))

    if not YAW:
        # what the CAD mass does to the closure fixed point, using frameless_motor.json's own coefficients
        m_yaw_cad = None
        yaw_json = os.path.join(ROOT, "cad", "actuator", "frameless-yaw.json")
        if os.path.exists(yaw_json):
            m_yaw_cad = json.load(open(yaw_json))["total_g"] / 1e3
        if m_yaw_cad:
            m_robot = FM["requirement"]["m_fixed"] + 12 * m_fk_cad + 6 * m_yaw_cad
            need = {j: FM["requirement"]["c_per_kg"][j] * m_robot for j in ("yaw", "femur", "knee")}
            T = {"yaw": FM["pick"]["T_joint_yaw"], "femur": FM["pick"]["T_joint_fk"], "knee": FM["pick"]["T_joint_fk"]}
            rec["closure_with_cad_mass"] = dict(
                m_fk_kg=round(m_fk_cad, 3), m_yaw_kg=round(m_yaw_cad, 3), m_robot_kg=round(m_robot, 1),
                m_robot_kg_in_9_17=round(FM["pick"]["m_robot"], 1),
                need_Nm={j: round(v, 1) for j, v in need.items()},
                margin={j: round(T[j] / need[j], 3) for j in need},
                margin_in_9_17={j: round(v, 3) for j, v in FM["pick"]["margin"].items()},
                note=("Arithmetic only, on frameless_motor.json's own c_per_kg and m_fixed: it re-evaluates the "
                      "closure at the CAD unit mass without the mass fixed point being re-solved (the leg and body "
                      "structure would also grow). analysis/ was not modified. A margin below 1.0 means the design "
                      "as drawn does not close and the parent session has to re-run the study."))

    json.dump(rec, open(os.path.join(ROOT, "cad", "actuator", f"{TAG}.json"), "w"), indent=1)

    stem = "frameless-cad-yaw" if YAW else "frameless-cad"
    draw_section(PARTS, os.path.join(FIG, f"{stem}-section.png"))
    draw_iso(PARTS, os.path.join(FIG, f"{stem}-iso.png"))
    draw_exploded(PARTS, os.path.join(FIG, f"{stem}-exploded.png"))
    if not YAW:
        draw_compare(PARTS, os.path.join(FIG, "frameless-cad-compare.png"))

    print(json.dumps({k: rec[k] for k in ("variant", "envelope", "mass_by_group_g", "total_g",
                                          "mass_assumed_in_closure_kg", "mass_delta_g")}, indent=1))
    print("\nchecks:")
    print(json.dumps(CH, indent=1))
    print(f"\nheight {H_TOTAL:.1f} mm, OD {2*R_OD:.0f} mm, mass {total_g/1e3:.3f} kg "
          f"against the {assumed_kg:.2f} kg §9.17 assumed ({(total_g/1e3 - assumed_kg)*1e3:+.0f} g)")

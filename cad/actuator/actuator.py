#!/opt/hw-py/bin/python
"""Actuator CAD: one femur/knee/yaw unit in build123d, from the numbers the
analysis scripts already agree on (hw/stator/geometry.json, analysis/cycloid.py,
analysis/rotor_field.py).

    /opt/hw-py/bin/python cad/actuator/actuator.py [femur|knee|yaw]

Topology (z up, the mounting face is z = 0, the output exits through it):
  * base: floor plate + outer wall (r 86-93) + the fixed pin cylinder (r 42-46)
    that rises through the open bottom of the rotor cup; the ring pins sit in
    half-grooves on its bore; the crossed-roller output bearing sits in its foot
  * rotor cup: bottom carrier ring (r 46.5-85.5) + drum (r 46.5-49.5, through
    the stator bore) + top carrier disc with a hub on the eccentric shaft;
    60 Halbach blocks 30x5x8 glued on each carrier
  * stator board clamped at its rim between the base wall and the upper ring
  * one cycloid disc on a needle bearing on the shaft's eccentric journal,
    output pins from the flange through its holes; the flange rides the
    crossed-roller bearing and carries the shaft's lower bearing
  * cover with the shaft's top bearing
Writes build/cad/<joint>/ (STEP + STL per colour group), cad/actuator/<joint>.json
(masses, envelope) and docs/design/actuator/cad-<joint>-{section,cutaway}.png.
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from build123d import (Align, Box, Compound, Cylinder, Location, Part, Plane, Polygon, Pos, Rot,  # noqa: E402
                       export_step, export_stl, extrude)
import cycloid as cy  # noqa: E402

GEO = json.load(open(os.path.join(ROOT, "hw", "stator", "geometry.json")))
RF = json.load(open(os.path.join(ROOT, "hw", "stator", "rotor_field.json")))
JOINT = sys.argv[1] if len(sys.argv) > 1 else "femur"
JOINT_MAGNET = {"femur": "rect 30x5x8 N48", "knee": "rect 30x5x8 N48", "yaw": "rect 30x5x8 N48"}   # round 6: the heavier robot needs the 8 mm blocks on the yaw too
MAGNET = JOINT_MAGNET[JOINT]
MAG_L, MAG_W, MAG_H = 30.0, RF[MAGNET]["w_b_mm"], RF[MAGNET]["h_m_mm"]
P = GEO["pole_pairs"]
N_MAG = 4 * P                                   # 4 Halbach segments per pole pair

# ---- radial (mm) ------------------------------------------------------------
R_OD = 93.0            # housing outside radius (Ø186)
R_WALL_IN = 86.0       # outer wall bore; the board rim r 86-87.5 is clamped on it
R_BOARD = GEO["r_board_mm"] if "r_board_mm" in GEO else 87.5
R_BORE = 50.0          # stator board bore
R_CARRIER_OUT = 85.5
R_MAG_OUT, R_MAG_IN = 85.0, 55.0
R_DRUM_OUT, R_DRUM_IN = 49.5, 46.5
R_CYL_OUT = 46.0
R_PINS = cy.R_PIN_CIRCLE                        # 43.5
R_OUT_BRG_OUT, R_OUT_BRG_IN, T_OUT_BRG = 40.0, 25.0, 13.0   # crossed roller RB5013 (50x80x13)
R_FLANGE = 41.5
R_OUT_PINS, OUT_PIN_D, N_OUT = cy.R_OUT, cy.OUT_PIN_D, cy.N_OUT
SHAFT_R = 12.5                                  # Ø25 shaft at the bearings and the hub
JOURNAL_R = 15.0                                # Ø30 eccentric journals (round 7: HK3012, review decision)
ECC_BRG_OUT, ECC_BRG_W = 18.5, 12.0             # HK3012 drawn-cup needle 30x37x12
SHAFT_BRG_OUT, SHAFT_BRG_W = 21.0, 9.0          # 6905 25x42x9
TOP_BRG_IN, TOP_BRG_OUT, TOP_BRG_W = 7.5, 12.0, 5.0    # 6802 15x24x5
N_BOLTS, BOLT_D, R_BOLTS = 12, 4.4, 89.5
N_OUT_BOLTS, R_OUT_BOLTS = 6, 19.5              # M4 tapped in the output face (r 13.5-25)

# ---- axial (mm) -------------------------------------------------------------
T_FLOOR = 3.0
CLR = 0.5                                       # rotor to fixed parts, magnet face to board
T_CARRIER = 4.5                                 # see carrier_deflection(): 0.06 mm under the rotor attraction
T_BOARD = 2.2                                   # 12-layer 3 oz finished thickness
DISC_T, N_DISCS = cy.DISC_T, cy.N_DISCS         # 8 mm, two discs 180 deg apart
Z_BRG0 = 0.0
Z_FLANGE0 = T_OUT_BRG                           # 13
T_FLANGE = 4.0
Z_DISC0 = Z_FLANGE0 + T_FLANGE + 0.2            # 17.2
CUP_OVER = (ECC_BRG_W - DISC_T) / 2             # needle cup protrudes this much each side of its disc
DISC_PITCH = DISC_T + 2 * CUP_OVER              # cups of neighbouring discs touch: 12 mm per disc
Z_DISCS = [Z_DISC0 + k * DISC_PITCH for k in range(N_DISCS)]
Z_CUP0 = Z_DISC0 - CUP_OVER
Z_CYL_TOP = Z_DISCS[-1] + DISC_T + CUP_OVER     # cylinder stops level with the last cup top
N_STATORS = 1 if "--stators=1" in sys.argv or "1s" in sys.argv else 2      # round 7: two stators is the canonical unit (review decision A)
TAG = JOINT + ("-1s" if N_STATORS == 1 else "")
# the motor band hangs from the top carrier, which must clear the upper disc's cup;
# with two stators the band is taller than the reducer and lifts the top carrier instead
H_BAND = N_STATORS * (T_BOARD + 2 * CLR + 2 * MAG_H) + (N_STATORS + 1) * T_CARRIER
Z_TOPCAR0 = max(Z_CYL_TOP + CLR, T_FLOOR + CLR + 1.5 + H_BAND - T_CARRIER)   # 1.5: the drum lip under the bottom ring
Z_TOPCAR1 = Z_TOPCAR0 + T_CARRIER
MAGS, BOARDS, CARRIERS = [], [], []               # (z0, z1) bottom -> top, built top-down
z = Z_TOPCAR0
for _k in range(N_STATORS):
    MAGS.insert(0, (z - MAG_H, z)); z -= MAG_H
    BOARDS.insert(0, (z - CLR - T_BOARD, z - CLR)); z -= CLR + T_BOARD + CLR
    MAGS.insert(0, (z - MAG_H, z)); z -= MAG_H
    if _k < N_STATORS - 1:
        CARRIERS.insert(0, (z - T_CARRIER, z)); z -= T_CARRIER      # middle rotor ring, magnets both faces
CARRIERS.insert(0, (z - T_CARRIER, z))                               # bottom carrier ring
CARRIERS.append((Z_TOPCAR0, Z_TOPCAR1))                              # top carrier with the hub
Z_BOTCAR0, Z_BOTMAG0 = CARRIERS[0]
Z_BOTMAG1 = MAGS[0][1]
Z_BOARD0, Z_BOARD1 = BOARDS[0]
Z_TOPMAG0 = MAGS[-1][0]
assert Z_BOTCAR0 >= T_FLOOR + CLR - 1e-9, (Z_BOTCAR0, "bottom carrier hits the floor plate")
Z_COVER0 = Z_TOPCAR1 + CLR
T_COVER = 2.0
Z_BOSS1 = Z_COVER0 + TOP_BRG_W                  # the top bearing sits in a boss on the cover
H_TOTAL = Z_BOSS1
AL, STEEL, NDFEB, FR4 = 2.70e-3, 7.85e-3, 7.50e-3, 1.85e-3   # g/mm^3
F_ATTRACT = RF[MAGNET]["attraction_N"]          # rotor-to-rotor pull from the Maxwell stress at the mid-plane
E_AL, NU_AL = 71e3, 0.33                        # MPa, 7075-T6


def carrier_deflection(t=T_CARRIER):
    """Magnet-carrying annulus r 49.5-85.5 as a cantilever plate strip from the
    drum, uniformly loaded by the attraction: w = q L^4 / (8 D).  Conservative
    (ignores the hoop stiffness of the annulus)."""
    L = R_CARRIER_OUT - R_DRUM_OUT
    q = F_ATTRACT / (math.pi * (R_MAG_OUT**2 - R_MAG_IN**2))     # N/mm^2 -> N/mm per unit width
    D = E_AL * t**3 / (12 * (1 - NU_AL**2))
    return q * L**4 / (8 * D)


R_LIGHT, LIGHT_D = 34.5, 9.0                    # disc lightening holes between the output-pin holes


def cyl(r, z0, z1, x=0.0, y=0.0):
    return Pos(x, y, z0) * Cylinder(r, z1 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN))


def ring(r_in, r_out, z0, z1):
    return cyl(r_out, z0, z1) - cyl(r_in, z0 - 1, z1 + 1)


def build(joint="femur"):
    N, T_cont, T_peak = cy.JOINTS[joint]
    d = cy.design(N, T_cont, T_peak)
    e, r_pin = d["e"], d["r_pin"]
    n_pins = N + 1
    parts, mass = {}, {}

    # ---- base: floor + outer wall + pin cylinder --------------------------------
    base = ring(R_OUT_BRG_OUT, R_OD, 0, T_FLOOR)
    base += ring(R_WALL_IN, R_OD, T_FLOOR, BOARDS[0][0])
    base += ring(R_OUT_BRG_OUT, R_CYL_OUT, 0, T_OUT_BRG)                 # bearing seat
    base += ring(R_PINS - r_pin, R_CYL_OUT, T_OUT_BRG, Z_CYL_TOP)       # pin cylinder
    for k in range(n_pins):
        a = 2 * math.pi * k / n_pins
        base -= cyl(r_pin + 0.03, Z_DISC0 - 1.5, Z_CYL_TOP + 1, R_PINS * math.cos(a), R_PINS * math.sin(a))
    for k in range(N_BOLTS):
        a = 2 * math.pi * (k + 0.5) / N_BOLTS
        base -= cyl(BOLT_D / 2 - 0.5, T_FLOOR, Z_BOARD0 + 1, R_BOLTS * math.cos(a), R_BOLTS * math.sin(a))  # tapped M4
    parts["base"] = base; mass["base"] = base.volume * AL

    pins = Part()
    for k in range(n_pins):
        a = 2 * math.pi * k / n_pins
        pins += cyl(r_pin, Z_DISC0 - 1.0, Z_CYL_TOP, R_PINS * math.cos(a), R_PINS * math.sin(a))
    parts["ring_pins"] = pins; mass["ring_pins"] = pins.volume * STEEL

    # ---- upper wall ring and cover ---------------------------------------------
    upper = ring(R_WALL_IN, R_OD, BOARDS[-1][1], Z_COVER0)
    for (a0, a1), (b0, b1) in zip(BOARDS[:-1], BOARDS[1:]):
        upper += ring(R_WALL_IN, R_OD, a1, b0)                          # clamp ring between two boards
    cover = cyl(R_OD, Z_COVER0, Z_COVER0 + T_COVER) - cyl(TOP_BRG_OUT, Z_COVER0 - 1, Z_BOSS1 + 1)
    cover += ring(TOP_BRG_OUT, TOP_BRG_OUT + 3.0, Z_COVER0 + T_COVER, Z_BOSS1)
    for k in range(N_BOLTS):
        a = 2 * math.pi * (k + 0.5) / N_BOLTS
        h = cyl(BOLT_D / 2, BOARDS[0][1] - 1, Z_BOSS1 + 1, R_BOLTS * math.cos(a), R_BOLTS * math.sin(a))
        upper -= h; cover -= h
    parts["upper_ring"] = upper; mass["upper_ring"] = upper.volume * AL
    parts["cover"] = cover; mass["cover"] = cover.volume * AL

    # ---- stator board -------------------------------------------------------------
    board = Part()
    for b0, b1 in BOARDS:
        board += ring(R_BORE, R_BOARD, b0, b1)
    parts["stator"] = board
    mass["stator"] = board.volume * FR4 + 68.0 * N_STATORS              # copper from asbuilt.json

    # ---- rotor cup: two parts, because the board has to go on over the drum ------------
    # top carrier + drum (one machined part, with a lip at the drum's foot) and the bottom
    # carrier ring, bonded onto the drum foot against the lip after the board is on
    Z_LIP0 = Z_BOTCAR0 - 1.5
    assert Z_LIP0 >= T_FLOOR + CLR - 1e-6, (Z_LIP0, "drum lip hits the floor plate")
    rotor = ring(R_DRUM_IN, R_DRUM_OUT, Z_LIP0, Z_TOPCAR0)                # drum through the stator bore
    rotor += ring(R_DRUM_IN, R_DRUM_OUT + 1.5, Z_LIP0, Z_BOTCAR0)         # lip the bottom ring bears on
    rotor += ring(SHAFT_R, R_CARRIER_OUT, Z_TOPCAR0, Z_TOPCAR1)           # top carrier with hub bore
    parts["rotor_top"] = rotor; mass["rotor_top"] = rotor.volume * AL
    bring = Part()
    for c0, c1 in CARRIERS[:-1]:                                         # bottom ring and any middle rings
        bring += ring(R_DRUM_OUT, R_CARRIER_OUT, c0, c1)
    parts["rotor_bottom_ring"] = bring; mass["rotor_bottom_ring"] = bring.volume * AL

    mags = Part()
    for z0, _z1 in MAGS:
        for k in range(N_MAG):
            a = 360.0 * k / N_MAG
            blk = Pos((R_MAG_IN + R_MAG_OUT) / 2, 0, z0) * Box(MAG_L, MAG_W, MAG_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
            mags += Rot(0, 0, a) * blk
    parts["magnets"] = mags; mass["magnets"] = mags.volume * NDFEB
    assert abs(mags.volume * NDFEB - RF[MAGNET]["magnet_mass_g"] * N_STATORS) < 5, "magnet count vs rotor_field.json"

    # ---- shaft: lower journal, one eccentric journal per disc (180 deg apart), hub, stub ---
    shaft = cyl(SHAFT_R, Z_FLANGE0 - SHAFT_BRG_W, Z_CUP0)
    ecc = [(e * math.cos(math.pi * k), e * math.sin(math.pi * k)) for k in range(N_DISCS)]
    for k, zd in enumerate(Z_DISCS):
        shaft += cyl(JOURNAL_R, zd - CUP_OVER, zd + DISC_T + CUP_OVER, *ecc[k])
    shaft += cyl(SHAFT_R, Z_CYL_TOP, Z_TOPCAR1)
    shaft -= Pos(SHAFT_R - 2.0, -20, Z_TOPCAR0) * Box(10, 40, T_CARRIER, align=(Align.MIN, Align.MIN, Align.MIN))  # D-flat
    shaft += cyl(TOP_BRG_IN, Z_TOPCAR1, Z_BOSS1)
    parts["shaft"] = shaft; mass["shaft"] = shaft.volume * STEEL

    # ---- cycloid discs on the eccentrics, 180 deg apart ----------------------------
    x, y = cy.profile(N, cy.R_PIN_CIRCLE, r_pin, e, n=720)
    disc = Part()
    for k, zd in enumerate(Z_DISCS):
        ex, ey = ecc[k]
        pts = [(float(px) + ex, float(py) + ey) for px, py in zip(x[:-1], y[:-1])]   # disc centre on its journal
        d = Pos(0, 0, zd) * extrude(Polygon(*pts, align=None), amount=DISC_T)
        d -= cyl(ECC_BRG_OUT, zd - 1, zd + DISC_T + 1, ex, ey)
        for i in range(N_OUT):
            a = 2 * math.pi * i / N_OUT
            d -= cyl(OUT_PIN_D / 2 + e, zd - 1, zd + DISC_T + 1, R_OUT_PINS * math.cos(a) + ex, R_OUT_PINS * math.sin(a) + ey)
            b = a + math.pi / N_OUT
            d -= cyl(LIGHT_D / 2, zd - 1, zd + DISC_T + 1, R_LIGHT * math.cos(b) + ex, R_LIGHT * math.sin(b) + ey)
        disc += d
    parts["disc"] = disc; mass["disc"] = disc.volume * STEEL

    # ---- output flange + pins ----------------------------------------------------
    flange = ring(SHAFT_R + 1.0, R_OUT_BRG_IN, 0, Z_FLANGE0)             # hub inside the crossed roller
    flange += ring(SHAFT_R + 0.5, R_FLANGE, Z_FLANGE0, Z_FLANGE0 + T_FLANGE)
    flange -= cyl(SHAFT_BRG_OUT, Z_FLANGE0 - SHAFT_BRG_W, Z_FLANGE0 + 0.5)  # lower shaft bearing pocket, from above
    for k in range(N_OUT_BOLTS):                                          # leg / pulley attachment, tapped M4 in the z = 0 face
        a = 2 * math.pi * k / N_OUT_BOLTS
        flange -= cyl(3.3 / 2, -1, 6, R_OUT_BOLTS * math.cos(a), R_OUT_BOLTS * math.sin(a))
    opins = Part()
    for k in range(N_OUT):
        a = 2 * math.pi * k / N_OUT
        opins += cyl(OUT_PIN_D / 2, Z_FLANGE0 + 1.0, Z_DISCS[-1] + DISC_T + 0.5, R_OUT_PINS * math.cos(a), R_OUT_PINS * math.sin(a))
        flange -= cyl(OUT_PIN_D / 2, Z_FLANGE0 + 0.5, Z_FLANGE0 + T_FLANGE + 1, R_OUT_PINS * math.cos(a), R_OUT_PINS * math.sin(a))
    parts["output_flange"] = flange; mass["output_flange"] = flange.volume * AL
    parts["output_pins"] = opins; mass["output_pins"] = opins.volume * STEEL

    # ---- bearings (as steel rings) ------------------------------------------
    brgs = ring(R_OUT_BRG_IN, R_OUT_BRG_OUT, 0, T_OUT_BRG)
    brgs += ring(SHAFT_R, SHAFT_BRG_OUT, Z_FLANGE0 - SHAFT_BRG_W, Z_FLANGE0)
    for k, zd in enumerate(Z_DISCS):
        brgs += ring(JOURNAL_R, ECC_BRG_OUT, zd - CUP_OVER, zd + DISC_T + CUP_OVER).moved(Location((ecc[k][0], ecc[k][1], 0)))
    brgs += ring(TOP_BRG_IN, TOP_BRG_OUT, Z_COVER0, Z_BOSS1)
    parts["bearings"] = brgs; mass["bearings"] = brgs.volume * STEEL * 0.8

    return parts, mass, dict(N=N, e=e, r_pin=r_pin, n_pins=n_pins)


GROUPS = {"housing": ["base", "upper_ring", "cover"], "rotor": ["rotor_top", "rotor_bottom_ring", "shaft"], "magnets": ["magnets"],
          "stator": ["stator"], "reducer": ["disc", "ring_pins", "output_flange", "output_pins"], "bearings": ["bearings"]}
COLORS = {"housing": "#9aa5ad", "rotor": "#d98c3a", "magnets": "#c0392b", "stator": "#0f9b8e", "reducer": "#3a3a3a", "bearings": "#e0e0e0"}


def section_polys(part, tol=0.4):
    """Outline polygons of the part's section by the XZ plane (y = 0)."""
    slab = Box(400, 0.02, 200, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(Location((0, 0, 40)))
    sec = part & slab
    polys = []
    for f in sec.faces():
        if abs(f.normal_at().Y) < 0.9:
            continue
        for w in f.wires():
            pts = [w.position_at(t) for t in np.linspace(0, 1, 160, endpoint=False)]
            polys.append([(p.X, p.Z) for p in pts])
    return polys


def draw_section(parts, joint, info, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for g, names in GROUPS.items():
        for n in names:
            for poly in section_polys(parts[n]):
                ax.fill(*zip(*poly), color=COLORS[g], lw=0.3, ec="k", alpha=0.95 if g != "bearings" else 0.8)
    ax.axhline(0, color="#b03a2e", lw=0.6, ls="--")
    ax.annotate("", (-R_OD, -6), (R_OD, -6), arrowprops=dict(arrowstyle="<->", color="#b03a2e", lw=0.8))
    ax.text(0, -9.5, f"Ø{2*R_OD:.0f} housing", ha="center", color="#b03a2e", fontsize=8)
    ax.annotate("", (R_OD + 5, 0), (R_OD + 5, H_TOTAL), arrowprops=dict(arrowstyle="<->", color="#b03a2e", lw=0.8))
    ax.text(R_OD + 7, H_TOTAL / 2, f"{H_TOTAL:.1f} mm\n(target 42)", color="#b03a2e", fontsize=8, va="center")
    ax.annotate("", (-R_OUT_BRG_OUT, -3), (R_OUT_BRG_OUT, -3), arrowprops=dict(arrowstyle="<->", color="#b03a2e", lw=0.8))
    ax.text(0, -5.2, "Ø80 output bearing", ha="center", color="#b03a2e", fontsize=7)
    labels = [(R_MAG_OUT - 15, (Z_BOTMAG0 + Z_BOTMAG1) / 2, f"Halbach rings ({len(MAGS)}), 60 × 30×5×{MAG_H:.0f} N48 each"),
              (R_BOARD - 5, (Z_BOARD0 + Z_BOARD1) / 2, "stator PCB, 12L 3 oz, clamped at the rim"),
              (R_CARRIER_OUT - 10, (Z_TOPCAR0 + Z_TOPCAR1) / 2, "rotor: top carrier + drum, one part"),
              (R_CARRIER_OUT - 10, (Z_BOTCAR0 + Z_BOTMAG0) / 2, "bottom carrier ring, bonded on the drum foot"),
              (R_PINS, Z_CYL_TOP - 4, f"{info['n_pins']} pins Ø{2*info['r_pin']:.0f} in the fixed cylinder"),
              (R_OUT_PINS, Z_DISCS[-1] + DISC_T / 2, f"{N_DISCS} discs {info['N']}:1 on HK3012, e = {info['e']:.2f}, 180° apart"),
              (R_OUT_BRG_IN + 7, T_OUT_BRG / 2, "RB5013 crossed roller"),
              (0, Z_BOSS1 - 2, "6802"), (0, Z_FLANGE0 - 4, "6905"),
              (R_BOLTS, Z_COVER0 + 1.5, "12 × M4"),
              (R_FLANGE - 8, Z_FLANGE0 + 2, "output flange → exits at z = 0")]
    for k, (x, z, t) in enumerate(labels):
        ax.annotate(t, (x, z), (-R_OD - 8 if k % 2 else R_OD + 30, 2 + 3.6 * k), fontsize=7, ha="right" if k % 2 else "left",
                    va="center", arrowprops=dict(arrowstyle="-", color="#777", lw=0.5))
    ax.set_aspect("equal"); ax.set_xlim(-R_OD - 75, R_OD + 95); ax.set_ylim(-12, H_TOTAL + 6)
    ax.set_xlabel("mm"); ax.set_ylabel("mm (mounting face at 0)")
    ax.set_title(f"{joint} actuator{', two stators' if N_STATORS == 2 else ''}, section through the axis (from the build123d model)", fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def _mesh(part, tol=0.8):
    v, t = part.tessellate(tol, 0.4)
    return np.array([(q.X, q.Y, q.Z) for q in v]), np.array(t, dtype=int)


def render(parts, out, title, azim=-50, elev=28, cutter=None, size=(1000, 800), scale=4.4):
    """Orthographic z-buffer raster of the tessellated parts (no renderer
    dependency; ~20 s).  azim/elev: camera direction in degrees."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W, H = size
    az, el = math.radians(azim), math.radians(elev)
    cam = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])   # towards the viewer
    right = np.cross([0, 0, 1], cam); right /= np.linalg.norm(right)
    up = np.cross(cam, right)
    light = cam + 0.6 * up + 0.3 * right; light /= np.linalg.norm(light)
    zbuf = np.full((H, W), -1e9); img = np.ones((H, W, 3))
    for g, names in GROUPS.items():
        base = np.array(matplotlib.colors.to_rgb(COLORS[g]))
        for n in names:
            p = parts[n] - cutter if cutter is not None else parts[n]
            v, t = _mesh(p)
            if len(t) == 0:
                continue
            tri = v[t]                                           # (n, 3, 3)
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            shade = 0.35 + 0.65 * np.clip(nrm @ light, 0, 1)
            # screen coordinates
            sx = W / 2 + scale * (tri @ right); sy = H / 2 - scale * (tri @ up - 12); sz = tri @ cam
            for k in range(len(tri)):
                x0, x1 = int(max(sx[k].min(), 0)), int(min(sx[k].max(), W - 1)) + 1
                y0, y1 = int(max(sy[k].min(), 0)), int(min(sy[k].max(), H - 1)) + 1
                if x1 <= x0 or y1 <= y0:
                    continue
                gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
                (ax_, ay_), (bx_, by_), (cx_, cy_) = sx[k][[0, 1, 2]].tolist() and list(zip(sx[k], sy[k]))
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
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.imshow(img); ax.set_axis_off(); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def draw_cutaway(parts, joint, out):
    cutter = Box(200, 200, 200, align=(Align.MIN, Align.MIN, Align.MIN)).moved(Location((0, -200, -50)))  # remove x>0, y<0
    render(parts, out, f"{joint} actuator, quarter cutaway (mounting/output face down)", azim=-45, elev=30, cutter=cutter)


def draw_iso(parts, joint, out):
    render(parts, out, f"{joint} actuator from below: output flange and crossed-roller bearing in the mounting face", azim=-45, elev=-35)


if __name__ == "__main__":
    joint = JOINT
    parts, mass, info = build(joint)
    out_dir = os.path.join(ROOT, "build", "cad", joint)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(ROOT, "docs", "design", "actuator")
    comp = Compound(children=[parts[n] for n in parts])
    export_step(comp, os.path.join(out_dir, f"actuator-{joint}.step"))
    for g, names in GROUPS.items():
        export_stl(Compound(children=[parts[n] for n in names]), os.path.join(out_dir, f"{g}.stl"), tolerance=0.5, angular_tolerance=0.3)
    total = sum(mass.values())
    bb = comp.bounding_box()
    rec = dict(joint=joint, stators=N_STATORS, **info, mass_g={k: round(v, 1) for k, v in mass.items()}, total_g=round(total, 1),
               od_mm=2 * R_OD, height_mm=H_TOTAL, bbox=[round(bb.size.X, 1), round(bb.size.Y, 1), round(bb.size.Z, 1)],
               z=dict(board=[Z_BOARD0, Z_BOARD1], bot_mag=[Z_BOTMAG0, Z_BOTMAG1], top_mag=[Z_TOPMAG0, Z_TOPCAR0],
                      carriers=[list(c) for c in CARRIERS], boards=[list(b) for b in BOARDS], mags=[list(m) for m in MAGS], discs=[[zd, zd + DISC_T] for zd in Z_DISCS], cylinder_top=Z_CYL_TOP, cover=[Z_COVER0, Z_BOSS1]),
               magnet=MAGNET, gap_magnet_to_magnet_mm=T_BOARD + 2 * CLR, rotor_attraction_N=round(F_ATTRACT),
               carrier_t_mm=T_CARRIER, carrier_deflection_mm=round(carrier_deflection(), 3))
    json.dump(rec, open(os.path.join(ROOT, "cad", "actuator", f"{TAG}.json"), "w"), indent=1)
    print(json.dumps(rec, indent=1))
    draw_section(parts, TAG, info, os.path.join(fig_dir, f"cad-{TAG}-section.png"))
    draw_cutaway(parts, TAG, os.path.join(fig_dir, f"cad-{TAG}-cutaway.png"))
    if TAG == "femur":
        draw_iso(parts, joint, os.path.join(fig_dir, f"cad-{joint}-iso.png"))
    print("height", H_TOTAL, "mass", round(total), "g")

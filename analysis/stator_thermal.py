#!/opt/hw-py/bin/python
"""Stator thermal path, as built: where the copper and eddy heat actually goes.

    /opt/hw-py/bin/python analysis/stator_thermal.py

The sizing assumed 1.5 K/W from the copper to the clamp ring.  This computes
it from the board (hw/stator/geometry.json) and the CAD (cad/actuator/femur.json):

  copper -> board faces        distributed heat in a 2.2 mm slab, k_z of FR4
  board faces -> magnet faces  conduction through the 0.5 mm air films (no
                               convection credit; the rotor churns the air, which
                               only helps)
  magnets/rotor -> housing     the rotor cup's outer faces and rim through their
                               0.5 mm air films to the cover, floor and wall
  board -> rim                 radial conduction through the FR4 beyond the coils,
                               bridged by the copper rings, then the clamp
  housing -> ambient           (a) bolted into a body that holds 45 C,
                               (b) free-standing unit in still air,
                               (c) a three-unit hip pod, natural convection

Two board variants: as built (only the live rings and M arcs beyond r 80.2)
and with floating thermal rings on every layer at every free radius (eddy loss
of those rings included, w^2 law).  Writes hw/stator/thermal.json and
docs/design/actuator/thermal-network.png.
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor_options as mo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = json.load(open(os.path.join(ROOT, "hw", "stator", "geometry.json")))
AB = json.load(open(os.path.join(ROOT, "hw", "stator", "asbuilt.json")))
CAD = json.load(open(os.path.join(ROOT, "cad", "actuator", "femur.json")))

K_CU, K_FR4_IP, K_FR4_TP, K_AIR = 385.0, 0.9, 0.3, 0.031     # W/mK; air at ~100 C
T_CU = 3 * mo.OZ
N_L = G["layers"]
T_BOARD = 2.2e-3
CLR = 0.5e-3
R_IN, R_OUT = G["r_in_mm"] * 1e-3, G["r_out_mm"] * 1e-3
R_BOARD = 91.9e-3                                             # v2 board (round 9)
R_CLAMP = 90.4e-3                                             # wall bore: the rim r 90.4-91.9 is clamped (v2)
H_CLAMP = 5000.0                                              # W/m2K, greased aluminium clamp faces
R_CARRIER_OUT, R_DRUM_IN, R_HUB = 85.5e-3, 46.5e-3, 12.5e-3
Z = CAD["z"]
P_ALLOW_DT = mo.T_CU_MAX_PCB - mo.T_AMB                       # 120 - 45 K, copper over ambient
R_PH = AB["R_ph_mohm"] * 1e-3
KT = AB["Kt"]
A_FACE = math.pi * (R_OUT**2 - R_IN**2)                       # coil annulus, one face


def r_radial_fr4(r1, r2):
    """Radial conduction through the full board thickness between two radii."""
    return math.log(r2 / r1) / (2 * math.pi * K_FR4_IP * T_BOARD)


def rim_path(copper):
    """Copper -> rim: the FR4 gaps left uncovered by the union of the copper
    rings' radial extents (a ring on any layer counts as a bridge: optimistic by
    < 0.1 K/W for the 3-layer live rings), in series, then the clamp contact."""
    iv = sorted((rc - w / 2, rc + w / 2) for rc, w in copper)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    R, r = 0.0, R_OUT
    for a, b in merged:
        if a > r:
            R += r_radial_fr4(r, a)
        r = max(r, b)
    if R_CLAMP > r:
        R += r_radial_fr4(r, R_CLAMP)
    A_clamp = 2 * 2 * math.pi * 0.5 * (R_CLAMP + R_BOARD) * (R_BOARD - R_CLAMP)
    R += 1 / (H_CLAMP * A_clamp)
    return R


# ---- the board variants -----------------------------------------------------------------
RINGS_LIVE = sorted([(G["r_m_mm"] * 1e-3, G["m_w_mm"] * 1e-3)] + [(v * 1e-3, G["ring_w_mm"] * 1e-3) for v in G["r_ring_mm"].values()])
W_T, GAP_T = 0.5e-3, 0.15e-3                                   # floating thermal rings, on the layers a radius is free
RINGS_THERMAL = []
rc = R_OUT + 0.6e-3 + W_T / 2                                  # first one just outside the terminal vias' clearance
while rc + W_T / 2 < R_CLAMP - GAP_T:
    if all(abs(rc - rl) > (W_T + wl) / 2 + GAP_T for rl, wl in RINGS_LIVE):
        RINGS_THERMAL.append((rc, W_T))
        rc += W_T + GAP_T
    else:
        rc += 0.05e-3
VARIANTS = {"as built": RINGS_LIVE, "with thermal rings": RINGS_LIVE + RINGS_THERMAL}

# ---- the common paths -------------------------------------------------------------------
R_slab = T_BOARD / (8 * K_FR4_TP * 2 * A_FACE)                 # distributed heat, both faces, mid-plane peak  (K per W)
R_face_air = CLR / (K_AIR * A_FACE)                            # one face to its magnet face
R_faces = R_face_air / 2                                       # two faces in parallel
A_top = math.pi * (R_CARRIER_OUT**2 - R_HUB**2)
A_bot = math.pi * (R_CARRIER_OUT**2 - R_DRUM_IN**2)
H_rim = (Z["carriers"][-1][1] - Z["carriers"][0][0]) * 1e-3
N_STATORS = CAD.get("stators", 1)                               # the boards share the rotor cup's path to the housing
A_rim = 2 * math.pi * R_CARRIER_OUT * H_rim
R_rotor_housing = 1 / (K_AIR / CLR * (A_top + A_bot + A_rim))
R_via_rotor = R_faces + N_STATORS * R_rotor_housing             # copper -> rotor -> housing, per board with every board running

# ---- housing to ambient ---------------------------------------------------------------------
OD, H = CAD["od_mm"] * 1e-3, CAD["height_mm"] * 1e-3
A_unit = 2 * math.pi * (OD / 2)**2 + math.pi * OD * H
H_NAT = 12.0                                                    # W/m2K natural convection + radiation, matte aluminium
A_POD = 0.16                                                    # m2, a three-unit hip pod's exposed surface (est.)
HOUSING = {"body holds 45 C": ("R", 0.0), "body at 60 C (15 K rise)": ("dT", 15.0), "unit alone in still air": ("R", 1 / (H_NAT * A_unit)),
           "three-unit hip pod (per unit)": ("R", 3 / (H_NAT * A_POD))}
ROBOT_BUDGET_W, N_UNITS = 300.0, 18 * N_STATORS                # what a ~1.2 m2 body sheds at ~25 K in still air, shared by every board


def ring_eddy(rings, rpm, B=0.95):
    """Eddy loss of floating copper rings in the fringe field beyond the coils (w^2 law, all layers)."""
    f = G["pole_pairs"] * rpm / 60
    P = 0.0
    for rc, w in rings:
        if rc > 85e-3:
            continue
        vol = 2 * math.pi * rc * w * T_CU * N_L
        P += mo.eddy_pv_strip(f, np.array([B]), w)[0] * vol
    return P


results = {}
for name, rings in VARIANTS.items():
    R_rim = rim_path(rings)
    R_int = 1 / (1 / R_rim + 1 / R_via_rotor)                   # board faces/rim to the housing
    R_total = R_slab + R_int                                    # copper peak to housing
    row = dict(R_slab=R_slab, R_faces=R_faces, R_rotor_housing=R_rotor_housing, R_rim=R_rim, R_to_housing=R_total,
               eddy_rings_W_1000rpm=ring_eddy([r for r in rings if r not in RINGS_LIVE], 1000),
               eddy_rings_W_2500rpm=ring_eddy([r for r in rings if r not in RINGS_LIVE], 2500), cases={})
    for hname, (kind, R_h) in HOUSING.items():
        if kind == "dT":                                        # a fixed housing rise
            P_allow = (P_ALLOW_DT - R_h) / R_total
            R_h = R_h / P_allow
        R_all = R_total + R_h
        P_allow = P_ALLOW_DT / R_all
        P_eddy = AB["ratings"]["1000"]["P_eddy"] + row["eddy_rings_W_1000rpm"]
        P_cu = max(P_allow - P_eddy, 0)
        I = math.sqrt(P_cu / (3 * R_PH))
        frac_rim = (1 / R_rim) / (1 / R_rim + 1 / R_via_rotor)
        T_rotor = mo.T_AMB + R_h * P_allow + R_rotor_housing * (1 - frac_rim) * P_allow
        row["cases"][hname] = dict(R_housing_amb=R_h, R_all=R_all, P_allow=P_allow, I_cont=I, T_cont=KT * I,
                                   T_magnet=T_rotor, share_via_rotor=1 - frac_rim)
    P_avg = ROBOT_BUDGET_W / N_UNITS
    I_avg = math.sqrt(max(P_avg - AB["ratings"]["1000"]["P_eddy"], 0) / (3 * R_PH))
    row["cases"][f"sustained: {ROBOT_BUDGET_W:.0f} W robot budget / {N_UNITS}"] = dict(R_housing_amb=None, R_all=None, P_allow=P_avg, I_cont=I_avg, T_cont=KT * I_avg,
                                                                                    T_magnet=mo.T_AMB + 25 + R_rotor_housing * P_avg, share_via_rotor=None)
    results[name] = row

json.dump(dict(variants=results, assumptions=dict(k_fr4_inplane=K_FR4_IP, k_fr4_through=K_FR4_TP, k_air=K_AIR, h_clamp=H_CLAMP,
                                                   h_nat=H_NAT, A_pod_m2=A_POD, A_unit_m2=A_unit, sizing_R_th=mo.R_TH_NOMINAL)),
          open(os.path.join(ROOT, "hw", "stator", "thermal.json"), "w"), indent=1)

# ---- figure: the network and the continuous torque per case ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axes[0]
ax.set_axis_off()
ab = results["as built"]; th = results["with thermal rings"]
lines = [
    "copper (peak, mid-plane)",
    f"  slab, both faces                 {ab['R_slab']:.2f} K/W",
    f"  faces -> magnets, 2 x 0.5 mm air {ab['R_faces']:.2f} K/W",
    f"  rotor -> housing, 0.5 mm air     {ab['R_rotor_housing']:.2f} K/W",
    f"  rim: FR4 gaps + clamp, as built  {ab['R_rim']:.2f} K/W",
    f"       with thermal rings          {th['R_rim']:.2f} K/W",
    f"copper -> housing, as built        {ab['R_to_housing']:.2f} K/W   (sizing assumed {mo.R_TH_NOMINAL:.1f})",
    f"                   thermal rings   {th['R_to_housing']:.2f} K/W   (+{th['eddy_rings_W_1000rpm']:.1f} W eddy at 1000 rpm, +{th['eddy_rings_W_2500rpm']:.0f} W at 2500)",
    "",
    "housing -> ambient",
] + [f"  {k:34s}{v:.2f} K/W" if kind == "R" else f"  {k:34s}+{v:.0f} K fixed" for k, (kind, v) in HOUSING.items()] + [
    f"  sustained: {ROBOT_BUDGET_W:.0f} W robot budget / {N_UNITS} units = {ROBOT_BUDGET_W/N_UNITS:.0f} W each",
    "", "The thermal rings are not worth it: 75 % of the heat", "already leaves through the 0.5 mm air films to the rotors."]
ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=8.6, va="top", transform=ax.transAxes)
ax.set_title("Thermal resistances from the built geometry", fontsize=10)
ax = axes[1]
cases = list(results["as built"]["cases"])
x = np.arange(len(cases))
for i, (name, col) in enumerate((("as built", "#0f9b8e"), ("with thermal rings", "#d98c3a"))):
    vals = [results[name]["cases"][c]["T_cont"] for c in cases]
    ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, color=col, label=name)
    for xx, v in zip(x + (i - 0.5) * 0.36, vals):
        ax.text(xx, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
ax.axhline(AB["ratings"]["1000"]["T_cont"], color="#b03a2e", ls="--", lw=0.9, label=f"rating at {mo.R_TH_NOMINAL:.1f} K/W: {AB['ratings']['1000']['T_cont']:.2f} N·m")
ax.axhline(135 / (60 * 0.88), color="#555", ls=":", lw=0.9, label="femur needs 2.56 N·m at 60:1")
ax.set_xticks(x); ax.set_xticklabels([c.replace(" (", "\n(").replace(": ", ":\n") for c in cases], fontsize=7)
ax.set_ylabel("continuous motor torque at 1000 rpm (N·m), copper at 120 °C")
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(axis="y", alpha=0.3)
ax.set_title("What the housing's own cooling does to the rating", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "design", "actuator", "thermal-network.png"), dpi=110)

if __name__ == "__main__":
    for name, row in results.items():
        print(f"{name}: slab {row['R_slab']:.2f}, faces {row['R_faces']:.2f}, rotor->housing {row['R_rotor_housing']:.2f}, rim {row['R_rim']:.2f}, "
              f"copper->housing {row['R_to_housing']:.2f} K/W; ring eddy {row['eddy_rings_W_1000rpm']:.1f} W @1000, {row['eddy_rings_W_2500rpm']:.1f} W @2500")
        for c, v in row["cases"].items():
            print(f"   {c:44s} P {v['P_allow']:.0f} W  I {v['I_cont']:.1f} A  T_cont {v['T_cont']:.2f} N·m  magnets {v['T_magnet']:.0f} C")
    print("thermal rings:", len(RINGS_THERMAL), "at", [round(r * 1e3, 2) for r, _ in RINGS_THERMAL])

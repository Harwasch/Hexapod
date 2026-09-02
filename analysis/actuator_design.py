#!/opt/hw-py/bin/python
"""Detailed design of the chosen actuator: A3 (12-layer 3 oz PCB stator, one
stator, two Halbach rotors, in-plane cycloid) with C1 (wound flat coils) as the
upgrade in the same rotors.

    /opt/hw-py/bin/python analysis/actuator_design.py

Builds on analysis/motor_options.py (the sub-agent's electromagnetic model),
analysis/cycloid.py and analysis/actuator_section.py.  Writes
docs/design/08-actuator-design.md and docs/design/actuator/*.png.
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
from matplotlib.patches import Polygon, Wedge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import motor_options as mo                       # noqa: E402
import cycloid as cy                             # noqa: E402
from hexapod_model import YAW_SWING_CAP          # noqa: E402
import actuator_section as sec                   # noqa: E402

DOC = os.path.join(ROOT, "docs", "design", "08-actuator-design.md")
FIG = os.path.join(ROOT, "docs", "design", "actuator")
os.makedirs(FIG, exist_ok=True)

ETA = 0.90 * 0.97                                # cycloid (20 lobes) x capstan; the yaw is cycloid only (0.90)
JOINTS = {name: dict(N=cy.TOTAL[name], N_cyc=cy.RATIOS[name], k2=cy.STAGE2[name], T_cont=cy.JOINT_REQ[name][0], T_peak=cy.JOINT_REQ[name][1], w_swing=cy.SWING[name])
          for name in ("yaw", "femur", "knee")}   # total ratio (cycloid x capstan), joint continuous / peak N·m and swing rad/s, from 01-sizing via cycloid.py

# ----------------------------------------------------------------------------
# 1. The motor design point (A3) and the upgrade (C1), from the study's model
# ----------------------------------------------------------------------------
STACK = mo.PCB_STACKS["PCB 12L 3oz"]
A3 = mo.optimise("pcb", "in-plane", 1, stack=STACK)
C1 = mo.optimise("wire", "in-plane", 1, coil_thick=mo.COIL_THICK)
# a lighter build for the yaw joint: same rotors, thinner magnets (a 16 mm axial budget)
mo.PACKAGING["in-plane-lean"] = dict(r_in=0.050, axial=0.016)
YAW_LEAN = mo.optimise("pcb", "in-plane-lean", 1, stack=mo.PCB_STACKS["PCB 6L 2oz"])

# pole-count trade at the A3 conductor width
POLE_TRADE = []
for p in range(8, 25):
    r = mo.eval_axial("pcb", "in-plane", p, A3["w"], 1, stack=STACK)
    if r:
        POLE_TRADE.append((p, r["n_coils"], r["T_cont"], r["f_e"] * mo.N_NOLOAD_TGT / mo.N_EVAL, r["P_eddy"], r["m_mag"]))

# sensitivity of A3 to the thermal resistance
RTH_TRADE = [(rth, mo.optimise("pcb", "in-plane", 1, stack=STACK, r_th=rth)["T_cont"]) for rth in (1.0, 1.5, 2.0, 3.0)]


def joint_points(m):
    """Per-joint operating points on motor m."""
    out = {}
    for name, j in JOINTS.items():
        t_c = j["T_cont"] / (j["N"] * ETA)
        t_p = j["T_peak"] / (j["N"] * ETA)
        i_c, i_p = t_c / m["Kt"], t_p / m["Kt"]
        out[name] = dict(t_cont=t_c, t_peak=t_p, i_cont=i_c, i_peak=i_p,
                         p_cu_cont=3 * i_c**2 * m["R_ph"], p_cu_peak=3 * i_p**2 * m["R_ph"],
                         margin_cont=m["T_cont"] / t_c, margin_peak=m["T_peak"] / t_p,
                         w_noload=m["n_noload"] * 2 * math.pi / 60 / j["N"], w_needed=j["w_swing"])
    return out


A3_J = joint_points(A3)
C1_J = joint_points(C1)

# ----------------------------------------------------------------------------
# 2. Mechanical: reducer (cycloid.py), bearings, stack-up, mass roll-up
# ----------------------------------------------------------------------------
CYC = {name: cy.design(cy.JOINTS[name][0], cy.JOINTS[name][1], cy.JOINTS[name][2]) for name in JOINTS}   # the cycloid stage at its own (reduced) torque
sec.T_BACKIRON = 3.0                                          # aluminium carrier behind a Halbach ring; 3 mm
sec.STATORS["pcb"].update(t_stator=STACK[2] * 1000, gap=mo.AIR_CLEAR * 1000, t_mag=A3["h_m"] * 1000, label="PCB stator 12-layer 3 oz")
sec.STATORS["wound"].update(t_stator=(mo.COIL_THICK + 2 * mo.COIL_SKIN) * 1000, gap=mo.AIR_CLEAR * 1000, t_mag=C1["h_m"] * 1000)
SEC_A3 = sec.draw("pcb", 1)
SEC_C1 = sec.draw("wound", 1)

RHO_STEEL, RHO_AL = 7.85e-6, 2.7e-6                          # kg/mm³

# ---- round 5: what was actually built and simulated ---------------------------------
import csv
AB = json.load(open(os.path.join(ROOT, "hw", "stator", "asbuilt.json")))
AB16 = json.load(open(os.path.join(ROOT, "hw", "stator", "variants", "16L-2oz", "asbuilt.json")))
AB8 = json.load(open(os.path.join(ROOT, "hw", "stator", "variants", "8t", "asbuilt.json")))
CL = json.load(open(os.path.join(ROOT, "hw", "stator", "closure.json")))
RF = json.load(open(os.path.join(ROOT, "hw", "stator", "rotor_field.json")))
TH = json.load(open(os.path.join(ROOT, "hw", "stator", "thermal.json")))
SUST = [k for k in TH["variants"]["as built"]["cases"] if k.startswith("sustained")][0]
CADJ = {j: json.load(open(os.path.join(ROOT, "cad", "actuator", f"{j}.json"))) for j in ("femur", "knee", "yaw", "femur-1s", "yaw-1s")}
BOM = list(csv.DictReader(open(os.path.join(ROOT, "docs", "design", "bom-actuator.csv"))))
BOM_UNIT = sum(float(r["qty_per_unit"]) * float(r["unit_price_usd"]) for r in BOM)
BOM_BEFORE = sum(float(r["qty_per_unit"]) * float(r["price_before_usd"]) for r in BOM)
CAP = json.load(open(os.path.join(ROOT, "hw", "stator", "capstan.json")))
HK2512 = dict(Cr=11.8e3, C0r=16.3e3, n_lim=6500)                # NTN sheet, docs/reference/ntn-hk2512.pdf
HK3012 = dict(Cr=11.5e3, C0r=17.3e3)                            # PTI HK-series catalogue, docs/reference/pti-hk-series.pdf (round 7)
RB5013 = dict(C=16.7e3, C0=20.9e3, mass=0.27)                    # THK catalogue 382-5E, docs/reference


def mass_rollup(m, n_discs=2, t_disc=8.0):
    R_disc = cy.R_PIN_CIRCLE - 3.0
    disc = math.pi * R_disc**2 * t_disc * RHO_STEEL * 0.80    # 20 % holes
    pins = 56 * math.pi * 1.7**2 * (n_discs * t_disc + 6) * RHO_STEEL
    ecc_shaft = 0.12
    needle_brgs = n_discs * 0.045
    output_brg = 0.18                                          # thin-section four-point contact, Ø~110
    plates = 2 * math.pi * 85**2 * sec.T_HOUSING * RHO_AL
    walls = (2 * math.pi * 85 * 36 * 1.5 + 2 * math.pi * 50 * 36 * 4) * RHO_AL
    encoder_driver = 0.06
    parts = {"motor (magnets, stator, carriers)": m["m_total"], f"cycloid discs ({n_discs} × {t_disc:.0f} mm steel)": n_discs * disc,
             "ring pins": pins, "eccentric shaft": ecc_shaft, "needle bearings": needle_brgs, "output bearing": output_brg,
             "housing plates": plates, "housing walls": walls, "encoder + driver share": encoder_driver}
    return parts, sum(parts.values())


MASS_A3, TOTAL_A3 = mass_rollup(A3)
MASS_YAW, TOTAL_YAW = mass_rollup(YAW_LEAN, n_discs=1, t_disc=8.0)
MASS_A3_THIN, TOTAL_A3_THIN = mass_rollup(A3, n_discs=2, t_disc=5.0)

# ----------------------------------------------------------------------------
# 3. Figures
# ----------------------------------------------------------------------------
def fig_layout():
    """Plan view of the stator PCB (coils) and one rotor face (Halbach blocks)."""
    r1, r2 = A3["r1"] * 1000, A3["r2"] * 1000
    rm1, rm2 = max(r1 - 1, 50), min(r2 + 1, 85)
    n_coils, p = A3["n_coils"], A3["p"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.6))
    # stator: n_coils concentrated spiral coils, three phases
    cols = ["#c0392b", "#2980b9", "#27ae60"]
    for k in range(n_coils):
        th0 = 2 * math.pi * k / n_coils
        dth = 2 * math.pi / n_coils * 0.92
        for t in range(2):                                     # two turns drawn, schematic
            f = 0.15 + 0.35 * t
            th_a, th_b = th0 - dth / 2 * (1 - f), th0 + dth / 2 * (1 - f)
            ra, rb = r1 + (r2 - r1) * f * 0.5, r2 - (r2 - r1) * f * 0.5
            ths = np.linspace(th_a, th_b, 12)
            pts = [(ra * math.cos(x), ra * math.sin(x)) for x in ths] + [(rb * math.cos(x), rb * math.sin(x)) for x in ths[::-1]]
            a.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=cols[k % 3], lw=0.8))
    for rr in (50, 85):
        a.add_patch(plt.Circle((0, 0), rr, fill=False, color="#0f9b8e", ls="--", lw=1))
    a.add_patch(plt.Circle((0, 0), 50, facecolor="#f4f3f0", edgecolor="none", zorder=0))
    a.text(0, 0, "Ø100 bore\n(cycloid)", ha="center", va="center", fontsize=9, color="#555")
    a.set_title(f"Stator PCB face — {n_coils} coils, 3 phases (colour), 12 layers 3 oz\n"
                f"active annulus r {r1:.0f}–{r2:.0f} mm, trace {A3['w']*1e3:.2f} mm / space {mo.TRACE_SPACE*1e3:.2f} mm, {A3['N_ph']:.0f} turns per phase", fontsize=8.5)
    # rotor: Halbach array, 4 segments per pole pair
    nb = A3["n_blocks"]
    arrows = ["↑", "→", "↓", "←"]
    for k in range(nb):
        th0 = 360 * k / nb
        w = Wedge((0, 0), rm2, th0, th0 + 360 / nb * 0.92, width=rm2 - rm1,
                  facecolor=["#f7e0da", "#e8eef7", "#f7e0da", "#e8eef7"][k % 4], edgecolor="#555", lw=0.5)
        b.add_patch(w)
        rc, thc = (rm1 + rm2) / 2, math.radians(th0 + 360 / nb * 0.46)
        b.text(rc * math.cos(thc), rc * math.sin(thc), arrows[k % 4], ha="center", va="center", fontsize=7, rotation=th0 + 360 / nb * 0.46 - 90)
    b.add_patch(plt.Circle((0, 0), rm1 - 2, facecolor="#dcdcdc", edgecolor="#777", lw=0.6))
    b.text(0, 0, "aluminium carrier\n(3 mm, no back-iron)", ha="center", va="center", fontsize=8.5, color="#555")
    b.set_title(f"One rotor face — {nb} NdFeB blocks, Halbach (4 per pole pair, {p} pole pairs)\n"
                f"block {A3['h_m']*1e3:.1f} mm thick, ~{A3['mag_arc']*1e3:.1f} mm arc at r{rm1:.0f}; B̂ gap {A3['B_pk']:.2f} T; arrows = magnetisation", fontsize=8.5)
    for ax in (a, b):
        ax.set_aspect("equal")
        ax.set_xlim(-92, 92)
        ax.set_ylim(-92, 92)
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p_ = os.path.join(FIG, "stator-rotor-layout.png")
    fig.savefig(p_, dpi=72)
    plt.close(fig)
    return p_


def fig_operating():
    """Torque–speed of the A3 motor, reflected to each joint, with the joint requirements."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (name, j) in zip(axes, JOINTS.items()):
        N = j["N"]
        w_nl = A3["n_noload"] * 2 * math.pi / 60 / N
        # simple PMSM envelope: constant torque to the base speed, then the voltage limit line
        t_c = A3["T_cont"] * N * ETA
        t_p = A3["T_peak"] * N * ETA
        # voltage limit: back-EMF plus the IR drop of the current for torque T
        # reach the phase voltage at w_lim(T) = w_nl * (1 - (T/Kt) R / V_ph)
        def w_lim(t_motor):
            return w_nl * (1 - (t_motor / A3["Kt"]) * A3["R_ph"] / mo.V_PH_MAX)
        w = np.linspace(0, w_nl, 200)
        t_v = np.array([max(0.0, min(t_p, A3["Kt"] * (mo.V_PH_MAX / A3["R_ph"]) * (1 - wi / w_nl) * N * ETA)) for wi in w])
        ax.fill_between(w, 0, np.minimum(t_c, t_v), color="#2980b9", alpha=0.25, label="continuous")
        ax.fill_between(w, np.minimum(t_c, t_v), t_v, color="#c0392b", alpha=0.18, label="2 s peak")
        ax.plot(w, t_v, color="#c0392b", lw=1)
        ax.axhline(j["T_cont"], color="#2980b9", ls="--", lw=1)
        ax.axhline(j["T_peak"], color="#c0392b", ls="--", lw=1)
        ax.axvline(j["w_swing"], color="#27ae60", ls=":", lw=1.2)
        ax.text(j["w_swing"] + 0.2, t_p * 0.95, "swing\nspeed", fontsize=7.5, color="#27ae60", va="top")
        ax.set_title(f"{name}: {N}:1 — need {j['T_cont']:.0f} / {j['T_peak']:.0f} N·m (dashed), get {t_c:.0f} / {t_p:.0f}", fontsize=9)
        ax.set_xlabel("joint speed (rad/s)")
        ax.set_ylabel("joint torque (N·m)")
        ax.set_ylim(0, max(t_p, j["T_peak"]) * 1.12)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"A3 motor ({A3['T_cont']:.2f} N·m cont / {A3['T_peak']:.2f} N·m peak, Kt {A3['Kt']:.3f} N·m/A, 48 V) through each joint's cycloid at {ETA*100:.0f} %", fontsize=10)
    fig.tight_layout()
    p_ = os.path.join(FIG, "operating-envelopes.png")
    fig.savefig(p_, dpi=120)
    plt.close(fig)
    return p_


# ----------------------------------------------------------------------------
# 4. Document
# ----------------------------------------------------------------------------
def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def rel(p):
    return os.path.relpath(p, os.path.dirname(DOC)).replace(os.sep, "/")


def write_doc(figs):
    m = A3
    em_rows = [
        ("Topology", "axial flux, ironless PCB stator between two Halbach rotors", "no cogging, no core loss, the cycloid fits in the bore"),
        ("Active annulus", f"r {m['r1']*1e3:.0f}–{m['r2']*1e3:.0f} mm (end turns take r 50–{m['r1']*1e3:.0f} and {m['r2']*1e3:.0f}–85)", "study §2"),
        ("Pole pairs / coils", f"{m['p']} / {m['n_coils']}", "12-slot/10-pole family, winding factor 0.933; see the pole-count trade below"),
        ("Magnets", f"{m['n_blocks']} blocks per rotor, {m['h_m']*1e3:.1f} mm thick, ~{m['mag_arc']*1e3:.1f} mm arc, N48 (Br 1.21 T at 80 °C)", "Halbach, 4 per pole pair, no back-iron"),
        ("Air gap, magnetic", f"{m['g_mag']*1e3:.1f} mm (board {STACK[2]*1e3:.1f} + 2 × {mo.AIR_CLEAR*1e3:.1f} clearance)", ""),
        ("Peak gap field", f"{m['B_pk']:.2f} T", "magnetic-circuit estimate; FEA to confirm"),
        ("Board", f"{STACK[0]} layers × {STACK[1]/mo.OZ:.0f} oz, {STACK[2]*1e3:.1f} mm; trace {m['w']*1e3:.2f} mm, space {mo.TRACE_SPACE*1e3:.2f} mm", "specialist stack-up; 12L 2 oz is the stock fallback (−10 % torque)"),
        ("Turns per phase", f"{m['N_ph']}", ""),
        ("Kt / Kv", f"{m['Kt']:.3f} N·m/A rms / {60/(2*math.pi*m['Kt']):.0f} rpm/V", f"no-load {m['n_noload']:.0f} rpm at 48 V"),
        ("Phase resistance", f"{m['R_ph']*1e3:.1f} mΩ at 120 °C", ""),
        ("Continuous torque", f"**{m['T_cont']:.2f} N·m** at {m['I_cont']:.0f} A rms, J = {m['J']/1e6:.0f} A/mm², {m['P_cu']:.0f} W copper + {m['P_eddy']:.0f} W eddy", f"R_th {mo.R_TH_NOMINAL} K/W stator→ambient, 45 °C ambient, 120 °C copper"),
        ("Peak torque, 2 s", f"**{m['T_peak']:.2f} N·m** at {3*m['I_cont']:.0f} A rms", f"3× continuous current; {m['peak_limit']}"),
        ("Electrical frequency", f"{m['f_e']*mo.N_NOLOAD_TGT/mo.N_EVAL:.0f} Hz at {mo.N_NOLOAD_TGT:.0f} rpm", "sets the drive's PWM and current-loop bandwidth"),
        ("Shear stress, continuous", f"{m['sigma']/1e3:.1f} kPa on the active annulus", "vs 1.5 kPa assumed in 01-sizing: the number the dyno must earn"),
        ("Motor mass", f"{m['m_total']*1e3:.0f} g (magnets {m['m_mag']*1e3:.0f} g)", "excl. housing, bearings, reducer"),
        ("Motor parts cost", f"~${m['cost']:.0f}", "magnets ~${m['m_mag']*1e3*mo.COST_MAGNET_PER_G:.0f}, board ~${mo.COST_PCB['PCB 12L 3oz']:.0f}"),
    ]
    jrows = []
    for name, j in JOINTS.items():
        a, c = A3_J[name], C1_J[name]
        jrows.append((name, f"{j['N']}:1", f"{a['t_cont']:.2f} / {a['t_peak']:.2f}", f"{a['i_cont']:.0f} / {a['i_peak']:.0f}",
                      f"{a['p_cu_cont']:.0f} / {a['p_cu_peak']:.0f}", f"{a['margin_cont']:.2f} / {a['margin_peak']:.2f}",
                      f"{a['w_noload']:.1f} (need {a['w_needed']:.1f})", f"{c['margin_cont']:.2f} / {c['margin_peak']:.2f}"))
    prow = [(p, nc, f"{t:.2f}", f"{fe:.0f}", f"{pe:.0f}", f"{mm*1e3:.0f}") for p, nc, t, fe, pe, mm in POLE_TRADE if p in (8, 10, 12, 14, 16, 19, 22)]
    rrow = [(f"{r:.1f}", f"{t:.2f}", f"{t/A3_J['femur']['t_cont']:.2f}") for r, t in RTH_TRADE]
    mrows = [(k, f"{v*1e3:.0f}") for k, v in MASS_A3.items()] + [("**total, femur / knee unit**", f"**{TOTAL_A3*1e3:.0f}**")]
    mrows2 = [(f"same with 5 mm discs", f"{TOTAL_A3_THIN*1e3:.0f}"), (f"yaw unit: 6-layer board, {YAW_LEAN['h_m']*1e3:.1f} mm magnets, one disc", f"{TOTAL_YAW*1e3:.0f}")]
    cad_mrows = [(k, f"{CADJ['femur']['mass_g'][k]:.0f}", f"{CADJ['yaw']['mass_g'][k]:.0f}" if k in CADJ['yaw']['mass_g'] else "—") for k in CADJ['femur']['mass_g']] + \
                [("**total**", f"**{CADJ['femur']['total_g']:.0f}**", f"**{CADJ['yaw']['total_g']:.0f}**")]
    b12, b16, b8 = AB, AB16, AB8
    brd_rows = [
        ("Layers × copper, turns per layer", f"{b12['layers']} × {b12['copper_oz']:.0f} oz, 10", f"{b8['layers']} × {b8['copper_oz']:.0f} oz, 8", f"{b16['layers']} × {b16['copper_oz']:.0f} oz, 10"),
        ("Finished thickness", f"{b12['t_board_mm']:.2f} mm", f"{b8['t_board_mm']:.2f} mm", f"{b16['t_board_mm']:.2f} mm"),
        ("Who can build it", "PCBWay-class heavy copper", "same", "JLCPCB standard (inner copper ≤ 2 oz)"),
        ("Used on", "every unit (two per unit)", "not used: the yaw swing requirement was lowered instead (round 7)", "fallback"),
        ("Series turns per phase", f"{b12['series_turns_per_phase']}", f"{b8['series_turns_per_phase']}", f"{b16['series_turns_per_phase']}"),
        ("Trace width", f"{0.28:.2f} mm", f"{0.39:.2f} mm", f"{0.28:.2f} mm"),
        ("Phase resistance at 120 °C", f"{b12['R_ph_mohm']:.1f} mΩ (interconnect {b12['R_inter_mohm']:.1f})", f"{b8['R_ph_mohm']:.1f} mΩ", f"{b16['R_ph_mohm']:.1f} mΩ"),
        ("Kt", f"{b12['Kt']:.3f} N·m/A rms", f"{b8['Kt']:.3f}", f"{b16['Kt']:.3f}"),
        ("No-load speed at 48 V", f"{b12['n_noload_rpm']:.0f} rpm", f"{b8['n_noload_rpm']:.0f} rpm", f"{b16['n_noload_rpm']:.0f} rpm"),
        ("Continuous torque, 1000 / 1600 / 2500 rpm", " / ".join(f"{b12['ratings'][k]['T_cont']:.2f}" for k in ("1000", "1600", "2500")) + " N·m",
         " / ".join(f"{b8['ratings'][k]['T_cont']:.2f}" for k in ("1000", "1600", "2500")), " / ".join(f"{b16['ratings'][k]['T_cont']:.2f}" for k in ("1000", "1600", "2500"))),
        ("Peak torque (3× current)", f"{b12['ratings']['1000']['T_peak']:.2f} N·m", f"{b8['ratings']['1000']['T_peak']:.2f}", f"{b16['ratings']['1000']['T_peak']:.2f}"),
        ("Eddy loss at 2500 rpm", f"{b12['ratings']['2500']['P_eddy']:.0f} W", f"{b8['ratings']['2500']['P_eddy']:.0f} W", f"{b16['ratings']['2500']['P_eddy']:.0f} W"),
        ("Copper mass", f"{b12['copper_mass_g']:.0f} g", f"{b8['copper_mass_g']:.0f} g", f"{b16['copper_mass_g']:.0f} g"),
    ]
    rf_rows = [(k, f"{v['w_b_mm']:.0f} × {v['h_m_mm']:.0f}", f"{v['B1_midplane']:.2f}", f"{v['B_peak_surface']:.2f}", f"{v['ratio_to_model']:.2f}",
                f"{v['magnet_mass_g']:.0f}", f"{v['attraction_N']/1e3:.1f}") for k, v in RF.items()]
    th = TH["variants"]["as built"]
    th_rows = [("copper → board faces (2.25 mm slab)", f"{th['R_slab']:.2f}"), ("faces → magnets, two 0.5 mm air films", f"{th['R_faces']:.2f}"),
               ("rotor cup → housing, 0.5 mm air films", f"{th['R_rotor_housing']:.2f}"), ("board rim: FR4 gaps and the clamp", f"{th['R_rim']:.2f}"),
               ("**copper → housing**", f"**{th['R_to_housing']:.2f}**")]
    th_case_rows = [(k, f"{v['P_allow']:.0f}", f"{v['I_cont']:.1f}", f"{v['T_cont']:.2f}", f"{v['T_magnet']:.0f}") for k, v in th["cases"].items()]
    joint_rows = []
    for name, j in JOINTS.items():
        ab = AB
        b1 = RF[CADJ[name]["magnet"]]["B1_midplane"] / RF["rect 30x5x8 N48"]["B1_midplane"]
        for ns, tag in ((2, name), (1, name + "-1s" if name != "knee" else "femur-1s")):
            t_cont = ab["ratings"]["1000"]["T_cont"] * b1 * ns
            t_peak = ab["ratings"]["1000"]["T_peak"] * b1 * ns
            joint_rows.append((f"{name}, {ns} stator{'s' if ns > 1 else ''}" + (" (chosen)" if ns == 2 else ""), f"{j['N_cyc']} × {j['k2']:.0f} = {j['N']:.0f}:1", "10-turn 3 oz", f"{t_cont:.2f} / {t_peak:.2f}",
                               f"{t_cont*j['N']*ETA:.0f} / {t_peak*j['N']*ETA:.0f}", f"{j['T_cont']:.0f} / {j['T_peak']:.0f}",
                               f"{t_cont*j['N']*ETA/j['T_cont']:.2f} / {t_peak*j['N']*ETA/j['T_peak']:.2f}",
                               f"{ab['n_noload_rpm']/j['N']*2*math.pi/60:.1f} (need {j['w_swing']:.1f})", f"{CADJ[tag]['total_g']/1000:.2f}", f"{CADJ[tag]['height_mm']:.0f}"))
    cl_rows = []
    for n, o in CL["options"].items():
        cl_rows.append((n, f"{o['m_fk']:.1f}, {o['h']:.0f}", f"{o['m_yaw']:.1f}, {o['h_yaw']:.0f}", f"{o['m_robot']:.0f}", f"{o['T_fk']:.0f} / {o['need_femur']:.0f}",
                        f"{o['margin_knee']:.2f}", f"{o['margin_yaw']:.2f}", "yes" if o["closes"] else "no"))
    case_rows = []
    for r in CL["cases"]:
        b, a = r["B: 1 stator, 3 oz boards"], r["A-cost: 2 stators, 2 oz boards (chosen)"]
        case_rows.append((r["label"], f"{b['need']['femur']:.0f} / {b['need']['knee']:.0f} / {b['need']['yaw']:.0f}", "yes" if b["closes"] else "no",
                          f"{a['need']['femur']:.0f} / {a['need']['knee']:.0f} / {a['need']['yaw']:.0f}", "yes" if a["closes"] else "no"))
    bom_rows = [(r["item"], r["qty_per_unit"], r["spec"][:90], f"{float(r['qty_per_unit'])*float(r['price_before_usd']):.0f}", f"{float(r['qty_per_unit'])*float(r['unit_price_usd']):.0f}", r["verified"]) for r in BOM]
    cap = CAP["joints"]["knee"]; capg = CAP["geometry"]
    crows = []
    for name, d in CYC.items():
        crows.append((name, f"{d['N']}:1", f"{d['N']+1} × Ø{2*d['r_pin']:.1f}", f"{d['e']:.2f}", f"{d['F_pin']:.0f}", f"{d['sigma_peak']:.0f}", f"{d['F_ecc']:.0f}", f"{d['F_out']:.0f}"))

    doc = f"""# 08 — The actuator: detailed design of the chosen options

*Generated by `analysis/actuator_design.py` on top of the motor study's
model (`analysis/motor_options.py`), the reducer sizing (`analysis/cycloid.py`)
and the stack-up (`analysis/actuator_section.py`). Every number here is a
model number until the dyno says otherwise.*

The study in [07-motor-options-in-envelope.md](07-motor-options-in-envelope.md)
ranked the options that fit the Ø170 × 42 mm pancake. This document takes the
two it recommended and designs them into the envelope with the cycloid, the
bearings and the housing: **A3**, a single 12-layer 3 oz PCB stator between
two Halbach rotors with the reducer in the bore, and **C1**, a wound flat-coil
stator that drops into the same rotors and housing as the torque upgrade.

## 1. What changed against the sizing round

* **One stator per joint, not two.** The study's magnetic circuit gives a
  peak gap field of {m['B_pk']:.2f} T and a thermally derived current density of
  {m['J']/1e6:.0f} A/mm², so a single PCB stator reaches {m['T_cont']:.2f} N·m continuous —
  {m['sigma']/1e3:.1f} kPa on the active annulus against the 1.5 kPa the sizing round
  assumed. That is enough for the femur and knee at their ratios with margin,
  and the two-stator variants do not fit the 42 mm envelope anyway
  (46.8 mm with 4 mm rotor plates; 39.8 mm only with 3 mm plates and 4 mm
  magnets). The sizing model has been updated to the study's values and the
  per-DOF plan is now 45 / 70 / 70:1 (round 6: the heavier robot moved the
  femur and knee to 70:1 and the yaw to 45:1 on a faster 8-turn board).
* **The actuator mass budget does not close at 1.1 kg.** A femur or knee unit
  is {CADJ['femur']['total_g']/1000:.2f} kg in the CAD (§5, §9.3). The magnets alone are {CADJ['femur']['mass_g']['magnets']:.0f} g. This has
  to go back to the mass budget.
* **The whole result rests on the thermal path.** §6 and §9.4: the copper
  to housing path is {TH['variants']['as built']['R_to_housing']:.2f} K/W as built, but the robot as a whole can only
  shed a few hundred watts, so the sustained torque is about half the rating.
* **Round 5 (this revision) designed the parts**: the stator board is laid
  out and DRC-clean with gerbers (`hw/stator`), the rotor field is simulated
  with off-the-shelf blocks, the unit is modelled in build123d with a bill of
  materials, and the numbers in §9 are from those artefacts. Where they
  disagree with the model numbers above, §9 wins.

### Why the radial ring tops the study's chart, and why that is not "radial beats axial"

The study's candidate E, a radial-flux ring of the same outer diameter, posts
the highest torque per litre (14 N·m/L) and the highest peak torque. That is
not a verdict on axial flux; it is a verdict on **iron**. E is a slotted,
laminated, iron-core machine: teeth concentrate the magnet flux across a
sub-millimetre gap and the copper sits deep in slots, so its airgap shear
stress is 12–16 kPa. Our axial candidates are *ironless* — a PCB or a potted
coil in a 3–7 mm magnetic gap — because that is what a small shop can make,
and ironless costs 3–4× in shear stress whatever the flux direction. The
literature's "axial flux is denser" claim is about *iron-core* axial machines
(yokeless segmented-armature, YASA-style) in pancake aspect ratios, where
the disc puts all of its gap area at large radius and drops the stator yoke.
The study's iron-core axial candidate D comes out below E only because the
42 mm envelope leaves it 30 mm for two rotors, two back-irons and a slotted
stator; commercial YASA machines are 60–100 mm thick and reach 10–15 N·m/kg
for the whole motor. So: within what we can build, A3 is the pick; if
buying is acceptable, an iron-core frameless torque motor (radial or a
vendor's axial) of this diameter is the 2–3× torque-density route, at
3–4 kg and $400–1200 per unit, and it still takes the cycloid in its bore.

## 2. Electromagnetic design point — A3

{md(("Quantity", "Value", "Note"), em_rows)}

![Stator and rotor layout]({rel(figs['layout'])})

### Pole count: torque against drive bandwidth

The study's optimiser picks {m['p']} pole pairs for the most torque, which puts
the electrical frequency at {m['f_e']*mo.N_NOLOAD_TGT/mo.N_EVAL:.0f} Hz at top speed. A field-oriented
drive wants 20 or more PWM periods per electrical cycle and a current loop
several times faster than that, so {m['p']} pole pairs means a ≥ 40 kHz PWM,
100 kHz-class current loop — achievable with a modern gate driver and MCU
(the moteus / ODrive Pro class runs 30–48 kHz), but it rules out the cheapest
hobby ESC hardware. Fewer poles cost torque and eddy loss falls:

{md(("Pole pairs", "Coils", "Continuous torque (N·m)", "f_e at 5000 rpm (Hz)", "Eddy loss (W)", "Magnet mass (g)"), prow)}

**Choice: {m['p']} pole pairs as designed, with the driver specified for it**;
14 pole pairs is the fallback if the driver has to be off-the-shelf, at about
{(1 - [t for p, nc, t, fe, pe, mm in POLE_TRADE if p == 14][0] / m['T_cont']) * 100:.0f} % less torque.

## 3. Each joint on this motor

Motor torque is the joint torque over the ratio and the {ETA*100:.0f} % cycloid
efficiency; current is torque over Kt; copper loss is 3 I² R.

{md(("Joint", "Ratio", "Motor torque cont / peak (N·m)", "Current cont / peak (A rms)", "Copper loss cont / peak (W)", "A3 margin cont / peak", "Joint speed no-load, rad/s", "C1 margin cont / peak"), jrows)}

![Operating envelopes]({rel(figs['operating'])})

The femur is the binding joint at {A3_J['femur']['margin_cont']:.2f}× continuous margin; its peak
current, {A3_J['femur']['i_peak']:.0f} A rms, is what the driver's FET stage has to carry for
2 s. The yaw joint runs at half the current of the others and is the
candidate for a lighter build (§5). Every joint has at least
{min(a['w_noload']/a['w_needed'] for a in A3_J.values()):.1f}× the swing speed it needs at 48 V.

## 4. The reducer in the bore

The cycloid sits inside the r < 50 mm bore with its ring pins in a fixed
cylinder that rises from the mounting plate, and **two 8 mm discs 180° apart**
on twin journals (round 5 tried one 10 mm disc to stay inside 42 mm of height;
the review chose bearing life over height, and the unit is now
{CADJ['femur']['height_mm']:.1f} mm tall). **Round 8 split the reduction**: the review found 71 lobes
and pins too many, so the cycloid now does {JOINTS['femur']['N_cyc']}:1 (21 pins of Ø6, 13 mm
pitch, 1.66 mm eccentricity) and a capstan cable stage at the joint does the
other {JOINTS['femur']['k2']:.0f}:1 (§9.9); the yaw stays a direct {JOINTS['yaw']['N_cyc']}:1 cycloid (31 pins).
The table is the cycloid stage at the torque it now sees, i.e. the joint
torque over the capstan ratio. From `analysis/cycloid.py`:

{md(("Joint", "Ratio", "Ring pins", "Eccentricity (mm)", "Peak ring-pin force (N, per disc)", "Hertz, peak (MPa; 1400 allowed)", "Eccentric bearing load, peak (N, per disc)", "Output-pin force, peak (N)"), crows)}

![Cycloid profiles]({rel(os.path.join(FIG, 'cycloid-profiles.png'))})

* **Discs**: two, 8 mm hardened steel (42CrMo4 or 1.2379, 58 HRC), wire-EDM
  profile from `cycloid.profile`, {CADJ['femur']['mass_g']['disc']:.0f} g the pair with eight lightening
  holes each, balancing each other at motor speed. Hertz stress at the femur
  peak is {CYC['femur']['sigma_peak']:.0f} MPa against 1400 allowed.
* **Ring pins**: standard hardened dowels, Ø{2*CYC['femur']['r_pin']:.0f} mm for femur and knee,
  Ø{2*CYC['yaw']['r_pin']:.0f} mm for yaw, in half-grooves in the fixed cylinder's bore. Needle
  rollers on the pins are the first upgrade if efficiency measures below 85 %.
* **Eccentric bearings**: back to one HK2512 drawn-cup needle per disc
  (25 × 32 × 12; Cr {HK2512['Cr']/1e3:.1f} kN, C0r {HK2512['C0r']/1e3:.1f} kN from the NTN sheet in `docs/reference`).
  The review had moved to HK3012 for life; with the capstan stage the cycloid
  sees a quarter of the torque, so the load per disc is {CYC['femur']['F_ecc']/1e3:.1f} kN at the femur
  peak (static margin {HK2512['C0r']/CYC['femur']['F_ecc']:.2f}) and {CYC['femur']['F_ecc']/1e3*JOINTS['femur']['T_cont']/JOINTS['femur']['T_peak']:.1f} kN at the continuous rating:
  L10 {(HK2512['Cr']/(CYC['femur']['F_ecc']*JOINTS['femur']['T_cont']/JOINTS['femur']['T_peak']))**(10/3)/1e6:.0f} million revolutions ({(HK2512['Cr']/(CYC['femur']['F_ecc']*JOINTS['femur']['T_cont']/JOINTS['femur']['T_peak']))**(10/3)/60/1000:.0f} h at 1000 rpm) and
  {(HK2512['Cr']/(0.4*CYC['femur']['F_ecc']*JOINTS['femur']['T_cont']/JOINTS['femur']['T_peak']))**(10/3)/60/1000:.0f} h at 40 % of it. The HK3012 stays in the record as what a direct
  70:1 cycloid would have needed.
* **Output bearing**: RB5013 crossed roller (50 × 80 × 13; C {RB5013['C']/1e3:.1f} kN, C0 {RB5013['C0']/1e3:.1f} kN,
  THK catalogue) in the mounting face; it carries the output flange and the
  drive torque. The coxa's 270 N·m overturning moment at the yaw output is a
  separate, larger bearing in the hip structure, not in the unit.
* **Ratio**: the yaw joint dropped to 30:1 this round (e = {CYC['yaw']['e']:.2f} mm, 31 pins Ø6):
  {AB['ratings']['1000']['T_cont']*RF['rect 30x5x6 N48']['B1_midplane']/RF['rect 30x5x8 N48']['B1_midplane']*30*ETA:.0f} N·m continuous against 55 needed, and 19 rad/s at the joint against the
  8.6 rad/s swing. The three units still share one disc blank.

## 5. Stack-up, mass and cost

The analytic stack-up of round 4 ({SEC_A3[1]['total']:.1f} mm) has been replaced by the
CAD in §9.3: **{CADJ['femur']['height_mm']:.1f} mm tall, Ø{CADJ['femur']['od_mm']:.0f}**. The masses below are
from the model's solids, not estimates.

{md(("Part", "Femur / knee (g)", "Yaw (g)"), cad_mrows)}

Eighteen two-stator actuators are about {(2*CADJ['femur']['total_g'] + CADJ['yaw']['total_g'])*6/1000:.0f} kg (single-stator:
{(2*CADJ['femur-1s']['total_g'] + CADJ['yaw-1s']['total_g'])*6/1000:.0f} kg) against the {18*1.1:.0f} kg the round-1 mass budget carried. The magnets
({CADJ['femur']['mass_g']['magnets']:.0f} g per stator), the rotor cup and the base are the big items. Per unit
that is {AB['ratings']['1000']['T_cont']*JOINTS['femur']['N']*ETA/CADJ['femur']['total_g']*1e3:.0f} N·m/kg continuous and {AB['ratings']['1000']['T_peak']*JOINTS['femur']['N']*ETA/CADJ['femur']['total_g']*1e3:.0f} N·m/kg peak at the joint. 01-sizing
now carries the CAD masses, and §9.7 shows what that does to the loop.

Parts cost per unit from the BOM in §9.6: about **${BOM_UNIT:.0f} at prototype quantities
including a ${[r for r in BOM if r['item']=='Motor driver'][0]['unit_price_usd']} driver**, ${BOM_UNIT*18/1000:.1f}k for eighteen. Machining is
~${sum(float(r['unit_price_usd']) for r in BOM if 'machined' in r['part'] or 'EDM' in r['part'] or 'turned' in r['part']):.0f} of it at one-off prices and falls 2–3× at 50 units.

## 6. Thermal: the assumption everything rests on

The continuous rating is set by the thermal resistance from the copper to
ambient. Round 4 assumed {mo.R_TH_NOMINAL} K/W; §9.4 computes it from the built geometry:
**{TH['variants']['as built']['R_to_housing']:.2f} K/W from the copper to the housing**, and three quarters of that heat
leaves through the 0.5 mm air films to the rotors, not through the board rim.
What the number really depends on is the housing's own path to ambient — the
body it is bolted into.

{md(("R_th stator→ambient (K/W)", "A3 continuous torque (N·m)", "Femur margin"), rrow)}

Three units share one hip pod. At the full continuous rating each stator
board dissipates ~45 W; thirty-six of them are 1.6 kW, which a ~1.2 m² body
in still air cannot shed. The sustained thermal budget of the robot is nearer
300 W, i.e. 8 W per board: {TH['variants']['as built']['cases'][SUST]['T_cont']:.2f} N·m per board, {2*TH['variants']['as built']['cases'][SUST]['T_cont']:.2f} per unit,
{2*TH['variants']['as built']['cases'][SUST]['T_cont']*JOINTS['femur']['N']*ETA:.0f} N·m at the femur, all joints at once, indefinitely. The
per-joint continuous rating is a rating for one joint at a time or for
minutes, not for the whole robot for hours; the gait's average torque is what
the body has to be able to cool. Design rules that follow: the stator rim
clamped, not glued, into the housing; the housing bolted face-to-face to the
hip pod plate; a temperature sensor on every stator; N48**H** magnets (they run
{TH['variants']['as built']['cases']['unit alone in still air']['T_magnet']:.0f} °C in a free-standing unit at its rating); and the dyno test that measures
R_th on the real housing before anything else is trusted.

## 7. C1, the upgrade in the same rotors

The wound flat-coil stator ({C1['n_coils']} coils of {C1['w']*1e3:.1f} mm wire, {mo.COIL_THICK*1e3:.0f} mm thick, potted)
gives {C1['T_cont']:.2f} N·m continuous and {C1['T_peak']:.2f} N·m peak in the same rotors with the
gap opened to {C1['g_mag']*1e3:.1f} mm — the study found it does not beat A3 in-plane once
eddy loss in 0.4 mm wire at {C1['f_e']:.0f} Hz is counted ({C1['P_eddy']:.0f} W of its budget).
It is the fallback if the 12-layer 3 oz board proves hard to source or the PCB
runs hotter than modelled, and it becomes the better option with litz wire or
if the pole count is reduced. Housing rule: the stator carrier is a separate
ring on the same bolt circle, sized for the thicker stator, so the swap is a
part change.

## 8. Electrical interface

* Bus 48 V; per joint a three-phase FOC stage rated {A3_J['knee']['i_cont']:.0f} A rms continuous and
  {A3_J['knee']['i_peak']:.0f} A rms for 2 s (knee), PWM ≥ 40 kHz for the {m['p']}-pole-pair machine,
  phase inductance low (ironless) so switching ripple needs attention.
* One driver board per hip pod, three axes, mounted on the pod plate.
* Absolute encoder on the motor (magnetic, off-axis ring on the rotor) and
  on the output (the cycloid has backlash and the transmission adds
  compliance); temperature sensor on the stator.

## 9. Round 5: the parts, as designed

### 9.1 The stator board — `hw/stator/stator.kicad_pcb`, DRC clean, gerbers in `hw/stator/gerbers`

Generated with KiCad's `pcbnew` API by `hw/stator/make_stator.py`: 36 coils,
15 pole pairs, 10-turn spirals on every layer (odd layers spiral in, even
layers spiral out, joined by a centre via), 0.28 mm trace / 0.15 mm space,
0.6 mm coil-to-coil gap. The winding respects what the study's sheet model
skipped: only coils with identical EMF phasors may be paralleled, so each
phase is four series turns-groups of two parallel coils per 12-coil repeat,
four repeats in parallel through three-layer phase rings at the rim and a
star ring at the bore. `analysis/stator_asbuilt.py` rates the copper that is
actually on the board (resistance element by element, Kt from every radial
leg in the simulated field):

{md(("Quantity", "12 L × 3 oz, 10 turns (canonical)", "12 L × 3 oz, 8 turns (yaw)", "16 L × 2 oz (fallback)"), brd_rows)}

The 8-turn board has the same copper mass and nearly the same torque at
1000 rpm, but its wider traces lose {b8['ratings']['2500']['P_eddy']/b12['ratings']['2500']['P_eddy']:.1f}× the eddy power at speed. It would have
bought the 7.7 rad/s the 1 m/s walk asked of the yaw at 45:1; the review chose
to lower that requirement to the {YAW_SWING_CAP:.1f} rad/s the shared 10-turn board gives
instead, so every unit carries the same board.

![Stator F.Cu]({rel(os.path.join(FIG, 'stator-F_Cu.svg'))})

![Stator In9.Cu, the M-arc and star layer]({rel(os.path.join(FIG, 'stator-In9_Cu.svg'))})

### 9.2 The rotor field with blocks you can buy — `analysis/rotor_field.py`

Two facing four-segment Halbach rings of rectangular blocks, 2-D
surface-charge model at the mean radius, gap {AB['h_m_mm']*0+3.2:.1f} mm magnet face to magnet face:

{md(("Blocks", "w × h (mm)", "B₁ at the board (T)", "B at the magnet face (T)", "vs the study's model", "Magnet mass (g)", "Rotor attraction (kN)"), rf_rows)}

![Rotor field]({rel(os.path.join(FIG, 'rotor-field.png'))})

Rectangular 5 mm blocks fill 71 % of the 7 mm segment at the mean radius and
give 84 % of the trapezoid field; 8 mm thick blocks (30 × 8 × 5, a standard
size stood on edge) buy that back. **Chosen: 30 × 8 × 5 N48H for the femur and
knee, 30 × 6 × 5 for the yaw.** Half the blocks are magnetised through their
5 mm dimension, half through the 8 mm — specify both. The {RF['rect 30x5x8 N48']['attraction_N']/1e3:.1f} kN
attraction between the rotors sizes the 4.5 mm carriers ({CADJ['femur']['carrier_deflection_mm']:.2f} mm deflection).

### 9.3 The unit — `cad/actuator/actuator.py`, STEP and STL in `build/cad/`

![Femur unit, section]({rel(os.path.join(FIG, 'cad-femur-section.png'))})

![Femur unit, cutaway]({rel(os.path.join(FIG, 'cad-femur-cutaway.png'))})

![Femur unit from below]({rel(os.path.join(FIG, 'cad-femur-iso.png'))})

The topology that makes "the reducer inside the motor" buildable: the fixed
pin cylinder rises from the mounting plate through the **open bottom of the
rotor cup** (a bottom carrier ring, a middle ring with magnets on both
faces, a drum through both board bores, a top carrier with the hub); the two
discs sit inside the cylinder and the two-stator motor stack sits above them,
because the hub has to clear the upper disc; the output flange rides the crossed roller in the mounting face and
exits through it, so the output and the mounting are on the same side, like a
flat harmonic unit. The shaft runs in a 6905 in the flange hub and a 6802 in
the cover. **{CADJ['femur']['height_mm']:.1f} mm tall against the 42 mm target (the review accepted the
height for the second disc); Ø{CADJ['femur']['od_mm']:.0f} against the Ø170 target (accepted)**, because
the stator's phase rings and pads sit at r 82–86 outside the magnets and the
board is clamped at its rim. The 12 mm below the rotor at r > 50 is empty; it
is where a future revision would put the driver if a slimmer one than the
ODrive S1 is chosen.
Assembly notes: the cup is two machined parts (the bottom ring bolts to the
drum after the board is in); the magnets go on with a printed jig because
neighbouring Halbach blocks repel; the pins drop into half-grooves.

### 9.4 Where the heat goes — `analysis/stator_thermal.py`

{md(("Path", "K/W"), th_rows)}

{md(("Housing condition", "Loss allowed (W)", "Current (A rms)", "Motor torque, cont. (N·m)", "Magnet temp (°C)"), th_case_rows)}

![Thermal network]({rel(os.path.join(FIG, 'thermal-network.png'))})

The copper-to-housing path is better than assumed because the 0.5 mm air
films to the rotors conduct; the housing-to-ambient path is the problem, and
it is a body-level one. Floating "thermal rings" on the board were evaluated
and are not worth it.

### 9.5 Each joint on the built motor

The requirement column is what 01-sizing derives **at the robot mass the CAD
implies** ({CL['options']['A-cost: 2 stators, 2 oz boards (chosen)']['m_robot']:.0f} kg with the two-stator units), not the 49 kg of round 1;
§9.7 explains why that is the right mass to check against.

{md(("Joint", "Cycloid × capstan", "Board", "Motor cont / peak (N·m)", "Joint cont / peak (N·m)", "Needed at the CAD mass", "Margin", "Joint speed no-load (rad/s)", "Unit mass (kg)", "Height (mm)"), joint_rows)}

With the cost-down boards (two 8-layer 2 oz per position, §9.10) every motor
torque above scales by {AB16['ratings']['1000']['T_cont']/AB['ratings']['1000']['T_cont']:.2f}; §9.7 carries that case as "A-cost".

### 9.6 Bill of materials — `docs/design/bom-actuator.csv`

{md(("Item", "Qty", "Spec", "Round 7 ($)", "Round 8 cost-down ($)", "Price verified"), bom_rows)}

**${BOM_UNIT:.0f} per unit after the cost-down (${BOM_BEFORE:.0f} before), ${BOM_UNIT*18/1000:.1f}k for eighteen** at
20-unit quantities. Verified prices are marked; the rest are estimates to be
replaced by quotes. The capstan sector, rope and tensioner are costed here
although they live on the joint.

### 9.9 The second stage: a capstan at the joint — `analysis/capstan.py`

![Capstan stage]({rel(os.path.join(FIG, 'capstan.png'))})

A Ø{2*capg['r_drum_mm']:.0f} drum on the unit's output face (below the mounting face, inside the
hip pod, on the yaw axis) drives a Ø{2*capg['r_sector_mm']:.0f} sector at the femur or knee pivot
through one pre-tensioned {capg['rope']['name']} rope, {capg['n_wrap']:.0f} wraps on the drum,
spliced eyes at the sector: {capg['ratio']:.0f}:1, no backlash, ~97 % efficient. The cycloid
therefore does {JOINTS['femur']['N_cyc']}:1 with 21 pins and sees a quarter of the joint torque, which
is what let the needle bearings go back to HK2512 and the discs to laser-cut
steel.

* Rope tension at the knee: {cap['F_rope_cont']/1e3:.1f} kN continuous, {cap['F_rope_peak']/1e3:.1f} kN peak, against the
  datasheet's {capg['rope']['break_kN']:.1f} kN minimum spliced strength: safety factor {cap['sf_cont']:.1f} / {cap['sf_peak']:.1f}.
  Steel wire at this tension would need a drum 25–40× its diameter; Dyneema
  is happy at D/d = {cap['D_over_d']:.0f}.
* Joint stiffness from the rope's stretch: {cap['k_joint_Nm_per_rad']/1e3:.0f} kN·m/rad, a wind-up of
  {cap['wind_up_deg_at_cont']:.2f}° at the continuous torque. That is compliance the controller sees;
  the output-side encoder on the joint (§8) is what makes it harmless.
* SK78 creeps {capg['rope']['creep_pct_per_yr_at_20pct']:.1f} % per year at 20 % load: a tensioner and a re-tension
  interval. The rope loses strength above {capg['rope']['T_critical_C']:.0f} °C, so it must not touch the
  housing, which runs at 60–100 °C at the rating.
* The drum sits on the yaw axis and the sector on the coxa, so a yaw rotation
  winds the drum by the yaw angle: femur angle = sector angle − yaw × (r_drum /
  r_sector). A known coupling, compensated in software.
* The sector at Ø{2*capg['r_sector_mm']:.0f} for {capg['joint_range_deg']:.0f}° of travel is the largest new part; it belongs
  to the transmission chunk (M1) and is costed in the BOM so the stage is complete.

### 9.10 Cost-down — what changed on every line

| Line | Round 7 | Round 8 | How |
|---|---|---|---|
| Stator boards | 2 × 12L 3 oz, $300 | 4 × 8L 2 oz JLCPCB, $80 | two thin boards per position, wired in parallel; −11 % torque, taken back by the 80:1 total ratio |
| Cycloid discs | 2 × wire-EDM, $180 | 2 × laser-cut + hardened, $44 | 20 lobes of 13 mm pitch tolerate laser-cut accuracy |
| Eccentric bearings | HK3012, $14 | HK2512, $8 | a quarter of the torque on the cycloid |
| Output bearing | THK RB5013, $196 | Chinese CRB5013, $45 | qualify on its own datasheet; THK as alternate |
| Base | 7075 billet, $180 | 6061, 20 pcs, $95 | die-cast at volume |
| Rotor cup | 7075 one-piece, $150 | turned drum + laser-cut plates, $60 | bonded, as before |
| Clamp rings, cover | machined, $125 | laser-cut plates, $33 | |
| Shaft, flange | $130 | $70 | straight Ø25 stock, 6061 flange |
| Driver | ODrive S1, $149 | ODrive-compatible board, $45 | ODrive S1 as the qualified alternate; a custom board at volume |
| Magnets | $144 | $72 | volume quote, 4320 pcs |
| Capstan stage | — | $61 | drum, sector, rope, tensioner |
| Encoder, misc | $27 | $18 | |

The magnets and the boards are now the two biggest lines and both are
physics, not machining: fewer magnets means less torque, and a cheaper board
means less copper. Everything else is a manufacturing-route choice that a
quote can confirm. Not done here: a custom driver board, casting the base,
and the volume pricing that would take the unit under $500.

### 9.7 Does the robot close? — `analysis/closure.py`

Renegotiating the mass budget is not a number to write down; it is a loop.
Joint torque is proportional to robot mass ({CL['torque_per_kg']['femur']:.2f} N·m/kg at the femur,
{CL['torque_per_kg']['knee']:.2f} at the knee, {CL['torque_per_kg']['yaw']:.2f} at the yaw, for the confirmed leg
and the walking load case), the robot carries eighteen units, and a unit gives
what its motor and ratio give. The fixed point for each unit option:

{md(("Configuration", "Femur/knee unit (kg, mm)", "Yaw unit (kg, mm)", "Robot (kg)", "Femur gives / needs (N·m)", "Knee margin", "Yaw margin", "Closes?"), cl_rows)}

![Closure]({rel(os.path.join(FIG, 'closure.png'))})

The single-stator unit cannot close the requirement as defined at any mass:
the torque it needs grows with the mass it adds faster than its own torque.
Two stators per unit ({CADJ['femur']['height_mm']:.0f} mm tall, {CADJ['femur']['total_g']/1000:.1f} kg) close it at
about {CL['options']['A-cost: 2 stators, 2 oz boards (chosen)']['m_robot']:.0f} kg. The other way to close would have been the definition of
"continuous": the walking load case is dyn 1.5 on three legs, on a 30° slope,
accelerating at 1 m/s², over the whole routine working volume, all at once.
Against gentler definitions:

{md(("Continuous load case", "B: 1 stator, femur / knee / yaw needed at " + f"{CL['options']['B: 1 stator, 3 oz boards']['m_robot']:.0f} kg", "B closes?", "A: 2 stators, needed at " + f"{CL['options']['A-cost: 2 stators, 2 oz boards (chosen)']['m_robot']:.0f} kg", "A closes?"), case_rows)}

**The review chose A** (round 6): two stators everywhere, {CADJ['femur']['height_mm']:.0f} mm units, a
~{CL['options']['A-cost: 2 stators, 2 oz boards (chosen)']['m_robot']:.0f} kg robot, the requirement as written. Round 8 then swapped the
3 oz boards for two 8-layer 2 oz JLCPCB boards per position (11 % less torque,
a quarter of the price) and raised the total ratio to 80:1 through the
capstan stage to keep the margin ("A-cost" above). The single-stator unit
stays in the CAD as the `-1s` variant (§9.3) and B stays here as the record of
the lighter robot that was not chosen. The 62 mm stack no longer fits the
200 mm body slab (three units and two gaps are 202 mm), so the slab is now
220 mm in `hexapod_model.py`; the drawings in 06 carry it.

### 9.8 Open items from this round (updated in round 8)

1. **OD 186, not 170** (§9.3): accepted by the review.
2. **Mass and closure** (§9.7): the review chose to renegotiate the mass
   budget; 01-sizing now carries the CAD masses, and the loop only closes
   with two stators per unit (A) or a gentler continuous load case (B).
   **This is the decision this round asks for.**
3. **Eccentric bearings**: the capstan stage settled this. With a quarter of
   the torque on the cycloid, HK2512 on a straight Ø25 shaft has a static
   margin of {HK2512['C0r']/CYC['femur']['F_ecc']:.1f} at the femur peak and {(HK2512['Cr']/(CYC['femur']['F_ecc']*JOINTS['femur']['T_cont']/JOINTS['femur']['T_peak']))**(10/3)/60/1000:.0f} h L10 at the
   continuous rating (§4). The HK3012 decision of round 7 is withdrawn.
4. **Sustained torque is set by the body's cooling**, not the motor (§6);
   the review accepted it: the gait is designed around a ~300 W average.
7. **The yaw joint was speed-limited by the 48 V bus**: 7.7 rad/s at 45:1
   needs 3300 rpm. The review lowered the yaw swing-speed requirement to the
   {YAW_SWING_CAP:.1f} rad/s the shared 10-turn board gives (`YAW_SWING_CAP` in
   `hexapod_model.py`); the 8-turn board stays in `hw/stator/variants/8t` as the
   record of the alternative.
5. **Heavy copper**: the canonical board needs a 3 oz multilayer house; the
   2 oz fallback costs 11 % torque and one ratio step.
6. **Magnet grade** N48H, not N48, because a free-standing unit runs its
   magnets near 100 °C at the rating.

## 10. What the first unit has to prove

1. R_th of the stator in its housing, on the bench, with a heater and then
   with current: the continuous rating is that number.
2. Kt and R_ph against this model (±10 % is a pass).
3. Cycloid efficiency and noise at 60:1 with the EDM disc and plain pins.
4. Demagnetisation margin at the 2 s peak, hot.
5. Mass, against the roll-up above.
"""
    with open(DOC, "w") as f:
        f.write(doc)


if __name__ == "__main__":
    figs = {"layout": fig_layout(), "operating": fig_operating()}
    write_doc(figs)
    print("wrote", DOC)
    print(f"A3: T {A3['T_cont']:.2f}/{A3['T_peak']:.2f}, p {A3['p']}, coils {A3['n_coils']}, Kt {A3['Kt']:.3f}, I {A3['I_cont']:.0f} A, mass {A3['m_total']*1e3:.0f} g; total {TOTAL_A3*1e3:.0f} g, thin {TOTAL_A3_THIN*1e3:.0f} g, yaw lean {TOTAL_YAW*1e3:.0f} g")
    print("femur margin", {k: round(v, 2) for k, v in A3_J["femur"].items()})
    print("Rth trade", RTH_TRADE)

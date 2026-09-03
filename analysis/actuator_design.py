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
AB3 = json.load(open(os.path.join(ROOT, "hw", "stator", "variants", "3x6L-2oz", "asbuilt.json")))
AB8 = json.load(open(os.path.join(ROOT, "hw", "stator", "variants", "8t", "asbuilt.json")))
CL = json.load(open(os.path.join(ROOT, "hw", "stator", "closure.json")))
RF = json.load(open(os.path.join(ROOT, "hw", "stator", "rotor_field.json")))
TH = json.load(open(os.path.join(ROOT, "hw", "stator", "thermal.json")))
SUST = [k for k in TH["variants"]["as built"]["cases"] if k.startswith("sustained")][0]
CADJ = {j: json.load(open(os.path.join(ROOT, "cad", "actuator", f"{j}.json"))) for j in ("femur", "knee", "yaw", "femur-1s", "yaw-1s")}
BOM = list(csv.DictReader(open(os.path.join(ROOT, "docs", "design", "bom-actuator.csv"))))
BOM_UNIT = sum(float(r["qty_per_unit"]) * float(r["unit_price_usd"]) for r in BOM)
BOM_BEFORE = sum(float(r["qty_per_unit"]) * float(r["price_before_usd"]) for r in BOM)
BOM_100 = sum(float(r["qty_per_unit"]) * float(r["price_100_usd"]) for r in BOM)
CAP = json.load(open(os.path.join(ROOT, "hw", "stator", "capstan.json")))
CS = json.load(open(os.path.join(ROOT, "hw", "stator", "cost_search.json")))
MM = json.load(open(os.path.join(ROOT, "hw", "stator", "motor_market.json")))
TO = json.load(open(os.path.join(ROOT, "hw", "stator", "transmission_options.json")))
TC = json.load(open(os.path.join(ROOT, "hw", "stator", "topology_compare.json")))
FM = json.load(open(os.path.join(ROOT, "hw", "stator", "frameless_motor.json")))
FCAD = json.load(open(os.path.join(ROOT, "cad", "actuator", "frameless.json"))) if os.path.exists(os.path.join(ROOT, "cad", "actuator", "frameless.json")) else None
ARR = json.load(open(os.path.join(ROOT, "hw", "arrangement.json"))) if os.path.exists(os.path.join(ROOT, "hw", "arrangement.json")) else None
SO = json.load(open(os.path.join(ROOT, "hw", "stator", "single_stator_opt.json")))       # round 13: the single-stator sweep at Ø190
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


def so_section(rel_fn, note=False):
    """Round 13: the single-stator sweep (analysis/single_stator_opt.py).  One text for the
    08 section and for docs/design/actuator/single-stator-opt.md; rel_fn makes the image paths."""
    ch, R, U, c80, c100, cf, ck = SO["chosen"], SO["chosen_rating"], SO["chosen_unit"], SO["closure_80"], SO["closure_100"], SO["closed_form"], SO["checks"]
    lim, conv = SO["limits"], SO["conventions"]
    ba, bj, bd, bk, bc = SO["best_any"], SO["best_jlcpcb_any_speed"], SO["best_per_dollar"], SO["best_per_kg"], SO["best_closure"]

    def brow(r, label=None):
        if r is None:
            return None
        return (label or r["stack"], f"{r['t_board_mm']:.2f}", f"{r['coils']} / {2*r['pp']}", f"{r['turns']} × {r['trace_mm']:.2f}", f"{r['h_m']:.0f}", f"{r['B_pk_mean']:.2f}",
                f"{r['Kt']:.3f}", f"{r['R_ph_mohm']:.0f}", f"{r['I_cont']:.1f}", f"{r['P_eddy']:.0f}", f"**{r['T_cont']:.2f}**", f"{r['n_noload']:.0f}" + ("" if r["swing_ok"] else " ✗"),
                f"{r['m_motor_kg']:.2f}", f"{r['cost20']:.0f} / {r['cost100']:.0f}", f"{r['margin_knee']:.2f}")
    hdr = ("Stack", "t (mm)", "Coils / poles", "Turns × trace (mm)", "Blocks (mm)", "B (T)", "Kt (N·m/A)", "R (mΩ)", "I (A)", "Eddy (W)", "T at 1000 rpm (N·m)",
           "No-load rpm (✗ = under the 2880 swing)", "Stator + rotor (kg)", "$ at 20 / 100", "Knee margin at 80:1")
    fast_rows = [brow(p["best_swing_ok"]) for p in SO["per_stack"] if p["best_swing_ok"]]
    any_rows = [brow(p["best_any"]) for p in SO["per_stack"]]
    rate_rows = [(k, f"{v['f_e']:.0f}", f"{v['P_eddy']:.1f}", f"{v['P_cu']:.1f}", f"{v['I_cont']:.1f}", f"{v['T_cont']:.2f}", f"{v['T_peak']:.2f}", f"{v['n_at_cont']:.0f}") for k, v in R["ratings"].items()]
    relief_rows = [(r["label"], f"{max(r['need']['femur'], r['need']['knee']):.0f} / {r['need']['yaw']:.0f}", f"{r['margin_fk']:.2f} / " + (f"{r['margin_yaw']:.2f}" if r['margin_yaw'] < 100 else "—"),
                    "yes" if r["closes_80"] else "no", "yes" if r["closes_100"] else "no") for r in SO["relief"]]
    sens_rows = [(k, f"{v['T_cont']:.2f}", f"{v['T_cont']/R['ratings']['1000']['T_cont']-1:+.1%}") for k, v in SO["sensitivity"].items()]
    fc = ck["field_vs_rotor_field"]
    check_rows = [(c["case"], f"{c['B1_here']:.3f} / {c['B1_rotor_field']:.3f}", f"{c['ratio_here']:.3f} / {c['ratio_rotor_field']:.3f}", f"{c['pull_here_N']:.0f} / {c['pull_rotor_field_N']:.0f}") for c in fc]
    kc, ac = ck["kt_vs_asbuilt"], SO["asbuilt_check"]
    n_sw = ch["n_swing"]
    _drc = os.path.join(ROOT, "hw", "stator", "variants", "1s-opt", "drc.json")
    drc = json.load(open(_drc)) if os.path.exists(_drc) else None
    drc_txt = (f"DRC with kicad-cli {drc['kicad_version']} at error severity: **{len(drc['violations'])} violations**, {len(drc.get('unconnected_items', []))} unconnected "
               f"(`hw/stator/variants/1s-opt/drc.json`)") if drc else "DRC not yet run on the variant"
    swing_factor = c80['ratio_fk_to_close'] * conv['w_swing_femur'] * 60 / (2 * math.pi) / R['n_noload_rpm']
    ps16 = [p for p in SO["per_stack"] if p["stack"] == "16L 2oz"][0]["best_swing_ok"]
    head = "" if not note else f"""# The optimised single-stator motor at Ø190 — round 13

Generated by `analysis/actuator_design.py` from `hw/stator/single_stator_opt.json`
(`analysis/single_stator_opt.py`); the same text is §9.15 of `08-actuator-design.md`.
The board is `hw/stator/variants/1s-opt/stator.kicad_pcb`.

"""
    return head + f"""### 9.15 Round 13: optimised single-stator motor at Ø190 — `analysis/single_stator_opt.py`

Asked: the best single-stator PCB motor for the unit with a hard limit of
**{lim['od_max_mm']:.0f} mm over the housing wall**. The wall is {lim['wall_r_mm']:.0f} mm of radius outside
the board (§9.3), so the board is Ø{lim['board_od_mm']:.0f} and every rim radius of layout v2 —
the coils' outer edge, the terminal vias, the M arcs, the three phase rings, the
pads and the edge — moves inward together by {-lim['rim_shift_mm']:.1f} mm (`make_stator.py --od {lim['board_od_mm']:.0f}`);
the magnet rings move with the coils. The clamp, the bolt circle and the wall
section of the CAD are unchanged, which is why the wall was kept rather than thinned:
the Ø4.4 bolt holes at r 93.2 would break out of a 3 mm wall.

The sweep lays out every board with the generator (coil count, turns, trace
width; {SO['n_candidates']} candidates after the stack, magnet, and ratio options) and rates each
one the way `stator_asbuilt.py` rates the canonical board: the same {conv['P_allow_W']:.0f} W
thermal budget ({conv['T_cu_max']:.0f} °C copper, {conv['T_amb']:.0f} °C ambient, {conv['R_th']:.1f} K/W), the same
element-by-element resistance, the same eddy loss, and Kt from every radial
leg in the field. Two things are new. The rotor field is the 2-D block model of
§9.2 run at each pole count and gap (it reproduces `rotor_field.py` exactly on
the same inputs — table at the end — and shows that `rotor_field.json` was
written at the v1 coil span and reads 3 % high for v2), and Kt is the closed
form of the numeric average ({kc['Kt_here']:.4f} against {kc['Kt_asbuilt']:.4f} N·m/A on the canonical
board with the same field ratio forced). The levers:

* **layers × copper**: 6, 8, 10, 12, 14, 16, 20 layers at 2 oz (JLCPCB), bonded
  pairs of 6, 8 and 10-layer boards, and 12 L 3 oz as the non-JLCPCB reference.
  Below 12 layers the generator now gives the phase rings one or two layers
  instead of three (`set_layers`), and the rating charges for it.
* **coils / poles**: 24/20, 36/30, 48/40. At 48/40 a 5 mm block no longer
  fits the Halbach segment at the ring's inner radius, so those rotors need
  {[r for r in SO['top40'] if r['coils']==48][0]['w_b'] if any(r['coils']==48 for r in SO['top40']) else 3.5:.1f} mm custom blocks; at 24/20 the 5 mm block fills under half the segment and an
  8 mm custom block is tried as well.
* **turns per layer** from 4 to what the leg fits, and **trace width** at
  100 / 70 / 50 / 35 % of the leg fill — the lever that lets a fast board keep
  its eddy loss down.
* **magnets** 30 × 5 × 6 / 8 / 10 mm N48; the rotor pull sizes the carrier
  plates (same 0.06 mm deflection as the CAD's 4.5 mm plate under 2.3 kN).
* **r_in** 53 → 52, the **leg gap** 0.6 → 0.4 and the **clearance**
  0.35 / 0.5 / 0.7 mm, as sensitivities on the chosen board.

The one lever that decides the outcome is not in that list: **the 48 V bus.**
The femur swing needs {conv['w_swing_femur']:.1f} rad/s at the joint, {n_sw:.0f} rpm at 80:1
({n_sw*1.25:.0f} at 100:1), and a board whose back-EMF allows that at 48 V has to have
few turns — and few turns with the trace filling the leg means wide traces and
an eddy loss that eats the thermal budget at 250 Hz. Every prior round tabled
this as "2.9 rad/s (need 3.8)" and moved on. Here it is the constraint:

{md(hdr, fast_rows)}

*Best of each stack that reaches the femur swing speed at 48 V (2 oz OTS blocks; the 12 L 3 oz row is the non-JLCPCB reference), all at r 53–{lim['r_out_mm']:.1f}. Knee margin at the robot mass the unit implies (§9.7's fixed point).*

The same stacks with the speed constraint dropped — what the board can do if
the femur swing is slowed or the bus raised:

{md(hdr, any_rows)}

![Single-stator sweep]({rel_fn('1s-opt-sweep.png')})

**Chosen: {ch['stack']}, {ch['coils']} coils / {2*ch['pp']} poles, {ch['turns']} turns × {ch['trace_mm']:.2f} mm, 30 × 5 × {ch['h_m']:.0f} mm blocks**,
r {ch['r_in']:g}–{ch['r_out']:.1f}, board {ch['t_board_mm']:.2f} mm, magnet gap {R['g_mag_mm']:.2f} mm — the most continuous torque at
1000 rpm of any JLCPCB single board that reaches {n_sw:.0f} rpm at 48 V. Kt {R['Kt']:.3f} N·m/A rms,
R {R['R_ph_mohm']:.0f} mΩ at 120 °C, B {R['B_pk_mean']:.2f} T, no-load {R['n_noload_rpm']:.0f} rpm, {U['m_motor_g']:.0f} g of stator + rotor
(board {U['m_board_g']:.0f}, magnets {U['m_magnets_g']:.0f}, carriers {U['m_carriers_g']:.0f} at {U['t_carrier_mm']:.1f} mm), {U['m_unit_fk_kg']:.2f} kg femur unit, {U['h_unit_mm']:.1f} mm tall
(the reducer sets the height; the band is {U['h_band_mm']:.1f} mm), boards + magnets + cup
${U['cost']['20']['total']:.0f} at 20 units / ${U['cost']['100']['total']:.0f} at 100 (board ${U['cost']['20']['boards']:.0f} / ${U['cost']['100']['boards']:.0f} from the reviewer's JLCPCB quote
scaled by layer count, magnets ${U['cost']['20']['magnets']:.0f} / ${U['cost']['100']['magnets']:.0f}). The board is generated by `make_stator.py --coils {ch['coils']}
--pp {ch['pp']} --turns {ch['turns']} --trace {ch['trace_mm']:g} --layers {ch['layers']} --oz {ch['oz']:g} --od {lim['board_od_mm']:.0f}` into `hw/stator/variants/1s-opt`;
{drc_txt}.

| Speed (rpm) | f_e (Hz) | Eddy (W) | Copper (W) | I (A rms) | T continuous (N·m) | T peak, 2 s (N·m) | Speed at I_cont, 48 V (rpm) |
|---|---|---|---|---|---|---|---|
{chr(10).join('| ' + ' | '.join(r) + ' |' for r in rate_rows)}

![Torque–speed envelope]({rel_fn('1s-opt-envelope.png')})

![Section of the single-stator unit]({rel_fn('1s-opt-section.png')})

![Chosen board, F.Cu]({rel_fn('1s-opt-F_Cu.svg')})

**The alternatives the sweep ranks above it on one axis each.** The most
torque of any board in the Ø190 housing is the {ba['stack']} reference with {ba['turns']} turns and
{ba['h_m']:.0f} mm blocks, {ba['T_cont']:.2f} N·m — no-load {ba['n_noload']:.0f} rpm, so it cannot swing the femur at 48 V,
and it is not a JLCPCB board. The most torque JLCPCB can build is {bj['stack']}, {bj['turns']} turns,
{bj['h_m']:.0f} mm blocks: {bj['T_cont']:.2f} N·m at {bj['n_noload']:.0f} rpm no-load. Best per dollar (speed-feasible):
{bd['stack']}, {bd['turns']} turns, {bd['h_m']:.0f} mm, {bd['T_cont']:.2f} N·m for ${bd['cost20']:.0f}; best per kilogram: {bk['stack']}, {bk['turns']} turns,
{bk['h_m']:.0f} mm, {bk['T_cont']:.2f} N·m from {bk['m_motor_kg']:.2f} kg. Trace width matters more than turns: on
the chosen stack the torque is flat in turns while the no-load speed is not
(the sweep figure's second panel), so the fast board is the few-turn,
narrow-trace one.

**Closure.** At §9.7's fixed point (29.2 kg + 12 femur/knee + 6 yaw units) the
chosen unit implies a **{c80['m_robot_kg']:.0f} kg robot**, which needs {c80['need']['femur']:.0f} / {c80['need']['knee']:.0f} / {c80['need']['yaw']:.0f} N·m
continuous at femur / knee / yaw. Through 80:1 at {conv['eta_cyc']*conv['eta_cap']:.2f} the unit gives
**{c80['T_fk']:.0f} N·m** — knee margin **{c80['margin']['knee']:.2f}**, yaw {c80['margin']['yaw']:.2f}: it **does not close**, and would
support a {c80['m_robot_supported_kg']:.0f} kg robot as written. At 100:1 (25 lobes, rated at 1250 rpm) the
motor gives {c100['T_fk']/(100*conv['eta_cyc']*conv['eta_cap']):.2f} N·m and the joint {c100['T_fk']:.0f}, margin {c100['margin']['knee']:.2f}, with the swing now needing
{n_sw*1.25:.0f} rpm{' (reached)' if R['n_noload_rpm'] >= n_sw*1.25 else ' (not reached)'}. The ratio that would close the requirement as written at this
mass is about {c80['ratio_fk_to_close']:.0f}:1 (more, since the stance speed rises with the ratio and the
torque falls) — {c80['ratio_fk_to_close']*conv['w_swing_femur']*60/(2*math.pi):.0f} rpm at the swing, {swing_factor:.1f}× what this board reaches at 48 V.
The best knee margin any JLCPCB single board reaches, speed ignored, is
{bc['margin_knee']:.2f} ({bc['stack']}, {bc['turns']} turns, {bc['h_m']:.0f} mm blocks at {bc['ratio']}:1, no-load {bc['n_noload']:.0f} rpm against {bc['n_swing']:.0f} needed).
This is §9.7's conclusion again with the sweep behind it: **a single PCB
stator does not close the requirement as written at any board in this
housing**, and the speed requirement makes it worse, not better. Which
definitions of "continuous" the chosen unit does close, at the fixed point
plus the {SO['conventions'].get('payload', 8):.0f} kg payload:

| Continuous load case | Femur/knee / yaw needed (N·m) | Margin fk / yaw at 80:1 | Closes at 80:1 | at 100:1 |
|---|---|---|---|---|
{chr(10).join('| ' + ' | '.join(r) + ' |' for r in relief_rows)}

What it would take: the two-stator unit (A-cost, §9.7) or the OTS outrunner
(§9.12) for the requirement as written. A single PCB stator closes only the
level walking case, and this board does so at 80:1 with no margin at all
({[r for r in SO['relief'] if r['label'].startswith('level')][0]['margin_fk']:.2f}). The torque-optimal JLCPCB board ({bj['stack']}, {bj['turns']} turns, {bj['h_m']:.0f} mm blocks, {bj['T_cont']:.2f} N·m,
knee margin {bj['margin_knee']:.2f} at 80:1) needs a ~{48*n_sw*1.03/bj['n_noload']:.0f} V bus for the femur swing, or a swing
requirement of {bj['n_noload']*2*math.pi/60/80:.1f} rad/s instead of {conv['w_swing_femur']:.1f}.

**Sensitivities on the chosen board** (continuous torque at 1000 rpm):

| Lever | T (N·m) | vs chosen |
|---|---|---|
{chr(10).join('| ' + ' | '.join(r) + ' |' for r in sens_rows)}

**Cross-checks.** The closed form T = k_w · N_coils · N_turns · B · I · L · r_mean
with k_w = {cf['k_w']:.3f}, B = {cf['B1']:.2f} T, L = {cf['L_m']*1e3:.1f} mm, r_mean = {cf['r_mean_m']*1e3:.1f} mm gives
Kt = {cf['Kt_textbook']:.3f} N·m/A, **{cf['discrepancy_textbook']:+.0%}** against the leg-by-leg integral's {cf['Kt_model']:.3f}. The
discrepancy is the spiral: the inner turns of a concentrated spiral coil span a
fraction of the pole, so their pitch factor is small (sin(p·α) per turn:
{', '.join(f'{k:.2f}' for k in cf['pitch_factors'])}); with each turn's own pitch factor the closed form gives {cf['Kt_with_turn_pitch']:.3f}
({cf['discrepancy_pitch']:+.0%}). `stator_asbuilt.py` run on the written variant board gives Kt {ac['Kt_asbuilt']:.4f}
(sweep {ac['Kt_sweep']:.4f}) and {ac['T1000_asbuilt']:.3f} N·m at 1000 rpm (sweep {ac['T1000_sweep']:.3f}). The block field
against `rotor_field.json` (P = 15, gap 3.2 mm; the JSON was written at r_m 66.6,
this at v2's 68.8):

| Blocks | B₁ here / JSON (T) | ratio to model here / JSON | pull here / JSON (N) |
|---|---|---|---|
{chr(10).join('| ' + ' | '.join(r) + ' |' for r in check_rows)}

**Open items from this round.**

1. The femur swing speed ({conv['w_swing_femur']:.1f} rad/s) is a requirement without a parent in the
   .sdoc set and has been tabled as unmet since round 8; it needs one, and a
   decision. This board reaches it with {R['n_noload_rpm']/n_sw-1:.0%} to spare and gives up
   {1-ch['T_cont']/bj['T_cont']:.0%} of torque for that; a higher bus or a slower swing buys the torque back.
2. `rotor_field.json` should be re-run at the v2 span; every rating since
   round 9 is ~3 % optimistic on field.
3. The thermal budget is the sizing's 1.5 K/W for every stack; §9.4's built
   path (0.8 K/W) would raise every torque here by the same {math.sqrt(1.5/0.8):.2f}, and a
   1.2 mm board conducts better through its faces than the 2.2 mm one.
4. The unit mass here is a roll-up (the round-9 femur parts less one stator
   stage), not a CAD run; `cad/actuator/actuator.py femur --stators=1` with
   this board's geometry is the check.
5. Custom-width blocks (48/40 poles) and the 3 oz reference are priced by
   volume and by an estimate; neither is a quote.
6. A 20-layer 2 oz board is scaled from the 6-layer quote; JLCPCB's copper
   options at that layer count are not confirmed. The 16-layer board is
   {ps16['T_cont']:.2f} N·m for ${ps16['cost20']:.0f} and the fallback if 20 layers cannot be had.

"""


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
    mm_rows = [(r["name"], f"{r['Kt']:.3f}", f"{r['R']:.3f}", f"{r['mass']:.2f}", f"{r['T_cont']:.2f}", f"{r['n_noload']:.0f}",
                f"{r['motors_per_unit'] if r['motors_per_unit'] else '—'}", f"{r['price20']}", f"{r['T_per_usd']*100:.1f}", f"{r['T_per_kg']:.1f}", r["flags"][:60])
               for r in MM["rows"]]
    to_rows = [(r["name"], f"{r['ratio']:.0f}:1", f"{r['eta']:.2f}", f"{r['T_joint']:.0f}", f"{r['T_need']:.0f} at {r['m_robot']:.0f} kg", f"{r['robot_mass_supported']:.0f}",
                f"{r['motor_drum_turns']:.1f}", f"{r['cost']:.0f}", "closes" if r["closes"] else ("short" if r["feasible"] else "cannot be built"), r["note"])
               for r in TO["rows"]]
    _cb = lambda names, q: sum(float(r["qty_per_unit"]) * float(r[{20: "unit_price_usd", 100: "price_100_usd"}[q]]) for r in BOM if r["item"] in names)
    _grp = (("Motor", ["Stator boards", "Rotor magnets", "Clamp rings", "Cover", "Rotor cup", "Adhesive"], None),
            ("Reducer: cycloid, eccentric, bearings, output crossed-roller, flange, pin cage", ["Eccentric bearing", "Shaft bearing (lower)", "Shaft bearing (top)", "Ring pins", "Output pins", "Cycloid disc", "Eccentric shaft", "Output flange", "Output bearing", "Bearing carrier", "Pin cage"], "same"),
            ("Capstan stage: drum, sector, rope, tensioner", ["Capstan drum", "Capstan sector", "Capstan rope", "Rope tensioner"], "same"),
            ("Housing: floor plate, wall, screws, connectors, thermistor", ["Floor plate", "Wall tube", "Housing screws", "Output screws", "Connectors", "Thermistor"], "same"),
            ("Electronics: driver, encoder", ["Motor driver", "Rotor encoder"], "same"))
    cb_rows = []
    for label, names, same in _grp:
        pcb = (_cb(names, 20), _cb(names, 100))
        ots = pcb if same else (CS["outrunner"]["price"]["20"] + 20, CS["outrunner"]["price"]["100"] + 20)
        cb_rows.append((label, f"{pcb[0]:.0f} / {pcb[1]:.0f}", f"{ots[0]:.0f} / {ots[1]:.0f}" + ("" if same else " (motor + $20 mount and heat-sink plate)")))
    _t_pcb = (sum(_cb(g[1], 20) for g in _grp), sum(_cb(g[1], 100) for g in _grp))
    _t_ots = (_t_pcb[0] - _cb(_grp[0][1], 20) + CS["outrunner"]["price"]["20"] + 20, _t_pcb[1] - _cb(_grp[0][1], 100) + CS["outrunner"]["price"]["100"] + 20)
    cb_rows.append(("**Unit total**", f"**{_t_pcb[0]:.0f} / {_t_pcb[1]:.0f}**", f"**{_t_ots[0]:.0f} / {_t_ots[1]:.0f}**"))
    tc_rows = [(u["name"], f"{u['T_cont']:.2f}", f"{u['kt_ratio_per_board']:.2f}", f"{u['boards']}", f"{u['magnet_g']:.0f}", f"{u['iron_kg']:.2f}" if u["iron_kg"] else "—",
                f"{u['mass_kg']:.2f}", f"{u['cost20']:.0f} / {u['cost100']:.0f}", f"{u['stack_mm']:.0f}", f"{u['T_per_kg']:.2f}", f"{u['T_per_100usd']:.2f}", u["note"]) for u in TC["units"]]
    tg_rows = [(g["grade"], f"{g['Br20']:.2f}", f"{g['Br_hot']:.2f}", f"{g['tmax_C']}", "yes" if g["usable"] else "no", f"x{g['torque_ratio']:.2f}", f"x{g['cost_ratio']}") for g in TC["grades"]]
    tt_rows = [(f"{n} turns, {t['opening_mm']:.1f} mm opening", f"x{t['air'] / TC['fields']['A  two Halbach rotors, one stator (1s-opt, baseline)']['dlam_dx_per_turn']:.2f}",
                f"x{t['tooth_linear'] / TC['fields']['A  two Halbach rotors, one stator (1s-opt, baseline)']['dlam_dx_per_turn']:.2f}", f"x{t['gain_vs_8t_air']:.2f}",
                f"x{t['gain_vs_8t_air'] / math.sqrt(t['copper_ratio']):.2f}", f"{t['B_tooth']:.1f}", f"{t['attraction_N'] / 1e3:.1f}") for n, t in TC["teeth"].items()]
    fm_chk = [("Kt from the quoted Ke (1.5 × Ke phase peak)", f"{FM['checks']['kt_from_ke']*1e3:.2f} mN·m/A", f"{FM['datasheet']['kt_per_A_peak']*1e3:.1f}", f"{FM['checks']['kt_err']*100:+.1f} %"),
              ("continuous torque from Kt × √2 × I", f"{FM['checks']['t_from_kt']:.3f} N·m", f"{FM['datasheet']['t_cont']:.2f}", f"{FM['checks']['t_err']*100:+.1f} %"),
              ("Km from Kt and R (Kt_rms/√(3R))", f"{FM['checks']['km_from_kt']*1e3:.1f} mN·m/√W", f"{FM['datasheet']['km']*1e3:.1f}", f"{FM['checks']['km_err']*100:+.1f} %"),
              ("no-load speed against the V_dc/√3 modulation ceiling", f"{FM['checks']['e_peak_at_noload']:.2f} V", f"{FM['checks']['v_svpwm_peak']:.2f} V", f"{FM['checks']['noload_err']*100:+.1f} %")]
    fm_stock = [(k, f"{v['m_robot']:.0f}", f"{v['margin']['femur']:.2f}", f"{v['margin']['knee']:.2f}", f"{v['margin']['yaw']:.2f}", f"{v['robot_supported']:.0f}")
                for k, v in FM["stock_unit"].items()]
    fm_sens = [(x["label"], f"{x['T_cont']:.2f}", f"{min(x['margin'].values()):.2f}", "closes" if x["closes"] else "**short**") for x in FM["sensitivity"]]
    ladder_rows = [(L["label"], f"{L['m_unit_fk']:.2f}", f"{L['leg_struct_kg']:.1f}", f"{L['m_robot']:.0f}",
                    f"**{L['worst']:.2f}**", "closes" if L["closes"] else "**does not close**") for L in FM["mass_ladder"]]
    cs_rows = [(r["option"], r["requirement"], f"{r['m_unit']:.2f}", f"{r['m_robot']:.0f}", f"{r['T_joint']:.0f} / {r['need_knee']:.0f}", f"{r['margin']:.2f}" + ("" if r["closes"] else " ✗"),
                f"{r['cost']:.0f}", f"{[x for x in CS['rows'] if x['option']==r['option'] and x['requirement']==r['requirement'] and x['qty']==100][0]['cost']:.0f}")
               for r in CS["rows"] if r["qty"] == 20]
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
        b, a = r["B: 1 stator, 3 oz boards, 8 mm magnets"], r["A-cost: 2 stators, 2 oz boards, 6 mm magnets (chosen)"]
        case_rows.append((r["label"], f"{b['need']['femur']:.0f} / {b['need']['knee']:.0f} / {b['need']['yaw']:.0f}", "yes" if b["closes"] else "no",
                          f"{a['need']['femur']:.0f} / {a['need']['knee']:.0f} / {a['need']['yaw']:.0f}", "yes" if a["closes"] else "no"))
    bom_rows = [(r["item"], r["qty_per_unit"], r["spec"][:90], f"{float(r['qty_per_unit'])*float(r['price_before_usd']):.0f}", f"{float(r['qty_per_unit'])*float(r['unit_price_usd']):.0f}", f"{float(r['qty_per_unit'])*float(r['price_100_usd']):.0f}", r["verified"]) for r in BOM]
    rf_ns = {k: v for k, v in RF.items() if v.get("kind") == "ns_iron"}
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
give 84 % of the trapezoid field; thicker blocks buy that back. **Chosen
(round 9): 30 × 6 × 5 N48H on every unit** — the v2 board's extra torque paid
for the thinner, lighter, cheaper block; 8 mm was the choice of rounds 5–8.
Half the blocks are magnetised through their 5 mm dimension, half through the
6 mm — specify both. The {RF['rect 30x5x8 N48']['attraction_N']/1e3:.1f} kN
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
implies** ({CL['options']['A-cost: 2 stators, 2 oz boards, 6 mm magnets (chosen)']['m_robot']:.0f} kg with the two-stator units), not the 49 kg of round 1;
§9.7 explains why that is the right mass to check against.

{md(("Joint", "Cycloid × capstan", "Board", "Motor cont / peak (N·m)", "Joint cont / peak (N·m)", "Needed at the CAD mass", "Margin", "Joint speed no-load (rad/s)", "Unit mass (kg)", "Height (mm)"), joint_rows)}

With the cost-down boards (two 8-layer 2 oz per position, §9.10) every motor
torque above scales by {AB16['ratings']['1000']['T_cont']/AB['ratings']['1000']['T_cont']:.2f}; §9.7 carries that case as "A-cost".

### 9.6 Bill of materials — `docs/design/bom-actuator.csv`

{md(("Item", "Qty", "Spec", "Round 7 ($)", "20 units ($)", "100 units ($)", "Price verified"), bom_rows)}

**${BOM_UNIT:.0f} per unit at 20 units (${BOM_BEFORE:.0f} before the cost-down), ${BOM_100:.0f} at 100 units;
${BOM_UNIT*18/1000:.1f}k for eighteen.** Verified prices are marked; the rest are estimates to be
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
quote can confirm.

**Calibration against a real quote (round 12).** The reviewer ran JLCPCB's
instant quote: 192 × 192 mm, 6 layers, 2 oz, $15 at 20 pieces and $11 at
100. The BOM had carried $20 / $15 per 8-layer board; it now carries $22 /
$16, scaled by layer count from that quote, and marks the line as partly
verified. The same quote suggested a cheaper stack — three 6-layer boards
per position instead of two 8-layer — so it was generated and rated
(`hw/stator/variants/3x6L-2oz`): 18 layers of 2 oz in 3.2 mm gives
{AB3['ratings']['1000']['T_cont']:.2f} N·m against {AB16['ratings']['1000']['T_cont']:.2f} for the two-board stack, at the same $45 per
position, because the 0.7 mm of extra gap costs more field than the extra
copper adds. Two 8-layer boards stay. Every other machined and laser-cut line
in the BOM is still an estimate and should get the same treatment: JLCPCB's
CNC and sheet-metal quote tools take the STEP and DXF files in `build/cad`.

### 9.11 Round 9: pushing further on the unit itself

The review asked for more. Three things were tried; two stayed.

**1. The stator laid out again (v2).** The first layout kept the phase rings,
M arcs and terminal vias inside the magnet span (r 80.9–86), so the coils'
radial legs stopped at r 80.2 and the outer 4.8 mm of magnet did no work.
Layout v2 moves the interconnect out to the clamp rim (rings at r 87.1–88.9,
pads at 90.4, board Ø184) and runs the legs to r 84.6. Torque per amp rises
with ∫B·r·dr, so this is +{(AB['Kt']/0.195 - 1)*100:.0f} % Kt and **+{(AB['ratings']['1000']['T_cont']/2.90 - 1)*100:.0f} % continuous torque for
the same copper**: {AB['ratings']['1000']['T_cont']:.2f} N·m per 12L 3 oz board, {AB16['ratings']['1000']['T_cont']:.2f} per position of two
8L 2 oz boards — the cheap boards now match what the expensive one did.
The housing grows to Ø192 for the wider clamp. DRC clean, gerbers
regenerated; the v1 board is kept in `hw/stator/variants/v1-r80`.

**2. A conventional N-S rotor on steel back plates, tried and rejected.**
Same blocks, all magnetised axially, two per pole on a laser-cut steel plate
(no Halbach jig, one magnetisation direction, cheaper plates):
{md(("Rotor", "Field at the board (T)", "Magnet mass per two rings (g)", "Note"), [(k, f"{v['B1_midplane']:.2f}", f"{v['magnet_mass_g']:.0f}", v['note'][:70]) for k, v in rf_ns.items()] + [("Halbach 30×5×8 (kept)", f"{RF['rect 30x5x8 N48']['B1_midplane']:.2f}", f"{RF['rect 30x5x8 N48']['magnet_mass_g']:.0f}", "no iron")])}
The field is within 2 %, but the back plates have to carry the pole flux
(≥ 4.5 mm of steel to stay under 1.6 T), which adds about 0.7 kg per unit —
12 kg on the robot, 8 % more torque needed — for a saving of perhaps $15 in
magnetisation and jigs. The Halbach ring is the mass-optimal choice; its cost
premium is small. Rejected.

**3. The base as four cheap parts.** The machined 7075 base (floor, wall
and pin cylinder in one billet) is now a laser-cut 6 mm floor plate, a slice
of Ø192 tube for the wall, a short turned bearing carrier, and a pin cage of
three laser-cut 8 mm steel rings whose 21 holes are open to the bore so the
pins press through — the half-grooves that needed a 4-axis mill are gone.
The output flange is a turned hub with a laser-cut plate. All in the CAD and
the BOM.

**4. With the v2 board's torque in hand, the magnets drop to 6 mm** (30 × 5 × 6,
{RF['rect 30x5x6 N48']['B1_midplane']:.2f} T at the board, {RF['rect 30x5x6 N48']['magnet_mass_g']:.0f} g per two rings instead of {RF['rect 30x5x8 N48']['magnet_mass_g']:.0f}): the unit is
{2*(RF['rect 30x5x8 N48']['magnet_mass_g']-RF['rect 30x5x6 N48']['magnet_mass_g'])/1000:.2f} kg lighter and 4 mm shorter, the robot ~10 kg lighter, and §9.7 still closes
with 1.09 at the knee. The 8 mm blocks stay in §9.7 as the margin option.

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
about {CL['options']['A-cost: 2 stators, 2 oz boards, 6 mm magnets (chosen)']['m_robot']:.0f} kg. The other way to close would have been the definition of
"continuous": the walking load case is dyn 1.5 on three legs, on a 30° slope,
accelerating at 1 m/s², over the whole routine working volume, all at once.
Against gentler definitions:

{md(("Continuous load case", "B: 1 stator, femur / knee / yaw needed at " + f"{CL['options']['B: 1 stator, 3 oz boards, 8 mm magnets']['m_robot']:.0f} kg", "B closes?", "A: 2 stators, needed at " + f"{CL['options']['A-cost: 2 stators, 2 oz boards, 6 mm magnets (chosen)']['m_robot']:.0f} kg", "A closes?"), case_rows)}

**The review chose A** (round 6): two stators everywhere, {CADJ['femur']['height_mm']:.0f} mm units, a
~{CL['options']['A-cost: 2 stators, 2 oz boards, 6 mm magnets (chosen)']['m_robot']:.0f} kg robot, the requirement as written. Round 8 then swapped the
3 oz boards for two 8-layer 2 oz JLCPCB boards per position (11 % less torque,
a quarter of the price) and raised the total ratio to 80:1 through the
capstan stage to keep the margin ("A-cost" above). The single-stator unit
stays in the CAD as the `-1s` variant (§9.3) and B stays here as the record of
the lighter robot that was not chosen. The 62 mm stack no longer fits the
200 mm body slab (three units and two gaps are 202 mm), so the slab is now
220 mm in `hexapod_model.py`; the drawings in 06 carry it.

### 9.12 Round 10: exhausting the cost routes — `analysis/cost_search.py`

The review asked for confidence that the minimum has been found, not another
line trimmed. So the design space was searched as a whole: motor family,
number of stators, reduction, quantity and the definition of "continuous",
each option at the robot mass it implies and with the torque it actually gives.

{md(("Option", "Requirement", "Unit (kg)", "Robot (kg)", "Joint gives / knee needs (N·m)", "Margin", "$ at 20", "$ at 100"), cs_rows)}

![Cost search]({rel(os.path.join(FIG, 'cost-search.png'))})

What the search says:

* **Two stators are needed only because the motor is ironless.** The PCB
  machine's shear stress is ~3 kPa; an iron-core outrunner's is 10–15 kPa. A
  single PCB stator closes the requirement as written at no mass (its own
  weight is what defeats it), but closes level walking with 1.3 margin at
  $497 / $320.
* **The cheapest option that closes the requirement as written is not the
  PCB machine.** One off-the-shelf 8318 outrunner (Ø92 × 40, 0.65 kg, 100 KV,
  0.055 Ω, ~$65 / $48) heat-sunk to an aluminium mount gives an estimated
  {CS['outrunner']['T_cont']:.1f} N·m continuous at {CS['outrunner']['I_cont']:.0f} A; through a 25-lobe cycloid and the same 4:1
  capstan (100:1) it closes at a **77 kg robot with 1.12 margin for
  ${[r for r in CS['rows'] if r['option'].startswith('1 x 8318') and r['requirement']=='as written' and r['qty']==20][0]['cost']:.0f} / ${[r for r in CS['rows'] if r['option'].startswith('1 x 8318') and r['requirement']=='as written' and r['qty']==100][0]['cost']:.0f}** — a lighter robot and a cheaper unit than anything the PCB machine
  reaches, because the motor mass drops from ~2.7 kg of boards, magnets and
  rotor plates to 0.65 kg. Two such motors on one eccentric shaft give 1.55
  margin at 89 kg for $503 / $347. The number that carries this is the
  outrunner's continuous current in a closed body, which the listings only
  quote with propeller airflow: it is an assumption ({CS['outrunner']['R_th']} K/W) until a motor
  is bolted to a plate and measured — the first bench test either way.
* **What the PCB machine keeps.** No cogging, a flat form, no dependence on a
  motor vendor, and the ability to make the active parts in-house. At
  $597 / $396 and 5.3 kg it is the more expensive and heavier way to the same
  torque; its case is self-manufacture and volume, not unit cost at 20.
* **The requirement is the largest lever of all.** Level walking at dyn 1.2
  instead of the 30° slope at dyn 1.5 lets the single-stator PCB unit close
  at 94 kg and the single outrunner carry 1.66 margin.

**On the rotor cup and the middle rotor** (asked in review): the middle rotor
is two Halbach rings back to back (red, orange, red in the cutaway is magnet,
4.5 mm aluminium carrier, magnet). Magnetically the carrier does nothing — a
Halbach array is one-sided and each ring feeds its own gap — and the two
2.3 kN attractions on the middle ring cancel, so it only has to hold the
blocks in place: a 2 mm laser-cut ring, or the two rings bonded back to back
with no plate at all. The whole rotor can be sheet and off-the-shelf parts:
laser-cut carrier plates, twelve turned standoffs instead of the drum, a
clamping hub instead of the machined hub. That takes 2.5 mm and ~100 g out of
the unit and the last turned part out of the rotor. It is not modelled,
because the search above says the motor decision comes first.

**Apples to apples: the outrunner price includes the gearbox.** Every unit
price in the search is stacked from the same BOM lines; the OTS unit keeps
the whole reducer, the capstan stage, the housing and the driver, and swaps
only the motor lines for the bought motor plus a mount. Per unit, at 20 / 100
units:

{md(("Lines", "PCB two-stator unit ($)", "1 × 8318 unit ($)"), cb_rows)}

The whole difference is the motor: a PCB motor costs {_cb(_grp[0][1], 20):.0f} in boards, magnets,
rotor and clamp parts against {CS["outrunner"]["price"]["20"] + 20:.0f} for the outrunner and its plate. The reducer
is the largest single block in either unit, and it is the same block.

**The honest statement of the minimum.** Within this design space and these
requirements, the cost floor is an OTS iron-core outrunner driving the
20–25-lobe cycloid and the 4:1 capstan, in a laser-cut housing, at roughly
$420 per unit at 20 and $280 at 100, on a ~77 kg robot — subject to one
measurement (the outrunner's heat-sunk continuous current) and one quote
(the motor at quantity). Below that lies only a custom driver board (−$20),
a cast base (−$30 at 500+), and relaxing the continuous load case.

### 9.13 Round 11: the market search — `analysis/motor_market.py`

Asked: the best off-the-shelf motor for this joint at the lowest price. The
joint needs, through the 25-lobe cycloid and the 4:1 capstan, {MM['assumptions']['need_T']['100:1']:.2f} N·m
continuous from one motor at a 77 kg robot (or {MM['assumptions']['need_T']['80:1']:.2f} N·m from two at 80:1 and
89 kg), and {MM['assumptions']['need_rpm']['100:1']:.0f} rpm no-load at 48 V for the femur swing. Every candidate is
rated the same way: not the listing's propeller-cooled current, but what its
copper can dissipate through a heat-sink path assumed at {MM['assumptions']['R_th_ref']} K/W for a
Ø92 × 40 stator and scaled with stator area. Listing numbers, not datasheets;
the flags say what was inferred.

{md(("Motor", "Kt (N·m/A)", "R (Ω)", "kg", "Heat-sunk cont. (N·m)", "rpm at 48 V", "Motors / unit", "$ each at 20", "N·m per $100", "N·m per kg", "Flags"), mm_rows)}

![Motor market]({rel(os.path.join(FIG, 'motor-market.png'))})

* **Winner on price for the requirement: the 8318 100KV class** (HL Q9XL,
  Alibaba clones): one motor per unit, 0.65 kg, about $50 at 20 pieces,
  {[r for r in MM['rows'] if r['name'].startswith('8318')][0]['T_cont']:.2f} N·m heat-sunk against {MM['assumptions']['need_T']['100:1']:.2f} needed. The Turnigy 9235-100KV is
  the same motor with a published resistance and a brand behind it, at $75.
  Buy one of each for the bench test; the 9235's listing is the better spec.
* The skateboard 63100 190KV would give more torque per motor but needs
  60 A continuous, beyond a $45 driver, and its resistance is an estimate.
  The 6384 120KV is heavier and needs two per unit.
* The premium drone motors (MAD 8108, T-Motor U8 II) are lighter per newton
  metre but three to five times the price and cannot carry the current
  without airflow: wrong market for us.
* The hoverboard hub motor is the cheapest torque on earth
  ({[r for r in MM['rows'] if 'hoverboard' in r['name']][0]['T_per_usd']*100:.0f} N·m per $100) and is excluded by speed: 1000 rpm no-load at
  48 V is a quarter of what the femur swing needs at any ratio that fits, and
  it weighs 2.9 kg.
* The PCB two-stator motor sits on the chart at 5.9 N·m and 2.7 kg: twice
  the torque of an 8318 at four times the mass and three to four times the
  price. It is the better machine per watt and the one we can make; it is
  not the cheapest way to this joint.

The one number all of this rests on is the heat-sunk continuous current of a
propeller motor in a closed body. One 8318 or 9235 on an aluminium plate,
locked rotor, 30 A, a thermocouple on the winding, is a $100 afternoon and
decides the motor family.

### 9.14 Round 12: does the outrunner still need the cycloid? — `analysis/transmission_options.py`

Asked: with these motors, do we need a gearbox, or can the motor drive the
capstan joint directly? The ratio a joint needs is joint torque divided by
motor continuous torque, whatever the motor. The 8318 gives
{TO['motor']['T_cont']:.2f} N·m heat-sunk against the PCB motor's 5.9, so it needs *more* ratio,
not less: about 90:1 at the load case as written, 100:1 as costed. The 4:1
capstan on its own would put {TO['rows'][0]['T_joint']:.0f} N·m at the femur, enough for a
{TO['rows'][0]['robot_mass_supported']:.0f} kg robot.

What limits a rope drive is not strength but wrap. The rope on the motor drum
has to hold (total ratio × joint travel) turns: at {TO['joint_range_deg']:.0f}° of travel that is
ratio / 2.8, and past about {TO['wrap_max_turns']:.0f} working turns the drum is a winch (fleet
angle, rope stacking, a long drum) rather than a capstan. So a pure rope drive
tops out near 10:1 whether it has one stage or two, which is why every
quasi-direct legged robot pairs it with a motor ten times ours. Each option
below is rated for the femur with one 8318 per unit, at the robot mass its own
weight implies, against the continuous load case as written; the cost is the
reducer plus capstan BOM lines at 20 units (cycloid lines ${TO['cost_cycloid_lines']:.0f}, capstan
lines ${TO['cost_capstan_lines']:.0f}).

{md(("Transmission", "Ratio", "η", "Joint (N·m)", "Needed", "Robot it supports (kg)", "Turns on motor drum", "$ / unit", "Verdict", "Note"), to_rows)}

![Transmission options for the outrunner]({rel(os.path.join(FIG, 'transmission-options.png'))})

* **Direct to the capstan: no.** A 4:1 capstan gives the joint a twentieth of
  what it needs; the largest single capstan the coxa pod could carry (a Ø300
  sector, a Ø14 drum on 1.8 mm rope) is {TO['rows'][1]['ratio']:.0f}:1, still a quarter short,
  and already {TO['rows'][1]['motor_drum_turns']:.1f} turns of rope on the motor drum.
* **Two rope stages: no.** 100:1 in rope means {TO['rows'][2]['motor_drum_turns']:.0f} turns on a drum
  spinning at 3600 rpm.
* **Belts instead of the cycloid: no.** Two 5:1 belt stages reach the ratio,
  but the second belt carries {TO['motor']['T_cont']*25*0.92:.0f} N·m continuous and its large
  pulley is a Ø300 HTD 8M, wider than the pod.
* **A motor big enough to skip the reducer: no.** A 10-inch hub motor on a
  10:1 capstan is the cheapest torque that would do it
  ({TO['hub']['T_cont']:.0f} N·m heat-sunk, ~$80), but twelve of them add 54 kg and the
  mass loop never closes; the hip pod would be Ø260.
* **What stays: the 25-lobe cycloid × 4:1 capstan**, ${TO['rows'][4]['cost']:.0f} of
  transmission per unit. The one OTS alternative is a 60-frame planetary
  (~$140, unverified) at its rated limit and a little heavier; it is worth a
  quote as the drop-in fallback if the laser-cut cycloid discs disappoint,
  not a cost-down.

{so_section(lambda f: rel(os.path.join(FIG, f)))}
### 9.16 Round 13b: single rotor between two stators, magnet material, iron in the coils — `analysis/topology_compare.py`

Three questions on the round-13 optimum, answered with one 2-D magnetostatic
model at the mean coil radius (scalar potential on a {TC['model']['grid_mm']:.1f} mm grid over the
12-coil / 5-pole-pair period, magnets as surface charges, iron as μr 1000
with the tooth flux capped at 1.6 T). The torque figure is the fundamental of
one coil's flux linkage as the rotor turns, which is the same as the per-leg
Lorentz sum for an air-core coil and the only right way to count a tooth.
Cross-check: the model gives {TC['model']['crosscheck_B1_FD']:.3f} T where `rotor_field.py` gives
{TC['model']['crosscheck_B1_rotor_field']:.3f} T on the same inputs. Every unit below has 10 mm N48 blocks and the
20-layer 2 oz board of §9.15, each board at the same copper-loss budget.

**1. One rotor between two stators, same Ø190 package.**

{md(("Topology", "T cont (N·m)", "Kt per board vs A", "Boards", "Magnets (g)", "Iron (kg)", "Stator+rotor (kg)", "$ at 20 / 100", "Stack (mm)", "N·m/kg", "N·m/$100", "Note"), tc_rows)}

![Topologies]({rel(os.path.join(FIG, 'topology-compare.png'))})

* **A stays.** The two-rotor single-stator unit is the best air-core
  arrangement per kilogram and per dollar. The second rotor is worth
  {TC['units'][0]['T_cont'] / TC['units'][6]['T_cont']:.1f}× over one rotor with nothing behind the board (E0) because a
  one-sided Halbach array only feeds the side it faces.
* **B, a through-magnetised rotor between two iron-backed boards**, gives
  {TC['units'][1]['T_cont']:.2f} N·m from two boards, {TC['units'][1]['T_cont'] / TC['units'][0]['T_cont']:.2f}× A for {TC['units'][1]['mass_kg'] / TC['units'][0]['mass_kg']:.1f}× the mass and
  {TC['units'][1]['cost20'] / TC['units'][0]['cost20']:.1f}× the cost: each board sees {TC['fields']['B  one N-S rotor, two iron-backed stators']['B1_board']:.2f} T instead of {TC['fields']['A  two Halbach rotors, one stator (1s-opt, baseline)']['B1_board']:.2f}, and the
  yokes see the rotor field at 250 Hz, so they have to be wound-strip
  laminations or SMC rings, not the steel plate the N-S-on-iron study of round 9
  already rejected. The canonical two-stator three-rotor unit does 5.9 N·m
  from the same two boards.
* **C, the canonical middle rotor on its own** (two Halbach rings back to back
  between two boards), is worse than A per board: each board gets one ring's
  field. Iron behind each board (D) recovers some of it and brings the 250 Hz
  yoke back.

**2. Magnet material.** Torque at fixed current scales with the remanence at
the magnet's working temperature ({TC['model']['T_magnet_C']:.0f} °C in every rating). Generic
grade-chart values; no vendor sheet in `docs/reference` yet, so these are
flagged, not verified:

{md(("Grade", "Br at 20 °C (T)", "Br at the working temperature (T)", "Max working (°C)", "Usable here", "Torque", "Cost"), tg_rows)}

Every rating so far used a typical N45 remanence; the N48H the BOM already
specifies is worth about 5 % more. N52 is 9 % more but cannot run at
{TC['model']['T_magnet_C']:.0f} °C, SH and UH grades buy temperature at a small torque cost, SmCo
loses 13 % and costs four times as much, ferrite loses 72 %. **Magnet grade
moves torque by ±5 %; thickness (6 → 10 mm, +12 % in §9.15) and topology
(×2) are the levers.**

**3. Iron in the centre of the coils.** A laminated tooth through the board
in each coil's opening, for coils of 8, 6 and 4 turns (the opening is what
the legs leave free, so a wider tooth means fewer turns and less copper):

{md(("Coil", "Air core, vs the 8-turn coil", "Tooth, linear iron", "Tooth, flux capped at 1.6 T", "At the same copper loss", "Tooth would see (T)", "Rotor pull (kN)"), tt_rows)}

The linear model wants {TC['teeth']['8']['B_tooth']:.1f} T in a 2.7 mm tooth, so saturation eats most of the
linear gain; the capped result is {TC['teeth']['6']['gain_vs_8t_air'] / math.sqrt(TC['teeth']['6']['copper_ratio']):.2f}× at the same copper loss for the
6-turn coil. Against that: the pull on each rotor rises from
{TC['fields']['A  two Halbach rotors, one stator (1s-opt, baseline)']['attraction_N'] / 1e3:.1f} kN to {TC['teeth']['6']['attraction_N'] / 1e3:.1f} kN (carriers, bearings and the glue joints all
resize), a Halbach rotor over teeth cogs and needs skew, the teeth carry
250 Hz flux and have to be lamination stacks or SMC plugs pressed into slots
in the board, which is not a JLCPCB process, and the eddy loss in the copper
next to a 1.6 T tooth edge goes up. **A 20–35 % gain for a different
manufacturing route and a heavier rotor structure; it is not the cheap lever,
and it is not modelled beyond this bound.**

### 9.17 Round 14: the Wheemo frameless motor as the basis — `analysis/frameless_motor.py`

> **Corrected by §9.18.** The unit mass in this section (2.84 kg) and the unit
> height (37 mm) were wrong: the reducer and housing masses were carried over
> from the Ø192 design without being re-solved for this one, and the height
> left out the reducer's own axial stack. The CAD says 4.09 kg and 49.7 mm.
> The scaling law, the datasheet decoding and the reducer loads below stand;
> the closure does not. Read §9.18 with this section.

The review supplied the datasheet for a **{FM['datasheet']['part']}** frameless
kit motor ([`docs/reference/wheemo-WxF70x24GT.pdf`](../reference/wheemo-WxF70x24GT.pdf),
filed in the manifest) and asked that our motors be built on it, scaled up if
the requirement needs more torque. It does need more, and the scaled motor
turns out to change the architecture rather than just the part number.

**The datasheet decodes cleanly.** It gives terminal quantities and no
geometry, so before using it the conventions were pinned by checking it
against itself four ways:

{md(("Check", "Computed", "Datasheet", "Error"), fm_chk)}

So Kt is quoted per amp of *peak* phase current, the {FM['datasheet']['current_A']:.1f} A is rms, and the
no-load speed is exactly where the peak phase back-EMF meets the
V_dc/√3 space-vector ceiling. Of the {FM['datasheet']['loss_cont_W']:.1f} W of quoted loss, {FM['checks']['P_cu']:.1f} W is
copper and the remaining {FM['checks']['P_fe_implied']:.1f} W is iron and windage at {FM['datasheet']['rpm_rated']:.0f} rpm.

**The stock motor is a fifth of what the joint needs.** {FM['datasheet']['t_cont']:.2f} N·m
continuous, {FM['datasheet']['mass_g']:.0f} g, and at every ratio the requirement stays out of reach:

{md(("Reduction", "Robot at the fixed point (kg)", "Femur", "Knee", "Yaw", "Robot it supports (kg)"), fm_stock)}

**Scaling it.** The datasheet gives no internal geometry, so the radial build
is inferred from the active mass: an annulus of laminations, copper at
realistic slot fill and sintered magnet sits between 5.8 and 7.2 g/cm³, which
brackets the bore at {FM['inferred']['bore_bracket_mm'][1]:.0f}–{FM['inferred']['bore_bracket_mm'][0]:.0f} mm and the total radial build at
{FM['inferred']['t_tot_bracket_mm'][0]:.1f}–{FM['inferred']['t_tot_bracket_mm'][1]:.1f} mm. That implies an airgap shear stress of
{FM['inferred']['sigma_cont_Pa']['nom']/1e3:.1f} kPa continuous and {FM['inferred']['sigma_peak_Pa']/1e3:.0f} kPa peak — squarely in the range a
conduction-cooled machine of this class runs at, which is the independent
sanity check on the inference.

The scaling law is the one that holds when a frameless torque motor grows at
**constant pole pitch**: the pole count rises with diameter, so the radial
build, the end-turn length per coil and the flux per pole all stay put. Then
torque goes as the square of the airgap diameter while copper loss goes only
as the mounting area, so the heat flux through the stator's outside face — the
only path a frameless stator has — never changes. **Torque per kilogram grows
linearly with diameter.** That is the whole argument for making the motor big
and thin, and it is the same argument the PCB axial motor was built on, now
with iron behind it.

![Frameless scaling]({rel(os.path.join(FIG, 'frameless-scaling.png'))})

**The design.** Sweeping diameter and stack length inside the actuator can and
solving the mass/torque fixed point, {FM['n_closing']} of {FM['n_rows']} combinations close the
requirement and {FM['n_robust']} still close if the quoted resistance turns out to be a
cold value. The pick:

| | |
|---|---|
| Motor | **Ø{FM['pick']['motor']['od_mm']:.0f} × {FM['pick']['motor']['len_mm']:.0f} mm**, bore Ø{FM['pick']['motor']['bore_mm']:.0f}, airgap Ø{FM['pick']['motor']['d_gap_mm']:.0f} |
| Torque | **{FM['pick']['motor']['T_cont']:.2f} N·m continuous**, {FM['pick']['motor']['T_peak']:.1f} N·m the iron could make |
| Active mass | {FM['pick']['motor']['mass_kg']*1e3:.0f} g → **{FM['pick']['motor']['T_per_kg']:.1f} N·m/kg** (the WxF70x24GT itself: {FM['datasheet']['t_cont']/(FM['datasheet']['mass_g']/1e3):.2f}; the 8318: 4.0) |
| Motor constant | Km {FM['pick']['motor']['Km']:.2f} N·m/√W (the WxF70x24GT: {FM['datasheet']['km']:.2f}) |
| Reduction | **{FM['pick']['ratio_name']}** |
| Winding | Kt up to {FM['pick']['kt_max_rms_at_speed']:.2f} N·m/A rms at the {FM['pick']['rpm_fk']:.0f} rpm the femur swing needs → {FM['pick']['I_at_cont']:.0f} A rms continuous |
| Unit | {FM['pick']['m_fk']:.2f} kg → robot **{FM['pick']['m_robot']:.1f} kg** |
| Margins | femur **{FM['pick']['margin']['femur']:.2f}**, knee **{FM['pick']['margin']['knee']:.2f}**, yaw {FM['pick']['margin']['yaw']:.2f} |

**The capstan can be deleted, and that is the real result.** The frameless
motor is an annulus with a Ø{FM['pick']['motor']['bore_mm']:.0f} hole, and the cycloid fits inside it rather
than underneath it. Two things follow. The unit gets shorter — {FM['pick']['motor']['len_mm']+12:.0f} mm against
54 for the PCB unit and 64 for the outrunner — and the cycloid's ring-pin
circle moves from r 43.5, all the Ø100 PCB-stator bore allowed, out to
r {FM['pick']['reducer']['r_pin_circle_mm']:.0f}. Deleting the capstan multiplies the cycloid's torque by four; moving
the pin circle out takes most of that back. At {FM['pick']['reducer']['lobes']} lobes on r {FM['pick']['reducer']['r_pin_circle_mm']:.0f} the pitch is
**{FM['pick']['reducer']['pitch_mm']:.1f} mm — coarser than the {CYC['femur']['pitch']:.1f} mm of the current design**, so the
round-8 worry about lobe count and laser-cut tolerance gets better, not worse.
The cycloid then carries {FM['pick']['reducer']['T_cyc_cont']:.0f} / {FM['pick']['reducer']['T_cyc_peak']:.0f} N·m at {FM['pick']['reducer']['hertz_peak']:.0f} MPa peak Hertz
(allowable {CYC['femur']['sigma_allow'] if 'sigma_allow' in CYC['femur'] else 1400:.0f}), and the eccentric bearing sees {FM['pick']['reducer']['F_ecc']/1e3:.1f} kN — a static
margin of {FM['pick']['reducer']['hk2512_static_margin']:.2f} on the HK2512 already in the BOM, {FM['pick']['reducer']['hk3012_static_margin']:.2f} on the HK3012.

That matters beyond the actuator: the leg study (§09) found the capstan rope
drive as drawn **cannot be wound**, with fleet angles of 5° to 53° where a
grooved drum wants 1.5°. Deleting the stage removes that blocker, the two
sectors, the two drums, four rope terminations and the tensioners.

![The frameless unit]({rel(os.path.join(FIG, 'frameless-unit.png'))})

**The yaw joint should not share the motor.** At the pick's size the yaw
margin is {FM['pick']['margin']['yaw']:.2f} — half a kilogram of iron per unit doing nothing. Sized on
its own the yaw wants Ø{FM['yaw']['motor']['od_mm']:.0f} × {FM['yaw']['motor']['len_mm']:.0f}, {FM['yaw']['motor']['T_cont']:.2f} N·m, margin {FM['yaw']['margin']:.2f}, saving
{FM['yaw']['saving_g']:.0f} g on each of six units.

**Thermally it fits, on an estimated duty split.** All eighteen units at their
own continuous ratings would put {FM['body']['all_at_continuous_W']:.0f} W into the body and holding the
requirement continuously would put {FM['body']['at_requirement_W']:.0f} W; weighted by a stance duty of
{FM['body']['duty']['fk']:.0%} for the femur and knee units and {FM['body']['duty']['yaw']:.0%} for yaw it is **{FM['body']['duty_weighted_W']:.0f} W against the
{FM['body']['budget_W']:.0f} W the body can shed**. The duty numbers are an estimate, not a gait
simulation, and they are the weakest link in that sentence.

**Peak is limited by the driver, not the motor.** The 40 A class board already
in the BOM gives {FM['driver_limit']['T_motor']:.1f} N·m at the motor and {FM['driver_limit']['T_joint']:.0f} N·m at the joint against
{FM['driver_limit']['T_joint_needed']:.0f} needed, a margin of {FM['driver_limit']['T_joint']/FM['driver_limit']['T_joint_needed']:.2f}. The iron could make {FM['pick']['motor']['T_peak']:.0f} N·m, well past what
the cycloid is sized for, so **the drive has to current-limit to protect the
reducer** — a firmware requirement this design creates.

**How much of this depends on the inferred geometry.** The two numbers that
were not in the datasheet were attacked directly:

{md(("Assumption", "Continuous torque (N·m)", "Worst margin", "Verdict"), fm_sens)}

The design survives every variant of the geometry inference. The one that
bites is thermal, not geometric: **the datasheet does not say at what winding
temperature the {FM['datasheet']['R_ph']*1e3:.1f} mΩ is quoted.** If it is a 20 °C value and the winding
runs at 120 °C, every torque here falls 15 %, which is why the pick was
required to show a margin of {FM['margin_min']:.2f} rather than 1.00.

**What is not known.** The datasheet carries **no price**, and that is the one
number that decides whether this replaces the outrunner on cost as well as on
performance. The frameless unit deletes the capstan stage (${FM['unit_8318_usd']:.0f} unit, of which
$58 is capstan lines) and the outrunner's mount, so **it undercuts the $423
8318 unit while the kit costs under ${FM['breakeven_motor_price_usd']:.0f}**. A Ø160 frameless kit is likely to
cost more than that, in which case this is a performance and packaging choice,
not a cost-down — and the round-10 cost floor still stands. Also missing:
the airgap diameter, the pole and slot count, and the thermal resistance or
reference ambient behind the {FM['datasheet']['loss_cont_W']:.1f} W rating. All are recorded as needed in
the manifest.

### 9.18 Round 14c: the correction, and the number the whole design now turns on

Building the unit in CAD (`cad/actuator/frameless.py`) and drawing the whole
robot around it (`analysis/arrangement.py`) contradicted §9.17 in three ways.
All three are the same mistake: a number that stood in for a measurement was
never re-solved after the design it came from changed.

**1. The unit is {FCAD['total_g']/1e3:.2f} kg, not {FCAD['mass_assumed_in_closure_kg']:.2f} kg.** §9.17 costed the unit as the
motor's active mass plus a 1.25 kg reducer and a 0.55 kg housing — both taken
from the CAD table of the **Ø192 can with a Ø100 bore**. But this design moves
the cycloid's pin circle from r 43.5 to r 59.3, which makes the discs much
bigger ({FCAD['mass_g']['cycloid_discs']:.0f} g for the pair), and puts them in a different can. The
built unit is {FCAD['total_g']/1e3:.2f} kg: housing {FCAD['mass_by_group_g']['housing']:.0f} g, reducer {FCAD['mass_by_group_g']['reducer']:.0f} g, bearings {FCAD['mass_by_group_g']['bearings']:.0f} g,
rotor carrier {FCAD['mass_by_group_g']['rotor']:.0f} g, motor {FCAD['mass_by_group_g']['stator'] + FCAD['mass_by_group_g']['magnets']:.0f} g.

**2. The unit is {FCAD['envelope']['height_mm']:.1f} mm tall, not 37 mm.** The 37 mm was the motor stack
plus 6 mm of case each end. It ignored the reducer's own axial stack — the
crossed roller, the output flange, two discs on the HK2512's 12 mm cup pitch,
and the rotor carrier. So the unit is not dramatically shorter than the PCB
machine's 54 mm; it is about 5 mm shorter.

**3. The mass fixed point was never re-solved.** `analysis/frameless_motor.py`
now takes the reducer and housing masses from the CAD point and **scales**
them with the pin circle and the can, so the sweep is honest at every size.
The result is a ladder, and it is the real state of the design:

{md(("What is assumed and what is measured", "Unit (kg)", "Leg structure (kg)", "Robot (kg)", "Worst margin", ""), ladder_rows)}

![The mass ladder]({rel(os.path.join(FIG, 'frameless-mass-ladder.png'))})

**On the unit as built, the design does not close at all — not at any leg mass.**
At the 1.2 kg a leg the budget has always assumed the margin is {FM['mass_ladder'][1]['worst']:.2f}; on the leg
that was actually measured it is {FM['mass_ladder'][2]['worst']:.2f}. Solving for the leg mass at which it
would close returns **{FM['leg_struct']['breakeven_kg']:.1f} kg** — the loop does not close even with a
weightless leg. The unit grew from the 2.84 kg §9.17 assumed to
{FM['cad_unit_kg']:.2f} kg as it was drawn, and that alone is enough to take the design
under.

**And a bigger motor does not rescue it.** Re-sweeping every diameter and
stack length inside the Ø{FM['limits']['od_max_mm']:.0f} the can allows, at the CAD-anchored unit mass and
the measured leg structure, **nothing closes**. Each extra newton metre costs
mass, the mass costs torque, and inside this envelope the loop does not
converge. That is a statement about the leg, not the motor: **the leg has to
get lighter before any motor choice can be validated.** No margin anywhere in
§9.17 or here can be trusted until a leg is designed for this actuator and
weighed.

**4. Deleting the capstan moved the motors out of the body.** This is the
architectural consequence §9.17 missed. The capstan was not only a reduction
stage; it was the only means by which a femur or knee motor sitting *in the
body* drove a joint *out on the leg*. Take it away and the femur and knee
units have to sit on their own joints. The project's founding premise is
"all eighteen motors in the body", and the round-14b design quietly breaks it.
The general arrangement (§10) draws the consequence: with a Ø{FCAD['envelope']['od_mm']:.0f} can on each
knee, **the robot is wider across its knee cans ({ARR['overall']['width_over_cans_mm'] if ARR and 'overall' in ARR and 'width_over_cans_mm' in ARR.get('overall', {}) else 1066:.0f} mm) than across its
feet ({ARR['overall']['width_over_feet_mm'] if ARR and 'overall' in ARR and 'width_over_feet_mm' in ARR.get('overall', {}) else 894:.0f} mm)**, and the eighteen cans no longer fit in the body slab.

**What building it also found.** Five things that only appear when the parts
are drawn to size, all recorded in `cad/actuator/frameless.json`:

1. **`analysis/cycloid.py::profile()` offset the disc the wrong way.** It
   applied the pin-radius offset along the outward normal, so every disc it
   has ever drawn came out **larger than its own pin circle** — 59.6 to 64.1 mm
   on an R 59.3 pin circle — and could not mesh; the pins buried 2.8 mm into
   the flank. The discs in `cad/actuator/actuator.py` therefore intersect the
   pin cage by about 3 mm and every disc mass in `cad/actuator/*.json` is high.
   Fixed in round 14c, with `cycloid.mesh_gap()` added as the check that
   catches it: a correct profile returns exactly the pin radius, and all three
   lobe counts now do. **The canonical PCB actuator CAD has not yet been
   re-run on the corrected profile.**
2. **The output bearing in the BOM is inadequate now the capstan is gone.**
   The RB5013 carries the whole joint moment instead of a quarter of it: a
   static safety factor of 1.16 against THK's 1–2 normal and 2–3 impact
   (cat. 382-5E p. 6). The CAD fits an **RB8016** (80 × 120 × 16, C0 42.1 kN)
   for 3.58, and that bearing is 350 g of the unit's overshoot.
3. **The eccentric bearing is now under this study's own threshold.** 11.16 kN
   per disc against the HK2512's 16.3 kN static rating is 1.46, below the 1.5
   required. The HK3012 gives 1.55 but needs a Ø30 journal and a Ø37 disc bore,
   which is not drawn.
4. **The output pins are cantilevered.** Eight Ø10 pins out of a 4 mm flange
   plate, through both discs, unsupported at the top: 3.6 kN per disc and
   748 MPa of root bending. The standard fix is a second carrier plate above
   the discs, which is about 6 mm more height and is not drawn either.
5. **The radial budget is exactly zero.** From the Ø131.6 bore to the r 59.3
   pin circle there are 6.5 mm for a Ø6 pin, its backing steel, clearance and
   the rotor carrier skirt — and they add to 6.5 mm with nothing spare. The
   ring pins sit in grooves only 2.05 mm deep, 34 % of their diameter.

**What still stands from §9.17.** The datasheet decoding (four checks within
1 %), the scaling law and its sensitivity, the shear stress, the reducer load
analysis, the thermal budget and the driver-limited peak are all unaffected —
they are about the motor, not about what it is bolted to. The frameless kit
is still the best motor found for this joint. What is not established is that
a robot built around it closes.

**What this round asks for.** Not another motor study. A leg designed for this
actuator, with the femur and knee units wherever they now have to go, weighed
honestly — and then the fixed point solved once, on measurements.

### 9.8 Open items from this round (updated in round 11)

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
    with open(os.path.join(FIG, "single-stator-opt.md"), "w") as f:            # round 13 design note, same text as §9.15
        f.write(so_section(lambda p: p, note=True))


if __name__ == "__main__":
    figs = {"layout": fig_layout(), "operating": fig_operating()}
    write_doc(figs)
    print("wrote", DOC)
    print(f"A3: T {A3['T_cont']:.2f}/{A3['T_peak']:.2f}, p {A3['p']}, coils {A3['n_coils']}, Kt {A3['Kt']:.3f}, I {A3['I_cont']:.0f} A, mass {A3['m_total']*1e3:.0f} g; total {TOTAL_A3*1e3:.0f} g, thin {TOTAL_A3_THIN*1e3:.0f} g, yaw lean {TOTAL_YAW*1e3:.0f} g")
    print("femur margin", {k: round(v, 2) for k, v in A3_J["femur"].items()})
    print("Rth trade", RTH_TRADE)

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
import actuator_section as sec                   # noqa: E402

DOC = os.path.join(ROOT, "docs", "design", "08-actuator-design.md")
FIG = os.path.join(ROOT, "docs", "design", "actuator")
os.makedirs(FIG, exist_ok=True)

ETA = 0.88                                       # cycloid efficiency assumed in the sizing
JOINTS = {                                       # ratio, joint continuous / peak N·m (01-sizing §6), swing rad/s
    "yaw":   dict(N=45, T_cont=55.0, T_peak=58.0, w_swing=8.6),
    "femur": dict(N=55, T_cont=135.0, T_peak=245.0, w_swing=3.8),
    "knee":  dict(N=60, T_cont=143.0, T_peak=300.0, w_swing=3.8),
}

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
CYC = {name: cy.design(j["N"], j["T_cont"], j["T_peak"]) for name, j in JOINTS.items()}
sec.T_BACKIRON = 3.0                                          # aluminium carrier behind a Halbach ring; 3 mm
sec.STATORS["pcb"].update(t_stator=STACK[2] * 1000, gap=mo.AIR_CLEAR * 1000, t_mag=A3["h_m"] * 1000, label="PCB stator 12-layer 3 oz")
sec.STATORS["wound"].update(t_stator=(mo.COIL_THICK + 2 * mo.COIL_SKIN) * 1000, gap=mo.AIR_CLEAR * 1000, t_mag=C1["h_m"] * 1000)
SEC_A3 = sec.draw("pcb", 1)
SEC_C1 = sec.draw("wound", 1)

RHO_STEEL, RHO_AL = 7.85e-6, 2.7e-6                          # kg/mm³


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
  per-DOF plan is now one stator at 45 / 55 / 60:1.
* **The actuator mass budget does not close at 1.1 kg.** A femur or knee unit
  rolls up to {TOTAL_A3*1e3:.0f} g with the reducer, bearings and housing (§5).
  The magnets alone are {m['m_mag']*1e3:.0f} g. This has to go back to the mass budget.
* **The whole result rests on the thermal path**: {mo.R_TH_NOMINAL} K/W from the stator
  copper to ambient through the board rim and the housing. §6 shows what
  1.0 to 3.0 K/W does.

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

The cycloid sits inside the r < 50 mm bore with its ring pins in the housing's
bore wall and two discs 180° apart on a twin-lobe eccentric, which balances the
disc mass at motor speed. From `analysis/cycloid.py`:

{md(("Joint", "Ratio", "Ring pins", "Eccentricity (mm)", "Peak ring-pin force (N, per disc)", "Hertz, peak (MPa; 1400 allowed)", "Eccentric bearing load, peak (N, per disc)", "Output-pin force, peak (N)"), crows)}

![Cycloid profiles]({rel(os.path.join(FIG, 'cycloid-profiles.png'))})

* **Discs**: 8 mm hardened steel (4140 through-hardened or case-hardened
  1018), profile laser- or water-jet cut then the lobes ground or left as-cut
  for the first units; Ø{2*(cy.R_PIN_CIRCLE-3):.0f} mm, ~{MASS_A3[list(MASS_A3)[1]]*1e3/2:.0f} g each. 5 mm discs save
  {(TOTAL_A3-TOTAL_A3_THIN)*1e3:.0f} g per actuator at {math.sqrt(8/5):.2f}× the Hertz stress, still inside the allowable.
* **Ring pins**: hardened dowel pins, Ø{2*CYC['femur']['r_pin']:.1f} mm for femur, Ø{2*CYC['yaw']['r_pin']:.1f} mm for yaw,
  pressed into the bore wall. Needle rollers on the pins would cut friction
  and are the first upgrade if efficiency measures below 85 %.
* **Eccentric bearings**: the input torque works through a {CYC['femur']['e']:.2f} mm
  eccentricity, so the radial load on each disc's bearing is
  {CYC['femur']['F_ecc']/1e3:.1f} kN at the femur peak and ~{CYC['femur']['F_ecc']/1e3*JOINTS['femur']['T_cont']/JOINTS['femur']['T_peak']:.1f} kN continuous. That is a
  drawn-cup needle bearing (HK4012 class, 40 × 47 × 12: C0 ≈ 25 kN, C ≈ 20 kN
  — catalogue values to verify), not a ball bearing. Life at the continuous
  load is well over 10⁹ revolutions.
* **Output bearing**: the yaw unit's output flange carries the coxa's
  overturning moment (270 N·m at the peak foot load), which needs a
  thin-section four-point-contact or cross-roller bearing of ~Ø110 mm on the
  output flange; the femur and knee outputs see only their drive torque.
* **Ratio note**: with a single stator at {m['T_cont']:.2f} N·m the yaw joint could drop
  to ~25:1 (e = 1.3 mm, lower bearing load, more transparent). Kept at 45:1
  this round so the three units share one disc family; revisit with the
  transmission.

## 5. Stack-up, mass and cost

![Actuator section, A3]({rel(SEC_A3[0])})

Axial stack for A3: {SEC_A3[1]['total']:.1f} mm of the 42 mm envelope, the cycloid's 18 mm
inside the motor's height. C1 needs {SEC_C1[1]['total']:.1f} mm.

{md(("Part", "Mass (g)"), mrows)}

{md(("Variant", "Mass (g)"), mrows2)}

So the femur and knee units land near {TOTAL_A3*1e3/1000:.1f} kg and a lean yaw unit near
{TOTAL_YAW*1e3/1000:.1f} kg: eighteen actuators at about {(2*TOTAL_A3 + TOTAL_YAW)*6:.0f} kg against the
{18*1.1:.0f} kg the mass budget carries. The magnets ({m['m_mag']*1e3:.0f} g per unit) and the
steel discs are the two big items; the levers are thinner discs, a smaller
magnet annulus on the yaw unit, and accepting a heavier robot. This is the
first thing to renegotiate with the mass budget.

Parts cost per unit, order of magnitude: motor ~${m['cost']:.0f}, cycloid discs and pins
~$80 (laser-cut and hardened), bearings ~$40, housing ~$60 (machined), encoder
and driver share ~$60 — about **$530 per actuator, $9.5k for eighteen**, against
the $8–11k off-the-shelf benchmark. The custom unit wins on the pancake stack
and on serviceability rather than on parts cost.

## 6. Thermal: the assumption everything rests on

The continuous rating is set by {mo.R_TH_NOMINAL} K/W from copper to ambient. The
board's own radial conduction is small (thin copper, short path to the rim),
so the number is really the bond from the board rim to the housing and from
the housing to the body.

{md(("R_th stator→ambient (K/W)", "A3 continuous torque (N·m)", "Femur margin"), rrow)}

Three motors share one hip pod, so at full continuous rating the pod has to
reject about {3*(m['P_cu']+m['P_eddy']):.0f} W. On a walk the RMS torque is well under the corner
rating the joints are sized to — at half the continuous torque the losses
are a quarter — so the pod's steady load is nearer {3*(m['P_cu']+m['P_eddy'])/4:.0f} W. The design
rules that follow: the stator rim clamped, not glued, into an aluminium ring
that is part of the housing; the housing bolted face-to-face to the hip pod
plate; a temperature sensor on every stator; and the dyno test that measures
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

## 9. What the first unit has to prove

1. R_th of the stator in its housing, on the bench, with a heater and then
   with current: the continuous rating is that number.
2. Kt and R_ph against this model (±10 % is a pass).
3. Cycloid efficiency and noise at 55:1 with laser-cut discs and plain pins.
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

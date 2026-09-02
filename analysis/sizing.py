#!/opt/hw-py/bin/python
"""Generate docs/design/01-sizing.md and its figures from analysis/hexapod_model.py.

    /opt/hw-py/bin/python analysis/sizing.py

Every number in the document comes from the model; iterate on the parameters
there, not on the prose here.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from hexapod_model import (ACT, BODY, ENERGY, GAITS, LOAD_CASES, MASS, STANCES, G, PcbMotor,
                           Reduction, ik, joint_from_motor, joint_speeds, knee_pos,
                           motor_speed_for, stance_torques, tip_angles)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "design", "01-sizing.md")
FIG = os.path.join(ROOT, "docs", "design", "sizing")
os.makedirs(FIG, exist_ok=True)
CASE = {c.rating: c for c in LOAD_CASES}
RIDER = CASE["10 min"]

# ----------------------------------------------------------------------------
# Derived actuator requirement.  Rating = worst joint across both stances.
# ----------------------------------------------------------------------------
def worst(rating: str) -> float:
    return max(max(stance_torques(s, CASE[rating]).values()) for s in STANCES)

REQ = {"continuous": worst("continuous"), "10 min": worst("10 min"), "peak": worst("peak")}
REACH_LIMIT = 300.0   # the controller lets the foot go this far out at full peak load
REQ["peak_reach"] = max(stance_torques(s, CASE["peak"], foot_reach=REACH_LIMIT)["femur"] for s in STANCES)
# Structural: overturning moment the yaw bearing and the coxa carry (not a motor torque)
COXA_MOMENT = max(CASE["peak"].foot_force_z * s.foot_radius() / 1000 for s in STANCES)

SPEED_REQ = {g.label: max(max(joint_speeds(s, g).values()) for s in STANCES) for g in GAITS}
JOINT_SPEED_NOMINAL = SPEED_REQ["Walk, tripod gait"]
JOINT_SPEED_FAST = SPEED_REQ["Fast walk, tripod gait (aspirational)"]
STANCE_RATE = max(joint_speeds(s, GAITS[1])["yaw_stance"] for s in STANCES)

MOTOR_RPM_MAX = 5500.0          # PCB stator eddy losses and rotor tip speed cap this
RATIOS = (30, 40, 50, 60, 80)
MOTOR_1 = PcbMotor(stators=1)
MOTOR_2 = PcbMotor(stators=2)


def joint_speed_at(ratio: float) -> float:
    return MOTOR_RPM_MAX * 2 * math.pi / 60 / ratio


def first_ratio_that_closes(m: PcbMotor):
    """Smallest ratio (1:1 steps) at which the motor meets both torque ratings."""
    for r in range(20, 121):
        j = joint_from_motor(m, Reduction(r))
        if j["cont"] >= REQ["continuous"] and j["peak"] >= REQ["peak"]:
            return r
    return None


R1 = first_ratio_that_closes(MOTOR_1)
R2 = first_ratio_that_closes(MOTOR_2)
# Nominal: the single stator if it closes with speed to spare, else two stators
if R1 is not None and joint_speed_at(R1) >= JOINT_SPEED_NOMINAL:
    MOTOR, RATIO = MOTOR_1, Reduction(float(max(R1, 50)))
else:
    MOTOR, RATIO = MOTOR_2, Reduction(float(R2 or 50))
JOINT = joint_from_motor(MOTOR, RATIO)
JOINT_SPEED_AT_MAX_RPM = joint_speed_at(RATIO.ratio)

# Electrical
KV_RPM_V = MOTOR_RPM_MAX * 1.15 / MOTOR.bus_v            # leave 15 % for FOC headroom
KT = 60 / (2 * math.pi * KV_RPM_V)                       # N·m/A (q-axis)
I_CONT = MOTOR.torque_cont / KT
I_PEAK = MOTOR.torque_peak / KT

# Power
P_JOINT_CONT = REQ["continuous"] * STANCE_RATE
P_JOINT_PEAK = REQ["peak"] * 5.0
P_WALK = ENERGY.walking_power(MASS.robot + MASS.mission_payload, 1.0)
P_RIDER = ENERGY.walking_power(MASS.robot + MASS.rider, 0.3)
PACK_WH = ENERGY.pack_wh(P_WALK)
PACK_KG = ENERGY.pack_kg(P_WALK)
PACK_AH = PACK_WH / MOTOR.bus_v
I_BUS_PEAK = 6 * P_JOINT_PEAK / MOTOR.bus_v * 0.5   # six joints near peak, 50 % coincidence


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_leg_side():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    for ax, st in zip(axes, STANCES):
        leg = st.leg
        r, z = st.foot_reach, -st.hip_height
        a1, k = ik(leg, r, z)
        kx, kz = knee_pos(leg, a1)
        ax.add_patch(plt.Rectangle((-leg.coxa - BODY.width / 2, 0), BODY.width / 2, BODY.height, color="#bdc3c7"))
        ax.annotate("body", (-leg.coxa - BODY.width / 4, BODY.height / 2), ha="center", fontsize=9, color="#555")
        ax.plot([-leg.coxa, 0], [0, 0], "k-", lw=6, solid_capstyle="round", label="coxa")
        ax.plot([0, kx], [0, kz], "-", color="#c0392b", lw=6, solid_capstyle="round", label="femur")
        ax.plot([kx, r], [kz, z], "-", color="#2980b9", lw=6, solid_capstyle="round", label="tibia")
        ax.plot([-leg.coxa - BODY.width / 2 - 20, 420], [z - 22, z - 22], "-", color="#7f8c8d", lw=2)
        for (x, y) in ((0, 0), (kx, kz), (-leg.coxa, 0)):
            ax.plot(x, y, "o", ms=10, color="k")
        t = stance_torques(st, RIDER)
        ax.annotate(f'femur {t["femur"]:.0f} N·m', (0, 0), (-10, -34), textcoords="offset points", fontsize=10, ha="right")
        ax.annotate(f'knee {t["knee"]:.0f} N·m', (kx, kz), (10, 6), textcoords="offset points", fontsize=10)
        ax.annotate(f'yaw {t["yaw"]:.0f} N·m', (-leg.coxa, 0), (-10, 14), textcoords="offset points", fontsize=10, ha="right")
        ax.annotate(f'{RIDER.foot_force_z:.0f} N', (r, z), (12, 8), textcoords="offset points", fontsize=10)
        ax.arrow(r, z - 150, 0, 110, width=6, color="#27ae60", length_includes_head=True)
        ax.annotate("", (r, -60), (0, -60), arrowprops=dict(arrowstyle="<->", color="#c0392b"))
        ax.annotate(f"arm {r:.0f} mm", (r / 2, -60), (0, -14), textcoords="offset points", ha="center", fontsize=9, color="#c0392b")
        ax.set_title(f"{st.name}\ncoxa {leg.coxa:.0f} / femur {leg.femur:.0f} / tibia {leg.tibia:.0f} mm\n"
                     f"femur {st.femur_deg:+.0f}°, knee {math.degrees(k):.0f}°, hip {st.hip_height:.0f} mm up", fontsize=10)
        ax.set_xlabel("r — outward in leg plane (mm)")
        ax.set_aspect("equal")
        ax.set_xlim(-leg.coxa - BODY.width / 2 - 40, 440)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("z (mm)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(f"Leg in its plane, neutral stance, rider load case ({RIDER.foot_force_z:.0f} N per foot, quasi-static)")
    fig.tight_layout()
    p = os.path.join(FIG, "leg-stance.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_torque_map():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    for row, st in zip(axes, STANCES):
        leg = st.leg
        rs = np.linspace(20, 420, 81)
        hs = np.linspace(120, leg.reach - 40, 81)
        fem = np.full((len(hs), len(rs)), np.nan)
        kne = np.full_like(fem, np.nan)
        Fz = RIDER.foot_force_z
        for i, h in enumerate(hs):
            for j, r in enumerate(rs):
                try:
                    a1, _ = ik(leg, r, -h)
                except ValueError:
                    continue
                kx, kz = knee_pos(leg, a1)
                fem[i, j] = abs(r * Fz) / 1000
                kne[i, j] = abs((r - kx) * Fz) / 1000
        for ax, data, name in ((row[0], fem, "femur"), (row[1], kne, "knee")):
            cs = ax.contourf(rs, hs, data, levels=np.arange(0, 280, 20), cmap="viridis")
            ax.contour(rs, hs, data, levels=[REQ["continuous"], REQ["10 min"], REQ["peak"]],
                       colors=["w", "orange", "r"], linewidths=1.5)
            ax.plot(st.foot_reach, st.hip_height, "o", ms=10, color="w", mec="k", label="neutral stance")
            ax.axvline(REACH_LIMIT, color="r", ls=":", lw=1)
            ax.set_xlabel("foot reach r from femur axis (mm)")
            ax.set_ylabel("hip height above ground (mm)")
            ax.set_title(f"{st.name}: {name} torque, vertical load only ({Fz:.0f} N)", fontsize=10)
            fig.colorbar(cs, ax=ax, label="N·m")
        row[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Quasi-static joint torque over the foot workspace at the rider load. "
                 f"Contours: {REQ['continuous']:.0f} (white) / {REQ['10 min']:.0f} (orange) / {REQ['peak']:.0f} (red) N·m; "
                 f"dotted: {REACH_LIMIT:.0f} mm reach limit", fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIG, "torque-map.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_ratio_trade():
    ratios = np.linspace(20, 100, 81)
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    for m, ls, lab in ((MOTOR_1, "--", "1 stator"), (MOTOR_2, "-", "2 stators")):
        ax1.plot(ratios, [joint_from_motor(m, Reduction(r))["cont"] for r in ratios], ls, color="#2980b9", label=f"continuous, {lab}")
        ax1.plot(ratios, [joint_from_motor(m, Reduction(r))["peak"] for r in ratios], ls, color="#c0392b", label=f"peak, {lab}")
    for key, col in (("continuous", "#2980b9"), ("peak", "#c0392b")):
        ax1.axhline(REQ[key], color=col, alpha=0.35, lw=8)
    ax1.set_xlabel("reduction ratio")
    ax1.set_ylabel("joint torque (N·m)")
    ax1.set_ylim(0, 500)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ratios, [joint_speed_at(r) for r in ratios], color="#27ae60", lw=2, label=f"joint speed at {MOTOR_RPM_MAX:.0f} rpm")
    ax2.axhline(JOINT_SPEED_NOMINAL, color="#27ae60", alpha=0.35, lw=8)
    ax2.axhline(JOINT_SPEED_FAST, color="#27ae60", alpha=0.2, lw=8, ls=":")
    ax2.set_ylabel("joint speed (rad/s)")
    ax2.set_ylim(0, 30)
    ax1.axvline(RATIO.ratio, color="k", ls=":")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax1.set_title(f"PCB axial-flux motor Ø{2*MOTOR.r_out:.0f} mm + in-plane cycloid: torque and speed vs ratio\n"
                  "bands = requirement (continuous / peak torque, nominal / fast joint speed); dotted = nominal ratio", fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIG, "ratio-trade.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------------
def rel(p):
    return os.path.relpath(p, os.path.dirname(DOC)).replace(os.sep, "/")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def short(st):
    return st.name.split(" — ")[0]


def write_doc(figs):
    m = MASS
    A, B = STANCES
    mass_rows = [
        ("18 × joint actuators", f"{m.actuators:.1f}", f"{ACT.mass:.1f} kg each, target"),
        ("Body structure", f"{m.body_structure:.1f}", "plates, hip pods, top deck"),
        ("6 × legs", f"{m.legs:.1f}", "links, feet, in-leg transmission share"),
        ("2 × hot-swap batteries", f"{m.batteries:.1f}", f"~{PACK_WH:.0f} Wh each, see §8"),
        ("Electronics", f"{m.electronics:.1f}", "compute, 18 drivers, sensors, harness"),
        ("Payload interface", f"{m.payload_interface:.1f}", "deck rails, tool mount, solar skin"),
        ("Margin", f"{m.margin:.1f}", ""),
        ("**Robot, unloaded**", f"**{m.robot:.1f}**", ""),
        ("Mission payload", f"{m.mission_payload:.1f}", "trash + gripper"),
        ("Rider (emergency / demo)", f"{m.rider:.1f}", "adult male"),
        ("**Gross, with rider**", f"**{m.robot + m.rider:.1f}**", ""),
    ]

    geom_rows = [
        ("Body slab, length × width × height", f"{BODY.length:.0f} × {BODY.slab_width(ACT):.0f} × {BODY.height:.0f} mm",
         "width = yaw-axis spacing + one actuator diameter + two side rails"),
        ("Hip yaw axes", f"x = {', '.join(f'{x:+.0f}' for x in BODY.hip_x)} mm, y = ±{BODY.width/2:.0f} mm",
         "six vertical axes, under the body"),
        ("Actuator envelope", f"Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm, {ACT.mass:.1f} kg", "pancake; three stacked per hip on the yaw axis"),
        ("Hip stack height", f"{3*ACT.thickness + 2*ACT.stack_gap:.0f} mm", f"fits inside the {BODY.height:.0f} mm slab"),
        ("Top deck height, neutral stance", f"{A.hip_height + BODY.height:.0f}–{B.hip_height + BODY.height:.0f} mm", "the rider's seat and the solar skin"),
    ]

    stance_rows = []
    for st in STANCES:
        leg = st.leg
        t0 = tip_angles(BODY, st, False)
        stance_rows.append((short(st), f"{leg.coxa:.0f} / {leg.femur:.0f} / {leg.tibia:.0f}", f"{leg.tibia_ratio:.1f}×",
                            f"{st.femur_deg:+.0f}° / {st.tibia_lean_deg:.0f}°", f"{st.hip_height:.0f}", f"{st.knee_height:.0f}",
                            f"**{st.foot_reach:.0f}**", f"{st.foot_radius():.0f}", f"{t0['stance_width_m']:.2f} × {t0['stance_length_m']:.2f}",
                            f"{leg.reach:.0f}"))

    load_rows = [(c.name, c.rating, f"{c.total_mass:.0f}", c.legs_down, f"{c.dyn_factor:.1f}", f"{c.slope_deg:.0f}° + {c.accel:.1f} m/s²",
                  f"**{c.foot_force_z:.0f}**", f"{c.foot_force_prop:.0f}") for c in LOAD_CASES]

    torque_rows = []
    for st in STANCES:
        for c in LOAD_CASES:
            t = stance_torques(st, c)
            torque_rows.append((short(st), c.name, c.rating, f"{t['yaw']:.0f}", f"**{t['femur']:.0f}**", f"{t['knee']:.0f}"))

    gait_rows = [(g.label, f"{g.speed:.1f}", f"{g.stride*1000:.0f}", f"{g.duty:.2f}", f"{g.cycle_hz:.2f}",
                  f"{g.t_swing*1000:.0f}", f"{g.swing_speed_peak():.1f}") for g in GAITS]
    speed_rows = []
    for st in STANCES:
        for g in GAITS:
            s = joint_speeds(st, g)
            speed_rows.append((short(st), g.label, f"{s['yaw_swing']:.1f}", f"{s['yaw_stance']:.1f}",
                               f"{s['pitch_swing']:.1f}", f"{s['pitch_stance']:.1f}"))

    stab_rows = []
    for st in STANCES:
        for rider in (False, True):
            t = tip_angles(BODY, st, rider)
            stab_rows.append((short(st), "with rider" if rider else "unloaded", f"{t['com_m']:.2f}",
                              f"{t['roll_deg']:.0f}°", f"{t['pitch_deg']:.0f}°"))

    ratio_rows = []
    for r in RATIOS:
        red = Reduction(r)
        j1, j2 = joint_from_motor(MOTOR_1, red), joint_from_motor(MOTOR_2, red)
        ratio_rows.append((f"{r}:1", f"{j1['cont']:.0f} / {j1['peak']:.0f}", f"{j2['cont']:.0f} / {j2['peak']:.0f}",
                           f"{joint_speed_at(r):.1f}", f"{motor_speed_for(JOINT_SPEED_NOMINAL, red):.0f}", f"{motor_speed_for(JOINT_SPEED_FAST, red):.0f}"))

    spec_rows = [
        ("Joint torque, continuous (thermal steady state)", f"**{REQ['continuous']:.0f} N·m**", "walk, tripod, 1.5× dynamic factor, 30° slope — any joint, either leg"),
        ("Joint torque, 10-minute rating", f"**{REQ['10 min']:.0f} N·m**", "rider on board, wave gait, 15° slope"),
        ("Joint torque, peak (≤ 2 s)", f"**{REQ['peak']:.0f} N·m**", "stumble onto two legs at 3× dynamic factor, neutral stance"),
        ("Joint torque, peak at extended reach", f"{REQ['peak_reach']:.0f} N·m", f"same case with the foot out at {REACH_LIMIT:.0f} mm; software-limited region"),
        ("Joint speed, loaded (stance)", f"≥ {STANCE_RATE:.0f} rad/s at continuous torque", "1 m/s walk"),
        ("Joint speed, unloaded (swing)", f"≥ **{JOINT_SPEED_NOMINAL:.0f} rad/s**", "1 m/s walk, 0.4 m stride; 2 m/s needs " f"{JOINT_SPEED_FAST:.0f} rad/s"),
        ("Joint range", "yaw ±60°, femur −60…+90°, knee 20…160°", "fold-flat for transport, step-over 300 mm obstacle, crouch to the deck"),
        ("Yaw bearing overturning moment (structural)", f"{COXA_MOMENT:.0f} N·m", "peak foot load × foot radius; carried by the coxa and the yaw bearing, not the motor"),
        ("Backdrivability / sensing", "joint-torque estimate ≤ 10 % error, foot contact detection", "high ratio ⇒ cannot rely on motor-current transparency; needs output-side sensing or SEA"),
        ("Envelope", f"Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm", "pancake, axis = hip yaw axis"),
        ("Mass", f"≤ {ACT.mass:.1f} kg", ""),
        ("Count", "18, identical", "yaw joints run at ⅓–½ of pitch torque; commonality wins at this stage"),
        ("Bus voltage", f"{MOTOR.bus_v:.0f} V", "13S Li-ion; 48 V drivers and connectors are commodity"),
        ("Continuous mechanical power", f"~{P_JOINT_CONT:.0f} W", "continuous torque × stance rate"),
        ("Peak mechanical power", f"~{P_JOINT_PEAK/1000:.1f} kW", "peak torque × 5 rad/s"),
        ("Environment", "IP54 body, IP65 legs, −10…+45 °C", "outdoor, mud, rain; heat rejection at 45 °C ambient sizes the stator copper"),
    ]

    one_stator_line = (f"closes both ratings at **{R1}:1**, where the joint's top speed at {MOTOR_RPM_MAX:.0f} rpm is "
                       f"{joint_speed_at(R1):.1f} rad/s against the {JOINT_SPEED_NOMINAL:.0f} rad/s the 1 m/s walk needs"
                       if R1 else "does not close both ratings at any ratio up to 120:1")
    two_stator_line = (f"closes at **{R2}:1** with {joint_speed_at(R2):.1f} rad/s to spare at the joint" if R2 else "does not close")
    nominal_desc = f"{MOTOR.stators} stator{'s' if MOTOR.stators > 1 else ''}, {RATIO.ratio:.0f}:1"

    doc = f"""# 01 — Top-level sizing: shape, loads, and what the actuator has to be

*Generated by `analysis/sizing.py` from `analysis/hexapod_model.py`. Change the
parameters, re-run, and every number and figure here moves together.*

This is the first design pass: it fixes the rough size and shape of the
hexapod, the loads that shape produces at the joints, and therefore the
torque, speed, reduction ratio and power an actuator must deliver. It stops
at the actuator requirement. Transmission (motors in the body, power to the
joints), electronics and software are downstream and are not decided here.

Two leg proportions are carried side by side — **A, tibia {A.leg.tibia_ratio:.1f}× femur** and
**B, tibia {B.leg.tibia_ratio:.1f}× femur** — because the [vision document](vision.md) puts
both in front of you as rendered skeletons. The actuator requirement below is
the envelope of both, so the actuator stage can start before the proportion
is settled.

## 1. What it is sized for

| Mission | What it drives |
|---|---|
| Complex outdoor terrain (first demo) | 300 mm step-over, 30° slope, foot workspace, joint speed, weather sealing |
| Trash pickup (second demo) | 8 kg mission payload with a tool on the top deck; a free leg or a deck arm — not decided |
| Carry an adult (if needed, slow) | 100 kg on the top deck at a wave-gait crawl: the 10-minute torque rating and roll stability |
| Large-dog size | ~{BODY.length/1000:.1f} m body, hips {A.hip_height/1000:.2f}–{B.hip_height/1000:.2f} m above ground, top deck at ~{(A.hip_height + BODY.height)/1000:.2f}–{(B.hip_height + BODY.height)/1000:.2f} m |

The rider case is what makes this a serious machine rather than a large
hobby hexapod: it roughly doubles the vertical load per foot. It is treated as a
**10-minute rating at crawl speed**, not as the continuous design point, so
it sets copper temperature and gearbox pin loads but not the battery or the
continuous speed.

### Mass budget

{md_table(("Item", "kg", "Basis"), mass_rows)}

Everything downstream scales with the unloaded mass, and the unloaded mass is
dominated by the eighteen actuators. **A 1.1 kg actuator target is the single
most important number in this budget**: at 1.6 kg each the robot is 56 kg
and every torque below grows by 15 %.

## 2. Shape

### The body

Putting all eighteen motors in the body, as pancakes stacked three-high on
each hip's yaw axis, is what sets the body's proportions: the hip stack is
{3*ACT.thickness + 2*ACT.stack_gap:.0f} mm tall and Ø{ACT.od:.0f} mm, so the body is a
{BODY.height:.0f} mm slab whose width is set by the stacks, not by the batteries.
The yaw axes sit under the body at ±{BODY.width/2:.0f} mm; the batteries, compute and
payload bay fill the space between the six stacks. The flat top is the solar
skin and the payload/rider deck.

{md_table(("Parameter", "Value", "Why"), geom_rows)}

### The leg

The joint torque from a vertical foot load is that load times the
**horizontal** distance from the joint axis to the foot — the link angles do
not enter. So the leg is shaped to keep the foot horizontally close to the
femur axis while the stance stays wide:

* a **coxa** of {A.leg.coxa:.0f} mm carries the leg out from under the body. It is a
  rigid arm on the yaw output, so it takes the vertical load as a bending
  moment into the yaw bearing, not as motor torque — the sprawl is free;
* the **femur** pitches up and out from the end of the coxa, at
  {A.femur_deg:.0f}–{B.femur_deg:.0f}° in the neutral stance, so the knee sits high and only
  `femur × cos(angle)` outboard of the femur axis;
* the **tibia**, at least twice the femur, comes straight down from the knee
  to the foot. Vertical, it adds nothing to the femur arm and puts no load
  torque on the knee at all.

The two proportions on the table:

| | Coxa / femur / tibia (mm) | Tibia ratio | Femur / tibia lean | Hip height (mm) | Knee height (mm) | Femur arm (mm) | Foot radius from yaw axis (mm) | Footprint W × L (m) | Leg reach (mm) |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join('| ' + ' | '.join(r) + ' |' for r in stance_rows)}

**A** is squatter: the knee just below deck level, a bigger femur arm, a
wider footprint. **B** is taller with the knee higher and the femur arm
{A.foot_reach - B.foot_reach:.0f} mm shorter, which is worth {(1 - B.foot_reach / A.foot_reach) * 100:.0f} % of femur torque in every load
case, at the price of a longer, heavier tibia to swing.

![Leg in its plane at the rider load]({rel(figs['leg'])})

## 3. Loads and joint torques

Quasi-static: a vertical ground reaction per supporting foot, plus a
propulsion force to hold a slope and accelerate. Dynamic factors stand in for
impact and gait dynamics until there is a simulation.

{md_table(("Load case", "Rating", "Mass on feet (kg)", "Legs down", "Dyn. factor", "Slope + accel", "Vertical / foot (N)", "Propulsion / foot (N)"), load_rows)}

Joint torques at each neutral stance (N·m):

{md_table(("Leg", "Load case", "Rating", "Yaw", "Femur", "Knee"), torque_rows)}

The femur is the sized joint in both, and its torque is almost exactly
`vertical load × femur arm`. The knee torque at the neutral stance comes
from the in-plane share of the propulsion force acting over the tibia's
height (the front and rear legs are yawed 30°, so half their push is in the
leg plane), not from the vertical load. The yaw joint sees a third of the
femur duty. All of that holds only at the neutral stance:

![Torque map over the foot workspace]({rel(figs['map'])})

The map is vertical load only, over each leg's reachable workspace. The
femur torque is simply reach × load, so the controller's job is to keep the
foot near the femur axis: inside the orange contour under a rider, inside
the red one otherwise, and never past the {REACH_LIMIT:.0f} mm dotted line at full
load. The knee is unloaded along the curve where the tibia is vertical and
picks up torque either side of it, as fast for a foot pulled in under the
body as for one reached out.

## 4. Speed

Stride and duty factor fix the cycle, and the cycle fixes the swing time —
the swing leg has to cover the stride plus its lift in the time the stance
legs take to walk the body forward.

{md_table(("Gait", "Body speed (m/s)", "Stride (mm)", "Duty", "Cycle (Hz)", "Swing time (ms)", "Peak foot speed in swing (m/s)"), gait_rows)}

Peak joint rates (rad/s), swing (unloaded) and stance (under load). With
radial leg planes the stride is a yaw sweep, so yaw is the fast joint:

{md_table(("Leg", "Gait", "Yaw, swing", "Yaw, stance", "Pitch, swing", "Pitch, stance"), speed_rows)}

The **1 m/s walk needs ~{JOINT_SPEED_NOMINAL:.0f} rad/s** unloaded at the yaw joint. The 2 m/s
aspiration needs ~{JOINT_SPEED_FAST:.0f} rad/s and, as §7 shows, is where the reduction
ratio and the motor's speed limit start fighting each other.

## 5. Stability

Static tip-over angles on the tripod support polygon, the narrowest polygon
the gaits use. With a rider the wave gait keeps five feet down and the
polygon is much larger; the tripod numbers are the floor.

{md_table(("Leg", "Load", "CoM height (m)", "Roll tip angle", "Pitch tip angle"), stab_rows)}

Roll is the limit and the coxa is what buys it: the {A.leg.coxa:.0f} mm coxa puts
{2*A.leg.coxa/1000:.2f} m into the stance width at no torque cost. Under a rider on a tripod
both legs are in the {tip_angles(BODY, B, True)['roll_deg']:.0f}–{tip_angles(BODY, A, True)['roll_deg']:.0f}° range; the wave gait
and splayed feet recover most of it.

## 6. The actuator requirement

Envelope of both legs and all load cases. This is the input to the
actuator design stage.

{md_table(("Quantity", "Requirement", "Driving case"), spec_rows)}

## 7. Motor and reduction: is the PCB axial-flux + in-plane cycloid concept in range?

The concept: a PCB stator between two Halbach-array rotors, the magnets on
an annulus at large radius (r = {MOTOR.r_in:.0f}–{MOTOR.r_out:.0f} mm), and a cycloidal reducer in the
same plane inside the annulus. Torque of an axial-flux motor scales as the
shear stress on the airgap times ∫r² dA, so a thin ring at large radius gives
up little torque and frees the centre for the reducer:

T = σ · (2π/3) · (r_out³ − r_in³) · (number of stators)

The shear stress σ is the whole question for a PCB stator. Iron-core radial
motors with good cooling reach 10–15 kPa continuous; an ironless PCB stator
has a magnetic airgap the thickness of the board, copper fill limited by
trace spacing, and ~1–1.5 mm of copper at best across a 12-layer board. With
B ≈ 0.6 T from dual Halbach rotors and 5–6 A/mm² continuous, the model
assumes **σ = {MOTOR.sigma_cont/1000:.1f} kPa continuous and {MOTOR.sigma_peak/1000:.1f} kPa for 2 s bursts** — numbers to
be earned on a dyno, not assumed for long. That gives:

| Motor | Continuous (N·m) | Peak (N·m) |
|---|---|---|
| 1 stator, 2 rotors | {MOTOR_1.torque_cont:.2f} | {MOTOR_1.torque_peak:.2f} |
| 2 stators, 3 rotors | {MOTOR_2.torque_cont:.2f} | {MOTOR_2.torque_peak:.2f} |

Against the requirement, through a cycloid at {RATIO.efficiency*100:.0f} % efficiency:

{md_table(("Ratio", "1 stator: cont / peak (N·m)", "2 stators: cont / peak (N·m)", f"Joint rad/s at {MOTOR_RPM_MAX:.0f} rpm", "Motor rpm for 1 m/s walk", "Motor rpm for 2 m/s"), ratio_rows)}

![Torque and speed vs reduction ratio]({rel(figs['ratio'])})

Reading the table against the {REQ['continuous']:.0f} / {REQ['peak']:.0f} N·m requirement and the {JOINT_SPEED_NOMINAL:.0f} rad/s
swing rate:

* **A single PCB stator** {one_stator_line}.
* **Two stators (three rotors)** {two_stator_line}.
* **Nominal actuator architecture: Ø{2*MOTOR.r_out:.0f} mm, {nominal_desc} in-plane cycloid,
  {MOTOR.bus_v:.0f} V** — {JOINT['cont']:.0f} N·m continuous, {JOINT['peak']:.0f} N·m peak, {JOINT_SPEED_AT_MAX_RPM:.1f} rad/s at
  the joint. That meets the walk, the rider rating and the peak, and leaves
  the 2 m/s aspiration short. The leg geometry decision in §2 is what brought
  a single stator into range; the second stator stays as the fallback if the
  dyno comes in under {MOTOR.sigma_cont/1000:.1f} kPa.
* The motor's speed ceiling is set by the PCB, not the rotor: at {MOTOR_RPM_MAX:.0f} rpm
  with ~12 pole pairs the electrical frequency is ~{MOTOR_RPM_MAX/60*12:.0f} Hz, and eddy
  losses in wide PCB traces grow with the square of that. Trace segmentation
  or higher-layer-count thin traces are the levers; the stator design will
  have to check this.
* Electrical: Kv ≈ {KV_RPM_V:.0f} rpm/V (Kt ≈ {KT:.3f} N·m/A) at {MOTOR.bus_v:.0f} V ⇒ {I_CONT:.0f} A continuous and
  {I_PEAK:.0f} A for the 2 s peak, per motor. Half-bridge MOSFETs at that rating
  are commodity; the bus capacitance and the connectors for eighteen of them
  are the real layout problem. A lower Kv trades peak current for top speed.

What the in-plane cycloid concept implies, to be taken into the actuator
stage rather than solved here:

* A single-stage cycloid ratio equals its lobe count. {RATIO.ratio:.0f} lobes on a disc that
  fits inside a Ø{2*MOTOR.r_in:.0f} mm bore means a lobe pitch of ~{2*math.pi*(MOTOR.r_in-8)/RATIO.ratio:.1f} mm and pins of
  ~3 mm: fine for a machined steel disc and hardened pins, marginal for
  printed or aluminium parts at {REQ['peak']:.0f} N·m.
* The output torque reacts through those pins at a radius of ~{MOTOR.r_in-8:.0f} mm, so the
  net pin load at peak is ~{REQ['peak']/((MOTOR.r_in-8)/1000)/1000:.1f} kN spread over roughly half the pins.
  A two-stage compound cycloid in the same plane would halve the lobe count
  per stage at the cost of an extra disc.
* High ratio means low transparency: the reflected motor inertia through
  {RATIO.ratio:.0f}:1 is {RATIO.ratio**2:,.0f}× the rotor's own. Foot-contact detection and
  torque control cannot come from motor current alone; plan on an output-side
  torque sensor, a series-elastic element, or foot force sensing.

**Off-the-shelf benchmark.** Actuators in the ~120 N·m peak, ~60–70:1, sub-1 kg
class (the CubeMars AK80-64 and MyActuator RMD-X8-Pro families are the usual
suspects; verify numbers against their datasheets before relying on them)
already cover this requirement at roughly $400–600 each, i.e. $8–11k for
eighteen. That is the bar the custom actuator has to beat on cost, on the
pancake form factor that lets it stack on the yaw axis, or on serviceability
— and the first prototype leg should probably be built on them so that the
transmission and the software are not waiting on the motor.

## 8. Power and energy

| Quantity | Value | Basis |
|---|---|---|
| Average electrical power, 1 m/s walk | {P_WALK:.0f} W | cost of transport {ENERGY.cost_of_transport:.1f} × m·g·v + {ENERGY.hotel_w:.0f} W hotel |
| Average electrical power, rider at 0.3 m/s | {P_RIDER:.0f} W | same model |
| Endurance target | {ENERGY.endurance_h:.0f} h walking | mission length |
| Battery, each of {ENERGY.packs} | **{PACK_WH:.0f} Wh, {PACK_AH:.0f} Ah at {MOTOR.bus_v:.0f} V, ~{PACK_KG:.1f} kg** | {ENERGY.pack_wh_per_kg:.0f} Wh/kg pack level |
| Peak bus current | ~{I_BUS_PEAK:.0f} A | six joints near peak power, 50 % coincidence |
| Hot-swap | either pack alone must carry the full peak | a pack coming out must not brown out the drivers |

A cost of transport of {ENERGY.cost_of_transport:.1f} is pessimistic for a good quadruped and
realistic for a geared hexapod with cycloids in the loop; it puts the design
in the ~{2*PACK_WH/1000:.1f} kWh, two {PACK_KG:.0f} kg packs range, which the mass budget carries.

## 9. What this pass decided and what it left open

Decided (or at least proposed for your review):

1. Robot ~{MASS.robot:.0f} kg, {BODY.length/1000:.1f} × {BODY.slab_width(ACT)/1000:.2f} × {BODY.height/1000:.1f} m slab body with the hips under it, hips {A.hip_height/1000:.2f}–{B.hip_height/1000:.2f} m up.
2. Sprawled legs: {A.leg.coxa:.0f} mm coxa out from the hip, femur up and out at {A.femur_deg:.0f}–{B.femur_deg:.0f}°, tibia vertical and {A.leg.tibia_ratio:.1f}–{B.leg.tibia_ratio:.1f}× the femur.
3. Eighteen identical pancake actuators, three per hip stacked on the yaw axis, in the body.
4. Actuator requirement: {REQ['continuous']:.0f} N·m continuous, {REQ['10 min']:.0f} N·m for 10 min, {REQ['peak']:.0f} N·m peak, {JOINT_SPEED_NOMINAL:.0f} rad/s, Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm, ≤ {ACT.mass:.1f} kg.
5. Nominal actuator architecture: Ø{2*MOTOR.r_out:.0f} mm PCB axial-flux motor, {nominal_desc} in-plane cycloid, {MOTOR.bus_v:.0f} V.
6. Two {PACK_WH:.0f} Wh hot-swap packs.

Open, in order of how much they change the actuator:

* **Which leg proportion** (A or B) — see the vision document. It moves the femur torque by {(1 - B.foot_reach / A.foot_reach) * 100:.0f} % and the knee height by {B.knee_height - A.knee_height:.0f} mm.
* **Whether the rider case stays.** Dropping it removes the 10-minute rating and halves the gearbox pin loads.
* **σ for the PCB stator.** The whole motor concept rests on ~{MOTOR.sigma_cont/1000:.1f} kPa continuous. The first actuator-stage task is a single-stator test motor on a dyno.
* **Transmission from the body to the femur and knee, through the coxa.** Not sized here; it adds losses (budget 10 %) and reflected inertia, and it decides whether the three pancakes really can be coaxial.
* **Dynamic simulation** to replace the 1.5× / 3× factors with numbers.
"""
    with open(DOC, "w") as f:
        f.write(doc)


if __name__ == "__main__":
    figs = {"leg": fig_leg_side(), "map": fig_torque_map(), "ratio": fig_ratio_trade()}
    write_doc(figs)
    print(f"wrote {DOC}")
    print("requirement:", {k: round(v) for k, v in REQ.items()}, "coxa moment", round(COXA_MOMENT))
    print("speeds:", {k: round(v, 1) for k, v in SPEED_REQ.items()})
    print(f"closing ratios: 1 stator {R1}, 2 stators {R2}; nominal {MOTOR.stators} stator(s) @ {RATIO.ratio:.0f}:1 ->",
          {k: round(v) for k, v in JOINT.items()}, "joint rad/s", round(JOINT_SPEED_AT_MAX_RPM, 1))
    print("power W walk/rider:", round(P_WALK), round(P_RIDER), "pack Wh/kg", round(PACK_WH), round(PACK_KG, 1))

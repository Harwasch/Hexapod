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
from hexapod_model import (ACT, BODY, ENERGY, GAITS, LEG, LOAD_CASES, MASS, STANCE_SAGITTAL,
                           STANCE_SPRAWL, G, PcbMotor, Reduction, com_height, ik, joint_from_motor,
                           joint_speeds, knee_pos, motor_speed_for, stance_torques, tip_angles)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "design", "01-sizing.md")
FIG = os.path.join(ROOT, "docs", "design", "sizing")
os.makedirs(FIG, exist_ok=True)
STANCES = (STANCE_SPRAWL, STANCE_SAGITTAL)

# ----------------------------------------------------------------------------
# Derived actuator requirement.  Rating = worst joint across both stances.
# ----------------------------------------------------------------------------
def worst(case_rating: str) -> float:
    return max(max(stance_torques(LEG, s, c).values())
               for s in STANCES for c in LOAD_CASES if c.rating == case_rating)

REQ = {
    "continuous": worst("continuous"),
    "10 min": worst("10 min"),
    "peak": worst("peak"),
}
# Reach limit: the controller lets the foot go to this reach at full peak load
REACH_LIMIT = 300.0
REQ["peak_reach"] = max(joint_torques_r := [
    stance_torques(LEG, type(s)(s.name, s.hip_height, REACH_LIMIT, s.leg_plane, s.yaw_deg), c)["femur"]
    for s in STANCES for c in LOAD_CASES if c.rating == "peak"])

SPEED_REQ = {g.label: max(max(joint_speeds(LEG, s, g).values()) for s in STANCES) for g in GAITS}
JOINT_SPEED_NOMINAL = SPEED_REQ["Walk, tripod gait"]
JOINT_SPEED_FAST = SPEED_REQ["Fast walk, tripod gait (aspirational)"]

MOTOR_RPM_MAX = 5500.0          # PCB stator eddy losses and rotor tip speed cap this
RATIOS = (30, 40, 50, 60, 80)
MOTOR_1 = PcbMotor(stators=1)
MOTOR_2 = PcbMotor(stators=2)

# Nominal choice
RATIO = Reduction(50)
MOTOR = MOTOR_2
JOINT = joint_from_motor(MOTOR, RATIO)
JOINT_SPEED_AT_MAX_RPM = MOTOR_RPM_MAX * 2 * math.pi / 60 / RATIO.ratio

# Electrical
KV_RPM_V = MOTOR_RPM_MAX * 1.15 / MOTOR.bus_v            # leave 15 % for FOC headroom
KT = 60 / (2 * math.pi * KV_RPM_V)                       # N·m/A (q-axis, rms-equivalent)
I_CONT = MOTOR.torque_cont / KT
I_PEAK = MOTOR.torque_peak / KT

# Power
P_JOINT_CONT = REQ["continuous"] * joint_speeds(LEG, STANCE_SPRAWL, GAITS[1])["yaw_stance"]
P_JOINT_PEAK = REQ["peak"] * 5.0
P_WALK = ENERGY.walking_power(MASS.robot + MASS.mission_payload, 1.0)
P_RIDER = ENERGY.walking_power(MASS.robot + MASS.rider, 0.3)
PACK_WH = ENERGY.pack_wh(P_WALK)
PACK_KG = ENERGY.pack_kg(P_WALK)
PACK_AH = PACK_WH / MOTOR.bus_v
I_BUS_PEAK = 6 * 1000 / MOTOR.bus_v * 0.5   # 6 joints at ~1 kW, 50 % coincidence


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_leg_side():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    rider = [c for c in LOAD_CASES if c.rating == "10 min"][0]
    for ax, st in zip(axes, STANCES):
        r, z = st.foot_reach, -st.hip_height
        a1, k = ik(LEG, r, z)
        kx, kz = knee_pos(LEG, a1)
        ax.plot([-LEG.coxa, 0], [0, 0], "k-", lw=6, solid_capstyle="round", label="coxa")
        ax.plot([0, kx], [0, kz], "-", color="#c0392b", lw=6, solid_capstyle="round", label="femur")
        ax.plot([kx, r], [kz, z], "-", color="#2980b9", lw=6, solid_capstyle="round", label="tibia")
        ax.plot([-LEG.coxa - 60, 420], [z - 20, z - 20], "-", color="#7f8c8d", lw=2)
        ax.add_patch(plt.Rectangle((-LEG.coxa - BODY.width / 2, -12), BODY.width / 2, BODY.height, color="#bdc3c7"))
        ax.annotate("body", (-LEG.coxa - BODY.width / 4, BODY.height / 2), ha="center", fontsize=9, color="#555")
        for (x, y), name in (((0, 0), "femur"), ((kx, kz), "knee"), ((-LEG.coxa, 0), "yaw")):
            ax.plot(x, y, "o", ms=10, color="k")
        t = stance_torques(LEG, st, rider)
        ax.annotate(f'femur {t["femur"]:.0f} N·m', (0, 0), (10, 60), textcoords="offset points", fontsize=10)
        ax.annotate(f'knee {t["knee"]:.0f} N·m', (kx, kz), (10, -30), textcoords="offset points", fontsize=10)
        ax.annotate(f'yaw {t["yaw"]:.0f} N·m', (-LEG.coxa, 0), (-30, 60), textcoords="offset points", fontsize=10)
        ax.annotate(f'{rider.foot_force_z:.0f} N', (r, z), (12, 8), textcoords="offset points", fontsize=10)
        ax.arrow(r, z - 140, 0, 100, width=6, color="#27ae60", length_includes_head=True)
        ax.set_title(f"{st.name}\nhip {st.hip_height:.0f} mm, reach {st.foot_reach:.0f} mm, "
                     f"femur {math.degrees(a1):+.0f}°, knee {math.degrees(k):.0f}°", fontsize=10)
        ax.set_xlabel("r — outward in leg plane (mm)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("z (mm)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(f"Leg in its plane at the rider load case ({rider.foot_force_z:.0f} N per foot, quasi-static)")
    fig.tight_layout()
    p = os.path.join(FIG, "leg-stance.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_torque_map():
    rider = [c for c in LOAD_CASES if c.rating == "10 min"][0]
    rs = np.linspace(60, 380, 65)
    hs = np.linspace(200, 560, 73)
    fem = np.full((len(hs), len(rs)), np.nan)
    kne = np.full_like(fem, np.nan)
    for i, h in enumerate(hs):
        for j, r in enumerate(rs):
            try:
                a1, _ = ik(LEG, r, -h)
            except ValueError:
                continue
            kx, kz = knee_pos(LEG, a1)
            Fz = rider.foot_force_z
            fem[i, j] = abs(r * Fz) / 1000
            kne[i, j] = abs((r - kx) * Fz) / 1000
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, name in ((axes[0], fem, "femur"), (axes[1], kne, "knee")):
        cs = ax.contourf(rs, hs, data, levels=np.arange(0, 260, 20), cmap="viridis")
        ax.contour(rs, hs, data, levels=[REQ["continuous"], REQ["10 min"], REQ["peak"]],
                   colors=["w", "orange", "r"], linewidths=1.5)
        for st, mk in zip(STANCES, ("o", "s")):
            ax.plot(st.foot_reach, st.hip_height, mk, ms=10, color="w", mec="k", label=st.name.split(" (")[0])
        ax.set_xlabel("foot reach r from femur axis (mm)")
        ax.set_ylabel("hip height above ground (mm)")
        ax.set_title(f"{name} torque, vertical load only ({rider.foot_force_z:.0f} N)")
        fig.colorbar(cs, ax=ax, label="N·m")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Quasi-static joint torque over the foot workspace at the rider load. "
                 f"Contours: {REQ['continuous']:.0f} (white) / {REQ['10 min']:.0f} (orange) / {REQ['peak']:.0f} (red) N·m",
                 fontsize=10)
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
    ax2.plot(ratios, [MOTOR_RPM_MAX * 2 * math.pi / 60 / r for r in ratios], color="#27ae60", lw=2, label=f"joint speed at {MOTOR_RPM_MAX:.0f} rpm")
    ax2.axhline(JOINT_SPEED_NOMINAL, color="#27ae60", alpha=0.35, lw=8)
    ax2.axhline(JOINT_SPEED_FAST, color="#27ae60", alpha=0.2, lw=8, ls=":")
    ax2.set_ylabel("joint speed (rad/s)")
    ax2.set_ylim(0, 30)
    ax1.axvline(RATIO.ratio, color="k", ls=":")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax1.set_title(f"PCB axial-flux motor Ø{2*MOTOR.r_out:.0f} mm + in-plane cycloid: torque and speed vs ratio\n"
                  "bands = requirement (continuous / peak torque, nominal / fast joint speed)", fontsize=10)
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


def write_doc(figs):
    m = MASS
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
         "width = yaw-axis spacing + one actuator diameter + two side rails; the hip stacks set the width"),
        ("Hip yaw axes", f"x = {', '.join(f'{x:+.0f}' for x in BODY.hip_x)} mm, y = ±{BODY.width/2:.0f} mm", "six vertical axes"),
        ("Coxa (yaw axis → femur axis)", f"{LEG.coxa:.0f} mm", "just enough to clear the hip pod"),
        ("Femur", f"{LEG.femur:.0f} mm", ""),
        ("Tibia", f"{LEG.tibia:.0f} mm", "longer than the femur: ground clearance, near-vertical under load"),
        ("Leg reach (femur + tibia)", f"{LEG.reach:.0f} mm", ""),
        ("Actuator envelope", f"Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm, {ACT.mass:.1f} kg", "pancake; three stacked per hip on the yaw axis"),
        ("Hip stack height", f"{3*ACT.thickness + 2*ACT.stack_gap:.0f} mm", f"fits inside the {BODY.height:.0f} mm slab"),
    ]

    stance_rows = []
    for st in STANCES:
        a1, k = ik(LEG, st.foot_reach, -st.hip_height)
        t0 = tip_angles(BODY, LEG, st, False)
        stance_rows.append((st.name, f"{st.hip_height:.0f}", f"{st.foot_reach:.0f}", f"{math.degrees(a1):+.0f}° / {math.degrees(k):.0f}°",
                            f"{t0['stance_width_m']:.2f} × {t0['stance_length_m']:.2f}", f"{st.hip_height - 0:.0f}"))

    load_rows = [(c.name, c.rating, f"{c.total_mass:.0f}", c.legs_down, f"{c.dyn_factor:.1f}", f"{c.slope_deg:.0f}° + {c.accel:.1f} m/s²",
                  f"**{c.foot_force_z:.0f}**", f"{c.foot_force_prop:.0f}") for c in LOAD_CASES]

    torque_rows = []
    for st in STANCES:
        for c in LOAD_CASES:
            t = stance_torques(LEG, st, c)
            torque_rows.append((st.name.split(" (")[0], c.name, c.rating, f"{t['yaw']:.0f}", f"{t['femur']:.0f}", f"{t['knee']:.0f}"))

    gait_rows = [(g.label, f"{g.speed:.1f}", f"{g.stride*1000:.0f}", f"{g.duty:.2f}", f"{g.cycle_hz:.2f}",
                  f"{g.t_swing*1000:.0f}", f"{g.swing_speed_peak():.1f}") for g in GAITS]
    speed_rows = []
    for st in STANCES:
        for g in GAITS:
            s = joint_speeds(LEG, st, g)
            speed_rows.append((st.name.split(" (")[0], g.label, f"{s['yaw_swing']:.1f}", f"{s['yaw_stance']:.1f}",
                               f"{s['pitch_swing']:.1f}", f"{s['pitch_stance']:.1f}"))

    stab_rows = []
    for st in STANCES:
        for rider in (False, True):
            t = tip_angles(BODY, LEG, st, rider)
            stab_rows.append((st.name.split(" (")[0], "with rider" if rider else "unloaded", f"{t['com_m']:.2f}",
                              f"{t['roll_deg']:.0f}°", f"{t['pitch_deg']:.0f}°"))

    ratio_rows = []
    for r in RATIOS:
        red = Reduction(r)
        j1, j2 = joint_from_motor(MOTOR_1, red), joint_from_motor(MOTOR_2, red)
        v = MOTOR_RPM_MAX * 2 * math.pi / 60 / r
        ratio_rows.append((f"{r}:1", f"{j1['cont']:.0f} / {j1['peak']:.0f}", f"{j2['cont']:.0f} / {j2['peak']:.0f}",
                           f"{v:.1f}", f"{motor_speed_for(JOINT_SPEED_NOMINAL, red):.0f}", f"{motor_speed_for(JOINT_SPEED_FAST, red):.0f}"))

    spec_rows = [
        ("Joint torque, continuous (thermal steady state)", f"**{REQ['continuous']:.0f} N·m**", "walk, tripod, 1.5× dynamic factor, 30° slope — any joint, either stance"),
        ("Joint torque, 10-minute rating", f"**{REQ['10 min']:.0f} N·m**", "rider on board, wave gait, 15° slope"),
        ("Joint torque, peak (≤ 2 s)", f"**{REQ['peak']:.0f} N·m**", "stumble onto two legs at 3× dynamic factor, nominal stance"),
        ("Joint torque, peak at extended reach", f"{REQ['peak_reach']:.0f} N·m", f"same case with the foot out at {REACH_LIMIT:.0f} mm; software-limited region"),
        ("Joint speed, loaded (stance)", f"≥ {max(joint_speeds(LEG, s, GAITS[1])['pitch_stance'] for s in STANCES):.0f} rad/s at continuous torque", "1 m/s walk"),
        ("Joint speed, unloaded (swing)", f"≥ **{JOINT_SPEED_NOMINAL:.0f} rad/s**", "1 m/s walk, 0.4 m stride; 2 m/s needs " f"{JOINT_SPEED_FAST:.0f} rad/s"),
        ("Joint range", "yaw ±60°, femur −90…+60°, knee 20…160°", "fold-flat for transport, step-over 300 mm obstacle"),
        ("Backdrivability / sensing", "joint-torque estimate ≤ 10 % error, foot contact detection", "high ratio ⇒ cannot rely on motor-current transparency; needs output-side sensing or SEA"),
        ("Envelope", f"Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm", "pancake, axis = hip yaw axis"),
        ("Mass", f"≤ {ACT.mass:.1f} kg", ""),
        ("Count", "18, identical", "yaw joints run at ⅓–½ of pitch torque; commonality wins at this stage"),
        ("Bus voltage", f"{MOTOR.bus_v:.0f} V", "13S Li-ion; 48 V drivers and connectors are commodity"),
        ("Continuous mechanical power", f"~{P_JOINT_CONT:.0f} W", "continuous torque × stance rate"),
        ("Peak mechanical power", f"~{P_JOINT_PEAK/1000:.1f} kW", "peak torque × 5 rad/s"),
        ("Environment", "IP54 body, IP65 legs, −10…+45 °C", "outdoor, mud, rain; heat rejection at 45 °C ambient sizes the stator copper"),
    ]

    doc = f"""# 01 — Top-level sizing: shape, loads, and what the actuator has to be

*Generated by `analysis/sizing.py` from `analysis/hexapod_model.py`. Change the
parameters, re-run, and every number and figure here moves together.*

This is the first design pass: it fixes the rough size and shape of the
hexapod, the loads that shape produces at the joints, and therefore the
torque, speed, reduction ratio and power an actuator must deliver. It stops
at the actuator requirement. Transmission (motors in the body, power to the
joints), electronics and software are downstream and are not decided here.

Two body stances are carried through side by side — **A, sprawl** and **B,
under-body** — because the choice between them changes which joints do the
work, and the [vision document](vision.md) puts both in front of you as
rendered skeletons. The actuator requirement below is the envelope of both,
so the actuator stage can start before the stance is settled.

## 1. What it is sized for

| Mission | What it drives |
|---|---|
| Complex outdoor terrain (first demo) | 300 mm step-over, 30° slope, foot workspace, joint speed, weather sealing |
| Trash pickup (second demo) | 8 kg mission payload with a tool on the top deck; a free leg or a deck arm — not decided |
| Carry an adult (if needed, slow) | 100 kg on the top deck at a wave-gait crawl: the 10-minute torque rating and roll stability |
| Large-dog size | ~{BODY.length/1000:.1f} m body, hips {STANCE_SPRAWL.hip_height/1000:.2f}–{STANCE_SAGITTAL.hip_height/1000:.2f} m above ground, top deck at ~{(STANCE_SPRAWL.hip_height + BODY.height)/1000:.2f} m |

The rider case is what makes this a serious machine rather than a large
hobby hexapod: it roughly triples the vertical load per foot. It is treated as a
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

Putting all eighteen motors in the body, as pancakes stacked three-high on
each hip's yaw axis, is what sets the body's proportions: the hip stack is
{3*ACT.thickness + 2*ACT.stack_gap:.0f} mm tall and Ø{ACT.od:.0f} mm, so the body is a
{BODY.height:.0f} mm slab whose width is set by the stacks, not by the batteries.
The batteries, compute and payload bay fill the space between the six
stacks. The flat top is the solar skin and the payload/rider deck.

{md_table(("Parameter", "Value", "Why"), geom_rows)}

### The two stances

| Stance | Hip height (mm) | Foot reach r (mm) | Femur / knee angle | Footprint W × L (m) | Ground clearance under slab (mm) |
|---|---|---|---|---|---|
{chr(10).join('| ' + ' | '.join(r) + ' |' for r in stance_rows)}

* **A — Sprawl.** Leg planes radiate from the body (front and rear legs
  yawed ±30°). The yaw joints do the propulsion, the pitch joints carry the
  weight with a {STANCE_SPRAWL.foot_reach:.0f} mm lever arm. Wide, stable, insect-like; the widest
  thing you have to get through a door.
* **B — Under-body.** Leg planes are fore-aft, feet under the hips. The pitch
  joints do both carrying and pushing; yaw only steers and crabs. Narrow,
  taller, mammal-like; more ground clearance, less roll margin.

![Leg in its plane at the rider load]({rel(figs['leg'])})

The leg proportions (tibia longer than femur, knee-up) put the tibia nearly
vertical under load in both stances, which is why the knee torque is small at
the nominal posture and the femur carries almost everything. That is a choice:
it concentrates the hard duty in one joint per leg, and the torque map in §3
shows how quickly the knee picks up load when the foot is pulled inward.

## 3. Loads and joint torques

Quasi-static: a vertical ground reaction per supporting foot, plus a
propulsion force to hold a grade and accelerate. Dynamic factors stand in for
impact and gait dynamics until there is a simulation.

{md_table(("Load case", "Rating", "Mass on feet (kg)", "Legs down", "Dyn. factor", "Grade + accel", "Vertical / foot (N)", "Propulsion / foot (N)"), load_rows)}

Joint torques at each nominal stance (N·m):

{md_table(("Stance", "Load case", "Rating", "Yaw", "Femur", "Knee"), torque_rows)}

Two things to notice. First, the under-body stance does *not* lower the femur
torque, even though its lever arm is shorter, because the propulsion force now
acts through the whole {STANCE_SAGITTAL.hip_height:.0f} mm leg height. Second, the yaw joint sees at
most half of the femur duty in either stance — the case for a second, smaller
actuator size exists, but not yet.

![Torque map over the foot workspace]({rel(figs['map'])})

The map is vertical load only. The femur torque is simply the foot reach
times the load; the knee torque is small along the diagonal where the tibia
is vertical and grows either side of it. The controller keeps the foot inside
the orange contour under a rider and inside the red one otherwise.

## 4. Speed

Stride and duty factor fix the cycle, and the cycle fixes the swing time —
the swing leg has to cover the stride plus its lift in the time the stance
legs take to walk the body forward.

{md_table(("Gait", "Body speed (m/s)", "Stride (mm)", "Duty", "Cycle (Hz)", "Swing time (ms)", "Peak foot speed in swing (m/s)"), gait_rows)}

Peak joint rates (rad/s), swing (unloaded) and stance (under load):

{md_table(("Stance", "Gait", "Yaw, swing", "Yaw, stance", "Pitch, swing", "Pitch, stance"), speed_rows)}

The **1 m/s walk needs ~{JOINT_SPEED_NOMINAL:.0f} rad/s** unloaded at whichever joint sweeps the
leg. The 2 m/s aspiration needs ~{JOINT_SPEED_FAST:.0f} rad/s and, as §7 shows, is where the
reduction ratio and the motor's speed limit start fighting each other.

## 5. Stability

Static tip-over angles on the tripod support polygon, the narrowest polygon
the gaits use. With a rider the wave gait keeps five feet down and the
polygon is much larger; the tripod numbers are the floor.

{md_table(("Stance", "Load", "CoM height (m)", "Roll tip angle", "Pitch tip angle"), stab_rows)}

Roll is the limit. The under-body stance with a rider is at {tip_angles(BODY, LEG, STANCE_SAGITTAL, True)['roll_deg']:.0f}° on a
tripod — it must crawl with a wave gait, and it will have to widen its feet
(the yaw joints allow it) on a side slope. The sprawl stance keeps a
{tip_angles(BODY, LEG, STANCE_SPRAWL, True)['roll_deg']:.0f}° margin even on a tripod.

## 6. The actuator requirement

Envelope of both stances and all load cases. This is the input to the
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

* **A single PCB stator at 50:1 misses the continuous rating by a factor of
  ~{REQ['continuous']/joint_from_motor(MOTOR_1, RATIO)['cont']:.1f}.** It only closes at 80:1 or above, where the joint's top speed at
  {MOTOR_RPM_MAX:.0f} rpm drops to {MOTOR_RPM_MAX*2*math.pi/60/80:.1f} rad/s — below the 1 m/s walk.
* **Two stators (three rotors) at {RATIO.ratio:.0f}:1** gives {JOINT['cont']:.0f} N·m continuous and
  {JOINT['peak']:.0f} N·m peak with {JOINT_SPEED_AT_MAX_RPM:.1f} rad/s at the joint. That meets the walk and
  the peak, meets the rider rating, and leaves the 2 m/s aspiration short.
  This is the **nominal actuator architecture**: Ø{2*MOTOR.r_out:.0f} mm, two stator
  PCBs, {RATIO.ratio:.0f}:1 in-plane cycloid, {MOTOR.bus_v:.0f} V.
* The motor's speed ceiling is set by the PCB, not the rotor: at {MOTOR_RPM_MAX:.0f} rpm
  with ~12 pole pairs the electrical frequency is ~{MOTOR_RPM_MAX/60*12:.0f} Hz, and eddy
  losses in wide PCB traces grow with the square of that. Trace segmentation
  or higher-layer-count thin traces are the levers; the stator design will
  have to check this.
* Electrical: Kv ≈ {KV_RPM_V:.0f} rpm/V (Kt ≈ {KT:.3f} N·m/A) at {MOTOR.bus_v:.0f} V ⇒ {I_CONT:.0f} A continuous and
  {I_PEAK:.0f} A for the 2 s peak, per motor. That is a 48 V FOC stage rated ~40 A
  continuous / ~120 A peak per joint — half-bridge MOSFETs at that rating are
  commodity; the bus capacitance and the connectors for eighteen of them are
  the real layout problem. A lower Kv trades peak current for top speed.

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
  {RATIO.ratio:.0f}:1 is {RATIO.ratio**2:,}× the rotor's own. Foot-contact detection and
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

1. Robot ~{MASS.robot:.0f} kg, {BODY.length/1000:.1f} × {BODY.slab_width(ACT)/1000:.2f} × {BODY.height/1000:.1f} m slab body, legs {LEG.coxa:.0f} / {LEG.femur:.0f} / {LEG.tibia:.0f} mm, hips {STANCE_SPRAWL.hip_height/1000:.2f}–{STANCE_SAGITTAL.hip_height/1000:.2f} m up.
2. Eighteen identical pancake actuators, three per hip stacked on the yaw axis, in the body.
3. Actuator requirement: {REQ['continuous']:.0f} N·m continuous, {REQ['10 min']:.0f} N·m for 10 min, {REQ['peak']:.0f} N·m peak, {JOINT_SPEED_NOMINAL:.0f} rad/s, Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm, ≤ {ACT.mass:.1f} kg.
4. Nominal actuator architecture: Ø{2*MOTOR.r_out:.0f} mm two-stator PCB axial-flux motor, {RATIO.ratio:.0f}:1 in-plane cycloid, {MOTOR.bus_v:.0f} V.
5. Two {PACK_WH:.0f} Wh hot-swap packs.

Open, in order of how much they change the actuator:

* **Which stance** (A or B) — see the vision document. It decides whether yaw or pitch is the propulsion joint.
* **Whether the rider case stays.** Dropping it halves the 10-minute rating and the gearbox pin loads, and would let a single-stator motor work at 50:1.
* **σ for the PCB stator.** The whole motor concept rests on ~{MOTOR.sigma_cont/1000:.1f} kPa continuous. The first actuator-stage task is a single-stator test motor on a dyno.
* **Transmission from the body to the femur and knee.** Not sized here; it adds losses (budget 10 %) and reflected inertia, and it decides whether the three pancakes really can be coaxial.
* **Dynamic simulation** to replace the 1.5× / 3× factors with numbers.
"""
    with open(DOC, "w") as f:
        f.write(doc)


if __name__ == "__main__":
    figs = {"leg": fig_leg_side(), "map": fig_torque_map(), "ratio": fig_ratio_trade()}
    write_doc(figs)
    print(f"wrote {DOC}")
    print("requirement:", {k: round(v) for k, v in REQ.items()})
    print("speeds:", {k: round(v, 1) for k, v in SPEED_REQ.items()})
    print("motor 2-stator @50:1:", {k: round(v) for k, v in JOINT.items()}, "joint rad/s", round(JOINT_SPEED_AT_MAX_RPM, 1))
    print("power W walk/rider:", round(P_WALK), round(P_RIDER), "pack Wh/kg", round(PACK_WH), round(PACK_KG, 1))

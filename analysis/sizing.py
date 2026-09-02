#!/opt/hw-py/bin/python
"""Generate docs/design/01-sizing.md and its figures from analysis/hexapod_model.py
and analysis/leg3d.py.

    /opt/hw-py/bin/python analysis/sizing.py

Every number in the document comes from the models; iterate on the
parameters there, not on the prose here.
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
from hexapod_model import (ACT, BODY, ENERGY, GAITS, LOAD_CASES, MASS, STANCE, YAW_RANGE_DEG, G, PcbMotor,
                           Reduction, ik, joint_from_motor, joint_speeds, knee_pos,
                           motor_speed_for, stance_torques, tip_angles)
from leg3d import CHOSEN, MAMMAL_MODE, ROUTINE, STUMBLE, Workspace, evaluate, force_set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "design", "01-sizing.md")
FIG = os.path.join(ROOT, "docs", "design", "sizing")
os.makedirs(FIG, exist_ok=True)
CASE = {c.rating: c for c in LOAD_CASES}
WALK, PEAK, RIDER = CASE["continuous"], CASE["peak"], CASE["stretch"]
ST = STANCE
LEG = ST.leg
DOFS = CHOSEN.joint_names          # ('yaw', 'femur', 'knee')

# ----------------------------------------------------------------------------
# Per-DOF requirement over the working volume (3-D model)
#   continuous: walking load anywhere in the routine volume
#   peak:       stumble load in the routine volume, or walking load anywhere in
#               the extreme (step-up) box
#   stretch:    the rider at the neutral stance, reported only
# ----------------------------------------------------------------------------
STEP_H = 300.0
STEP_R_MIN = math.sqrt(max((LEG.tibia - LEG.femur) ** 2 - (ST.hip_height - STEP_H) ** 2, 0))   # closest foot for that height
STEP_DY = STEP_R_MIN - ST.foot_reach
STEP = Workspace(dx=(-200.0, 200.0), dy=(STEP_DY + 2, STEP_DY + 60), dz=(STEP_H, STEP_H), n=(9, 3, 1))
EV_ROUTINE = evaluate(CHOSEN, WALK.foot_force_z, WALK.foot_force_prop, ROUTINE)
EV_EXTREME = evaluate(CHOSEN, WALK.foot_force_z, WALK.foot_force_prop, STEP)
EV_PEAK = evaluate(CHOSEN, PEAK.foot_force_z, PEAK.foot_force_prop, STUMBLE)
# the same leg reconfigured into the mammal stance must also walk (review decision)
EV_MAMMAL = evaluate(MAMMAL_MODE, WALK.foot_force_z, WALK.foot_force_prop, ROUTINE)
EV_MAMMAL_PEAK = evaluate(MAMMAL_MODE, PEAK.foot_force_z, PEAK.foot_force_prop, STUMBLE)
EV_RIDER = np.max([CHOSEN.torques((0, 0, 0), F) for F in force_set(RIDER.foot_force_z, RIDER.foot_force_prop)], axis=0)
DOF_CONT_SPRAWL = dict(zip(DOFS, EV_ROUTINE["max"]))
DOF_CONT_MAMMAL = dict(zip(DOFS, EV_MAMMAL["max"]))
DOF_CONT = {d: max(DOF_CONT_SPRAWL[d], DOF_CONT_MAMMAL[d]) for d in DOFS}
DOF_PEAK = {d: max(a, b, c) for d, a, b, c in zip(DOFS, EV_EXTREME["max"], EV_PEAK["max"], EV_MAMMAL_PEAK["max"])}
DOF_NEUTRAL = dict(zip(DOFS, EV_ROUTINE["neutral"]))
DOF_RIDER = dict(zip(DOFS, EV_RIDER))
REQ = {"continuous": max(DOF_CONT.values()), "peak": max(DOF_PEAK.values())}
COXA_MOMENT = PEAK.foot_force_z * ST.foot_radius() / 1000

# Speeds
SPEED_REQ = {g.label: max(joint_speeds(ST, g).values()) for g in GAITS}
JOINT_SPEED_NOMINAL = SPEED_REQ["Walk, tripod gait"]
JOINT_SPEED_FAST = SPEED_REQ["Fast walk, tripod gait (aspirational)"]
_js = joint_speeds(ST, GAITS[1])
DOF_SWING = {"yaw": _js["yaw_swing"], "femur": _js["pitch_swing"], "knee": _js["pitch_swing"]}
DOF_STANCE_RATE = {"yaw": _js["yaw_stance"], "femur": _js["pitch_stance"], "knee": _js["pitch_stance"]}

# Motor and per-DOF ratio
MOTOR_RPM_MAX = 5500.0          # PCB stator eddy losses and rotor tip speed cap this
RATIOS = (30, 40, 50, 60, 80)
MOTOR_1 = PcbMotor(stators=1)
MOTOR_2 = PcbMotor(stators=2)


def joint_speed_at(ratio: float) -> float:
    return MOTOR_RPM_MAX * 2 * math.pi / 60 / ratio


MAX_RATIO = 80                  # single-stage in-plane cycloid: lobe pitch gets silly above this
MIN_RATIO = {"yaw": 30, "femur": 55, "knee": 60}   # the disc family chosen in 08-actuator-design (yaw 30:1 for swing speed); torque margin comes out as margin


def ratio_for(m: PcbMotor, cont: float, peak: float) -> float:
    """Smallest 5:1 step that meets both ratings through the cycloid."""
    eff = Reduction(1).efficiency
    need = max(cont / (m.torque_cont * eff), peak / (m.torque_peak * eff))
    return 5 * math.ceil(need / 5)


DOF_PLAN = {}
for d in DOFS:
    for m, label in ((MOTOR_1, "1 stator"), (MOTOR_2, "2 stators")):
        r = max(ratio_for(m, DOF_CONT[d], DOF_PEAK[d]), MIN_RATIO[d])
        if r <= MAX_RATIO and joint_speed_at(r) >= DOF_SWING[d]:
            break
    j = joint_from_motor(m, Reduction(r))
    DOF_PLAN[d] = {"motor": m, "label": label, "ratio": r, "cont": j["cont"], "peak": j["peak"],
                   "speed": joint_speed_at(r), "ok": joint_speed_at(r) >= DOF_SWING[d]}

MOTOR = MOTOR_2 if any(p["motor"].stators == 2 for p in DOF_PLAN.values()) else MOTOR_1
KV_RPM_V = MOTOR_RPM_MAX * 1.15 / MOTOR.bus_v
KT = 60 / (2 * math.pi * KV_RPM_V)
I_CONT = MOTOR.torque_cont / KT
I_PEAK = MOTOR.torque_peak / KT
FEMUR = DOF_PLAN["femur"]

# Power
P_JOINT_CONT = DOF_CONT["femur"] * DOF_STANCE_RATE["femur"] + DOF_CONT["yaw"] * DOF_STANCE_RATE["yaw"]
P_JOINT_PEAK = REQ["peak"] * 5.0
P_WALK = ENERGY.walking_power(MASS.robot + MASS.mission_payload, 1.0)
P_RIDER = ENERGY.walking_power(MASS.robot + MASS.rider, 0.3)
PACK_WH = ENERGY.pack_wh(P_WALK)
PACK_KG = ENERGY.pack_kg(P_WALK)
PACK_AH = PACK_WH / MOTOR.bus_v
I_BUS_PEAK = 6 * P_JOINT_PEAK / MOTOR.bus_v * 0.5


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_leg_side():
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    leg = LEG
    r, z = ST.foot_reach, -ST.hip_height
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
    t = stance_torques(ST, WALK)
    ax.annotate(f'femur {t["femur"]:.0f} N·m', (0, 0), (-10, -34), textcoords="offset points", fontsize=10, ha="right")
    ax.annotate(f'knee {t["knee"]:.0f} N·m', (kx, kz), (10, 6), textcoords="offset points", fontsize=10)
    ax.annotate(f'yaw {t["yaw"]:.0f} N·m', (-leg.coxa, 0), (-10, 14), textcoords="offset points", fontsize=10, ha="right")
    ax.annotate(f'{WALK.foot_force_z:.0f} N', (r, z), (12, 8), textcoords="offset points", fontsize=10)
    ax.arrow(r, z - 150, 0, 110, width=6, color="#27ae60", length_includes_head=True)
    ax.annotate("", (r, -60), (0, -60), arrowprops=dict(arrowstyle="<->", color="#c0392b"))
    ax.annotate(f"arm {r:.0f} mm", (r / 2, -60), (0, -14), textcoords="offset points", ha="center", fontsize=9, color="#c0392b")
    ax.set_title(f"coxa {leg.coxa:.0f} / femur {leg.femur:.0f} / tibia {leg.tibia:.0f} mm\n"
                 f"femur {ST.femur_deg:+.0f}°, knee {math.degrees(k):.0f}°, hip {ST.hip_height:.0f} mm up, knee {ST.knee_height:.0f} mm up", fontsize=10)
    ax.set_xlabel("r — outward in leg plane (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_aspect("equal")
    ax.set_xlim(-leg.coxa - BODY.width / 2 - 40, 440)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.suptitle(f"The agreed leg at the walking load ({WALK.foot_force_z:.0f} N per foot, quasi-static)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIG, "leg-stance.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_torque_map():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    leg = LEG
    rs = np.linspace(20, 460, 89)
    hs = np.linspace(60, leg.reach - 40, 81)
    fem = np.full((len(hs), len(rs)), np.nan)
    kne = np.full_like(fem, np.nan)
    Fz = WALK.foot_force_z
    for i, h in enumerate(hs):
        for j, r in enumerate(rs):
            try:
                a1, _ = ik(leg, r, -h)
            except ValueError:
                continue
            kx, kz = knee_pos(leg, a1)
            fem[i, j] = abs(r * Fz) / 1000
            kne[i, j] = abs((r - kx) * Fz) / 1000
    step_r = math.sqrt(max((leg.tibia - leg.femur) ** 2 - (ST.hip_height - 300) ** 2, 0))
    for ax, data, name in ((axes[0], fem, "femur"), (axes[1], kne, "knee")):
        cs = ax.contourf(rs, hs, data, levels=np.arange(0, 200, 10), cmap="viridis")
        ax.contour(rs, hs, data, levels=[DOF_CONT[name]], colors=["w"], linewidths=1.5)
        ax.plot(ST.foot_reach, ST.hip_height, "o", ms=10, color="w", mec="k", label="neutral stance")
        ax.plot(step_r, ST.hip_height - 300, "s", ms=9, color="orange", mec="k", label="standing on a 300 mm step")
        ax.set_xlabel("foot reach r from femur axis (mm)")
        ax.set_ylabel("femur axis above the foothold (mm)")
        ax.set_title(f"{name} torque, vertical walking load only ({Fz:.0f} N)", fontsize=10)
        fig.colorbar(cs, ax=ax, label="N·m")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Quasi-static joint torque over the reachable foot workspace. White contour: the DOF's continuous rating "
                 f"({DOF_CONT['femur']:.0f} / {DOF_CONT['knee']:.0f} N·m). Blank: closer than the {leg.tibia - leg.femur:.0f} mm minimum extension.", fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG, "torque-map.png")
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def fig_ratio_trade():
    ratios = np.linspace(20, 100, 81)
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    for m, ls, lab in ((MOTOR_1, "--", "1 stator"), (MOTOR_2, "-", "2 stators")):
        ax1.plot(ratios, [joint_from_motor(m, Reduction(r))["cont"] for r in ratios], ls, color="#2980b9", label=f"continuous, {lab}")
        ax1.plot(ratios, [joint_from_motor(m, Reduction(r))["peak"] for r in ratios], ls, color="#c0392b", label=f"peak, {lab}")
    cols = {"yaw": "#3a3a3a", "femur": "#c0392b", "knee": "#2980b9"}
    for d in DOFS:
        ax1.axhline(DOF_CONT[d], color=cols[d], alpha=0.25, lw=7)
        ax1.text(21, DOF_CONT[d] + 4, f"{d} continuous {DOF_CONT[d]:.0f}", fontsize=8, color=cols[d])
        ax1.axvline(DOF_PLAN[d]["ratio"], color=cols[d], ls=":", lw=1.2)
        ax1.text(DOF_PLAN[d]["ratio"] + 0.6, 470 - 22 * DOFS.index(d), f"{d} {DOF_PLAN[d]['ratio']:.0f}:1", fontsize=8, color=cols[d])
    ax1.set_xlabel("reduction ratio")
    ax1.set_ylabel("joint torque (N·m)")
    ax1.set_ylim(0, 500)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ratios, [joint_speed_at(r) for r in ratios], color="#27ae60", lw=2, label=f"joint speed at {MOTOR_RPM_MAX:.0f} rpm")
    ax2.axhline(DOF_SWING["yaw"], color="#27ae60", alpha=0.35, lw=7)
    ax2.axhline(DOF_SWING["femur"], color="#27ae60", alpha=0.2, lw=7, ls=":")
    ax2.set_ylabel("joint speed (rad/s)")
    ax2.set_ylim(0, 30)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax1.set_title(f"One Ø{2*MOTOR.r_out:.0f} mm PCB axial-flux motor, three cycloid ratios: torque and speed vs ratio\n"
                  "bands = per-DOF continuous requirement and swing-speed need (yaw solid, pitch dotted); dotted verticals = chosen ratios", fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG, "ratio-trade.png")
    fig.savefig(p, dpi=120)
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
        ("18 × joint actuators", f"{m.actuators:.1f}", f"{ACT.mass:.1f} kg each, target (per-DOF sizing should bring the yaw and knee below this)"),
        ("Body structure", f"{m.body_structure:.1f}", "plates, hip pods, top deck"),
        ("6 × legs", f"{m.legs:.1f}", "links, feet, in-leg transmission share"),
        ("2 × hot-swap batteries", f"{m.batteries:.1f}", f"~{PACK_WH:.0f} Wh each, see §8"),
        ("Electronics", f"{m.electronics:.1f}", "compute, drivers, sensors, harness"),
        ("Payload interface", f"{m.payload_interface:.1f}", "deck rails, tool mount, solar skin"),
        ("Margin", f"{m.margin:.1f}", ""),
        ("**Robot, unloaded**", f"**{m.robot:.1f}**", ""),
        ("Mission payload", f"{m.mission_payload:.1f}", "trash + gripper"),
        ("Rider (stretch goal)", f"{m.rider:.1f}", "adult male; reported, not a design driver"),
    ]

    geom_rows = [
        ("Body slab, length × width × height", f"{BODY.length:.0f} × {BODY.slab_width(ACT):.0f} × {BODY.height:.0f} mm",
         "width = yaw-axis spacing + one actuator diameter + two side rails"),
        ("Hip yaw axes", f"x = {', '.join(f'{x:+.0f}' for x in BODY.hip_x)} mm, y = ±{BODY.width/2:.0f} mm", "six vertical axes, under the body"),
        ("Coxa / femur / tibia", f"{LEG.coxa:.0f} / {LEG.femur:.0f} / {LEG.tibia:.0f} mm", f"tibia {LEG.tibia_ratio:.1f}× femur, agreed in review"),
        ("Neutral stance", f"femur {ST.femur_deg:.0f}° up, tibia vertical", "a posture, adjustable at run time"),
        ("Hip height / knee height", f"{ST.hip_height:.0f} / {ST.knee_height:.0f} mm", "femur axis and knee above ground"),
        ("Femur arm / foot radius", f"{ST.foot_reach:.0f} / {ST.foot_radius():.0f} mm", "horizontal, femur axis → foot / yaw axis → foot"),
        ("Minimum leg extension", f"{LEG.tibia - LEG.femur:.0f} mm", "tibia − femur; the closest the foot can come to the femur axis"),
        ("Actuator envelope", f"Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm, ≤ {ACT.mass:.1f} kg", "pancake; three stacked per hip on the yaw axis"),
        ("Top deck height", f"{ST.hip_height + BODY.height:.0f} mm", "payload deck and solar skin"),
    ]

    load_rows = [(c.name, c.rating, f"{c.total_mass:.0f}", c.legs_down, f"{c.dyn_factor:.1f}", f"{c.slope_deg:.0f}° + {c.accel:.1f} m/s²",
                  f"**{c.foot_force_z:.0f}**", f"{c.foot_force_prop:.0f}") for c in LOAD_CASES]

    torque_rows = []
    for c in LOAD_CASES:
        t = stance_torques(ST, c)
        torque_rows.append((c.name, c.rating, f"{t['yaw']:.0f}", f"**{t['femur']:.0f}**", f"{t['knee']:.0f}"))

    gait_rows = [(g.label, f"{g.speed:.1f}", f"{g.stride*1000:.0f}", f"{g.duty:.2f}", f"{g.cycle_hz:.2f}",
                  f"{g.t_swing*1000:.0f}", f"{g.swing_speed_peak():.1f}") for g in GAITS]
    speed_rows = []
    for g in GAITS:
        s = joint_speeds(ST, g)
        speed_rows.append((g.label, f"{s['yaw_swing']:.1f}", f"{s['yaw_stance']:.1f}", f"{s['pitch_swing']:.1f}", f"{s['pitch_stance']:.1f}"))

    stab_rows = []
    for rider in (False, True):
        t = tip_angles(BODY, ST, rider)
        stab_rows.append(("with rider (stretch)" if rider else "unloaded", f"{t['com_m']:.2f}", f"{t['roll_deg']:.0f}°", f"{t['pitch_deg']:.0f}°",
                          f"{t['stance_width_m']:.2f} × {t['stance_length_m']:.2f}"))

    dof_rows = []
    for d in DOFS:
        p = DOF_PLAN[d]
        dof_rows.append((d, f"{DOF_NEUTRAL[d]:.0f}", f"{DOF_CONT_SPRAWL[d]:.0f}", f"{DOF_CONT_MAMMAL[d]:.0f}", f"**{DOF_CONT[d]:.0f}**", f"**{DOF_PEAK[d]:.0f}**", f"{DOF_RIDER[d]:.0f}",
                         f"{DOF_SWING[d]:.0f}", f"{p['label']}, **{p['ratio']:.0f}:1**", f"{p['cont']:.0f} / {p['peak']:.0f}",
                         f"{p['speed']:.1f}" + ("" if p["ok"] else " ✗")))

    ratio_rows = []
    for r in RATIOS:
        red = Reduction(r)
        j1, j2 = joint_from_motor(MOTOR_1, red), joint_from_motor(MOTOR_2, red)
        ratio_rows.append((f"{r}:1", f"{j1['cont']:.0f} / {j1['peak']:.0f}", f"{j2['cont']:.0f} / {j2['peak']:.0f}",
                           f"{joint_speed_at(r):.1f}", f"{motor_speed_for(JOINT_SPEED_NOMINAL, red):.0f}", f"{motor_speed_for(JOINT_SPEED_FAST, red):.0f}"))

    spec_rows = [
        ("Joint range", f"yaw ±{YAW_RANGE_DEG:.0f}°, femur −70…+90°, knee 20…160°", "yaw 90° puts the leg plane fore-aft for the mammal stance; fold-flat for transport; crouch to the deck"),
        ("Yaw bearing overturning moment (structural)", f"{COXA_MOMENT:.0f} N·m", "peak foot load × foot radius; carried by the coxa and the yaw bearing, not the motor"),
        ("Backdrivability / sensing", "joint-torque estimate ≤ 10 % error, foot contact detection", "high ratio ⇒ cannot rely on motor-current transparency; needs output-side sensing or SEA"),
        ("Envelope", f"Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm", "pancake, axis = hip yaw axis"),
        ("Mass", f"≤ {ACT.mass:.1f} kg (femur); less for yaw and knee", "one motor, three ratios"),
        ("Bus voltage", f"{MOTOR.bus_v:.0f} V", "13S Li-ion; 48 V drivers and connectors are commodity"),
        ("Continuous mechanical power, per leg", f"~{P_JOINT_CONT:.0f} W", "femur and yaw continuous torque × their stance rates"),
        ("Peak mechanical power, per joint", f"~{P_JOINT_PEAK/1000:.1f} kW", "peak torque × 5 rad/s"),
        ("Environment", "IP54 body, IP65 legs, −10…+45 °C", "outdoor, mud, rain; heat rejection at 45 °C ambient sizes the stator copper"),
    ]

    doc = f"""# 01 — Top-level sizing: shape, loads, and what each actuator has to be

*Generated by `analysis/sizing.py` from `analysis/hexapod_model.py` and
`analysis/leg3d.py`. Change the parameters, re-run, and every number and
figure here moves together.*

This is the first design pass: it fixes the rough size and shape of the
hexapod, the loads that shape produces at the joints, and therefore the
torque, speed, reduction ratio and power each joint's actuator must deliver.
It stops at the actuator requirement. Transmission (motors in the body,
power to the joints), electronics and software are downstream and are not
decided here. The leg topology choice is in [02-leg-topology.md](02-leg-topology.md)
and the wider architecture levers in [03-architecture-levers.md](03-architecture-levers.md).

## 1. What it is sized for

| Mission | What it drives |
|---|---|
| Complex outdoor terrain (first demo) | 300 mm step-up, 30° slope, foot working volume, joint speed, weather sealing |
| Trash pickup (second demo) | 8 kg mission payload with a tool on the top deck; a free leg or a deck arm — not decided |
| Carry an adult (stretch goal) | 100 kg on the deck at a wave-gait crawl; reported as margin, not a rating |
| Large-dog size | ~{BODY.length/1000:.1f} m body, hips {ST.hip_height/1000:.2f} m above ground, top deck at {(ST.hip_height + BODY.height)/1000:.2f} m |

### Mass budget

{md_table(("Item", "kg", "Basis"), mass_rows)}

Everything downstream scales with the unloaded mass, and the unloaded mass is
dominated by the eighteen actuators. **The actuator mass is the single most
important number in this budget**: at 1.6 kg each the robot is 56 kg and
every torque below grows by 15 %.

## 2. Shape

### The body

Putting all eighteen motors in the body, as pancakes stacked three-high on
each hip's yaw axis, is what sets the body's proportions: the hip stack is
{3*ACT.thickness + 2*ACT.stack_gap:.0f} mm tall and Ø{ACT.od:.0f} mm, so the body is a
{BODY.height:.0f} mm slab whose width is set by the stacks, not by the batteries.
The yaw axes sit under the body at ±{BODY.width/2:.0f} mm; the batteries, compute and
payload bay fill the space between the six stacks. The flat top is the solar
skin and the payload deck.

### The leg

Two stances, one leg. In the **sprawl** stance the leg planes radiate from
the body and the yaw joints sweep the stride. Yawed 90° the same leg stands
in a **mammal** stance: leg plane fore-aft, femur down and forward, foot
under the hip, hips {MAMMAL_MODE.hip_height:.0f} mm up instead of {ST.hip_height:.0f} mm and a {BODY.width/1000:.2f} m stance
instead of {(BODY.width + 2*ST.foot_radius())/1000:.2f} m. That is the tall, narrow mode for a doorway or a deep
obstacle field, and the yaw range and the ratings below are set so the
controller can move between the two at run time.

The joint torque from a vertical foot load is that load times the
**horizontal** distance from the joint axis to the foot — the link angles do
not enter. So the leg keeps the foot horizontally close to the femur axis
while the stance stays wide: a {LEG.coxa:.0f} mm **coxa** carries the leg out from
under the body (a rigid arm on the yaw output, so it takes the load as
bending into the yaw bearing, not as motor torque); the **femur** pitches up
and out from the end of the coxa at {ST.femur_deg:.0f}° in the neutral stance, so the
knee sits high and only `femur × cos(angle)` outboard; and the **tibia**,
{LEG.tibia_ratio:.1f}× the femur, comes straight down from the knee to the foot.

{md_table(("Parameter", "Value", "Why"), geom_rows)}

![The agreed leg at the walking load]({rel(figs['leg'])})

The price of the long tibia shows in the last geometry row: the knee cannot
fold the foot closer than {LEG.tibia - LEG.femur:.0f} mm to the femur axis, so a high step is
taken with the foot moved outward. §6 rates the actuators over the working
volume, so that cost is in the numbers; the topology document quantifies it
against other proportions.

## 3. Loads and joint torques at the neutral stance

Quasi-static: a vertical ground reaction per supporting foot, plus a
propulsion force to hold a slope and accelerate. Dynamic factors stand in for
impact and gait dynamics until there is a simulation.

{md_table(("Load case", "Rating", "Mass on feet (kg)", "Legs down", "Dyn. factor", "Slope + accel", "Vertical / foot (N)", "Propulsion / foot (N)"), load_rows)}

Joint torques at the neutral stance (N·m). The femur torque is `vertical
load × femur arm`; the knee's comes from the in-plane share of the
propulsion force acting over the tibia's height (front and rear legs are
yawed 30°, so half their push is in the leg plane); the yaw joint sees only
propulsion:

{md_table(("Load case", "Rating", "Yaw", "Femur", "Knee"), torque_rows)}

![Torque map over the foot workspace]({rel(figs['map'])})

The map is the walking load only, over the reachable workspace. Femur torque
is reach × load, so the controller's job is to keep the foot near the femur
axis. The knee is unloaded along the curve where the tibia is vertical and
picks up torque either side of it. The orange square is the foot standing on
a 300 mm step: the leg cannot fold enough to keep the foot where it was, so
it is out at {math.sqrt(max((LEG.tibia - LEG.femur)**2 - (ST.hip_height - 300)**2, 0)):.0f} mm and the femur pays.

## 4. Speed

Stride and duty factor fix the cycle, and the cycle fixes the swing time —
the swing leg has to cover the stride plus its lift in the time the stance
legs take to walk the body forward.

{md_table(("Gait", "Body speed (m/s)", "Stride (mm)", "Duty", "Cycle (Hz)", "Swing time (ms)", "Peak foot speed in swing (m/s)"), gait_rows)}

Peak joint rates (rad/s), swing (unloaded) and stance (under load). With
radial leg planes the stride is a yaw sweep, so yaw is the fast joint:

{md_table(("Gait", "Yaw, swing", "Yaw, stance", "Pitch, swing", "Pitch, stance"), speed_rows)}

The **1 m/s walk needs ~{JOINT_SPEED_NOMINAL:.0f} rad/s** unloaded at the yaw joint and
~{DOF_SWING['femur']:.0f} rad/s at the pitch joints. The 2 m/s aspiration needs ~{JOINT_SPEED_FAST:.0f} rad/s
at the yaw joint.

## 5. Stability

Static tip-over angles on the tripod support polygon, the narrowest polygon
the gaits use.

{md_table(("Load", "CoM height (m)", "Roll tip angle", "Pitch tip angle", "Footprint W × L (m)"), stab_rows)}

Roll is the limit and the coxa is what buys it: {LEG.coxa:.0f} mm of coxa is
{2*LEG.coxa/1000:.2f} m of stance width at no torque cost.

## 6. The actuator requirement, per DOF

Each joint is rated on its own. The 3-D leg model puts the foot everywhere in
two working volumes with the full force (vertical, propulsion and 30 % of
that sideways) and records the worst moment about each joint's axis:

* **continuous** — the walking load anywhere in the *routine* volume
  (±200 mm fore-aft, ±100 mm lateral, 0–150 mm up from the neutral foot);
* **peak** — the stumble load with the feet near their nominal ring (±200 mm
  fore-aft, ±50 mm lateral, up to 50 mm up), or the walking load standing on
  a {STEP_H:.0f} mm step with the foot as close in as the leg can fold
  ({STEP_R_MIN:.0f}–{STEP_R_MIN + 60:.0f} mm from the femur axis, ±200 mm fore-aft);
* **mammal stance** — the same leg yawed 90° with the femur down and forward
  (hips {MAMMAL_MODE.hip_height:.0f} mm up, feet under the hips), walking load over its own
  routine volume. The review asked for this reconfiguration to be designed
  in, so it is part of the continuous envelope;
* **rider** — the stretch case at the neutral stance, for margin only.

{md_table(("DOF", "Neutral, walk (N·m)", "Sprawl routine (N·m)", "Mammal-stance routine (N·m)", "Continuous (N·m)", "Peak (N·m)", "Rider, neutral (N·m)", "Swing speed (rad/s)", "Motor, ratio", "Gives cont / peak (N·m)", f"Joint rad/s at {MOTOR_RPM_MAX:.0f} rpm"), dof_rows)}

Two things the neutral-stance numbers hid. The **knee** is not the light
joint it looks: whenever the foot is raised the long tibia has to fold, the
knee swings out the far side of the femur axis, and the {LEG.tibia:.0f} mm tibia
becomes the lever. At the corners of the routine volume and on a {STEP_H:.0f} mm step
the knee's torque is above the femur's. And the **yaw** joint, which never
sees weight, still needs {DOF_CONT['yaw']:.0f} N·m for propulsion on a 30° slope over a
{ST.foot_radius():.0f} mm foot radius — half the femur, and it is the fast joint.

The remaining requirements common to all three:

{md_table(("Quantity", "Requirement", "Driving case"), spec_rows)}

## 7. Motor and reduction: one PCB axial-flux motor, three ratios

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
uses **σ = {MOTOR.sigma_cont/1000:.1f} kPa continuous and {MOTOR.sigma_peak/1000:.1f} kPa for 2 s bursts** on the
full annulus, back-fitted to the motor study's A3 design point
([07](07-motor-options-in-envelope.md), [08](08-actuator-design.md)); the
first sizing round assumed 1.5 / 4.5 kPa. Numbers to be earned on a dyno,
not assumed for long. That gives:

| Motor | Continuous (N·m) | Peak (N·m) |
|---|---|---|
| 1 stator, 2 rotors | {MOTOR_1.torque_cont:.2f} | {MOTOR_1.torque_peak:.2f} |
| 2 stators, 3 rotors | {MOTOR_2.torque_cont:.2f} | {MOTOR_2.torque_peak:.2f} |

Through a cycloid at {Reduction(1).efficiency*100:.0f} % efficiency:

{md_table(("Ratio", "1 stator: cont / peak (N·m)", "2 stators: cont / peak (N·m)", f"Joint rad/s at {MOTOR_RPM_MAX:.0f} rpm", "Motor rpm for 1 m/s walk", "Motor rpm for 2 m/s"), ratio_rows)}

![Torque and speed vs reduction ratio]({rel(figs['ratio'])})

The per-DOF table in §6 picks, for each joint, the smallest 5:1 step that
meets both its ratings, capped at {MAX_RATIO}:1 for a single-stage cycloid, with
one stator if that leaves enough speed for the joint's swing and two
stators otherwise. The result is **one stator and rotor design, stacked
once or twice, with three cycloid discs**: yaw {DOF_PLAN['yaw']['label']} at {DOF_PLAN['yaw']['ratio']:.0f}:1,
femur {DOF_PLAN['femur']['label']} at {DOF_PLAN['femur']['ratio']:.0f}:1, knee {DOF_PLAN['knee']['label']} at {DOF_PLAN['knee']['ratio']:.0f}:1. The femur gives
{FEMUR['cont']:.0f} N·m continuous and {FEMUR['peak']:.0f} N·m peak with {FEMUR['speed']:.1f} rad/s at the joint. A
two-stator unit is the same PCB twice and one more rotor, so the part count
stays at one stator, one rotor, one driver and three discs.

* The motor's speed ceiling is set by the PCB, not the rotor: at {MOTOR_RPM_MAX:.0f} rpm
  with ~12 pole pairs the electrical frequency is ~{MOTOR_RPM_MAX/60*12:.0f} Hz, and eddy
  losses in wide PCB traces grow with the square of that. Trace segmentation
  or higher-layer-count thin traces are the levers; the stator design will
  have to check this.
* Electrical: Kv ≈ {KV_RPM_V:.0f} rpm/V (Kt ≈ {KT:.3f} N·m/A) at {MOTOR.bus_v:.0f} V ⇒ {I_CONT:.0f} A continuous and
  {I_PEAK:.0f} A for the 2 s peak, per motor. Half-bridge MOSFETs at that rating
  are commodity; the bus capacitance and the connectors for eighteen of them
  are the real layout problem. A lower Kv trades peak current for top speed.
* A single-stage cycloid ratio equals its lobe count. {FEMUR['ratio']:.0f} lobes on a disc that
  fits inside a Ø{2*MOTOR.r_in:.0f} mm bore means a lobe pitch of ~{2*math.pi*(MOTOR.r_in-8)/FEMUR['ratio']:.1f} mm and pins of
  ~3 mm: fine for a machined steel disc and hardened pins, marginal for
  printed or aluminium parts at {REQ['peak']:.0f} N·m. The output torque reacts through
  those pins at ~{MOTOR.r_in-8:.0f} mm radius, so the net pin load at peak is ~{REQ['peak']/((MOTOR.r_in-8)/1000)/1000:.1f} kN
  over roughly half the pins.
* High ratio means low transparency: reflected motor inertia through
  {FEMUR['ratio']:.0f}:1 is {FEMUR['ratio']**2:,.0f}× the rotor's own. Foot-contact detection and torque
  control cannot come from motor current alone; plan on an output-side
  torque sensor, a series-elastic element, or foot force sensing.

**Off-the-shelf benchmark.** Actuators in the ~120 N·m peak, ~60–70:1, sub-1 kg
class (the CubeMars AK80-64 and MyActuator RMD-X8-Pro families are the usual
suspects; verify numbers against their datasheets before relying on them)
cover the femur and knee requirement at roughly $400–600 each, and a smaller
family covers the yaw. That is the bar the custom actuator has to beat on
cost, on the pancake form factor that lets it stack on the yaw axis, or on
serviceability — and the first prototype leg should probably be built on
them so that the transmission and the software are not waiting on the motor.

## 8. Power and energy

| Quantity | Value | Basis |
|---|---|---|
| Average electrical power, 1 m/s walk | {P_WALK:.0f} W | cost of transport {ENERGY.cost_of_transport:.1f} × m·g·v + {ENERGY.hotel_w:.0f} W hotel |
| Average electrical power, rider at 0.3 m/s (stretch) | {P_RIDER:.0f} W | same model |
| Endurance target | {ENERGY.endurance_h:.0f} h walking | mission length |
| Battery, each of {ENERGY.packs} | **{PACK_WH:.0f} Wh, {PACK_AH:.0f} Ah at {MOTOR.bus_v:.0f} V, ~{PACK_KG:.1f} kg** | {ENERGY.pack_wh_per_kg:.0f} Wh/kg pack level |
| Peak bus current | ~{I_BUS_PEAK:.0f} A | six joints near peak power, 50 % coincidence |
| Hot-swap | either pack alone must carry the full peak | a pack coming out must not brown out the drivers |

A cost of transport of {ENERGY.cost_of_transport:.1f} is pessimistic for a good quadruped and
realistic for a geared hexapod with cycloids in the loop; it puts the design
in the ~{2*PACK_WH/1000:.1f} kWh, two {PACK_KG:.0f} kg packs range, which the mass budget carries.

## 9. What this pass decided and what it left open

Decided in review, or proposed here:

1. Robot ~{MASS.robot:.0f} kg, {BODY.length/1000:.1f} × {BODY.slab_width(ACT)/1000:.2f} × {BODY.height/1000:.1f} m slab body with the hips under it, hips {ST.hip_height/1000:.2f} m up.
2. Sprawled yaw–pitch–pitch legs (agreed): {LEG.coxa:.0f} mm coxa, femur {LEG.femur:.0f} mm up and out at {ST.femur_deg:.0f}°, tibia {LEG.tibia:.0f} mm vertical ({LEG.tibia_ratio:.1f}×, shortened from 2.5× for the knee's sake), with ±{YAW_RANGE_DEG:.0f}° of yaw so the same leg stands in a mammal stance when asked (agreed).
3. The rider is a stretch goal, not a rating (agreed).
4. Per-DOF actuators (agreed): yaw {DOF_CONT['yaw']:.0f} / {DOF_PEAK['yaw']:.0f}, femur {DOF_CONT['femur']:.0f} / {DOF_PEAK['femur']:.0f}, knee {DOF_CONT['knee']:.0f} / {DOF_PEAK['knee']:.0f} N·m continuous / peak; {JOINT_SPEED_NOMINAL:.0f} rad/s at the yaw; Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm.
5. One Ø{2*MOTOR.r_out:.0f} mm PCB axial-flux stator design, {DOF_PLAN['yaw']['label']} / {DOF_PLAN['femur']['label']} / {DOF_PLAN['knee']['label']} for yaw / femur / knee, cycloid ratios {DOF_PLAN['yaw']['ratio']:.0f} / {DOF_PLAN['femur']['ratio']:.0f} / {DOF_PLAN['knee']['ratio']:.0f}:1, {MOTOR.bus_v:.0f} V.
6. Two {PACK_WH:.0f} Wh hot-swap packs.

Open, in order of how much they change the actuator:

* **The step-up cost of the 2.5× tibia** — the leg reaches out to stand high; see the topology document §3 for what a 2.0× leg of the same femur would do instead.
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
    for d in DOFS:
        p = DOF_PLAN[d]
        print(f"  {d:6s} neutral {DOF_NEUTRAL[d]:5.0f} cont {DOF_CONT[d]:5.0f} peak {DOF_PEAK[d]:5.0f} rider {DOF_RIDER[d]:5.0f} "
              f"swing {DOF_SWING[d]:4.1f} -> {p['label']} {p['ratio']:.0f}:1 gives {p['cont']:.0f}/{p['peak']:.0f} at {p['speed']:.1f} rad/s {'ok' if p['ok'] else 'SLOW'}")
    print("power W walk/rider:", round(P_WALK), round(P_RIDER), "pack Wh/kg", round(PACK_WH), round(PACK_KG, 1))

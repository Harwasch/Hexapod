#!/opt/hw-py/bin/python
"""Does the robot close?  Joint torque scales with robot mass, robot mass
carries eighteen actuators, and an actuator's torque is what its motor and
ratio give.  This finds the fixed point for each unit option and the
requirement relief that would let the single-stator unit close.

    /opt/hw-py/bin/python analysis/closure.py

Inputs: cad/actuator/*.json (unit masses), hw/stator/asbuilt.json (motor
torque), hw/stator/rotor_field.json (magnet options), analysis/sizing.py
(joint torque per kilogram of robot at the confirmed leg geometry and load
cases; the mass the sizing currently carries is whatever hexapod_model.py
says, and the torque-per-kilogram is what is used here).
Writes hw/stator/closure.json and docs/design/actuator/closure.png.
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
import sizing as sz                                  # noqa: E402  (runs the sizing model)
import hexapod_model as hm                           # noqa: E402
import leg3d                                         # noqa: E402
import cycloid as cy                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = json.load(open(os.path.join(ROOT, "hw", "stator", "asbuilt.json")))
RF = json.load(open(os.path.join(ROOT, "hw", "stator", "rotor_field.json")))
AB16 = json.load(open(os.path.join(ROOT, "hw", "stator", "variants", "16L-2oz", "asbuilt.json")))   # two 8L 2 oz JLCPCB boards per position
CAD = {}
for tag in ("femur", "yaw", "femur-1s", "yaw-1s"):
    p = os.path.join(ROOT, "cad", "actuator", f"{tag}.json")
    if os.path.exists(p):
        CAD[tag] = json.load(open(p))
ETA_CYC = 0.90                                        # 20-lobe cycloid: larger eccentricity, fewer contacts than the 70-lobe (0.88)
M_ROBOT_NOW = hm.MASS.robot
M_FIXED = M_ROBOT_NOW - hm.MASS.actuators             # body, legs, batteries, electronics, margin
C = {d: float(sz.DOF_CONT[d]) / M_ROBOT_NOW for d in ("yaw", "femur", "knee")}      # N·m per kg of robot, continuous
C_PEAK = {d: float(sz.DOF_PEAK[d]) / M_ROBOT_NOW for d in ("yaw", "femur", "knee")}
T_MOTOR = AB["ratings"]["1000"]["T_cont"]             # one 12L 3 oz stator, 8 mm blocks, at 1000 rpm
T_MOTOR_CHEAP = AB16["ratings"]["1000"]["T_cont"]     # one position of two 8L 2 oz boards (the 16L 2 oz variant), the cost-down board
N_FK, N_YAW = cy.TOTAL["femur"], cy.TOTAL["yaw"]     # total ratio: cycloid x capstan
ETA_FK = ETA_CYC * cy.ETA2["femur"]
ETA_YAW = ETA_CYC * cy.ETA2["yaw"]


def fixed_point(m_fk, m_yaw):
    """Robot mass with 12 femur/knee units of m_fk and 6 yaw units of m_yaw."""
    return M_FIXED + 12 * m_fk + 6 * m_yaw


OPTIONS = {}
m2, my2 = CAD["femur"]["total_g"] / 1000, CAD["yaw"]["total_g"] / 1000          # canonical: two stators (round 7)
T1, Ty1 = T_MOTOR * N_FK * ETA_FK, T_MOTOR * N_YAW * ETA_YAW
T1c, Ty1c = T_MOTOR_CHEAP * N_FK * ETA_FK, T_MOTOR_CHEAP * N_YAW * ETA_YAW
OPTIONS["A-cost: 2 stators, 2 oz boards (chosen)"] = dict(m_fk=m2, m_yaw=my2, T_fk=2 * T1c, T_yaw=2 * Ty1c, h=CAD["femur"]["height_mm"], h_yaw=CAD["yaw"]["height_mm"])
B6 = RF["rect 30x5x6 N48"]["B1_midplane"] / RF["rect 30x5x8 N48"]["B1_midplane"]          # 6 mm blocks: lighter, cheaper, less field
DM6 = 2 * (RF["rect 30x5x8 N48"]["magnet_mass_g"] - RF["rect 30x5x6 N48"]["magnet_mass_g"]) / 1000   # kg saved per two-stator unit
OPTIONS["A-cost, 6 mm magnets"] = dict(m_fk=m2 - DM6, m_yaw=my2 - DM6, T_fk=2 * T1c * B6, T_yaw=2 * Ty1c * B6, h=CAD["femur"]["height_mm"] - 4, h_yaw=CAD["yaw"]["height_mm"] - 4)
OPTIONS["A: 2 stators, 3 oz boards"] = dict(m_fk=m2, m_yaw=my2, T_fk=2 * T1, T_yaw=2 * Ty1, h=CAD["femur"]["height_mm"], h_yaw=CAD["yaw"]["height_mm"])
if "femur-1s" in CAD and "yaw-1s" in CAD:
    m1, my = CAD["femur-1s"]["total_g"] / 1000, CAD["yaw-1s"]["total_g"] / 1000
    OPTIONS["B: 1 stator, 3 oz boards"] = dict(m_fk=m1, m_yaw=my, T_fk=T1, T_yaw=Ty1, h=CAD["femur-1s"]["height_mm"], h_yaw=CAD["yaw-1s"]["height_mm"])
for name, o in OPTIONS.items():
    m = fixed_point(o["m_fk"], o["m_yaw"])
    o["m_robot"] = m
    o["need_femur"], o["need_knee"], o["need_yaw"] = C["femur"] * m, C["knee"] * m, C["yaw"] * m
    o["margin_femur"] = o["T_fk"] / o["need_femur"]
    o["margin_knee"] = o["T_fk"] / o["need_knee"]
    o["margin_yaw"] = o["T_yaw"] / o["need_yaw"]
    o["closes"] = min(o["margin_femur"], o["margin_knee"], o["margin_yaw"]) >= 1.0

# ---- requirement relief: which definition of "continuous" closes ----------------------------
# The continuous requirement is the torque maximum over the routine working volume (stride
# ±200 mm, lateral ±100, step-up 150) under the walking load case: dyn 1.5 on three legs, on a
# 30 deg slope with 1 m/s^2 of acceleration.  Torque is linear in the load, so each candidate
# definition gives a torque per kilogram; the unit closes at the robot mass where c*m <= T_unit.
CASES = [
    ("as defined: dyn 1.5, 30° slope, 1 m/s², stride ±200", 1.5, 30.0, 1.0, 1.0),
    ("dyn 1.5, 30° slope, 1 m/s², stride ±150", 1.5, 30.0, 1.0, 0.75),
    ("dyn 1.2, 20° slope, 0.5 m/s², stride ±200", 1.2, 20.0, 0.5, 1.0),
    ("dyn 1.2, 15° slope, 0.5 m/s², stride ±150", 1.2, 15.0, 0.5, 0.75),
    ("level: dyn 1.2, 0°, 0.5 m/s², stride ±200", 1.2, 0.0, 0.5, 1.0),
    ("static tripod stance, level", 1.0, 0.0, 0.0, 1.0),
]
relief = []
for label, dyn, slope, accel, sc in CASES:
    lc = hm.LoadCase("case", 1.0, legs_down=3, dyn_factor=dyn, slope_deg=slope, accel=accel, rating="continuous")   # per kg of mass on the feet
    ws = leg3d.Workspace(dx=tuple(v * sc for v in leg3d.ROUTINE.dx), dy=leg3d.ROUTINE.dy, dz=leg3d.ROUTINE.dz, n=leg3d.ROUTINE.n)
    Fz, Fp = lc.foot_force_z, lc.foot_force_prop
    t = np.maximum(leg3d.evaluate(leg3d.CHOSEN, Fz, Fp, ws)["max"], leg3d.evaluate(leg3d.MAMMAL_MODE, Fz, Fp, ws)["max"])
    c = dict(yaw=float(t[0]), femur=float(t[1]), knee=float(t[2]))                    # N·m per kg on the feet
    payload = hm.MASS.mission_payload
    row = dict(label=label, dyn=dyn, slope=slope, accel=accel, stride_scale=sc, c_per_kg=c)
    for oname, o in OPTIONS.items():
        m = o["m_robot"] + payload
        need = {d: max(c[d] * m, 1e-6) for d in c}                 # yaw carries nothing in a static stance
        row[oname] = dict(need=need, margin_femur=o["T_fk"] / need["femur"], margin_knee=o["T_fk"] / need["knee"], margin_yaw=o["T_yaw"] / need["yaw"],
                          closes=min(o["T_fk"] / need["femur"], o["T_fk"] / need["knee"], o["T_yaw"] / need["yaw"]) >= 1.0)
    relief.append(row)
T_single = OPTIONS["B: 1 stator, 3 oz boards"]["T_fk"] if "B: 1 stator, 3 oz boards" in OPTIONS else T1
RELIEF_SCALE = None

out = dict(m_fixed=M_FIXED, torque_per_kg=C, torque_per_kg_peak=C_PEAK, T_motor=T_MOTOR, options=OPTIONS, cases=relief, T_single_unit=T_single)
json.dump(out, open(os.path.join(ROOT, "hw", "stator", "closure.json"), "w"), indent=1)

# ---- figure -----------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axes[0]
names = list(OPTIONS)
x = np.arange(len(names))
ax.bar(x - 0.2, [OPTIONS[n]["need_femur"] for n in names], 0.4, color="#b03a2e", label="femur needs (continuous)")
ax.bar(x + 0.2, [OPTIONS[n]["T_fk"] for n in names], 0.4, color="#0f9b8e", label="femur unit gives")
for i, n in enumerate(names):
    ax.text(i, max(OPTIONS[n]["need_femur"], OPTIONS[n]["T_fk"]) + 8, f"{OPTIONS[n]['m_robot']:.0f} kg robot\n{OPTIONS[n]['m_fk']:.1f} kg unit, {OPTIONS[n]['h']:.0f} mm\nyaw {OPTIONS[n]['margin_yaw']:.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([n.replace(", ", ",\n").replace(": ", ":\n") for n in names], fontsize=7)
ax.set_ylabel("N·m at the femur joint"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
ax.set_title("Each unit option at its own fixed-point robot mass", fontsize=10)
ax = axes[1]
labels = [r["label"].replace(", stride", ",\nstride").replace(": ", ":\n") for r in relief]
y = np.arange(len(relief))
for k, (oname, col) in enumerate((("B: 1 stator, 3 oz boards", "#0f9b8e"), ("A-cost: 2 stators, 2 oz boards (chosen)", "#d98c3a"))):
    if oname not in OPTIONS:
        continue
    vals = [max(r[oname]["need"]["femur"], r[oname]["need"]["knee"]) for r in relief]
    ax.barh(y + (k - 0.5) * 0.38, vals, 0.38, color=col, label=f"{oname}: worst of femur/knee needed at {OPTIONS[oname]['m_robot']:.0f} kg")
    ax.axvline(OPTIONS[oname]["T_fk"], color=col, ls="--", lw=1)
    for yy, r in zip(y, relief):
        if not r[oname]["closes"]:
            ax.text(vals[int(yy)] + 3, yy + (k - 0.5) * 0.38, "✗" if r[oname]["margin_yaw"] >= 1 else "✗ yaw", va="center", fontsize=7, color=col)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7); ax.invert_yaxis()
ax.set_xlabel("continuous joint torque needed (N·m); dashed = what the unit gives")
ax.legend(fontsize=7, loc="lower right"); ax.grid(axis="x", alpha=0.3)
ax.set_title("Which definition of 'continuous' each unit closes", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "design", "actuator", "closure.png"), dpi=110)

if __name__ == "__main__":
    print(f"fixed mass {M_FIXED:.1f} kg; torque per kg: femur {C['femur']:.2f}, knee {C['knee']:.2f}, yaw {C['yaw']:.2f} N·m/kg; motor {T_MOTOR:.2f} N·m")
    for n, o in OPTIONS.items():
        print(f"  {n:40s} unit {o['m_fk']:.2f} kg, robot {o['m_robot']:.0f} kg: femur {o['T_fk']:.0f}/{o['need_femur']:.0f} ({o['margin_femur']:.2f}), knee {o['margin_knee']:.2f}, yaw {o['T_yaw']:.0f}/{o['need_yaw']:.0f} ({o['margin_yaw']:.2f}) -> {'closes' if o['closes'] else 'does not close'}")
    for r in relief:
        line = f"  {r['label']:52s}"
        for oname in OPTIONS:
            o = r[oname]
            line += f" | {oname[:12]}: f {o['need']['femur']:.0f} k {o['need']['knee']:.0f} y {o['need']['yaw']:.0f} {'OK' if o['closes'] else '--'}"
        print(line)

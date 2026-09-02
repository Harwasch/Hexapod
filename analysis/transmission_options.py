#!/opt/hw-py/bin/python
"""Does an off-the-shelf outrunner still need the cycloid, or can it drive the
capstan joint directly?

    /opt/hw-py/bin/python analysis/transmission_options.py

The ratio a joint needs is joint torque / motor continuous torque, whatever
the motor: the 8318 class gives less continuous torque than the PCB motor
(2.6 vs 5.9 N·m heat-sunk), so it needs more ratio, not less.  What limits a
rope drive is not strength but wrap: the rope on the motor drum has to hold
(total ratio x joint travel) turns, and a capstan stops being a capstan and
becomes a winch beyond a few turns.  Each option below is evaluated for the
femur joint with one 8318 per unit, at the robot mass its own weight implies
(closure.py's fixed point), against the continuous load case as written.
Writes hw/stator/transmission_options.json and
docs/design/actuator/transmission-options.png.
"""
import csv
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = json.load(open(os.path.join(ROOT, "hw", "stator", "closure.json")))
MM = json.load(open(os.path.join(ROOT, "hw", "stator", "motor_market.json")))
CAP = json.load(open(os.path.join(ROOT, "hw", "stator", "capstan.json")))["geometry"]
M8318 = [r for r in MM["rows"] if r["name"].startswith("8318")][0]

C_FK = CL["torque_per_kg"]["femur"]          # N·m of continuous femur torque per kg of robot, load case as written
M_FIXED = CL["m_fixed"]                      # kg: body, legs, batteries, electronics, margin
M_YAW_OTS = 2.45                             # kg: an 8318 yaw unit (motor 0.65 + 30:1 cycloid 1.25 + laser-cut housing 0.55), as in cost_search.py
JOINT_RANGE = CAP["joint_range_deg"]         # 130 deg of femur / knee travel
T_CONT = M8318["T_cont"]                     # N·m, heat-sunk (assumed 1.2 K/W; the bench test)
T_PEAK = 3.0 * T_CONT                        # N·m, short bursts at ~80 A (assumed; saturation not modelled)
ETA_CYC, ETA_CAP, ETA_BELT = 0.90, 0.97, 0.96
WRAP_MAX = 4.0                               # working turns on a drum before it is a winch: fleet angle, rope stacking, a long drum
ROPE_BREAK_5MM = CAP["rope"]["break_kN"] * 1e3
R_SECTOR_MAX = 150.0                         # mm: a Ø300 sector at the femur pivot, the most the 150 mm coxa pod can plausibly carry

# BOM lines (docs/design/bom-actuator.csv, 20-unit column) for the reducer and the capstan stage
BOM = list(csv.DictReader(open(os.path.join(ROOT, "docs", "design", "bom-actuator.csv"))))
def lines(names):
    return sum(float(r["qty_per_unit"]) * float(r["unit_price_usd"]) for r in BOM if r["item"] in names)
COST_CYC = lines({"Cycloid disc", "Eccentric shaft", "Eccentric bearing", "Ring pins", "Output pins", "Pin cage"})
COST_CAP = lines({"Capstan drum", "Capstan sector", "Capstan rope", "Rope tensioner"})

def rope_d_for(F_peak, sf=3.0):
    """Smallest Dyneema diameter (mm) whose spliced strength, scaled by d^2 from the 5 mm datasheet number, gives sf at F_peak."""
    return 5.0 * math.sqrt(sf * F_peak / ROPE_BREAK_5MM)

def robot_mass(m_unit):
    return M_FIXED + 12 * m_unit + 6 * M_YAW_OTS

def closes(T_joint, m_unit):
    return T_joint >= C_FK * robot_mass(m_unit), C_FK * robot_mass(m_unit)

rows = []
def add(name, ratio, eta, m_unit, cost, turns, note, feasible=True):
    T_joint = T_CONT * ratio * eta
    ok, need = closes(T_joint, m_unit)
    rows.append(dict(name=name, ratio=ratio, eta=eta, T_joint=T_joint, m_unit=m_unit, m_robot=robot_mass(m_unit), T_need=need,
                     closes=ok and feasible, feasible=feasible, cost=cost, motor_drum_turns=turns, robot_mass_supported=T_joint / C_FK, note=note))

# 1. the existing drum and sector, motor straight onto the drum
k = CAP["ratio"]
add("Direct: motor on the Ø60 drum, 4:1 capstan", k, ETA_CAP, 0.65 + 0.55 + 0.4, COST_CAP, k * JOINT_RANGE / 360,
    "no reducer at all; the joint gets 4 x the motor's torque")

# 2. the largest single capstan the leg allows: Ø300 sector, the thinnest rope the peak tension permits, drum at 8 x rope
d_rope = rope_d_for(T_PEAK / (0.008))            # first guess at an 8 mm drum radius
r_drum = max(8.0 * d_rope / 2, 6.0)
d_rope = rope_d_for(T_PEAK / (r_drum * 1e-3)); r_drum = max(8.0 * d_rope / 2, 6.0)
k1 = R_SECTOR_MAX / r_drum
turns1 = k1 * JOINT_RANGE / 360
add(f"Largest single capstan: Ø{2*r_drum:.0f} drum, Ø{2*R_SECTOR_MAX:.0f} sector ({k1:.0f}:1)", k1, ETA_CAP, 0.65 + 0.55 + 0.6, COST_CAP + 15, turns1,
    f"{d_rope:.1f} mm rope at SF 3 on the burst torque; {turns1:.1f} working turns on the motor drum (a winch, not a capstan)", feasible=turns1 <= WRAP_MAX)

# 3. two rope stages, 10:1 each
add("Two rope stages, 10 x 10 (100:1)", 100, ETA_CAP ** 2, 0.65 + 0.55 + 1.0, 2 * COST_CAP, 100 * JOINT_RANGE / 360,
    f"{100 * JOINT_RANGE / 360:.0f} turns on the motor drum for 130 deg of joint travel: a multi-layer winch at 3600 rpm", feasible=False)

# 4. two timing-belt stages and the capstan: the second belt has to carry the drum torque
T_belt2 = T_CONT * 5 * ETA_BELT * 5 * ETA_BELT            # N·m continuous on the 25:1 output, before the capstan
add("Belt 5:1 + belt 5:1 + capstan 4:1 (100:1)", 100, ETA_BELT ** 2 * ETA_CAP, 0.65 + 0.55 + 1.6, 2 * 45 + COST_CAP, 4 * JOINT_RANGE / 360,
    f"stage-2 pulley carries {T_belt2:.0f} N·m continuous / {3*T_belt2:.0f} burst: HTD 8M, 30 mm wide, a Ø300 large pulley that does not fit the Ø192 pod", feasible=False)

# 5. as designed for the outrunner in cost_search: 25-lobe cycloid x 4:1 capstan
add("25-lobe cycloid + 4:1 capstan (100:1, as costed)", 100, ETA_CYC * ETA_CAP, 0.65 + 1.25 + 0.55 + 0.3, COST_CYC + COST_CAP, k * JOINT_RANGE / 360,
    f"the cost-search pick: reducer lines ${COST_CYC:.0f}, capstan lines ${COST_CAP:.0f} at 20 units")

# 6. OTS planetary instead of the cycloid
add("OTS 25:1 planetary (PLE/PLF-60 class) + 4:1 capstan", 100, 0.94 * ETA_CAP, 0.65 + 1.6 + 0.55 + 0.3, 140 + COST_CAP, k * JOINT_RANGE / 360,
    "a 60-frame two-stage planetary is listed around $120-160 (est, unverified) and rated ~40-60 N·m; the 66 N·m continuous here is at its limit, an 80-frame is $200+")

# 7. quasi-direct: what motor would let the capstan alone do it?  A 10-inch e-scooter hub motor at 10:1
KV_HUB, R_HUB, D_HUB, H_HUB, M_HUB = 16.7, 0.25, 250, 60, 4.5           # est from listings: ~800 rpm no-load at 48 V, ~0.25 ohm line-line
R_th_hub = max(MM["assumptions"]["R_th_ref"] * MM["assumptions"]["A_ref_m2"] / (math.pi * D_HUB * 1e-3 * H_HUB * 1e-3), 0.3)
I_hub = math.sqrt(MM["assumptions"]["dT_K"] / R_th_hub / (1.5 * R_HUB))
T_hub = 60 / (2 * math.pi * KV_HUB) * I_hub
k_hub = R_SECTOR_MAX / 15.0                                          # Ø30 drum on the hub, 5 mm rope
T_joint_hub = T_hub * k_hub * ETA_CAP
ok_hub, need_hub = closes(T_joint_hub, M_HUB + 1.0)
rows.append(dict(name=f"Quasi-direct: 10-inch hub motor on a {k_hub:.0f}:1 capstan", ratio=k_hub, eta=ETA_CAP, T_joint=T_joint_hub, m_unit=M_HUB + 1.0,
                 m_robot=robot_mass(M_HUB + 1.0), T_need=need_hub, closes=False, feasible=False, cost=80 + COST_CAP + 20,
                 motor_drum_turns=k_hub * JOINT_RANGE / 360, robot_mass_supported=T_joint_hub / C_FK,
                 note=f"{T_hub:.0f} N·m heat-sunk from a Ø{D_HUB} x {H_HUB} hub (est), 12 of them add 54 kg; the pod would be Ø260 and the sector Ø300; "
                      f"{KV_HUB * 48 * 2 * math.pi / 60 / k_hub:.1f} rad/s joint no-load is fast enough but the mass never closes"))

json.dump(dict(motor=dict(name=M8318["name"], T_cont=T_CONT, T_peak_assumed=T_PEAK), c_per_kg_femur=C_FK, m_fixed=M_FIXED, wrap_max_turns=WRAP_MAX,
               joint_range_deg=JOINT_RANGE, r_sector_max_mm=R_SECTOR_MAX, cost_cycloid_lines=COST_CYC, cost_capstan_lines=COST_CAP,
               hub=dict(kv=KV_HUB, R=R_HUB, R_th=R_th_hub, I_cont=I_hub, T_cont=T_hub), rows=rows),
          open(os.path.join(ROOT, "hw", "stator", "transmission_options.json"), "w"), indent=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
ax = axes[0]
y = np.arange(len(rows))
cols = ["#0f9b8e" if r["closes"] else ("#d98c3a" if r["feasible"] else "#b03a2e") for r in rows]
ax.barh(y, [r["T_joint"] for r in rows], color=cols, height=0.55)
for i, r in enumerate(rows):
    ax.plot([r["T_need"], r["T_need"]], [i - 0.35, i + 0.35], color="#222", lw=2)
    ax.text(max(r["T_joint"], r["T_need"]) + 5, i, f"{r['T_joint']:.0f} N·m vs {r['T_need']:.0f} needed at {r['m_robot']:.0f} kg; ${r['cost']:.0f}", va="center", fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels([r["name"].split(" (")[0] for r in rows], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("continuous femur torque at the joint, one 8318 per unit (N·m)"); ax.set_xlim(0, 420)
ax.set_title("Bar: what the joint gets. Black tick: what it needs at that option's robot mass.\n"
             "Green closes; orange is buildable but short; red cannot be built as a capstan", fontsize=9)
ax.grid(axis="x", alpha=0.3)
ax = axes[1]
ratios = np.linspace(1, 120, 200)
ax.plot(ratios, ratios * JOINT_RANGE / 360, color="#0f9b8e", label=f"working turns on the motor drum for {JOINT_RANGE:.0f}° of joint travel")
ax.axhline(WRAP_MAX, color="#b03a2e", ls="--", label=f"{WRAP_MAX:.0f} turns: beyond this the drum is a winch (fleet angle, stacking, length)")
ax.axvline(T_CONT and (C_FK * robot_mass(0.65 + 1.25 + 0.55 + 0.3)) / (T_CONT * ETA_CYC * ETA_CAP), color="#555", ls=":", label="ratio the 8318 needs at its own fixed-point mass")
ax.axvline(k, color="#d98c3a", ls=":", label="4:1, the capstan as designed")
for r in rows:
    if r["motor_drum_turns"] < 40:
        ax.scatter(r["ratio"], r["motor_drum_turns"], s=40, color="#222", zorder=3)
ax.set_xlabel("total rope-drive ratio, motor to joint"); ax.set_ylabel("turns the rope must wrap on the motor drum")
ax.set_ylim(0, 40); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("Why a pure rope drive tops out near 10:1, whatever the motor", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "design", "actuator", "transmission-options.png"), dpi=110)

if __name__ == "__main__":
    print(f"8318: {T_CONT:.2f} N·m cont, {T_PEAK:.1f} burst assumed; femur needs {C_FK:.2f} N·m/kg; cycloid lines ${COST_CYC:.0f}, capstan lines ${COST_CAP:.0f}")
    for r in rows:
        print(f"{r['name']:62s} {r['ratio']:6.1f}:1 eta {r['eta']:.2f} joint {r['T_joint']:6.1f} N·m need {r['T_need']:5.0f} at {r['m_robot']:5.1f} kg "
              f"supports {r['robot_mass_supported']:5.1f} kg  turns {r['motor_drum_turns']:5.1f}  ${r['cost']:.0f}  {'CLOSES' if r['closes'] else ('short' if r['feasible'] else 'infeasible')}")
        print("    " + r["note"])

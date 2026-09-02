#!/opt/hw-py/bin/python
"""The second reduction stage: a capstan cable drive from the actuator's output
drum to a sector on the joint, so the in-plane cycloid drops from 70 to 20
lobes and sees 3.5x less torque.

    /opt/hw-py/bin/python analysis/capstan.py

Geometry: drum of radius r_d on the actuator output (in the hip pod, below the
mounting face), sector of radius r_s at the joint, ratio r_s / r_d.  One rope,
pre-tensioned, wraps the drum n_wrap turns and terminates on the sector at both
ends (a "capstan" or "cable-cabestan" drive: no slip, no backlash, the rope's
stretch is the only compliance).  The rope is Dyneema SK78 (Marlow D12 78,
docs/reference): steel wire at this tension would need a drum 25-40x its
diameter, which the pod does not have.

Loads: rope tension = cycloid output torque / r_d (continuous and peak);
safety factor against the rope's average break load; bending ratio D/d;
stretch under load -> joint compliance; the yaw-to-pitch coupling because the
drum sits on the yaw axis and the sector on the coxa (the rope run turns with
the coxa, so a yaw rotation winds the drum by the same angle: the controller
adds yaw to the femur/knee command).
Writes hw/stator/capstan.json and docs/design/actuator/capstan.png.
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
import cycloid as cy  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "design", "actuator")

# ---- the rope (Marlow D12 78, 12-strand Dyneema SK78; docs/reference/manifest.yaml) ----------
ROPE = {"name": "Marlow D12 Max 78, 5 mm", "d_mm": 5.0, "break_kN": 29.2, "break_linear_kN": 32.4, "mass_g_per_m": 15.6,
        "E_eff_GPa": 60.0,                 # working-range modulus after bedding in: the datasheet's used-rope curve, ~1 % at 30 % MBL
        "creep_pct_per_yr_at_20pct": 0.5,  # SK78 at 20 % load, 16 C (datasheet); re-tension on a schedule
        "T_critical_C": 80.0}              # datasheet: permanent strength loss above 80 C -> keep the rope off the housing
# break_kN is the minimum SPLICED strength (the rope terminates in splices); the linear MBL is 32.4 kN
SF_MIN_CONT, SF_MIN_PEAK = 5.0, 3.0        # rope working-load rules of thumb for cyclic loads
D_OVER_D_MIN = 8.0                         # Dyneema on a drum: >= 8 keeps the bending fatigue negligible

# ---- geometry (mm) ----------------------------------------------------------------------------
R_DRUM = 30.0                              # Ø60 drum on the unit's output flange (cad/actuator)
K2 = cy.STAGE2["femur"]
R_SECTOR = R_DRUM * K2                     # 105 mm: a Ø210 sector at the femur / knee pivot
JOINT_RANGE_DEG = 130.0                    # femur / knee travel the sector has to cover
N_WRAP = 3.0                               # rope turns on the drum, to spread the load and hold pretension by friction
L_RUN = 320.0                              # rope run from the drum to the sector, along the coxa (150 mm coxa + pod)

results = {}
for joint in ("femur", "knee"):
    N_cyc, T_c, T_p = cy.JOINTS[joint]                          # cycloid output torque (cont / peak) = drum torque
    F_c, F_p = T_c / (R_DRUM * 1e-3), T_p / (R_DRUM * 1e-3)      # rope tension, N
    sf_c, sf_p = ROPE["break_kN"] * 1e3 / F_c, ROPE["break_kN"] * 1e3 / F_p
    A = math.pi * (ROPE["d_mm"] / 2) ** 2 * 0.6                  # load-bearing area of a 12-strand braid ~60 % of the circle
    k_rope = ROPE["E_eff_GPa"] * 1e3 * A / L_RUN                 # N/mm, one run
    k_joint = 2 * k_rope * (R_SECTOR * 1e-3) ** 2 * 1e3          # N·m/rad: both runs act about the sector (pretensioned)
    wind_up_deg = math.degrees(T_c * cy.STAGE2[joint] / k_joint)  # at the continuous joint torque
    sector_arc = math.radians(JOINT_RANGE_DEG) * R_SECTOR + 2 * math.pi * R_DRUM * N_WRAP
    results[joint] = dict(T_drum_cont=T_c, T_drum_peak=T_p, F_rope_cont=F_c, F_rope_peak=F_p, sf_cont=sf_c, sf_peak=sf_p,
                          ok=sf_c >= SF_MIN_CONT and sf_p >= SF_MIN_PEAK, D_over_d=2 * R_DRUM / ROPE["d_mm"],
                          k_joint_Nm_per_rad=k_joint, wind_up_deg_at_cont=wind_up_deg, rope_length_mm=sector_arc + 2 * L_RUN,
                          drum_turns_for_range=JOINT_RANGE_DEG / 360 * K2, pretension_N=0.15 * F_p)
GEO = dict(r_drum_mm=R_DRUM, r_sector_mm=R_SECTOR, ratio=K2, n_wrap=N_WRAP, run_mm=L_RUN, joint_range_deg=JOINT_RANGE_DEG,
           d_over_d_min=D_OVER_D_MIN, rope=ROPE, yaw_coupling="joint = sector angle - yaw angle x (r_drum / r_sector)")
json.dump(dict(geometry=GEO, joints=results), open(os.path.join(ROOT, "hw", "stator", "capstan.json"), "w"), indent=1)

# ---- figure: the drive in the coxa plane and the tension budget -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax = axes[0]
xd, xs = 0.0, 150.0 + R_DRUM                                 # drum on the yaw axis, sector at the femur pivot along the coxa
ax.add_patch(plt.Circle((xd, 0), R_DRUM, fill=False, lw=2, color="#6c8e3a"))
ax.add_patch(plt.Circle((xd, 0), 12.5, color="#d98c3a"))
th = np.linspace(math.radians(90 - JOINT_RANGE_DEG / 2 - 20), math.radians(90 + JOINT_RANGE_DEG / 2 + 20), 60)
ax.plot(xs + R_SECTOR * np.cos(th), R_SECTOR * np.sin(th), color="#c0392b", lw=3)
ax.plot([xs, xs + R_SECTOR * math.cos(th[0])], [0, R_SECTOR * math.sin(th[0])], color="#c0392b", lw=1)
ax.plot([xs, xs + R_SECTOR * math.cos(th[-1])], [0, R_SECTOR * math.sin(th[-1])], color="#c0392b", lw=1)
ax.add_patch(plt.Circle((xs, 0), 8, color="#3a3a3a"))
# the two rope runs: tangent from the drum to the sector (drawn as straight lines to the sector rim)
for sgn in (1, -1):
    ax.plot([xd, xs], [sgn * R_DRUM, sgn * R_SECTOR], color="#555", lw=1.2, ls="-")
ax.annotate("", (xd, -R_DRUM - 12), (xs, -R_DRUM - 12), arrowprops=dict(arrowstyle="<->", color="#b03a2e", lw=0.8))
ax.text((xd + xs) / 2, -R_DRUM - 22, "150 mm coxa", ha="center", fontsize=8, color="#b03a2e")
ax.text(xd, R_DRUM + 8, f"drum Ø{2*R_DRUM:.0f} on the unit output\n{N_WRAP:.0f} wraps, on the yaw axis", ha="center", fontsize=8)
ax.text(xs, -R_SECTOR - 14, f"sector Ø{2*R_SECTOR:.0f} at the femur / knee pivot\n{JOINT_RANGE_DEG:.0f}° of travel, ratio {K2:.1f}:1", ha="center", fontsize=8)
ax.set_aspect("equal"); ax.set_xlim(-60, xs + R_SECTOR + 30); ax.set_ylim(-R_SECTOR - 40, R_SECTOR + 40)
ax.set_xlabel("mm, in the coxa plane"); ax.grid(alpha=0.2)
ax.set_title("Capstan stage: drum on the actuator, sector on the joint, one pre-tensioned Dyneema rope", fontsize=10)
ax = axes[1]
labels, F_c, F_p = [], [], []
for j, r in results.items():
    labels.append(j); F_c.append(r["F_rope_cont"] / 1e3); F_p.append(r["F_rope_peak"] / 1e3)
x = np.arange(len(labels))
ax.bar(x - 0.2, F_c, 0.4, color="#0f9b8e", label="rope tension, continuous")
ax.bar(x + 0.2, F_p, 0.4, color="#d98c3a", label="rope tension, peak")
ax.axhline(ROPE["break_kN"] / SF_MIN_CONT, color="#0f9b8e", ls="--", lw=1, label=f"break / {SF_MIN_CONT:.0f}")
ax.axhline(ROPE["break_kN"] / SF_MIN_PEAK, color="#d98c3a", ls="--", lw=1, label=f"break / {SF_MIN_PEAK:.0f}")
for i, j in enumerate(labels):
    ax.text(i, F_p[i] + 0.2, f"SF {results[j]['sf_cont']:.1f} / {results[j]['sf_peak']:.1f}\nwind-up {results[j]['wind_up_deg_at_cont']:.2f}°", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("kN"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"{ROPE['name']}: {ROPE['break_kN']:.1f} kN break, D/d = {2*R_DRUM/ROPE['d_mm']:.0f}", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "capstan.png"), dpi=110)

if __name__ == "__main__":
    for j, r in results.items():
        print(f"{j}: drum torque {r['T_drum_cont']:.0f}/{r['T_drum_peak']:.0f} N·m, rope {r['F_rope_cont']/1e3:.2f}/{r['F_rope_peak']/1e3:.2f} kN, SF {r['sf_cont']:.1f}/{r['sf_peak']:.1f} "
              f"({'ok' if r['ok'] else 'NOT ok'}), D/d {r['D_over_d']:.0f}, joint stiffness {r['k_joint_Nm_per_rad']:.0f} N·m/rad, wind-up {r['wind_up_deg_at_cont']:.2f}° at cont, rope {r['rope_length_mm']/1e3:.2f} m, pretension {r['pretension_N']:.0f} N")

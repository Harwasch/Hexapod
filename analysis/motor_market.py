#!/opt/hw-py/bin/python
"""Market search: which off-the-shelf motor gives the joint the most for the
least, at 48 V, heat-sunk in a closed body, through the 25-lobe cycloid and
the 4:1 capstan (100:1) or the 20-lobe x 4 (80:1).

    /opt/hw-py/bin/python analysis/motor_market.py

Every candidate's numbers are from a product listing found on 2026-09-02 (the
URL is recorded); none is a datasheet, and the ones marked "est" are inferred
(a resistance scaled by turns^2 from a sibling KV, a price from a similar
listing).  The continuous torque is NOT the listing's propeller-cooled
rating: it is what the copper can dissipate through an assumed thermal
resistance from the stator to a heat-sunk aluminium mount, which scales with
the stator's outside area.  That assumption is the one bench test this study
needs.  Writes hw/stator/motor_market.json and docs/design/actuator/motor-market.png.
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
import motor_options as mo   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = mo.T_CU_MAX_PCB - mo.T_AMB                 # 75 K copper over a 45 C body
R_TH_REF, A_REF = 1.2, math.pi * 0.092 * 0.040  # K/W assumed for a Ø92 x 40 outrunner stator heat-sunk to an aluminium plate
NEED_RPM = {"80:1": 3.8 * 80 * 60 / (2 * math.pi), "100:1": 3.8 * 100 * 60 / (2 * math.pi)}   # femur swing 3.8 rad/s
NEED_T = {"80:1": 236 / (80 * 0.90 * 0.97), "100:1": 205 / (100 * 0.90 * 0.97)}                # femur continuous at the outrunner family's fixed-point mass: 77 kg (one motor, 100:1) / 89 kg (two, 80:1), 08 s9.12

# name: kv (rpm/V), R_ll (ohm, line-line), mass (kg), d x h (mm), price at 1 / at 20 (USD), source, flags
CANDIDATES = [
    dict(name="8318 100KV (HL Q9XL / Alibaba class)", kv=100, R=0.055, mass=0.65, d=92, h=40, price1=65, price20=50,
         src="https://www.himodel.com/electric/HL_Q9XL_8318_100KV_8-14S_Outrunner_Brushless_Motor.html", flags="R and mass from the listing; price est from GBP 49 and Alibaba"),
    dict(name="Turnigy Multistar 9235-100KV", kv=100, R=0.055, mass=0.674, d=92, h=39, price1=86, price20=75,
         src="https://hobbyking.com/en_us/9235-100kv-turnigy-multistar-brushless-multi-rotor-motor.html", flags="R, mass, 57 A prop-cooled from the listing; price from an eBay listing"),
    dict(name="6384 120KV (skateboard, sensored)", kv=120, R=0.125, mass=0.95, d=63, h=84, price1=70, price20=55,
         src="https://www.amazon.com/Outrunner-Brushless-Sensored-Balancing-Skateboard/dp/B081Z7YWLF", flags="R est: 0.05 ohm of the 190KV sibling x (190/120)^2; mass est 0.9-1.05 kg; price est"),
    dict(name="63100 190KV (skateboard, sensored)", kv=190, R=0.02, mass=1.1, d=63, h=100, price1=95, price20=80,
         src="https://www.amazon.com/Sensored-Outrunner-Brushless-Skateboard-Longboard/dp/B0BPCBPTFV", flags="R and mass est from the class; price from the listing"),
    dict(name="MAD M8S C08 8108 EEE 100KV", kv=100, R=0.12, mass=0.303, d=87, h=29, price1=269, price20=230,
         src="https://www.amazon.com/M8S-100KV-brushless-Drone-Motor/dp/B0DCK17NFK", flags="mass, 31.8 A prop-cooled and price from the listing; R est"),
    dict(name="T-Motor U8 II KV100", kv=100, R=0.7, mass=0.277, d=87, h=29, price1=250, price20=220,
         src="https://store.tmotor.com/product/tmotor-u8-v2-u-efficiency-kv100.html", flags="mass and R from the listing (0.7 ohm as quoted; suspiciously high for KV100), price est"),
    dict(name="6.5 in hoverboard hub motor, 36 V 350 W", kv=22, R=0.30, mass=2.9, d=165, h=70, price1=40, price20=25,
         src="https://www.alibaba.com/product-detail/6-5-inch-Hoverboard-Motor-Wheel_62386612847.html", flags="15 pole pairs, 800 rpm no-load at 36 V, rated 5 N·m / 10 peak from listings; KV and R est from those; price from Alibaba"),
]

rows = []
for c in CANDIDATES:
    Kt = 60 / (2 * math.pi * c["kv"])
    A_stator = math.pi * c["d"] * 1e-3 * c["h"] * 1e-3
    R_th = R_TH_REF * A_REF / A_stator                       # bigger stator surface, better heat-sinking
    R_th = max(R_th, 0.6)
    P_allow = DT / R_th
    I_cont = math.sqrt(P_allow / (1.5 * c["R"]))
    T_cont = Kt * I_cont
    Km = Kt / math.sqrt(1.5 * c["R"])
    n_noload = c["kv"] * 48.0
    ratio = "100:1" if n_noload >= NEED_RPM["100:1"] else ("80:1" if n_noload >= NEED_RPM["80:1"] else "too slow")
    if n_noload >= NEED_RPM["100:1"] and T_cont >= NEED_T["100:1"]:
        n_needed, ratio = 1, "100:1"
    elif n_noload >= NEED_RPM["80:1"] and 2 * T_cont >= NEED_T["80:1"]:
        n_needed, ratio = 2, "80:1"
    else:
        n_needed = None
    fits = c["d"] <= 120 and c["h"] <= 100
    rows.append(dict(c, Kt=Kt, Km=Km, R_th=R_th, P_allow=P_allow, I_cont=I_cont, T_cont=T_cont, n_noload=n_noload, ratio=ratio,
                     motors_per_unit=n_needed, fits=fits, T_per_usd=T_cont / c["price20"], T_per_kg=T_cont / c["mass"],
                     unit_motor_cost=(n_needed or 0) * c["price20"], unit_motor_mass=(n_needed or 0) * c["mass"]))
rows.sort(key=lambda r: (r["motors_per_unit"] is None, r["unit_motor_cost"] if r["motors_per_unit"] else 1e9))
json.dump(dict(assumptions=dict(dT_K=DT, R_th_ref=R_TH_REF, A_ref_m2=A_REF, need_rpm=NEED_RPM, need_T=NEED_T), rows=rows),
          open(os.path.join(ROOT, "hw", "stator", "motor_market.json"), "w"), indent=1)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
ax = axes[0]
for r in rows:
    ok = r["motors_per_unit"] is not None
    ax.scatter(r["mass"], r["T_cont"], s=110, color="#0f9b8e" if ok else "#b03a2e", marker="o" if r["fits"] else "x")
    ax.annotate(f"{r['name'].split(' (')[0]}\n${r['price20']}, {r['ratio']}", (r["mass"], r["T_cont"]), (6, 4), textcoords="offset points", fontsize=7)
ax.axhline(NEED_T["100:1"], color="#555", ls="--", lw=0.9, label=f"one motor per unit needs {NEED_T['100:1']:.1f} N·m (100:1, 77 kg robot)")
ax.axhline(NEED_T["80:1"] / 2, color="#999", ls=":", lw=0.9, label=f"two per unit need {NEED_T['80:1']/2:.1f} N·m each (80:1, 89 kg)")
ax.scatter([2.7], [5.9], s=110, color="#555", marker="D"); ax.annotate("PCB 2-stator motor\n(as designed) $150-240", (2.7, 5.9), (6, 4), textcoords="offset points", fontsize=7)
ax.set_xlabel("motor mass (kg)"); ax.set_ylabel("continuous torque, heat-sunk in a closed body (N·m)")
ax.set_title("Candidates: what the copper can carry through an assumed heat-sink path", fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax = axes[1]
names = [r["name"].split(" (")[0] for r in rows]
y = np.arange(len(rows))
ax.barh(y - 0.2, [r["T_per_usd"] * 100 for r in rows], 0.4, color="#d98c3a", label="continuous N·m per $100 (20 pcs)")
ax.barh(y + 0.2, [r["T_per_kg"] for r in rows], 0.4, color="#0f9b8e", label="continuous N·m per kg")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=8)
ax.set_title("Torque per dollar and per kilogram", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "design", "actuator", "motor-market.png"), dpi=110)

if __name__ == "__main__":
    for r in rows:
        print(f"{r['name']:42s} Kt {r['Kt']:.3f} Km {r['Km']:.2f} R_th {r['R_th']:.2f} I {r['I_cont']:.0f} A T {r['T_cont']:.2f} N·m {r['n_noload']:.0f} rpm {r['ratio']:>8s} "
              f"motors/unit {r['motors_per_unit']} ${r['price20']} {r['T_per_usd']*100:.1f} N·m/$100 {r['T_per_kg']:.1f} N·m/kg fits {r['fits']}")

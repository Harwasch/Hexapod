#!/opt/hw-py/bin/python
"""Exhaust the cost routes: a parametric cost + mass + closure model over the
actuator design space, so the minimum for the requirement is a computed thing
rather than the last idea tried.

    /opt/hw-py/bin/python analysis/cost_search.py

Axes:
  motor       PCB axial, 2 stators, layout v2, 2 oz boards (as designed)
              PCB axial, 1 stator, 3 oz board (the lighter unit that did not close)
              OTS outrunner 8318 100KV, one or two per unit, heat-sunk stator
  reduction   20-lobe cycloid x 4:1 capstan (80:1) or 25-lobe x 4:1 (100:1) for the
              slower-KV-less outrunner; the yaw a direct cycloid
  structure   laser-cut / tube / turned (20 units) or cast base + plates (100+)
  driver      ODrive-compatible clone ($45) or a custom FOC board ($25 at 100+)
  quantity    20 / 100 units
  requirement the continuous load case as written, or level walking at dyn 1.2

For every combination: unit mass, the robot's fixed-point mass (12 femur/knee
units, 6 yaw units, the fixed 29 kg), the joint torque that mass needs, the
torque the unit gives, the margin, and the unit cost.  Costs are the BOM's
lines re-used where the part is the same and marked estimates where not;
nothing here is a quote.  Writes hw/stator/cost_search.json and
docs/design/actuator/cost-search.png.
"""
import csv
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
CL = json.load(open(os.path.join(ROOT, "hw", "stator", "closure.json")))
CAD = json.load(open(os.path.join(ROOT, "cad", "actuator", "femur.json")))
CAD_YAW = json.load(open(os.path.join(ROOT, "cad", "actuator", "yaw.json")))
BOM = list(csv.DictReader(open(os.path.join(ROOT, "docs", "design", "bom-actuator.csv"))))
M_FIXED = CL["m_fixed"]
C = CL["torque_per_kg"]                              # N·m per kg of robot, continuous, requirement as written
CASE_LEVEL = [c for c in CL["cases"] if c["label"].startswith("level")][0]["c_per_kg"]   # per kg on the feet, level walking dyn 1.2
PAYLOAD = 8.0
ETA_CYC, ETA_CAP = 0.90, 0.97
DT = mo.T_CU_MAX_PCB - mo.T_AMB                      # 75 K copper over ambient


def bom(items, col):
    return sum(float(r["qty_per_unit"]) * float(r[col]) for r in BOM if r["item"] in items)


COL = {20: "unit_price_usd", 100: "price_100_usd"}
MOTOR_LINES = ["Stator boards", "Rotor magnets", "Clamp rings", "Cover", "Rotor cup", "Adhesive"]
REDUCER_LINES = ["Eccentric bearing", "Shaft bearing (lower)", "Shaft bearing (top)", "Ring pins", "Output pins", "Cycloid disc",
                 "Eccentric shaft", "Output flange", "Output bearing", "Bearing carrier", "Pin cage"]
CAPSTAN_LINES = ["Capstan drum", "Capstan sector", "Capstan rope", "Rope tensioner"]
HOUSING_LINES = ["Floor plate", "Wall tube", "Housing screws", "Output screws", "Connectors", "Thermistor"]
ELEC_LINES = ["Motor driver", "Rotor encoder"]

# ---- the OTS outrunner anchor: 8318 100KV (HL Q9XL / Alibaba class; specs from the listings, not a datasheet) ----
OUT = dict(name="8318 100KV outrunner", kv=100.0, R_ll=0.055, mass_kg=0.65, d_mm=92, h_mm=40, price={20: 65.0, 100: 48.0},
           R_th=1.2,            # K/W stator to a heat-sunk aluminium mount, ASSUMED (the listings quote 58 A only with propeller airflow)
           T_cu_max=120.0)
OUT["Kt"] = 60 / (2 * math.pi * OUT["kv"])                                   # N·m/A (line-to-line, sinusoidal drive)
OUT["I_cont"] = math.sqrt((OUT["T_cu_max"] - mo.T_AMB) / OUT["R_th"] / (1.5 * OUT["R_ll"]))
OUT["T_cont"] = OUT["Kt"] * OUT["I_cont"]
OUT["n_noload"] = OUT["kv"] * 48.0
PCB_T2 = CL["options"]["A-cost: 2 stators, 2 oz boards, 6 mm magnets (chosen)"]["T_fk"] / (80 * ETA_CYC * ETA_CAP)   # N·m of the 2-stator motor
PCB_T1_3OZ = CL["options"]["B: 1 stator, 3 oz boards, 8 mm magnets"]["T_fk"] / (80 * ETA_CYC * ETA_CAP)

OPTIONS = []


def add(name, motor_desc, m_fk, m_yaw, T_motor_fk, T_motor_yaw, ratio_fk, ratio_yaw, cost_fn, note, family):
    OPTIONS.append(dict(name=name, motor=motor_desc, m_fk=m_fk, m_yaw=m_yaw, T_motor_fk=T_motor_fk, T_motor_yaw=T_motor_yaw,
                        ratio_fk=ratio_fk, ratio_yaw=ratio_yaw, cost_fn=cost_fn, note=note, family=family))


m_pcb2, m_pcb2y = CAD["total_g"] / 1000, CAD_YAW["total_g"] / 1000
m_pcb1 = json.load(open(os.path.join(ROOT, "cad", "actuator", "femur-1s.json")))["total_g"] / 1000
add("PCB 2-stator v2, 2 oz boards (as designed)", "PCB axial, 2 stators, 6 mm Halbach", m_pcb2, m_pcb2y, PCB_T2, PCB_T2, 80, 30,
    lambda q: bom(MOTOR_LINES + REDUCER_LINES + CAPSTAN_LINES + HOUSING_LINES + ELEC_LINES, COL[q]), "the round-9 unit", "pcb")
add("PCB 1-stator v2, 3 oz board", "PCB axial, 1 stator, 8 mm Halbach", m_pcb1, m_pcb1 - 0.1, PCB_T1_3OZ, PCB_T1_3OZ, 80, 30,
    lambda q: bom(MOTOR_LINES + REDUCER_LINES + CAPSTAN_LINES + HOUSING_LINES + ELEC_LINES, COL[q]) - bom(["Stator boards", "Rotor magnets"], COL[q]) / 2 + (150 if q == 20 else 110) - bom(["Stator boards"], COL[q]) / 2,
    "one board (3 oz, $150/$110) and two rings", "pcb")
# one stator, but everything else of round 9 (v2 board, 2 oz, 6 mm magnets), at 100:1 with the 8-turn board for speed
add("PCB 1-stator v2, 2 oz, 8-turn board, 100:1", "PCB axial, 1 stator, 6 mm Halbach, 25-lobe x 4", m_pcb1 - 0.27, m_pcb1 - 0.37, PCB_T2 / 2 * 0.955, PCB_T2 / 2 * 0.955, 100, 40,
    lambda q: bom(MOTOR_LINES + REDUCER_LINES + CAPSTAN_LINES + HOUSING_LINES + ELEC_LINES, COL[q]) - bom(["Stator boards", "Rotor magnets"], COL[q]) / 2 - bom(["Clamp rings"], COL[q]) / 2 - 12,
    "joint speed 3.6 rad/s against 3.8 needed (0.95); the 8-turn board loses 6x the eddy power at speed", "pcb")
# outrunner units: reducer + capstan + a laser-cut housing around the motor; the cycloid sits beside the motor
m_red = 1.25                                                                  # kg: discs, shaft, pins, cage, bearings, flange, drum (from the CAD's mass table)
m_hsg = 0.55                                                                  # kg: plates, standoffs, screws
for n_mot, ratio in ((1, 100), (2, 80)):
    add(f"{n_mot} x 8318 outrunner, {ratio}:1", f"{n_mot} x {OUT['name']}, heat-sunk", n_mot * OUT["mass_kg"] + m_red + m_hsg + 0.3,
        n_mot * OUT["mass_kg"] + m_red + m_hsg, n_mot * OUT["T_cont"], n_mot * OUT["T_cont"], ratio, 30 if n_mot == 2 else 40,
        (lambda q, n=n_mot: n * OUT["price"][q] + bom(REDUCER_LINES + CAPSTAN_LINES + HOUSING_LINES + ELEC_LINES, COL[q]) + (15 if n == 2 else 0) + 20),
        f"{'two motors share the eccentric shaft through a belt; ' if n_mot == 2 else ''}continuous torque assumes {OUT['R_th']} K/W to a heat-sunk mount ({OUT['I_cont']:.0f} A, unverified)", "ots")

rows = []
for o in OPTIONS:
    for req_name, req in (("as written", None), ("level walking, dyn 1.2", CASE_LEVEL)):
        m_robot = M_FIXED + 12 * o["m_fk"] + 6 * o["m_yaw"]
        if req is None:
            need = {d: C[d] * m_robot for d in ("femur", "knee", "yaw")}
        else:
            need = {d: req[d] * (m_robot + PAYLOAD) for d in ("femur", "knee", "yaw")}
        T_fk = o["T_motor_fk"] * o["ratio_fk"] * ETA_CYC * ETA_CAP
        T_yaw = o["T_motor_yaw"] * o["ratio_yaw"] * ETA_CYC
        margins = dict(femur=T_fk / need["femur"], knee=T_fk / need["knee"], yaw=T_yaw / max(need["yaw"], 1e-6))
        for q in (20, 100):
            rows.append(dict(option=o["name"], motor=o["motor"], family=o["family"], requirement=req_name, qty=q, m_unit=o["m_fk"], m_robot=m_robot,
                             T_joint=T_fk, need_femur=need["femur"], need_knee=need["knee"], margin=min(margins.values()), margins=margins,
                             cost=o["cost_fn"](q), closes=min(margins.values()) >= 1.0, note=o["note"]))

json.dump(dict(outrunner=OUT, rows=rows), open(os.path.join(ROOT, "hw", "stator", "cost_search.json"), "w"), indent=1)

# ---- figure: cost against robot mass, closing options filled -------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.6))
mk = {"pcb": "o", "ots": "s"}
for r in rows:
    if r["requirement"] != "as written":
        continue
    col = "#0f9b8e" if r["qty"] == 20 else "#d98c3a"
    ax.scatter(r["m_robot"], r["cost"], marker=mk[r["family"]], s=110, facecolors=col if r["closes"] else "none", edgecolors=col, lw=1.6)
    ax.annotate(f"{r['option']}\n{'closes' if r['closes'] else 'does not close'}: margin {r['margin']:.2f}", (r["m_robot"], r["cost"]), (6, 6),
                textcoords="offset points", fontsize=7)
ax.scatter([], [], marker="o", color="#0f9b8e", label="20 units (filled = closes the requirement as written)")
ax.scatter([], [], marker="o", color="#d98c3a", label="100 units")
ax.scatter([], [], marker="s", color="#555", facecolors="none", label="OTS outrunner options")
ax.set_xlabel("robot mass at the fixed point (kg)"); ax.set_ylabel("actuator unit cost ($, BOM-based estimate)")
ax.set_title("Cost search: every motor/quantity option at the robot mass it implies (requirement as written)", fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "design", "actuator", "cost-search.png"), dpi=110)

if __name__ == "__main__":
    print(f"{OUT['name']}: Kt {OUT['Kt']:.3f} N·m/A, I_cont {OUT['I_cont']:.0f} A at {OUT['R_th']} K/W, T_cont {OUT['T_cont']:.2f} N·m, no-load {OUT['n_noload']:.0f} rpm")
    print(f"PCB 2-stator v2 motor: {PCB_T2:.2f} N·m; 1-stator 3 oz: {PCB_T1_3OZ:.2f}")
    for r in rows:
        print(f"  {r['option']:42s} {r['requirement']:22s} q{r['qty']:>3}: unit {r['m_unit']:.2f} kg, robot {r['m_robot']:.0f} kg, joint {r['T_joint']:.0f} vs {r['need_knee']:.0f} N·m, margin {r['margin']:.2f} {'OK' if r['closes'] else '--'}, ${r['cost']:.0f}")

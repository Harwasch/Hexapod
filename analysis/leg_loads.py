#!/opt/hw-py/bin/python
"""Leg assembly: mass roll-up, closure, structural loads, rope and bearing
margins, and the range-of-motion coverage of the sizing workspaces, for the
leg built by cad/leg/leg.py.

    /opt/hw-py/bin/python analysis/leg_loads.py

Reads cad/leg/leg.json (masses, geometry, sections, clearances), hw/stator/
closure.json (fixed mass, torque per kilogram), hw/stator/cost_search.json
(the outrunner and its unit), hw/stator/capstan.json (the rope), and the
sizing model (analysis/hexapod_model.py, leg3d.py) for the load cases and the
workspaces.  Writes hw/leg/leg_loads.json and docs/design/leg/leg-loads.png.

Design load (the task's definition): three legs down, dyn 1.5, on the 30 deg
slope with 1 m/s^2 (the continuous walking case of closure.py), times a 1.5
drop factor; and the stumble case (two legs, dyn 3.0) as the second candidate.
Structural checks use whichever is worse.  Joint torques come from closure's
torque-per-kilogram at the robot mass this leg implies, so the loop is closed
the same way the cost search closed it.

Every material number here is a handbook value, marked as such in the JSON;
every bearing number is from the sheet in docs/reference.
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
import cycloid as cy                     # noqa: E402
import hexapod_model as hm               # noqa: E402
import leg3d                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEG = json.load(open(os.path.join(ROOT, "cad", "leg", "leg.json")))
CL = json.load(open(os.path.join(ROOT, "hw", "stator", "closure.json")))
CS = json.load(open(os.path.join(ROOT, "hw", "stator", "cost_search.json")))
CAP = json.load(open(os.path.join(ROOT, "hw", "stator", "capstan.json")))
OUT_JSON = os.path.join(ROOT, "hw", "leg", "leg_loads.json")
OUT_FIG = os.path.join(ROOT, "docs", "design", "leg", "leg-loads.png")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
G = hm.G

# ---- materials (handbook, not datasheet) ------------------------------------------------------
MAT = {
    "6061-T6": dict(E=69e3, sy=240.0, note="handbook: yield 240-276 MPa, taken at 240 as the design allowable (HAZ at welds would be lower; this design is bolted/bonded)"),
    "42CrMo4 QT": dict(E=207e3, sy=700.0, note="handbook: yield 650-900 MPa QT, taken at 700; shear allowable 0.58 sy"),
    "bolt 8.8": dict(sut=800.0, note="ISO 898-1 class 8.8: UTS 800 MPa; shear strength taken as 0.6 UTS on the shank area"),
    "rubber pad": dict(note="60 Shore A natural rubber, replaceable, bonded to a bolt-on aluminium disc"),
}
BEARINGS = {  # kN, from docs/reference (manifest.yaml); PTI's HK2512 row is lower than NTN's
    "HK2512": dict(Cr=11.8, C0r=16.3, src="ntn-hk2512"),
    "HK3012": dict(Cr=11.5, C0r=17.3, src="pti-hk-series"),
    "HK3512": dict(Cr=12.45, C0r=20.25, src="pti-hk-series"),
    "HK4012": dict(Cr=13.3, C0r=23.1, src="pti-hk-series"),
    "RB7013": dict(C=19.4, C0=27.7, dp=84.0, mass_kg=0.35, src="thk-cross-roller-ring-382-5E"),
}
ROPE = CAP["geometry"]["rope"]                       # Marlow D12 Max 78, 5 mm: 29.2 kN min spliced
SF_ROPE_CONT, SF_ROPE_PEAK = 5.0, 3.0                # capstan.py's rules

# ---- the leg as built ---------------------------------------------------------------------------
TR = LEG["transmission"]
R_DRUM, R_SECTOR, R_LINK = TR["r_drum"], TR["r_sector"], TR["r_link"]
RATIO = TR["capstan_ratio"]
ETA_CAP, ETA_CYC, ETA_BELT = 0.97, 0.90, 0.98        # capstan.py / closure.py / the 1:1 HTD belt between motor and eccentric (estimate)
LOBES = LEG["body"]["lobes"]
MOTOR = CS["outrunner"]
T_MOTOR = MOTOR["T_cont"]                            # 2.63 N·m heat-sunk at the ASSUMED 1.2 K/W
T_MOTOR_PEAK = 3.0 * T_MOTOR                         # 2 s burst at 3x the continuous current (thermal mass), an assumption to be measured
m_leg = LEG["leg_total_g"] / 1000                    # three units + everything below the floor plate
m_struct = LEG["leg_structure_g"] / 1000
m_units = 3 * LEG["unit_mass_g"] / 1000 + sum(LEG["unit_extras_g"].values()) / 1000

# ---- closure at this leg's mass ------------------------------------------------------------------
M_FIXED = CL["m_fixed"]                              # 29.18 kg: body, legs (6 x 1.2), batteries, electronics, margin
LEG_ALLOWANCE = hm.MASS.legs / 6                     # 1.2 kg per leg of structure in the budget the cost search closed on
CS_ROW = [r for r in CS["rows"] if r["option"].startswith("1 x 8318") and r["requirement"] == "as written" and r["qty"] == 20][0]
m_leg_cost_search = (CS_ROW["m_robot"] - M_FIXED) / 6 + LEG_ALLOWANCE      # what the cost search carried per leg (units + capstan + the 1.2 kg)
M_FIXED_NO_LEGS = M_FIXED - hm.MASS.legs
m_robot = M_FIXED_NO_LEGS + 6 * m_leg
C, C_PEAK = CL["torque_per_kg"], CL["torque_per_kg_peak"]
T_JOINT = {j: T_MOTOR * LOBES[j] * RATIO[j] * ETA_CYC * ETA_CAP * ETA_BELT for j in ("femur", "knee")}
T_JOINT["yaw"] = T_MOTOR * LOBES["yaw"] * ETA_CYC * ETA_BELT
T_JOINT_PEAK = {j: v * T_MOTOR_PEAK / T_MOTOR for j, v in T_JOINT.items()}
need = {j: C[j] * m_robot for j in C}
need_peak = {j: C_PEAK[j] * m_robot for j in C}
margin = {j: T_JOINT[j] / need[j] for j in C}
margin_peak = {j: T_JOINT_PEAK[j] / need_peak[j] for j in C}
m_close = min(T_JOINT[j] / C[j] for j in ("femur", "knee", "yaw"))          # the robot mass at which the weakest joint just closes
m_leg_close = (m_close - M_FIXED_NO_LEGS) / 6
# levers: two motors per unit at 80:1 (cost_search's second OTS option), or the level-walking requirement
CASE_LEVEL = [c for c in CL["cases"] if c["label"].startswith("level")][0]["c_per_kg"]
LEVERS = {}
LEVERS["as built"] = dict(m_robot=m_robot, margin=margin, closes=min(margin.values()) >= 1)
m2 = m_robot + 6 * 3 * (MOTOR["mass_kg"] + 0.10)                        # a second 8318 + belt/pulley per unit, all three joints
T2 = {j: 2 * T_MOTOR * (20 if j != "yaw" else 30) * (RATIO.get(j, 1.0)) * ETA_CYC * (ETA_CAP if j != "yaw" else 1) * ETA_BELT for j in C}
LEVERS["2 x 8318 per unit, 80:1 (20-lobe x 4) / yaw 30-lobe"] = dict(m_robot=m2, margin={j: T2[j] / (C[j] * m2) for j in C}, closes=min(T2[j] / (C[j] * m2) for j in C) >= 1)
m_lvl = m_robot + hm.MASS.mission_payload
LEVERS["as built, level walking dyn 1.2 (closure.py case)"] = dict(m_robot=m_robot, margin={j: T_JOINT[j] / max(CASE_LEVEL[j] * m_lvl, 1e-6) for j in C},
                                                                  closes=min(T_JOINT[j] / max(CASE_LEVEL[j] * m_lvl, 1e-6) for j in C) >= 1)

# ---- load cases on the foot ---------------------------------------------------------------------
m_total = m_robot + hm.MASS.mission_payload
walk = hm.LoadCase("walk", m_total, legs_down=3, dyn_factor=1.5, slope_deg=30.0, accel=1.0, rating="continuous")
stumble = hm.LoadCase("stumble", m_total, legs_down=2, dyn_factor=3.0, slope_deg=0.0, accel=2.0, rating="peak")
DROP = 1.5
F_cases = {
    "walk x 1.5 drop": dict(Fz=walk.foot_force_z * DROP, Fp=walk.foot_force_prop * DROP, Fy=0.3 * walk.foot_force_prop * DROP),
    "stumble": dict(Fz=stumble.foot_force_z, Fp=stumble.foot_force_prop, Fy=0.3 * stumble.foot_force_prop),
}
F_design = max(F_cases.values(), key=lambda c: c["Fz"])
# joint torques for the structure: continuous (closure's per-kg maximum over the workspace) x drop, or the peak per kg, whichever is larger
tau_design = {j: max(DROP * C[j] * m_robot, C_PEAK[j] * m_robot) for j in C}
tau_cont = {j: C[j] * m_robot for j in C}

# ---- sections -------------------------------------------------------------------------------------
def rect_tube_props(w, h, t):
    """w across x (out of the leg plane), h in the leg plane.  Strong axis bends in the leg plane."""
    A = w * h - (w - 2 * t) * (h - 2 * t)
    I_strong = (w * h ** 3 - (w - 2 * t) * (h - 2 * t) ** 3) / 12
    I_weak = (h * w ** 3 - (h - 2 * t) * (w - 2 * t) ** 3) / 12
    return dict(A=A, I_strong=I_strong, I_weak=I_weak, c_strong=h / 2, c_weak=w / 2)


FEM = rect_tube_props(*LEG["links"]["femur_beam_section"])
TIB = rect_tube_props(*LEG["links"]["tibia_section"])
E_AL, SY_AL = MAT["6061-T6"]["E"], MAT["6061-T6"]["sy"]
L_COXA, L_FEMUR, L_TIBIA = LEG["links"]["coxa"], LEG["links"]["femur"], LEG["links"]["tibia"]
BEAM_V = 65.0                                                          # beam offset from the pivot-knee line (cad/leg/leg.py BEAM_V)

# femur beam: in-plane moment = femur joint torque (the beam carries the foot moment to the pivot); axial ~ the foot force
# resolved along the femur at the stance (45 deg) adds a P x offset moment; out-of-plane: the lateral force x its lever
Fz, Fp, Fy = F_design["Fz"], F_design["Fp"], F_design["Fy"]
P_femur = (Fz + Fp) * math.cos(math.radians(45))                      # conservative axial resultant along the femur
lever_lat_femur = math.hypot(L_FEMUR * math.cos(math.radians(45)), L_TIBIA - L_FEMUR * math.sin(math.radians(45)))   # pivot to foot in the stance
M_femur = tau_design["femur"] * 1e3 + P_femur * BEAM_V
sig_femur = M_femur * FEM["c_strong"] / FEM["I_strong"] + P_femur / FEM["A"]
sig_femur_lat = Fy * lever_lat_femur * FEM["c_weak"] / FEM["I_weak"]
sig_femur_comb = sig_femur + sig_femur_lat
Pcr_femur = math.pi ** 2 * E_AL * FEM["I_weak"] / (1.0 * (BEAM_U := 160.0)) ** 2        # pinned-pinned over the free beam length between carrier and cheeks
# tibia: cantilever from the knee; the knee torque is the in-plane root moment; lateral at the knee = Fy x tibia; axial = foot force
M_tibia = tau_design["knee"] * 1e3
sig_tibia = M_tibia * TIB["c_strong"] / TIB["I_strong"] + Fz / TIB["A"]
sig_tibia_lat = Fy * L_TIBIA * TIB["c_weak"] / TIB["I_weak"]
sig_tibia_comb = sig_tibia + sig_tibia_lat
Pcr_tibia = math.pi ** 2 * E_AL * TIB["I_weak"] / (2.0 * (L_TIBIA - 45.0)) ** 2         # fixed-free (the knee holds it), K = 2
STRUCT = {
    "femur beam 30x60x3 (in-plane bending + axial + lateral)": dict(stress_MPa=sig_femur_comb, allow_MPa=SY_AL, SF=SY_AL / sig_femur_comb,
                                                                     detail=f"M {M_femur / 1e3:.0f} N·m incl. {P_femur:.0f} N x {BEAM_V:.0f} mm offset; lateral {Fy:.0f} N x {lever_lat_femur:.0f} mm -> {sig_femur_lat:.0f} MPa"),
    "femur beam buckling (weak axis, free length 160)": dict(load_N=P_femur, crit_N=Pcr_femur, SF=Pcr_femur / P_femur),
    "tibia tube 30x50x3 (root bending + axial + lateral)": dict(stress_MPa=sig_tibia_comb, allow_MPa=SY_AL, SF=SY_AL / sig_tibia_comb,
                                                                 detail=f"M {M_tibia / 1e3:.0f} N·m (knee design torque); lateral {Fy:.0f} N x {L_TIBIA:.0f} mm -> {sig_tibia_lat:.0f} MPa"),
    "tibia buckling (fixed-free, K = 2)": dict(load_N=Fz, crit_N=Pcr_tibia, SF=Pcr_tibia / Fz),
}

# ---- transmission loads ---------------------------------------------------------------------------
# capstan ropes: tension = joint torque / (eta_cap x r_sector); the drum sees the same rope
def rope_sf(F):
    return ROPE["break_kN"] * 1e3 / F


ROPES = {}
for j in ("femur", "knee"):
    Fc = tau_cont[j] / (ETA_CAP * R_SECTOR[j] * 1e-3)
    Fpk = tau_design[j] / (ETA_CAP * R_SECTOR[j] * 1e-3)
    ROPES[f"{j} capstan (drum r {R_DRUM[j]:.0f} -> sector r {R_SECTOR[j]:.0f})"] = dict(
        F_cont_N=Fc, F_peak_N=Fpk, SF_cont=rope_sf(Fc), SF_peak=rope_sf(Fpk), ok=rope_sf(Fc) >= SF_ROPE_CONT and rope_sf(Fpk) >= SF_ROPE_PEAK,
        D_over_d=2 * R_DRUM[j] / ROPE["d_mm"], pretension_N=0.15 * Fpk, drum_torque_peak_Nm=tau_design[j] / (ETA_CAP * RATIO[j]))
Fc_link, Fp_link = tau_cont["knee"] / (R_LINK * 1e-3), tau_design["knee"] / (R_LINK * 1e-3)
ROPES[f"knee link loop (1:1, pulleys r {R_LINK:.0f}, one loop on the +x side)"] = dict(
    F_cont_N=Fc_link, F_peak_N=Fp_link, SF_cont=rope_sf(Fc_link), SF_peak=rope_sf(Fp_link), ok=rope_sf(Fc_link) >= SF_ROPE_CONT and rope_sf(Fp_link) >= SF_ROPE_PEAK,
    D_over_d=2 * R_LINK / ROPE["d_mm"], pretension_N=0.15 * Fp_link)
# rope stiffness -> joint wind-up (capstan.py's model: 60 GPa on 60 % of the circle, two runs)
A_rope = math.pi * (ROPE["d_mm"] / 2) ** 2 * 0.6
run_len = TR["run_lengths_stance_mm"]
WINDUP = {}
for j in ("femur", "knee"):
    k_run = ROPE["E_eff_GPa"] * 1e3 * A_rope / run_len[f"{j}_A"]
    k_joint = 2 * k_run * (R_SECTOR[j] * 1e-3) ** 2 * 1e3
    WINDUP[j] = dict(k_joint_Nm_per_rad=k_joint, windup_deg_at_cont=math.degrees(tau_cont[j] * RATIO[j] / (k_joint * RATIO[j])))
k_run = ROPE["E_eff_GPa"] * 1e3 * A_rope / L_FEMUR
k_link = 2 * k_run * (R_LINK * 1e-3) ** 2 * 1e3
WINDUP["knee link"] = dict(k_joint_Nm_per_rad=k_link, windup_deg_at_cont=math.degrees(tau_cont["knee"] / k_link))
WINDUP["knee total (crank + link in series)"] = dict(k_joint_Nm_per_rad=1 / (1 / WINDUP["knee"]["k_joint_Nm_per_rad"] + 1 / k_link),
                                                     windup_deg_at_cont=WINDUP["knee"]["windup_deg_at_cont"] + WINDUP["knee link"]["windup_deg_at_cont"])

# the knee's 40-lobe cycloid on the bigger pin circle of the hollow module (R 48.5: module r 55 - 4 wall - 2.5)
T_cyc_knee = (tau_cont["knee"] / (RATIO["knee"] * ETA_CAP), tau_design["knee"] / (RATIO["knee"] * ETA_CAP))
T_cyc_femur = (tau_cont["femur"] / (RATIO["femur"] * ETA_CAP), tau_design["femur"] / (RATIO["femur"] * ETA_CAP))
CYC = {"knee 40-lobe, R 48.5 (hollow module)": cy.design(LOBES["knee"], *T_cyc_knee, R=48.5),
       "femur 25-lobe, R 43.5": cy.design(LOBES["femur"], *T_cyc_femur),
       "yaw 40-lobe, R 53.5 (hollow module)": cy.design(LOBES["yaw"], tau_cont["yaw"], tau_design["yaw"], R=53.5)}
for k, d in CYC.items():
    d["ecc_bearing"] = "HK3512" if "knee" in k else ("HK4012" if "yaw" in k else "HK2512")
    d["ecc_static_SF"] = BEARINGS[d["ecc_bearing"]]["C0r"] * 1e3 / d["F_ecc"]

# ---- pivots: shafts, bearings, keys, bolt groups --------------------------------------------------
PIN_D, PIN_BORE = LEG["links"]["pin_d"], LEG["links"]["pin_bore"]
J_pin = math.pi * (PIN_D ** 4 - PIN_BORE ** 4) / 32
I_pin = J_pin / 2
tau_shaft = tau_design["knee"] * 1e3 * (PIN_D / 2) / J_pin                                   # the crank shaft carries the whole knee torque to the drive pulley
# crank shaft bending: the link loop pulls on the +x overhang (x 54..66) outside the cheek bearing at x 40..52
F_link_total = Fp_link + 0.15 * Fp_link
M_shaft = F_link_total * (60.0 - 46.0)
sig_shaft = M_shaft * (PIN_D / 2) / I_pin
SY_ST = MAT["42CrMo4 QT"]["sy"]
# radial loads on the crank shaft into the two cheek bearings (HK3012): the two capstan rope resultants (both runs pull
# toward the drum: taut run + pretensioned slack run), the link loop resultant, and the foot load through the femur carrier
F_femur_rope = ROPES[[k for k in ROPES if k.startswith("femur")][0]]
F_knee_rope = ROPES[[k for k in ROPES if k.startswith("knee capstan")][0]]
R_rope_f = F_femur_rope["F_peak_N"] + F_femur_rope["pretension_N"]
R_rope_k = F_knee_rope["F_peak_N"] + F_knee_rope["pretension_N"]
R_joint = math.hypot(Fz, Fp)
R_cheeks = R_rope_f + R_rope_k + F_link_total + R_joint        # scalar sum: conservative (the directions differ)
PIVOT = {
    "crank shaft Ø30/18 42CrMo4: torsion (knee design torque)": dict(stress_MPa=tau_shaft, allow_MPa=0.58 * SY_ST, SF=0.58 * SY_ST / tau_shaft),
    "crank shaft: bending at the cheek from the link loop overhang": dict(stress_MPa=sig_shaft, allow_MPa=SY_ST, SF=SY_ST / sig_shaft, detail=f"{F_link_total:.0f} N x 14 mm"),
    "cheek bearings 2 x HK3012 (crank shaft in the coxa): static": dict(load_N=R_cheeks, C0_N=2 * BEARINGS["HK3012"]["C0r"] * 1e3, SF=2 * BEARINGS["HK3012"]["C0r"] * 1e3 / R_cheeks,
                                                                       detail=f"femur rope {R_rope_f:.0f} + crank rope {R_rope_k:.0f} + link {F_link_total:.0f} + joint {R_joint:.0f} N, summed as scalars"),
    "femur carrier bearings 2 x HK3012: static": dict(load_N=R_rope_f + R_joint, C0_N=2 * BEARINGS["HK3012"]["C0r"] * 1e3, SF=2 * BEARINGS["HK3012"]["C0r"] * 1e3 / (R_rope_f + R_joint)),
    "knee pin Ø30/18: torsion (knee design torque from the knee pulley)": dict(stress_MPa=tau_shaft, allow_MPa=0.58 * SY_ST, SF=0.58 * SY_ST / tau_shaft),
    "knee bearings 2 x HK3012 (knee pin in the femur cheeks): static": dict(load_N=F_link_total + R_joint, C0_N=2 * BEARINGS["HK3012"]["C0r"] * 1e3, SF=2 * BEARINGS["HK3012"]["C0r"] * 1e3 / (F_link_total + R_joint)),
    "knee drum bearings HK4012 + HK3012: static (rope resultant)": dict(load_N=R_rope_k, C0_N=(BEARINGS["HK4012"]["C0r"] + BEARINGS["HK3012"]["C0r"]) * 1e3, SF=(BEARINGS["HK4012"]["C0r"] + BEARINGS["HK3012"]["C0r"]) * 1e3 / R_rope_k),
    "femur drum bearings 2 x HK2512: static (rope resultant)": dict(load_N=R_rope_f, C0_N=2 * BEARINGS["HK2512"]["C0r"] * 1e3, SF=2 * BEARINGS["HK2512"]["C0r"] * 1e3 / R_rope_f),
}
# yaw output bearing: the foot moment about the hip (force x horizontal reach) plus the axial load
reach = L_COXA + L_FEMUR * math.cos(math.radians(45))
M_yaw_brg = Fz * reach * 1e-3
M0 = BEARINGS["RB7013"]["C0"] * 1e3 * BEARINGS["RB7013"]["dp"] * 1e-3 / 2
PIVOT["yaw bearing RB7013: static moment (foot force x 327 mm reach)"] = dict(load_Nm=M_yaw_brg, M0_Nm=M0, SF=M0 / M_yaw_brg, detail="M0 = C0 x dp / 2 (THK 382-5E)")
# keys and bolt groups
KEY = {}
for name, T, r_shaft, n_keys, key_len, key_w in (("crank plates x 2 on the crank shaft (8 x 7 key, 20 mm)", tau_design["knee"], PIN_D / 2, 2, 20.0, 8.0),
                                                 ("drive pulley on the crank shaft (8 x 7 key, 12 mm)", tau_design["knee"], PIN_D / 2, 1, 12.0, 8.0),
                                                 ("knee pulley on the knee pin (8 x 7 key, 12 mm)", tau_design["knee"], PIN_D / 2, 1, 12.0, 8.0),
                                                 ("tibia carrier on the knee pin (2 x 8 x 7 keys, 30 mm)", tau_design["knee"], PIN_D / 2, 2, 30.0, 8.0)):
    F = T * 1e3 / (r_shaft * n_keys)
    shear = F / (key_len * key_w)
    bearing = F / (key_len * 3.5)                                     # 7 mm key, half in the hub
    KEY[name] = dict(force_N=F, key_shear_MPa=shear, hub_bearing_MPa=bearing, SF_shear=(0.58 * SY_ST) / shear, SF_hub_bearing_6061=SY_AL / bearing)
BOLTS = {}
for name, T, n, r, d in (("femur sector plates to the carrier: 6 x M6 at r 32 (shear, per plate pair)", tau_design["femur"], 6, 32.0, 6.0),
                         ("knee crank: plates keyed; drive pulley flange 6 x M6 at r 30 (alternative to the key)", tau_design["knee"], 6, 30.0, 6.0),
                         ("coxa cheek to hub plates: 8 x M5 at ~60 mm, rope pull of both capstans", (R_rope_f + R_rope_k) * 0.06, 8, 60.0, 5.0)):
    F = T * 1e3 / (n * r)
    A_shank = math.pi * d ** 2 / 4
    BOLTS[name] = dict(force_per_bolt_N=F, shear_MPa=F / A_shank, SF_vs_0p6UTS=0.6 * MAT["bolt 8.8"]["sut"] / (F / A_shank),
                       plate_bearing_MPa=F / (d * 6.0), SF_plate_bearing=SY_AL / (F / (d * 6.0)))

# ---- the knee-motor speed with the parallelogram --------------------------------------------------
n_noload = MOTOR["n_noload"]
w_motor = n_noload * 2 * math.pi / 60
SPEED = dict(femur_joint_rad_s_noload=w_motor / (LOBES["femur"] * RATIO["femur"]),
             tibia_absolute_rad_s_noload=w_motor / (LOBES["knee"] * RATIO["knee"]),
             yaw_rad_s_noload=w_motor / LOBES["yaw"],
             note="the knee motor drives the tibia's ABSOLUTE angle (parallelogram): motor rate = 100 x d(tau)/dt + 40 x d(psi)/dt; the femur "
                  "motor rate = 100 x d(phi)/dt + 25 x d(psi)/dt.  Lifting the foot with a hanging tibia costs the knee motor nothing; "
                  "a knee flex with the femur fixed costs the full 3.8 rad/s the sizing asked.  Under load the motor holds ~75 % of no-load.")

# ---- workspace coverage under the joint windows -----------------------------------------------------
FR, TAUR, KL = LEG["femur_range_deg"], LEG["tau_range_deg"], LEG["knee_limits_deg"]
YAW_LIM = hm.YAW_RANGE_DEG


def joint_angles(topo, key, sol):
    q0, q1, q2 = np.degrees(sol)
    if key == "sprawl":
        phi, tau, yaw = 45 + q1, 270 + q1 + q2, q0
    else:
        phi, tau, yaw = -45 - q1, 360 - 110.7 - q2, 90 + q0
    return yaw, phi, tau, (tau - phi - 180) % 360


COVER = {}
C_REACH = {}
walk_lc = hm.LoadCase("walk", 1.0, legs_down=3, dyn_factor=1.5, slope_deg=30.0, accel=1.0, rating="continuous")   # per kg on the feet
for topo, key in ((leg3d.CHOSEN, "sprawl"), (leg3d.MAMMAL_MODE, "mammal")):
    for wsn, ws in (("routine", leg3d.ROUTINE), ("stumble", leg3d.STUMBLE)):
        nf, q = topo.neutral_foot, np.zeros(3)
        n_ik = n_ok = 0
        tmax_all, tmax_reach = np.zeros(3), np.zeros(3)
        blocked = {"femur": 0, "tau": 0, "knee": 0, "yaw": 0}
        for d in ws.grid():
            sol = topo.ik(nf + d, q0=q)
            if sol is None:
                sol = topo.ik(nf + d)
            if sol is None:
                continue
            q = sol
            n_ik += 1
            yaw, phi, tau, th = joint_angles(topo, key, sol)
            t = np.max([topo.torques(sol, F) for F in leg3d.force_set(walk_lc.foot_force_z, walk_lc.foot_force_prop)], axis=0)
            tmax_all = np.maximum(tmax_all, t)
            ok = True
            for nm, v, lo, hi in (("femur", phi, FR[0], FR[1]), ("tau", tau, TAUR[0], TAUR[1]), ("knee", th, KL[0], KL[1]), ("yaw", yaw, -YAW_LIM, YAW_LIM)):
                if not lo <= v <= hi:
                    blocked[nm] += 1
                    ok = False
            if ok:
                n_ok += 1
                tmax_reach = np.maximum(tmax_reach, t)
        COVER[f"{key} {wsn}"] = dict(ik_reached=n_ik, of=int(np.prod(ws.n)), within_joint_windows=n_ok, blocked_by=blocked,
                                     torque_per_kg_all=dict(zip(("yaw", "femur", "knee"), tmax_all.round(3).tolist())),
                                     torque_per_kg_reachable=dict(zip(("yaw", "femur", "knee"), tmax_reach.round(3).tolist())))
# the requirement over what the leg can actually reach (per kg on the feet -> per kg of robot as closure does: x (m + payload) / m)
c_reach = {j: max(COVER[k]["torque_per_kg_reachable"][j] for k in COVER if k.endswith("routine")) for j in ("yaw", "femur", "knee")}
c_all = {j: max(COVER[k]["torque_per_kg_all"][j] for k in COVER if k.endswith("routine")) for j in ("yaw", "femur", "knee")}
need_reach = {j: c_reach[j] * (m_robot + hm.MASS.mission_payload) for j in c_reach}
margin_reach = {j: T_JOINT[j] / max(need_reach[j], 1e-6) for j in c_reach}

# ---- roll-up and output -----------------------------------------------------------------------------
GROUPS = LEG["group_mass_g"]
out = dict(
    inputs=dict(leg_json="cad/leg/leg.json", motor=MOTOR["name"], T_motor_cont=T_MOTOR, T_motor_peak_assumed=T_MOTOR_PEAK, R_th_assumed=MOTOR["R_th"],
                eta=dict(cycloid=ETA_CYC, capstan=ETA_CAP, belt=ETA_BELT), materials=MAT, bearings=BEARINGS, rope=ROPE, drop_factor=DROP),
    mass=dict(leg_structure_kg=m_struct, units_and_extras_kg=m_units, leg_total_kg=m_leg, groups_g=GROUPS,
              cost_search_leg_kg=m_leg_cost_search, cost_search_structure_allowance_kg=LEG_ALLOWANCE + 2 * 0.3,
              over_cost_search_kg=m_leg - m_leg_cost_search, m_fixed_no_legs_kg=M_FIXED_NO_LEGS),
    closure=dict(m_robot=m_robot, m_robot_cost_search=CS_ROW["m_robot"], torque_per_kg=C, torque_per_kg_peak=C_PEAK, joint_gives=T_JOINT, joint_gives_peak=T_JOINT_PEAK,
                 need=need, need_peak=need_peak, margin=margin, margin_peak=margin_peak, closes=min(margin.values()) >= 1.0,
                 m_robot_that_closes=m_close, leg_mass_that_closes_kg=m_leg_close, levers=LEVERS,
                 requirement_over_reachable_set=dict(c_per_kg_on_feet_all=c_all, c_per_kg_on_feet_reachable=c_reach, need=need_reach, margin=margin_reach,
                                                     closes=min(margin_reach.values()) >= 1.0)),
    load_cases=dict(m_total=m_total, cases=F_cases, design=F_design, joint_torque_design=tau_design, joint_torque_cont=tau_cont),
    structure=STRUCT, ropes=ROPES, windup=WINDUP, cycloid=CYC, pivots=PIVOT, keys=KEY, bolts=BOLTS, speed=SPEED,
    coupling=TR["coupling"], workspace_coverage=COVER, joint_windows=dict(femur=FR, tau=TAUR, knee=KL, yaw=[-YAW_LIM, YAW_LIM]),
    clearances=LEG["clearances"] if "skipped" not in LEG["clearances"] else "sweep not run",
)
json.dump(out, open(OUT_JSON, "w"), indent=1, default=float)

# ---- figure -------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
ax = axes[0, 0]
names = [g for g in GROUPS if g != "body_stub"]
vals = [GROUPS[g] / 1000 for g in names]
ax.barh(names, vals, color="#d98c3a")
ax.barh(["3 x actuator unit + extras"], [m_units], color="#555b61")
ax.axvline(m_leg_cost_search, color="#b03a2e", ls="--", lw=1)
ax.text(m_leg_cost_search + 0.1, 0.5, f"cost search: {m_leg_cost_search:.1f} kg per leg\n(units {3 * LEG['unit_mass_g'] / 1000:.2f} + capstan 0.6 + structure {LEG_ALLOWANCE:.1f})", fontsize=7, color="#b03a2e")
for i, v in enumerate(vals + [m_units]):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlabel("kg"); ax.set_title(f"Mass roll-up: {m_leg:.2f} kg per leg ({m_struct:.2f} structure + {m_units:.2f} units); robot {m_robot:.0f} kg", fontsize=10)
ax.grid(axis="x", alpha=0.3)
ax = axes[0, 1]
x = np.arange(3); J = ("femur", "knee", "yaw")
ax.bar(x - 0.3, [need[j] for j in J], 0.3, color="#b03a2e", label=f"needs at {m_robot:.0f} kg (continuous, as written)")
ax.bar(x, [need_reach[j] for j in J], 0.3, color="#e0a090", label="needs over the poses this leg reaches")
ax.bar(x + 0.3, [T_JOINT[j] for j in J], 0.3, color="#0f9b8e", label="unit gives (2.63 N·m x ratio x η)")
for i, j in enumerate(J):
    ax.text(i, max(need[j], T_JOINT[j]) + 6, f"margin {margin[j]:.2f}\n({margin_reach[j]:.2f} reachable)", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(J); ax.set_ylabel("N·m"); ax.legend(fontsize=7, loc="upper right"); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"Closure: {'closes' if out['closure']['closes'] else 'does not close'} as written; closes at {m_close:.0f} kg robot = {m_leg_close:.2f} kg per leg", fontsize=10)
ax = axes[1, 0]
rows = [(k.split(" (")[0][:44], v["SF"]) for k, v in STRUCT.items()] + [(k.split(":")[0][:44], v["SF"]) for k, v in PIVOT.items()] + \
       [(k[:44], v["SF_shear"]) for k, v in KEY.items()] + [(k.split(":")[0][:44], v["SF_vs_0p6UTS"]) for k, v in BOLTS.items()] + \
       [(f"{k.split(',')[0]} Hertz", cy.SIGMA_H_ALLOW / v["sigma_peak"]) for k, v in CYC.items()] + [(f"{k.split(',')[0]} ecc. bearing static", v["ecc_static_SF"]) for k, v in CYC.items()]
lab, sf = zip(*rows)
cols = ["#0f9b8e" if s >= 1.5 else ("#d98c3a" if s >= 1.0 else "#b03a2e") for s in sf]
ax.barh(range(len(lab)), np.clip(sf, 0, 12), color=cols)
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=6.5); ax.invert_yaxis()
for i, s in enumerate(sf):
    ax.text(min(s, 12) + 0.1, i, f"{s:.1f}", va="center", fontsize=7)
ax.axvline(1.5, color="#555", ls="--", lw=0.8); ax.set_xlim(0, 13.5); ax.set_xlabel("safety factor at the design load (clipped at 12)")
ax.set_title(f"Structure, pivots, keys, bolts, cycloids at foot {Fz:.0f} N / torques femur {tau_design['femur']:.0f}, knee {tau_design['knee']:.0f} N·m", fontsize=9.5)
ax = axes[1, 1]
rn = list(ROPES)
x = np.arange(len(rn))
ax.bar(x - 0.2, [ROPES[k]["F_cont_N"] / 1e3 for k in rn], 0.4, color="#0f9b8e", label="continuous")
ax.bar(x + 0.2, [ROPES[k]["F_peak_N"] / 1e3 for k in rn], 0.4, color="#d98c3a", label="design (peak / drop)")
ax.axhline(ROPE["break_kN"] / SF_ROPE_CONT, color="#0f9b8e", ls="--", lw=1, label=f"break / {SF_ROPE_CONT:.0f}")
ax.axhline(ROPE["break_kN"] / SF_ROPE_PEAK, color="#d98c3a", ls="--", lw=1, label=f"break / {SF_ROPE_PEAK:.0f}")
for i, k in enumerate(rn):
    ax.text(i, ROPES[k]["F_peak_N"] / 1e3 + 0.2, f"SF {ROPES[k]['SF_cont']:.1f} / {ROPES[k]['SF_peak']:.1f}\nD/d {ROPES[k]['D_over_d']:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([k.split(" (")[0] for k in rn], fontsize=8); ax.set_ylabel("kN"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"{ROPE['name']}: {ROPE['break_kN']:.1f} kN spliced; wind-up femur {WINDUP['femur']['windup_deg_at_cont']:.2f}°, knee {WINDUP['knee total (crank + link in series)']['windup_deg_at_cont']:.2f}° at continuous", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=110)

if __name__ == "__main__":
    print(f"leg {m_leg:.2f} kg ({m_struct:.2f} structure + {m_units:.2f} units/extras); cost search carried {m_leg_cost_search:.2f} -> robot {m_robot:.1f} kg (cost search {CS_ROW['m_robot']:.1f})")
    for j in ("femur", "knee", "yaw"):
        print(f"  {j}: gives {T_JOINT[j]:.0f} / needs {need[j]:.0f} N·m -> margin {margin[j]:.2f} (peak {margin_peak[j]:.2f}); over the reachable set needs {need_reach[j]:.0f} -> {margin_reach[j]:.2f}")
    print(f"  closes: {out['closure']['closes']}; closes at robot {m_close:.1f} kg = {m_leg_close:.2f} kg per leg")
    for k, v in LEVERS.items():
        print(f"  lever {k}: robot {v['m_robot']:.0f} kg, margins {', '.join(f'{j} {m:.2f}' for j, m in v['margin'].items())} -> {'closes' if v['closes'] else 'no'}")
    print(f"design foot load {Fz:.0f} N (from {[k for k, v in F_cases.items() if v is F_design][0]}), torques {tau_design}")
    for sect in (STRUCT, PIVOT, KEY, BOLTS):
        for k, v in sect.items():
            sf = v.get("SF", v.get("SF_shear", v.get("SF_vs_0p6UTS")))
            print(f"  {k}: SF {sf:.2f}")
    for k, v in ROPES.items():
        print(f"  {k}: {v['F_cont_N'] / 1e3:.2f} / {v['F_peak_N'] / 1e3:.2f} kN, SF {v['SF_cont']:.1f} / {v['SF_peak']:.1f}, {'ok' if v['ok'] else 'NOT ok'}")
    for k, v in CYC.items():
        print(f"  {k}: pin Ø{2 * v['r_pin']:.0f}, e {v['e']:.2f}, Hertz {v['sigma_peak']:.0f} MPa (allow {cy.SIGMA_H_ALLOW:.0f}), F_ecc {v['F_ecc'] / 1e3:.1f} kN -> {v['ecc_bearing']} static SF {v['ecc_static_SF']:.1f}")
    for k, v in COVER.items():
        print(f"  {k}: IK {v['ik_reached']}/{v['of']}, within windows {v['within_joint_windows']}, blocked by {v['blocked_by']}")
    print("speed", SPEED)

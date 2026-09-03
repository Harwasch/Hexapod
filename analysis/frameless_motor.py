#!/opt/hw-py/bin/python
"""The Wheemo WxF70x24GT frameless kit motor as the basis for the actuator, and
what a scaled-up member of the same family has to be to close the requirement.

    /opt/hw-py/bin/python analysis/frameless_motor.py

Everything about the WxF70x24GT comes from its datasheet
(docs/reference/wheemo-WxF70x24GT.pdf, filed in the manifest).  The datasheet
gives terminal quantities and no internal geometry, so this script

  1. checks the datasheet against itself three ways (Kt vs Ke, copper loss vs
     the quoted loss, Km vs Kt and R) to be sure the conventions are decoded;
  2. infers the radial build from the active mass, bracketed by the density a
     laminated-steel/copper/magnet annulus can physically have;
  3. scales it on the one law that holds for a frameless torque motor whose
     pole pitch is held constant as it grows -- torque with the square of the
     airgap diameter, copper loss with the mounting area, radial build and
     therefore mass per unit circumference unchanged;
  4. sweeps outside diameter and stack length inside the actuator can, solves
     the mass/torque fixed point for the robot, and reports which designs close
     the requirement as written and at what ratio.

The scaling law is stated rather than assumed away: at constant current
density and constant electric loading, torque goes as D_g^2 L and copper loss
as D L, so the heat flux through the stator's mounting face -- the only path a
frameless stator has -- is unchanged.  That is why the shear stress may be held
at the datasheet's own value, and it is the assumption the sensitivity sweep
attacks.

Writes hw/stator/frameless_motor.json and docs/design/actuator/frameless-*.png.
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
import sizing as sz                                   # noqa: E402
import hexapod_model as hm                            # noqa: E402
import cycloid as cy                                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "design", "actuator")

# ---------------------------------------------------------------- the datasheet
DS = dict(part="Wheemo WxF70x24GT", od_mm=70.0, len_mm=23.7, mass_g=377.0, volt=48.0,
          t_cont=1.60, t_peak=5.05, rpm_rated=2500.0, rpm_noload=2903.0, loss_cont_W=23.8,
          current_A=8.2, R_ph=0.0967, L_tt=531.8e-6, L_d=272.6e-6,
          ke_mV_per_rpm=9.55, km=0.3621, kt_per_A_peak=0.1367, temp_range_C=(-40, 85))

# --- check 1: Kt against Ke.  For a sinusoidal three-phase machine the torque
# per amp of PEAK phase current is 1.5x the peak phase back-EMF constant.
ke_Vs_per_rad = DS["ke_mV_per_rpm"] * 1e-3 * 60 / (2 * math.pi)
kt_from_ke = 1.5 * ke_Vs_per_rad
# --- check 2: the quoted current is rms, so torque = Kt_peak * sqrt(2) * I_rms
t_from_kt = DS["kt_per_A_peak"] * math.sqrt(2) * DS["current_A"]
# --- check 3: copper loss and Km
P_CU = 3 * DS["current_A"] ** 2 * DS["R_ph"]
kt_rms = DS["t_cont"] / DS["current_A"]                       # N.m per A rms, the useful form
km_from_kt = kt_rms / math.sqrt(3 * DS["R_ph"])
P_FE = DS["loss_cont_W"] - P_CU                               # what is left is iron + windage at 2500 rpm
# --- check 4: no-load speed.  With space-vector modulation the largest phase-
# to-neutral fundamental a wye winding can see is V_dc/sqrt(3) peak, so the
# no-load speed is where the peak phase back-EMF reaches it.
e_peak_at_noload = DS["ke_mV_per_rpm"] * 1e-3 * DS["rpm_noload"]
v_svpwm_peak = DS["volt"] / math.sqrt(3)
CHECKS = dict(e_peak_at_noload=e_peak_at_noload, v_svpwm_peak=v_svpwm_peak,
              noload_err=e_peak_at_noload / v_svpwm_peak - 1, kt_from_ke=kt_from_ke, kt_datasheet=DS["kt_per_A_peak"], kt_err=kt_from_ke / DS["kt_per_A_peak"] - 1,
              t_from_kt=t_from_kt, t_datasheet=DS["t_cont"], t_err=t_from_kt / DS["t_cont"] - 1,
              km_from_kt=km_from_kt, km_datasheet=DS["km"], km_err=km_from_kt / DS["km"] - 1,
              P_cu=P_CU, P_fe_implied=P_FE, kt_rms=kt_rms)

# ------------------------------------------------- inferring the radial build
# Active mass fills the annulus between the outside diameter and the bore.  A
# stator+rotor annulus (laminations at ~0.97 stacking, copper at 40-50 % slot
# fill, sintered NdFeB, slot openings and end-winding air) sits between about
# 5.8 and 7.2 g/cm^3.  That brackets the bore, and with it the total radial
# build t_tot = (OD - bore)/2, which is the quantity the scaling needs.
RHO_LO, RHO_HI = 5.8e3, 7.2e3          # kg/m^3, the physically defensible bracket


def bore_for_density(rho):
    """Bore (m) that makes the annulus mass come out at the datasheet value."""
    v = DS["mass_g"] * 1e-3 / rho
    od = DS["od_mm"] * 1e-3
    inner_sq = od ** 2 - 4 * v / (math.pi * DS["len_mm"] * 1e-3)
    return math.sqrt(max(inner_sq, 0.0))


BORE_LO, BORE_HI = bore_for_density(RHO_HI), bore_for_density(RHO_LO)   # dense -> small annulus -> big bore
T_TOT_LO = (DS["od_mm"] * 1e-3 - BORE_LO) / 2
T_TOT_HI = (DS["od_mm"] * 1e-3 - BORE_HI) / 2
T_TOT = 0.5 * (T_TOT_LO + T_TOT_HI)                    # nominal total radial build, m
RHO_EFF = DS["mass_g"] * 1e-3 / ((math.pi / 4) * ((DS["od_mm"] * 1e-3) ** 2 - (DS["od_mm"] * 1e-3 - 2 * T_TOT) ** 2) * DS["len_mm"] * 1e-3)

# f = the share of the radial build that lies outside the airgap.  0.5 splits it
# evenly; an inner-rotor machine with a deep stator is higher, an outer-rotor
# machine lower.  The datasheet does not say which, so the sweep carries all three.
F_NOM, F_LO, F_HI = 0.50, 0.35, 0.65


def gap_diameter(od, t_tot=T_TOT, f=F_NOM):
    return od - 2 * f * t_tot


def shear_stress(t_cont, od, length, t_tot=T_TOT, f=F_NOM):
    """Airgap shear stress (Pa) implied by a torque: T = sigma * (pi/2) * D_g^2 * L."""
    d_g = gap_diameter(od, t_tot, f)
    return t_cont / ((math.pi / 2) * d_g ** 2 * length)


SIGMA = {k: shear_stress(DS["t_cont"], DS["od_mm"] * 1e-3, DS["len_mm"] * 1e-3, f=v) for k, v in
         (("nom", F_NOM), ("lo_f", F_LO), ("hi_f", F_HI))}
SIGMA_PEAK = SIGMA["nom"] * DS["t_peak"] / DS["t_cont"]

# ------------------------------------------------------------ the scaling law
# Hold: pole pitch (so the radial build, and the end-winding length per coil,
# do not change), current density, electric loading, slot proportions.  Then
#   torque  T   = sigma * (pi/2) * D_g^2 * L          (D_g = OD - 2 f t_tot)
#   copper  Pcu = Pcu_0 * (D L) / (D_0 L_0)           (constant flux at the mount)
#   mass    m   = rho_eff * (pi/4) (OD^2 - bore^2) L  (bore = OD - 2 t_tot)
#   iron    Pfe = Pfe_0 * (D/D_0)*(L/L_0) * (p/p_0)^1.6 * (n/n_0)^1.6, p ~ D
RHO_CU_TEMPCO = 0.00393        # 1/K, copper


def motor(od_mm, len_mm, f=F_NOM, t_tot=T_TOT):
    od, L = od_mm * 1e-3, len_mm * 1e-3
    od0, L0 = DS["od_mm"] * 1e-3, DS["len_mm"] * 1e-3
    d_g = gap_diameter(od, t_tot, f)
    bore = od - 2 * t_tot
    sigma = shear_stress(DS["t_cont"], od0, L0, t_tot, f)
    t_cont = sigma * (math.pi / 2) * d_g ** 2 * L
    t_peak = t_cont * DS["t_peak"] / DS["t_cont"]
    p_cu = P_CU * (od * L) / (od0 * L0)
    mass = RHO_EFF * (math.pi / 4) * (od ** 2 - bore ** 2) * L
    km = t_cont / math.sqrt(p_cu)
    return dict(od_mm=od_mm, len_mm=len_mm, d_gap_mm=d_g * 1e3, bore_mm=bore * 1e3, sigma_Pa=sigma,
                T_cont=t_cont, T_peak=t_peak, P_cu=p_cu, mass_kg=mass, Km=km,
                T_per_kg=t_cont / mass, f=f)


def iron_loss(m, rpm, poles_ref=14):
    """Iron loss scaled from the datasheet's implied 4 W at 2500 rpm: mass with
    D*L, electrical frequency with pole count (~D) and speed, loss with f^1.6."""
    d_ratio = m["od_mm"] / DS["od_mm"]
    mass_ratio = d_ratio * (m["len_mm"] / DS["len_mm"])
    f_ratio = d_ratio * (rpm / DS["rpm_rated"])
    return max(P_FE, 0.0) * mass_ratio * f_ratio ** 1.6


# ------------------------------------------------------- the requirement side
C = {d: float(sz.DOF_CONT[d]) / hm.MASS.robot for d in ("yaw", "femur", "knee")}
C_PEAK = {d: float(sz.DOF_PEAK[d]) / hm.MASS.robot for d in ("yaw", "femur", "knee")}
M_FIXED = hm.MASS.robot - hm.MASS.actuators            # 29.2 kg: body, legs, batteries, electronics, margin
SWING = dict(sz.DOF_SWING)
V_BUS = 48.0
RING_WALL, PIN_CLEAR = 4.0, 2.5      # mm, ring wall and pin clearance inside the bore (cycloid.py's convention)
SUSTAINED_W = 300.0                                   # the robot's whole thermal budget (docs 08 s6)
ETA_CYC, ETA_CAP = 0.90, 0.97

# What the actuator can hold.  The can is OD 192 with a 4 mm wall; a frameless
# stator is bonded or shrunk into a carrier, so the motor OD tops out ~14 mm
# inside the can.  The bore has to clear the cycloid's pin ring (r 46 -> D 92)
# when the motor sits concentric with the reducer instead of stacked above it.
OD_MAX, BORE_MIN_CONCENTRIC = 170.0, 100.0
M_DRUM = 0.30                                         # kg, the capstan drum, when there is one

# Round 14c.  The first pass took the reducer at 1.25 kg and the housing at
# 0.55 kg from the CAD mass table of the Ø192 can with a Ø100 bore -- and then
# moved the cycloid's pin circle from r 43.5 to r 59.3 and changed the can.
# Both masses scale with that, so the numbers were wrong: cad/actuator/frameless.py
# builds the unit at 4.09 kg against the 2.84 kg assumed.  The mass model below
# is anchored on that CAD point and scales, so the sweep is honest at every size:
#   reducer + bearings + rotor carrier ~ the disc area, r_pin^2
#   housing                            ~ the can's surface, OD x height
#   height                             = the reducer's own axial stack + the motor
CAD = j_cad = None
_p = os.path.join(ROOT, "cad", "actuator", "frameless.json")
if os.path.exists(_p):
    CAD = json.load(open(_p))
CAD_R_PIN, CAD_OD, CAD_H = 59.3, 172.0, 49.7
CAD_ROT = (CAD["mass_by_group_g"]["reducer"] + CAD["mass_by_group_g"]["bearings"] + CAD["mass_by_group_g"]["rotor"]) / 1e3 if CAD else 2.054
CAD_HOUSE = CAD["mass_by_group_g"]["housing"] / 1e3 if CAD else 0.997
H_REDUCER = CAD_H - 25.0                              # mm of axial stack the reducer needs whatever the motor is


def can(od_motor_mm, len_motor_mm):
    """Housing outside diameter and unit height for a motor of this size."""
    return od_motor_mm + 12.0, len_motor_mm + H_REDUCER


def unit(m_motor, capstan, r_pin_mm=CAD_R_PIN, od_motor_mm=160.0, len_motor_mm=25.0):
    """Unit mass from the CAD point, scaled. The frameless kit still saves its own
    case and bearings -- the rotor runs on the cycloid's eccentric shaft -- but the
    reducer and the can are this design's, not the Ø192 design's."""
    od_can, h_can = can(od_motor_mm, len_motor_mm)
    m_rot = CAD_ROT * (r_pin_mm / CAD_R_PIN) ** 2
    m_house = CAD_HOUSE * (od_can / CAD_OD) * (h_can / CAD_H)
    return m_motor + m_rot + m_house + (M_DRUM if capstan else 0.0)


def closure(m, ratio_fk, ratio_yaw, capstan, m_fixed=None):
    """Fixed point: robot mass carries 12 femur/knee units and 6 yaw units."""
    eta_fk = ETA_CYC * (ETA_CAP if capstan else 1.0)
    r_pin = max(20.0, m["bore_mm"] / 2 - RING_WALL - PIN_CLEAR)
    kw = dict(r_pin_mm=r_pin, od_motor_mm=m["od_mm"], len_motor_mm=m["len_mm"])
    m_fk, m_yaw = unit(m["mass_kg"], capstan, **kw), unit(m["mass_kg"], False, **kw)
    m_robot = (M_FIXED if m_fixed is None else m_fixed) + 12 * m_fk + 6 * m_yaw
    t_fk = m["T_cont"] * ratio_fk * eta_fk
    t_yaw = m["T_cont"] * ratio_yaw * ETA_CYC
    need = {d: C[d] * m_robot for d in C}
    margin = {"femur": t_fk / need["femur"], "knee": t_fk / need["knee"], "yaw": t_yaw / need["yaw"]}
    rpm_fk = SWING["femur"] * ratio_fk * 60 / (2 * math.pi)
    rpm_yaw = SWING["yaw"] * ratio_yaw * 60 / (2 * math.pi)
    # Kt is a free design variable (turns).  Its ceiling is the bus: the peak
    # phase back-EMF may not exceed V_dc/sqrt(3) at the top of the swing, and
    # Kt_rms = 1.5*sqrt(2)*Ke_peak.  Verified against the datasheet's own no-load speed.
    kt_max_fk = 1.5 * math.sqrt(2) * (V_BUS / math.sqrt(3)) / (rpm_fk * 2 * math.pi / 60)
    kt_max_yaw = 1.5 * math.sqrt(2) * (V_BUS / math.sqrt(3)) / (rpm_yaw * 2 * math.pi / 60) if rpm_yaw else float("inf")
    return dict(m_fk=m_fk, m_yaw=m_yaw, m_robot=m_robot, T_joint_fk=t_fk, T_joint_yaw=t_yaw, need=need,
                margin=margin, closes=min(margin.values()) >= 1.0, rpm_fk=rpm_fk, rpm_yaw=rpm_yaw,
                can_od_mm=can(m["od_mm"], m["len_mm"])[0], can_h_mm=can(m["od_mm"], m["len_mm"])[1],
                kt_max_rms_yaw=kt_max_yaw, I_yaw=m["T_cont"] / kt_max_yaw,
                T_at_body_budget=m["T_cont"] * math.sqrt(min(1.0, (SUSTAINED_W / 18) / m["P_cu"])),
                robot_supported=min(t_fk / C["femur"], t_fk / C["knee"], t_yaw / C["yaw"]),
                P_cu_unit=m["P_cu"], P_fe_unit=iron_loss(m, rpm_fk), kt_needed_rms=m["T_cont"] / (m["T_cont"] / kt_max_fk) if kt_max_fk else 0.0,
                kt_max_rms_at_speed=kt_max_fk, I_at_cont=m["T_cont"] / kt_max_fk if kt_max_fk > 0 else float("nan"))


# ------------------------------------------- what the reducer then has to take
# Deleting the capstan multiplies the cycloid's output torque by four.  That is
# only survivable because the frameless motor is an annulus: its bore is where
# the cycloid goes, so the pin circle moves out from r 43.5 (in the Ø100 bore of
# the PCB stator) to whatever the new bore allows, and both the pin force and the
# eccentric-bearing load fall with the radius.
HK2512_C0R, HK3012_C0R = 16300.0, 17300.0     # N, static radial ratings from the filed datasheets


def reducer(m, ratio_fk, capstan, m_robot, lobes):
    """Cycloid loads for this motor's bore, at the joint torque the fixed point asks for."""
    r_pin = max(20.0, m["bore_mm"] / 2 - RING_WALL - PIN_CLEAR)
    stage2 = (4.0 * ETA_CAP) if capstan else 1.0
    t_c, t_p = C["knee"] * m_robot / stage2, C_PEAK["knee"] * m_robot / stage2
    d = cy.design(lobes, t_c, t_p, R=r_pin)
    return dict(r_pin_circle_mm=r_pin, lobes=lobes, pitch_mm=d["pitch"], e_mm=d["e"],
                T_cyc_cont=t_c, T_cyc_peak=t_p, F_pin=d["F_pin"], F_ecc=d["F_ecc"],
                hertz_peak=d["sigma_peak"], hertz_cont=d["sigma_cont"], hertz_ok=d["ok"],
                hk2512_static_margin=HK2512_C0R / d["F_ecc"], hk3012_static_margin=HK3012_C0R / d["F_ecc"])


# ----------------------------------------------------------------- the sweep
ODS = np.arange(80.0, OD_MAX + 1, 5.0)
LENS = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
RATIOS = [("25-lobe cycloid x 4:1 capstan", 100.0, 30.0, True),
          ("20-lobe cycloid x 4:1 capstan (as designed)", 80.0, 30.0, True),
          ("25-lobe cycloid alone, no capstan", 25.0, 30.0, False),
          ("30-lobe cycloid alone, no capstan", 30.0, 30.0, False),
          ("40-lobe cycloid alone, no capstan", 40.0, 30.0, False)]
# The pick has to survive the one datasheet number we do not have: if R_ph is
# quoted cold and the winding runs at 120 C, every torque here falls 15 %.  So
# the margin a design must show is 1/0.85, not 1.0.  And the review's round-8
# decision was to REDUCE the lobe count, so a design that closes on 25 lobes
# beats one that needs 40.
MARGIN_MIN = 1.0 / math.sqrt(1 + RHO_CU_TEMPCO * 100.0)
LOBES = {"25-lobe cycloid x 4:1 capstan": 25, "20-lobe cycloid x 4:1 capstan (as designed)": 20,
         "25-lobe cycloid alone, no capstan": 25, "30-lobe cycloid alone, no capstan": 30,
         "40-lobe cycloid alone, no capstan": 40}
rows = []
for od in ODS:
    for L in LENS:
        m = motor(od, L)
        if m["bore_mm"] < BORE_MIN_CONCENTRIC:
            continue                                   # cannot sit around the reducer; would need a stacked, taller unit
        for name, r_fk, r_yaw, cap in RATIOS:
            c = closure(m, r_fk, r_yaw, cap)
            red = reducer(m, r_fk, cap, c["m_robot"], LOBES[name])
            rows.append(dict(motor=m, ratio_name=name, ratio_fk=r_fk, ratio_yaw=r_yaw, capstan=cap, reducer=red, **c))

closing = [r for r in rows if r["closes"]]
ECC_MARGIN_MIN = 1.5                    # static margin on the eccentric needle bearing at the joint's peak
robust = [r for r in rows if min(r["margin"].values()) >= 1 / MARGIN_MIN
          and r["reducer"]["hertz_ok"] and r["reducer"]["hk2512_static_margin"] >= ECC_MARGIN_MIN]
# The pick, in order: survives the cold-R case; no capstan (the leg study found
# the rope drive as drawn cannot be wound); fewest lobes; then the lightest robot.
def rank(r):
    return (r["capstan"], LOBES[r["ratio_name"]], r["m_robot"], r["motor"]["od_mm"], r["motor"]["len_mm"])


PICK = sorted(robust, key=rank)[0] if robust else (sorted(closing, key=rank)[0] if closing else None)
# Round 14c.  Once the unit mass is taken from the CAD the sweep would re-pick a
# slightly different size, but the CAD, the general-arrangement drawings and the
# reducer analysis are all of the Ø160 x 25 / 25-lobe design.  Reporting a
# different motor than the one that has been drawn would be worse than useless,
# and the re-pick is moot anyway until the leg mass is known (see the ladder
# below).  So pin the reported design to the built one when its CAD exists.
if CAD:
    _want = (CAD["motor"]["od_mm"], CAD["motor"]["length_mm"], CAD["reducer"]["lobes"])
    _match = [r for r in rows if (r["motor"]["od_mm"], r["motor"]["len_mm"], LOBES[r["ratio_name"]]) == _want and not r["capstan"]]
    if _match:
        PICK = _match[0]

# ------------------------------------------------- the yaw motor, sized on its own
YAW = None
if PICK:
    need_yaw = C["yaw"] * PICK["m_robot"] / (PICK["ratio_yaw"] * ETA_CYC) / MARGIN_MIN
    for od in ODS:
        for L in LENS:
            m = motor(od, L)
            if m["bore_mm"] >= BORE_MIN_CONCENTRIC and m["T_cont"] >= need_yaw:
                rpm = SWING["yaw"] * PICK["ratio_yaw"] * 60 / (2 * math.pi)
                kt_max = 1.5 * math.sqrt(2) * (V_BUS / math.sqrt(3)) / (rpm * 2 * math.pi / 60)
                YAW = dict(motor=m, T_needed=need_yaw, margin=m["T_cont"] * PICK["ratio_yaw"] * ETA_CYC / (C["yaw"] * PICK["m_robot"]),
                           rpm=rpm, kt_max_rms=kt_max, I=m["T_cont"] / kt_max,
                           saving_g=(PICK["motor"]["mass_kg"] - m["mass_kg"]) * 1e3)
                break
        if YAW:
            break

# ------------------------------------------------------------- the mass ladder
# Round 14c.  Three of the numbers behind the closure were assumptions standing
# in for measurements, and two have since been measured.  This is what the
# margin does as each is replaced -- it is the honest state of the design.
LEG_STRUCT_MODEL = hm.MASS.legs / 6            # kg per leg, what hexapod_model has always carried
LEG_STRUCT_CAD = None
_lp = os.path.join(ROOT, "cad", "leg", "leg.json")
if os.path.exists(_lp):
    LEG_STRUCT_CAD = json.load(open(_lp)).get("leg_structure_g", 0) / 1e3


def margin_at(m, ratio_fk, ratio_yaw, capstan, m_unit_fk, m_unit_yaw, leg_struct_kg):
    """Worst joint margin with a stated unit mass and a stated per-leg structure."""
    m_fixed = M_FIXED - hm.MASS.legs + 6 * leg_struct_kg
    m_robot = m_fixed + 12 * m_unit_fk + 6 * m_unit_yaw
    eta_fk = ETA_CYC * (ETA_CAP if capstan else 1.0)
    t_fk = m["T_cont"] * ratio_fk * eta_fk
    t_yaw = m["T_cont"] * ratio_yaw * ETA_CYC
    marg = {"femur": t_fk / (C["femur"] * m_robot), "knee": t_fk / (C["knee"] * m_robot),
            "yaw": t_yaw / (C["yaw"] * m_robot)}
    return m_robot, marg


LADDER = []
if PICK:
    m, rfk, ryw, cap = PICK["motor"], PICK["ratio_fk"], PICK["ratio_yaw"], PICK["capstan"]
    m_cad = (CAD["total_g"] / 1e3) if CAD else None
    steps = [("as published in §9.17: reducer and housing borrowed from the Ø192 design",
              2.839, 2.839, LEG_STRUCT_MODEL, "assumed"),
             ("with the unit as actually built in CAD", m_cad, (m_cad - 0.4) if m_cad else None, LEG_STRUCT_MODEL, "measured unit"),
             ("and with the leg structure the round-14 leg CAD measured", m_cad, (m_cad - 0.4) if m_cad else None, LEG_STRUCT_CAD, "measured leg")]
    for label, u_fk, u_yaw, ls, kind in steps:
        if u_fk is None or ls is None:
            continue
        mr, mg = margin_at(m, rfk, ryw, cap, u_fk, u_yaw, ls)
        LADDER.append(dict(label=label, kind=kind, m_unit_fk=u_fk, leg_struct_kg=ls, m_robot=mr,
                           margin=mg, worst=min(mg.values()), closes=min(mg.values()) >= 1.0))
    # and the break-even: how heavy may a leg's structure be before the design stops closing?
    if m_cad:
        lo, hi = 0.0, 30.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            _, mg = margin_at(m, rfk, ryw, cap, m_cad, m_cad - 0.4, mid)
            if min(mg.values()) >= 1.0:
                lo = mid
            else:
                hi = mid
        LEG_STRUCT_BREAKEVEN = lo
    else:
        LEG_STRUCT_BREAKEVEN = None

# ---------------- what would close, on the leg structure we have measured
# The constructive question. Re-sweep with the CAD-anchored unit mass AND the
# leg structure the round-14 leg CAD actually weighed, and see what size of
# frameless motor closes -- if any does inside the can.
CLOSERS = []
if CAD and LEG_STRUCT_CAD:
    m_fixed_measured = M_FIXED - hm.MASS.legs + 6 * LEG_STRUCT_CAD
    for od in ODS:
        for L in LENS:
            mm_ = motor(od, L)
            if mm_["bore_mm"] < BORE_MIN_CONCENTRIC:
                continue
            for name, r_fk, r_yaw, cap in RATIOS:
                if cap:
                    continue                      # the capstan is gone in this family
                c = closure(mm_, r_fk, r_yaw, cap, m_fixed=m_fixed_measured)
                rd = reducer(mm_, r_fk, cap, c["m_robot"], LOBES[name])
                if min(c["margin"].values()) >= 1 / MARGIN_MIN and rd["hertz_ok"] and rd["hk2512_static_margin"] >= ECC_MARGIN_MIN:
                    CLOSERS.append(dict(od_mm=od, len_mm=L, lobes=LOBES[name], ratio=r_fk, T_cont=mm_["T_cont"],
                                        m_unit=c["m_fk"], m_robot=c["m_robot"], worst=min(c["margin"].values()),
                                        can_od=c["can_od_mm"], can_h=c["can_h_mm"], P_cu=mm_["P_cu"],
                                        ecc_margin=rd["hk2512_static_margin"], pitch=rd["pitch_mm"]))
    CLOSERS.sort(key=lambda r: (r["lobes"], r["m_robot"], r["od_mm"]))

# --------------------------------------------- the body's own thermal budget
# 18 units at their continuous rating is not the load the body sees.  With three
# legs down of six, a femur/knee unit carries the stance load about half the
# time and a swing leg's units carry almost nothing; the yaw units hold a
# near-static heading torque.  DUTY is that share, and it is an estimate.
DUTY = dict(fk=0.5, yaw=0.25)
BODY = None
if PICK:
    # A unit does not sit at its own continuous rating; it delivers what the
    # requirement asks, which is 1/margin of it, and copper loss goes as torque
    # squared.  Iron loss does not -- it is set by speed, so it is carried whole.
    def unit_heat(m, margin, rpm):
        return m["P_cu"] / margin ** 2 + iron_loss(m, rpm)

    p_fk_rated = PICK["P_cu_unit"] + PICK["P_fe_unit"]
    p_fk = unit_heat(PICK["motor"], min(PICK["margin"]["femur"], PICK["margin"]["knee"]), PICK["rpm_fk"])
    p_yaw = unit_heat(YAW["motor"], YAW["margin"], YAW["rpm"]) if YAW else p_fk
    BODY = dict(per_fk_at_rating_W=p_fk_rated, per_fk_at_requirement_W=p_fk, per_yaw_at_requirement_W=p_yaw,
                all_at_continuous_W=12 * p_fk_rated + 6 * (YAW["motor"]["P_cu"] + iron_loss(YAW["motor"], YAW["rpm"]) if YAW else p_fk_rated),
                at_requirement_W=12 * p_fk + 6 * p_yaw,
                duty_weighted_W=12 * DUTY["fk"] * p_fk + 6 * DUTY["yaw"] * p_yaw, duty=DUTY, budget_W=SUSTAINED_W)
    BODY["fits_budget"] = BODY["duty_weighted_W"] <= SUSTAINED_W

# ------------------------------------------------------- what it may cost
# The datasheet carries no price.  This is the break-even: the frameless unit
# deletes the capstan stage and the outrunner's mount, so it is cheaper than the
# 8318 unit whenever the kit costs less than this.
I_DRIVER_MAX = 40.0                     # A rms, the 48 V / 40 A class driver already in the BOM
COST_8318, COST_CAPSTAN_LINES, COST_MOUNT, UNIT_8318 = 50.0, 58.0, 20.0, 423.0
BREAKEVEN = COST_8318 + COST_CAPSTAN_LINES + COST_MOUNT

# sensitivity of the pick to the two inferred numbers
SENS = []
if PICK:
    for label, f, t_tot in (("nominal (f 0.50, build 14.0 mm)", F_NOM, T_TOT),
                            ("outer-rotor split (f 0.35)", F_LO, T_TOT),
                            ("deep-stator split (f 0.65)", F_HI, T_TOT),
                            ("densest annulus (7.2 g/cm3, thin build)", F_NOM, T_TOT_LO),
                            ("lightest annulus (5.8 g/cm3, thick build)", F_NOM, T_TOT_HI)):
        m = motor(PICK["motor"]["od_mm"], PICK["motor"]["len_mm"], f=f, t_tot=t_tot)
        c = closure(m, PICK["ratio_fk"], PICK["ratio_yaw"], PICK["capstan"])
        SENS.append(dict(label=label, T_cont=m["T_cont"], mass_kg=m["mass_kg"], bore_mm=m["bore_mm"],
                         margin=c["margin"], m_robot=c["m_robot"], closes=c["closes"]))

# hot-winding penalty: the datasheet does not say at what temperature R_ph is
# quoted.  If it is 20 C and the winding actually runs at 120 C, the same loss
# budget buys 1/sqrt(1.393) of the current, so torque falls 15 %.
HOT = {}
for t_ref in (20.0, 100.0):
    ratio = (1 + RHO_CU_TEMPCO * (120.0 - t_ref))
    HOT[f"R quoted at {t_ref:.0f} C, winding at 120 C"] = dict(R_ratio=ratio, torque_ratio=1 / math.sqrt(ratio))

out = dict(datasheet=DS, checks=CHECKS, inferred=dict(rho_bracket=[RHO_LO, RHO_HI], bore_bracket_mm=[BORE_LO * 1e3, BORE_HI * 1e3],
                                                      t_tot_mm=T_TOT * 1e3, t_tot_bracket_mm=[T_TOT_LO * 1e3, T_TOT_HI * 1e3],
                                                      rho_eff=RHO_EFF, sigma_cont_Pa=SIGMA, sigma_peak_Pa=SIGMA_PEAK),
           stock_unit={name: closure(motor(DS["od_mm"], DS["len_mm"]), r_fk, r_yaw, cap)
                       for name, r_fk, r_yaw, cap in RATIOS},
           stock_motor=motor(DS["od_mm"], DS["len_mm"]),
           pick=PICK, yaw=YAW, body=BODY, breakeven_motor_price_usd=BREAKEVEN, unit_8318_usd=UNIT_8318,
           closers_on_measured_leg=CLOSERS[:12], n_closers=len(CLOSERS), mass_ladder=LADDER, leg_struct=dict(model_kg=LEG_STRUCT_MODEL, cad_kg=LEG_STRUCT_CAD,
                                               breakeven_kg=LEG_STRUCT_BREAKEVEN if PICK else None),
           cad_unit_kg=(CAD["total_g"] / 1e3) if CAD else None,
           ecc_margin_min=ECC_MARGIN_MIN, driver_limit=(dict(
               I_max=I_DRIVER_MAX, T_motor=I_DRIVER_MAX * PICK["kt_max_rms_at_speed"],
               T_joint=I_DRIVER_MAX * PICK["kt_max_rms_at_speed"] * PICK["ratio_fk"] * ETA_CYC * (ETA_CAP if PICK["capstan"] else 1.0),
               T_joint_needed=C_PEAK["knee"] * PICK["m_robot"]) if PICK else None),
           alternatives=[dict(ratio_name=r["ratio_name"], od_mm=r["motor"]["od_mm"], len_mm=r["motor"]["len_mm"],
                                         T_cont=r["motor"]["T_cont"], mass_kg=r["motor"]["mass_kg"], m_robot=r["m_robot"],
                                         margin=r["margin"], T_at_body_budget=r["T_at_body_budget"])
                                    for r in sorted(robust, key=rank)[:8]],
           margin_min=1 / MARGIN_MIN, n_robust=len(robust), sensitivity=SENS, hot_winding=HOT, n_closing=len(closing), n_rows=len(rows),
           limits=dict(od_max_mm=OD_MAX, bore_min_mm=BORE_MIN_CONCENTRIC, sustained_W=SUSTAINED_W,
                       per_unit_sustained_W=SUSTAINED_W / 18),
           requirement=dict(c_per_kg=C, c_peak_per_kg=C_PEAK, m_fixed=M_FIXED, swing=SWING))
json.dump(out, open(os.path.join(ROOT, "hw", "stator", "frameless_motor.json"), "w"), indent=1, default=float)

# -------------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
ax = axes[0]
for L in LENS:
    ods = [od for od in ODS if motor(od, L)["bore_mm"] >= BORE_MIN_CONCENTRIC]
    ax.plot(ods, [motor(od, L)["T_cont"] for od in ods], marker="o", ms=3, label=f"{L:.0f} mm stack")
ax.axhline(DS["t_cont"], color="#555", ls="--", lw=1, label=f"the WxF70x24GT itself, {DS['t_cont']:.2f} N·m")
ax.axhline(2.63, color="#b03a2e", ls=":", lw=1, label="8318 outrunner, 2.63 N·m (assumed heat-sink)")
if PICK:
    ax.scatter([PICK["motor"]["od_mm"]], [PICK["motor"]["T_cont"]], s=140, marker="*", color="#0f9b8e", zorder=5)
    ax.annotate(f"pick: Ø{PICK['motor']['od_mm']:.0f} × {PICK['motor']['len_mm']:.0f}\n{PICK['motor']['T_cont']:.2f} N·m",
                (PICK["motor"]["od_mm"], PICK["motor"]["T_cont"]), (-95, 6), textcoords="offset points", fontsize=8)
ax.set_xlabel("motor outside diameter (mm)"); ax.set_ylabel("continuous torque (N·m)")
ax.set_title(f"Torque with diameter at the datasheet's shear stress\n({SIGMA['nom']/1e3:.1f} kPa continuous, radial build held at {T_TOT*1e3:.1f} mm)", fontsize=9)
ax.grid(alpha=0.3); ax.legend(fontsize=7)

ax = axes[1]
for L in LENS:
    ods = [od for od in ODS if motor(od, L)["bore_mm"] >= BORE_MIN_CONCENTRIC]
    ax.plot(ods, [motor(od, L)["T_per_kg"] for od in ods], marker="o", ms=3, label=f"{L:.0f} mm stack")
ax.axhline(DS["t_cont"] / (DS["mass_g"] / 1e3), color="#555", ls="--", lw=1, label=f"the WxF70x24GT, {DS['t_cont']/(DS['mass_g']/1e3):.2f} N·m/kg")
ax.axhline(2.63 / 0.65, color="#b03a2e", ls=":", lw=1, label="8318, 4.0 N·m/kg")
ax.set_xlabel("motor outside diameter (mm)"); ax.set_ylabel("continuous N·m per kg of active mass")
ax.set_title("Why diameter is the lever: torque grows with the square of the\nairgap diameter, mass only with its circumference", fontsize=9)
ax.grid(alpha=0.3); ax.legend(fontsize=7)

ax = axes[2]
mk = {"25-lobe cycloid alone, no capstan": ("o", "#0f9b8e"), "40-lobe cycloid alone, no capstan": ("s", "#2a78d6"),
      "25-lobe cycloid x 4:1 capstan": ("^", "#d98c3a"), "20-lobe cycloid x 4:1 capstan (as designed)": ("v", "#b03a2e")}
for name, (marker, col) in mk.items():
    sel = [r for r in rows if r["ratio_name"] == name and r["motor"]["len_mm"] == 25.0]
    if sel:
        ax.plot([r["motor"]["od_mm"] for r in sel], [min(r["margin"].values()) for r in sel],
                marker=marker, ms=4, color=col, label=name)
ax.axhline(1.0, color="#222", lw=1.2)
ax.set_xlabel("motor outside diameter (mm), 25 mm stack"); ax.set_ylabel("worst joint margin at the fixed-point robot mass")
ax.set_title("Closure: torque, unit mass and robot mass solved together\n(≥ 1 closes the requirement as written)", fontsize=9)
ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "frameless-scaling.png"), dpi=110)

# ---- second figure: the architecture the frameless motor makes possible ----
if PICK:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15), left=0.06, right=0.97, top=0.93, bottom=0.07, hspace=0.30, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])

    def draw(ax, x0, label, blocks, colour_map):
        for (r0, r1, z0, z1, tag) in blocks:
            for sgn in (+1, -1):
                lo = x0 + (r0 if sgn > 0 else -r1)
                ax.add_patch(plt.Rectangle((lo, z0), r1 - r0, z1 - z0, fc=colour_map.get(tag, "#ccc"), ec="#333", lw=0.6, alpha=0.92))
        ax.text(x0, 112, label, ha="center", va="top", fontsize=8.5)

    CM = {"motor": "#0f9b8e", "cyc": "#d98c3a", "case": "#9aa0a6", "drum": "#b03a2e", "board": "#2a78d6"}
    m, r = PICK["motor"], PICK["reducer"]
    fr = [(m["bore_mm"] / 2, m["od_mm"] / 2, 0, m["len_mm"], "motor"),
          (0, r["r_pin_circle_mm"] + 4, 2, 22, "cyc"),
          (0, m["od_mm"] / 2 + 6, -6, 0, "case"), (0, m["od_mm"] / 2 + 6, m["len_mm"], m["len_mm"] + 6, "case")]
    pcb = [(50, 92, 6, 12, "board"), (50, 92, 20, 26, "board"), (50, 96, 0, 6, "motor"), (50, 96, 13, 19, "motor"),
           (50, 96, 27, 33, "motor"), (0, 46, 33, 49, "cyc"), (0, 96, -6, 0, "case"), (0, 96, 49, 55, "case"),
           (0, 30, 55, 81, "drum")]
    ots = [(0, 46, 0, 40, "motor"), (0, 46, 42, 58, "cyc"), (0, 96, -6, 0, "case"), (0, 96, 58, 64, "case"),
           (0, 30, 64, 90, "drum")]
    draw(ax, 0, f"Wheemo-family frameless Ø{m['od_mm']:.0f} × {m['len_mm']:.0f}\n{PICK['m_fk']:.2f} kg, {m['len_mm']+12:.0f} mm tall, no capstan\n{m['T_cont']:.1f} N·m motor, 25:1, {PICK['T_joint_fk']:.0f} N·m at the joint", fr, CM)
    draw(ax, 330, "PCB two-stator (round 9)\n5.33 kg, 54 mm tall, + capstan\n5.9 N·m motor, 80:1", pcb, CM)
    draw(ax, 600, "8318 outrunner (round 10)\n2.45 kg, 64 mm tall, + capstan\n2.6 N·m motor, 100:1", ots, CM)
    ax.set_xlim(-130, 720); ax.set_ylim(-34, 118); ax.set_aspect("equal"); ax.axis("off")
    for tag, name in (("motor", "motor"), ("cyc", "cycloid"), ("board", "PCB stator"), ("drum", "capstan drum"), ("case", "housing")):
        ax.plot([], [], "s", color=CM[tag], label=name)
    ax.legend(fontsize=8, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    ax.set_title("Three actuators, same scale, half-sections in mm: the frameless motor is an annulus, so the cycloid lives inside it "
                 "instead of underneath it", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    radii = np.linspace(30, 70, 60)
    for cap, col, lab in ((False, "#b03a2e", "no capstan: the cycloid carries the whole joint torque"),
                          (True, "#0f9b8e", "with the 4:1 capstan (round 8, as designed)")):
        st = (4.0 * ETA_CAP) if cap else 1.0
        f = [cy.design(25, C["knee"] * PICK["m_robot"] / st, C_PEAK["knee"] * PICK["m_robot"] / st, R=rr)["F_ecc"] / 1e3 for rr in radii]
        ax.plot(radii, f, color=col, lw=2, label=lab)
    ax.axhline(HK2512_C0R / 1e3, color="#555", ls="--", lw=1, label="HK2512 static rating, 16.3 kN (datasheet)")
    ax.axhline(HK2512_C0R / 1e3 / ECC_MARGIN_MIN, color="#555", ls=":", lw=1, label=f"the same with a {ECC_MARGIN_MIN:.1f}× static margin")
    ax.axvline(43.5, color="#999", lw=1)
    ax.annotate("r 43.5 — all the Ø100\nPCB-stator bore allowed", (43.5, 19.2), (-4, 0), textcoords="offset points",
                fontsize=7.5, va="top", ha="right", color="#555")
    ax.axvline(r["r_pin_circle_mm"], color="#0f9b8e", lw=1)
    ax.annotate(f"r {r['r_pin_circle_mm']:.0f} — what the Ø{m['bore_mm']:.0f}\nmotor bore allows", (r["r_pin_circle_mm"], 19.2), (5, 0),
                textcoords="offset points", fontsize=7.5, va="top", ha="left", color="#0f9b8e")
    ax.scatter([r["r_pin_circle_mm"]], [r["F_ecc"] / 1e3], s=150, marker="*", color="#0f9b8e", zorder=5)
    ax.annotate(f"{r['F_ecc']/1e3:.1f} kN, margin {r['hk2512_static_margin']:.2f}", (r["r_pin_circle_mm"], r["F_ecc"] / 1e3),
                (8, -12), textcoords="offset points", fontsize=8)
    ax.set_xlabel("cycloid ring-pin circle radius (mm)"); ax.set_ylabel("eccentric bearing load at the knee peak (kN, per disc)")
    ax.set_ylim(0, 20); ax.set_xlim(30, 70); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("Why the capstan can go: deleting it puts 4× the torque on the\ncycloid, and the bigger bore takes it back off", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    labels = ["every unit at its own\ncontinuous rating", "every unit holding the\nrequirement, continuously", f"duty-weighted\n(femur/knee {DUTY['fk']:.0%}, yaw {DUTY['yaw']:.0%})"]
    vals = [BODY["all_at_continuous_W"], BODY["at_requirement_W"], BODY["duty_weighted_W"]]
    ax.bar(labels, vals, color=["#b03a2e" if v > SUSTAINED_W else "#0f9b8e" for v in vals])
    ax.axhline(SUSTAINED_W, color="#222", lw=1.5)
    ax.text(-0.45, SUSTAINED_W + 20, f"the body's {SUSTAINED_W:.0f} W cooling budget", fontsize=8.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 20, f"{v:.0f} W", ha="center", fontsize=9.5)
    ax.set_ylabel("heat into the body (W)"); ax.tick_params(axis="x", labelsize=8)
    ax.set_title("The eighteen units against the body's cooling.\nThe duty split is an estimate, not a gait simulation", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1080)
    fig.savefig(os.path.join(FIG, "frameless-unit.png"), dpi=110)

# ---- third figure: the mass ladder, which is the real state of the design ----
if PICK and LADDER:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=(1.05, 1.0)))
    ax = axes[0]
    y = np.arange(len(LADDER))[::-1]
    cols = ["#0f9b8e" if L["closes"] else "#b03a2e" for L in LADDER]
    ax.barh(y, [L["worst"] for L in LADDER], color=cols, height=0.5)
    ax.axvline(1.0, color="#222", lw=1.6)
    ax.text(1.02, len(LADDER) - 0.35, "1.00 — closes", fontsize=8.5)
    for yy, L in zip(y, LADDER):
        ax.text(0.02, yy + 0.34, L["label"], fontsize=8.5, va="bottom")
        ax.text(L["worst"] + 0.02, yy, f"{L['worst']:.2f}   robot {L['m_robot']:.0f} kg", va="center", fontsize=8.5)
    ax.set_yticks([]); ax.set_xlim(0, 1.62); ax.set_ylim(-0.6, len(LADDER) - 0.05)
    ax.set_xlabel("worst joint margin"); ax.grid(axis="x", alpha=0.3)
    ax.set_title("What the margin does as assumptions are replaced by measurements", fontsize=10)

    ax = axes[1]
    ls = np.linspace(0, 10, 120)
    m_cad = CAD["total_g"] / 1e3
    worst = [min(margin_at(PICK["motor"], PICK["ratio_fk"], PICK["ratio_yaw"], PICK["capstan"], m_cad, m_cad - 0.4, x)[1].values()) for x in ls]
    ax.plot(ls, worst, color="#0f9b8e", lw=2.2)
    ax.axhline(1.0, color="#222", lw=1.4)
    ax.fill_between(ls, 0, worst, where=np.array(worst) >= 1, color="#0f9b8e", alpha=0.12)
    for x, lab, col in ((LEG_STRUCT_MODEL, f"what the mass budget has always\ncarried: {LEG_STRUCT_MODEL:.1f} kg a leg", "#999"),
                        (LEG_STRUCT_BREAKEVEN, f"break-even: {LEG_STRUCT_BREAKEVEN:.1f} kg", "#2a78d6"),
                        (LEG_STRUCT_CAD, f"what the only leg we have\nbuilt weighs: {LEG_STRUCT_CAD:.1f} kg", "#b03a2e")):
        ax.axvline(x, color=col, ls="--", lw=1.2)
        ax.annotate(lab, (x, 1.45 if x == LEG_STRUCT_BREAKEVEN else 1.62), (4, 0), textcoords="offset points",
                    fontsize=8, color=col, va="top")
    ax.set_xlabel("structure and transmission mass of one leg (kg), actuators excluded")
    ax.set_ylabel("worst joint margin"); ax.set_ylim(0, 1.75); ax.set_xlim(0, 10); ax.grid(alpha=0.3)
    ax.set_title("The whole design now turns on one number nobody has designed to:\nhow heavy a leg is allowed to be", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "frameless-mass-ladder.png"), dpi=110)

if __name__ == "__main__":
    print(f"{DS['part']}: datasheet checks")
    print(f"  Kt from Ke      {CHECKS['kt_from_ke']*1e3:7.2f} mN·m/A vs {DS['kt_per_A_peak']*1e3:.1f} quoted   ({CHECKS['kt_err']*100:+.1f} %)")
    print(f"  T from Kt, I    {CHECKS['t_from_kt']:7.3f} N·m      vs {DS['t_cont']:.2f} quoted    ({CHECKS['t_err']*100:+.1f} %)")
    print(f"  Km from Kt, R   {CHECKS['km_from_kt']*1e3:7.1f} mN·m/√W vs {DS['km']*1e3:.1f} quoted  ({CHECKS['km_err']*100:+.1f} %)")
    print(f"  copper loss {P_CU:.1f} W of the {DS['loss_cont_W']:.1f} W quoted -> {P_FE:.1f} W iron+windage at {DS['rpm_rated']:.0f} rpm")
    print(f"inferred: bore {BORE_LO*1e3:.0f}-{BORE_HI*1e3:.0f} mm, radial build {T_TOT_LO*1e3:.1f}-{T_TOT_HI*1e3:.1f} (nom {T_TOT*1e3:.1f}) mm, "
          f"rho_eff {RHO_EFF/1e3:.2f} g/cm3")
    print(f"          shear stress {SIGMA['nom']/1e3:.1f} kPa continuous ({SIGMA['hi_f']/1e3:.1f}-{SIGMA['lo_f']/1e3:.1f} over the split), {SIGMA_PEAK/1e3:.0f} kPa peak")
    print()
    sm = out["stock_motor"]
    print(f"the stock motor as our actuator ({sm['T_cont']:.2f} N·m, {sm['mass_kg']*1e3:.0f} g):")
    for name, c in out["stock_unit"].items():
        print(f"  {name:44s} robot {c['m_robot']:5.1f} kg  margins f{c['margin']['femur']:.2f} k{c['margin']['knee']:.2f} y{c['margin']['yaw']:.2f}  "
              f"{'CLOSES' if c['closes'] else 'short'}  supports {c['robot_supported']:.0f} kg")
    print()
    print(f"  no-load speed: E_peak {CHECKS['e_peak_at_noload']:.2f} V vs the V_dc/sqrt(3) ceiling {CHECKS['v_svpwm_peak']:.2f} V ({CHECKS['noload_err']*100:+.1f} %)")
    print(f"{len(closing)} of {len(rows)} swept designs close; {len(robust)} still close if R is quoted cold (margin >= {1/MARGIN_MIN:.2f}).")
    if PICK:
        m, p = PICK["motor"], PICK
        print(f"PICK: Ø{m['od_mm']:.0f} x {m['len_mm']:.0f} mm, bore Ø{m['bore_mm']:.0f}, airgap Ø{m['d_gap_mm']:.0f}; {p['ratio_name']}")
        print(f"  {m['T_cont']:.2f} N·m continuous, {m['T_peak']:.2f} peak, {m['mass_kg']*1e3:.0f} g active, Km {m['Km']:.2f} N·m/√W, {m['T_per_kg']:.1f} N·m/kg")
        print(f"  unit {p['m_fk']:.2f} kg -> robot {p['m_robot']:.1f} kg; joint {p['T_joint_fk']:.0f} N·m vs {p['need']['knee']:.0f} needed at the knee")
        print(f"  margins femur {p['margin']['femur']:.2f}, knee {p['margin']['knee']:.2f}, yaw {p['margin']['yaw']:.2f}")
        print(f"  {p['rpm_fk']:.0f} rpm at femur swing; wind for Kt <= {p['kt_max_rms_at_speed']:.3f} N·m/A_rms -> {p['I_at_cont']:.0f} A rms at the continuous point")
        print(f"  yaw {p['rpm_yaw']:.0f} rpm, Kt <= {p['kt_max_rms_yaw']:.3f} -> {p['I_yaw']:.0f} A")
        print(f"  at the body's own {SUSTAINED_W:.0f} W budget ({SUSTAINED_W/18:.1f} W per unit) the same motor holds {p['T_at_body_budget']:.2f} N·m")
        print(f"  losses {p['P_cu_unit']:.0f} W copper + {p['P_fe_unit']:.0f} W iron per unit; 18 units = {(p['P_cu_unit']+p['P_fe_unit'])*18:.0f} W against the {SUSTAINED_W:.0f} W body budget")
        print("  sensitivity to the inferred geometry:")
        for s in SENS:
            print(f"    {s['label']:42s} {s['T_cont']:5.2f} N·m  bore Ø{s['bore_mm']:5.1f}  worst margin {min(s['margin'].values()):.2f}  {'closes' if s['closes'] else 'SHORT'}")
        for k, v in HOT.items():
            print(f"    {k:42s} torque x{v['torque_ratio']:.2f}")
        r = PICK["reducer"]
        print(f"  reducer with no capstan: {r['lobes']} lobes on a r{r['r_pin_circle_mm']:.0f} pin circle in the Ø{PICK['motor']['bore_mm']:.0f} bore, {r['pitch_mm']:.1f} mm pitch, e {r['e_mm']:.2f}")
        print(f"    cycloid carries {r['T_cyc_cont']:.0f} / {r['T_cyc_peak']:.0f} N·m; Hertz {r['hertz_peak']:.0f} MPa peak (allow {cy.SIGMA_H_ALLOW:.0f}); "
              f"eccentric {r['F_ecc']/1e3:.1f} kN -> HK2512 static margin {r['hk2512_static_margin']:.2f}, HK3012 {r['hk3012_static_margin']:.2f}")
        if YAW:
            y = YAW
            print(f"  yaw motor sized on its own: Ø{y['motor']['od_mm']:.0f} x {y['motor']['len_mm']:.0f}, {y['motor']['T_cont']:.2f} N·m, "
                  f"{y['motor']['mass_kg']*1e3:.0f} g (saves {y['saving_g']:.0f} g/unit), margin {y['margin']:.2f}, {y['rpm']:.0f} rpm, {y['I']:.0f} A")
        if BODY:
            print(f"  body heat: {BODY['all_at_continuous_W']:.0f} W if every unit sat at its continuous rating, "
                  f"{BODY['at_requirement_W']:.0f} W if every unit held the requirement all the time, "
                  f"{BODY['duty_weighted_W']:.0f} W duty-weighted (fk {DUTY['fk']:.0%}, yaw {DUTY['yaw']:.0%}) "
                  f"against the {SUSTAINED_W:.0f} W budget -> {'fits' if BODY['fits_budget'] else 'OVER'}")
        dl = out["driver_limit"]
        print(f"  peak is driver-limited, not motor-limited: {I_DRIVER_MAX:.0f} A gives {dl['T_motor']:.1f} N·m at the motor -> "
              f"{dl['T_joint']:.0f} N·m at the joint against {dl['T_joint_needed']:.0f} needed (margin {dl['T_joint']/dl['T_joint_needed']:.2f}); "
              f"the motor itself could make {PICK['motor']['T_peak']:.0f} N·m, so the drive must current-limit to protect the reducer")
        print("  MASS LADDER — what the margin does as assumptions are replaced by measurements:")
        for L in LADDER:
            print(f"    {L['label'][:66]:66s} unit {L['m_unit_fk']:.2f} kg, leg struct {L['leg_struct_kg']:.1f} kg -> "
                  f"robot {L['m_robot']:5.1f} kg, worst margin {L['worst']:.2f}  {'closes' if L['closes'] else '*** DOES NOT CLOSE ***'}")
        if LEG_STRUCT_BREAKEVEN is not None:
            print(f"    the design stops closing once a leg's structure passes {LEG_STRUCT_BREAKEVEN:.1f} kg "
                  f"(the model carries {LEG_STRUCT_MODEL:.1f}, the round-14 leg CAD measured {LEG_STRUCT_CAD:.1f})")
        if CLOSERS:
            print(f"  ON THE LEG WE MEASURED ({LEG_STRUCT_CAD:.1f} kg of structure a leg), {len(CLOSERS)} sizes still close. The smallest, by lobe count then mass:")
            for c in CLOSERS[:4]:
                print(f"    Ø{c['od_mm']:.0f} x {c['len_mm']:.0f} mm, {c['lobes']}-lobe {c['ratio']:.0f}:1 -> {c['T_cont']:5.2f} N·m, unit {c['m_unit']:.2f} kg, "
                      f"robot {c['m_robot']:5.1f} kg, margin {c['worst']:.2f}, can Ø{c['can_od']:.0f} x {c['can_h']:.0f} mm, {c['P_cu']:.0f} W copper")
        elif CAD:
            print(f"  ON THE LEG WE MEASURED: nothing inside the Ø{OD_MAX:.0f} limit closes. The leg has to get lighter.")
        print(f"  price: the datasheet has none. The frameless unit beats the $ {UNIT_8318:.0f} 8318 unit while the kit costs under $ {BREAKEVEN:.0f}")

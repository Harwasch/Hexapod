#!/usr/bin/env python3
"""
Electromagnetic sizing study for the Hexapod joint actuator motor.

Question: inside a Ø170 x 42 mm pancake, which motor topology delivers the
per-joint motor-shaft torque of docs/design/01-sizing.md section 6/7, in each of
two packaging variants (reducer in-plane inside the motor, or reducer stacked
axially beside it)?

Method: airgap shear stress.  For an axial-flux machine the tangential shear
stress on the airgap is

    sigma(r) = (k_w / sqrt(2)) * B_pk(r) * A_rms(r)          [Pa]
    A_rms(r) = fill(r) * t_cu_total * J_rms                  [A/m]
    T        = n_stators * k_edge * 2*pi * INT sigma(r) r^2 dr

A_rms is the rms linear current density: rms current per conductor times
conductors per metre of circumference, which for a window of axial copper
thickness t_cu_total filled to fraction `fill` at current density J is exactly
fill * t_cu * J.  The sqrt(2) is the sinusoidal-commutation form factor
(sigma = 0.5 * B_pk * A_pk, A_pk = sqrt(2) A_rms).

J_rms is NOT a free parameter: it is set by the thermal balance
    P_copper(J) + P_eddy + P_core = (T_cu_max - T_ambient) / R_th
so every candidate is compared at the same heat rejection.  Because
    T ~ fill*t_cu*J*B   and   P_cu ~ J^2 * fill*t_cu*Area,
continuous torque scales as  B * sqrt(P_allow * fill * t_cu).  That square root
is the headline of this study: doubling the copper buys 1.41x torque, not 2x.

Run:  /opt/hw-py/bin/python analysis/motor_options.py
Writes: docs/design/motor/torque-density.png ; prints markdown tables.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# 1. ASSUMPTIONS -- every number used anywhere in the script is declared here.
#    [D] = derived in this file.  [E] = engineering estimate / judgement.
#    [V] = verified against a fetched datasheet (see docs 07 for the link).
#    [M] = taken from memory, NOT verified -- flagged in the write-up.
# =============================================================================

# ---- Envelope (fixed by the robot geometry, docs/design/01-sizing.md) --------
R_OUT            = 0.085     # m   outer radius of the 170 mm pancake
ENVELOPE_AXIAL   = 0.042     # m   total actuator height incl. housing+reducer

# packaging variant -> (inner radius available to the magnetics, axial for magnetics)
PACKAGING = {
    "in-plane": dict(r_in=0.050, axial=0.030),   # cycloid inside the motor, r<50 mm
    "stacked":  dict(r_in=0.020, axial=0.018),   # cycloid axially beside the motor
}

# ---- Bus / drive ------------------------------------------------------------
V_DC          = 48.0
V_PH_MAX      = V_DC / math.sqrt(6) * 0.95   # [D] SVPWM linear range, 5% modulation margin
N_NOLOAD_TGT  = 5000.0                       # rpm, winding scaled to this no-load speed
N_EVAL        = 3500.0                       # rpm at which f_e and eddy loss are evaluated
                                             #   (the fastest joint needs 3450 rpm)

# ---- Thermal ----------------------------------------------------------------
# R_th justification: the verified Kollmorgen TBM2G-11526 datasheet gives
# R_th(winding->ambient) = 1.21 K/W for a 115 mm OD / 26 mm motor in an aluminium
# housing bolted to a 305x305x12 mm aluminium plate.  Our stator is a 170 mm disc
# (2.3x the surface) bonded to an aluminium housing that is a structural part of
# a body with far more area than that plate, but it sits between two rotors with
# air on both faces and the heat has to travel radially out through the copper
# first.  1.5 K/W nominal, 1.0-2.0 K/W swept in the sensitivity section.
R_TH_NOMINAL  = 1.5          # K/W  stator copper -> ambient  [E, anchored on [V]]
R_TH_OUTRUNNER= 2.0          # K/W  radial machine: stator is inboard, worse path [E]
T_AMB         = 45.0         # degC  (spec: -10..+45 degC environment)
T_CU_MAX_PCB  = 120.0        # degC  FR4/copper long-term (IPC Tg-limited)      [E]
T_CU_MAX_WIRE = 150.0        # degC  class-F enamelled wire                     [E]

# ---- Copper -----------------------------------------------------------------
RHO_CU_20     = 1.724e-8     # ohm.m
ALPHA_CU      = 0.00393      # 1/K
RHO_CU_DENS   = 8960.0       # kg/m3
CP_CU         = 385.0        # J/kg.K (adiabatic 2 s burst check)

# ---- Magnets ----------------------------------------------------------------
# N45 sintered NdFeB, SH temperature grade so the 2 s burst has demag margin.
BR_20         = 1.32         # T   N45 typical Br at 20 degC                    [M]
BR_TEMPCO     = -0.0012      # 1/K                                             [M]
T_MAGNET      = 90.0         # degC assumed magnet body temperature            [E]
BR = BR_20 * (1 + BR_TEMPCO * (T_MAGNET - 20.0))     # -> 1.21 T               [D]
MU_R_MAG      = 1.05
RHO_MAG       = 7500.0       # kg/m3
K_HALBACH_SEG = 0.900        # sin(pi/M)/(pi/M) for M = 4 segments/wavelength   [D]
HALBACH_SEGS  = 4            # magnet blocks per wavelength (per pole pair)
MAG_ARC_MIN   = 2.5e-3       # m   smallest magnet block arc we will glue by hand [E]
H_M_LAMBDA    = 0.20         # magnet thickness cap as a fraction of wavelength.
                             # (1-exp(-2 pi h/lam)) = 0.72 at 0.20 vs 0.79 at 0.25:
                             # 9% less field for 20% less magnet mass and cost.
K_3D          = 0.90         # finite array, glue gaps, magnet tolerance        [E]
MAG_PACK      = 0.95         # ring fill of the magnet annulus                 [E]
MU0           = 4e-7 * math.pi

# ---- Winding / stator build -------------------------------------------------
K_W           = 0.933        # winding factor, 12-slot/10-pole family (Nc/2p = 1.2)
TRACE_SPACE   = 0.15e-3      # m   PCB copper-to-copper space                   [E]
W_MIN_PCB     = 0.15e-3      # m   minimum usable trace width
OZ            = 34.8e-6      # m   1 oz copper thickness
PCB_STACKS = {   # name -> (n_layers, copper per layer, finished board thickness)
    "PCB 6L 2oz":  (6,  2 * OZ, 1.00e-3),
    "PCB 12L 2oz": (12, 2 * OZ, 1.80e-3),
    "PCB 12L 3oz": (12, 3 * OZ, 2.20e-3),
}
RHO_FR4       = 1850.0       # kg/m3

WIRE_FILL     = 0.45         # copper fraction inside a wound flat coil (0.40-0.55) [E]
WIRE_PACK     = 0.90         # coil-to-coil clearance in the annulus               [E]
COIL_THICK    = 5.0e-3       # m   wound pancake coil axial thickness              [E]
COIL_SKIN     = 0.5e-3       # m   carrier skin each side of the coil array
RHO_POTTING   = 1800.0       # kg/m3 epoxy/glass carrier

AIR_CLEAR     = 0.5e-3       # m   mechanical clearance, stator face to magnet face
T_CARRIER_IN  = 3.0e-3       # m   rotor carrier (aluminium) - in-plane variant
T_CARRIER_ST  = 2.5e-3       # m   rotor carrier - stacked variant
RHO_AL        = 2700.0
HUB_ALLOW     = 1.30         # x carrier annulus mass, for the hub and spokes      [E]

# ---- Iron-core (YASA-style) candidate D -------------------------------------
SLOT_FRACTION = 0.45         # circumferential fraction of the pitch that is slot  [E]
IRON_FILL     = 0.45         # copper fill inside the slot (concentrated coil)     [E]
B_YOKE_MAX    = 1.6          # T   rotor back-iron flux density ceiling
RHO_SMC       = 7400.0       # kg/m3 soft magnetic composite
RHO_STEEL     = 7650.0
SMC_LOSS_REF  = 25.0         # W/kg at 1.0 T, 400 Hz (Somaloy-class)              [M]
                             # D's viability hinges on this one -- see the sensitivity note
CARTER_LEAK   = 0.95 * 0.90  # slot-opening + 3-D leakage derate                  [E]
IRON_ET_BAND  = 4.0e-3       # m radial band lost to the tooth-wound coil ends    [E]

# ---- Radial-flux reference (candidate E) ------------------------------------
# Anchored on the verified Kollmorgen TBM2G-11526: Tc = 6.03 N.m at stall,
# 115 mm OD, 26 mm stack, 1.43 kg, R_th = 1.21 K/W at 130 K rise.  Taking the
# airgap radius as 0.39*OD gives sigma_ref = T/(2 pi r_g^2 L).
TBM_T_CONT, TBM_OD, TBM_STACK, TBM_MASS, TBM_RTH, TBM_RISE = 6.03, 0.115, 0.026, 1.43, 1.21, 130.0
SLOTS_RADIAL  = 48           # concentrated-wound slots, sets the end-winding overhang
BACKIRON_RAD  = 4.0e-3       # m radial thickness of rotor back iron + stator yoke  [E]

# ---- Cost (small-batch, ~100 off; all [E]) ----------------------------------
COST_MAGNET_PER_G = 0.22
COST_PCB = {"PCB 6L 2oz": 18.0, "PCB 12L 2oz": 45.0, "PCB 12L 3oz": 65.0}
COST_WIRE_PER_KG  = 30.0
COST_CARRIER_EACH = 30.0
COST_POTTED_STATOR= 25.0
COST_SMC_SET      = 250.0
COST_BACKIRON_EACH= 40.0

# ---- Ratings convention -----------------------------------------------------
PEAK_CURRENT_MULT = 3.0      # 2 s burst = 3x continuous current (stated convention)
PEAK_BURST_S      = 2.0
PEAK_ADIABATIC_DT = 30.0     # K allowed adiabatic copper rise in the 2 s burst

# ---- Joint requirements at the MOTOR SHAFT (01-sizing.md section 6) ----------
JOINTS = [
    # name,  ratio, T_cont, T_peak, min motor rpm
    ("yaw",   45, 1.4, 1.5, 3450),
    ("femur", 55, 2.8, 5.1, 2100),
    ("knee",  60, 2.7, 5.7, 2300),
]

NGRID = 400   # radial integration points


# =============================================================================
# 2. PHYSICS
# =============================================================================

def rho_cu(T_c):
    return RHO_CU_20 * (1 + ALPHA_CU * (T_c - 20.0))


def halbach_B(r, p, h_m, g_mag):
    """Peak airgap flux density at the mid-plane between two opposed Halbach
    arrays.  For one ideal array of wavelength lam the surface field decays as

        B(y) = Br * Kseg * (1 - exp(-2 pi h_m/lam)) * exp(-2 pi y/lam)

    Two arrays facing each other add at the mid-plane (y = g/2 each), hence
    the factor 2 and exp(-pi g/lam).  lam = 2 pi r / p at radius r.
    """
    lam = 2 * math.pi * r / p
    return (2.0 * BR * K_HALBACH_SEG * K_3D
            * (1.0 - np.exp(-2 * math.pi * h_m / lam))
            * np.exp(-math.pi * g_mag / lam))


def surface_magnet_B(h_m, g_mech):
    """1-D magnetic circuit for surface magnets on back iron (candidate D)."""
    return BR * h_m / (h_m + MU_R_MAG * g_mech) * CARTER_LEAK


def eddy_pv_strip(f, B_pk, w):
    """Classical eddy loss per unit volume in a thin strip of width w with the
    field normal to its face: pi^2 f^2 B^2 w^2 / (6 rho).  Valid while the
    thickness << skin depth (66 um copper vs 1.9 mm at 1.2 kHz)."""
    return math.pi**2 * f**2 * B_pk**2 * w**2 / (6 * rho_cu(T_CU_MAX_PCB))


def eddy_pv_wire(f, B_pk, d):
    """Proximity eddy loss per unit volume in a round wire of diameter d in a
    transverse alternating field: pi^2 f^2 B^2 d^2 / (8 rho)."""
    return math.pi**2 * f**2 * B_pk**2 * d**2 / (8 * rho_cu(T_CU_MAX_WIRE))


def end_turn_band(r, n_coils):
    """Radial band consumed by the coil end turns.  A coil of N turns per layer
    nests N conductors of pitch (w+s) at the inner/outer radius, and the pitch
    is 2 pi r / (2 Nc N), so the band is pi r / Nc regardless of N."""
    return math.pi * r / n_coils


# =============================================================================
# 3. AXIAL-FLUX CANDIDATE MODEL (A, B, C, D)
# =============================================================================

def eval_axial(kind, pack, p, w_cond, n_stators, stack=None, coil_thick=None,
               t_bi=4.0e-3, h_m_iron=4.0e-3, r_th=R_TH_NOMINAL, fill_scale=1.0):
    """Evaluate one axial-flux design point.  Returns a dict or None if infeasible.

    kind: 'pcb' | 'wire' | 'iron'
    p:    pole pairs;  n_coils = 2.4 p  (12-slot/10-pole family, k_w = 0.933)
    w_cond: PCB trace width / wire diameter / slot-conductor bundle width [m]
    """
    geo = PACKAGING[pack]
    r_bi, axial = geo["r_in"], geo["axial"]
    n_coils = int(round(2.4 * p))
    if n_coils % 3:
        n_coils += 3 - n_coils % 3
    n_rotors = n_stators + 1

    # ---- radial layout: end turns eat into the annulus at both ends ----------
    # Ironless: the coil is a flat spiral, so its N turns nest RADIALLY at each
    # end and the band is pi r / Nc (independent of N -- see end_turn_band()).
    # Iron core: the coil is wound around a tooth, so its end turns wrap in the
    # axial direction and only the coil build (~IRON_ET_BAND) is lost radially.
    if kind == "iron":
        r1, r2 = r_bi + IRON_ET_BAND, R_OUT - IRON_ET_BAND
    else:
        r1 = r_bi / (1 - math.pi / n_coils)      # r1 = r_bi + pi r1 / Nc
        r2 = R_OUT / (1 + math.pi / n_coils)
    L_act = r2 - r1
    if L_act < 6e-3:
        return None

    # ---- axial stack-up -> magnet thickness ---------------------------------
    t_carrier = T_CARRIER_IN if pack == "in-plane" else T_CARRIER_ST
    if kind == "pcb":
        n_layers, t_layer, t_board = stack
        t_stator = t_board
        t_cu_tot = n_layers * t_layer
        fill = (w_cond / (w_cond + TRACE_SPACE)) * fill_scale
        T_cu_max = T_CU_MAX_PCB
    elif kind == "wire":
        t_stator = coil_thick + 2 * COIL_SKIN
        t_cu_tot = coil_thick * WIRE_FILL * WIRE_PACK * fill_scale
        fill = 1.0
        T_cu_max = T_CU_MAX_WIRE
    elif kind == "iron":
        # the axial budget is split between two back-iron rings, two magnets and
        # the toothed stator; back iron and magnet thickness are swept by optimise()
        slot_depth = axial - 2 * (t_bi + h_m_iron) - 2 * AIR_CLEAR
        if slot_depth < 4.0e-3:
            return None
        t_stator = slot_depth
        t_cu_tot = slot_depth * SLOT_FRACTION * IRON_FILL * fill_scale
        fill = 1.0
        T_cu_max = T_CU_MAX_WIRE
    else:
        raise ValueError(kind)

    if kind in ("pcb", "wire"):
        # ironless Halbach: n_rotors carriers, 2*n_stators magnet faces
        left = axial - n_rotors * t_carrier - n_stators * (t_stator + 2 * AIR_CLEAR)
        h_m = left / (2 * n_stators)
        lam_m = 2 * math.pi * (0.5 * (r1 + r2)) / p
        h_m = min(h_m, H_M_LAMBDA * lam_m)    # past ~lam/5 the extra magnet is wasted
        if h_m < 2.0e-3:
            return None
        # buildability: the Halbach blocks must be wide enough to handle and glue
        if 2 * math.pi * r1 / (HALBACH_SEGS * p) < MAG_ARC_MIN:
            return None
        g_mag = t_stator + 2 * AIR_CLEAR      # magnet face to magnet face
        t_bi = 0.0
        axial_used = (n_rotors * t_carrier + 2 * n_stators * h_m
                      + n_stators * (t_stator + 2 * AIR_CLEAR))
    else:
        h_m = h_m_iron
        g_mag = 0.0
        axial_used = 2 * (t_bi + h_m) + slot_depth + 2 * AIR_CLEAR
        # back-iron saturation: half the pole flux flows each way round the yoke
        flux_pole = surface_magnet_B(h_m, AIR_CLEAR + 0.2e-3) * math.pi * (r2**2 - r1**2) / (2 * p)
        B_yoke = flux_pole / (2 * t_bi * (r2 - r1))
        if B_yoke > B_YOKE_MAX:
            return None

    # ---- radial integration --------------------------------------------------
    r = np.linspace(r1, r2, NGRID)
    if kind in ("pcb", "wire"):
        B = halbach_B(r, p, h_m, g_mag)
    else:
        B = np.full_like(r, surface_magnet_B(h_m, AIR_CLEAR + 0.2e-3))

    # Edge factor: an ironless Halbach field falls off over a length lam/(2 pi)
    # at the inner and outer edges of a finite annulus.  With iron teeth the flux
    # is guided and only the slot-opening fringe is lost.
    lam_mean = 2 * math.pi * (0.5 * (r1 + r2)) / p
    if kind == "iron":
        k_edge = 0.97
    else:
        k_edge = float(np.clip(1 - lam_mean / (2 * math.pi * L_act), 0.30, 1.0))

    # torque per unit current density: T = n_st*k_edge*2pi INT sigma r^2 dr
    A_per_J = fill * t_cu_tot                        # A_rms per (A/m2) of J
    dT_dJ = (n_stators * k_edge * 2 * math.pi
             * np.trapezoid((K_W / math.sqrt(2)) * B * A_per_J * r**2, r))

    # ---- copper volume (active + end turns) ---------------------------------
    vol_act = n_stators * float(np.trapezoid(fill * t_cu_tot * 2 * math.pi * r, r))
    b_in, b_out = end_turn_band(r1, n_coils), end_turn_band(r2, n_coils)
    vol_et = n_stators * fill * t_cu_tot * math.pi * (
        (r1**2 - max(r1 - b_in, 1e-4)**2) + ((r2 + b_out)**2 - r2**2))
    vol_cu = vol_act + vol_et

    # ---- frequency-dependent losses at N_EVAL --------------------------------
    f_e = p * N_EVAL / 60.0
    if kind == "pcb":
        pv = eddy_pv_strip(f_e, B, w_cond)
    else:
        pv = eddy_pv_wire(f_e, B, w_cond)
    P_eddy = n_stators * float(np.trapezoid(pv * fill * t_cu_tot * 2 * math.pi * r, r))

    P_core = 0.0
    m_core = 0.0
    if kind == "iron":
        tooth_frac = 1 - SLOT_FRACTION
        B_tooth = float(B[0]) / tooth_frac
        m_core = RHO_SMC * tooth_frac * math.pi * (r2**2 - r1**2) * slot_depth
        P_core = m_core * SMC_LOSS_REF * (f_e / 400.0)**1.5 * (B_tooth / 1.0)**2

    # ---- thermal balance -> J -----------------------------------------------
    P_allow = (T_cu_max - T_AMB) / r_th
    P_cu_allow = P_allow - P_eddy - P_core
    if P_cu_allow <= 0.05 * P_allow:
        return None
    J = math.sqrt(P_cu_allow / (rho_cu(T_cu_max) * vol_cu))
    if J > 30e6:      # sanity cap; never binds in practice here
        J = 30e6
        P_cu_allow = rho_cu(T_cu_max) * vol_cu * J**2

    T_cont = dT_dJ * J

    # ---- peak (2 s) ----------------------------------------------------------
    J_pk = PEAK_CURRENT_MULT * J
    dT_adiabatic = J_pk**2 * rho_cu(T_cu_max) * PEAK_BURST_S / (RHO_CU_DENS * CP_CU)
    J_adiabatic = math.sqrt(PEAK_ADIABATIC_DT * RHO_CU_DENS * CP_CU
                            / (rho_cu(T_cu_max) * PEAK_BURST_S))
    # demagnetising armature field at the magnet
    K_pk = math.sqrt(2) * fill * t_cu_tot * J_pk
    tau_p = math.pi * 0.5 * (r1 + r2) / p
    if kind == "iron":
        H_a = (K_pk * tau_p / math.pi) / (h_m + MU_R_MAG * (AIR_CLEAR + 0.2e-3))
    else:
        H_a = K_pk / 2.0        # ironless: field of a current sheet, no concentration
    B_demag = MU0 * H_a
    demag_ok = B_demag < 0.35 * BR
    sat_mult = 2.5 if kind == "iron" else PEAK_CURRENT_MULT   # iron saturates first
    peak_mult = min(PEAK_CURRENT_MULT, sat_mult, J_adiabatic / J)
    peak_limit = ("3x I_cont" if peak_mult >= PEAK_CURRENT_MULT - 1e-6
                  else ("saturation" if kind == "iron" else "adiabatic 2 s"))
    if not demag_ok:
        peak_limit = "demagnetisation"
    T_peak = dT_dJ * J * peak_mult

    # ---- winding scaling -> Kt, R, speeds -----------------------------------
    w_ph = 2 * math.pi * N_NOLOAD_TGT / 60.0
    Kt_target = 3 * V_PH_MAX / w_ph
    I_target = T_cont / Kt_target
    # series turns per phase implied by that Kt
    I_total = J * fill * t_cu_tot * 2 * math.pi * r1     # total ampere-conductors
    N_ph = I_total / (6 * I_target)
    if kind == "pcb":
        N_ph_max = n_layers * (2 * math.pi * r1 / (w_cond + TRACE_SPACE)) / 6.0
    else:
        N_ph_max = 1e9
    N_ph = max(1.0, min(N_ph, N_ph_max))
    I_cont = I_total / (6 * N_ph)
    Kt = T_cont / I_cont
    R_ph = P_cu_allow / (3 * I_cont**2)
    n_noload = 3 * V_PH_MAX / Kt * 60 / (2 * math.pi)
    n_at_cont = max(0.0, 3 * (V_PH_MAX - I_cont * R_ph) / Kt * 60 / (2 * math.pi))
    n_at_peak = max(0.0, 3 * (V_PH_MAX - peak_mult * I_cont * R_ph) / Kt * 60 / (2 * math.pi))

    # ---- masses --------------------------------------------------------------
    rm1, rm2 = max(r1 - 1e-3, r_bi), min(r2 + 1e-3, R_OUT)
    n_faces = 2 * n_stators
    m_mag = n_faces * math.pi * (rm2**2 - rm1**2) * h_m * MAG_PACK * RHO_MAG
    m_cu = vol_cu * RHO_CU_DENS
    if kind == "pcb":
        vol_board = n_stators * math.pi * (R_OUT**2 - r_bi**2) * t_board
        m_stator = m_cu + max(vol_board - vol_cu, 0) * RHO_FR4
        cost_stator = n_stators * COST_PCB[stack_name_of(stack)]
    elif kind == "wire":
        vol_stator = n_stators * math.pi * (R_OUT**2 - r_bi**2) * (coil_thick + 2 * COIL_SKIN)
        m_stator = m_cu + max(vol_stator - vol_cu, 0) * RHO_POTTING
        cost_stator = n_stators * (m_cu * COST_WIRE_PER_KG + COST_POTTED_STATOR)
    else:
        m_stator = m_cu + m_core
        cost_stator = COST_SMC_SET + m_cu * COST_WIRE_PER_KG

    m_carrier = n_rotors * math.pi * (R_OUT**2 - r_bi**2) * t_carrier * RHO_AL * HUB_ALLOW
    m_bi = 0.0
    if kind == "iron":
        m_bi = 2 * math.pi * (rm2**2 - rm1**2) * t_bi * RHO_STEEL
    m_total = m_mag + m_stator + m_carrier + m_bi

    cost = (m_mag * 1000 * COST_MAGNET_PER_G + cost_stator
            + n_rotors * COST_CARRIER_EACH + (2 * COST_BACKIRON_EACH if kind == "iron" else 0))

    # torque density is quoted against the axial length the magnetics actually use,
    # not the budget -- several designs leave axial room that the reducer could take.
    vol_env = math.pi * R_OUT**2 * axial_used * 1000.0

    return dict(
        p=p, n_coils=n_coils, w=w_cond, h_m=h_m, r1=r1, r2=r2, g_mag=g_mag,
        n_blocks=(HALBACH_SEGS * p if kind in ("pcb", "wire") else 2 * p),
        mag_arc=2 * math.pi * r1 / (HALBACH_SEGS * p) if kind in ("pcb", "wire") else 0.0,
        B_pk=float(np.mean(B)), k_edge=k_edge, J=J, T_cont=T_cont, T_peak=T_peak,
        peak_limit=peak_limit, Kt=Kt, I_cont=I_cont, R_ph=R_ph, N_ph=N_ph,
        P_cu=P_cu_allow, P_eddy=P_eddy, P_core=P_core, P_allow=P_allow,
        f_e=f_e, n_noload=n_noload, n_at_cont=n_at_cont, n_at_peak=n_at_peak,
        m_mag=m_mag, m_total=m_total, sigma=T_cont / (2 * math.pi / 3 * (r2**3 - r1**3) * n_stators),
        t_per_kg=T_cont / m_total, t_per_l=T_cont / vol_env, cost=cost,
        vol_env=vol_env, axial_used=axial_used, axial_budget=axial,
        dT_adiabatic=dT_adiabatic, B_demag=B_demag,
    )


_STACK_LOOKUP = {v: k for k, v in PCB_STACKS.items()}
def stack_name_of(stack):
    return _STACK_LOOKUP[stack]


def optimise(kind, pack, n_stators, **kw):
    """Sweep pole pairs and conductor width; keep the highest continuous torque."""
    best = None
    p_list = range(4, 29)
    if kind == "pcb":
        w_list = [0.125e-3, 0.15e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.6e-3, 0.8e-3, 1.2e-3, 2.0e-3]
    elif kind == "wire":
        w_list = [0.4e-3, 0.5e-3, 0.6e-3, 0.8e-3]
    else:
        w_list = [0.5e-3, 0.8e-3, 1.2e-3]
    # the iron-core stator has two extra degrees of freedom inside the same axial
    # budget: how much of it goes to back iron (which must not saturate) and how
    # much to magnet, with whatever is left becoming slot depth (i.e. copper).
    iron_sweep = ([(bi, hm) for bi in (3e-3, 4e-3, 5e-3, 6e-3, 8e-3)
                            for hm in (2.5e-3, 3e-3, 4e-3, 5e-3)]
                  if kind == "iron" else [(None, None)])
    for p in p_list:
        for w in w_list:
            for bi, hm in iron_sweep:
                extra = {} if bi is None else dict(t_bi=bi, h_m_iron=hm)
                res = eval_axial(kind, pack, p, w, n_stators, **kw, **extra)
                if res and (best is None or res["T_cont"] > best["T_cont"]):
                    best = res
    return best


# =============================================================================
# 4. RADIAL-FLUX REFERENCE (candidate E) -- scaled from a verified datasheet
# =============================================================================

SIGMA_TBM = TBM_T_CONT / (2 * math.pi * (0.39 * TBM_OD)**2 * TBM_STACK)   # Pa

def eval_radial(pack, r_th=R_TH_OUTRUNNER):
    geo = PACKAGING[pack]
    axial = geo["axial"]
    r_g = R_OUT - BACKIRON_RAD - 2e-3            # airgap radius inside the rotor ring
    overhang = 0.5 * (2 * math.pi * r_g / SLOTS_RADIAL)   # end-winding, each end
    L_act = axial - 2 * overhang
    if L_act <= 2e-3:
        return None
    # sigma ~ sqrt(P_allowed / copper volume).  Keeping the slot depth the same,
    # copper volume scales as (airgap circumference x active length), i.e. r_g*L.
    P_ref = TBM_RISE / TBM_RTH
    P_ours = (T_CU_MAX_WIRE - T_AMB) / r_th
    # copper volume ~ (airgap circumference) x (active length + both end windings)
    tbm_overhang = 0.5 * (2 * math.pi * (0.39 * TBM_OD) / 12)   # 12 slots assumed
    vcu_ratio = ((r_g * axial)
                 / (0.39 * TBM_OD * (TBM_STACK + 2 * tbm_overhang)))
    sigma = SIGMA_TBM * math.sqrt((P_ours / P_ref) / vcu_ratio)
    T_cont = sigma * 2 * math.pi * r_g**2 * L_act
    # mass scaled from the TBM datapoint by active volume, +15% for the larger bore
    m_total = TBM_MASS * ((r_g**2 * L_act) / ((0.39 * TBM_OD)**2 * TBM_STACK)) * 1.15
    p = 14
    f_e = p * N_EVAL / 60.0
    vol_env = math.pi * R_OUT**2 * axial * 1000.0
    Kt = 3 * V_PH_MAX / (2 * math.pi * N_NOLOAD_TGT / 60.0)
    I_cont = T_cont / Kt
    P_cu = (T_CU_MAX_WIRE - T_AMB) / r_th
    return dict(p=p, sigma=sigma, T_cont=T_cont, T_peak=2.5 * T_cont,
                peak_limit="saturation", Kt=Kt, I_cont=I_cont,
                R_ph=P_cu / (3 * I_cont**2), P_cu=P_cu, P_eddy=0.0, P_core=0.0,
                P_allow=P_cu, f_e=f_e, n_noload=N_NOLOAD_TGT,
                n_at_cont=N_NOLOAD_TGT * 0.93, n_at_peak=N_NOLOAD_TGT * 0.80,
                m_mag=0.09 * m_total, m_total=m_total, J=float('nan'),
                t_per_kg=T_cont / m_total, t_per_l=T_cont / vol_env,
                cost=350.0, vol_env=vol_env, h_m=3e-3, B_pk=0.87,
                r1=r_g, r2=r_g, L_act=L_act, overhang=overhang, N_ph=float('nan'),
                k_edge=1.0, w=0.6e-3, n_coils=SLOTS_RADIAL, g_mag=0.0,
                n_blocks=2 * p, mag_arc=2 * math.pi * r_g / (2 * p),
                axial_used=axial, axial_budget=axial,
                dT_adiabatic=float('nan'), B_demag=float('nan'))


# =============================================================================
# 5. CANDIDATE LIST
# =============================================================================

BUILDABILITY = {
 "A1": "Easiest. Any 6-layer fab, magnets glued into a printed jig, no winding. "
       "Runs the highest current density of the set, so the bond to the housing is critical.",
 "A2": "Same process as A1, 12-layer 2 oz is a stock JLC/PCBWay option. Best effort/return of the PCB family.",
 "A3": "12-layer 3 oz is a specialist stack-up (heavy copper on inner layers); "
       "longer lead time and ~1.5x the board cost, no new skill needed.",
 "B":  "Two identical boards and a third rotor. Doubles the assembly and halves the magnet thickness; "
       "the middle rotor needs magnets on both faces and a stiff, thin, non-magnetic carrier.",
 "C1": "CNC- or hand-wound flat coils potted in a printed carrier. ~2 h of labour per stator and a winding "
       "jig, but no fab lead time and the same rotors as A. This is the DIY axial-flux recipe.",
 "C2": "As C1 twice, in an axial budget that leaves 2 mm magnets. Not worth the labour.",
 "D":  "Not buildable in a small shop: needs SMC or wound-strip laminated tooth cores. "
       "Cogging and core loss as well. Contract-build only.",
 "E":  "Nothing to build - buy a frameless torque motor. But it is an iron ring at 160 mm bore: heavy.",
}

CANDIDATES = [
    ("A1  PCB 6L 2oz, 1 stator",      dict(kind="pcb",  n_stators=1, stack=PCB_STACKS["PCB 6L 2oz"])),
    ("A2  PCB 12L 2oz, 1 stator",     dict(kind="pcb",  n_stators=1, stack=PCB_STACKS["PCB 12L 2oz"])),
    ("A3  PCB 12L 3oz, 1 stator",     dict(kind="pcb",  n_stators=1, stack=PCB_STACKS["PCB 12L 3oz"])),
    ("B   PCB 12L 2oz, 2 stators",    dict(kind="pcb",  n_stators=2, stack=PCB_STACKS["PCB 12L 2oz"])),
    ("C1  Wound coils, 1 stator",     dict(kind="wire", n_stators=1, coil_thick=COIL_THICK)),
    ("C2  Wound coils, 2 stators",    dict(kind="wire", n_stators=2, coil_thick=4.0e-3)),
    ("D   Iron-core YASA, 1 stator",  dict(kind="iron", n_stators=1)),
]


def build_results():
    rows = []
    for name, spec in CANDIDATES:
        for pack in PACKAGING:
            kw = dict(spec)
            kind = kw.pop("kind")
            n_st = kw.pop("n_stators")
            res = optimise(kind, pack, n_st, **kw)
            rows.append((name, pack, res))
    for pack in PACKAGING:
        rows.append(("E   Radial-flux ring, same OD", pack, eval_radial(pack)))
    return rows


# =============================================================================
# 6. TABLES
# =============================================================================

def fmt(v, n=2):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{n}f}"


def main_table(rows):
    hdr = ("| Candidate | Packaging | sigma cont (kPa) | T cont (N.m) | T peak 2 s (N.m) | "
           "peak set by | P_cu (W) | P_eddy (W) | f_e at 3500 rpm (Hz) | "
           "n max at T_cont (rpm) | axial used (mm) | magnet (g) | motor mass (g) | N.m/kg | N.m/L | "
           "parts $ | buildability |")
    sep = "|" + "---|" * 17
    out = [hdr, sep]
    for name, pack, r in rows:
        if r is None:
            out.append(f"| {name} | {pack} | — | — | — | no feasible design in this envelope "
                       f"| — | — | — | — | — | — | — | — | — | — | {BUILDABILITY[name.split()[0]]} |")
            continue
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, pack, fmt(r["sigma"] / 1000, 2), fmt(r["T_cont"], 2), fmt(r["T_peak"], 2),
            r["peak_limit"], fmt(r["P_cu"], 0), fmt(r["P_eddy"], 0), fmt(r["f_e"], 0),
            fmt(r["n_at_cont"], 0), fmt(r["axial_used"] * 1000, 1),
            fmt(r["m_mag"] * 1000, 0), fmt(r["m_total"] * 1000, 0),
            fmt(r["t_per_kg"], 2), fmt(r["t_per_l"], 1), fmt(r["cost"], 0),
            BUILDABILITY[name.split()[0]]))
    return "\n".join(out)


def electrical_table(rows):
    hdr = ("| Candidate | Packaging | pole pairs | coils | conductor w/d (mm) | magnet h (mm) | "
           "airgap (mm) | B_pk (T) | magnet blocks/rotor | J cont (A/mm2) | Kt (N.m/A_rms) | "
           "I cont (A_rms) | R_ph (mohm) | turns/phase | n no-load (rpm) |")
    sep = "|" + "---|" * 15
    out = [hdr, sep]
    for name, pack, r in rows:
        if r is None:
            continue
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, pack, r["p"], r["n_coils"], fmt(r["w"] * 1000, 3), fmt(r["h_m"] * 1000, 1),
            fmt(r["g_mag"] * 1000, 1), fmt(r["B_pk"], 2), r["n_blocks"],
            fmt(r["J"] / 1e6, 1), fmt(r["Kt"], 4), fmt(r["I_cont"], 1),
            fmt(r["R_ph"] * 1000, 1), fmt(r["N_ph"], 0), fmt(r["n_noload"], 0)))
    return "\n".join(out)


def joint_table(rows):
    hdr = "| Candidate | Packaging | " + " | ".join(
        f"{j[0]} ({j[2]}/{j[3]} N.m, {j[4]} rpm)" for j in JOINTS) + " |"
    sep = "|" + "---|" * (2 + len(JOINTS))
    out = [hdr, sep]
    for name, pack, r in rows:
        cells = []
        for _, _, tc, tp, nmin in JOINTS:
            if r is None:
                cells.append("—"); continue
            ok_c, ok_p = r["T_cont"] >= tc, r["T_peak"] >= tp
            ok_n = r["n_at_cont"] >= nmin
            if ok_c and ok_p and ok_n:
                cells.append("**yes**")
            else:
                miss = []
                if not ok_c: miss.append(f"cont {r['T_cont']:.2f}")
                if not ok_p: miss.append(f"peak {r['T_peak']:.2f}")
                if not ok_n: miss.append(f"speed {r['n_at_cont']:.0f}")
                cells.append("no (" + ", ".join(miss) + ")")
        out.append(f"| {name} | {pack} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def catalogue_table():
    return "\n".join([
        "| Motor | Source | Size | Mass | Continuous torque | Implied N.m/kg | Implied sigma |",
        "|---|---|---|---|---|---|---|",
        "| Kollmorgen TBM2G-11526 (frameless, slotted radial) | **verified** - TBM2G Selection Guide PDF, "
        "115 Series parameter page | 115 mm OD x 26 mm stack | 1.43 kg | 6.03 N.m at stall, 130 K rise, "
        "on a 305 x 305 x 12 mm Al plate (R_th 1.21 K/W) | 4.2 | "
        f"{SIGMA_TBM/1000:.1f} kPa (r_g taken as 0.39 x OD) |",
        "| Kollmorgen TBM2G-11508 | **verified**, same source | 115 mm OD x 8 mm stack | 0.644 kg | "
        "1.90 N.m, R_th 1.83 K/W | 3.0 | ~18 kPa |",
        "| T-Motor U15 II KV80 (radial outrunner) | **partly verified** - vendor store page, not a datasheet | "
        "147.5 x 64 mm | 1.74 kg | 143 A / 8580 W \"180 s\" rating -> ~14.9 N.m from Kt = 8.3/Kv "
        "(propeller-cooled burst, NOT a still-air continuous rating) | ~8.6 (with forced air) | ~22 kPa |",
        "| CubeMars AK80-64 (motor + 64:1 planetary) | **partly verified** - vendor page | 98 x 61.9 mm | "
        "0.85 kg | 48 N.m rated / 120 N.m peak at the joint | 56 at the joint | n/a (geared) |",
        "| MyActuator RMD-X8-Pro (9:1) | **partly verified** - vendor pages | pancake, ~110 mm | 0.71-1.2 kg | "
        "13 N.m nominal / 25 N.m peak | ~18 at the joint | n/a (geared) |",
    ])


# =============================================================================
# 7. SENSITIVITY
# =============================================================================

def sensitivity():
    """T_cont ~ B * sqrt(P_allow * fill * t_cu), so it goes as sqrt(1/R_th) and
    sqrt(fill).  Check that numerically on the two leading candidates."""
    cases = [("C1 wound, in-plane", "wire", "in-plane", 1, dict(coil_thick=COIL_THICK)),
             ("A3 PCB 12L 3oz, in-plane", "pcb", "in-plane", 1, dict(stack=PCB_STACKS["PCB 12L 3oz"])),
             ("C1 wound, stacked", "wire", "stacked", 1, dict(coil_thick=COIL_THICK)),
             ("A3 PCB 12L 3oz, stacked", "pcb", "stacked", 1, dict(stack=PCB_STACKS["PCB 12L 3oz"]))]
    out = ["| Case | R_th 1.0 | R_th 1.5 (nominal) | R_th 2.0 | fill x0.8 | fill x1.2 |",
           "|---|---|---|---|---|---|"]
    for label, kind, pack, n, kw in cases:
        vals = []
        for rth in (1.0, 1.5, 2.0):
            r = optimise(kind, pack, n, r_th=rth, **kw)
            vals.append(r["T_cont"] if r else float("nan"))
        for fs in (0.8, 1.2):
            r = optimise(kind, pack, n, fill_scale=fs, **kw)
            vals.append(r["T_cont"] if r else float("nan"))
        out.append("| {} | {} |".format(label, " | ".join(f"{v:.2f}" for v in vals)))
    return "\n".join(out)


# =============================================================================
# 8. FIGURE
# =============================================================================

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
C_INPLANE, C_STACKED = "#2a78d6", "#eb6834"


def figure(rows, path):
    names = []
    for n, _, _ in rows:
        if n not in names:
            names.append(n)
    def get(n, p, key):
        for a, b, r in rows:
            if a == n and b == p:
                return 0.0 if r is None else r[key]
        return 0.0

    y = np.arange(len(names))
    h = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=SURFACE)
    for ax, key, title, unit in (
            (axes[0], "t_per_kg", "Continuous torque per kilogram", "N·m/kg"),
            (axes[1], "t_per_l", "Continuous torque per litre of envelope", "N·m/L")):
        vi = [get(n, "in-plane", key) for n in names]
        vs = [get(n, "stacked", key) for n in names]
        ax.barh(y - h / 2, vi, h, color=C_INPLANE, label="in-plane (reducer inside)", zorder=3)
        ax.barh(y + h / 2, vs, h, color=C_STACKED, label="stacked (reducer beside)", zorder=3)
        span = max(max(vi), max(vs)) or 1
        for yy, v in list(zip(y - h / 2, vi)) + list(zip(y + h / 2, vs)):
            if v > 0:
                ax.text(v + span * 0.015, yy, f"{v:.2f}" if key == "t_per_kg" else f"{v:.1f}",
                        va="center", ha="left", fontsize=8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axes[0] else [], fontsize=8.5, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, span * 1.18)
        ax.set_xlabel(unit, fontsize=9, color=INK2)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color="#e3e2df", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c4")
        ax.tick_params(colors=INK2, length=0)
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper right", bbox_to_anchor=(0.995, 0.965), ncol=2,
               frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Joint-actuator motor candidates in the Ø170 × 42 mm envelope "
                 "— continuous rating at R_th = 1.5 K/W, 45 °C ambient",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.018, "A missing bar means no feasible design in that packaging's axial budget. "
             "Torque per litre uses the axial length the magnetics actually occupy. "
             "Mass excludes housing, bearings and reducer.", fontsize=8, color=INK2)
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# =============================================================================
# 9. MAIN
# =============================================================================

if __name__ == "__main__":
    rows = build_results()
    print(f"\nDerived constants: Br({T_MAGNET:.0f} C) = {BR:.3f} T, "
          f"V_ph,max = {V_PH_MAX:.2f} V, P_allow(PCB) = {(T_CU_MAX_PCB-T_AMB)/R_TH_NOMINAL:.0f} W, "
          f"P_allow(wire) = {(T_CU_MAX_WIRE-T_AMB)/R_TH_NOMINAL:.0f} W, "
          f"sigma_ref(TBM2G-11526) = {SIGMA_TBM/1000:.1f} kPa\n")
    print("### Results\n"); print(main_table(rows))
    print("\n### Electrical and magnetic detail\n"); print(electrical_table(rows))
    print("\n### Does it meet the joint? (motor-shaft ratings)\n"); print(joint_table(rows))
    print("\n### Off-the-shelf calibration points\n"); print(catalogue_table())
    print("\n### Sensitivity of continuous torque (N.m)\n"); print(sensitivity())
    figure(rows, "/home/user/Hexapod/docs/design/motor/torque-density.png")
    print("\nFigure written to docs/design/motor/torque-density.png")

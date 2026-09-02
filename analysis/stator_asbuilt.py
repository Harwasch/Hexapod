#!/opt/hw-py/bin/python
"""As-built stator analysis: resistance, Kt, torque, losses from the generated
board geometry (hw/stator/geometry.json + the generator's constants).

    /opt/hw-py/bin/python analysis/stator_asbuilt.py

Kt is computed from every radial conductor's position in the Halbach field:
a leg at angle a (from the coil centre) between r_a and r_b, carrying current i,
gives torque i * INT B(r) cos(p (a - th_rotor)) r dr; summed over the turns,
coils, phases with balanced sinusoidal currents at the best commutation angle.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor_options as mo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "hw", "stator", "geometry.json")
G = json.load(open(GEOM))
N_COILS, P, N_LAYERS, N_T = G["coils"], G["pole_pairs"], G["layers"], G["turns_per_layer"]
TRACE, SPACE, GAP = G["trace_mm"] * 1e-3, G["space_mm"] * 1e-3, G.get("gap_mm", 0.8) * 1e-3
R_IN, R_OUT = G["r_in_mm"] * 1e-3, G["r_out_mm"] * 1e-3
T_CU = 3 * mo.OZ                       # 3 oz copper
PITCH = TRACE + SPACE
T_BOARD = 2.2e-3                       # 12-layer 3 oz board
G_MAG = T_BOARD + 2 * mo.AIR_CLEAR
# magnets: the OTS block chosen in analysis/rotor_field.py; its simulated fundamental
# (two facing 4-segment Halbach rings of rectangular blocks) scales the analytic field
MAGNET = "rect 30x5x8 N48"
_rf_path = os.path.join(ROOT, "hw", "stator", "rotor_field.json")
if os.path.exists(_rf_path):
    _rf = json.load(open(_rf_path))[MAGNET]
    H_M = _rf["h_m_mm"] * 1e-3
    B_SCALE = _rf["ratio_to_model"]
    MAGNET_MASS_G = _rf["magnet_mass_g"]
else:
    _m = mo.eval_axial("pcb", "in-plane", P, TRACE, 1, stack=mo.PCB_STACKS["PCB 12L 3oz"])
    H_M = _m["h_m"] if _m else 4.4e-3      # magnet thickness the study's axial budget allows at this pole count
    B_SCALE, MAGNET_MASS_G = 1.0, None
R_TH, T_CU_MAX, T_AMB, V_PH = mo.R_TH_NOMINAL, mo.T_CU_MAX_PCB, mo.T_AMB, mo.V_PH_MAX

# the 12-coil repeat: (local coil, sign) per phase, from the star of slots
PHASE_COILS = {"A": [(0, +1), (6, -1), (7, +1), (1, -1)],
               "B": [(8, +1), (2, -1), (3, +1), (9, -1)],
               "C": [(4, +1), (10, -1), (11, +1), (5, -1)]}

# ---- resistance -----------------------------------------------------------------
L_coil_layer = G["coil_track_mm_per_coil_layer"] * 1e-3            # m, one spiral
rho = mo.rho_cu(T_CU_MAX)
A_tr = TRACE * T_CU
R_layer = rho * L_coil_layer / A_tr
R_coil = 2 * R_layer / (N_LAYERS / 2)          # 6 odd layers parallel, in series with 6 even parallel
R_repeat = 2 * (R_coil / 2)                    # two parallel pairs in series
N_REP = G.get("repeats", 4)
# interconnect, element by element (equivalent series resistance seen by the phase current I):
#   phase ring: each repeat feeds I/N_REP into the ring at a different angle; path to the pad
#   averages half of the half-circumference; 3 layers in parallel
r_ring = G["r_ring_mm"]["A"] * 1e-3
R_ring_path = rho * (math.pi * r_ring / 2) / (3 * G["ring_w_mm"] * 1e-3 * T_CU)
R_inter_ring = N_REP * (1.0 / N_REP) ** 2 * R_ring_path
#   M arc: carries the repeat current I/N_REP over its length on one layer
L_marc = G["marc_mm_per_phase"]["A"] * 1e-3 / N_REP
R_marc = rho * L_marc / (G["m_w_mm"] * 1e-3 * T_CU)
R_inter_m = N_REP * (1.0 / N_REP) ** 2 * R_marc
#   star jumpers: I/(2 N_REP) each, r_via_t -> r_n on 3 layers
L_j = (G["r_via_t_mm"] - G["r_n_mm"]) * 1e-3
R_j = rho * L_j / (3 * G["jumper_w_mm"] * 1e-3 * T_CU)
R_inter_j = 2 * N_REP * (1.0 / (2 * N_REP)) ** 2 * R_j
R_inter = R_inter_ring + R_inter_m + R_inter_j
R_ph = R_repeat / N_REP + R_inter

# ---- field and torque per amp ------------------------------------------------------
r = np.linspace(R_IN, R_OUT, 300)
B = mo.halbach_B(r, P, H_M, G_MAG) * B_SCALE                        # peak axial field (T) vs radius, OTS-block corrected


def half_ang(rr, i):
    return ((2 * math.pi * rr / N_COILS - GAP) / 2 - i * PITCH) / rr


LEG_ANG = np.array([[half_ang(rr, i) for rr in r] for i in range(N_T)])   # (turn, r)


def coil_tpa(th):
    """Torque per amp of one coil (both leg groups, all turns of one layer) with
    its centre at angle 0 and the rotor at electrical position p*th."""
    tot = 0.0
    for i in range(N_T):
        ang = LEG_ANG[i]
        tot += np.trapezoid(B * np.cos(P * (-ang - th)) * r, r)      # left leg, current inward (+)
        tot -= np.trapezoid(B * np.cos(P * (ang - th)) * r, r)       # right leg, current outward
    return tot


def torque(I_rms, phi, th):
    """Total torque with balanced currents; phase current splits over 4 repeats
    and 2 parallel coils (I/8 per coil); each coil has 2*N_T series turns per
    layer-group... i.e. the layer groups are in series (x2) and each group's
    N_T turns are already summed in coil_tpa."""
    I_pk = I_rms * math.sqrt(2)
    t = 0.0
    for ph, k_off in (("A", 0.0), ("B", -2 * math.pi / 3), ("C", 2 * math.pi / 3)):
        cur = I_pk * math.cos(P * th + phi + k_off) / (2 * N_REP)
        for k, sgn in PHASE_COILS[ph]:
            thc = 2 * math.pi * k / N_COILS
            t += cur * sgn * 2 * coil_tpa(th - thc) * N_REP      # 2 layer groups in series, N_REP repeats
    return t


ths = np.linspace(0, 2 * math.pi / P, 24, endpoint=False)
Kt = max(np.mean([torque(1.0, phi, th) for th in ths]) for phi in np.linspace(0, 2 * math.pi, 72, endpoint=False))
ripple = None
best_phi = max(np.linspace(0, 2 * math.pi, 72, endpoint=False), key=lambda phi: np.mean([torque(1.0, phi, th) for th in ths]))
tr = [torque(1.0, best_phi, th) for th in ths]
ripple = (max(tr) - min(tr)) / np.mean(tr)

# ---- losses and ratings --------------------------------------------------------------
vol_cu = G["coil_track_mm_total"] * 1e-3 * A_tr
m_cu = vol_cu * mo.RHO_CU_DENS


def rating(rpm):
    f = P * rpm / 60
    pv = mo.eddy_pv_strip(f, B, TRACE)
    P_eddy = float(np.trapezoid(pv * (TRACE / PITCH) * N_LAYERS * T_CU * 2 * math.pi * r, r))
    P_allow = (T_CU_MAX - T_AMB) / R_TH
    P_cu = max(P_allow - P_eddy, 0.0)
    I = math.sqrt(P_cu / (3 * R_ph))
    return dict(rpm=rpm, f_e=f, P_eddy=P_eddy, P_cu=P_cu, I_cont=I, T_cont=Kt * I, T_peak=Kt * 3 * I,
                drag_Nm=P_eddy / (rpm * 2 * math.pi / 60))


n_noload = 3 * V_PH / Kt * 60 / (2 * math.pi)
out = dict(series_turns_per_phase=4 * N_T, R_ph_mohm=R_ph * 1e3, R_inter_mohm=R_inter * 1e3, Kt=Kt, torque_ripple=ripple,
           n_noload_rpm=n_noload, copper_mass_g=m_cu * 1e3, copper_volume_cm3=vol_cu * 1e6, B_pk_mean=float(np.mean(B)),
           ratings={str(rpm): rating(rpm) for rpm in (1000, 1600, 2500, 3500)})
out["h_m_mm"] = H_M * 1e3
out["magnet"] = MAGNET
out["magnet_mass_g"] = MAGNET_MASS_G
out["B_scale_from_rotor_field"] = B_SCALE
json.dump(out, open(GEOM.replace("geometry", "asbuilt"), "w"), indent=1)
if __name__ == "__main__":
    print(f"magnets {MAGNET}: h_m {H_M*1e3:.1f} mm, field scale {B_SCALE:.2f}, mass {MAGNET_MASS_G} g")
    print(f"p {P}, {N_COILS} coils, n_t {N_T}, trace {TRACE*1e3:.2f}: series turns/phase {4*N_T}, R_ph {R_ph*1e3:.1f} mΩ (coil {R_repeat/N_REP*1e3:.1f} + ring {R_inter_ring*1e3:.1f} + M {R_inter_m*1e3:.1f} + jumpers {R_inter_j*1e3:.1f}), Kt {Kt:.3f} N·m/A_rms, "
          f"ripple {ripple*100:.1f} %, no-load {n_noload:.0f} rpm, copper {m_cu*1e3:.0f} g, B_pk mean {np.mean(B):.2f} T")
    for k, v in out["ratings"].items():
        print(f"  {k} rpm: f_e {v['f_e']:.0f} Hz, eddy {v['P_eddy']:.0f} W, cu {v['P_cu']:.0f} W, I {v['I_cont']:.1f} A, "
              f"T_cont {v['T_cont']:.2f}, T_peak {v['T_peak']:.2f} N·m, drag {v['drag_Nm']:.2f} N·m")

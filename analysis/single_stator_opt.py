#!/opt/hw-py/bin/python
"""Round 13: the best single-stator PCB axial-flux motor inside a Ø190 housing.

    /opt/hw-py/bin/python analysis/single_stator_opt.py            # full sweep (~3 min)
    /opt/hw-py/bin/python analysis/single_stator_opt.py --quick    # coarse sweep for a smoke test

The hard limit is the housing: 190 mm over the wall.  The wall is 4 mm of
radius outside the board (cad/actuator/actuator.py: R_OD 95.9 over a Ø184
board), so the board is Ø182 and every rim radius of the v2 layout moves
inward by 0.9 mm together (hw/stator/make_stator.py --od 182).

What is swept, on boards the generator actually lays out (make_stator.py is
run for every coil count / turns / r_in / leg-gap combination, so the coil
track length, the trace width the leg allows and the interconnect are the
built numbers, not a sheet model), and rated the way analysis/stator_asbuilt.py
rates the canonical board (same field, same resistance, same eddy loss, same
50 W thermal budget: 120 C copper, 45 C ambient, 1.5 K/W):

  * coils / pole pairs        24/10, 36/15, 48/20 (12-slot/10-pole family)
  * layers x copper           6, 8, 10, 12, 14, 16, 20 layers at 2 oz (JLCPCB), a
                              bonded pair of 6, 8 or 10-layer boards, and 12L 3 oz
                              as the non-JLCPCB reference
  * turns per layer           every count the leg fits at >= 0.15 mm trace
  * r_in                      53 (v2) and 52 (the star ring still fits inside)
  * coil-to-coil leg gap      0.6 (v2) and 0.4 mm
  * magnets                   30x5x6, 30x5x8, 30x5x10 N48 blocks (the field from the
                              same 2-D block model as analysis/rotor_field.py, run
                              here at each pole count and gap; the rotor pull sizes
                              the carrier plates)
  * mechanical clearance      0.35 / 0.5 / 0.7 mm, on the chosen board

Kt is the closed form of stator_asbuilt.py's numeric average: with a sinusoidal
field the torque of one coil is A sin(p th), so the mean over rotor position at
the best commutation angle is A |S| / sqrt 2 with S the star-of-slots phasor sum
(checked against the numeric value for the canonical board below).

Writes hw/stator/single_stator_opt.json and docs/design/actuator/1s-opt-{sweep,
envelope,section}.png.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor_options as mo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "design", "actuator")
GEN = os.path.join(ROOT, "hw", "stator", "make_stator.py")
QUICK = "--quick" in sys.argv
SCRATCH = os.environ.get("HEXAPOD_SCRATCH", os.path.join(ROOT, "build", "stator-sweep"))
os.makedirs(SCRATCH, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
RF = json.load(open(os.path.join(ROOT, "hw", "stator", "rotor_field.json")))
CL = json.load(open(os.path.join(ROOT, "hw", "stator", "closure.json")))
CAD2 = json.load(open(os.path.join(ROOT, "cad", "actuator", "femur.json")))          # the canonical two-stator femur (round 9 parts)
CADY = json.load(open(os.path.join(ROOT, "cad", "actuator", "yaw.json")))
AB = json.load(open(os.path.join(ROOT, "hw", "stator", "asbuilt.json")))
G0 = json.load(open(os.path.join(ROOT, "hw", "stator", "geometry.json")))

# ---- the limit and the housing --------------------------------------------------------
OD_MAX = 190.0                    # mm over the housing wall, the hard limit
WALL_R = 4.0                      # mm of radius from the board edge to the housing outside (cad/actuator: 95.9 - 91.9)
BOARD_OD = OD_MAX - 2 * WALL_R    # 182
R_BOARD = BOARD_OD / 2            # 91.0
RIM_SHIFT = R_BOARD - G0["board_od_mm"] / 2      # -0.9: every rim radius of v2 moves in by this
R_BORE = 50.0
R_MAG_IN0, R_MAG_OUT0 = 55.0, 85.0               # v2 magnet span; the rings move in with the coils
MAG_L = 30.0                                     # block length, radial
# ---- rating conventions (stator_asbuilt.py) ----------------------------------------------
R_TH, T_CU_MAX, T_AMB, V_PH = mo.R_TH_NOMINAL, mo.T_CU_MAX_PCB, mo.T_AMB, mo.V_PH_MAX
P_ALLOW = (T_CU_MAX - T_AMB) / R_TH              # 50 W per board
PEAK_MULT = 3.0
SPACE = 0.15
# ---- the joints -----------------------------------------------------------------------------
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import cycloid as cy                              # noqa: E402  (ratios, efficiencies, the sizing's joint speeds)
ETA_CYC, ETA_CAP = 0.90, cy.ETA2["femur"]         # closure.py / cost_search.py conventions
RATIO_FK = {80: (20, 4.0), 100: (25, 4.0)}        # total ratio -> (cycloid lobes, capstan)
RATIO_YAW = 30
RPM_RATE = {80: 1000.0, 100: 1250.0}              # the stance speed the continuous torque is rated at (1000 rpm at 80:1, as every prior round)
W_SWING_FK, W_SWING_YAW = cy.SWING["femur"], cy.SWING["yaw"]    # 3.77 / 6.4 rad/s at the joint
SWING_MARGIN = 1.03                               # no-load speed at 48 V must exceed the swing speed by this: the IR drop of the swing current (leg inertia) is a few %
C_PER_KG = CL["torque_per_kg"]                    # N·m per kg of robot, continuous: femur 2.53, knee 2.67, yaw 1.02
M_FIXED = CL["m_fixed"]                           # 29.2 kg: body, legs, batteries, electronics, margin
PAYLOAD = 8.0                                     # kg, mission payload the relief cases carry (closure.py)
# ---- costs ------------------------------------------------------------------------------------
PCB_6L_2OZ = {20: 15.0, 100: 11.0}                # the reviewer's JLCPCB instant quote, 192 x 192, 6L 2 oz; scaled by layer count
PCB_12L_3OZ = {20: 65.0, 100: 50.0}               # PCBWay-class heavy copper, estimate (motor_options COST_PCB; 100-off guessed)
MAG_6MM = {20: 0.25, 100: 0.20}                   # per 30x5x6 block, scaled by volume
ROTOR_CUP_1S = {20: 45.0, 100: 28.0}              # two carrier plates + drum (the two-stator cup with three plates is $60 / $38 in the BOM)
# ---- masses (g) -----------------------------------------------------------------------------
AL, STEEL, NDFEB, FR4 = 2.70e-3, 7.85e-3, 7.50e-3, 1.85e-3      # g/mm3
R_CARRIER_OUT, R_DRUM_OUT, R_DRUM_IN, R_HUB = 85.5, 49.5, 46.5, 12.5
M_TOP_PER_MM = math.pi * (R_CARRIER_OUT**2 - R_HUB**2) * AL         # top carrier disc, g per mm of thickness
M_BOT_PER_MM = math.pi * (R_CARRIER_OUT**2 - R_DRUM_OUT**2) * AL    # bottom carrier ring
M_DRUM_PER_MM = math.pi * (R_DRUM_OUT**2 - R_DRUM_IN**2) * AL       # drum
M_WALL_PER_MM = math.pi * (95.0**2 - 90.4**2) * AL                   # wall tube / upper ring at Ø190
T_CARRIER0, F_ATTRACT0, DEFL0 = 4.5, 2300.0, 0.06                    # the CAD's carrier: 4.5 mm 6061 deflects 0.06 mm under 2.3 kN (8 mm blocks)
# the 1s unit: the canonical femur's round-9 parts minus the second stator stage (wall height, one board, two rings, one carrier)
_m2 = CAD2["mass_g"]
M_FIXED_FK = sum(_m2[k] for k in ("floor_plate", "bearing_carrier", "pin_cage", "ring_pins", "cover", "shaft", "disc", "capstan_drum", "output_flange", "output_pins", "bearings"))
_my = CADY["mass_g"]
M_FIXED_YAW = sum(_my[k] for k in ("floor_plate", "bearing_carrier", "pin_cage", "ring_pins", "cover", "shaft", "disc", "output_flange", "output_pins", "bearings"))
H_WALL_2S = (_m2["wall_tube"] + _m2["upper_ring"]) / (math.pi * (95.9**2 - 90.4**2) * AL)    # mm of wall the two-stator unit carries
H_BAND_2S = 2 * (2.2 + 1.0 + 2 * 6.0) + 3 * 4.5                                              # its motor band (6 mm blocks, 2.2 mm boards, 4.5 mm carriers)
Z_TOPCAR0_1S = 39.7                               # the reducer sets the height of a one-stator unit (cad/actuator femur-1s: two discs, cylinder top 39.2 + 0.5)
Z_BAND_FLOOR = 3.0 + 0.5 + 1.5                    # floor plate 3 + clearance + the drum lip: the lowest the bottom carrier may go


# =====================================================================================
# 1. Rotor field: the 2-D block model of analysis/rotor_field.py, vectorised, at any pole count and gap
# =====================================================================================
_FIELD_CACHE = {}


def block_field(P, r_m_mm, w_b_mm, h_m_mm, gap_mm, nper=5, npts=240):
    """Two facing 4-segment Halbach rings of rectangular blocks at the mean radius:
    fundamental B1 at the mid-plane, the study's analytic field there, their ratio,
    and the Maxwell-stress pull between the rotors (kPa).  Same charges, same
    discretisation (40 line charges per face) as rotor_field.py."""
    key = (P, round(r_m_mm, 1), w_b_mm, h_m_mm, round(gap_mm, 2))
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]
    M = mo.BR / mo.MU0
    lam = 2 * math.pi * r_m_mm * 1e-3 / P
    seg = lam / 4
    w, h, g = w_b_mm * 1e-3, h_m_mm * 1e-3, gap_mm * 1e-3
    x = np.linspace(0, lam, npts, endpoint=False)
    pattern = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    # every face charge of every block of both rotors: arrays of (xq, zq, q)
    xs, zs, qs = [], [], []
    n_pts = 40
    for k in range(-nper * 4, nper * 4 + 4):
        xc = k * seg
        mx, mz = pattern[k % 4]
        for zc, sx in ((-(g / 2 + h / 2), 1), (+(g / 2 + h / 2), -1)):     # rotor 2 mirrors the tangential component
            mxx = mx * sx
            if mz:
                xf = np.linspace(xc - w / 2, xc + w / 2, n_pts)
                xs += [xf, xf]; zs += [np.full_like(xf, zc + h / 2), np.full_like(xf, zc - h / 2)]
                qs += [np.full_like(xf, mz * M * w / n_pts), np.full_like(xf, -mz * M * w / n_pts)]
            if mxx:
                zf = np.linspace(zc - h / 2, zc + h / 2, n_pts)
                xs += [np.full_like(zf, xc + w / 2), np.full_like(zf, xc - w / 2)]; zs += [zf, zf]
                qs += [np.full_like(zf, mxx * M * h / n_pts), np.full_like(zf, -mxx * M * h / n_pts)]
    xq, zq, q = np.concatenate(xs), np.concatenate(zs), np.concatenate(qs)

    def field(xp, zp):
        dx = xp[:, None] - xq[None, :]
        dz = zp[:, None] - zq[None, :]
        r2 = dx * dx + dz * dz + 1e-12
        c = mo.MU0 / (2 * math.pi) * q[None, :]
        return (c * dx / r2).sum(1), (c * dz / r2).sum(1)

    bx0, bz0 = field(x, np.zeros_like(x))
    b1 = 2 * np.trapezoid(bz0 * np.cos(2 * math.pi * x / lam), x) / lam
    sigma = float(np.mean((bz0**2 - bx0**2) / (2 * mo.MU0)))
    out = dict(B1=float(b1), B_peak=float(np.max(np.abs(bz0))), attraction_kPa=sigma * 1e-3, lambda_mm=lam * 1e3, seg_mm=seg * 1e3)
    _FIELD_CACHE[key] = out
    return out


def field_ratio(P, r_in, r_out, w_b, h_m, gap):
    """Block-to-model ratio the way stator_asbuilt.py uses it: B1 of the block model at r_m
    over the mean of motor_options.halbach_B across the coil span."""
    r_m = 0.5 * (r_in + r_out)
    bf = block_field(P, r_m, w_b, h_m, gap)
    model = float(np.mean(mo.halbach_B(np.linspace(r_in, r_out, 50) * 1e-3, P, h_m * 1e-3, gap * 1e-3)))
    area = math.pi * (((r_m + 15) * 1e-3)**2 - ((r_m - 15) * 1e-3)**2)
    return dict(ratio=bf["B1"] / model, B1=bf["B1"], attraction_N=bf["attraction_kPa"] * 1e3 * area, attraction_kPa=bf["attraction_kPa"], seg_mm=bf["seg_mm"])


def magnet_block(P, r_in):
    """Block widths to try at this pole count: the 5 mm OTS listing (30x5xH) if it fits the
    Halbach segment at the rings' inner radius, and the widest 0.5-mm-step custom block
    that does (0.3 mm glue gap).  Returns [(w_b, ots?), ...]."""
    seg_in = 2 * math.pi * (R_MAG_IN0 + RIM_SHIFT) / (4 * P)          # segment arc at the magnet ring's inner radius
    w_max = math.floor((seg_in - 0.3) * 2) / 2
    out = []
    if w_max >= 5.0:
        out.append((5.0, True))
    if w_max != 5.0:
        out.append((w_max, False))
    return out


# =====================================================================================
# 2. Boards: run the generator, read what it built
# =====================================================================================
def gen_board(coils, pp, turns, r_in, gap, od=BOARD_OD, layers=12, oz=2.0, out_dir=None, t_board=None, trace=None):
    """Lay the board out with make_stator.py and return its geometry.json (cached).  The
    per-layer coil length and the interconnect do not depend on the layer count, so the
    sweep lays each coil geometry out once (12 layers) and rates every stack from it."""
    tag = f"c{coils}p{pp}t{turns}r{r_in:g}g{gap:g}od{od:g}L{layers}oz{oz:g}" + (f"tr{trace:g}" if trace else "")
    d = out_dir or os.path.join(SCRATCH, tag)
    geo = os.path.join(d, "geometry.json")
    if not os.path.exists(geo) or out_dir:
        os.makedirs(d, exist_ok=True)
        cmd = ["python3", GEN, "--coils", str(coils), "--pp", str(pp), "--turns", str(turns), "--r-in", str(r_in), "--gap", str(gap),
               "--od", str(od), "--layers", str(layers), "--oz", str(oz), "--out", os.path.join(d, "stator.kicad_pcb")]
        if t_board:
            cmd += ["--t-board", str(t_board)]
        if trace:
            cmd += ["--trace", str(trace)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return None
    return json.load(open(geo))


def board_thickness(n, oz):
    return round(n * oz * 0.035 + (n - 1) * 0.09, 2)


STACKS = [  # name, layers, oz, boards bonded, JLCPCB?
    ("6L 2oz", 6, 2.0, 1, True), ("8L 2oz", 8, 2.0, 1, True), ("10L 2oz", 10, 2.0, 1, True), ("12L 2oz", 12, 2.0, 1, True),
    ("14L 2oz", 14, 2.0, 1, True), ("16L 2oz", 16, 2.0, 1, True), ("20L 2oz", 20, 2.0, 1, True),
    ("2x6L 2oz", 12, 2.0, 2, True), ("2x8L 2oz", 16, 2.0, 2, True), ("2x10L 2oz", 20, 2.0, 2, True),
    ("12L 3oz", 12, 3.0, 1, False),
]


def stack_thickness(layers, oz, boards):
    if boards == 1:
        return board_thickness(layers, oz)
    return round(boards * board_thickness(layers // boards, oz) + 0.1 * (boards - 1), 2)    # 0.1 mm adhesive film per joint


def board_cost(layers, oz, boards, qty):
    if oz >= 3:
        return PCB_12L_3OZ[qty] * layers / 12
    return PCB_6L_2OZ[qty] * layers / 6           # linear in layer count from the 6-layer quote; a pair is two boards of half the count


# =====================================================================================
# 3. Rating: stator_asbuilt.py's formulas with the field and the stack as parameters
# =====================================================================================
PHASE_COILS = {"A": [(0, +1), (6, -1), (7, +1), (1, -1)], "B": [(8, +1), (2, -1), (3, +1), (9, -1)], "C": [(4, +1), (10, -1), (11, +1), (5, -1)]}


def rate(G, layers, oz, t_board, h_m, w_b, clr=mo.AIR_CLEAR * 1e3, rpms=(1000.0, 1250.0), ring_per=None, nm_layers=3, ratio_override=None):
    P, N_COILS, N_T = G["pole_pairs"], G["coils"], G["turns_per_layer"]
    TRACE, GAP = G["trace_mm"] * 1e-3, G["gap_mm"] * 1e-3
    R_IN, R_OUT = G["r_in_mm"] * 1e-3, G["r_out_mm"] * 1e-3
    T_CU = oz * mo.OZ
    PITCH = TRACE + SPACE * 1e-3
    N_REP = G["repeats"]
    g_mag = t_board + 2 * clr                                       # mm, magnet face to magnet face
    fr = field_ratio(P, G["r_in_mm"], G["r_out_mm"], w_b, h_m, g_mag)
    if ratio_override is not None:
        fr = dict(fr, ratio=ratio_override)
    rho = mo.rho_cu(T_CU_MAX)
    if ring_per is None:
        ring_per = min(3, max(1, (layers - 3) // 3))                # make_stator.set_layers
    # resistance, element by element
    A_tr = TRACE * T_CU
    R_layer = rho * G["coil_track_mm_per_coil_layer"] * 1e-3 / A_tr
    R_coil = 2 * R_layer / (layers / 2)
    R_repeat = 2 * (R_coil / 2)
    r_ring = G["r_ring_mm"]["A"] * 1e-3
    R_ring_path = rho * (math.pi * r_ring / 2) / (ring_per * G["ring_w_mm"] * 1e-3 * T_CU)
    R_inter_ring = N_REP * (1.0 / N_REP)**2 * R_ring_path
    L_marc = G["marc_mm_per_phase"]["A"] * 1e-3 / N_REP
    R_inter_m = N_REP * (1.0 / N_REP)**2 * rho * L_marc / (G["m_w_mm"] * 1e-3 * T_CU)
    L_j = (G["r_via_t_mm"] - G["r_n_mm"]) * 1e-3
    R_inter_j = 2 * N_REP * (1.0 / (2 * N_REP))**2 * rho * L_j / (nm_layers * G["jumper_w_mm"] * 1e-3 * T_CU)
    R_inter = R_inter_ring + R_inter_m + R_inter_j
    R_ph = R_repeat / N_REP + R_inter
    # field and Kt (closed form of the numeric average in stator_asbuilt.py)
    r = np.linspace(R_IN, R_OUT, 300)
    B = mo.halbach_B(r, P, h_m * 1e-3, g_mag * 1e-3) * fr["ratio"]
    A = 0.0
    for i in range(N_T):
        ang = ((2 * math.pi * r / N_COILS - GAP) / 2 - i * PITCH) / r
        A += 2 * np.trapezoid(B * r * np.sin(P * ang), r)
    S = 0.0
    for ph, k_off in (("A", 0.0), ("B", -2 * math.pi / 3), ("C", 2 * math.pi / 3)):
        for k, sgn in PHASE_COILS[ph]:
            S += sgn * np.exp(1j * (k_off + P * 2 * math.pi * k / N_COILS))
    Kt = float(A * abs(S) / math.sqrt(2))                          # N·m per A rms, 2 layer groups in series x N_REP repeats / (2 N_REP) current split cancel
    k_w = abs(S) / 12                                                # 4 coils per phase x 3 phases, unity if all phasors aligned
    # losses and ratings
    vol_cu = G["coil_track_mm_total"] * 1e-3 * A_tr * layers / G["layers"]
    m_cu = vol_cu * mo.RHO_CU_DENS * 1e3
    n_noload = 3 * V_PH / Kt * 60 / (2 * math.pi)

    def at(rpm):
        f = P * rpm / 60
        pv = mo.eddy_pv_strip(f, B, TRACE)
        P_eddy = float(np.trapezoid(pv * (TRACE / PITCH) * layers * T_CU * 2 * math.pi * r, r))
        P_cu = max(P_ALLOW - P_eddy, 0.0)
        I = math.sqrt(P_cu / (3 * R_ph))
        n_v = max(0.0, 3 * (V_PH - I * R_ph) / Kt * 60 / (2 * math.pi))          # voltage-limited speed at the continuous current
        return dict(rpm=rpm, f_e=f, P_eddy=P_eddy, P_cu=P_cu, I_cont=I, T_cont=Kt * I, T_peak=Kt * PEAK_MULT * I, n_at_cont=n_v)

    return dict(Kt=Kt, k_w=k_w, R_ph_mohm=R_ph * 1e3, R_inter_mohm=R_inter * 1e3, B_pk_mean=float(np.mean(B)), B1_block=fr["B1"], field_ratio=fr["ratio"],
                attraction_N=fr["attraction_N"], n_noload_rpm=n_noload, copper_mass_g=m_cu, g_mag_mm=g_mag, ring_layers=ring_per,
                ratings={f"{rpm:g}": at(rpm) for rpm in rpms}, A_int=float(A))


# =====================================================================================
# 4. Mass, cost, the unit and the robot
# =====================================================================================
def carrier_t(F):
    """Carrier plate thickness for the same 0.06 mm deflection as the CAD's 4.5 mm plate under 2.3 kN (w ~ F / t^3), 0.5 mm steps, 3 mm floor."""
    return max(3.0, math.ceil(2 * T_CARRIER0 * (F / F_ATTRACT0)**(1 / 3)) / 2)


def unit(G, R, layers, oz, boards, h_m, w_b, ratio_fk):
    """Masses (g) and costs ($) of the stator + rotor and of the whole femur / yaw unit."""
    P = G["pole_pairs"]
    t_b = R["g_mag_mm"] - 2 * mo.AIR_CLEAR * 1e3
    m_board = math.pi * (R_BOARD**2 - R_BORE**2) * t_b * FR4 + R["copper_mass_g"]
    n_mag = 2 * 4 * P
    m_mag = n_mag * MAG_L * w_b * h_m * NDFEB
    t_c = carrier_t(R["attraction_N"])
    h_band = t_b + 2 * mo.AIR_CLEAR * 1e3 + 2 * h_m + 2 * t_c
    m_car = M_TOP_PER_MM * t_c + M_BOT_PER_MM * t_c + M_DRUM_PER_MM * (h_band + 3.0)
    z_botcar0 = Z_TOPCAR0_1S - (h_band - t_c)
    fits = z_botcar0 >= Z_BAND_FLOOR
    h_unit = Z_TOPCAR0_1S + t_c + 0.5 + 5.0
    h_wall = H_WALL_2S - (H_BAND_2S - h_band)
    m_wall = M_WALL_PER_MM * h_wall
    m_fk = M_FIXED_FK + m_board + m_mag + m_car + m_wall
    m_yaw = M_FIXED_YAW + m_board + m_mag + m_car + m_wall
    cost = {}
    for qty in (20, 100):
        c_b = board_cost(layers, oz, boards, qty)
        c_m = n_mag * MAG_6MM[qty] * (h_m / 6.0) * (w_b / 5.0)
        cost[qty] = dict(boards=c_b, magnets=c_m, cup=ROTOR_CUP_1S[qty], total=c_b + c_m + ROTOR_CUP_1S[qty])
    return dict(t_board_mm=t_b, m_board_g=m_board, m_magnets_g=m_mag, n_magnets=n_mag, t_carrier_mm=t_c, m_carriers_g=m_car, m_motor_g=m_board + m_mag + m_car,
                h_band_mm=h_band, band_fits=fits, h_unit_mm=h_unit, m_unit_fk_kg=m_fk / 1e3, m_unit_yaw_kg=m_yaw / 1e3, cost=cost)


def closure(T_motor_fk, T_motor_yaw, m_fk, m_yaw, ratio_fk):
    """The fixed point: 29.2 kg + 12 femur/knee + 6 yaw units; needs per kg from the sizing."""
    m = M_FIXED + 12 * m_fk + 6 * m_yaw
    T_fk = T_motor_fk * ratio_fk * ETA_CYC * ETA_CAP
    T_yaw = T_motor_yaw * RATIO_YAW * ETA_CYC
    need = {d: C_PER_KG[d] * m for d in ("femur", "knee", "yaw")}
    margin = dict(femur=T_fk / need["femur"], knee=T_fk / need["knee"], yaw=T_yaw / need["yaw"])
    m_support = min(T_fk / C_PER_KG["femur"], T_fk / C_PER_KG["knee"], T_yaw / C_PER_KG["yaw"])     # the robot mass the unit's torque would carry
    ratio_to_close = max(C_PER_KG["femur"], C_PER_KG["knee"]) * m / max(T_motor_fk * ETA_CYC * ETA_CAP, 1e-9)
    return dict(m_robot_kg=m, T_fk=T_fk, T_yaw=T_yaw, need=need, margin=margin, closes=min(margin.values()) >= 1.0,
                m_robot_supported_kg=m_support, ratio_fk_to_close=ratio_to_close)


# =====================================================================================
# 5. The sweep
# =====================================================================================
def sweep():
    rows = []
    coil_sets = [(24, 10), (36, 15), (48, 20)]
    r_in, gap = 53.0, 0.6                                  # r_in and the leg gap are sensitivities on the chosen board (small levers, see main)
    fills = [1.0] if QUICK else [1.0, 0.7, 0.5, 0.35]      # trace width as a fraction of what fills the leg: narrow traces cut the eddy loss of a fast (few-turn) board
    mags = [6.0, 8.0] if QUICK else [6.0, 8.0, 10.0]
    stacks = [s for s in STACKS if s[0] in ("8L 2oz", "12L 2oz", "12L 3oz")] if QUICK else STACKS
    for coils, pp in coil_sets:
        leg = (2 * math.pi * r_in / coils - gap) / 2
        t_max = int(leg / (0.15 + SPACE))
        turns_list = list(range(4, t_max + 1)) if not QUICK else [4, max(4, t_max - 4), t_max]
        for turns in turns_list:
            for fill in fills:
                trace_full = math.floor((leg / turns - SPACE) * 200) / 200
                trace = None if fill == 1.0 else math.floor(trace_full * fill * 200) / 200
                if trace is not None and trace < 0.15:
                    continue
                G = gen_board(coils, pp, turns, r_in, gap, trace=trace)
                if G is None:
                    continue
                if True:
                    for (w_b, ots), h_m in [(wb, h) for wb in magnet_block(pp, r_in) for h in mags]:
                        for name, layers, oz, boards, jlc in stacks:
                            t_b = stack_thickness(layers, oz, boards)
                            R = rate(G, layers, oz, t_b, h_m, w_b)
                            for ratio in (80, 100):
                                r1 = R["ratings"][f"{RPM_RATE[ratio]:g}"]
                                if r1["T_cont"] < 0.05:                                   # eddy loss alone exceeds the 50 W budget
                                    continue
                                U = unit(G, R, layers, oz, boards, h_m, w_b, ratio)
                                if not U["band_fits"]:
                                    continue
                                C = closure(r1["T_cont"], r1["T_cont"], U["m_unit_fk_kg"], U["m_unit_yaw_kg"], ratio)
                                n_swing = W_SWING_FK * ratio * 60 / (2 * math.pi)
                                rows.append(dict(coils=coils, pp=pp, turns=turns, trace_mm=G["trace_mm"], trace_fill=fill, r_in=r_in, r_out=G["r_out_mm"], leg_gap=gap, stack=name, layers=layers, oz=oz,
                                                 boards=boards, jlcpcb=jlc, t_board_mm=t_b, h_m=h_m, w_b=w_b, ots_block=ots, ratio=ratio, rpm=RPM_RATE[ratio],
                                                 Kt=R["Kt"], R_ph_mohm=R["R_ph_mohm"], B_pk_mean=R["B_pk_mean"], I_cont=r1["I_cont"], P_cu=r1["P_cu"], P_eddy=r1["P_eddy"],
                                                 T_cont=r1["T_cont"], T_peak=r1["T_peak"], n_noload=R["n_noload_rpm"], n_at_cont=r1["n_at_cont"], n_swing=n_swing,
                                                 swing_ok=R["n_noload_rpm"] >= SWING_MARGIN * n_swing, attraction_N=R["attraction_N"], t_carrier=U["t_carrier_mm"],
                                                 m_motor_kg=U["m_motor_g"] / 1e3, m_unit_kg=U["m_unit_fk_kg"], cost20=U["cost"][20]["total"], cost100=U["cost"][100]["total"],
                                                 T_per_dollar20=r1["T_cont"] / U["cost"][20]["total"], T_per_kg=r1["T_cont"] / (U["m_motor_g"] / 1e3),
                                                 m_robot=C["m_robot_kg"], margin_knee=C["margin"]["knee"], margin_yaw=C["margin"]["yaw"], closes=C["closes"],
                                                 m_supported=C["m_robot_supported_kg"], ratio_to_close=C["ratio_fk_to_close"], copper_g=R["copper_mass_g"]))
    return rows


# =====================================================================================
# 6. Checks against the existing tools
# =====================================================================================
def checks():
    out = {}
    # (a) the block field against rotor_field.json (P = 15, gap 3.2 mm, the canonical coil span)
    fc = []
    for name in ("rect 30x5x6 N48", "rect 30x5x8 N48", "rect 30x5x10 N48"):
        ref = RF[name]
        fr = field_ratio(15, G0["r_in_mm"], G0["r_out_mm"], 5.0, ref["h_m_mm"], 3.2)
        fc.append(dict(case=name, B1_here=fr["B1"], B1_rotor_field=ref["B1_midplane"], ratio_here=fr["ratio"], ratio_rotor_field=ref["ratio_to_model"],
                       pull_here_N=fr["attraction_N"], pull_rotor_field_N=ref["attraction_N"]))
    out["field_vs_rotor_field"] = fc
    out["field_note"] = ("rotor_field.json was written at the v1 coil span (r 53-80.2, r_m 66.6, a 7.0 mm segment) and not re-run for v2 (r_m 68.8); "
                         "the same block model at v2's mean radius gives ratios ~3 % lower, which is what this sweep uses at every pole count and gap")
    # (b) the closed-form Kt against stator_asbuilt.py's numeric average on the canonical board
    R = rate(G0, G0["layers"], G0["copper_oz"], G0["t_board_mm"], 8.0, 5.0, ratio_override=AB["B_scale_from_rotor_field"])
    out["kt_vs_asbuilt"] = dict(note="same field ratio as asbuilt.json forced, so this isolates the closed-form Kt against the numeric average",
                                Kt_here=R["Kt"], Kt_asbuilt=AB["Kt"], R_ph_here=R["R_ph_mohm"], R_ph_asbuilt=AB["R_ph_mohm"],
                                T1000_here=R["ratings"]["1000"]["T_cont"], T1000_asbuilt=AB["ratings"]["1000"]["T_cont"],
                                field_ratio_here=R["field_ratio"], field_ratio_asbuilt=AB["B_scale_from_rotor_field"])
    return out


def closed_form(G, R, layers, w_b, h_m):
    """T = k_w · N_coils · N_turns · B · I · L · r_mean, the textbook estimate, against the leg-by-leg integral."""
    P, N_COILS, N_T = G["pole_pairs"], G["coils"], G["turns_per_layer"]
    N_REP = G["repeats"]
    L = (G["r_out_mm"] - G["r_in_mm"]) * 1e-3
    r_m = 0.5 * (G["r_in_mm"] + G["r_out_mm"]) * 1e-3
    B1 = R["B_pk_mean"]                                        # fundamental peak, mean over the span
    I_coil_pk = math.sqrt(2) / (2 * N_REP)                     # per coil, per A rms of phase current
    n_series = 2 * N_T                                         # two layer groups in series
    # (3/2) x [coils per phase] x turns x (two legs: 2 B I L r) with the winding factor
    T_text = 1.5 * mo.K_W * (N_COILS / 3) * n_series * 2 * B1 * I_coil_pk * L * r_m
    # the same with each turn's own pitch factor sin(p * half-angle at r_m): the inner turns of a spiral span a fraction of the pole
    PITCH = (G["trace_mm"] + G["space_mm"]) * 1e-3
    GAP = G["gap_mm"] * 1e-3
    kp = [math.sin(P * (((2 * math.pi * r_m / N_COILS - GAP) / 2 - i * PITCH) / r_m)) for i in range(N_T)]
    T_pitch = 1.5 * (R["k_w"] / max(kp[0], 1e-9)) * (N_COILS / 3) * 2 * 2 * B1 * I_coil_pk * L * r_m * sum(kp)
    return dict(Kt_textbook=T_text, Kt_with_turn_pitch=T_pitch, Kt_model=R["Kt"], discrepancy_textbook=T_text / R["Kt"] - 1, discrepancy_pitch=T_pitch / R["Kt"] - 1,
                k_w=mo.K_W, B1=B1, L_m=L, r_mean_m=r_m, pitch_factors=kp)


# =====================================================================================
# 7. Figures
# =====================================================================================
def fig_sweep(rows, chosen, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    pref_w = {}
    for coils in (24, 36, 48):
        ws = sorted(set((r["w_b"], r["ots_block"]) for r in rows if r["coils"] == coils), key=lambda t: (not t[1], -t[0]))
        pref_w[coils] = ws[0][0] if ws else None
    sub = [r for r in rows if r["ratio"] == 80 and r["r_in"] == chosen["r_in"] and r["leg_gap"] == chosen["leg_gap"] and r["w_b"] == pref_w[r["coils"]]]
    wlab = {c: f"{c} coils" + ("" if any(r["ots_block"] for r in sub if r["coils"] == c) else f" ({pref_w[c]:.1f} mm custom blocks)") for c in (24, 36, 48)}
    stacks = [s[0] for s in STACKS if any(r["stack"] == s[0] for r in sub)]
    x = np.arange(len(stacks))
    cols = {24: "#d98c3a", 36: "#0f9b8e", 48: "#6c5ce7"}
    ls = {6.0: ":", 8.0: "--", 10.0: "-"}
    ax = axes[0, 0]
    for coils in (24, 36, 48):
        for h_m in (6.0, 8.0, 10.0):
            ys = []
            for s in stacks:
                c = [r for r in sub if r["coils"] == coils and r["h_m"] == h_m and r["stack"] == s and r["swing_ok"]]
                ys.append(max((r["T_cont"] for r in c), default=np.nan))
            ax.plot(x, ys, ls[h_m], color=cols[coils], marker="o", ms=3, lw=1.2, label=f"{wlab[coils]}, {h_m:.0f} mm blocks")
    ys = [max((r["T_cont"] for r in sub if r["coils"] == chosen["coils"] and r["h_m"] == chosen["h_m"] and r["stack"] == s), default=np.nan) for s in stacks]
    ax.plot(x, ys, "-", color="#999", marker=".", ms=3, lw=1.0, label=f"{chosen['coils']} coils, {chosen['h_m']:.0f} mm, speed constraint dropped")
    ax.plot([stacks.index(chosen["stack"])], [chosen["T_cont"]], "*", ms=16, color="#b03a2e", zorder=5, label="chosen")
    ax.axhline(AB["ratings"]["1000"]["T_cont"], color="#555", lw=0.8, ls="-.", label="v2 12L 3 oz, Ø184, 8 mm (3.30)")
    ax.set_xticks(x); ax.set_xticklabels(stacks, rotation=35, fontsize=8)
    ax.set_ylabel("continuous torque at 1000 rpm (N·m), best turns / trace"); ax.grid(alpha=0.3)
    ax.set_title(f"Torque of every stack that reaches the femur swing speed at 48 V, Ø{BOARD_OD:.0f} board in Ø{OD_MAX:.0f}", fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    # turns at the chosen stack / coils / magnet
    ax = axes[0, 1]
    for coils in (24, 36, 48):
        c = sorted([r for r in sub if r["coils"] == coils and r["h_m"] == chosen["h_m"] and r["stack"] == chosen["stack"]], key=lambda r: r["turns"])
        if c:
            turns = sorted(set(r["turns"] for r in c))
            best = [max([r for r in c if r["turns"] == t], key=lambda r: r["T_cont"]) for t in turns]
            full = [r for r in c if r["trace_fill"] == 1.0]
            ax.plot(turns, [r["T_cont"] for r in best], "o-", color=cols[coils], ms=4, label=f"{coils} coils: T at 1000 rpm, best trace width")
            ax.plot([r["turns"] for r in full], [r["T_cont"] for r in full], "x--", color=cols[coils], ms=4, lw=0.7, label=f"{coils} coils: trace fills the leg")
            ax.plot(turns, [r["n_noload"] / 1000 for r in best], "s:", color=cols[coils], ms=3, label=f"{coils} coils: no-load krpm at 48 V")
    ax.axhline(chosen["n_swing"] / 1000, color="#27ae60", lw=1, ls="--", label=f"femur swing needs {chosen['n_swing']:.0f} rpm at 80:1")
    ax.set_xlabel("turns per layer"); ax.set_ylabel("N·m  /  krpm"); ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
    ax.set_title(f"Turns: torque is flat, speed is not ({chosen['stack']}, {chosen['h_m']:.0f} mm blocks)", fontsize=9)
    # cost and mass
    ax = axes[1, 0]
    for coils in (24, 36, 48):
        c = [max([r for r in sub if r["coils"] == coils and r["h_m"] == chosen["h_m"] and r["stack"] == s], key=lambda r: r["T_cont"], default=None) for s in stacks]
        ax.plot(x, [r["cost20"] if r else np.nan for r in c], "o-", color=cols[coils], ms=3, label=f"{coils} coils: $ at 20 (boards + magnets + cup)")
        ax.plot(x, [r["cost100"] if r else np.nan for r in c], "o--", color=cols[coils], ms=3, lw=0.8, label=f"{coils} coils: $ at 100")
    ax2 = ax.twinx()
    for coils in (24, 36, 48):
        c = [max([r for r in sub if r["coils"] == coils and r["h_m"] == chosen["h_m"] and r["stack"] == s], key=lambda r: r["T_cont"], default=None) for s in stacks]
        ax2.plot(x, [r["m_motor_kg"] if r else np.nan for r in c], "^:", color=cols[coils], ms=4, lw=0.8)
    ax2.set_ylabel("stator + rotor mass (kg, triangles)")
    ax.set_xticks(x); ax.set_xticklabels(stacks, rotation=35, fontsize=8); ax.set_ylabel("$ per unit"); ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
    ax.set_title(f"Cost and mass of the stator + rotor ({chosen['h_m']:.0f} mm blocks)", fontsize=9)
    # per dollar and per kg, all candidates
    ax = axes[1, 1]
    for coils in (24, 36, 48):
        c = [r for r in sub if r["coils"] == coils]
        import random
        c = random.Random(0).sample(c, len(c) // 3)                                  # a third of the candidates, fixed seed: the cloud reads the same, the PNG stays small
        ax.scatter([r["T_per_kg"] for r in c], [r["T_per_dollar20"] for r in c], s=[8 + 25 * (r["h_m"] - 5) for r in c], color=cols[coils], alpha=0.45, lw=0, label=f"{coils} coils (size: block thickness)")
    ax.plot([chosen["T_per_kg"]], [chosen["T_per_dollar20"]], "*", ms=16, color="#b03a2e", zorder=5, label="chosen")
    ax.set_xlabel("N·m per kg of stator + rotor"); ax.set_ylabel("N·m per $ at 20 units"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title("Every candidate at 80:1 (JLCPCB and reference stacks)", fontsize=9)
    fig.suptitle(f"Single-stator sweep in a Ø{OD_MAX:.0f} housing: {len(rows)} candidates, boards laid out by make_stator.py, rated as stator_asbuilt.py", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    shrink_png(path)


def shrink_png(path, colours=128):
    """The review page refuses images over ~400 kB; a palette PNG of a line chart is a third of the size and looks the same."""
    from PIL import Image
    im = Image.open(path).convert("RGB").quantize(colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    im.save(path, optimize=True)


def fig_envelope(G, R, chosen, cl80, cl100, path):
    Kt, R_ph = R["Kt"], R["R_ph_mohm"] * 1e-3
    n_nl = R["n_noload_rpm"]
    rpm = np.linspace(1, n_nl * 1.02, 300)
    P = G["pole_pairs"]
    r = np.linspace(G["r_in_mm"], G["r_out_mm"], 300) * 1e-3
    B = mo.halbach_B(r, P, chosen["h_m"] * 1e-3, R["g_mag_mm"] * 1e-3) * R["field_ratio"]
    TRACE, PITCH = G["trace_mm"] * 1e-3, (G["trace_mm"] + G["space_mm"]) * 1e-3
    layers, T_CU = chosen["layers"], chosen["oz"] * mo.OZ

    def t_cont(n):
        pe = float(np.trapezoid(mo.eddy_pv_strip(P * n / 60, B, TRACE) * (TRACE / PITCH) * layers * T_CU * 2 * math.pi * r, r))
        return Kt * math.sqrt(max(P_ALLOW - pe, 0) / (3 * R_ph)), pe

    tc = np.array([t_cont(n)[0] for n in rpm])
    tv = np.array([max(0.0, Kt * (V_PH / R_ph) * (1 - n / n_nl)) for n in rpm])
    tp = np.minimum(PEAK_MULT * tc, tv)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.fill_between(rpm, 0, np.minimum(tc, tv), color="#2980b9", alpha=0.25, label="continuous (50 W: copper + eddy at 120 °C)")
    ax.fill_between(rpm, np.minimum(tc, tv), tp, color="#c0392b", alpha=0.15, label="2 s peak (3× current)")
    ax.plot(rpm, tv, color="#c0392b", lw=1.2, label=f"48 V limit: no-load {n_nl:.0f} rpm")
    ax.plot(rpm, tc, color="#2980b9", lw=1.2)
    pts = []
    for ratio, cl, mk in ((80, cl80, "o"), (100, cl100, "s")):
        n_st = RPM_RATE[ratio]
        t_need = cl["need"]["knee"] / (ratio * ETA_CYC * ETA_CAP)
        n_sw = W_SWING_FK * ratio * 60 / (2 * math.pi)
        ax.plot([n_st], [t_need], mk, color="#b03a2e", ms=8, zorder=5)
        ax.annotate(f"femur/knee stance at {ratio}:1\nneeds {t_need:.2f} N·m at {n_st:.0f} rpm\n(robot {cl['m_robot_kg']:.0f} kg, margin {cl['margin']['knee']:.2f})", (n_st, t_need),
                    (200, 7.3) if ratio == 80 else (1500, 6.4), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#777", lw=0.5))
        ax.plot([n_sw], [0.12], mk, color="#27ae60", ms=8, zorder=5)
        ax.annotate(f"femur swing {W_SWING_FK:.1f} rad/s at {ratio}:1 = {n_sw:.0f} rpm\n({'reached' if n_sw <= n_nl else 'NOT reached: past no-load'})", (n_sw, 0.12), (n_sw - 80, 0.55 + (0.5 if ratio == 100 else 0)),
                    fontsize=7.5, ha="right", arrowprops=dict(arrowstyle="-", color="#777", lw=0.5))
    n_yaw = W_SWING_YAW * RATIO_YAW * 60 / (2 * math.pi)
    t_yaw = cl80["need"]["yaw"] / (RATIO_YAW * ETA_CYC)
    n_yaw_st = 3.06 * RATIO_YAW * 60 / (2 * math.pi)
    ax.plot([n_yaw_st], [t_yaw], "D", color="#8e44ad", ms=7, zorder=5)
    ax.annotate(f"yaw stance at 30:1: {t_yaw:.2f} N·m at {n_yaw_st:.0f} rpm (margin {cl80['margin']['yaw']:.2f})", (n_yaw_st, t_yaw), (200, 5.6), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#777", lw=0.5))
    ax.plot([n_yaw], [0.12], "D", color="#8e44ad", ms=7, zorder=5)
    ax.annotate(f"yaw swing 6.4 rad/s at 30:1 = {n_yaw:.0f} rpm", (n_yaw, 0.12), (n_yaw + 60, 0.35), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#777", lw=0.5))
    ax.set_xlabel("motor speed (rpm)"); ax.set_ylabel("motor torque (N·m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, n_nl * 1.02); ax.set_ylim(0, max(tp) * 1.1)
    ax.set_title(f"{chosen['stack']}, {chosen['coils']} coils / {2*chosen['pp']} poles, {chosen['turns']} turns, {chosen['h_m']:.0f} mm blocks: Kt {Kt:.3f} N·m/A, R {R_ph*1e3:.0f} mΩ, "
                 f"{chosen['T_cont']:.2f} N·m continuous at 1000 rpm", fontsize=9.5)
    fig.tight_layout(); fig.savefig(path, dpi=115); plt.close(fig)


def fig_section(chosen, U, G, path):
    """Dimensioned half-section of the single-stator unit stack, mirrored about the axis."""
    t_b, h_m, t_c = U["t_board_mm"], chosen["h_m"], U["t_carrier_mm"]
    clr = mo.AIR_CLEAR * 1e3
    z_top0 = Z_TOPCAR0_1S
    z = z_top0
    top_mag = (z - h_m, z); z -= h_m
    board = (z - clr - t_b, z - clr); z -= clr + t_b + clr
    bot_mag = (z - h_m, z); z -= h_m
    bot_car = (z - t_c, z)
    z_top1 = z_top0 + t_c
    z_cover0 = z_top1 + 0.5
    H = z_cover0 + 5.0
    r_od = OD_MAX / 2
    r_mag_in, r_mag_out = R_MAG_IN0 + RIM_SHIFT, R_MAG_OUT0 + RIM_SHIFT
    fig, ax = plt.subplots(figsize=(13, 4.6))

    def box(r0, r1, z0, z1, color, label=None, hatch=None, alpha=0.95):
        for s in (1, -1):
            ax.add_patch(Rectangle((min(s * r0, s * r1), z0), abs(r1 - r0), z1 - z0, facecolor=color, edgecolor="k", lw=0.4, hatch=hatch, alpha=alpha))
    AL_C, ST_C, MAG_C, BRD_C, BRG_C, RED_C = "#9aa5ad", "#d98c3a", "#c0392b", "#0f9b8e", "#e0e0e0", "#3a3a3a"
    box(40, r_od, 0, 6, AL_C)                                    # floor plate
    box(90.4, r_od, 6, board[0], AL_C)                           # wall tube
    box(90.4, r_od, board[1], z_cover0, AL_C)                    # upper ring (clamps the board rim)
    box(12, r_od, z_cover0, z_cover0 + 2, AL_C)                  # cover
    box(12, 15, z_cover0 + 2, H, AL_C)                           # bearing boss
    box(R_BORE, R_BOARD, *board, BRD_C)                          # stator board
    box(r_mag_in, r_mag_out, *top_mag, MAG_C); box(r_mag_in, r_mag_out, *bot_mag, MAG_C)
    box(R_HUB, R_CARRIER_OUT, z_top0, z_top1, ST_C)              # top carrier
    box(R_DRUM_OUT, R_CARRIER_OUT, *bot_car, ST_C)               # bottom carrier ring
    box(R_DRUM_IN, R_DRUM_OUT, bot_car[0] - 1.5, z_top0, ST_C)   # drum through the bore
    box(R_DRUM_IN, R_DRUM_OUT + 1.5, bot_car[0] - 1.5, bot_car[0], ST_C)
    box(25, 40, 0, 13, BRG_C)                                    # crossed roller
    box(41, 46, 15, 39.2, RED_C, hatch="////", alpha=0.7)        # pin cage
    box(13, 43, 17.2, 37.2, RED_C, hatch="..", alpha=0.5)        # two cycloid discs (schematic)
    box(0, 12.5, 4, H, "#777")                                    # shaft
    ax.axhline(0, color="#b03a2e", lw=0.6, ls="--")

    def dim_h(x0, x1, y, text, color="#b03a2e", dy=-3.2, fs=8):
        ax.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", color=color, lw=0.8))
        ax.text((x0 + x1) / 2, y + dy, text, ha="center", color=color, fontsize=fs)

    def dim_v(x, y0, y1, text, color="#1f4e79", dx=2.5, fs=7.5):
        ax.annotate("", (x, y0), (x, y1), arrowprops=dict(arrowstyle="<->", color=color, lw=0.8))
        ax.text(x + dx, (y0 + y1) / 2, text, va="center", color=color, fontsize=fs)
    dim_h(-r_od, r_od, -7, f"Ø{OD_MAX:.0f} over the housing wall (the limit)", fs=9)
    dim_h(-R_BOARD, R_BOARD, -12.5, f"Ø{BOARD_OD:.0f} board (Ø{G0['board_od_mm']:.0f} in v2)")
    dim_h(-R_BORE, R_BORE, -17.5, "Ø100 bore for the reducer")
    dim_h(-r_mag_out, -r_mag_in, H + 4, f"magnets r {r_mag_in:.1f}–{r_mag_out:.1f}", dy=1.2, fs=7)
    dim_h(G["r_in_mm"], G["r_out_mm"], H + 4, f"coils r {G['r_in_mm']:g}–{G['r_out_mm']:.1f}", dy=1.2, fs=7)
    x_d = r_od + 6
    dim_v(x_d, z_top0, z_top1, f"{t_c:.1f} mm top carrier + drum, 6061")
    dim_v(x_d, *top_mag, f"{h_m:.0f} mm  30×{chosen['w_b']:.0f}×{h_m:.0f} N48 block, {U['n_magnets']//2} per ring")
    dim_v(x_d, *board, f"{t_b:.2f} mm board: {chosen['stack']}, {chosen['coils']} coils, {chosen['turns']} turns / layer, {clr:.1f} mm clearance each side")
    dim_v(x_d, *bot_mag, f"{h_m:.0f} mm  lower Halbach ring")
    dim_v(x_d, *bot_car, f"{t_c:.1f} mm carrier ring ({U['m_carriers_g']:.0f} g of carriers hold {chosen['attraction_N']/1e3:.1f} kN of pull)")
    ax.text(x_d + 2.5, bot_car[0] - 3.8, f"motor band {U['h_band_mm']:.1f} mm, floor to cover", color="#555", fontsize=7.5)
    dim_v(-r_od - 6, 0, H, f"{H:.1f} mm\n(reducer-limited:\nthe band could be\n{z_top0 - Z_BAND_FLOOR + t_c:.0f} mm tall)", dx=-30)
    dim_v(-r_od - 6, 0, 13, "RB5013 crossed roller", dx=-34, color="#555")
    ax.annotate(f"wall: {WALL_R:.0f} mm of radius outside the board", (r_od - 1, 8), (r_od + 4, 2.5), fontsize=7, ha="left", color="#333", arrowprops=dict(arrowstyle="-", color="#777", lw=0.5))
    ax.text(0, 27, "20-lobe cycloid, two discs\n180° apart on HK2512", ha="center", va="center", fontsize=7.5, color="w")
    ax.set_aspect("equal"); ax.set_xlim(-r_od - 42, r_od + 100); ax.set_ylim(-20, H + 8)
    ax.set_xlabel("mm"); ax.set_ylabel("mm (mounting face at 0)"); ax.grid(alpha=0.2)
    ax.set_title(f"Single-stator femur/knee unit, Ø{OD_MAX:.0f}: {chosen['stack']} board between two Halbach rings — "
                 f"{U['m_unit_fk_kg']:.2f} kg unit, {chosen['T_cont']:.2f} N·m at 1000 rpm", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


# =====================================================================================
# 8. Main
# =====================================================================================
def main():
    CH = checks()
    for c in CH["field_vs_rotor_field"]:
        print(f"field check {c['case']}: B1 {c['B1_here']:.3f} vs {c['B1_rotor_field']:.3f} T, ratio {c['ratio_here']:.3f} vs {c['ratio_rotor_field']:.3f}, pull {c['pull_here_N']:.0f} vs {c['pull_rotor_field_N']:.0f} N")
    k = CH["kt_vs_asbuilt"]
    print(f"Kt check: {k['Kt_here']:.4f} vs asbuilt {k['Kt_asbuilt']:.4f}; R_ph {k['R_ph_here']:.1f} vs {k['R_ph_asbuilt']:.1f} mΩ; T1000 {k['T1000_here']:.3f} vs {k['T1000_asbuilt']:.3f}")
    rows = sweep()
    print(f"{len(rows)} candidates")
    json.dump(rows, open(os.path.join(SCRATCH, "rows.json"), "w"), default=float)      # every candidate, for queries; not committed
    # ---- selection ------------------------------------------------------------------
    # 1. JLCPCB-buildable, one board, 80:1, reaches the femur swing speed at 48 V: the most continuous torque at 1000 rpm
    ok = [r for r in rows if r["ratio"] == 80 and r["jlcpcb"] and r["boards"] == 1 and r["swing_ok"] and r["ots_block"]]
    if not ok:                                                                                                       # nothing reaches the swing speed: say so, take the fastest
        ok = [r for r in rows if r["ratio"] == 80 and r["jlcpcb"] and r["boards"] == 1 and r["ots_block"]]
        print("WARNING: no JLCPCB single board reaches the femur swing speed at 48 V; choosing on torque alone")
    chosen = max(ok, key=lambda r: r["T_cont"])
    best_any = max([r for r in rows if r["ratio"] == 80], key=lambda r: r["T_cont"])                               # torque with no constraint (reference)
    best_jlc = max([r for r in rows if r["ratio"] == 80 and r["jlcpcb"]], key=lambda r: r["T_cont"])              # incl. bonded pairs, ignoring speed
    best_per_dollar = max(ok, key=lambda r: r["T_per_dollar20"])
    best_per_kg = max(ok, key=lambda r: r["T_per_kg"])
    best_closure = max([r for r in rows if r["jlcpcb"] and r["boards"] == 1 and r["ots_block"]], key=lambda r: r["margin_knee"])   # either ratio, speed or not
    # ---- the chosen board in detail -------------------------------------------------------
    tr = None if chosen["trace_fill"] == 1.0 else chosen["trace_mm"]
    G = gen_board(chosen["coils"], chosen["pp"], chosen["turns"], chosen["r_in"], chosen["leg_gap"], trace=tr)
    R = rate(G, chosen["layers"], chosen["oz"], chosen["t_board_mm"], chosen["h_m"], chosen["w_b"], rpms=(1000.0, 1250.0, 1600.0, 2500.0))
    U = unit(G, R, chosen["layers"], chosen["oz"], chosen["boards"], chosen["h_m"], chosen["w_b"], 80)
    cl80 = closure(R["ratings"]["1000"]["T_cont"], R["ratings"]["1000"]["T_cont"], U["m_unit_fk_kg"], U["m_unit_yaw_kg"], 80)
    cl100 = closure(R["ratings"]["1250"]["T_cont"], R["ratings"]["1000"]["T_cont"], U["m_unit_fk_kg"], U["m_unit_yaw_kg"], 100)
    # which definition of continuous closes, at the fixed point + payload (closure.py's relief table)
    relief = []
    for case in CL["cases"]:
        c = case["c_per_kg"]
        m = cl80["m_robot_kg"] + PAYLOAD
        need = {d: max(c[d] * m, 1e-6) for d in c}
        relief.append(dict(label=case["label"], need=need, margin_fk=cl80["T_fk"] / max(need["femur"], need["knee"]), margin_yaw=cl80["T_yaw"] / need["yaw"],
                           closes_80=min(cl80["T_fk"] / need["femur"], cl80["T_fk"] / need["knee"], cl80["T_yaw"] / need["yaw"]) >= 1.0,
                           closes_100=min(cl100["T_fk"] / need["femur"], cl100["T_fk"] / need["knee"], cl100["T_yaw"] / need["yaw"]) >= 1.0))
    # sensitivity on the chosen board: clearance, r_in, leg gap
    sens = {}
    for clr in (0.35, 0.5, 0.7):
        Rs = rate(G, chosen["layers"], chosen["oz"], chosen["t_board_mm"], chosen["h_m"], chosen["w_b"], clr=clr)
        sens[f"clearance {clr} mm"] = dict(T_cont=Rs["ratings"]["1000"]["T_cont"], B=Rs["B_pk_mean"], attraction_N=Rs["attraction_N"])
    for r_in, gap in ((53.0, 0.6), (52.0, 0.6), (53.0, 0.4), (52.0, 0.4)):
        Gs = gen_board(chosen["coils"], chosen["pp"], chosen["turns"], r_in, gap, trace=tr)
        if Gs:
            Rs = rate(Gs, chosen["layers"], chosen["oz"], chosen["t_board_mm"], chosen["h_m"], chosen["w_b"])
            sens[f"r_in {r_in:g}, leg gap {gap:g}"] = dict(T_cont=Rs["ratings"]["1000"]["T_cont"], trace=Gs["trace_mm"], Kt=Rs["Kt"], R_ph_mohm=Rs["R_ph_mohm"])
    CF = closed_form(G, R, chosen["layers"], chosen["w_b"], chosen["h_m"])
    # ---- write the chosen board where the deliverables live ------------------------------------
    var_dir = os.path.join(ROOT, "hw", "stator", "variants", "1s-opt")
    Gv = gen_board(chosen["coils"], chosen["pp"], chosen["turns"], chosen["r_in"], chosen["leg_gap"], layers=chosen["layers"], oz=chosen["oz"], out_dir=var_dir,
                   t_board=chosen["t_board_mm"] if chosen["boards"] > 1 else None, trace=tr)
    mag_case = f"rect 30x5x{chosen['h_m']:.0f} N48"
    subprocess.run(["/opt/hw-py/bin/python", os.path.join(ROOT, "analysis", "stator_asbuilt.py"), os.path.join(var_dir, "geometry.json"), mag_case, f"{R['field_ratio']:.6f}"],
                   check=True, capture_output=True, text=True)
    ABv = json.load(open(os.path.join(var_dir, "asbuilt.json")))
    asbuilt_check = dict(Kt_sweep=R["Kt"], Kt_asbuilt=ABv["Kt"], R_sweep=R["R_ph_mohm"], R_asbuilt=ABv["R_ph_mohm"], T1000_sweep=R["ratings"]["1000"]["T_cont"], T1000_asbuilt=ABv["ratings"]["1000"]["T_cont"])
    # ---- figures ----------------------------------------------------------------------------------
    fig_sweep(rows, chosen, os.path.join(FIG, "1s-opt-sweep.png"))
    fig_envelope(G, R, chosen, cl80, cl100, os.path.join(FIG, "1s-opt-envelope.png"))
    fig_section(chosen, U, G, os.path.join(FIG, "1s-opt-section.png"))
    # ---- the record -------------------------------------------------------------------------------
    rows_sorted = sorted(rows, key=lambda r: -r["T_cont"])
    per_stack = []
    for name, layers, oz, boards, jlc in STACKS:
        c80 = [r for r in rows if r["ratio"] == 80 and r["stack"] == name and r["ots_block"]]
        if not c80:
            continue
        fast = [r for r in c80 if r["swing_ok"]]
        per_stack.append(dict(stack=name, jlcpcb=jlc, boards=boards, t_board_mm=c80[0]["t_board_mm"],
                              best_any=max(c80, key=lambda r: r["T_cont"]), best_swing_ok=max(fast, key=lambda r: r["T_cont"]) if fast else None))
    out = dict(limits=dict(od_max_mm=OD_MAX, wall_r_mm=WALL_R, board_od_mm=BOARD_OD, rim_shift_mm=RIM_SHIFT, r_out_mm=G["r_out_mm"]),
               conventions=dict(P_allow_W=P_ALLOW, R_th=R_TH, T_cu_max=T_CU_MAX, T_amb=T_AMB, V_ph=V_PH, rpm_rate=RPM_RATE, w_swing_femur=W_SWING_FK, w_swing_yaw=W_SWING_YAW,
                                swing_margin=SWING_MARGIN, payload=PAYLOAD, eta_cyc=ETA_CYC, eta_cap=ETA_CAP, c_per_kg=C_PER_KG, m_fixed=M_FIXED, pcb_quote=PCB_6L_2OZ, pcb_3oz=PCB_12L_3OZ, magnet_6mm=MAG_6MM, cup=ROTOR_CUP_1S,
                                m_fixed_fk_g=M_FIXED_FK, m_fixed_yaw_g=M_FIXED_YAW, h_wall_2s_mm=H_WALL_2S),
               checks=CH, n_candidates=len(rows), chosen=chosen, chosen_rating=R, chosen_unit=U, closure_80=cl80, closure_100=cl100, relief=relief, sensitivity=sens,
               closed_form=CF, asbuilt_check=asbuilt_check, best_any=best_any, best_jlcpcb_any_speed=best_jlc, best_per_dollar=best_per_dollar, best_per_kg=best_per_kg,
               best_closure=best_closure, per_stack=per_stack, top40=rows_sorted[:40], geometry_chosen=Gv)
    json.dump(out, open(os.path.join(ROOT, "hw", "stator", "single_stator_opt.json"), "w"), indent=1, default=float)
    # ---- report ------------------------------------------------------------------------------------
    print(f"\nchosen: {chosen['stack']}, {chosen['coils']} coils / {chosen['pp']} pp, {chosen['turns']} turns x {chosen['trace_mm']:.2f} mm, r {chosen['r_in']:g}-{chosen['r_out']:.1f}, "
          f"{chosen['h_m']:.0f} mm blocks ({'OTS' if chosen['ots_block'] else 'custom width'} {chosen['w_b']:.0f} mm), board {chosen['t_board_mm']:.2f} mm, gap {R['g_mag_mm']:.2f} mm")
    print(f"  Kt {R['Kt']:.3f}, R_ph {R['R_ph_mohm']:.1f} mΩ, B {R['B_pk_mean']:.2f} T, no-load {R['n_noload_rpm']:.0f} rpm; swing needs {chosen['n_swing']:.0f} rpm")
    for k2, v in R["ratings"].items():
        print(f"  {k2} rpm: eddy {v['P_eddy']:.1f} W, cu {v['P_cu']:.1f} W, I {v['I_cont']:.1f} A, T {v['T_cont']:.2f} / peak {v['T_peak']:.2f} N·m, n at I_cont {v['n_at_cont']:.0f} rpm")
    print(f"  stator+rotor {U['m_motor_g']:.0f} g (board {U['m_board_g']:.0f}, magnets {U['m_magnets_g']:.0f}, carriers {U['m_carriers_g']:.0f} at {U['t_carrier_mm']:.1f} mm), unit {U['m_unit_fk_kg']:.2f} kg, "
          f"cost ${U['cost'][20]['total']:.0f} / ${U['cost'][100]['total']:.0f}")
    print(f"  closure 80:1: robot {cl80['m_robot_kg']:.0f} kg, femur {cl80['T_fk']:.0f} / {cl80['need']['femur']:.0f}, knee margin {cl80['margin']['knee']:.2f}, yaw {cl80['margin']['yaw']:.2f} -> "
          f"{'closes' if cl80['closes'] else 'does not close'}; supports {cl80['m_robot_supported_kg']:.0f} kg; ratio to close {cl80['ratio_fk_to_close']:.0f}:1")
    print(f"  closure 100:1: robot {cl100['m_robot_kg']:.0f} kg, knee margin {cl100['margin']['knee']:.2f}, yaw {cl100['margin']['yaw']:.2f} -> {'closes' if cl100['closes'] else 'does not close'}")
    for rr in relief:
        print(f"    {rr['label']:55s} margin {rr['margin_fk']:.2f} / yaw {rr['margin_yaw']:.2f}: 80:1 {'OK' if rr['closes_80'] else '--'}, 100:1 {'OK' if rr['closes_100'] else '--'}")
    print(f"  closed form: textbook {CF['Kt_textbook']:.3f} ({CF['discrepancy_textbook']*100:+.0f} %), with turn pitch {CF['Kt_with_turn_pitch']:.3f} ({CF['discrepancy_pitch']*100:+.0f} %) vs model {CF['Kt_model']:.3f}")
    print(f"  asbuilt.py on the written board: Kt {asbuilt_check['Kt_asbuilt']:.4f} vs {asbuilt_check['Kt_sweep']:.4f}, T1000 {asbuilt_check['T1000_asbuilt']:.3f} vs {asbuilt_check['T1000_sweep']:.3f}")
    print(f"  best torque of all (ignoring buildability/speed): {best_any['stack']} {best_any['coils']}c {best_any['turns']}t {best_any['h_m']:.0f}mm: {best_any['T_cont']:.2f} N·m; "
          f"best JLCPCB any speed: {best_jlc['stack']} {best_jlc['coils']}c {best_jlc['turns']}t {best_jlc['h_m']:.0f}mm: {best_jlc['T_cont']:.2f}")
    print(f"  best per $: {best_per_dollar['stack']} {best_per_dollar['coils']}c {best_per_dollar['turns']}t {best_per_dollar['h_m']:.0f}mm {best_per_dollar['T_cont']:.2f} N·m ${best_per_dollar['cost20']:.0f}; "
          f"best per kg: {best_per_kg['stack']} {best_per_kg['coils']}c {best_per_kg['turns']}t {best_per_kg['h_m']:.0f}mm {best_per_kg['T_cont']:.2f} N·m {best_per_kg['m_motor_kg']:.2f} kg")
    print(f"  best knee margin (any ratio/speed): {best_closure['stack']} {best_closure['coils']}c {best_closure['turns']}t {best_closure['h_m']:.0f}mm at {best_closure['ratio']}:1: margin {best_closure['margin_knee']:.2f}, swing {'ok' if best_closure['swing_ok'] else 'NOT reached'}")
    for name, v in sens.items():
        print(f"  sensitivity {name}: {v}")


if __name__ == "__main__":
    main()

#!/opt/hw-py/bin/python
"""The two-machine system: a tracked carrier that carries a small legged scout.

    /opt/hw-py/bin/python analysis/marsupial.py

Round 16 found that the two requirements the brief puts on one machine --
cover distance efficiently, and touch sensitive ground lightly -- are in
conflict only because we assumed one machine has to do both.  This sizes the
pair that splits them, and it changes the project more than any round so far.

The consequence worth stating first: **the legged machine stops being a 147 kg
hexapod and becomes a ~22 kg quadruped scout.**  Joint torque scales with mass
times leg length, so a scout at a fifth the mass and half the leg length needs
roughly a tenth the joint torque.  Fourteen rounds of actuator design were
aimed at a machine this architecture does not contain.  This script computes
what the scout's actuator actually has to be, so the review can see whether any
of that work survives.

It also asks the question that decides whether the split is real: **what can a
22 kg scout actually do?**  It cannot cut and carry vegetation -- that is force
and payload the carrier has.  So the tasks divide, and the division is the
design.

Writes hw/marsupial.json and docs/design/platform/marsupial-*.png.
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
import hexapod_model as hm                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "design", "platform")
os.makedirs(FIG, exist_ok=True)
G = 9.81
PT = json.load(open(os.path.join(ROOT, "hw", "platform_topology.json")))
HM = json.load(open(os.path.join(ROOT, "hw", "hybrid_mobility.json")))
FM = json.load(open(os.path.join(ROOT, "hw", "stator", "frameless_motor.json")))
A = PT["assumptions"]
C = FM["requirement"]["c_per_kg"]                 # N.m of joint torque per kg of robot, at the hexapod's leg length
C_PK = FM["requirement"]["c_peak_per_kg"]
HEX_LEG_M = (hm.LEG.femur + hm.LEG.tibia) / 1e3   # 0.75 m: the length the c_per_kg above is defined at
HEX_COT = PT["hexapod"]["cot"]
PACK = A["pack_wh_per_kg"]

# ------------------------------------------------------------------ the scout
# A quadruped sized to work sensitive ground, not to carry a day's battery.
SCOUT = dict(mass_kg=22.0, legs=4, femur_mm=170.0, tibia_mm=260.0, coxa_mm=90.0,
             foot_dia_mm=110.0, stride_m=0.40, speed_ms=0.6,
             tool_kg=3.0, tool_W=40.0, compute_W=25.0, pack_kg=1.6)
SCOUT["leg_m"] = (SCOUT["femur_mm"] + SCOUT["tibia_mm"]) / 1e3


def scout_joint_torque():
    """Joint torque scales with mass x leg length: the c_per_kg from the hexapod
    study is defined at its own 0.75 m leg, so scale by the scout's."""
    k = SCOUT["leg_m"] / HEX_LEG_M
    m = SCOUT["mass_kg"] + SCOUT["tool_kg"] + SCOUT["pack_kg"]
    return {d: dict(cont=C[d] * m * k, peak=C_PK[d] * m * k) for d in ("yaw", "femur", "knee")}


SCOUT_T = scout_joint_torque()
HEX_T = {d: dict(cont=C[d] * 147.0, peak=C_PK[d] * 147.0) for d in ("yaw", "femur", "knee")}
TORQUE_RATIO = {d: HEX_T[d]["cont"] / SCOUT_T[d]["cont"] for d in SCOUT_T}

# What reduction the scout needs, given the motors the round-11 market search already priced.
OTS = json.load(open(os.path.join(ROOT, "hw", "stator", "motor_market.json")))
BEST_OTS = sorted([r for r in OTS["rows"] if r["motors_per_unit"]], key=lambda r: r["price20"])[0]
ETA_RED = 0.90
scout_ratio = {d: SCOUT_T[d]["cont"] / (BEST_OTS["T_cont"] * ETA_RED) for d in SCOUT_T}

# ---------------------------------------------------------------- the carrier
CARRIER = dict(base_kg=55.0, track_w_mm=150.0, track_l_mm=600.0, drive_act=3,
               cot=(A["f_roll"]["grass/pasture"] + A["track_internal"]) / A["eta_drive"] * 1.05,
               steering="articulated centre pivot: no skid shear", speed_ms=1.2,
               tool_W=900.0, compute_W=60.0)

# ------------------------------------------------- what each machine can do
# The division of labour is the design. Force and payload live on the carrier;
# access and gentleness live on the scout.
TASKS = [
    dict(task="Transit between work sites", carrier=4, scout=1,
         why="the carrier is 5x more efficient per km and carries everything"),
    dict(task="Broad survey along a route", carrier=4, scout=2,
         why="sensor mast on the carrier covers ground faster"),
    dict(task="Cutting woody vegetation", carrier=4, scout=0,
         why=f"needs ~{CARRIER['tool_W']:.0f} W at the tool and a base to react against; a {SCOUT['mass_kg']:.0f} kg machine cannot"),
    dict(task="Moving cut material", carrier=4, scout=0,
         why="payload. The carrier takes ~50 % of its mass; the scout takes 3 kg"),
    dict(task="Close inspection in tight or broken ground", carrier=1, scout=4,
         why="the scout goes where a 1.1 m tracked machine cannot"),
    dict(task="Spot-treating invasive plants on sensitive ground", carrier=0, scout=4,
         why="this is the whole reason the scout exists"),
    dict(task="Sampling and placing sensors", carrier=1, scout=4,
         why="precision placement, light touch"),
    dict(task="Working wet or fragile ground", carrier=0, scout=4,
         why="the carrier stays on the firm route"),
    dict(task="Power, comms relay and recharge", carrier=4, scout=0,
         why="the carrier is the mothership: battery, uplink, shelter"),
]

# --------------------------------------------- the scout's operating envelope
# Standing power is what decided every earlier round, so it must be computed
# from a real actuator, not scaled from the hexapod's. A scout that stands on
# a small motor through a modest reduction burns a LOT: copper loss goes as
# the square of motor torque, and motor torque is joint torque over the ratio.
# Two levers: more reduction, or a brake. The brake is decisive.
MOTOR = dict(name=BEST_OTS["name"], T_cont=BEST_OTS["T_cont"], Kt=BEST_OTS["Kt"], R=BEST_OTS["R"],
             I_cont=BEST_OTS["I_cont"], mass=BEST_OTS["mass"], price=BEST_OTS["price20"])
MOTOR["P_cu_cont"] = 3 * MOTOR["I_cont"] ** 2 * MOTOR["R"]
# Standing a quadruped on four legs, no dynamic factor: each leg carries m/4,
# where the c_per_kg load case carries m/3 x 1.5 = 0.5 m. So half the torque.
STAND_FRACTION = 0.25 / 0.50


def scout_standing_W(ratio, braked=False):
    """Copper loss holding a static stance, per the whole machine."""
    if braked:
        return 0.0
    tot = 0.0
    for d in ("femur", "knee"):
        t_joint = SCOUT_T[d]["cont"] * STAND_FRACTION
        t_motor = t_joint / (ratio * ETA_RED)
        tot += SCOUT["legs"] * MOTOR["P_cu_cont"] * (t_motor / MOTOR["T_cont"]) ** 2
    return tot


RATIOS = np.arange(8, 81, 2.0)
STAND_VS_RATIO = [dict(ratio=float(r), W=scout_standing_W(r)) for r in RATIOS]
RATIO_PICK = 30.0          # what a compact single-stage cycloid gives, and 4x less standing loss than 16:1
BRAKE = dict(kind="a small holding brake or a non-backdrivable stage on each pitch joint",
             mass_kg_per_joint=0.10, standing_W=0.0,
             why="the scout is stopped for 70 % of its mission; this is the same reason a tracked "
                 "platform beat the hexapod in round 15")


def scout_envelope(pack_kg=SCOUT["pack_kg"], dwell_frac=0.70, ratio=RATIO_PICK, braked=True):
    wh = pack_kg * PACK
    m = SCOUT["mass_kg"] + SCOUT["tool_kg"] + pack_kg
    p_walk = HEX_COT * m * G * SCOUT["speed_ms"]
    p_stand = scout_standing_W(ratio, braked)
    p_avg = ((1 - dwell_frac) * (p_walk + SCOUT["compute_W"])
             + dwell_frac * (p_stand + SCOUT["compute_W"] + SCOUT["tool_W"]))
    hours = wh / p_avg
    moving_h = hours * (1 - dwell_frac)
    radius_m = moving_h * 3600 * SCOUT["speed_ms"] / 2
    return dict(pack_kg=pack_kg, wh=wh, mass_kg=m, p_walk=p_walk, p_stand=p_stand, p_avg=p_avg,
                hours=hours, radius_m=radius_m, dwell_frac=dwell_frac, ratio=ratio, braked=braked)


ENV = scout_envelope()
ENV_UNBRAKED = scout_envelope(braked=False)
PACKS = np.linspace(0.6, 6.0, 40)
ENV_SWEEP = [scout_envelope(p) for p in PACKS]
ENV_SWEEP_UNBRAKED = [scout_envelope(p, braked=False) for p in PACKS]

# --------------------------------------------------- the pair over a day
# The carrier repositions; the scout works a disc around each stop.
def day(n_sorties=6, dwell_frac=0.70):
    """The scout's radius is a LEASH, not a work rate: it is how far it could walk
    out and back before the pack runs down. What it can actually treat in that
    time is set by how long each plant takes, which nobody has measured, so the
    area below is the ground it can REACH, not the ground it can work."""
    e = ENV
    reach_m2 = math.pi * (e["radius_m"] ** 2)
    return dict(sorties=n_sorties, hours_per_sortie=e["hours"], leash_m=e["radius_m"],
                reachable_m2_per_sortie=reach_m2, reachable_ha_per_day=reach_m2 * n_sorties / 1e4,
                scout_energy_day_wh=n_sorties * e["wh"], recharges=n_sorties - 1,
                caveat="reachable, not worked: the treatment rate per plant is unmeasured")


DAY = day()
CARRIER_DAY = [r for r in HM["rows"] if r["key"] == "marsupial"][0]

out = dict(scout=SCOUT, scout_torque=SCOUT_T, hexapod_torque=HEX_T, torque_ratio=TORQUE_RATIO,
           scout_ratio=scout_ratio, best_ots_motor=BEST_OTS, carrier=CARRIER, tasks=TASKS,
           motor=MOTOR, standing_vs_ratio=STAND_VS_RATIO, ratio_pick=RATIO_PICK, brake=BRAKE,
           envelope=ENV, envelope_unbraked=ENV_UNBRAKED, envelope_sweep=[dict(pack_kg=float(p), **{k: v for k, v in e.items() if k != "pack_kg"})
                                         for p, e in zip(PACKS, ENV_SWEEP)],
           day=DAY, pair_from_hybrid_study=CARRIER_DAY, hex_leg_m=HEX_LEG_M)
json.dump(out, open(os.path.join(ROOT, "hw", "marsupial.json"), "w"), indent=1, default=float)

# ============================================================== figure 1
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), gridspec_kw=dict(width_ratios=(0.85, 1.0, 1.35)))
ax = axes[0]
d = ["yaw", "femur", "knee"]
x = np.arange(3)
ax.bar(x - 0.2, [HEX_T[k]["cont"] for k in d], 0.38, color="#b03a2e", label="the 147 kg hexapod")
ax.bar(x + 0.2, [SCOUT_T[k]["cont"] for k in d], 0.38, color="#0f9b8e", label=f"the {SCOUT['mass_kg']:.0f} kg scout")
for i, k in enumerate(d):
    ax.text(i - 0.2, HEX_T[k]["cont"] + 6, f"{HEX_T[k]['cont']:.0f}", ha="center", fontsize=8.5)
    ax.text(i + 0.2, SCOUT_T[k]["cont"] + 6, f"{SCOUT_T[k]['cont']:.0f}", ha="center", fontsize=8.5)
    ax.text(i, HEX_T[k]["cont"] * 0.5, f"÷{TORQUE_RATIO[k]:.0f}", ha="center", fontsize=12, fontweight="bold", color="#fff")
ax.set_xticks(x); ax.set_xticklabels(["yaw", "femur", "knee"])
ax.set_ylabel("continuous joint torque (N·m)")
ax.set_title("1. What splitting the machine does\nto the actuator", fontsize=9.5)
ax.set_ylim(0, 460)
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.plot([s_["ratio"] for s_ in STAND_VS_RATIO], [s_["W"] for s_ in STAND_VS_RATIO], color="#b03a2e", lw=2.2,
        label="unbraked: copper loss holding the stance")
ax.axhline(0, color="#0f9b8e", lw=2.6, label="with a holding brake on each pitch joint")
ax.axvline(RATIO_PICK, color="#222", ls="--", lw=1.2)
w_at = scout_standing_W(RATIO_PICK)
ax.annotate(f"at {RATIO_PICK:.0f}:1 the scout burns {w_at:.0f} W\nstanding still — more than it uses walking.\n"
            f"A brake makes it zero.", (RATIO_PICK, w_at), (14, 40), textcoords="offset points", fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#444"))
ax.set_xlabel("reduction ratio at the scout's pitch joints")
ax.set_ylabel("power to stand still (W)")
ax.set_ylim(0, min(600, max(s_["W"] for s_ in STAND_VS_RATIO) * 1.05))
ax.set_title("2. The scout is stopped 70 % of the time.\nA brake on each joint is the single biggest decision on this page.", fontsize=9.5)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

ax = axes[2]
y = np.arange(len(TASKS))
ax.barh(y - 0.2, [t["carrier"] for t in TASKS], 0.38, color="#2a78d6", label="carrier")
ax.barh(y + 0.2, [t["scout"] for t in TASKS], 0.38, color="#0f9b8e", label="scout")
ax.set_yticks(y); ax.set_yticklabels([t["task"] for t in TASKS], fontsize=8); ax.invert_yaxis()
ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xticklabels(["cannot", "poor", "ok", "good", "best"], fontsize=8)
ax.set_title("3. The division of labour IS the design.\nForce and payload on the carrier; access and gentleness on the scout.", fontsize=9.5)
ax.set_xlim(0, 4.9)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.97); ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "marsupial-system.png"), dpi=110)

# ============================================================== figure 2: to scale
# Everything in mm at one scale, including the person. The earlier draft drew the
# figure at 0.42x and the legs at arbitrary factors, which defeated the point.
fig, ax = plt.subplots(figsize=(14, 4.6))


def box(ax, x0, y0, w, h, fc, label=None, fs=8):
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fc=fc, ec="#333", lw=1.2, alpha=0.92))
    if label:
        ax.text(x0 + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=fs)


# --- carrier, side view
CW, TRACK_H, BODY_H = 1100.0, 190.0, 210.0
box(ax, 0, 0, CW, TRACK_H, "#5b6770", "tracks, 150 mm wide")
box(ax, 55, TRACK_H, CW - 110, BODY_H, "#2a78d6", "carrier: battery, compute, arm, tools")
# the scout sits in a cradle on the deck
SB_L, SB_H = SCOUT["coxa_mm"] * 2 + 240, 150.0
box(ax, CW - SB_L - 90, TRACK_H + BODY_H, SB_L, SB_H, "#0f9b8e", "scout, docked", fs=7.5)
ax.annotate("", (0, -70), (CW, -70), arrowprops=dict(arrowstyle="<->", color="#444"))
ax.text(CW / 2, -105, f"{CW:.0f} mm", ha="center", fontsize=8)
ax.text(CW / 2, -215, f"carrier {[r for r in HM['rows'] if r['key']=='marsupial'][0]['m_total'] - 26:.0f} kg\n{CARRIER['steering']}",
        ha="center", va="top", fontsize=8.5)

# --- scout, deployed, standing, true limb lengths
sx = CW + 760
FEM, TIB = SCOUT["femur_mm"], SCOUT["tibia_mm"]
# a standing pose: femur 45 deg from horizontal, tibia vertical
fem_dx, fem_dy = FEM * math.cos(math.radians(50)), FEM * math.sin(math.radians(50))
hip_y = TIB + fem_dy
box(ax, sx, hip_y, SB_L, SB_H, "#0f9b8e", "scout body", fs=7.5)
for dx in (35, SB_L - 35):
    hx = sx + dx
    kx, ky = hx + (fem_dx if dx > SB_L / 2 else -fem_dx), hip_y - fem_dy
    ax.plot([hx, kx], [hip_y, ky], color="#222", lw=3.2)
    ax.plot([kx, kx], [ky, 0], color="#222", lw=3.2)
    ax.add_patch(plt.Circle((kx, 0), SCOUT["foot_dia_mm"] / 2, fc="#b03a2e", ec="#333", lw=0.8))
ax.annotate("", (sx - 30, -70), (sx + SB_L + 30, -70), arrowprops=dict(arrowstyle="<->", color="#444"))
ax.text(sx + SB_L / 2, -105, f"{SB_L + 60:.0f} mm", ha="center", fontsize=8)
ax.text(sx + SB_L / 2, -215, f"scout {SCOUT['mass_kg']:.0f} kg\nfemur {FEM:.0f} + tibia {TIB:.0f} mm, Ø{SCOUT['foot_dia_mm']:.0f} feet",
        ha="center", va="top", fontsize=8.5)

# --- a 1.8 m person, at the same scale
px = sx + SB_L + 780
PH = 1800.0
ax.plot([px, px], [0, PH * 0.82], color="#555", lw=4)
ax.add_patch(plt.Circle((px, PH * 0.9), PH * 0.055, fc="#555", ec="none"))
ax.plot([px - 170, px + 170], [PH * 0.72, PH * 0.72], color="#555", lw=3)
ax.plot([px, px - 110], [PH * 0.42, 0], color="#555", lw=3.5)
ax.plot([px, px + 110], [PH * 0.42, 0], color="#555", lw=3.5)
ax.text(px, -105, "1.8 m", ha="center", fontsize=8)
ax.text(px, -215, "for scale", ha="center", fontsize=8.5)

ax.plot([-80, px + 420], [0, 0], color="#8a7a5a", lw=2)
ax.set_xlim(-140, px + 480); ax.set_ylim(-290, 1980); ax.set_aspect("equal"); ax.axis("off")
hexr = [r for r in HM["rows"] if r["key"] == "hex"][0]
mar = [r for r in HM["rows"] if r["key"] == "marsupial"][0]
ax.set_title("4. The pair, to scale. The carrier does the distance, the force and the payload; the scout does the sensitive ground.\n"
             f"Together {mar['m_total']:.0f} kg and {mar['wh']:.0f} Wh a day, against the hexapod's {hexr['m_total']:.0f} kg and {hexr['wh']:.0f} Wh — "
             f"and {mar['pressure_kPa']:.0f} kPa on the sensitive ground instead of {hexr['pressure_kPa']:.0f}.", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "marsupial-scale.png"), dpi=110)

if __name__ == "__main__":
    print(f"SCOUT {SCOUT['mass_kg']:.0f} kg, legs {SCOUT['femur_mm']:.0f}+{SCOUT['tibia_mm']:.0f} mm "
          f"({SCOUT['leg_m']:.2f} m vs the hexapod's {HEX_LEG_M:.2f} m)")
    print(f"{'joint':8s}{'hexapod cont':>14s}{'scout cont':>12s}{'ratio':>8s}{'scout peak':>12s}{'ratio to a ' + BEST_OTS['name'].split(' (')[0]:>26s}")
    for d_ in ("yaw", "femur", "knee"):
        print(f"{d_:8s}{HEX_T[d_]['cont']:14.0f}{SCOUT_T[d_]['cont']:12.1f}{TORQUE_RATIO[d_]:8.0f}{SCOUT_T[d_]['peak']:12.1f}{scout_ratio[d_]:26.1f}")
    print(f"\nthe cheapest OTS motor that already closes: {BEST_OTS['name']} at ${BEST_OTS['price20']}, {BEST_OTS['T_cont']:.2f} N·m")
    print(f"so the scout's joints need {min(scout_ratio.values()):.0f}:1 to {max(scout_ratio.values()):.0f}:1, against the hexapod's 25:1 to 100:1\n")
    print(f"standing power at {RATIO_PICK:.0f}:1 unbraked: {scout_standing_W(RATIO_PICK):.0f} W; braked: 0 W")
    print(f"  braked  -> {ENV['hours']:.1f} h, {ENV['radius_m']:.0f} m radius")
    print(f"  unbraked-> {ENV_UNBRAKED['hours']:.1f} h, {ENV_UNBRAKED['radius_m']:.0f} m radius")
    e = ENV
    print(f"SCOUT ENVELOPE on a {e['pack_kg']:.1f} kg pack ({e['wh']:.0f} Wh): walk {e['p_walk']:.0f} W, stand {e['p_stand']:.0f} W, "
          f"average {e['p_avg']:.0f} W -> {e['hours']:.1f} h, {e['radius_m']:.0f} m working radius")
    print(f"PAIR over a day: {DAY['sorties']} sorties of {DAY['hours_per_sortie']:.1f} h, leash {DAY['leash_m']:.0f} m, "
          f"{DAY['recharges']} recharges from the carrier")
    print(f"     (the leash is how far it can walk out and back, NOT a work rate)")
    print(f"     carrier + scout: {CARRIER_DAY['m_total']:.0f} kg, {CARRIER_DAY['wh']:.0f} Wh/day, "
          f"{CARRIER_DAY['pressure_kPa']:.0f} kPa and {CARRIER_DAY['disturbed_m2_per_m']:.3f} m2/m on the sensitive ground")

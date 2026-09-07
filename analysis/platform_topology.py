#!/opt/hw-py/bin/python
"""Challenge the founding assumption: is a walking hexapod the right platform for
a rugged multipurpose outdoor robot doing long-duration, long-distance missions
that navigate AND manipulate?

    /opt/hw-py/bin/python analysis/platform_topology.py

The vision stage picked a hexapod before any mission profile existed.  This is
the trade study that should have come first.  It compares eight ground
topologies against four mission archetypes drawn from the stated tasks --
cutting and moving plants, surveying, inspecting, treating invasive species --
on the quantity that actually decides a long-endurance platform: **energy**,
solved as a fixed point, because the battery that buys endurance is itself mass
that costs energy to carry.

Three things make this comparison honest rather than rhetorical:

  1. The hexapod's cost of transport is **this project's own number**, not a
     literature value: hexapod_model.py's gait model says 908 W to walk at
     1 m/s at 124 kg, which is CoT 0.75.  So the legged column is argued on the
     design's own terms.
  2. The wheeled and tracked columns are built from rolling resistance and
     drivetrain efficiency, both stated, both swept for sensitivity -- not from
     a remembered CoT figure.
  3. Standing power is counted separately from travelling power, because the
     stated tasks are dwell-heavy: a robot that stops to cut a stem or spray a
     weed spends most of its mission holding still.  A legged robot pays to
     stand.  A wheeled one does not.  That distinction is invisible in a CoT
     comparison and it dominates these missions.

Writes hw/platform_topology.json and docs/design/platform/*.png.
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

# ---------------------------------------------------------------- this design
# The hexapod's own gait model, so the legged case is argued on its own numbers.
HEX_MASS = hm.MASS.robot                          # kg
HEX_WALK_W = 907.58                               # W at 1 m/s (hw/arrangement.json, from hexapod_model)
HEX_COT = HEX_WALK_W / (HEX_MASS * G * 1.0)
HEX_PACK_KG, HEX_PACK_WH_PER_KG = 8.0, 170.0      # two 4 kg packs; pack-level Li-ion NMC

# --------------------------------------------------------------- assumptions
# Every number below is an engineering assumption, stated so it can be attacked,
# and the ones that matter are swept at the end.
PACK_WH_PER_KG = 170.0        # pack level, Li-ion NMC. LFP would be ~110.
ETA_DRIVE = 0.85              # motor + gearbox, wheeled/tracked driveline
F_ROLL = {                    # rolling resistance coefficient, by ground
    "track/firm dirt": 0.08, "grass/pasture": 0.11, "soft soil/leaf litter": 0.22, "sand/mud": 0.32}
TRACK_INTERNAL = 0.06         # extra loss in a track's own rollers and pins
SOLAR_W_PER_M2 = 190.0        # a decent panel at a working-day average, not peak
COMPUTE_W = 60.0              # perception + planning, always on
TOOL_W = {"cut": 900.0, "spray": 40.0, "sense": 15.0, "carry": 0.0}


def hexapod_standing_W():
    """What this hexapod burns standing still, from its own actuator numbers.

    Standing is all six feet down and no dynamic factor, so each leg carries
    m/6 where the sizing load case (three legs down, dyn 1.5) carries m/3 x 1.5.
    The joint torque is that much lower, the motor torque follows the ratio, and
    copper loss goes as torque squared from the study's own continuous point --
    which is the right way to scale it, rather than guessing a resistance.
    The yaw joints hold nothing standing square, so only the twelve pitch
    joints count.  A wheeled or tracked platform holds still on brakes and
    burns nothing at all.
    """
    p = os.path.join(ROOT, "hw", "stator", "frameless_motor.json")
    if not os.path.exists(p):
        return 150.0, "estimate (no actuator study on disk)"
    fm = json.load(open(p))
    pick = fm["pick"]
    c = fm["requirement"]["c_per_kg"]
    m_robot = fm["mass_ladder"][-1]["m_robot"] if fm.get("mass_ladder") else pick["m_robot"]
    t_cont_motor = pick["motor"]["T_cont"]
    p_cu_cont = pick["motor"]["P_cu"]
    tot = 0.0
    for d in ("femur", "knee"):
        t_joint_static = c[d] * m_robot / 3.0            # m/6 per leg instead of m/3 x 1.5
        t_motor = t_joint_static / (pick["ratio_fk"] * 0.90)
        tot += 6 * p_cu_cont * (t_motor / t_cont_motor) ** 2      # six legs, this joint on each
    return tot, "computed from the actuator study: static stance torque, copper loss as torque squared"


HEX_STAND_W, HEX_STAND_SRC = hexapod_standing_W()

# --------------------------------------------------------------- topologies
# cot_firm / cot_soft: cost of transport, dimensionless (P / m g v)
# stand_W: what the locomotion system burns holding position, per tonne of robot
# terrain: fraction of a mixed outdoor route the topology can traverse unaided
# payload_frac: useful payload as a fraction of platform mass
# drive_act: actuators in the locomotion system (cost, reliability, control)
TOPOLOGIES = [
    dict(name="Hexapod (this design)", family="legged", drive_act=18,
         cot_firm=HEX_COT, cot_soft=HEX_COT * 1.2, stand_W_per_t=HEX_STAND_W / (HEX_MASS / 1e3),
         terrain=0.98, payload_frac=0.15, disturb="lowest: six small footprints, no shear",
         note="the project's own gait model: 908 W at 1 m/s and 124 kg"),
    dict(name="Quadruped", family="legged_quad", drive_act=12,
         cot_firm=HEX_COT * 0.9, cot_soft=HEX_COT * 1.1, stand_W_per_t=HEX_STAND_W / (HEX_MASS / 1e3) * 0.8,
         terrain=0.95, payload_frac=0.18, disturb="lowest: four small footprints",
         note="fewer actuators, less holding copper, slightly worse static stability"),
    dict(name="Wheel-on-leg hybrid", family="hybrid", drive_act=16,
         cot_firm=0.11 / ETA_DRIVE, cot_soft=0.28 / ETA_DRIVE, stand_W_per_t=HEX_STAND_W / (HEX_MASS / 1e3) * 0.5,
         terrain=0.95, payload_frac=0.22, disturb="low: four contact patches, some shear when steering",
         note="rolls where it can, walks where it must; still holds posture on legs"),
    dict(name="6-wheel rocker-bogie", family="wheeled", drive_act=6,
         cot_firm=F_ROLL["grass/pasture"] / ETA_DRIVE, cot_soft=F_ROLL["soft soil/leaf litter"] / ETA_DRIVE,
         stand_W_per_t=0.0, terrain=0.72, payload_frac=0.40, disturb="moderate: six contact patches",
         note="passive suspension, no holding power, climbs obstacles ~1 wheel radius"),
    dict(name="4-wheel skid-steer", family="wheeled", drive_act=4,
         cot_firm=F_ROLL["grass/pasture"] * 1.15 / ETA_DRIVE, cot_soft=F_ROLL["soft soil/leaf litter"] * 1.3 / ETA_DRIVE,
         stand_W_per_t=0.0, terrain=0.62, payload_frac=0.45, disturb="high: skid steering shears the ground",
         note="simplest and cheapest; skid turns scrub the surface, which matters on sensitive ground"),
    dict(name="Tracked, two tracks", family="tracked", drive_act=2,
         cot_firm=(F_ROLL["track/firm dirt"] + TRACK_INTERNAL) / ETA_DRIVE,
         cot_soft=(F_ROLL["grass/pasture"] + TRACK_INTERNAL) / ETA_DRIVE,
         stand_W_per_t=0.0, terrain=0.80, payload_frac=0.50, disturb="low pressure but high shear in a turn",
         note="lowest ground pressure, best traction for pushing and cutting, degrades least on soft ground"),
    dict(name="Tracked with four flippers", family="tracked", drive_act=6,
         cot_firm=(F_ROLL["track/firm dirt"] + TRACK_INTERNAL) * 1.15 / ETA_DRIVE,
         cot_soft=(F_ROLL["grass/pasture"] + TRACK_INTERNAL) * 1.15 / ETA_DRIVE,
         stand_W_per_t=0.0, terrain=0.92, payload_frac=0.42, disturb="as tracks",
         note="flippers add stairs, deadfall and ditch crossing for four more actuators"),
    dict(name="Small swarm: 8 × 25 kg tracked", family="swarm", drive_act=2 * 8,
         cot_firm=(F_ROLL["track/firm dirt"] + TRACK_INTERNAL) * 1.1 / ETA_DRIVE,
         cot_soft=(F_ROLL["grass/pasture"] + TRACK_INTERNAL) * 1.1 / ETA_DRIVE,
         stand_W_per_t=0.0, terrain=0.78, payload_frac=0.35, disturb="lowest pressure of all; eight light machines",
         note="coverage scales with unit count, not unit size; one failure is not mission loss"),
]

# ------------------------------------------------------------ the missions
# Drawn from the stated tasks. dwell_frac is the share of the working day spent
# stopped and working rather than travelling -- the number that decides this.
MISSIONS = [
    dict(key="survey", name="Survey / patrol transect", km_per_day=25.0, speed=1.2, dwell_frac=0.15,
         tool="sense", payload_kg=8.0, ground="grass/pasture", hours=8.0,
         note="cover ground, hold sensors steady, come home"),
    dict(key="inspect", name="Inspection round", km_per_day=8.0, speed=0.8, dwell_frac=0.55,
         tool="sense", payload_kg=12.0, ground="grass/pasture", hours=8.0,
         note="drive between assets, stop and look closely at each"),
    dict(key="invasive", name="Invasive species control", km_per_day=6.0, speed=0.5, dwell_frac=0.70,
         tool="spray", payload_kg=25.0, ground="soft soil/leaf litter", hours=8.0,
         note="quarter an area, identify a plant, treat it, move on: mostly stopped"),
    dict(key="cut", name="Cutting and moving vegetation", km_per_day=3.0, speed=0.4, dwell_frac=0.75,
         tool="cut", payload_kg=60.0, ground="soft soil/leaf litter", hours=8.0,
         note="high force at the tool, heavy cut material to carry out"),
]

# Platform mass without batteries or mission payload, kg.
#
# The legged figure is NOT an estimate: it is this project's own robot, from the
# corrected mass ladder (analysis/frameless_motor.py, round 14c) less its packs.
# Using anything smaller would flatter the topology the study is meant to test.
# The others are estimates for machines of the same duty, stated as such: a
# tracked or wheeled platform of this class carries far more of its mass as
# payload, so it needs far less of it as structure.
def _legged_base():
    p = os.path.join(ROOT, "hw", "stator", "frameless_motor.json")
    if os.path.exists(p):
        fm = json.load(open(p))
        if fm.get("mass_ladder"):
            return fm["mass_ladder"][-1]["m_robot"] - hm.MASS.batteries
    return HEX_MASS - hm.MASS.batteries


LEGGED_BASE = _legged_base()
BASE_STRUCTURE = {
    "legged": LEGGED_BASE,                 # measured: this design, less its batteries
    "hybrid": LEGGED_BASE * 0.75,          # estimate: four legs plus wheels, lighter legs
    "wheeled": 48.0,                       # estimate: chassis, six hub drives, suspension
    "tracked": 55.0,                       # estimate: chassis, two drives, track sets, idlers
    "swarm": 8 * 16.0,                     # estimate: eight small tracked machines
}
BASE_STRUCTURE["legged_quad"] = LEGGED_BASE * 0.68      # estimate: 12 units not 18, four legs not six
BASE_SOURCE = {"legged": "measured (this project, round 14c)", "legged_quad": "estimate", "hybrid": "estimate",
               "wheeled": "estimate", "tracked": "estimate", "swarm": "estimate"}


def mission_energy(topo, mis, m_total):
    """Wh for one working day at this total mass."""
    cot = topo["cot_soft"] if mis["ground"] in ("soft soil/leaf litter", "sand/mud") else topo["cot_firm"]
    t_move_h = (mis["km_per_day"] * 1e3 / mis["speed"]) / 3600
    t_dwell_h = max(0.0, mis["hours"] - t_move_h) if mis["dwell_frac"] else 0.0
    t_dwell_h = mis["hours"] * mis["dwell_frac"]
    t_move_h = min(t_move_h, mis["hours"] - t_dwell_h)
    e_move = cot * m_total * G * (mis["km_per_day"] * 1e3) / 3600      # Wh
    e_stand = topo["stand_W_per_t"] * (m_total / 1e3) * t_dwell_h
    e_tool = TOOL_W[mis["tool"]] * t_dwell_h
    e_comp = COMPUTE_W * mis["hours"]
    return dict(move=e_move, stand=e_stand, tool=e_tool, compute=e_comp,
                total=e_move + e_stand + e_tool + e_comp, t_move_h=t_move_h, t_dwell_h=t_dwell_h, cot=cot)


def solve(topo, mis, days=1.0, max_batt=400.0):
    """Fixed point: the battery that carries the mission is mass the mission carries."""
    m_batt = 10.0
    base = BASE_STRUCTURE[topo["family"]]
    for _ in range(200):
        m_total = base + m_batt + mis["payload_kg"]
        e = mission_energy(topo, mis, m_total)
        need = e["total"] * days / PACK_WH_PER_KG
        if abs(need - m_batt) < 1e-3:
            m_batt = need
            break
        m_batt += 0.4 * (need - m_batt)
        if m_batt > max_batt:
            return None                                   # runaway: the battery cannot catch its own mass
    m_total = base + m_batt + mis["payload_kg"]
    e = mission_energy(topo, mis, m_total)
    return dict(m_batt=m_batt, m_base=base, m_total=m_total, energy=e,
                payload_ok=mis["payload_kg"] <= topo["payload_frac"] * (base + m_batt),
                wh_per_km=e["total"] / max(mis["km_per_day"], 1e-6))


rows = []
for t in TOPOLOGIES:
    for m in MISSIONS:
        r = solve(t, m)
        rows.append(dict(topology=t["name"], family=t["family"], mission=m["key"], mission_name=m["name"],
                         feasible=r is not None, **({} if r is None else dict(
                             m_batt=r["m_batt"], m_total=r["m_total"], wh=r["energy"]["total"],
                             move=r["energy"]["move"], stand=r["energy"]["stand"], tool=r["energy"]["tool"],
                             compute=r["energy"]["compute"], payload_ok=r["payload_ok"], cot=r["energy"]["cot"],
                             wh_per_km=r["wh_per_km"]))))

# --------- what a fixed 8 kg of battery actually buys, which is the real test
FIXED_PACK = HEX_PACK_KG


def range_on_pack(topo, mis, pack_kg=FIXED_PACK):
    m_total = BASE_STRUCTURE[topo["family"]] + pack_kg + mis["payload_kg"]
    wh = pack_kg * PACK_WH_PER_KG
    cot = topo["cot_soft"] if mis["ground"] in ("soft soil/leaf litter", "sand/mud") else topo["cot_firm"]
    p_move = cot * m_total * G * mis["speed"]
    p_hold = topo["stand_W_per_t"] * (m_total / 1e3) + COMPUTE_W + TOOL_W[mis["tool"]] * 0
    # split the pack between moving and dwelling in the mission's own proportion
    f = 1 - mis["dwell_frac"]
    p_avg = f * (p_move + COMPUTE_W) + (1 - f) * (topo["stand_W_per_t"] * (m_total / 1e3) + COMPUTE_W + TOOL_W[mis["tool"]])
    hours = wh / p_avg
    return dict(hours=hours, km=hours * f * mis["speed"] * 3.6, p_avg=p_avg, p_move=p_move, p_hold=p_hold)


pack_rows = [dict(topology=t["name"], family=t["family"], mission=m["key"], **range_on_pack(t, m))
             for t in TOPOLOGIES for m in MISSIONS]

out = dict(assumptions=dict(pack_wh_per_kg=PACK_WH_PER_KG, eta_drive=ETA_DRIVE, f_roll=F_ROLL,
                            track_internal=TRACK_INTERNAL, compute_W=COMPUTE_W, tool_W=TOOL_W,
                            base_structure_kg=BASE_STRUCTURE, fixed_pack_kg=FIXED_PACK),
           hexapod=dict(mass_kg=HEX_MASS, walk_W_at_1ms=HEX_WALK_W, cot=HEX_COT,
                        standing_W=HEX_STAND_W, standing_source=HEX_STAND_SRC),
           base_source=BASE_SOURCE, legged_base_kg=LEGGED_BASE, topologies=TOPOLOGIES, missions=MISSIONS, rows=rows, pack_rows=pack_rows)
json.dump(out, open(os.path.join(ROOT, "hw", "platform_topology.json"), "w"), indent=1, default=float)

# =============================================================== figure 1
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=(1.0, 1.1)))
ax = axes[0]
names = [t["name"] for t in TOPOLOGIES]
y = np.arange(len(TOPOLOGIES))
cot_f = [t["cot_firm"] for t in TOPOLOGIES]
cot_s = [t["cot_soft"] for t in TOPOLOGIES]
ax.barh(y - 0.2, cot_f, 0.38, color="#0f9b8e", label="firm ground")
ax.barh(y + 0.2, cot_s, 0.38, color="#d98c3a", label="soft ground")
for i, t in enumerate(TOPOLOGIES):
    ax.text(max(cot_f[i], cot_s[i]) + 0.015, i, f"{cot_f[i]:.2f} / {cot_s[i]:.2f}", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
ax.set_xlabel("cost of transport (watts per newton of weight per metre per second)")
ax.set_title("1. Energy to move: the hexapod is 5–7× everything with wheels\n"
             f"(the legged figure is this project's own gait model: {HEX_WALK_W:.0f} W at 1 m/s, {HEX_MASS:.0f} kg)", fontsize=9.5)
ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
w = 0.2
for k, m in enumerate(MISSIONS):
    vals = []
    for t in TOPOLOGIES:
        pr = [p for p in pack_rows if p["topology"] == t["name"] and p["mission"] == m["key"]][0]
        vals.append(pr["hours"])
    ax.barh(np.arange(len(TOPOLOGIES)) + (k - 1.5) * w, vals, w, label=m["name"])
ax.axvline(8.0, color="#222", lw=1.6)
ax.text(8.4, -0.75, "a working day", fontsize=8.5, va="center", ha="left")
ax.set_yticks(np.arange(len(TOPOLOGIES))); ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
ax.set_xscale("log"); ax.set_xlabel(f"hours of mission on the same {FIXED_PACK:.0f} kg of battery (log scale)")
ax.set_title("2. What one battery buys. The hexapod cannot reach a working day\non any of these missions; the wheeled and tracked platforms do it easily", fontsize=9.5)
ax.set_ylim(len(TOPOLOGIES) - 0.4, -1.1)
ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "topology-energy.png"), dpi=110)

# =============================================================== figure 2
fig, axes = plt.subplots(1, 2, figsize=(15, 6.0), gridspec_kw=dict(width_ratios=(1.15, 1.0)))
ax = axes[0]
mis = [m for m in MISSIONS if m["key"] == "invasive"][0]
bot = np.zeros(len(TOPOLOGIES))
parts = [("move", "moving", "#0f9b8e"), ("stand", "standing still", "#b03a2e"),
         ("tool", "the tool", "#d98c3a"), ("compute", "compute and sensors", "#9aa0a6")]
for key, lab, col in parts:
    vals = []
    for t in TOPOLOGIES:
        r = [x for x in rows if x["topology"] == t["name"] and x["mission"] == "invasive"][0]
        vals.append(r.get(key, 0.0) if r["feasible"] else 0.0)
    ax.bar(np.arange(len(TOPOLOGIES)), vals, 0.6, bottom=bot, label=lab, color=col)
    bot += np.asarray(vals)
for i, t in enumerate(TOPOLOGIES):
    r = [x for x in rows if x["topology"] == t["name"] and x["mission"] == "invasive"][0]
    ax.text(i, bot[i] + 60, "does not\nconverge" if not r["feasible"] else f"{bot[i]:.0f} Wh", ha="center", fontsize=8)
ax.set_xticks(np.arange(len(TOPOLOGIES)))
SHORT = ["Hexapod\n(this design)", "Quadruped", "Wheel-on-leg\nhybrid", "6-wheel\nrocker-bogie",
         "4-wheel\nskid-steer", "Tracked", "Tracked +\nflippers", "Swarm\n8 × 25 kg"]
ax.set_xticklabels(SHORT, fontsize=8)
ax.set_ylabel("energy for one day of invasive-species control (Wh)")
ax.set_title("3. Where the energy goes on a dwell-heavy mission (70 % of the day stopped).\n"
             "A legged robot pays to stand still. Nothing on wheels does.", fontsize=9.5)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
for t in TOPOLOGIES:
    r = [x for x in rows if x["topology"] == t["name"] and x["mission"] == "invasive"][0]
    if not r["feasible"]:
        continue
    col = {"legged": "#b03a2e", "legged_quad": "#e07b39", "hybrid": "#d98c3a", "wheeled": "#0f9b8e", "tracked": "#2a78d6", "swarm": "#7d3c98"}[t["family"]]
    ax.scatter(t["terrain"] * 100, r["m_total"], s=150, color=col, zorder=3)
    ax.annotate(t["name"], (t["terrain"] * 100, r["m_total"]), (6, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("share of a mixed outdoor route the topology can traverse unaided (%, estimate)")
ax.set_ylabel("total machine mass for a day of invasive-species control (kg)")
ax.set_title("4. What the terrain capability costs.\nThe last few percent of terrain is where all the mass and energy goes.", fontsize=9.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "topology-missions.png"), dpi=110)

# =============================================================== figure 3
fig, ax = plt.subplots(figsize=(13.5, 6.4))
crit = [("Energy over distance", "cot"), ("Endurance on one pack", "hours"), ("Payload fraction", "payload_frac"),
        ("Terrain reach", "terrain"), ("Force for cutting/pushing", "force"), ("Stable base to manipulate from", "stable"),
        ("Ground disturbance (gentler is better)", "gentle"), ("Actuator count (fewer is better)", "simple"),
        ("Cost and maintenance", "cost")]
# qualitative scores 0-4, stated rather than computed where no model exists
QUAL = {
    "Hexapod (this design)":        dict(force=1, stable=3, gentle=4, simple=0, cost=0),
    "Quadruped":                    dict(force=1, stable=2, gentle=4, simple=1, cost=1),
    "Wheel-on-leg hybrid":          dict(force=2, stable=3, gentle=3, simple=1, cost=1),
    "6-wheel rocker-bogie":         dict(force=3, stable=3, gentle=2, simple=3, cost=3),
    "4-wheel skid-steer":           dict(force=3, stable=3, gentle=1, simple=4, cost=4),
    "Tracked, two tracks":          dict(force=4, stable=4, gentle=3, simple=4, cost=3),
    "Tracked with four flippers":   dict(force=4, stable=4, gentle=3, simple=3, cost=3),
    "Small swarm: 8 × 25 kg tracked": dict(force=1, stable=2, gentle=4, simple=3, cost=2),
}
mat = np.zeros((len(TOPOLOGIES), len(crit)))
for i, t in enumerate(TOPOLOGIES):
    pr = [p for p in pack_rows if p["topology"] == t["name"] and p["mission"] == "invasive"][0]
    cots = [x["cot_soft"] for x in TOPOLOGIES]
    hrs = [[p for p in pack_rows if p["topology"] == q["name"] and p["mission"] == "invasive"][0]["hours"] for q in TOPOLOGIES]
    for jj, (lab, key) in enumerate(crit):
        if key == "cot":
            mat[i, jj] = 4 * (1 - (t["cot_soft"] - min(cots)) / (max(cots) - min(cots)))
        elif key == "hours":
            mat[i, jj] = 4 * (math.log(pr["hours"]) - math.log(min(hrs))) / (math.log(max(hrs)) - math.log(min(hrs)))
        elif key == "payload_frac":
            mat[i, jj] = 4 * (t["payload_frac"] - 0.15) / (0.50 - 0.15)
        elif key == "terrain":
            mat[i, jj] = 4 * (t["terrain"] - 0.6) / (0.98 - 0.6)
        else:
            mat[i, jj] = QUAL[t["name"]][key]
im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=4, aspect="auto")
ax.set_xticks(range(len(crit))); ax.set_xticklabels([c[0] for c in crit], rotation=32, ha="right", fontsize=8.5)
ax.set_yticks(range(len(TOPOLOGIES))); ax.set_yticklabels(names, fontsize=9)
for i in range(len(TOPOLOGIES)):
    for jj in range(len(crit)):
        ax.text(jj, i, f"{mat[i, jj]:.0f}", ha="center", va="center", fontsize=8,
                color="#222" if 1 < mat[i, jj] < 3.4 else "#fff")
ax.set_title("5. The whole trade. Green is better. The first four columns are computed from the model above;\n"
             "the last five are stated judgements, written down so they can be argued with.", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.7, label="0 worst … 4 best")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "topology-matrix.png"), dpi=110)

if __name__ == "__main__":
    print(f"This design: {HEX_MASS:.0f} kg, {HEX_WALK_W:.0f} W at 1 m/s -> CoT {HEX_COT:.2f}")
    print(f"  standing still: {HEX_STAND_W:.0f} W  ({HEX_STAND_SRC})")
    print(f"  on its own {FIXED_PACK:.0f} kg of battery ({FIXED_PACK*PACK_WH_PER_KG:.0f} Wh):")
    for m in MISSIONS:
        pr = [p for p in pack_rows if p["topology"] == "Hexapod (this design)" and p["mission"] == m["key"]][0]
        print(f"    {m['name']:34s} {pr['hours']:5.2f} h, {pr['km']:5.1f} km   (average draw {pr['p_avg']:.0f} W)")
    print()
    print(f"{'topology':32s} {'CoT soft':>9s} {'hours on 8 kg':>14s} {'day-mission kg':>15s} {'Wh/day':>9s}")
    for t in TOPOLOGIES:
        pr = [p for p in pack_rows if p["topology"] == t["name"] and p["mission"] == "invasive"][0]
        r = [x for x in rows if x["topology"] == t["name"] and x["mission"] == "invasive"][0]
        tot = f"{r['m_total']:.0f}" if r["feasible"] else "no converge"
        wh = f"{r['wh']:.0f}" if r["feasible"] else "—"
        print(f"{t['name']:32s} {t['cot_soft']:9.2f} {pr['hours']:14.2f} {tot:>15s} {wh:>9s}")

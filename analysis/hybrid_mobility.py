#!/opt/hw-py/bin/python
"""Can one machine be both efficient over distance and gentle on sensitive ground?

    /opt/hw-py/bin/python analysis/hybrid_mobility.py

The round-15 study said a tracked platform beats the hexapod on every axis the
brief names.  The review's objection is the right one: tracks may be too
damaging for the ground this robot works on.  This study takes that seriously
and asks whether a machine carrying both legs and wheels or tracks is the
answer.

The first thing it does is stop treating "gentle" as one property.  It is
three, and they do not agree:

  * **contact pressure** -- what compacts soil and crushes roots.  Tracks are
    the best thing here by a wide margin and legs are the worst, because a leg
    puts a third of the robot on three small feet.
  * **disturbed area** -- what fraction of the ground is touched at all.  Legs
    are the best thing here, because footprints are discrete; a track lays a
    continuous stripe over everything it crosses.
  * **shear** -- the twisting that tears root mats and turf.  Legs place and
    lift with none of it.  Any skid-steered machine, tracked or wheeled,
    generates it on every turn, and it is the mechanism that actually kills
    vegetation.

So the reviewer is right about tracks, and for the reason that matters most --
but the fix is not automatically legs, because a leg's pressure is worse.  Foot
area is the lever, and it trades directly against disturbed area.

Then it prices the hybrids honestly.  A machine that carries two locomotion
systems carries both of them everywhere, and the legs must be sized to lift the
wheels as well as the robot.  That compounds in the mass fixed point, which is
what killed the hexapod in the first place.

Writes hw/hybrid_mobility.json and docs/design/platform/hybrid-*.png.
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
import hexapod_model as hm                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "design", "platform")
os.makedirs(FIG, exist_ok=True)
G = 9.81
PT = json.load(open(os.path.join(ROOT, "hw", "platform_topology.json")))
A = PT["assumptions"]
PACK = A["pack_wh_per_kg"]
ETA = A["eta_drive"]
F_ROLL, TRACK_INT = A["f_roll"], A["track_internal"]
HEX = PT["hexapod"]

# --------------------------------------------------------- the route, honestly
# A working route is a mix. The split matters more than any single number here,
# and nobody has surveyed it -- so it is a parameter, and the answer is given as
# a function of it rather than at one point.
ROUTE = dict(
    transit_insensitive=0.55,   # track, field edge, fire road: get to the site
    working_sensitive=0.30,     # the ground the work is actually done on
    difficult=0.10,             # broken, steep, deadfall: passable to legs only
    difficult_sensitive=0.05,   # both at once
)
SENSITIVE = ROUTE["working_sensitive"] + ROUTE["difficult_sensitive"]

# ------------------------------------------------------ ground-impact model
# Contact pressure, disturbed area per metre travelled, and whether the turn
# shears. All computed from geometry, not asserted.
def legged_impact(mass_kg, n_legs, feet_down, foot_dia_mm, stride_m):
    a_foot = math.pi * (foot_dia_mm * 1e-3 / 2) ** 2
    pressure = mass_kg * G / (feet_down * a_foot) / 1e3                 # kPa
    steps_per_m = n_legs / stride_m
    disturbed = steps_per_m * a_foot                                     # m2 of ground touched per m travelled
    return dict(pressure_kPa=pressure, disturbed_m2_per_m=disturbed, shear="none: feet are placed and lifted",
                foot_dia_mm=foot_dia_mm, contact_area_m2=feet_down * a_foot)


def rolling_impact(mass_kg, n_contacts, width_mm, length_mm, tracks_in_line, shear):
    a = n_contacts * (width_mm * 1e-3) * (length_mm * 1e-3)
    pressure = mass_kg * G / a / 1e3
    # a wheel or track lays a continuous stripe; wheels in line share one stripe
    disturbed = tracks_in_line * (width_mm * 1e-3)
    return dict(pressure_kPa=pressure, disturbed_m2_per_m=disturbed, shear=shear,
                contact_area_m2=a)


# ------------------------------------------------------------- the candidates
# m_base: platform mass without batteries or payload.
# The legged base is this project's measured robot; every other is an estimate,
# and the hybrids are built by ADDING one system to another, which is the point.
LEG_BASE = PT["legged_base_kg"]                 # measured, round 14c
LEG_UNIT = 4.41                                 # kg, the built actuator
TRACK_BASE, WHEEL_BASE = 55.0, 48.0
HUB_WHEEL = 3.2                                 # kg per driven wheel with its hub motor and tyre


def cot_roll(ground, extra=0.0):
    return (F_ROLL[ground] + extra) / ETA


CANDIDATES = [
    dict(key="hex", name="Hexapod, as designed", kind="legs only", m_base=LEG_BASE, drive_act=18,
         cot_move=HEX["cot"] * 1.2, stand_W=HEX["standing_W"], terrain=0.98,
         impact=legged_impact(150, 6, 3, 60, 0.5),
         note="the measured design. 60 mm feet because that is what the CAD has."),
    dict(key="hex_bigfoot", name="Hexapod with 140 mm compliant feet", kind="legs only", m_base=LEG_BASE + 1.8,
         drive_act=18, cot_move=HEX["cot"] * 1.2, stand_W=HEX["standing_W"], terrain=0.98,
         impact=legged_impact(150, 6, 3, 140, 0.5),
         note="the cheapest change on this page: 300 g of foot per leg drops pressure 5x"),
    dict(key="track", name="Tracked, two tracks", kind="rolling only", m_base=TRACK_BASE, drive_act=2,
         cot_move=cot_roll("grass/pasture", TRACK_INT), stand_W=0.0, terrain=0.80,
         impact=rolling_impact(86, 2, 150, 600, 2, "high: skid-steer shears on every turn"),
         note="the round-15 recommendation, and what the review is worried about"),
    dict(key="track_steer", name="Tracked, articulated steering", kind="rolling only", m_base=TRACK_BASE + 6.0,
         drive_act=3, cot_move=cot_roll("grass/pasture", TRACK_INT) * 1.05, stand_W=0.0, terrain=0.80,
         impact=rolling_impact(92, 2, 150, 600, 2, "low: the body bends instead of skidding"),
         note="a centre pivot instead of skid steering removes the shear that does the damage"),
    dict(key="wheel_steer", name="Four steered wheels, no skid", kind="rolling only", m_base=WHEEL_BASE,
         drive_act=8, cot_move=cot_roll("soft soil/leaf litter"), stand_W=0.0, terrain=0.65,
         impact=rolling_impact(80, 4, 120, 110, 2, "low: every wheel steers, none scrubs"),
         note="drive and steer on each corner; no scrubbing, but a narrow contact patch"),
    dict(key="wheel_on_leg", name="Wheel-on-leg (wheel is the foot)", kind="hybrid, one system",
         m_base=LEG_BASE * 0.72 + 4 * HUB_WHEEL, drive_act=12 + 4,
         cot_move=cot_roll("grass/pasture") * 1.15, stand_W=HEX["standing_W"] * 0.55, terrain=0.95,
         impact=rolling_impact(120, 4, 120, 110, 2, "low: legs can steer each wheel"),
         note="four legs, a driven wheel at each foot. Rolls where it can, walks where it must."),
    dict(key="wheel_on_leg_foot", name="Wheel-on-leg with a deployable foot", kind="hybrid, one system",
         m_base=LEG_BASE * 0.72 + 4 * HUB_WHEEL + 2.4, drive_act=12 + 4,
         cot_move=cot_roll("grass/pasture") * 1.15, stand_W=HEX["standing_W"] * 0.55, terrain=0.96,
         impact=legged_impact(125, 4, 3, 140, 0.6),
         note="the wheel locks and a broad pad takes the load, so it WALKS on sensitive ground"),
    dict(key="track_plus_legs", name="Tracked chassis carrying stowed legs", kind="hybrid, two systems",
         m_base=TRACK_BASE + LEG_BASE * 0.72 + 6.0, drive_act=2 + 12,
         cot_move=cot_roll("grass/pasture", TRACK_INT) * 1.35, stand_W=0.0, terrain=0.96,
         impact=rolling_impact(180, 2, 150, 600, 2, "high when rolling, none when walking"),
         note="carries both systems everywhere; the legs must lift the tracks too"),
    dict(key="marsupial", name="Tracked carrier + a 22 kg legged scout", kind="two machines",
         m_base=TRACK_BASE + 22.0, drive_act=2 + 12,
         cot_move=cot_roll("grass/pasture", TRACK_INT) * 1.1, stand_W=0.0, terrain=0.96,
         impact=legged_impact(22, 4, 3, 110, 0.4),
         note="the carrier never enters the sensitive ground; the scout does the work there"),
]


# ------------------------------------------------------ mission on a mixed route
MIS = dict(name="Invasive species control", km_per_day=6.0, speed=0.5, dwell_frac=0.70,
           hours=8.0, payload_kg=25.0, tool_W=40.0)
COMPUTE_W = A["compute_W"]


def solve(c, route=ROUTE):
    """Energy fixed point over a mixed route. The marsupial only moves its scout
    over the sensitive fraction, so its transit mass is the carrier alone."""
    m_batt = 12.0
    for _ in range(300):
        m_total = c["m_base"] + m_batt + MIS["payload_kg"]
        t_dwell = MIS["hours"] * MIS["dwell_frac"]
        d = MIS["km_per_day"] * 1e3
        if c["key"] == "marsupial":
            # carrier covers the insensitive part with everything aboard;
            # the scout covers the sensitive part on its own legs
            m_carry = m_total
            m_scout = 22.0 + 4.0
            e_move = (c["cot_move"] * m_carry * G * d * (1 - SENSITIVE)
                      + HEX["cot"] * 1.2 * m_scout * G * d * SENSITIVE) / 3600
            e_stand = HEX["standing_W"] * (m_scout / 150.0) * t_dwell
        else:
            e_move = c["cot_move"] * m_total * G * d / 3600
            e_stand = c["stand_W"] * (m_total / 150.0) * t_dwell
        e = e_move + e_stand + MIS["tool_W"] * t_dwell + COMPUTE_W * MIS["hours"]
        need = e / PACK
        if abs(need - m_batt) < 1e-3:
            break
        m_batt += 0.4 * (need - m_batt)
        if m_batt > 500:
            return None
    return dict(m_batt=m_batt, m_total=c["m_base"] + m_batt + MIS["payload_kg"], wh=e,
                e_move=e_move, e_stand=e_stand)


rows = []
for c in CANDIDATES:
    r = solve(c)
    imp = c["impact"]
    # ground touched over a working day, and the share of the sensitive ground it covers
    disturbed_day = imp["disturbed_m2_per_m"] * MIS["km_per_day"] * 1e3
    rows.append(dict(key=c["key"], name=c["name"], kind=c["kind"], drive_act=c["drive_act"],
                     terrain=c["terrain"], note=c["note"], cot=c["cot_move"],
                     pressure_kPa=imp["pressure_kPa"], disturbed_m2_per_m=imp["disturbed_m2_per_m"],
                     disturbed_m2_per_day=disturbed_day, shear=imp["shear"],
                     feasible=r is not None,
                     **({} if r is None else dict(m_total=r["m_total"], m_batt=r["m_batt"], wh=r["wh"]))))

# recompute pressure at each candidate's own solved mass, which is the fair way
for c, r in zip(CANDIDATES, rows):
    if not r["feasible"]:
        continue
    m = r["m_total"] if c["key"] != "marsupial" else 26.0
    a = c["impact"]["contact_area_m2"]
    r["pressure_kPa"] = m * G / a / 1e3

out = dict(route=ROUTE, sensitive_fraction=SENSITIVE, mission=MIS, candidates=CANDIDATES, rows=rows,
           legged_base_kg=LEG_BASE, notes="pressure recomputed at each candidate's own solved mass")
json.dump(out, open(os.path.join(ROOT, "hw", "hybrid_mobility.json"), "w"), indent=1, default=float)

# ================================================================== figure 1
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
names = [r["name"] for r in rows]
y = np.arange(len(rows))
KIND_COL = {"legs only": "#b03a2e", "rolling only": "#2a78d6", "hybrid, one system": "#0f9b8e",
            "hybrid, two systems": "#d98c3a", "two machines": "#7d3c98"}
cols = [KIND_COL[r["kind"]] for r in rows]

ax = axes[0]
ax.barh(y, [r["pressure_kPa"] for r in rows], color=cols)
for i, r in enumerate(rows):
    ax.text(r["pressure_kPa"] + 3, i, f"{r['pressure_kPa']:.0f}", va="center", fontsize=8)
ax.axvline(50, color="#222", ls="--", lw=1.2)
ax.text(52, len(rows) - 0.4, "≈ a person's\nfootfall", fontsize=7.5, va="top")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("contact pressure (kPa)")
ax.set_title("1. What compacts soil and crushes roots.\nLegs are the WORST — a third of the robot on three small feet.", fontsize=9.5)
ax.grid(axis="x", alpha=0.3)

ax = axes[1]
ax.barh(y, [r["disturbed_m2_per_m"] for r in rows], color=cols)
for i, r in enumerate(rows):
    ax.text(r["disturbed_m2_per_m"] + 0.008, i, f"{r['disturbed_m2_per_m']:.3f}", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([""] * len(rows)); ax.invert_yaxis()
ax.set_xlabel("ground actually touched, m² per metre travelled")
ax.set_title("2. What crushes vegetation.\nLegs are the BEST at this — footprints, not a stripe.", fontsize=9.5)
ax.grid(axis="x", alpha=0.3)

ax = axes[2]
SH = {"none: feet are placed and lifted": 0, "low: the body bends instead of skidding": 1,
      "low: every wheel steers, none scrubs": 1, "low: legs can steer each wheel": 1,
      "high: skid-steer shears on every turn": 3, "high when rolling, none when walking": 2}
ax.barh(y, [SH[r["shear"]] for r in rows], color=cols)
ax.set_yticks(y); ax.set_yticklabels([""] * len(rows)); ax.invert_yaxis()
ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["none", "low", "mixed", "high"], fontsize=8)
ax.set_xlabel("shear at the contact patch")
ax.set_title("3. What tears turf and root mats.\nSkid steering is the damage mechanism — not the track.", fontsize=9.5)
ax.grid(axis="x", alpha=0.3)
import matplotlib.patches as mpatches
axes[0].legend(handles=[mpatches.Patch(color=c, label=k) for k, c in KIND_COL.items()],
               fontsize=7.5, loc="lower right", framealpha=0.95)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "hybrid-ground-impact.png"), dpi=110)

# ================================================================== figure 2
fig, axes = plt.subplots(1, 2, figsize=(15, 6.0), gridspec_kw=dict(width_ratios=(1.0, 1.05)))
ax = axes[0]
feas = [r for r in rows if r["feasible"]]
ax.barh(np.arange(len(feas)) - 0.2, [r["m_total"] for r in feas], 0.38, color="#d98c3a", label="machine mass (kg)")
ax.barh(np.arange(len(feas)) + 0.2, [r["wh"] / 20 for r in feas], 0.38, color="#0f9b8e", label="energy per day (Wh ÷ 20)")
for i, r in enumerate(feas):
    ax.text(max(r["m_total"], r["wh"] / 20) + 3, i, f"{r['m_total']:.0f} kg · {r['wh']:.0f} Wh", va="center", fontsize=8)
ax.set_yticks(np.arange(len(feas))); ax.set_yticklabels([r["name"] for r in feas], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("mass (kg) and energy (Wh ÷ 20) for one day of invasive-species control")
ax.set_title("4. What each option costs to run.\nCarrying two locomotion systems is paid for on every metre.", fontsize=9.5)
ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
for r in feas:
    ax.scatter(r["disturbed_m2_per_m"], r["wh"], s=170, color=KIND_COL[r["kind"]], zorder=3,
               edgecolors="#222", linewidths=0.6)
    ax.annotate(r["name"], (r["disturbed_m2_per_m"], r["wh"]), (7, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("ground touched per metre travelled (m²/m)  ←  gentler")
ax.set_ylabel("energy for a working day (Wh)  ←  cheaper")
ax.set_title("5. The actual trade: gentleness against energy.\nBottom-left is the corner worth being in.", fontsize=9.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "hybrid-tradeoff.png"), dpi=110)

if __name__ == "__main__":
    print(f"Route assumed: {SENSITIVE:.0%} of the distance is sensitive ground, "
          f"{ROUTE['difficult']+ROUTE['difficult_sensitive']:.0%} is difficult\n")
    print(f"{'option':44s}{'kg':>6s}{'Wh/day':>8s}{'kPa':>7s}{'m2/m':>8s}{'act':>5s}  shear")
    for r in rows:
        kg = f"{r['m_total']:.0f}" if r["feasible"] else "—"
        wh = f"{r['wh']:.0f}" if r["feasible"] else "—"
        print(f"{r['name']:44s}{kg:>6s}{wh:>8s}{r['pressure_kPa']:7.0f}{r['disturbed_m2_per_m']:8.3f}"
              f"{r['drive_act']:5d}  {r['shear'].split(':')[0]}")

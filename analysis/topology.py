#!/opt/hw-py/bin/python
"""Leg topology trade study -> docs/design/02-leg-topology.md + figures.

    /opt/hw-py/bin/python analysis/topology.py

Answers: which leg architecture performs best in complicated, high-obstacle
terrain while carrying the most payload — and what 'rotating the first motor
axis 90°' actually does.  Everything comes from analysis/leg3d.py.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from hexapod_model import BODY, LOAD_CASES, MASS, G  # noqa: E402
from leg3d import CHOSEN, EXTREME, ROUTINE, STUMBLE, TOPOLOGIES, evaluate, sprawl_ypp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "design", "02-leg-topology.md")
FIG = os.path.join(ROOT, "docs", "design", "topology")
os.makedirs(FIG, exist_ok=True)
CASE = {c.rating: c for c in LOAD_CASES}
WALK, PEAK, RIDER = CASE["continuous"], CASE["peak"], CASE["stretch"]

# ----------------------------------------------------------------------------
# Evaluate every topology
# ----------------------------------------------------------------------------
RESULTS = {}
for t in TOPOLOGIES:
    routine = evaluate(t, WALK.foot_force_z, WALK.foot_force_prop, ROUTINE)
    extreme = evaluate(t, WALK.foot_force_z, WALK.foot_force_prop, EXTREME)
    peak = evaluate(t, PEAK.foot_force_z, PEAK.foot_force_prop, STUMBLE)
    half_w = (BODY.width / 2 + t.neutral_foot[1]) / 1000.0
    com = t.hip_height / 1000.0 + 0.05
    RESULTS[t.key] = {
        "topo": t, "routine": routine, "extreme": extreme, "peak": peak,
        "stance_width": 2 * half_w, "roll_tip_deg": math.degrees(math.atan2(half_w / 2, com)),
        "sum_cont": sum(routine["max"]), "sum_peak": sum(peak["max"]),
    }

# ----------------------------------------------------------------------------
# Step-up cost vs leg proportion (chosen topology)
# ----------------------------------------------------------------------------
STEP = 300.0
PROPORTIONS = [
    ("250 / 500 (2.0×)", sprawl_ypp(femur=250, tibia=500, femur_deg=45, key="p1")),
    ("250 / 625 (2.5×) — agreed", sprawl_ypp(femur=250, tibia=625, femur_deg=55, key="p2")),
    ("300 / 600 (2.0×)", sprawl_ypp(femur=300, tibia=600, femur_deg=50, key="p3")),
    ("300 / 480 (1.6×)", sprawl_ypp(femur=300, tibia=480, femur_deg=40, key="p4")),
]


def step_cost(t):
    """Standing on a STEP-high foothold: how far out the foot must go and what
    the femur pays.  Minimum extension of the knee = tibia − femur."""
    _, femur, tibia = t.link_lengths()
    coxa = t.link_lengths()[0]
    e_min = tibia - femur
    dz = t.hip_height - STEP                       # femur axis above the foothold
    r_neutral = t.neutral_foot[1] - coxa           # femur-axis -> foot, neutral
    r_min = math.sqrt(max(e_min**2 - dz**2, 0.0))  # closest the foot can be, horizontally
    r = max(r_neutral, r_min)
    return {"e_min": e_min, "r_step": r, "outward": r - r_neutral,
            "femur_step": WALK.foot_force_z * r / 1000.0, "femur_neutral": WALK.foot_force_z * r_neutral / 1000.0,
            "hip": t.hip_height}


STEPS = [(label, t, step_cost(t)) for label, t in PROPORTIONS]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_torques():
    keys = [t.key for t in TOPOLOGIES]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [3, 1.2]})
    w = 0.26
    cols = ["#3a3a3a", "#c0392b", "#2980b9"]
    x = np.arange(len(keys))
    for j in range(3):
        mx = [RESULTS[k]["routine"]["max"][j] for k in keys]
        nt = [RESULTS[k]["routine"]["neutral"][j] for k in keys]
        ax.bar(x + (j - 1) * w, mx, w, color=cols[j], alpha=0.35, edgecolor=cols[j])
        ax.bar(x + (j - 1) * w, nt, w, color=cols[j])
        for xi, k in zip(x, keys):
            ax.text(xi + (j - 1) * w, RESULTS[k]["routine"]["max"][j] + 3, RESULTS[k]["topo"].joint_names[j], ha="center", fontsize=7, rotation=90, va="bottom")
    for xi, k in zip(x, keys):
        ax.text(xi, max(RESULTS[k]["routine"]["max"]) + 42, f'Σ {RESULTS[k]["sum_cont"]:.0f} N·m', ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    short = {"sprawl": "Sprawl YPP\nfemur 55° (agreed)", "sprawl_narrow": "Sprawl YPP\nfemur 75°, feet in",
             "mammal": "Mammal RPP\nfeet under", "lizard": "Sprawl RYP\nroll hip"}
    ax.set_xticklabels([short.get(k, k) for k in keys], fontsize=9)
    ax.set_ylabel("joint torque (N·m)")
    ax.set_ylim(0, max(max(RESULTS[k]["routine"]["max"]) for k in keys) + 70)
    ax.set_title(f"Walking load ({WALK.foot_force_z:.0f} N + {WALK.foot_force_prop:.0f} N push per foot): solid = neutral stance, "
                 "pale = max over the routine working volume", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax2.bar(x, [RESULTS[k]["extreme"]["reach_fraction"] * 100 for k in keys], 0.5, color="#0f9b8e")
    ax2.set_xticks(x)
    ax2.set_xticklabels([short.get(k, k).split("\n")[0] for k in keys], fontsize=8)
    ax2.set_ylabel("% of the step-up box reachable")
    ax2.set_ylim(0, 105)
    ax2.set_title("Reach: ±200 fore-aft, −100…+150 lateral,\n0…300 mm up", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG, "torques.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def fig_step():
    fig, axes = plt.subplots(1, len(STEPS), figsize=(12, 5.2), sharey=True)
    for ax, (label, t, s) in zip(axes, STEPS):
        coxa, femur, tibia = t.link_lengths()
        a = math.atan2(t.joints[1].link[2], t.joints[1].link[1])
        kx, kz = femur * math.cos(a), femur * math.sin(a)
        g = -t.hip_height
        ax.plot([-coxa - 120, 650], [g, g], color="#7f8c8d", lw=1.5)
        ax.add_patch(plt.Rectangle((s["r_step"] - 100, g), 420, STEP, color="#d5d8dc"))
        # neutral leg, faint
        ax.plot([-coxa, 0], [0, 0], color="#bbb", lw=4, solid_capstyle="round")
        ax.plot([0, kx], [0, kz], color="#bbb", lw=4, solid_capstyle="round")
        ax.plot([kx, kx], [kz, g], color="#bbb", lw=4, solid_capstyle="round")
        # the same leg standing on the step
        r = s["r_step"]
        zf = g + STEP
        d = math.hypot(r, zf)
        clamp = lambda v: max(-1.0, min(1.0, v))
        a1 = math.atan2(zf, r) + math.acos(clamp((femur**2 + d**2 - tibia**2) / (2 * femur * d)))
        kx2, kz2 = femur * math.cos(a1), femur * math.sin(a1)
        ax.plot([kx2, r], [kz2, zf], "-", color="#2980b9", lw=6, solid_capstyle="round", label="tibia")
        ax.plot([-coxa, 0], [0, 0], "k-", lw=6, solid_capstyle="round", label="coxa")
        ax.plot([0, kx2], [0, kz2], "-", color="#c0392b", lw=6, solid_capstyle="round", label="femur")
        ax.plot([0, kx2, r], [0, kz2, zf], "o", color="k", ms=6)
        ax.annotate("knee", (kx2, kz2), (0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(f"{label}\nhip {s['hip']:.0f} mm, min. extension {s['e_min']:.0f} mm", fontsize=9)
        ax.text(-coxa - 120, g - 80, f"foot {s['outward']:+.0f} mm outward\nfemur {s['femur_step']:.0f} N·m on the step\n({s['femur_neutral']:.0f} N·m at neutral)",
                fontsize=8.5, va="top")
        ax.set_aspect("equal")
        ax.set_xlim(-coxa - 140, 680)
        ax.set_ylim(-760, 330)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"Standing on a {STEP:.0f} mm step at the walking load ({WALK.foot_force_z:.0f} N): grey = neutral leg, coloured = on the step", fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIG, "step-up.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------------
def rel(p):
    return os.path.relpath(p, os.path.dirname(DOC)).replace(os.sep, "/")


def write_doc(figs):
    rows = []
    for t in TOPOLOGIES:
        R = RESULTS[t.key]
        jn = t.joint_names
        rows.append((t.name,
                     " / ".join(f"{n} {v:.0f}" for n, v in zip(jn, R["routine"]["neutral"])),
                     " / ".join(f"{v:.0f}" for v in R["routine"]["max"]),
                     f"**{R['sum_cont']:.0f}**",
                     " / ".join(f"{v:.0f}" for v in R["peak"]["max"]),
                     f"{R['extreme']['reach_fraction']*100:.0f} %",
                     f"{R['stance_width']:.2f}", f"{R['roll_tip_deg']:.0f}°", f"{t.hip_height:.0f}"))
    table = "\n".join(["| Topology | Neutral, walk (N·m) | Routine max, walk (N·m) | Σ routine max | Stumble peak (N·m) | Step-up box reachable | Stance width (m) | Roll tip angle | Hip height (mm) |",
                       "|---|---|---|---|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows])

    srows = []
    for label, t, s in STEPS:
        srows.append((label, f"{s['hip']:.0f}", f"{s['e_min']:.0f}", f"{s['outward']:+.0f}", f"{s['femur_neutral']:.0f}", f"**{s['femur_step']:.0f}**"))
    stable = "\n".join(["| Femur / tibia (mm) | Hip height (mm) | Min. extension (mm) | Foot moves outward to stand on a 300 mm step (mm) | Femur torque, neutral (N·m) | Femur torque on the step (N·m) |",
                        "|---|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in srows])

    S, M, L = RESULTS["sprawl"], RESULTS["mammal"], RESULTS["lizard"]
    doc = f"""# 02 — Leg topology: what the weight levers against, and what wins on rough ground

*Generated by `analysis/topology.py` from `analysis/leg3d.py`. Every torque
here is the moment of the foot force about a real joint axis, with the foot
anywhere in a working volume, not just at the neutral stance.*

The review asked two things: whether rotating the first motor axis by 90°
helps, given that most of the weight is levered against a motor axis and its
mount; and which configuration performs best in complicated, high-obstacle
terrain while carrying the most payload.

## 1. The one rule

A joint sees only the component of the foot's moment about its own axis:

τ = axis · ((foot − joint) × F)

For the vertical part of the foot force — the weight — that means:

* a **vertical axis never sees it**, whatever the foot does. That is the yaw
  joint in the sprawl leg, and it is why the coxa can be as long as stability
  wants at no torque cost. The weight goes into the coxa as bending and into
  the yaw bearing as an overturning moment, which is structure, not copper;
* a **horizontal axis sees weight × the horizontal distance from that axis to
  the foot**, at right angles to the axis. Link angles do not enter. A
  fore-aft axis (a hip *roll* joint) sees the foot's *lateral* offset; a
  lateral axis (a hip *pitch* joint) sees the foot's *fore-aft* offset.

So "rotate the first motor axis 90°" — a roll joint at the body instead of a
yaw joint — does not remove the lever; it moves it. With the same sprawled
leg the roll joint sees the full {L['topo'].neutral_foot[1]:.0f} mm lateral offset of the
foot: {L['routine']['neutral'][0]:.0f} N·m at the neutral stance under the walking load, against
{S['routine']['neutral'][0]:.0f} N·m for the yaw joint (propulsion only). It becomes a good idea only if
the feet also move in under the hips, which is the mammal layout — a
different robot with a different stance width, not a motor rotated in place.

## 2. Four topologies under the same load

All four use the same body ({BODY.width:.0f} mm between hip axes), the same walking load
({WALK.foot_force_z:.0f} N vertical, {WALK.foot_force_prop:.0f} N push, 30 % of that sideways) and the same
working volumes: *routine* is ±200 mm fore-aft, ±100 mm lateral and 0–150 mm
up from the neutral foot; the *step-up box* extends to +150 mm outboard and
300 mm up. The sprawl legs use the agreed 150 / 250 / 625 mm links; the mammal
leg uses proportions that suit it (100 mm carrier, 300 / 300 mm) because the
comparison is between the best each can do, not the same links forced into
both.

{table}

![Joint torques by topology]({rel(figs['torques'])})

Reading it:

* **Sprawl, yaw–pitch–pitch (A, agreed).** Nothing on the yaw joint but
  propulsion; the femur carries the weight over a {CHOSEN.neutral_foot[1]-150:.0f} mm arm at neutral.
  The pale bars are the corners of the routine volume: pulling the foot out
  or up puts the load on the knee through the tibia's lean, so the knee's
  routine maximum is on par with the femur's. Widest stance, best roll
  margin, highest body.
* **Same leg with the femur at 75° (feet in).** Lower femur torque, {RESULTS['sprawl_narrow']['stance_width']:.2f} m
  stance, and the fold limit bites harder — only {RESULTS['sprawl_narrow']['extreme']['reach_fraction']*100:.0f} % of the step-up box.
  It shows the trade inside the chosen topology: the femur angle is a
  posture the controller can choose, not a build decision.
* **Mammal, roll–pitch–pitch (B).** Lowest sum at the neutral stance and the
  whole step-up box reachable, because 300 / 300 mm links fold to nothing.
  But the routine maximum is not lower than the sprawl's: as soon as the foot
  steps sideways the roll joint picks up weight × lateral offset, and the
  femur carries the whole propulsion force over the leg's height. Its stance
  is {M['stance_width']:.2f} m against {S['stance_width']:.2f} m, so roll margin is {M['roll_tip_deg']:.0f}° against {S['roll_tip_deg']:.0f}°. With a
  payload on the deck that is the number that decides whether the robot
  walks a side slope or crawls it.
* **Sprawl with a roll hip (roll–yaw–pitch).** The literal "first axis turned
  90°". The roll joint sees {L['routine']['neutral'][0]:.0f} N·m standing still and {L['routine']['max'][0]:.0f} N·m in the routine
  volume — the worst of both worlds. Not a candidate.

## 3. What a high step really costs: the leg proportion, not the topology

The step-up box exposed something the neutral-stance sizing did not: a knee
cannot fold the foot closer to the femur axis than **tibia − femur**. With a
625 mm tibia and a 250 mm femur that is {STEPS[1][2]['e_min']:.0f} mm, so a foot that has to
come up 300 mm cannot stay where it was — it has to move outward, and the
femur pays for the reach.

{stable}

![Step-up cost by leg proportion]({rel(figs['step'])})

The agreed 2.5× leg is the tallest and clears the most, and it steps high by
reaching out: {STEPS[1][2]['outward']:+.0f} mm outward on a 300 mm foothold, at {STEPS[1][2]['femur_step']:.0f} N·m on the
femur under the walking load. A 2.0× leg of the same femur folds further and
stands on the same step at {STEPS[0][2]['femur_step']:.0f} N·m, but from a {STEPS[0][2]['hip']:.0f} mm hip instead of {STEPS[1][2]['hip']:.0f} mm.
This is the real terrain trade-off and it is inside the chosen topology; the
actuator ratings in `01-sizing.md` §6 are taken over the working volume so
they already carry it.

## 4. Answer

For complicated, high-obstacle terrain with the most payload, **the sprawl
yaw–pitch–pitch leg is the right topology**, for three reasons that survive
the numbers above: the weight never loads the first joint, so stance width is
free; the roll margin with a deck payload is roughly double the mammal's; and
foot placement is a ring around each hip rather than a slot under it, which
is what sparse footholds need. Its costs are real and now quantified: the
knee must be rated with the femur, the coxa and yaw bearing carry a
{PEAK.foot_force_z * CHOSEN.foot_radius / 1000:.0f} N·m overturning moment at the peak load, and the femur and knee
drives have to cross the yaw axis to reach their joints.

The mammal layout wins on one thing that matters to the "motors in the body"
dream: its femur and knee motors sit coaxially on the hip carrier and need no
transmission through a coxa. That is an argument for building the first
prototype leg of the sprawl design with the same coaxial idea — both pitch
drives concentric on the yaw axis, then along the coxa — rather than for
changing the topology. See `03-architecture-levers.md`.

What was not varied here: a fourth joint per leg (a knee-yaw or an ankle),
non-serial legs (a five-bar or pantograph puts both pitch motors at the hip
and can change the torque split), and dynamic loads. Each is a lever the
actuator stage can pull; none changes the conclusion about the first axis.
"""
    with open(DOC, "w") as f:
        f.write(doc)


if __name__ == "__main__":
    figs = {"torques": fig_torques(), "step": fig_step()}
    write_doc(figs)
    print("wrote", DOC)
    for k, R in RESULTS.items():
        print(f"{k:14s} neutral {[round(v) for v in R['routine']['neutral']]} routine {[round(v) for v in R['routine']['max']]} "
              f"sum {R['sum_cont']:.0f} reach {R['extreme']['reach_fraction']*100:.0f}% stance {R['stance_width']:.2f} tip {R['roll_tip_deg']:.0f}")
    for label, t, s in STEPS:
        print(label, {k: round(v) for k, v in s.items()})

#!/opt/hw-py/bin/python
"""Figures for docs/design/05-bio-inspiration.md.

    /opt/hw-py/bin/python analysis/bio_figure.py

(a) an insect leg and our leg drawn to the same scheme, joint axes named;
(b) what pointing the ground reaction force at the hip does to our femur
    torque — the cockroach's trick, computed on our leg with analysis/leg3d.py.
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
from hexapod_model import LOAD_CASES, STANCE  # noqa: E402
from leg3d import CHOSEN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "design", "bio")
os.makedirs(FIG, exist_ok=True)
WALK = [c for c in LOAD_CASES if c.rating == "continuous"][0]


def fig_legs():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2))
    # --- (a) insect leg, schematic, posterior view (body on the left)
    a.set_title("Insect leg (schematic, posterior view)\nsegments and joint axes of a cockroach / stick insect", fontsize=10)
    body = plt.Rectangle((-1.6, -0.35), 1.6, 0.7, color="#bdc3c7")
    a.add_patch(body)
    a.text(-0.8, 0, "thorax", ha="center", va="center", fontsize=9, color="#555")
    pts = {"ThC": (0.0, 0.0), "CTr": (0.35, -0.1), "FTi": (1.55, 0.55), "TiTa": (2.15, -0.9), "toe": (2.7, -1.0)}
    segs = [("coxa", "ThC", "CTr", "#3a3a3a", 9), ("femur", "CTr", "FTi", "#c0392b", 9), ("tibia", "FTi", "TiTa", "#2980b9", 8), ("tarsus", "TiTa", "toe", "#8e44ad", 5)]
    for name, p, q, col, lw in segs:
        (x1, y1), (x2, y2) = pts[p], pts[q]
        a.plot([x1, x2], [y1, y2], "-", color=col, lw=lw, solid_capstyle="round")
        a.text((x1 + x2) / 2 + (0.22 if name == "tibia" else 0), (y1 + y2) / 2 + (0.16 if name != "tibia" else 0), name, ha="center", fontsize=9, color=col)
    for k, (x, y) in pts.items():
        if k != "toe":
            a.plot(x, y, "o", color="k", ms=7)
    a.annotate("ThC: slanted axis (~30° off vertical)\nprotraction / retraction — the stride", pts["ThC"], (-1.55, 1.25), fontsize=8, arrowprops=dict(arrowstyle="-", color="#777"))
    a.annotate("CTr: hinge, levation / depression\n— the power joint (propulsion by pressing down)", pts["CTr"], (-1.75, -1.2), fontsize=8, arrowprops=dict(arrowstyle="-", color="#777"))
    a.annotate("FTi: hinge, flexion / extension\n— posture, aims the push", pts["FTi"], (1.2, 1.25), fontsize=8, arrowprops=dict(arrowstyle="-", color="#777"))
    a.annotate("tarsus: 5 passive segments,\nclaws, pads, tibial spines", pts["TiTa"], (0.9, -1.85), fontsize=8, arrowprops=dict(arrowstyle="-", color="#777"))
    # ground reaction force toward the coxa
    a.arrow(3.0, -1.55, -0.45, 0.55, width=0.03, color="#27ae60", length_includes_head=True)
    a.text(2.75, -1.95, "ground reaction force:\ntoward the coxa,\n20–26° off vertical", fontsize=8, color="#27ae60")
    a.set_xlim(-1.8, 3.6)
    a.set_ylim(-2.3, 1.7)
    a.set_aspect("equal")
    a.axis("off")

    # --- (b) our leg, same scheme
    b.set_title("Our leg (same scheme)\ncoxa 150 / femur 250 / tibia 500 mm, tibia vertical", fontsize=10)
    s = 1 / 300.0
    coxa, femur, tibia = STANCE.leg.coxa, STANCE.leg.femur, STANCE.leg.tibia
    kx, kz = STANCE.knee
    pts2 = {"yaw": (0.0, 0.0), "femur": (coxa * s, 0.0), "knee": ((coxa + kx) * s, kz * s), "foot": ((coxa + kx) * s, (kz - tibia) * s)}
    body = plt.Rectangle((-1.6, 0.0), 1.6, 0.66, color="#bdc3c7")
    b.add_patch(body)
    b.text(-0.8, 0.33, "body", ha="center", va="center", fontsize=9, color="#555")
    for name, p, q, col in (("coxa", "yaw", "femur", "#3a3a3a"), ("femur", "femur", "knee", "#c0392b"), ("tibia", "knee", "foot", "#2980b9")):
        (x1, y1), (x2, y2) = pts2[p], pts2[q]
        b.plot([x1, x2], [y1, y2], "-", color=col, lw=9, solid_capstyle="round")
        b.text((x1 + x2) / 2 + (0.22 if name == "tibia" else -0.22 if name == "femur" else 0), (y1 + y2) / 2 + (0.16 if name == "coxa" else 0), name, ha="center", fontsize=9, color=col)
    for k in ("yaw", "femur", "knee"):
        b.plot(*pts2[k], "o", color="k", ms=7)
    b.plot(*pts2["foot"], "o", color="#2980b9", ms=9)
    b.annotate("yaw: vertical axis — the stride;\nnever sees weight", pts2["yaw"], (-1.55, 1.25), fontsize=8, arrowprops=dict(arrowstyle="-", color="#777"))
    b.annotate("femur pitch: the power joint\n(levation / depression)", pts2["femur"], (0.05, -1.0), fontsize=8, arrowprops=dict(arrowstyle="-", color="#777"))
    b.annotate("knee: posture, aims the push", pts2["knee"], (1.3, 1.25), fontsize=8, arrowprops=dict(arrowstyle="-", color="#777"))
    b.annotate("foot: to add — compliant,\nclawed / spined, passive", pts2["foot"], (1.75, -1.45), fontsize=8, arrowprops=dict(arrowstyle="-", color="#777"))
    fx, fz = pts2["foot"]
    b.arrow(fx, fz - 0.6, 0, 0.45, width=0.03, color="#27ae60", length_includes_head=True)
    b.text(fx + 0.1, fz - 0.55, "vertical load\n(as sized)", fontsize=8, color="#27ae60")
    b.set_xlim(-1.8, 3.6)
    b.set_ylim(-2.3, 1.7)
    b.set_aspect("equal")
    b.axis("off")
    fig.tight_layout()
    p = os.path.join(FIG, "insect-leg.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def _planar(kx, kz, h, r, Fz, f):
    """Planar joint torques (N·m) for a foot at (r, -h) from the femur axis, knee at
    (kx, kz), vertical load Fz and an in-plane force f·Fz pointing toward the body."""
    fem = abs(r * Fz - h * f * Fz) / 1000.0
    knee = abs((r - kx) * Fz - (h + kz) * f * Fz) / 1000.0
    return fem, knee


def fig_grf():
    """The cockroach's trick — point the ground reaction force at the hip —
    computed on our leg in both stances."""
    Fz = WALK.foot_force_z
    fracs = np.linspace(0, 0.6, 61)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    # (a) sprawl stance, neutral: foot under the knee, knee above the femur axis
    kx, kz = STANCE.knee
    h = STANCE.hip_height
    fem = [_planar(kx, kz, h, STANCE.foot_reach, Fz, f)[0] for f in fracs]
    knee = [_planar(kx, kz, h, STANCE.foot_reach, Fz, f)[1] for f in fracs]
    a.plot(fracs * 100, fem, color="#c0392b", lw=2.5, label="femur")
    a.plot(fracs * 100, knee, color="#2980b9", lw=2.5, label="knee")
    a.plot(fracs * 100, np.add(fem, knee), "k--", lw=1.8, label="femur + knee")
    a.axvspan(35, 52, color="#27ae60", alpha=0.12)
    a.text(43.5, max(np.add(fem, knee)) * 0.97, "cockroach: lateral\n35–52 % of vertical", ha="center", va="top", fontsize=8, color="#27ae60")
    a.set_title(f"Sprawl stance, neutral ({Fz:.0f} N): inward push toward the body\n"
                "knee above the femur axis, tibia vertical — the push moves torque to the knee", fontsize=9.5)
    a.set_xlabel("inward foot force, % of the vertical load")
    a.set_ylabel("joint torque (N·m)")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="upper left")
    # (b) mammal stance, foot 200 mm ahead of the hip, push toward the hip (braking)
    am = math.radians(-45.0)
    kxm, kzm = STANCE.leg.femur * math.cos(am), STANCE.leg.femur * math.sin(am)
    hm = -kzm + math.sqrt(STANCE.leg.tibia**2 - kxm**2)
    r = 200.0
    fem2 = [_planar(kxm, kzm, hm, r, Fz, f)[0] for f in fracs]
    knee2 = [_planar(kxm, kzm, hm, r, Fz, f)[1] for f in fracs]
    b.plot(fracs * 100, fem2, color="#c0392b", lw=2.5, label="femur")
    b.plot(fracs * 100, knee2, color="#2980b9", lw=2.5, label="knee")
    b.plot(fracs * 100, np.add(fem2, knee2), "k--", lw=1.8, label="femur + knee")
    i = int(np.argmin(np.add(fem2, knee2)))
    b.axvline(fracs[i] * 100, color="k", ls=":", lw=1)
    b.annotate(f"minimum at {fracs[i]*100:.0f} %: {fem2[i] + knee2[i]:.0f} N·m\n(vs {fem2[0] + knee2[0]:.0f} N·m with no push)",
               (fracs[i] * 100, fem2[i] + knee2[i]), (fracs[i] * 100 + 6, (fem2[0] + knee2[0]) * 0.9), fontsize=9,
               arrowprops=dict(arrowstyle="->", color="k"))
    b.set_title(f"Mammal stance, foot {r:.0f} mm ahead of the hip ({Fz:.0f} N): push toward the hip\n"
                "knee below the femur axis — the push aligns the force with the leg", fontsize=9.5)
    b.set_xlabel("horizontal foot force toward the hip, % of the vertical load")
    b.set_ylabel("joint torque (N·m)")
    b.grid(alpha=0.3)
    b.legend(fontsize=8, loc="upper right")
    fig.suptitle("Pointing the ground reaction force at the hip, on our leg: it pays where the knee is below the hip, not where it is above", fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIG, "grf-direction.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p, fracs[i], fem2[0] + knee2[0], fem2[i] + knee2[i], fem[0], knee[-1]


if __name__ == "__main__":
    print(fig_legs())
    p, f, s0, smin, fem0, knee_end = fig_grf()
    print(p, f"mammal: push {f*100:.0f}% -> sum {s0:.0f} -> {smin:.0f} N·m; sprawl: femur {fem0:.0f} at f=0, knee {knee_end:.0f} at 60%")

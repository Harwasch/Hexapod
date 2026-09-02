#!/opt/hw-py/bin/python
"""Dimensioned orthographic drawings of the current geometry.

    /opt/hw-py/bin/python analysis/drawing.py

Writes docs/design/drawing/{top,front,side,hip-stack}.png and
docs/design/06-geometry.md.  Every dimension is read from the model
(analysis/hexapod_model.py, analysis/leg3d.py, concepts/hexapod_skeleton.py),
so the drawing cannot disagree with the sizing.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "concepts"))
from hexapod_model import ACT, BODY, STANCE, YAW_RANGE_DEG, HUMAN_HEIGHT  # noqa: E402
from leg3d import CHOSEN, MAMMAL_MODE  # noqa: E402
from hexapod_skeleton import (BATTERY, COMPUTE, HIP_DROP, LEG_YAW, PLATE_T, leg_points)  # noqa: E402

OUT = os.path.join(ROOT, "docs", "design", "drawing")
DOC = os.path.join(ROOT, "docs", "design", "06-geometry.md")
os.makedirs(OUT, exist_ok=True)

INK = "#1f1f1f"
DIM = "#b03a2e"
FILL = "#d9dcdf"
ACTC = "#0f9b8e"
LEGC = {"coxa": "#3a3a3a", "femur": "#c0392b", "tibia": "#2980b9"}
W = BODY.slab_width(ACT)
L, H = BODY.length, BODY.height
GROUND_SPRAWL = -HIP_DROP - CHOSEN.hip_height
GROUND_MAMMAL = -HIP_DROP - MAMMAL_MODE.hip_height


# ----------------------------------------------------------------------------
# dimensioning helpers
# ----------------------------------------------------------------------------
def dim(ax, p1, p2, offset, text=None, fs=8, ext=True):
    """A dimension line between p1 and p2, offset perpendicular by `offset`
    (positive = to the left of p1->p2).  Text is the length unless given."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    a, b = p1 + n * offset, p2 + n * offset
    if ext:
        for p, q in ((p1, a), (p2, b)):
            ax.plot([p[0], q[0] + n[0] * 6 * np.sign(offset)], [p[1], q[1] + n[1] * 6 * np.sign(offset)], color=DIM, lw=0.6)
    ax.annotate("", xy=a, xytext=b, arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.8, shrinkA=0, shrinkB=0))
    mid = (a + b) / 2
    t = text if text is not None else f"{np.linalg.norm(d):.0f}"
    ang = math.degrees(math.atan2(d[1], d[0]))
    if ang > 90 or ang < -90:
        ang += 180
    ax.text(mid[0] + n[0] * 12, mid[1] + n[1] * 12, t, color=DIM, fontsize=fs, ha="center", va="center", rotation=ang, rotation_mode="anchor")


def label(ax, xy, text, dxy=(0, 0), fs=8, color=INK, ha="left"):
    ax.annotate(text, xy, (xy[0] + dxy[0], xy[1] + dxy[1]), fontsize=fs, color=color, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.6) if dxy != (0, 0) else None)


def frame(ax, title, xl, yl):
    ax.set_aspect("equal")
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")


def legs_2d(topo, stance_radial, proj):
    """All six legs' joint points projected: proj in {'top': (x,y), 'front': (y,z), 'side': (x,z)}."""
    out = []
    for i, hx in enumerate(BODY.hip_x):
        for s in (1, -1):
            hip = (hx, s * BODY.width / 2, -HIP_DROP)
            P, _ = leg_points(topo, hip, s, LEG_YAW[i] if stance_radial else 0.0, (not stance_radial) and hx < 0)
            if proj == "top":
                out.append([(p[0], p[1]) for p in P])
            elif proj == "front":
                out.append([(p[1], p[2]) for p in P])
            else:
                out.append([(p[0], p[2]) for p in P])
    return out


def draw_leg(ax, pts, lw=5):
    for (p, q), name in zip(zip(pts[:-1], pts[1:]), ("coxa", "femur", "tibia")):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=LEGC[name], lw=lw, solid_capstyle="round", zorder=3)
    for p in pts[:3]:
        ax.plot(p[0], p[1], "o", color=INK, ms=4, zorder=4)
    ax.plot(pts[3][0], pts[3][1], "o", color=LEGC["tibia"], ms=6, mec=INK, zorder=4)


# ----------------------------------------------------------------------------
# views
# ----------------------------------------------------------------------------
def view_top():
    fig, ax = plt.subplots(figsize=(11, 8.5))
    # slab
    ax.add_patch(Rectangle((-L / 2, -W / 2), L, W, facecolor=FILL, edgecolor=INK, lw=1.2, zorder=1))
    # hip stacks and yaw axes
    for hx in BODY.hip_x:
        for s in (1, -1):
            ax.add_patch(Circle((hx, s * BODY.width / 2), ACT.od / 2, facecolor="#cfe9e6", edgecolor=ACTC, lw=1.2, zorder=2))
            ax.plot(hx, s * BODY.width / 2, "+", color=ACTC, ms=8, mew=1.2, zorder=3)
    # batteries, compute (hidden lines)
    for s in (1, -1):
        ax.add_patch(Rectangle((s * 165 - BATTERY[0] / 2, -BATTERY[1] / 2), BATTERY[0], BATTERY[1], fill=False, edgecolor="#777", ls="--", lw=0.8, zorder=2))
    ax.add_patch(Rectangle((-COMPUTE[0] / 2, -COMPUTE[1] / 2), COMPUTE[0], COMPUTE[1], fill=False, edgecolor="#777", ls="--", lw=0.8, zorder=2))
    # legs, sprawl stance
    legs = legs_2d(CHOSEN, True, "top")
    for pts in legs:
        draw_leg(ax, pts)
    feet = np.array([p[3] for p in legs])
    # foot circle radius around one hip
    r = CHOSEN.foot_radius
    ax.add_patch(Circle((BODY.hip_x[1], BODY.width / 2), r, fill=False, edgecolor=LEGC["tibia"], ls=":", lw=0.8))
    # dimensions
    dim(ax, (-L / 2, -W / 2), (L / 2, -W / 2), -700, f"body length {L:.0f}")
    dim(ax, (L / 2, -W / 2), (L / 2, W / 2), -130, f"slab width {W:.0f}")
    dim(ax, (BODY.hip_x[2], -BODY.width / 2), (BODY.hip_x[2], BODY.width / 2), 300, f"hip spacing {BODY.width:.0f}")
    dim(ax, (BODY.hip_x[2], W / 2), (BODY.hip_x[1], W / 2), 560, f"{BODY.hip_x[1] - BODY.hip_x[2]:.0f}")
    dim(ax, (BODY.hip_x[1], W / 2), (BODY.hip_x[0], W / 2), 560, f"{BODY.hip_x[0] - BODY.hip_x[1]:.0f}")
    xs, ys = feet[:, 0], feet[:, 1]
    dim(ax, (xs.min(), ys.min()), (xs.max(), ys.min()), -230, f"stance length {xs.max() - xs.min():.0f}")
    dim(ax, (xs.max(), ys.min()), (xs.max(), ys.max()), -220, f"stance width {ys.max() - ys.min():.0f}")
    dim(ax, (BODY.hip_x[1], BODY.width / 2), (BODY.hip_x[1], BODY.width / 2 + r), -90, f"foot radius {r:.0f}")
    dim(ax, (BODY.hip_x[0] - ACT.od / 2, -BODY.width / 2), (BODY.hip_x[0] + ACT.od / 2, -BODY.width / 2), -200, f"Ø{ACT.od:.0f}")
    # leg-plane yaw
    ax.annotate(f"front / rear legs yawed ±{LEG_YAW[0]:.0f}°\nyaw range ±{YAW_RANGE_DEG:.0f}°", (BODY.hip_x[0], BODY.width / 2), (BODY.hip_x[0] - 330, W / 2 + 330),
                fontsize=8, arrowprops=dict(arrowstyle="-", color="#888", lw=0.6))
    label(ax, (165, 0), "battery (×2)", (0, 0), fs=7, color="#555", ha="center")
    label(ax, (0, 0), "compute", (0, 0), fs=7, color="#555", ha="center")
    label(ax, (BODY.hip_x[2], -BODY.width / 2), "hip stack:\n3 × Ø170 pancakes", (-150, -290), fs=8)
    ax.text(0, -W / 2 - 900, "+x forward →", fontsize=8, ha="center", color="#555")
    frame(ax, f"TOP VIEW — sprawl stance, all six legs at neutral (mm)", (-L / 2 - 500, L / 2 + 500), (-W / 2 - 1000, W / 2 + 720))
    fig.tight_layout()
    p = os.path.join(OUT, "top.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def view_front():
    fig, ax = plt.subplots(figsize=(11, 7))
    g = GROUND_SPRAWL
    # ground
    ax.plot([-1000, 1000], [g, g], color="#7f8c8d", lw=1.5)
    ax.add_patch(Rectangle((-1000, g - 25), 2000, 25, facecolor="#e8e8e8", edgecolor="none"))
    # slab section (front view sees the slab end)
    ax.add_patch(Rectangle((-W / 2, 0), W, H, facecolor=FILL, edgecolor=INK, lw=1.2, zorder=1))
    # hip stacks (hidden) at y = ±120
    for s in (1, -1):
        for i in range(3):
            zc = PLATE_T + 4 + i * (ACT.thickness + ACT.stack_gap)
            ax.add_patch(Rectangle((s * BODY.width / 2 - ACT.od / 2, zc), ACT.od, ACT.thickness, fill=False, edgecolor=ACTC, ls="--", lw=0.8, zorder=2))
        ax.plot([s * BODY.width / 2] * 2, [-HIP_DROP - 30, H + 40], color=ACTC, lw=0.6, ls="-.")
    # mid legs (the pair in the y–z plane), front view
    legs = legs_2d(CHOSEN, True, "front")
    for pts in (legs[2], legs[3]):     # mid legs
        draw_leg(ax, pts)
    for pts in (legs[0], legs[1], legs[4], legs[5]):
        draw_leg(ax, pts, lw=2)
    # figure of a person for scale (6 ft)
    hx = -W / 2 - 720
    ax.add_patch(Rectangle((hx - 110, g), 220, HUMAN_HEIGHT, facecolor="#eeeeee", edgecolor="#bbb", lw=0.8))
    ax.text(hx, g + HUMAN_HEIGHT + 30, "6 ft", ha="center", fontsize=8, color="#888")
    # dimensions (right side)
    kz = legs[2][2][1]
    dim(ax, (W / 2 + 40, g), (W / 2 + 40, -HIP_DROP), -520, f"femur axis height {-HIP_DROP - g:.0f}")
    dim(ax, (W / 2 + 40, g), (W / 2 + 40, kz), -640, f"knee height {kz - g:.0f}")
    dim(ax, (W / 2 + 40, g), (W / 2 + 40, H), -760, f"deck height {H - g:.0f}")
    dim(ax, (-W / 2, H + 30), (W / 2, H + 30), 110, f"slab width {W:.0f}")
    dim(ax, (W / 2 + 130, 0), (W / 2 + 130, H), -60, f"{H:.0f}")
    feet = [p[3] for p in (legs[2], legs[3])]
    dim(ax, (feet[1][0], g - 60), (feet[0][0], g - 60), -160, f"stance width (mid legs) {feet[0][0] - feet[1][0]:.0f}")
    # leg link dimensions on the right mid leg
    P = legs[2]
    dim(ax, P[0], P[1], -90, f"coxa {CHOSEN.link_lengths()[0]:.0f}")
    dim(ax, P[1], P[2], 110, f"femur {CHOSEN.link_lengths()[1]:.0f}")
    dim(ax, P[2], P[3], -110, f"tibia {CHOSEN.link_lengths()[2]:.0f}")
    ax.annotate(f"femur {STANCE.femur_deg:.0f}° above horizontal,\ntibia vertical", P[1], (P[1][0] - 420, P[1][1] - 200), fontsize=8, arrowprops=dict(arrowstyle="-", color="#888", lw=0.6), ha="left")
    frame(ax, "FRONT VIEW (looking along −x) — sprawl stance (mm)", (-W / 2 - 950, W / 2 + 950), (g - 250, max(H, HUMAN_HEIGHT + g) + 150))
    fig.tight_layout()
    p = os.path.join(OUT, "front.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def view_side():
    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    for ax, topo, radial, g, title in ((axes[0], CHOSEN, True, GROUND_SPRAWL, "SIDE VIEW (looking along −y) — sprawl stance (mm)"),
                                       (axes[1], MAMMAL_MODE, False, GROUND_MAMMAL, "SIDE VIEW — same robot, mammal stance: legs yawed 90°, femur −45° (mm)")):
        ax.plot([-1300, 1300], [g, g], color="#7f8c8d", lw=1.5)
        ax.add_patch(Rectangle((-L / 2, 0), L, H, facecolor=FILL, edgecolor=INK, lw=1.2, zorder=1))
        for hx in BODY.hip_x:
            for i in range(3):
                zc = PLATE_T + 4 + i * (ACT.thickness + ACT.stack_gap)
                ax.add_patch(Rectangle((hx - ACT.od / 2, zc), ACT.od, ACT.thickness, fill=False, edgecolor=ACTC, ls="--", lw=0.8, zorder=2))
        legs = legs_2d(topo, radial, "side")
        for k, pts in enumerate(legs):
            draw_leg(ax, pts, lw=5 if k % 2 == 0 else 2)
        feet = np.array([p[3] for p in legs])
        dim(ax, (-L / 2, H + 30), (L / 2, H + 30), 110, f"body length {L:.0f}")
        dim(ax, (L / 2 + 60, g), (L / 2 + 60, -HIP_DROP), -300, f"femur axis {-HIP_DROP - g:.0f}")
        dim(ax, (L / 2 + 60, g), (L / 2 + 60, H), -440, f"deck {H - g:.0f}")
        dim(ax, (feet[:, 0].min(), g - 60), (feet[:, 0].max(), g - 60), -120, f"stance length {feet[:, 0].max() - feet[:, 0].min():.0f}")
        for hx in BODY.hip_x:
            ax.plot([hx, hx], [-HIP_DROP - 20, H + 20], color=ACTC, lw=0.6, ls="-.")
        if radial:
            dim(ax, (BODY.hip_x[2], H), (BODY.hip_x[1], H), 50, f"hip pitch {BODY.hip_x[1] - BODY.hip_x[2]:.0f}", ext=False)
            dim(ax, (BODY.hip_x[1], H), (BODY.hip_x[0], H), 50, f"{BODY.hip_x[0] - BODY.hip_x[1]:.0f}", ext=False)
        else:
            kz = max(p[2][1] for p in legs)
            ax.annotate(f"knees forward on front and mid legs, back on the rear\nfoot under the hip; {topo.hip_height:.0f} mm femur-axis height", (BODY.hip_x[0], kz), (BODY.hip_x[0] - 250, kz - 330), fontsize=8, arrowprops=dict(arrowstyle="-", color="#888", lw=0.6))
        frame(ax, title, (-L / 2 - 700, L / 2 + 700), (g - 200, H + 260))
    fig.tight_layout()
    p = os.path.join(OUT, "side.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def view_hip_stack():
    """Section through one hip: the three pancakes on the yaw axis, the yaw
    output through the floor, the coxa leaving below."""
    fig, ax = plt.subplots(figsize=(9, 7))
    r = ACT.od / 2
    # floor and deck plates (section)
    for z in (0, H - PLATE_T):
        ax.add_patch(Rectangle((-r - 60, z), 2 * r + 120, PLATE_T, facecolor="#999", edgecolor=INK, lw=0.8, hatch="////"))
    # the three pancakes
    names = ("yaw actuator (bottom, drives the coxa directly)", "femur actuator", "knee actuator (top)")
    for i in range(3):
        z0 = PLATE_T + 4 + i * (ACT.thickness + ACT.stack_gap)
        ax.add_patch(Rectangle((-r, z0), 2 * r, ACT.thickness, facecolor="#cfe9e6", edgecolor=ACTC, lw=1.2))
        # magnet annulus and cycloid bore, indicative
        for s in (1, -1):
            ax.add_patch(Rectangle((s * 55 if s > 0 else -85, z0 + 6), 30, ACT.thickness - 12, facecolor="#9fd3cc", edgecolor="none"))
        ax.add_patch(Rectangle((-50, z0 + 6), 100, ACT.thickness - 12, facecolor="#f4e1d2", edgecolor="#b03a2e", lw=0.6, ls=":"))
        ax.text(r + 30, z0 + ACT.thickness / 2, names[i], fontsize=8, va="center")
    # yaw axis, output hub, coxa
    ax.plot([0, 0], [-HIP_DROP - 60, H + 30], color=ACTC, lw=0.7, ls="-.")
    ax.add_patch(Rectangle((-40, -HIP_DROP - 3), 80, HIP_DROP + 6, facecolor="#bbb", edgecolor=INK, lw=0.8))
    ax.add_patch(Rectangle((-28, -HIP_DROP - 17), 200, 34, facecolor=LEGC["coxa"], edgecolor="none"))
    ax.text(180, -HIP_DROP + 30, "coxa (150 mm) → femur axis", fontsize=8)
    ax.annotate("femur and knee drives cross the yaw axis\nconcentrically and run along the coxa\n(transmission chunk M1)", (0, PLATE_T + 4), (-r - 40, -HIP_DROP - 150), fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.6))
    # dimensions
    dim(ax, (-r, H + 20), (r, H + 20), 70, f"Ø{ACT.od:.0f}")
    z1 = PLATE_T + 4
    dim(ax, (-r - 20, z1), (-r - 20, z1 + ACT.thickness), 60, f"{ACT.thickness:.0f}")
    dim(ax, (-r - 20, z1 + ACT.thickness), (-r - 20, z1 + ACT.thickness + ACT.stack_gap), 60, f"{ACT.stack_gap:.0f} gap")
    dim(ax, (-r - 20, z1), (-r - 20, z1 + 3 * ACT.thickness + 2 * ACT.stack_gap), 130, f"stack {3 * ACT.thickness + 2 * ACT.stack_gap:.0f}")
    dim(ax, (-r - 20, 0), (-r - 20, H), 200, f"slab {H:.0f}")
    zt = PLATE_T + 4 + 3 * ACT.thickness + 2 * ACT.stack_gap + 8
    dim(ax, (-85, zt), (-55, zt), 22, "magnet ring 30", fs=7)
    dim(ax, (-50, zt), (50, zt), 22, "Ø100 reducer bore", fs=7)
    frame(ax, "HIP STACK SECTION — three coaxial pancake actuators per hip (mm)", (-r - 300, r + 360), (-HIP_DROP - 210, H + 120))
    fig.tight_layout()
    p = os.path.join(OUT, "hip-stack.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def write_doc(figs):
    c, f, t = CHOSEN.link_lengths()
    doc = f"""# 06 — The geometry, dimensioned

*Generated by `analysis/drawing.py` from the models. All dimensions in mm.*

The current rough geometry from four views. The body is a {L:.0f} × {W:.0f} × {H:.0f} mm slab
with six hip stacks under it on a {BODY.width:.0f} mm spacing, {BODY.hip_x[0] - BODY.hip_x[1]:.0f} mm apart along the
body. Each hip is three coaxial pancake actuators, Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm, on the yaw axis.
Each leg is coxa {c:.0f} / femur {f:.0f} / tibia {t:.0f} mm. The sprawl stance stands with
the femur axis {CHOSEN.hip_height:.0f} mm and the deck {CHOSEN.hip_height + HIP_DROP + H:.0f} mm above ground; the mammal
stance, same robot, {MAMMAL_MODE.hip_height:.0f} mm and {MAMMAL_MODE.hip_height + HIP_DROP + H:.0f} mm.

## Top view

![top]({os.path.relpath(figs['top'], os.path.dirname(DOC))})

## Front view, sprawl stance

![front]({os.path.relpath(figs['front'], os.path.dirname(DOC))})

## Side views, both stances

![side]({os.path.relpath(figs['side'], os.path.dirname(DOC))})

## The hip stack — the volume given to the motors

![hip stack]({os.path.relpath(figs['hip'], os.path.dirname(DOC))})

Each actuator has a Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm envelope including housing, bearings and
reducer. With the reducer in-plane the magnetically active annulus is
r = 55–85 mm (a 30 mm wide ring) and the reducer has a Ø100 mm bore; with
the reducer stacked axially the motor may use the whole disc but only part of
the height. That envelope is the input to the motor study in
[07-motor-options-in-envelope.md](07-motor-options-in-envelope.md).

| Quantity | Value |
|---|---|
| Actuator envelope | Ø{ACT.od:.0f} × {ACT.thickness:.0f} mm, ≤ {ACT.mass:.1f} kg |
| Hip stack | 3 actuators + 2 × {ACT.stack_gap:.0f} mm = {3*ACT.thickness + 2*ACT.stack_gap:.0f} mm tall inside the {H:.0f} mm slab |
| Envelope volume per actuator | {math.pi * (ACT.od/2)**2 * ACT.thickness / 1e6:.2f} L |
| Active annulus (in-plane reducer) | r 55–85 mm, ~30 mm of axial length available to magnetics |
| Yaw axis spacing / hip pitch | {BODY.width:.0f} mm across, {BODY.hip_x[0] - BODY.hip_x[1]:.0f} mm along |
| Coxa / femur / tibia | {c:.0f} / {f:.0f} / {t:.0f} mm |
| Femur axis height, sprawl / mammal | {CHOSEN.hip_height:.0f} / {MAMMAL_MODE.hip_height:.0f} mm |
| Stance width, sprawl / mammal | {BODY.width + 2*CHOSEN.foot_radius:.0f} / {BODY.width + 2*MAMMAL_MODE.neutral_foot[1]:.0f} mm |
"""
    with open(DOC, "w") as fh:
        fh.write(doc)


if __name__ == "__main__":
    figs = {"top": view_top(), "front": view_front(), "side": view_side(), "hip": view_hip_stack()}
    write_doc(figs)
    print("wrote", DOC, "and", list(figs.values()))

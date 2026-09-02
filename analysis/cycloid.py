#!/opt/hw-py/bin/python
"""In-plane cycloidal reducer inside the actuator's Ø100 mm bore: geometry,
loads and a profile drawing for each joint's ratio.

    /opt/hw-py/bin/python analysis/cycloid.py

Writes docs/design/actuator/cycloid-profiles.png and prints a markdown
table used by docs/design/08-actuator-design.md.

Model (standard single-stage cycloid, fixed ring pins, disc on an eccentric,
output through pins in holes):
  N lobes on the disc, N+1 ring pins on a circle of radius R, pin radius r,
  eccentricity e.  Ratio = N:1 with the ring fixed and the disc driving the
  output.  The disc profile is the inner offset (by r) of an epitrochoid.
  Profile has no cusp when  e·(N+1) < R  (design margin: e·(N+1) ≤ 0.8·R).
  Loads at output torque T:
    ring-pin force, peak            F_pin ≈ 4·T / ((N+1)·R)      (half the pins share, sinusoidally)
    output-pin force, peak          F_out ≈ 4·T / (n_out·R_out)  (half of n_out pins loaded)
    eccentric bearing radial load   F_ecc ≈ T / R + T / (N·e)     (pin reaction + input tangential force)
  Hertz line contact between a pin (radius r) and the disc lobe (concave/convex
  radius of curvature ~ 2r..4r taken as r_eq = 1.5 r):  σ_H = sqrt(F·E' / (π·b·r_eq)),
  b = disc thickness, E' = E/(1−ν²)/2 for steel on steel.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "design", "actuator")
os.makedirs(FIG, exist_ok=True)

# ---- the bore and materials ------------------------------------------------
BORE_R = 50.0           # mm, inside the magnet ring (r < 50 mm available)
RING_WALL = 4.0         # mm, housing wall carrying the ring pins
R_PIN_CIRCLE = BORE_R - RING_WALL - 2.5   # mm, ring-pin circle radius ~43.5
DISC_T = 8.0            # mm, each disc (two discs 180° apart)
N_DISCS = 2
E_STEEL = 207e3         # MPa
NU = 0.3
E_PRIME = E_STEEL / (1 - NU**2) / 2
SIGMA_H_ALLOW = 1400.0  # MPa, hardened steel rollers/dowels on hardened disc, grease
R_OUT = 22.0            # mm, output-pin circle radius
N_OUT = 8               # output pins
SAFETY = 1.0            # loads already include the peak case


# ---- per-joint requirement (joint torque / ratio; from 01-sizing.md §6) -----
JOINTS = {
    # name: (ratio N, joint continuous N·m, joint peak N·m)
    "yaw":   (45, 55.0, 58.0),
    "femur": (55, 135.0, 245.0),
    "knee":  (60, 143.0, 300.0),
}


def design(N: int, T_cont: float, T_peak: float, R=R_PIN_CIRCLE):
    """Pick pin radius and eccentricity for N lobes on a pin circle R."""
    pitch = 2 * math.pi * R / (N + 1)            # mm between ring pins
    r = 0.40 * pitch                             # pin radius: pins take ~80 % of the pitch? no — diameter 0.8·pitch is too tight
    r = min(r, 0.35 * pitch)                     # keep ≥ 30 % of the pitch as lobe clearance
    e = 0.8 * R / (N + 1)                        # eccentricity at the no-cusp margin
    F_pin = 4 * T_peak * 1000 / ((N + 1) * R) / N_DISCS       # N, per disc (two discs share)
    F_out = 4 * T_peak * 1000 / (N_OUT * R_OUT) / N_DISCS
    F_ecc = (T_peak * 1000 / R + T_peak * 1000 / (N * e)) / N_DISCS
    sigma = math.sqrt(F_pin * E_PRIME / (math.pi * DISC_T * 1.5 * r))
    F_pin_c = 4 * T_cont * 1000 / ((N + 1) * R) / N_DISCS
    sigma_c = math.sqrt(F_pin_c * E_PRIME / (math.pi * DISC_T * 1.5 * r))
    return {"N": N, "R": R, "pitch": pitch, "r_pin": r, "e": e, "F_pin": F_pin, "F_out": F_out, "F_ecc": F_ecc,
            "sigma_peak": sigma, "sigma_cont": sigma_c, "T_cont": T_cont, "T_peak": T_peak,
            "ok": sigma <= SIGMA_H_ALLOW}


def profile(N: int, R: float, r: float, e: float, n=2000):
    """Disc profile (x, y) in the disc frame: epitrochoid of the pin-circle
    rolling, offset inward by the pin radius.  Standard parametric form."""
    t = np.linspace(0, 2 * math.pi, n)
    # epitrochoid generated with N+1 pins on radius R and eccentricity e
    x0 = R * np.cos(t) - e * np.cos((N + 1) * t)
    y0 = R * np.sin(t) - e * np.sin((N + 1) * t)
    dx = -R * np.sin(t) + e * (N + 1) * np.sin((N + 1) * t)
    dy = R * np.cos(t) - e * (N + 1) * np.cos((N + 1) * t)
    nrm = np.hypot(dx, dy)
    # inward normal offset by the pin radius
    x = x0 + r * dy / nrm
    y = y0 - r * dx / nrm
    return x, y


def fig_profiles(designs):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (name, d) in zip(axes, designs.items()):
        N, R, r, e = d["N"], d["R"], d["r_pin"], d["e"]
        x, y = profile(N, R, r, e)
        x = x - e                                   # the disc sits on the eccentric
        ax.plot(x, y, color="#c0392b", lw=1.2)
        ax.fill(x, y, color="#f7e0da", zorder=0)
        # ring pins
        for k in range(N + 1):
            a = 2 * math.pi * k / (N + 1)
            ax.add_patch(plt.Circle((R * math.cos(a), R * math.sin(a)), r, color="#3a3a3a"))
        # output pins and holes (hole radius = pin radius + e)
        r_o = 0.5 * (2 * math.pi * R_OUT / N_OUT) * 0.5
        for k in range(N_OUT):
            a = 2 * math.pi * k / N_OUT
            ax.add_patch(plt.Circle((R_OUT * math.cos(a) - e, R_OUT * math.sin(a)), r_o + e, fill=False, color="#2980b9", lw=0.8))
            ax.add_patch(plt.Circle((R_OUT * math.cos(a), R_OUT * math.sin(a)), r_o, color="#2980b9"))
        ax.add_patch(plt.Circle((-e, 0), 12, color="#999"))           # eccentric bearing (disc centre)
        ax.plot(0, 0, "k+", ms=10)
        ax.add_patch(plt.Circle((0, 0), BORE_R, fill=False, color="#0f9b8e", ls="--", lw=1))
        ax.set_aspect("equal")
        ax.set_xlim(-BORE_R - 6, BORE_R + 6)
        ax.set_ylim(-BORE_R - 6, BORE_R + 6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}: {N}:1 — {N+1} pins Ø{2*r:.1f} on R{R:.1f}, e = {e:.2f} mm\n"
                     f"peak {d['T_peak']:.0f} N·m: pin {d['F_pin']:.0f} N, Hertz {d['sigma_peak']:.0f} MPa", fontsize=9)
    fig.suptitle("In-plane cycloid inside the Ø100 mm bore (teal dashed): disc (red), ring pins (dark), output pins in oversized holes (blue), eccentric (grey)", fontsize=9.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(FIG, "cycloid-profiles.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def table(designs):
    rows = ["| Joint | Ratio | Ring pins | Pin Ø (mm) | Pitch (mm) | Eccentricity (mm) | Ring-pin force, peak (N, per disc) | Hertz, peak / cont. (MPa) | Output-pin force, peak (N) | Eccentric bearing load, peak (N, per disc) | OK? |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in designs.items():
        rows.append(f"| {name} | {d['N']}:1 | {d['N']+1} | {2*d['r_pin']:.1f} | {d['pitch']:.2f} | {d['e']:.2f} | {d['F_pin']:.0f} | "
                    f"{d['sigma_peak']:.0f} / {d['sigma_cont']:.0f} | {d['F_out']:.0f} | {d['F_ecc']:.0f} | {'yes' if d['ok'] else 'NO'} |")
    return "\n".join(rows)


if __name__ == "__main__":
    designs = {name: design(N, tc, tp) for name, (N, tc, tp) in JOINTS.items()}
    print(fig_profiles(designs))
    print(table(designs))
    print(f"\nassumptions: pin circle R {R_PIN_CIRCLE:.1f} mm in a Ø{2*BORE_R:.0f} bore, {N_DISCS} discs × {DISC_T:.0f} mm, "
          f"output {N_OUT} pins on R{R_OUT:.0f}, allowable Hertz {SIGMA_H_ALLOW:.0f} MPa")

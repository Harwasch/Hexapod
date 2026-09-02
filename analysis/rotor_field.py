#!/opt/hw-py/bin/python
"""Halbach rotor field with real magnet blocks: 2-D superposition at the mean
radius, two arrays facing across the stator gap.

    /opt/hw-py/bin/python analysis/rotor_field.py

Model: each block is a uniformly magnetised rectangle (width w_b along the
circumference, thickness h_m axially); its field is that of magnetic surface
charges sigma = M.n on its faces, discretised into line charges (2-D, the
radial direction is the invariant one).  The array is periodic with period
lambda = 2 tau_p (one pole pair = 4 Halbach segments); 5 periods either side
are summed.  The second rotor is the mirror image across the gap so the
axial fields add in the stator plane.  Output: B_z at the mid-plane, its
fundamental, and the comparison with motor_options.halbach_B.
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
import motor_options as mo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "design", "actuator")
os.makedirs(FIG, exist_ok=True)
AS = json.load(open(os.path.join(ROOT, "hw", "stator", "asbuilt.json")))
GEO = json.load(open(os.path.join(ROOT, "hw", "stator", "geometry.json")))

P = GEO["pole_pairs"]
R1, R2 = GEO["r_in_mm"] * 1e-3, GEO["r_out_mm"] * 1e-3
R_M = 0.5 * (R1 + R2)
BR = mo.BR                      # remanence at the magnet temperature the study uses (N48 hot)
M = BR / mo.MU0                 # A/m
T_BOARD = 2.2e-3
G_MAG = T_BOARD + 2 * mo.AIR_CLEAR
LAMBDA = 2 * math.pi * R_M / P  # one pole pair along the circumference at r_m
SEG = LAMBDA / 4                # 4 Halbach segments per pole pair
NPER = 5                        # periods summed either side

CASES = {
    # name: (block width along the circumference at r_m [m], thickness [m], fill note)
    "rect 30x5x6 N48": (5.0e-3, 6.0e-3, "rectangular OTS block, 5 mm wide: 71 % of the 7.0 mm segment at r_m; 90 % at r_in, 58 % at r_out"),
    "trapezoid full": (SEG * 0.96, 6.0e-3, "custom trapezoidal segment, 0.3 mm glue gaps"),
    "rect 30x5x4 N48": (5.0e-3, 4.0e-3, "thinner OTS block"),
    "rect 30x5x8 N48": (5.0e-3, 8.0e-3, "thicker OTS block (30x8x5 listing)"),
    "rect 30x5x10 N48": (5.0e-3, 10.0e-3, "30x10x5 listing stood on edge: +4 mm each side in the axial stack"),
}


def field_of_block(x, z, xc, zc, w, h, mdir):
    """B (Bx, Bz) at points (x, z) from a rectangle centred (xc, zc), width w
    (x), height h (z), magnetisation direction mdir in {(0,1),(1,0),(0,-1),(-1,0)}.
    Surface charge sigma_m = M (m . n) on faces with normal n; field of a
    line charge lambda_m at r: B = mu0/(2 pi) * lambda_m * r_vec/|r|^2."""
    mx, mz = mdir
    Bx = np.zeros_like(x); Bz = np.zeros_like(z)
    n_pts = 40
    faces = []
    if mz:
        # top and bottom faces carry +/- M
        xs = np.linspace(xc - w / 2, xc + w / 2, n_pts)
        faces.append((xs, np.full_like(xs, zc + h / 2), mz * M * (w / n_pts)))
        faces.append((xs, np.full_like(xs, zc - h / 2), -mz * M * (w / n_pts)))
    if mx:
        zs = np.linspace(zc - h / 2, zc + h / 2, n_pts)
        faces.append((np.full_like(zs, xc + w / 2), zs, mx * M * (h / n_pts)))
        faces.append((np.full_like(zs, xc - w / 2), zs, -mx * M * (h / n_pts)))
    for fx, fz, q in faces:
        for xi, zi in zip(fx, fz):
            dx, dz = x - xi, z - zi
            r2 = dx * dx + dz * dz + 1e-12
            Bx += mo.MU0 / (2 * math.pi) * q * dx / r2
            Bz += mo.MU0 / (2 * math.pi) * q * dz / r2
    return Bx, Bz


def array_field(x, z, w_b, h_m, gap):
    """Two facing Halbach arrays; rotor 1 below the gap, rotor 2 above."""
    Bx = np.zeros_like(x); Bz = np.zeros_like(z)
    pattern = [(0, 1), (-1, 0), (0, -1), (1, 0)]         # rotor 1: +z, -x, -z, +x concentrates the field upward (into the gap); the other sign cancels the fundamental
    for k in range(-NPER * 4, NPER * 4 + 4):
        xc = k * SEG
        mdir = pattern[k % 4]
        # rotor 1 (below): centre at z = -(gap/2 + h/2)
        bx, bz = field_of_block(x, z, xc, -(gap / 2 + h_m / 2), w_b, h_m, mdir)
        Bx += bx; Bz += bz
        # rotor 2 (above): mirror in z -> x-components of M flip sign so B_z adds at the mid-plane
        mdir2 = (-mdir[0], mdir[1])
        bx, bz = field_of_block(x, z, xc, +(gap / 2 + h_m / 2), w_b, h_m, mdir2)
        Bx += bx; Bz += bz
    return Bx, Bz


def fundamental(xs, bz):
    return 2 * np.trapezoid(bz * np.cos(2 * math.pi * xs / LAMBDA), xs) / LAMBDA


results = {}
xs = np.linspace(0, LAMBDA, 240, endpoint=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for name, (w_b, h_m, note) in CASES.items():
    _, bz0 = array_field(xs, np.zeros_like(xs), w_b, h_m, G_MAG)
    b1 = fundamental(xs, bz0)
    # field at the magnet surface of rotor 1 (worst place for the copper eddy loss estimate) and inside a magnet (demag)
    _, bz_surf = array_field(xs, np.full_like(xs, -G_MAG / 2 + 0.2e-3), w_b, h_m, G_MAG)
    model = float(np.mean(mo.halbach_B(np.linspace(R1, R2, 50), P, h_m, G_MAG)))
    # rotor-to-rotor attraction: Maxwell stress on the mid-plane, (Bz^2 - Bx^2)/(2 mu0), over the magnet annulus
    bx0, _ = array_field(xs, np.zeros_like(xs), w_b, h_m, G_MAG)
    sigma_zz = float(np.mean((bz0**2 - bx0**2) / (2 * mo.MU0)))
    area = math.pi * ((R_M + 15e-3)**2 - (R_M - 15e-3)**2)
    results[name] = dict(w_b_mm=w_b * 1e3, h_m_mm=h_m * 1e3, B1_midplane=float(b1), B_peak_midplane=float(np.max(np.abs(bz0))),
                         B_peak_surface=float(np.max(np.abs(bz_surf))), model_halbach_B=model, ratio_to_model=float(b1 / model),
                         magnet_mass_g=2 * 4 * P * (w_b * 30e-3 * h_m) * 7500 * 1e3, note=note,
                         attraction_kPa=sigma_zz * 1e-3, attraction_N=sigma_zz * area)
    axes[0].plot(xs * 1e3, bz0, label=f"{name}: B1 {b1:.2f} T")
axes[0].axhline(0, color="#999", lw=0.6)
axes[0].set_xlabel(f"circumference at r = {R_M*1e3:.0f} mm (mm), one pole pair = {LAMBDA*1e3:.1f} mm")
axes[0].set_ylabel("B_z at the stator mid-plane (T)")
axes[0].set_title(f"Two facing 4-segment Halbach arrays, {P} pole pairs, gap {G_MAG*1e3:.1f} mm", fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8)
# field map for the OTS rectangular case
w_b, h_m, _ = CASES["rect 30x5x6 N48"]
X, Z = np.meshgrid(np.linspace(0, LAMBDA, 120), np.linspace(-(G_MAG / 2 + h_m + 1e-3), G_MAG / 2 + h_m + 1e-3, 90))
Bx, Bz = array_field(X, Z, w_b, h_m, G_MAG)
Bmag = np.hypot(Bx, Bz)
cs = axes[1].contourf(X * 1e3, Z * 1e3, Bmag, levels=np.linspace(0, 1.4, 15), cmap="magma")
axes[1].streamplot(X * 1e3, Z * 1e3, Bx, Bz, color="w", density=0.9, linewidth=0.5, arrowsize=0.6)
for k in range(4):
    for zc in (-(G_MAG / 2 + h_m / 2), G_MAG / 2 + h_m / 2):
        axes[1].add_patch(plt.Rectangle(((k * SEG - w_b / 2) * 1e3, (zc - h_m / 2) * 1e3), w_b * 1e3, h_m * 1e3, fill=False, edgecolor="w", lw=0.8))
axes[1].axhspan(-T_BOARD / 2 * 1e3, T_BOARD / 2 * 1e3, color="#0f9b8e", alpha=0.35)
axes[1].set_xlabel("circumference (mm)")
axes[1].set_ylabel("axial (mm)")
axes[1].set_title("|B| with 30×5×6 mm blocks (white), stator board (teal)", fontsize=10)
fig.colorbar(cs, ax=axes[1], label="T")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "rotor-field.png"), dpi=110)
json.dump(results, open(os.path.join(ROOT, "hw", "stator", "rotor_field.json"), "w"), indent=1)
for k, v in results.items():
    print(f"{k:18s} w {v['w_b_mm']:.1f} h {v['h_m_mm']:.0f}: B1 {v['B1_midplane']:.3f} T (peak {v['B_peak_midplane']:.2f}), surface {v['B_peak_surface']:.2f} T, model {v['model_halbach_B']:.3f} -> ratio {v['ratio_to_model']:.2f}, magnets {v['magnet_mass_g']:.0f} g, rotor attraction {v['attraction_N']/1e3:.2f} kN")

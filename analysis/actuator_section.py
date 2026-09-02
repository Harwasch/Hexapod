#!/opt/hw-py/bin/python
"""Dimensioned cross-section of one pancake actuator: axial-flux motor on the
outer annulus, in-plane cycloid in the bore, bearings, housing.

    /opt/hw-py/bin/python analysis/actuator_section.py [--stator pcb|wound] [--stators 1|2]

Writes docs/design/actuator/section-<variant>.png.  The axial stack-up is
computed from the parameters below and checked against the 42 mm envelope;
the script prints the stack table used in 08-actuator-design.md.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "design", "actuator")
os.makedirs(FIG, exist_ok=True)

# ---- envelope --------------------------------------------------------------
OD = 170.0            # mm
H_ENV = 42.0          # mm, total axial
R_MAG_OUT = 85.0      # magnet ring outer radius (housing wall outside it)
R_MAG_IN = 55.0
R_BORE = 50.0         # cycloid bore
R_PIN = 43.5          # ring-pin circle
R_OUT_PINS = 22.0

# ---- axial stack-up (mm) ---------------------------------------------------
STATORS = {
    # stator thickness, mechanical clearance each side, magnet thickness per rotor
    "pcb":   {"t_stator": 3.2, "gap": 0.6, "t_mag": 5.0, "label": "PCB stator 12-layer 3 oz"},
    "wound": {"t_stator": 5.5, "gap": 0.8, "t_mag": 5.0, "label": "wound flat-coil stator"},
}
T_BACKIRON = 4.0      # steel rotor plate behind each magnet ring (Halbach needs less; kept for stiffness)
T_HOUSING = 2.5       # end plates
T_DISC = 8.0          # each cycloid disc
N_DISC = 2
T_ECC_BRG = 12.0      # eccentric bearing width (e.g. 6008: 40×68×15 → use a thin-section 12)
T_OUT_BRG = 10.0      # output cross-roller / thin-section bearing


def layers_for(stator: str, n_stators: int):
    """Axial layer list from the bottom rotor plate to the top one.  Inner
    rotors of a two-stator stack carry magnets on both faces of one plate."""
    s = STATORS[stator]
    L = [("backiron", T_BACKIRON), ("magnet", s["t_mag"]), ("gap", s["gap"]), ("stator", s["t_stator"]), ("gap", s["gap"]), ("magnet", s["t_mag"])]
    for _ in range(n_stators - 1):
        L += [("backiron", T_BACKIRON), ("magnet", s["t_mag"]), ("gap", s["gap"]), ("stator", s["t_stator"]), ("gap", s["gap"]), ("magnet", s["t_mag"])]
    L += [("backiron", T_BACKIRON)]
    return L


def stack(stator: str, n_stators: int):
    s = STATORS[stator]
    L = layers_for(stator, n_stators)
    motor = sum(t for _, t in L)
    reducer = N_DISC * T_DISC + 2.0
    total = motor + 2 * T_HOUSING + 1.0          # 0.5 mm clearance to each end plate
    return {"motor_axial": motor, "reducer_axial": reducer, "total": total, "n_rotors": n_stators + 1, **s,
            "fits": total <= H_ENV, "reducer_fits_in_motor_height": reducer <= motor, "layers": L}


def draw(stator: str, n_stators: int):
    st = stack(stator, n_stators)
    s = STATORS[stator]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    colors = {"backiron": "#7f8c8d", "magnet": "#c0392b", "gap": "none", "stator": "#0f9b8e"}
    z_bot = T_HOUSING + 0.5
    z_top = z_bot + st["motor_axial"] + 0.5
    total = st["total"]
    # housing: end plates, outer wall, bore wall (ring-pin carrier)
    ax.add_patch(Rectangle((-OD / 2, 0), OD, T_HOUSING, facecolor="#999", edgecolor="k", lw=0.6, hatch="////"))
    ax.add_patch(Rectangle((-OD / 2, total - T_HOUSING), OD, T_HOUSING, facecolor="#999", edgecolor="k", lw=0.6, hatch="////"))
    for side in (1, -1):
        ax.add_patch(Rectangle((side * OD / 2 - (1.5 if side > 0 else 0), 0), 1.5, total, facecolor="#999", edgecolor="k", lw=0.4))
        ax.add_patch(Rectangle((side * R_BORE - (0 if side > 0 else 4), T_HOUSING), 4, total - 2 * T_HOUSING, facecolor="#999", edgecolor="k", lw=0.4, hatch="////"))
    # motor layers on the annulus, both sides of the axis
    z = z_bot
    first = {}
    for name, t in st["layers"]:
        if name != "gap":
            for side in (1, -1):
                if name == "stator":
                    r0, r1 = R_BORE + 4.5, R_MAG_OUT + 0.5
                else:
                    r0, r1 = R_MAG_IN, R_MAG_OUT
                x0 = r0 if side > 0 else -r1
                ax.add_patch(Rectangle((x0, z), r1 - r0, t, facecolor=colors[name], edgecolor="k", lw=0.4))
            if name == "backiron":       # rotor plate web to the hub, inside the magnets
                for side in (1, -1):
                    x0 = R_BORE + 4.5 if side > 0 else -R_MAG_IN
                    ax.add_patch(Rectangle((x0, z + t * 0.3), R_MAG_IN - R_BORE - 4.5, t * 0.4, facecolor="#bdc3c7", edgecolor="none"))
        first.setdefault(name, z + t / 2)
        z += t
    # cycloid in the bore
    zc = (total - st["reducer_axial"]) / 2
    for k in range(N_DISC):
        zk = zc + k * (T_DISC + 2.0)
        off = 0.6 * (1 if k == 0 else -1)
        ax.add_patch(Rectangle((-R_PIN + 3 + off, zk), 2 * (R_PIN - 3), T_DISC, facecolor="#f7e0da", edgecolor="#c0392b", lw=0.8))
        ax.add_patch(Rectangle((-20 + off, zk - 1), 40, T_DISC + 2, facecolor="#e8e8e8", edgecolor="k", lw=0.5))
    for side in (1, -1):
        ax.add_patch(Rectangle((side * R_PIN - 1.7, zc - 3), 3.4, st["reducer_axial"] + 6, facecolor="#3a3a3a", edgecolor="none"))
        ax.add_patch(Rectangle((side * R_OUT_PINS - 1.5, zc - 3), 3.0, st["reducer_axial"] + 6, facecolor="#2980b9", edgecolor="none"))
    ax.add_patch(Rectangle((-6, T_HOUSING), 12, total - 2 * T_HOUSING, facecolor="#555", edgecolor="k", lw=0.4))
    ax.add_patch(Rectangle((-R_OUT_PINS - 4, -4), 2 * R_OUT_PINS + 8, 4, facecolor="#2980b9", edgecolor="k", lw=0.4))
    # labels: right side = motor layers (leader lines), left side = reducer and housing
    right = [(first["backiron"], "rotor plate (back-iron)"), (first["magnet"], "magnet ring, r 55–85"), (first["stator"], s["label"]),
             (first["gap"], f"air gap {s['gap']} mm each side")]
    for k, (zz, t) in enumerate(sorted(right)):
        yt = -2 + k * 7.5 + 6
        ax.annotate(t, (R_MAG_OUT + 1, zz), (OD / 2 + 12, yt), fontsize=7.5, va="center", arrowprops=dict(arrowstyle="-", color="#888", lw=0.5))
    left = [(zc + T_DISC / 2, "cycloid discs (2, 180° apart) on eccentric bearings"), (zc + st["reducer_axial"] + 1, "ring pins in the bore wall"),
            (zc - 1, "output pins → output flange"), (total - T_HOUSING / 2, "housing end plate = heat sink to the hip pod"), (T_HOUSING / 2, "housing / floor side")]
    for k, (zz, t) in enumerate(sorted(left)):
        yt = -2 + k * 7.5 + 6
        ax.annotate(t, (-R_BORE - 4, zz), (-OD / 2 - 12, yt), fontsize=7.5, va="center", ha="right", arrowprops=dict(arrowstyle="-", color="#888", lw=0.5))
    # dimensions
    ax.annotate("", (-OD / 2, total + 6), (OD / 2, total + 6), arrowprops=dict(arrowstyle="<->", color="#b03a2e", lw=0.8))
    ax.text(0, total + 8, f"Ø{OD:.0f}", ha="center", color="#b03a2e", fontsize=8)
    ax.annotate("", (OD / 2 + 4, 0), (OD / 2 + 4, total), arrowprops=dict(arrowstyle="<->", color="#b03a2e", lw=0.8))
    ax.text(OD / 2 + 6, total + 2, f"{total:.1f} of {H_ENV:.0f}", color="#b03a2e", fontsize=8, ha="left")
    ax.annotate("", (-R_BORE, -8), (R_BORE, -8), arrowprops=dict(arrowstyle="<->", color="#b03a2e", lw=0.8))
    ax.text(0, -11, f"Ø{2*R_BORE:.0f} bore", ha="center", color="#b03a2e", fontsize=8)
    ax.annotate("", (R_MAG_IN, -8), (R_MAG_OUT, -8), arrowprops=dict(arrowstyle="<->", color="#b03a2e", lw=0.8))
    ax.text((R_MAG_IN + R_MAG_OUT) / 2, -11, "30", ha="center", color="#b03a2e", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-OD / 2 - 95, OD / 2 + 95)
    ax.set_ylim(-16, max(total, 42) + 14)
    ax.axis("off")
    ax.set_title(f"Actuator section — {n_stators} × {s['label']}, in-plane cycloid; axial stack {total:.1f} mm "
                 f"({'fits' if st['fits'] else 'DOES NOT FIT'} the {H_ENV:.0f} mm envelope)", fontsize=10, loc="left")
    fig.tight_layout()
    p = os.path.join(FIG, f"section-{stator}-{n_stators}.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p, st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stator", default="pcb", choices=list(STATORS))
    ap.add_argument("--stators", type=int, default=1)
    ap.add_argument("--plate", type=float, default=T_BACKIRON, help="rotor plate thickness, mm (3 mm is enough behind a Halbach ring)")
    ap.add_argument("--mag", type=float, default=5.0, help="magnet thickness, mm")
    a = ap.parse_args()
    T_BACKIRON = a.plate
    for v in STATORS.values():
        v["t_mag"] = a.mag
    p, st = draw(a.stator, a.stators)
    print(p)
    print(f"| {a.stators} × {STATORS[a.stator]['label']}, {a.plate:.0f} mm plates, {a.mag:.0f} mm magnets | motor {st['motor_axial']:.1f} mm | reducer {st['reducer_axial']:.1f} mm (inside the motor height: {'yes' if st['reducer_fits_in_motor_height'] else 'no'}) | total {st['total']:.1f} of {H_ENV:.0f} mm | {'fits' if st['fits'] else 'does not fit'} |")

#!/opt/hw-py/bin/python
"""Can a cable drive turn continuously, to drive a wheel or a track?

    /opt/hw-py/bin/python analysis/cable_drive_continuous.py

A capstan joint drive (analysis/capstan.py) terminates the rope at both ends of
a sector.  That termination is what makes it a capstan: no slip ever, at any
torque, and no backlash.  It also caps travel at (working wraps / ratio) output
revolutions, which is exactly why it cannot drive a wheel.

This asks what you get if you give the termination up, and what it costs.  Five
architectures, all sized against the marsupial carrier's drive sprocket:

  A  terminated capstan, as designed            -- how far the wheel turns before it stops
  B  endless spliced loop, round groove         -- friction only: Euler-Eytelwein
  C  endless loop in a wedge (V) groove         -- mu' = mu / sin(beta/2); the elevator sheave
  D  toothed belt                               -- positive and continuous (Gates 8MGT)
  E  two terminated capstans + one-way clutches -- keeps the termination, rectifies the stroke

Method notes, because the first pass got two things wrong:

  * The wrap is NOT solved by pinning the tight side at the rope's allowable.
    A friction drive is specified the other way round: choose the wrap, compute
    the pretension needed to pass the load, then see what that pretension does
    to the shafts.  Low friction shows up as a bearing load, not a broken rope.
  * Rope diameter and drive-pulley diameter are ONE decision, not two.  The
    rope must hold the tight-side tension at the working safety factor, and the
    pulley must still give D/d >= 8.  A groove that raises effective friction
    lowers the tight-side tension, which permits a thinner rope, which permits a
    smaller pulley, which raises the ratio.  Each architecture is therefore
    sized from its own friction, not handed a rope.

Bend-over-sheave life is known to us only as an order of magnitude from a
CAPTCHA-blocked page (docs/reference/manifest.yaml).  It is used only to size
the gap, never extrapolated, and the conclusion is also stated as a ratio
against the walking joint whose rope this project has already accepted.

Writes hw/cable_drive.json, docs/design/platform/cable-drive-architectures.png
and docs/design/platform/cable-drive-tradeoff.png.
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "design", "platform")
G = 9.81

MARS = json.load(open(os.path.join(ROOT, "hw", "marsupial.json")))
CAP = json.load(open(os.path.join(ROOT, "hw", "stator", "capstan.json")))
ROPE5 = CAP["geometry"]["rope"]                          # the leg joint's rope, the datasheet anchor

# ---- what the carrier's drive sprocket has to do -------------------------------------------------
CARRIER, MOTOR = MARS["carrier"], MARS["best_ots_motor"]
M_PAIR = MARS["pair_from_hybrid_study"]["m_total"]       # kg the tracks carry: carrier + scout + battery
V_CARRIER, COT = CARRIER["speed_ms"], CARRIER["cot"]
N_DRIVE = 2                                              # two driven tracks
R_SPROCKET = 0.080                                       # m: Ø160 sprocket for the 150 mm track (assumed)
GRADE_DEG = 30.0                                         # peak duty: the slope it must pull away on

F_level = COT * M_PAIR * G
F_grade = M_PAIR * G * math.sin(math.radians(GRADE_DEG)) + F_level
T_SPR_CONT = F_level * R_SPROCKET / N_DRIVE              # N·m at one sprocket
T_SPR_PEAK = F_grade * R_SPROCKET / N_DRIVE
N_SPR = V_CARRIER / (2 * math.pi * R_SPROCKET)           # rev/s
REV_PER_KM = 1000.0 / (2 * math.pi * R_SPROCKET)
K_MIN_TORQUE = T_SPR_PEAK / MOTOR["T_cont"]              # ratio the slope demands of the market-search motor

# ---- the rope, and the rule that couples it to the pulley ----------------------------------------
MU_AL = 0.085                                            # Dyneema on an aluminium drum -- unconfirmed, see manifest
MU_SWEEP = np.linspace(0.05, 0.25, 60)
BETA_V = math.radians(36.0)
MU_V = MU_AL / math.sin(BETA_V / 2)
D_OVER_D_MIN = CAP["geometry"]["d_over_d_min"]           # 8, Dyneema on a drum
SF_CONT = 5.0                                            # the working-load rule analysis/capstan.py already uses
ROPE_SERIES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)        # mm, D12 sizes
R_DRIVEN = 0.120                                         # m: Ø240 driven pulley is what the hull allows
CENTRE = 0.30
THETA_PLAIN = math.pi                                    # a half wrap is what two pulleys give for free


def mbl_for(d_mm):
    """Minimum spliced strength, scaled d^2 from the 5 mm datasheet row."""
    return ROPE5["break_kN"] * 1e3 * (d_mm / ROPE5["d_mm"]) ** 2


def tensions(F, mu, theta):
    """Tight side, slack side, pretension and shaft load to pass F over wrap theta.

    Euler-Eytelwein the useful way round: at the point of slip T1/T2 = e^(mu.theta),
    so F = T1 - T2 = T1 (1 - e^-mu.theta).
    """
    e = math.exp(-mu * theta)
    T1 = F / (1.0 - e)
    return T1, T1 * e, 0.5 * T1 * (1 + e), T1 * (1 + e)


def loop_length(r_a, r_b, centre):
    alpha = math.asin((r_b - r_a) / centre)
    return (2 * math.sqrt(centre ** 2 - (r_b - r_a) ** 2)
            + r_a * (math.pi - 2 * alpha) + r_b * (math.pi + 2 * alpha))


def bends_per_km(r_drive, r_driven, centre, v=V_CARRIER, r_sprocket=R_SPROCKET, n_pulleys=2):
    """Bend cycles at one element of an endless loop, per km driven.

    Each element makes a circuit in L/v_cable seconds and is bent once at each
    pulley on the way round.  Pure geometry: no fatigue curve enters this.
    """
    v_cable = (v / r_sprocket) * r_driven
    return n_pulleys * (v_cable / loop_length(r_drive, r_driven, centre)) * (1000.0 / v)


def size_loop(mu, theta=THETA_PLAIN, r_driven=R_DRIVEN):
    """Size rope and drive pulley together for a friction loop at this friction.

    The tight side sets the smallest rope that holds SF; D/d >= 8 then sets the
    smallest drive pulley that rope may run on; the ratio is what is left.
    """
    F = T_SPR_PEAK / r_driven
    T1, T2, T0, shaft = tensions(F, mu, theta)
    d = next((x for x in ROPE_SERIES if mbl_for(x) / SF_CONT >= T1), None)
    if d is None:
        return None
    r_drive = D_OVER_D_MIN * d * 1e-3 / 2
    return dict(F_tangential=F, T_tight_N=T1, T_slack_N=T2, pretension_N=T0, shaft_load_N=shaft,
                rope_d_mm=d, mbl_N=mbl_for(d), sf=mbl_for(d) / T1,
                r_drive_m=r_drive, ratio=r_driven / r_drive, D_over_d=2 * r_drive * 1e3 / d,
                bends_per_km=bends_per_km(r_drive, r_driven, CENTRE),
                loop_m=loop_length(r_drive, r_driven, CENTRE))


ROUND, VEE = size_loop(MU_AL), size_loop(MU_V)

# ---- the five architectures ----------------------------------------------------------------------
arch = {}
n_wrap = CAP["geometry"]["n_wrap"]
out_rev_A = n_wrap / VEE["ratio"]
arch["A_terminated"] = dict(
    name="Terminated capstan, as designed", continuous=False,
    output_rev=out_rev_A, travel_m=out_rev_A * 2 * math.pi * R_SPROCKET,
    note=(f"{n_wrap:.0f} working wraps at {VEE['ratio']:.0f}:1 is {out_rev_A:.2f} output revolutions: the "
          f"wheel turns {out_rev_A*360:.0f}° and stops. Termination is the whole problem."))

arch["B_round_loop"] = dict(
    name="Endless spliced loop, round groove", continuous=True, wrap_rad=THETA_PLAIN, **ROUND,
    note=("Works on a half wrap, but Dyneema's low friction on metal means the shafts carry a large "
          "pretension whenever the machine is switched on, load or no load, and the thick rope that "
          "tension needs forces a bigger drive pulley, which spends the ratio."))

arch["C_vee_loop"] = dict(
    name="Endless loop in a 36° wedge groove", continuous=True, wrap_rad=THETA_PLAIN,
    mu_effective=MU_V, **VEE,
    creep_heat_W=0.01 * T_SPR_PEAK * 2 * math.pi * N_SPR, rope_T_critical_C=ROPE5["T_critical_C"],
    note=("The wedge multiplies effective friction by 1/sin(beta/2). That cuts the shaft load "
          f"{ROUND['shaft_load_N']/VEE['shaft_load_N']:.1f}x AND lets a thinner rope carry it, which lets a "
          f"smaller pulley drive it, which raises the ratio from {ROUND['ratio']:.0f}:1 to {VEE['ratio']:.0f}:1. "
          "This is the elevator traction sheave, which has done exactly this job for a century."))

PD_MIN_8MGT, GROOVES_MAX = 0.0560, 144.0                 # Gates 8MGT: smallest stock P22, largest stock P144
K_BELT_MAX = GROOVES_MAX / 22.0
arch["D_toothed_belt"] = dict(
    name="Toothed belt (Gates 8MGT)", continuous=True,
    r_drive_m=PD_MIN_8MGT / 2, ratio=K_BELT_MAX,
    stages_for_target=math.ceil(math.log(VEE["ratio"]) / math.log(K_BELT_MAX)),
    bends_per_km=bends_per_km(PD_MIN_8MGT / 2, PD_MIN_8MGT / 2 * K_BELT_MAX, CENTRE),
    note=("Positive, no slip, no creep, and its tension member is cord: a toothed belt IS a cable drive "
          "that solved termination by moulding teeth on it and embedding the cord in elastomer, which "
          f"is also what protects it. Capped at {K_BELT_MAX:.1f}:1 per stage by the smallest stock sprocket."))

arch["E_clutch_pair"] = dict(
    name="Two terminated capstans + one-way clutches", continuous=True,
    stroke_output_rev=out_rev_A, strokes_per_km=REV_PER_KM / out_rev_A,
    note=("Keeps the termination, so it keeps no-slip and zero backlash, and the output turns forever. "
          "But every stroke is a full load reversal in the rope, the output torque pulses at the stroke "
          "rate, and reversing needs switchable clutches. Most parts, most failure modes."))

# ---- the comparison the whole thing turns on ------------------------------------------------------
GAIT_STRIDE, GAIT_DUTY = 0.40, 0.5                        # hexapod_model.GAITS, walk / tripod
gait_cycles_per_km = 1000.0 / (GAIT_STRIDE / GAIT_DUTY)
JOINT_BENDS_PER_KM = 2 * gait_cycles_per_km
JOINT = dict(gait_cycles_per_km=gait_cycles_per_km, bends_per_km=JOINT_BENDS_PER_KM,
             D_over_d=CAP["joints"]["knee"]["D_over_d"],
             note="analysis/capstan.py's rope in the femur/knee joint, walking at 1 m/s on a 0.4 m stride")

LIFE_KM_CARRIER = 2000.0                                  # a season of transit
LIFE_KM_SCOUT = MARS["day"]["leash_m"] * 2 * MARS["day"]["sorties"] / 1e3 * 60
BOS_BAND = (1e3, 1e4)                                     # the band the blocked page reports, bare braid near WLL
OPTIMISM = 100.0                                          # if a lightly loaded thin braid does far better
life = dict(
    life_km_carrier=LIFE_KM_CARRIER, life_km_scout_legs=LIFE_KM_SCOUT,
    known_band=list(BOS_BAND), optimism_factor=OPTIMISM,
    km_per_rope_carrier=[BOS_BAND[0] / VEE["bends_per_km"], BOS_BAND[1] / VEE["bends_per_km"]],
    km_per_rope_carrier_optimistic=[BOS_BAND[0] * OPTIMISM / VEE["bends_per_km"],
                                    BOS_BAND[1] * OPTIMISM / VEE["bends_per_km"]],
    km_per_rope_joint=[BOS_BAND[0] / JOINT_BENDS_PER_KM, BOS_BAND[1] / JOINT_BENDS_PER_KM],
    caveat=("The 10^3-10^4 band is order-of-magnitude, from a source this VM could not open, measured on "
            "large offshore ropes near their working load. It sizes the gap; it does not close it. Even "
            "100x optimistic for a lightly loaded 1.5 mm braid, the rope is a scheduled replacement item. "
            "This number has to be measured on the actual rope, groove and load before anything is built."))

# ---- levers not left unswept ---------------------------------------------------------------------
sweep = [dict(r_drive_mm=r * 1e3, ratio=R_DRIVEN / r, D_over_d=2 * r * 1e3 / VEE["rope_d_mm"],
              bends_per_km=bends_per_km(r, R_DRIVEN, CENTRE))
         for r in np.linspace(0.005, 0.060, 100)]

OUT = dict(
    carrier=dict(m_pair_kg=M_PAIR, v_ms=V_CARRIER, cot=COT, r_sprocket_m=R_SPROCKET, grade_deg=GRADE_DEG,
                 n_drive=N_DRIVE, T_sprocket_cont=T_SPR_CONT, T_sprocket_peak=T_SPR_PEAK,
                 sprocket_rev_per_s=N_SPR, sprocket_rev_per_km=REV_PER_KM, k_min_for_grade=K_MIN_TORQUE,
                 motor=MOTOR["name"], motor_T_cont=MOTOR["T_cont"], motor_n_noload=MOTOR["n_noload"],
                 motor_rpm_at_vee_ratio=N_SPR * 60 * VEE["ratio"]),
    rope=dict(leg_rope_d_mm=ROPE5["d_mm"], mu_aluminium=MU_AL, mu_vee=MU_V, sf=SF_CONT,
              d_over_d_min=D_OVER_D_MIN,
              mu_note="unconfirmed; swept 0.05-0.25 in the figure. See docs/reference/manifest.yaml"),
    geometry=dict(r_driven_m=R_DRIVEN, centre_m=CENTRE, theta_plain_rad=THETA_PLAIN),
    architectures=arch, walking_joint=JOINT, life=life, sweep=sweep)
json.dump(OUT, open(os.path.join(ROOT, "hw", "cable_drive.json"), "w"), indent=1)


def save(fig, name, dpi=110):
    """Save, then palette-quantise: these are flat line figures, so 256 colours is
    visually identical and roughly halves the file the review page has to embed."""
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=dpi)
    from PIL import Image
    Image.open(path).convert("RGB").quantize(colors=256, method=Image.MEDIANCUT).save(path, optimize=True)


TEAL, ORANGE, RED, GREEN, GREY = "#0f9b8e", "#d98c3a", "#c0392b", "#6c8e3a", "#8a8a85"

# ================================ figure 1: the five architectures =================================
fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6))
ra, rb, c = 0.030, 0.115, 0.26


def frame(ax, title, sub):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-ra - 0.04, c + rb + 0.04); ax.set_ylim(-rb - 0.095, rb + 0.05)
    ax.set_title(title, fontsize=10, pad=6)
    ax.text(c / 2, -rb - 0.02, sub, ha="center", va="top", fontsize=8, color="#52514e")


ax = axes[0][0]
ax.add_patch(plt.Circle((0, 0), ra, fill=False, lw=2.2, color=TEAL))
ax.add_patch(plt.Circle((c, 0), rb, fill=False, lw=2.2, color=GREEN))
for s in (1, -1):
    ax.plot([0, c], [s * ra, s * rb], color=GREY, lw=1.6)
    t = np.linspace(0, s * 2.0, 30)
    ax.plot(c + rb * np.cos(t), rb * np.sin(t), color=RED, lw=2.6)
    ax.plot([c + rb * math.cos(s * 2.0)], [rb * math.sin(s * 2.0)], marker="s", ms=7, color=RED)
ax.text(c, 0, "anchored\nboth ends", ha="center", va="center", fontsize=8, color=RED)
frame(ax, "A · Terminated capstan — what we have",
      f"no slip, no backlash, and no continuous rotation:\nthe wheel turns {out_rev_A*360:.0f}° "
      f"({arch['A_terminated']['travel_m']*1e3:.0f} mm of ground) and stops")

ax = axes[0][1]
th = np.linspace(math.pi / 2, 3 * math.pi / 2, 60); th2 = np.linspace(-math.pi / 2, math.pi / 2, 60)
ax.add_patch(plt.Circle((0, 0), ra, fill=False, lw=2.2, color=ORANGE))
ax.add_patch(plt.Circle((c, 0), rb, fill=False, lw=2.2, color=ORANGE))
ax.plot(ra * np.cos(th), ra * np.sin(th), color=ORANGE, lw=2.6)
ax.plot(c + rb * np.cos(th2), rb * np.sin(th2), color=ORANGE, lw=2.6)
for s in (1, -1):
    ax.plot([0, c], [s * ra, s * rb], color=ORANGE, lw=2.6)
ax.annotate("splice — the loop is endless", (c / 2, ra + 0.006), (c / 2 - 0.015, ra + 0.040),
            fontsize=8, color=ORANGE, ha="center", arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
frame(ax, "B · Endless spliced loop, round groove",
      f"{ROUND['shaft_load_N']:.0f} N of pretension on the shafts, always,\nand a {ROUND['rope_d_mm']:.1f} mm "
      f"rope needs a Ø{2*ROUND['r_drive_m']*1e3:.0f} pulley → only {ROUND['ratio']:.0f}:1")

ax = axes[0][2]
ax.add_patch(plt.Circle((0, 0), ra, fill=False, lw=2.2, color=GREEN))
ax.add_patch(plt.Circle((c, 0), rb, fill=False, lw=2.2, color=GREEN))
ax.plot(ra * np.cos(th), ra * np.sin(th), color=GREEN, lw=3.2)
ax.plot(c + rb * np.cos(th2), rb * np.sin(th2), color=GREEN, lw=3.2)
for s in (1, -1):
    ax.plot([0, c], [s * ra, s * rb], color=GREEN, lw=3.2)
axin = ax.inset_axes([0.01, 0.17, 0.17, 0.21]); axin.axis("off")
axin.plot([-1, 0, 1], [1.4, 0, 1.4], color=GREY, lw=2)
axin.add_patch(plt.Circle((0, 0.72), 0.42, color=GREEN))
axin.set_xlim(-1.6, 1.6); axin.set_ylim(-0.35, 1.9); axin.set_aspect("equal")
axin.text(0, -0.3, "β = 36°", ha="center", fontsize=7.5, color="#52514e")
frame(ax, "C · Endless loop in a wedge groove — the best rope answer",
      f"μ' = {MU_V:.2f} → shaft load only {VEE['shaft_load_N']:.0f} N, and a {VEE['rope_d_mm']:.1f} mm rope\n"
      f"on a Ø{2*VEE['r_drive_m']*1e3:.0f} pulley gives {VEE['ratio']:.0f}:1 in ONE stage.\n"
      "This is the elevator traction sheave.")

ax = axes[1][0]
ra_b, rb_b = ra * 1.45, rb * 0.82          # drawn at the panel's scale; the real sizes are in the caption
t3 = np.linspace(math.pi / 2, 3 * math.pi / 2, 40); t4 = np.linspace(-math.pi / 2, math.pi / 2, 40)
ax.add_patch(plt.Circle((0, 0), ra_b, fill=False, lw=2.2, color=TEAL))
ax.add_patch(plt.Circle((c, 0), rb_b, fill=False, lw=2.2, color=TEAL))
ax.plot(ra_b * np.cos(t3), ra_b * np.sin(t3), color=TEAL, lw=3)
ax.plot(c + rb_b * np.cos(t4), rb_b * np.sin(t4), color=TEAL, lw=3)
for s in (1, -1):
    ax.plot([0, c], [s * ra_b, s * rb_b], color=TEAL, lw=3)
for t in np.linspace(0.03, c - 0.01, 13):
    ax.plot([t, t], [ra_b + 0.004 + (rb_b - ra_b) * t / c, ra_b + 0.011 + (rb_b - ra_b) * t / c],
            color=TEAL, lw=1.3)
frame(ax, "D · Toothed belt (Gates 8MGT)",
      f"positive, no pretension, cord protected in elastomer —\nbut the smallest stock sprocket caps a "
      f"stage at {K_BELT_MAX:.1f}:1,\nso {VEE['ratio']:.0f}:1 takes {arch['D_toothed_belt']['stages_for_target']} stages")

ax = axes[1][1]
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.07, 0.35); ax.set_ylim(-0.21, 0.15)
for y in (0.075, -0.075):
    ax.add_patch(plt.Circle((0, y), 0.028, fill=False, lw=2.2, color=ORANGE))
    for s in (1, -1):
        ax.plot([0, 0.175], [y + s * 0.028, y + s * 0.028], color=GREY, lw=1.6)
    ax.text(0, y, "↕", ha="center", va="center", fontsize=11, color=ORANGE)
ax.add_patch(plt.Circle((0.235, 0), 0.062, fill=False, lw=2.2, color=GREEN))
ax.text(0.235, 0, "one-way\nclutches", ha="center", va="center", fontsize=7.5, color=GREEN)
ax.set_title("E · Two terminated capstans + one-way clutches", fontsize=10, pad=6)
ax.text(0.14, -0.115, f"keeps no-slip and zero backlash, turns forever —\nbut "
        f"{arch['E_clutch_pair']['strokes_per_km']:,.0f} full rope reversals per km,\npulsing torque, and "
        "switchable clutches to reverse", ha="center", va="top", fontsize=8, color="#52514e")

ax = axes[1][2]
ax.plot(MU_SWEEP, [tensions(T_SPR_PEAK / R_DRIVEN, m, THETA_PLAIN)[3] for m in MU_SWEEP],
        color=ORANGE, lw=2, label="round groove, half wrap")
ax.plot(MU_SWEEP, [tensions(T_SPR_PEAK / R_DRIVEN, m / math.sin(BETA_V / 2), THETA_PLAIN)[3] for m in MU_SWEEP],
        color=GREEN, lw=2, label="36° wedge, half wrap")
ax.plot(MU_SWEEP, [tensions(T_SPR_PEAK / R_DRIVEN, m, 2 * math.pi)[3] for m in MU_SWEEP],
        color=ORANGE, lw=1.4, ls="--", label="round groove, full wrap")
ax.axvline(MU_AL, color=RED, ls=":", lw=1.2)
ax.text(MU_AL + 0.005, 2500, f"Dyneema on aluminium,\nμ ≈ {MU_AL:.3f} (unconfirmed —\nswept, so nothing "
        "rests on it)", fontsize=7.5, color=RED)
ax.set_xlabel("coefficient of friction, rope on drum")
ax.set_ylabel("shaft load from pretension, N")
ax.set_title("Low friction is paid for in bearings, not broken rope", fontsize=10)
ax.legend(fontsize=7.5); ax.grid(alpha=0.3); ax.set_ylim(0, 3200)
fig.suptitle(f"Five ways to make a cable drive turn continuously — sized on the carrier's drive sprocket: "
             f"{T_SPR_CONT:.0f} N·m level, {T_SPR_PEAK:.0f} N·m on a {GRADE_DEG:.0f}° slope, "
             f"{N_SPR*60:.0f} rpm at {V_CARRIER} m/s", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.955))
save(fig, "cable-drive-architectures.png")

# ================================ figure 2: what it costs =========================================
fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))

ax = axes[0]
rr = [s["r_drive_mm"] for s in sweep]
ax.plot(rr, [s["ratio"] for s in sweep], color=TEAL, lw=2)
ax.set_xlabel("drive-pulley radius, mm")
ax.set_ylabel("single-stage ratio, on a Ø240 driven pulley", color=TEAL)
ax.tick_params(axis="y", labelcolor=TEAL); ax.grid(alpha=0.3); ax.set_ylim(0, 26)
ax.axhline(K_BELT_MAX, color=GREY, ls="--", lw=1.2)
ax.text(rr[-1], K_BELT_MAX + 0.5, f"one toothed-belt stage stops here ({K_BELT_MAX:.1f}:1)",
        fontsize=7.5, color=GREY, ha="right")
for res, col, lab in ((VEE, GREEN, "wedge"), (ROUND, ORANGE, "round")):
    ax.plot([res["r_drive_m"] * 1e3], [res["ratio"]], marker="o", ms=8, color=col)
    ax.annotate(f"{lab} groove\n{res['rope_d_mm']:.1f} mm rope → Ø{2*res['r_drive_m']*1e3:.0f}\n"
                f"{res['ratio']:.0f}:1", (res["r_drive_m"] * 1e3, res["ratio"]),
                (res["r_drive_m"] * 1e3 + 3, res["ratio"] + 1.5), fontsize=8, color=col)
ax.set_title("The rope's advantage is the small pulley — and\nthe groove decides how small it may be",
             fontsize=9.5)

ax = axes[1]
names = ["C wedge loop\n(carrier)", "D toothed belt\n(carrier)", "capstan joint\n(walking leg)"]
vals = [VEE["bends_per_km"], arch["D_toothed_belt"]["bends_per_km"], JOINT_BENDS_PER_KM]
b = ax.bar(names, vals, color=[GREEN, TEAL, GREY])
ax.bar_label(b, fmt="%.0f", fontsize=9)
ax.set_ylabel("bend cycles at one element, per km travelled")
ax.set_ylim(0, max(vals) * 1.35)
ax.set_title("Per km, the drivetrain is not the outlier", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.text(0.5, 0.97, "The leg rope this project already accepted is cycled\njust as hard. What differs is how "
        "far each machine must go:\ncarrier ~2000 km a season, scout legs ~700 km.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#52514e")

ax = axes[2]
ax.axis("off")
rows = [
    ("WHAT THE WEDGE-GROOVE ROPE LOOP BUYS", ""),
    ("ratio in one stage", f"{VEE['ratio']:.0f}:1"),
    ("stages a toothed belt needs for that", f"{arch['D_toothed_belt']['stages_for_target']}"),
    ("rope this load actually needs", f"{VEE['rope_d_mm']:.1f} mm"),
    ("vs the leg joint's rope", f"{(ROPE5['d_mm']/VEE['rope_d_mm'])**2:.0f}x lighter"),
    ("shaft pretension, round → wedge", f"{ROUND['shaft_load_N']:.0f} → {VEE['shaft_load_N']:.0f} N"),
    ("", ""),
    ("WHAT IT COSTS — THE ROPE IS A WEAR ITEM", ""),
    ("km per rope if the blocked band transfers", f"{life['km_per_rope_carrier'][0]:.1f} – "
                                                  f"{life['km_per_rope_carrier'][1]:.0f}"),
    (f"km per rope if it is {OPTIMISM:.0f}x optimistic", f"{life['km_per_rope_carrier_optimistic'][0]:.0f} – "
                                                         f"{life['km_per_rope_carrier_optimistic'][1]:,.0f}"),
    ("carrier transit life wanted", f"{LIFE_KM_CARRIER:,.0f} km"),
    ("creep heat in the rope at peak", f"{arch['C_vee_loop']['creep_heat_W']:.0f} W"),
    ("rope's critical temperature", f"{ROPE5['T_critical_C']:.0f} °C"),
]
y = 0.99
for k, v in rows:
    if not k:
        y -= 0.03; continue
    head = not v
    ax.text(0.0, y, k, fontsize=8.4 if not head else 8.8, va="top",
            color="#0b0b0b" if head else "#52514e", weight="bold" if head else "normal")
    if v:
        ax.text(1.0, y, v, fontsize=8.6, color="#0b0b0b", ha="right", va="top", family="monospace")
    y -= 0.070
ax.text(0.0, y - 0.01, "Rope life is order-of-magnitude, from a page this VM could\nnot open, measured on "
        "large offshore ropes near their working\nload. It sizes the gap; it does not close it. Measure it\n"
        "on the real rope, groove and load before building anything.", fontsize=7.6, color=RED, va="top")
ax.set_title("The honest ledger", fontsize=10)
fig.suptitle("A cable drive can turn continuously, and for the carrier it is not even hard. "
             "The rope becomes a wear item — the question is whether that is acceptable.", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "cable-drive-tradeoff.png")

if __name__ == "__main__":
    print(f"carrier {M_PAIR:.0f} kg at {V_CARRIER} m/s, Ø{2*R_SPROCKET*1e3:.0f} sprocket:")
    print(f"  {T_SPR_CONT:.1f} N·m level, {T_SPR_PEAK:.1f} N·m on {GRADE_DEG:.0f}°, {N_SPR*60:.0f} rpm")
    print(f"  {MOTOR['name']}: {K_MIN_TORQUE:.1f}:1 minimum for the slope")
    for tag, r in (("round", ROUND), ("wedge", VEE)):
        print(f"  {tag}: tight {r['T_tight_N']:.0f} N, shaft {r['shaft_load_N']:.0f} N, rope "
              f"{r['rope_d_mm']:.1f} mm (SF {r['sf']:.1f}), pulley Ø{2*r['r_drive_m']*1e3:.0f} "
              f"(D/d {r['D_over_d']:.1f}) -> {r['ratio']:.0f}:1, {r['bends_per_km']:.0f} bends/km")
    print(f"  belt: {K_BELT_MAX:.1f}:1 per stage, {arch['D_toothed_belt']['stages_for_target']} stages, "
          f"{arch['D_toothed_belt']['bends_per_km']:.0f} bends/km")
    print(f"  clutch pair: {arch['E_clutch_pair']['strokes_per_km']:,.0f} rope reversals/km")
    print(f"walking joint baseline: {JOINT_BENDS_PER_KM:.0f} bends/km at D/d {JOINT['D_over_d']:.0f}")
    print(f"rope life {BOS_BAND[0]:.0f}-{BOS_BAND[1]:.0f} cycles -> carrier "
          f"{life['km_per_rope_carrier'][0]:.1f}-{life['km_per_rope_carrier'][1]:.0f} km/rope "
          f"({life['km_per_rope_carrier_optimistic'][0]:.0f}-{life['km_per_rope_carrier_optimistic'][1]:.0f} "
          f"at {OPTIMISM:.0f}x), against {LIFE_KM_CARRIER:.0f} km wanted")

#!/opt/hw-py/bin/python
"""Three questions from the round-13 review, answered with one 2-D magnetostatic
finite-difference model at the mean coil radius:

  1. Single rotor between two stators, in the same Ø190 package: how does it
     compare with the two-rotor single-stator optimum (1s-opt)?
  2. Does the magnet material help?
  3. Does iron in the centre of the PCB coils help?

Model: scalar potential, div(mu grad phi) = div(M) on a grid periodic over one
pole pair at r_m, far-field Dirichlet in z; iron as mu_r = 1000 with a soft
saturation knee (Picard iterations), magnets as mu_r = 1 with M = Br/mu0.  The
torque figure is the coil's flux-linkage derivative dlambda/dx for the actual
concentrated spiral (N(x) ramps across the legs), which is the same as the
per-leg Lorentz sum for an air-core coil and the only right way to count a
tooth.  Two cross-checks against rotor_field.py are printed.  Writes
hw/stator/topology_compare.json and docs/design/actuator/topology-compare.png.

    /opt/hw-py/bin/python analysis/topology_compare.py
"""
import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor_options as mo   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO = json.load(open(os.path.join(ROOT, "hw", "stator", "variants", "1s-opt", "geometry.json")))
AB = json.load(open(os.path.join(ROOT, "hw", "stator", "variants", "1s-opt", "asbuilt.json")))
RF = json.load(open(os.path.join(ROOT, "hw", "stator", "rotor_field.json")))

P = GEO["pole_pairs"]; N_COILS = GEO["coils"]
R1, R2 = GEO["r_in_mm"] * 1e-3, GEO["r_out_mm"] * 1e-3
R_M = 0.5 * (R1 + R2); L_RAD = R2 - R1
LAM = 2 * math.pi * R_M / P                      # one pole pair at r_m
COIL_PITCH = 2 * math.pi * R_M / N_COILS
COIL_W = COIL_PITCH - GEO["gap_mm"] * 1e-3       # copper envelope of one coil at r_m
T_BOARD = GEO["t_board_mm"] * 1e-3
CLR = mo.AIR_CLEAR
BR = mo.BR                                        # N48 at the hot magnet temperature, as every rating uses
W_B = 5.0e-3                                      # the 30x5xh block, 5 mm along the circumference
H_M = 10.0e-3
H = 0.2e-3                                        # grid pitch
IRON_MU = 1000.0
B_KNEE, B_SAT = 1.3, 1.9                          # laminated steel: mu_r falls past the knee
MAG_RHO = 7.5e3; STEEL_RHO = 7.85e3
AREA_MAG = math.pi * ((R_M + 15e-3) ** 2 - (R_M - 15e-3) ** 2)
PRICE_BLOCK_10 = 0.25 * 10 / 6                    # $ per 30x5x10 block, volume-scaled from the $0.25 30x5x6 estimate
BOARD_PRICE = {20: 50.0, 100: 37.0}               # the 1s-opt 20L 2 oz board (round 13, scaled from the reviewer's JLCPCB quote)
T_CONT_1S = AB["T_cont"] if "T_cont" in AB else 2.81

NDOM = 5                                          # pole pairs in the domain: 5 pole pairs = 12 coils, the true period of the 12/10 winding
LAM_DOM = NDOM * LAM
nx = int(round(LAM_DOM / H))
dx = LAM_DOM / nx


def factor(mu_r, nz):
    """Assemble and LU-factorise div(mu grad) with periodic x and phi = 0 beyond the z ends."""
    N = nx * nz
    idx = lambda i, j: (i % nx) * nz + j
    mu_xf = 2 * mu_r * np.roll(mu_r, -1, axis=0) / (mu_r + np.roll(mu_r, -1, axis=0))
    mu_zf = np.ones_like(mu_r); mu_zf[:, :-1] = 2 * mu_r[:, :-1] * mu_r[:, 1:] / (mu_r[:, :-1] + mu_r[:, 1:])
    I, J = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    rows, cols, vals = [], [], []
    diag = np.zeros((nx, nz))
    for shift, muf in ((1, mu_xf), (-1, np.roll(mu_xf, 1, axis=0))):
        rows.append(idx(I, J).ravel()); cols.append(idx(I + shift, J).ravel()); vals.append((muf / dx ** 2).ravel())
        diag -= muf / dx ** 2
    up = np.zeros((nx, nz)); up[:, :-1] = mu_zf[:, :-1] / H ** 2
    dn = np.zeros((nx, nz)); dn[:, 1:] = mu_zf[:, :-1] / H ** 2
    rows.append(idx(I, J)[:, :-1].ravel()); cols.append(idx(I, J + 1)[:, :-1].ravel()); vals.append(up[:, :-1].ravel())
    rows.append(idx(I, J)[:, 1:].ravel()); cols.append(idx(I, J - 1)[:, 1:].ravel()); vals.append(dn[:, 1:].ravel())
    diag -= up; diag -= dn
    diag[:, 0] -= mu_r[:, 0] / H ** 2; diag[:, -1] -= mu_r[:, -1] / H ** 2       # Dirichlet ghosts
    rows.append(idx(I, J).ravel()); cols.append(idx(I, J).ravel()); vals.append(diag.ravel())
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    return spla.splu(A)


def apply(lu, mu_r, Mx, Mz, nz):
    divM = (np.roll(Mx, -1, axis=0) - np.roll(Mx, 1, axis=0)) / (2 * dx)
    divM[:, 1:-1] += (Mz[:, 2:] - Mz[:, :-2]) / (2 * H)
    phi = lu.solve(divM.ravel()).reshape(nx, nz)
    # B on cell faces from the face permeability (normal B is continuous there), then averaged to the cell
    mu_xf = 2 * mu_r * np.roll(mu_r, -1, axis=0) / (mu_r + np.roll(mu_r, -1, axis=0))
    Bx_f = mo.MU0 * (-mu_xf * (np.roll(phi, -1, axis=0) - phi) / dx + 0.5 * (Mx + np.roll(Mx, -1, axis=0)))
    Bx = 0.5 * (Bx_f + np.roll(Bx_f, 1, axis=0))
    mu_zf = 2 * mu_r[:, :-1] * mu_r[:, 1:] / (mu_r[:, :-1] + mu_r[:, 1:])
    Bz_f = mo.MU0 * (-mu_zf * (phi[:, 1:] - phi[:, :-1]) / H + 0.5 * (Mz[:, :-1] + Mz[:, 1:]))
    Bz = np.zeros_like(phi); Bz[:, 1:-1] = 0.5 * (Bz_f[:, :-1] + Bz_f[:, 1:]); Bz[:, 0] = Bz_f[:, 0]; Bz[:, -1] = Bz_f[:, -1]
    return Bx, Bz


def solve(mu_r, Mx, Mz, z0, nz, iters=1):
    """Picard iterations on the iron permeability (saturation knee), then the field."""
    mu_r = mu_r.copy()
    for it in range(max(1, iters)):
        lu = factor(mu_r, nz)
        Bx, Bz = apply(lu, mu_r, Mx, Mz, nz)
        if iters > 1:
            Bm = np.hypot(Bx, Bz)
            iron = mu_r > 1.5
            # laminated steel: mu_r 1000 below the knee, falling so that B cannot run much past B_SAT
            mu_new = np.where(Bm < B_KNEE, IRON_MU, np.maximum(1.0, IRON_MU * np.exp(-(Bm - B_KNEE) / (0.25 * (B_SAT - B_KNEE)))))
            mu_r = np.where(iron, np.exp(0.5 * np.log(mu_r) + 0.5 * np.log(np.maximum(mu_new, 1.0))), mu_r)   # log-space relaxation
    return Bx, Bz, lu, mu_r


class Grid:
    def __init__(self, z_lo, z_hi):
        self.z0 = z_lo; self.nz = int(round((z_hi - z_lo) / H))
        self.mu = np.ones((nx, self.nz)); self.Mx = np.zeros((nx, self.nz)); self.Mz = np.zeros((nx, self.nz))
        self.x = (np.arange(nx) + 0.5) * dx; self.z = z_lo + (np.arange(self.nz) + 0.5) * H

    def block(self, xc, zc, w, h, mdir, shift=0.0):
        xs = ((self.x - xc - shift + LAM_DOM / 2) % LAM_DOM) - LAM_DOM / 2
        m = (np.abs(xs)[:, None] <= w / 2) & (np.abs(self.z - zc)[None, :] <= h / 2)
        self.Mx[m] = mdir[0] * BR / mo.MU0; self.Mz[m] = mdir[1] * BR / mo.MU0

    def iron(self, z_lo, z_hi, x_lo=None, x_hi=None):
        m = (self.z >= z_lo) & (self.z <= z_hi)
        if x_lo is None:
            self.mu[:, m] = IRON_MU
        else:
            for k in range(int(round(LAM_DOM / COIL_PITCH))):
                xs = ((self.x - k * COIL_PITCH + LAM_DOM / 2) % LAM_DOM) - LAM_DOM / 2
                mx = (xs >= x_lo) & (xs <= x_hi)
                self.mu[np.ix_(mx, m)] = IRON_MU

    def halbach(self, z_face, up, h=H_M, shift=0.0):
        """One-sided Halbach ring whose active face is at z_face; field goes up (+z) if up."""
        pattern = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        zc = z_face - h / 2 if up else z_face + h / 2
        for k in range(4 * NDOM):
            mdir = pattern[k % 4]
            if not up:
                mdir = (-mdir[0], mdir[1])
            self.block(k * LAM / 4, zc, W_B, h, mdir, shift)

    def ns(self, zc, h=H_M, n_per_pole=2, shift=0.0):
        for k in range(2 * NDOM):
            for j in range(n_per_pole):
                self.block(k * LAM / 2 + (j - (n_per_pole - 1) / 2) * W_B, zc, W_B, h, (0, 1 if k % 2 == 0 else -1), shift)


def coil_turns_profile(x, n_turns, opening=None, trace=None, space=0.15e-3):
    """N(x): turns enclosing x for a concentrated spiral centred at x = 0 with copper envelope COIL_W."""
    leg = (COIL_W - (opening if opening is not None else 0)) / 2 if opening is not None else n_turns * ((trace or GEO["trace_mm"] * 1e-3) + space)
    op = COIL_W - 2 * leg
    xa = np.abs(((x + LAM_DOM / 2) % LAM_DOM) - LAM_DOM / 2)       # the one coil at x = 0
    N = np.where(xa <= op / 2, n_turns, np.where(xa <= COIL_W / 2, n_turns * (COIL_W / 2 - xa) / leg, 0.0))
    return N, op, leg


B_CAP = 1.6                                       # T: what a laminated tooth carries before saturation eats the gain

def linkage_derivative(g, build, n_turns, opening, iron_teeth=False, board_z=(-T_BOARD / 2, T_BOARD / 2), stator_faces=None, iters=1, tooth_w=None):
    """Peak dlambda/dx per turn-metre of radial length, for the coil on the board occupying board_z.
    build(g, shift) places the magnets; the rotor is swept over half a period with one factorisation.
    Iron is linear (mu_r 1000); a tooth's flux is capped at B_CAP x its width, which is the honest
    bound a saturating tooth gives (the linear number is also returned)."""
    Nx, op, leg = coil_turns_profile(g.x, n_turns, opening=opening)
    zb = (g.z >= board_z[0]) & (g.z <= board_z[1])
    xa = np.abs(((g.x + LAM_DOM / 2) % LAM_DOM) - LAM_DOM / 2)
    tooth = (xa <= (tooth_w or 0) / 2) if tooth_w else np.zeros_like(g.x, dtype=bool)
    shifts = np.linspace(0, LAM, 17)[:-1]
    lam, lam_cap, bt = [], [], []
    g.Mx[:] = 0; g.Mz[:] = 0; build(g, 0.0)
    Bx, Bz, lu, mu = solve(g.mu, g.Mx, g.Mz, g.z0, g.nz, iters=1)
    first = (Bx.copy(), Bz.copy(), mu.copy())
    for s in shifts:
        g.Mx[:] = 0; g.Mz[:] = 0
        build(g, s)
        Bx, Bz = apply(lu, mu, g.Mx, g.Mz, g.nz)
        bz_board = Bz[:, zb].mean(axis=1)
        l = float(np.sum(Nx * bz_board) * dx)
        lam.append(l)
        if tooth_w:
            phi_t = float(np.sum(bz_board[tooth]) * dx)                       # flux through this coil's tooth
            phi_cap = float(np.clip(phi_t, -B_CAP * tooth_w, B_CAP * tooth_w))
            lam_cap.append(l - n_turns * (phi_t - phi_cap))
            bt.append(float(np.abs(bz_board[tooth]).max()))
        else:
            lam_cap.append(l)
    lam = np.array(lam); lam_cap = np.array(lam_cap)
    # torque with sinusoidal current in phase with the EMF is set by the fundamental of lambda(theta): k * lambda_1
    k = 2 * math.pi / LAM
    f1 = lambda a: k * float(np.hypot(np.mean(a * np.cos(k * shifts)), np.mean(a * np.sin(k * shifts))) * 2)
    return f1(lam_cap), first, lam, shifts, f1(lam), (max(bt) if bt else 0.0)


def attraction(Bx, Bz, g, z_plane):
    j = int(round((z_plane - g.z0) / H))
    return float(np.mean((Bz[:, j] ** 2 - Bx[:, j] ** 2) / (2 * mo.MU0)) * AREA_MAG)


def fundamental(bz):
    x = (np.arange(nx) + 0.5) * dx
    return float(2 * np.hypot(np.mean(bz * np.cos(2 * math.pi * x / LAM)), np.mean(bz * np.sin(2 * math.pi * x / LAM))))


results = {}
def run(name, z_lo, z_hi, build, n_turns=8, opening=None, boards=((-T_BOARD / 2, T_BOARD / 2),), teeth=None, irons=(), iters=1, extras=None):
    g = Grid(z_lo, z_hi)
    for zi in irons:
        g.iron(*zi)
    if teeth is not None:
        g.iron(teeth[0], teeth[1], -teeth[2] / 2, teeth[2] / 2)
    per_board = []
    lin = []
    for bz in boards:
        dl, first, lam, shifts, dl_lin, b_tooth = linkage_derivative(g, build, n_turns, opening, board_z=bz, iters=iters, tooth_w=(teeth[2] if teeth else None))
        per_board.append(dl); lin.append(dl_lin)
    Bx, Bz, mu = first
    jb = (g.z >= boards[0][0]) & (g.z <= boards[0][1])
    b1 = fundamental(Bz[:, jb].mean(axis=1))
    r = dict(dlam_dx_per_turn=sum(per_board), dlam_dx_linear=sum(lin), boards=len(boards), n_turns=n_turns, B1_board=b1, B_tooth_linear=b_tooth,
             B_peak_iron=float(np.hypot(Bx, Bz)[mu > 1.5].max()) if (mu > 1.5).any() else 0.0)
    if extras:
        r.update(extras(g, Bx, Bz))
    results[name] = r
    results[name]["_field"] = (g.x, g.z, Bz, mu)
    print(f"{name:58s} dλ/dx {sum(per_board):8.3f} (linear {sum(lin):.3f})  B1 {b1:.3f} T  iron peak {r['B_peak_iron']:.2f} T  tooth {b_tooth:.2f} T")
    return r


G1 = T_BOARD + 2 * CLR                            # magnet face to magnet face across one board
# ---- A: the 1s-opt baseline: two Halbach rings facing one board ----
def build_A(g, s):
    g.halbach(-G1 / 2, up=True, shift=s); g.halbach(+G1 / 2, up=False, shift=s)
A = run("A  two Halbach rotors, one stator (1s-opt, baseline)", -(G1 / 2 + H_M + 8e-3), (G1 / 2 + H_M + 8e-3), build_A,
        extras=lambda g, Bx, Bz: dict(attraction_N=attraction(Bx, Bz, g, 0.0)))

# ---- cross-check 1: rotor_field.py's numbers at its own gap and pole pitch ----
# (it used the v1 coil span: r_m 66.6 mm, gap 2.2 + 1.0 mm; B1 at the mid-plane 1.021 T for 10 mm blocks)
def crosscheck():
    global LAM, LAM_DOM, nx, dx
    LAM0, LD0, nx0, dx0 = LAM, LAM_DOM, nx, dx
    LAM = 2 * math.pi * 66.6e-3 / P; LAM_DOM = NDOM * LAM; nx = int(round(LAM_DOM / H)); dx = LAM_DOM / nx
    Gc = 2.2e-3 + 2 * CLR
    g = Grid(-(Gc / 2 + H_M + 8e-3), Gc / 2 + H_M + 8e-3)
    g.halbach(-Gc / 2, up=True); g.halbach(Gc / 2, up=False)
    Bx, Bz, _, _ = solve(g.mu, g.Mx, g.Mz, g.z0, g.nz)
    j0 = int(round((0 - g.z0) / H))
    b1 = fundamental(Bz[:, j0])
    LAM, LAM_DOM, nx, dx = LAM0, LD0, nx0, dx0
    return b1
B1_XC = crosscheck()
print(f"cross-check: FD mid-plane B1 {B1_XC:.3f} T vs rotor_field.py {RF['rect 30x5x10 N48']['B1_midplane']:.3f} T (10 mm blocks, v1 span, 3.2 mm gap)")

# ---- B: one N-S rotor, through-magnetised, between two boards, iron behind each board ----
# stack: iron | clr | board | clr | magnets | clr | board | clr | iron ; two boards, each 0.5 mm from the magnet
ZB = H_M / 2 + CLR + T_BOARD / 2                  # board centre
T_YOKE = 5.0e-3
def build_B(g, s):
    g.ns(0.0, shift=s)
B = run("B  one N-S rotor, two iron-backed stators", -(ZB + T_BOARD / 2 + CLR + T_YOKE + 6e-3), (ZB + T_BOARD / 2 + CLR + T_YOKE + 6e-3), build_B,
        boards=((-ZB - T_BOARD / 2, -ZB + T_BOARD / 2), (ZB - T_BOARD / 2, ZB + T_BOARD / 2)),
        irons=((ZB + T_BOARD / 2 + CLR, ZB + T_BOARD / 2 + CLR + T_YOKE), (-(ZB + T_BOARD / 2 + CLR + T_YOKE), -(ZB + T_BOARD / 2 + CLR))), iters=14,
        extras=lambda g, Bx, Bz: dict(attraction_N=attraction(Bx, Bz, g, H_M / 2 + CLR / 2), yoke_B=float(np.abs(Bx[:, int(round((ZB + T_BOARD / 2 + CLR + T_YOKE / 2 - g.z0) / H))]).max())))

# ---- C: one double-sided Halbach rotor (two rings back to back), two stators, no iron ----
def build_C(g, s):
    g.halbach(+H_M / 2, up=True, h=H_M / 2, shift=s)      # upper ring, 5 mm blocks in [0, 5], faces up
    g.halbach(-H_M / 2, up=False, h=H_M / 2, shift=s)     # lower ring in [-5, 0], faces down
ZC = H_M / 2 + CLR + T_BOARD / 2                          # board centres for the 2 x 5 mm rotor
C = run("C  one back-to-back Halbach rotor (2 x 5 mm), two stators, no iron", -(ZC + T_BOARD / 2 + 10e-3), (ZC + T_BOARD / 2 + 10e-3), build_C,
        boards=((-ZC - T_BOARD / 2, -ZC + T_BOARD / 2), (ZC - T_BOARD / 2, ZC + T_BOARD / 2)))
def build_C10(g, s):
    g.halbach(+H_M, up=True, h=H_M, shift=s); g.halbach(-H_M, up=False, h=H_M, shift=s)
ZC10 = H_M + CLR + T_BOARD / 2
C10 = run("C' same with two 10 mm rings (20 mm of magnet)", -(ZC10 + T_BOARD / 2 + 10e-3), (ZC10 + T_BOARD / 2 + 10e-3), build_C10,
          boards=((-ZC10 - T_BOARD / 2, -ZC10 + T_BOARD / 2), (ZC10 - T_BOARD / 2, ZC10 + T_BOARD / 2)))

# ---- D: C with laminated iron behind each stator ----
D = run("D  back-to-back Halbach rotor, two stators, iron behind each", -(ZC + T_BOARD / 2 + CLR + T_YOKE + 6e-3), (ZC + T_BOARD / 2 + CLR + T_YOKE + 6e-3), build_C,
        boards=((-ZC - T_BOARD / 2, -ZC + T_BOARD / 2), (ZC - T_BOARD / 2, ZC + T_BOARD / 2)),
        irons=((ZC + T_BOARD / 2 + CLR, ZC + T_BOARD / 2 + CLR + T_YOKE), (-(ZC + T_BOARD / 2 + CLR + T_YOKE), -(ZC + T_BOARD / 2 + CLR))), iters=6)

# ---- E: one Halbach rotor, one stator, iron behind the stator (the simplest unit + a back plate) ----
def build_E(g, s):
    g.halbach(-G1 / 2, up=True, shift=s)
E = run("E  one Halbach rotor, one stator, iron behind the board", -(G1 / 2 + H_M + 8e-3), (G1 / 2 + CLR + T_YOKE + 6e-3), build_E,
        irons=((G1 / 2, G1 / 2 + T_YOKE),), iters=6)
E0 = run("E0 one Halbach rotor, one stator, nothing behind the board", -(G1 / 2 + H_M + 8e-3), (G1 / 2 + 10e-3), build_E)

# ---- F: iron teeth in the coil centres of the baseline (A), for coils with 8 / 6 / 4 turns ----
teeth = {}
for n_t in (8, 6, 4):
    _, op, leg = coil_turns_profile(np.zeros(1), n_t, opening=None, trace=None)
    op = COIL_W - 2 * n_t * (GEO["trace_mm"] * 1e-3 + 0.15e-3)
    air = run(f"F0 baseline coil with {n_t} turns (opening {op*1e3:.1f} mm), air core", -(G1 / 2 + H_M + 8e-3), (G1 / 2 + H_M + 8e-3), build_A, n_turns=n_t, opening=op)
    tooth = run(f"F  same with a laminated tooth {op*1e3:.1f} mm wide through the board", -(G1 / 2 + H_M + 8e-3), (G1 / 2 + H_M + 8e-3), build_A, n_turns=n_t, opening=op,
                teeth=(-T_BOARD / 2, T_BOARD / 2, op), iters=14, extras=lambda g, Bx, Bz: dict(attraction_N=attraction(Bx, Bz, g, -(G1 / 2 - CLR / 2))))
    teeth[n_t] = dict(opening_mm=op * 1e3, air=air["dlam_dx_per_turn"], tooth=tooth["dlam_dx_per_turn"], tooth_linear=tooth["dlam_dx_linear"],
                      gain=tooth["dlam_dx_per_turn"] / air["dlam_dx_per_turn"], gain_vs_8t_air=tooth["dlam_dx_per_turn"] / base if 'base' in globals() else tooth["dlam_dx_per_turn"] / results["A  two Halbach rotors, one stator (1s-opt, baseline)"]["dlam_dx_per_turn"],
                      copper_ratio=n_t / 8, B_tooth=tooth["B_tooth_linear"], attraction_N=tooth["attraction_N"])

# ---- roll-up: torque per unit at the rating convention (each board at its own copper-loss budget), mass, cost, stack height ----
base = A["dlam_dx_per_turn"]
def unit(name, key, n_rings_10mm, boards, iron_kg, stack_mm, note, ring_mass_g=None):
    r = results[key]
    kt_ratio_per_board = r["dlam_dx_per_turn"] / boards / base
    T = T_CONT_1S * kt_ratio_per_board * boards         # each board carries the same copper loss as the 1s-opt board
    mag_g = ring_mass_g if ring_mass_g is not None else n_rings_10mm * 60 * (30 * 5 * 10) * 1e-3 * MAG_RHO / 1e3
    board_g = AB.get("board_mass_g", 200.0)
    carriers_g = 547.0 * (2 if n_rings_10mm >= 2 else 1) / 2
    mass = (mag_g + boards * board_g + carriers_g) / 1e3 + iron_kg
    cost20 = boards * BOARD_PRICE[20] + n_rings_10mm * 60 * PRICE_BLOCK_10 + iron_kg * 60 + 45
    cost100 = boards * BOARD_PRICE[100] + n_rings_10mm * 60 * PRICE_BLOCK_10 * 0.8 + iron_kg * 45 + 30
    return dict(name=name, T_cont=T, kt_ratio_per_board=kt_ratio_per_board, boards=boards, magnet_g=mag_g, iron_kg=iron_kg, mass_kg=mass,
                cost20=cost20, cost100=cost100, stack_mm=stack_mm, T_per_kg=T / mass, T_per_100usd=T / cost20 * 100, note=note,
                attraction_N=r.get("attraction_N", 0.0), B_iron=r["B_peak_iron"])

yoke_kg = 2 * math.pi * ((R_M + 18e-3) ** 2 - (R_M - 18e-3) ** 2) * T_YOKE * STEEL_RHO
UNITS = [
    unit("A  two Halbach rotors + one stator (1s-opt)", "A  two Halbach rotors, one stator (1s-opt, baseline)", 2, 1, 0.0, 2 * (4.5 + 10 + 0.5) + 3.11,
         "the round-13 optimum; no iron, no cogging, 2.1 kN pull per rotor on its own carrier"),
    unit("B  one N-S rotor + two iron-backed stators", "B  one N-S rotor, two iron-backed stators", 2, 2, yoke_kg, 10 + 2 * (0.5 + 3.11 + 0.5 + 5) + 2.5,
         "through-magnetised blocks (2 x 30x5x10 per pole, 69 % fill); the yokes see 250 Hz at 1000 rpm and must be wound-strip laminations or SMC, not plate"),
    unit("C  one back-to-back Halbach rotor (2 x 5 mm) + two stators", "C  one back-to-back Halbach rotor (2 x 5 mm), two stators, no iron", 1, 2, 0.0, 10 + 2 * (0.5 + 3.11) + 2.5 + 4,
         "the canonical middle rotor on its own: each board sees one ring instead of two"),
    unit("C' same with two 10 mm rings", "C' same with two 10 mm rings (20 mm of magnet)", 2, 2, 0.0, 20 + 2 * (0.5 + 3.11) + 2.5 + 4,
         "20 mm of magnet in the middle, two boards, no iron"),
    unit("D  back-to-back Halbach rotor + two stators + iron behind each", "D  back-to-back Halbach rotor, two stators, iron behind each", 1, 2, yoke_kg, 10 + 2 * (0.5 + 3.11 + 0.5 + 5) + 2.5,
         "the iron images the ring; same 250 Hz yoke problem as B"),
    unit("E  one Halbach rotor + one stator + iron behind the board", "E  one Halbach rotor, one stator, iron behind the board", 1, 1, yoke_kg / 2, 4.5 + 10 + 0.5 + 3.11 + 0.5 + 5,
         "half the magnets of A"),
    unit("E0 one Halbach rotor + one stator, nothing behind", "E0 one Halbach rotor, one stator, nothing behind the board", 1, 1, 0.0, 4.5 + 10 + 0.5 + 3.11 + 2,
         "the cheapest possible unit"),
]
for u in UNITS:
    print(f"{u['name']:62s} T {u['T_cont']:.2f} N·m  mass {u['mass_kg']:.2f} kg  ${u['cost20']:.0f}/{u['cost100']:.0f}  stack {u['stack_mm']:.0f} mm  {u['T_per_kg']:.2f} N·m/kg  {u['T_per_100usd']:.2f} N·m/$100")

# ---- magnet material: torque at fixed current scales with Br at the magnet's working temperature ----
# Generic grade-chart values (no vendor datasheet in docs/reference yet: flagged), Br at 20 C, reversible tempco, max working temperature
T_MAG = mo.T_MAGNET
GRADES = [
    ("N45 typical (what every rating so far assumed: Br 1.32 T at 20 C)", mo.BR_20, mo.BR_TEMPCO, 80, 1.0),
    ("N48H", 1.39, -0.0012, 120, 1.05),
    ("N52 (no H grade exists)", 1.44, -0.0012, 65, 1.15),
    ("N45SH", 1.34, -0.0012, 150, 1.35),
    ("N42UH", 1.30, -0.0012, 180, 1.6),
    ("SmCo 2:17 (Sm2Co17 28)", 1.08, -0.0003, 300, 4.0),
    ("Ferrite Y30", 0.40, -0.0020, 250, 0.08),
]
grade_rows = []
for nm, br20, tc, tmax, cost in GRADES:
    br_hot = br20 * (1 + tc * (T_MAG - 20))
    ok = T_MAG <= tmax
    grade_rows.append(dict(grade=nm, Br20=br20, Br_hot=br_hot, tmax_C=tmax, usable=ok, torque_ratio=br_hot / BR, cost_ratio=cost))
    print(f"{nm:44s} Br20 {br20:.2f} Br({T_MAG:.0f} C) {br_hot:.2f} T  torque x{br_hot / BR:.2f}  {'ok' if ok else 'NOT usable at ' + str(T_MAG) + ' C'}  cost x{cost}")

out = dict(model=dict(grid_mm=H * 1e3, lam_mm=LAM * 1e3, r_m_mm=R_M * 1e3, coil_w_mm=COIL_W * 1e3, t_board_mm=T_BOARD * 1e3, Br_hot=BR, T_magnet_C=T_MAG,
                      crosscheck_B1_FD=B1_XC, crosscheck_B1_rotor_field=RF["rect 30x5x10 N48"]["B1_midplane"]),
           fields={k: {kk: vv for kk, vv in v.items() if kk != "_field"} for k, v in results.items()},
           units=UNITS, teeth=teeth, grades=grade_rows, T_cont_1s=T_CONT_1S)
json.dump(out, open(os.path.join(ROOT, "hw", "stator", "topology_compare.json"), "w"), indent=1)

# ---- figure ----
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.15))
maps = [("A  two Halbach rotors, one stator (1s-opt, baseline)", "A: two Halbach rotors, one stator"),
        ("B  one N-S rotor, two iron-backed stators", "B: one N-S rotor, two iron-backed stators"),
        ("F  same with a laminated tooth 7.1 mm wide through the board", "F: iron teeth in the coil centres (4-turn coil)")]
for i, (key, title) in enumerate(maps):
    if key not in results:
        key = [k for k in results if k.startswith("F  ")][-1]
    ax = fig.add_subplot(gs[0, i])
    x, z, Bz, mu = results[key]["_field"]
    sel = x <= 2 * LAM
    im = ax.pcolormesh(x[sel] * 1e3, z * 1e3, Bz[sel].T, cmap="RdBu_r", vmin=-1.3, vmax=1.3, shading="auto")
    if (mu > 1.5).any():
        ax.contour(x[sel] * 1e3, z * 1e3, (mu[sel] > 1.5).T.astype(float), levels=[0.5], colors="k", linewidths=1.0)
    ax.set_title(title, fontsize=9); ax.set_xlabel("x along the circumference at r_m (mm)"); ax.set_ylabel("z (mm)")
    ax.set_aspect("equal")
fig.colorbar(im, ax=fig.axes[:3], shrink=0.6, label="B_z (T), one pole pair; black = iron")
ax = fig.add_subplot(gs[1, 0:2])
names = [u["name"] for u in UNITS]; y = np.arange(len(UNITS))
ax.barh(y - 0.22, [u["T_cont"] for u in UNITS], 0.42, color="#0f9b8e", label="continuous torque per unit (N·m), each board at the 1s-opt copper-loss budget")
ax.barh(y + 0.22, [u["mass_kg"] for u in UNITS], 0.42, color="#d98c3a", label="stator + rotor + iron mass (kg)")
for i, u in enumerate(UNITS):
    ax.text(max(u["T_cont"], u["mass_kg"]) + 0.1, i, f"${u['cost20']:.0f} / ${u['cost100']:.0f}, {u['stack_mm']:.0f} mm stack" + (f", iron {u['B_iron']:.1f} T" if u["iron_kg"] else ""), va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 8.5); ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Topologies in the same Ø190 package, 10 mm N48 blocks, 20L 2 oz boards", fontsize=10)
ax = fig.add_subplot(gs[1, 2])
gr = grade_rows
ax.barh(np.arange(len(gr)), [g["torque_ratio"] for g in gr], color=["#0f9b8e" if g["usable"] else "#b03a2e" for g in gr])
ax.set_yticks(np.arange(len(gr))); ax.set_yticklabels([g["grade"].split(" (")[0] for g in gr], fontsize=8); ax.invert_yaxis()
ax.axvline(1.0, color="#222", lw=0.8)
for i, g in enumerate(gr):
    ax.text(g["torque_ratio"] + 0.02, i, f"x{g['torque_ratio']:.2f}, {g['tmax_C']} °C max, cost x{g['cost_ratio']}", va="center", fontsize=7.5)
ax.set_xlim(0, 1.6); ax.set_xlabel(f"torque at fixed current, magnets at {T_MAG:.0f} °C (red: grade cannot run that hot)")
ax.set_title("Magnet material: ±5 % within NdFeB", fontsize=10)
tl = "\n".join(f"{n_t}-turn coil, {t['opening_mm']:.1f} mm tooth: x{t['gain_vs_8t_air']:.2f} vs the 8-turn air coil\n   (linear iron x{t['tooth_linear']/base:.2f}; tooth would see {t['B_tooth']:.1f} T, capped at {B_CAP} T), pull {t['attraction_N']/1e3:.1f} kN/rotor" for n_t, t in teeth.items())
fig.text(0.68, 0.30, "Iron teeth in the coil centres (F):\n" + tl, fontsize=8, va="top", bbox=dict(fc="#f4f3f0", ec="#c3c2b7"))
fig.savefig(os.path.join(ROOT, "docs", "design", "actuator", "topology-compare.png"), dpi=110)
print("wrote topology-compare.png")

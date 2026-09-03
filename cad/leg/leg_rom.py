#!/opt/hw-py/bin/python
"""Range-of-motion and clearance sweep of the leg built by cad/leg/leg.py.

    /opt/hw-py/bin/python cad/leg/leg_rom.py            # 7 x 7 femur x tibia grid at five yaws (~10 min)
    /opt/hw-py/bin/python cad/leg/leg_rom.py --coarse   # 4 x 4 grid, three yaws, for a quick look

Imports the leg model (no geometry is duplicated here) and, for every pose on
the femur x tibia-absolute-angle grid at yaw -90, -45, 0, +45, +90:
  * assembles the leg and finds the minimum distance between each moving
    group and everything it can meet - femur + sectors, tibia + foot, the knee
    pulley, and the six rope runs - against the body slab (floor plate, side
    rail, deck, modules, motors), the hip pod, the coxa, the drums and each
    other.  Distances are exact (OCCT BRepExtrema) with a bounding-box
    prefilter; the closest pair of parts is named.
  * checks the capstan runs at the three drum departure heights the wrapped
    band puts them at (mid, and the two ends of the band's walk), not just
    the drum's mid-height that leg.py draws.
Pairs of leg-side parts whose x-extents do not overlap cannot collide (all
leg-side motion is rotation about x-parallel axes; yaw carries them together),
so their x-gaps are reported once as the designed axial gaps and left out of
the pose sweep, where they would otherwise mask every pose-dependent minimum.

Writes the results into cad/leg/leg.json ("clearances") and renders
docs/design/leg/leg-rom.png.  analysis/leg_loads.py copies them through.
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leg as L                                  # noqa: E402
from OCP.BRepExtrema import BRepExtrema_DistShapeShape   # noqa: E402

COARSE = "--coarse" in sys.argv
N_F, N_T = (4, 4) if COARSE else (7, 7)
YAWS = (-45.0, 0.0, 45.0) if COARSE else (-90.0, -45.0, 0.0, 45.0, 90.0)
BLOCK_MM = 3.0                                    # leg.py's threshold: a pose with less than this is "blocked"
ROOT = L.ROOT
LEG_JSON = os.path.join(ROOT, "cad", "leg", "leg.json")
OUT_PNG = os.path.join(L.FIG_DIR, "leg-rom.png")


# ---- exact distance with the closest points ----------------------------------------------------
def dist_pts(a, b):
    ex = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
    ex.Perform()
    if not ex.IsDone() or ex.NbSolution() == 0:
        return float(a.distance_to(b)), None, None
    p, q = ex.PointOnShape1(1), ex.PointOnShape2(1)
    return float(ex.Value()), (p.X(), p.Y(), p.Z()), (q.X(), q.Y(), q.Z())


def bbox_gap(b1, b2):
    """Distance between two axis-aligned boxes (0 when they overlap)."""
    g = np.maximum(0.0, np.maximum(b1[0] - b2[1], b2[0] - b1[1]))
    return float(np.linalg.norm(g))


def bb_of(s):
    bb = s.bounding_box()
    return np.array([[bb.min.X, bb.min.Y, bb.min.Z], [bb.max.X, bb.max.Y, bb.max.Z]])


def group_min(A, names_a, names_b, cache):
    """Exact minimum distance between two lists of parts, with a bounding-box prefilter.
    Returns (d, part a, part b, point on a, point on b)."""
    cands = []
    for na in names_a:
        if na not in A:
            continue
        ba = cache.setdefault(na, bb_of(A[na][0]))
        for nb in names_b:
            if nb not in A or nb == na:
                continue
            bbb = cache.setdefault(nb, bb_of(A[nb][0]))
            cands.append((bbox_gap(ba, bbb), na, nb))
    cands.sort(key=lambda c: c[0])
    best = (math.inf, None, None, None, None)
    for g, na, nb in cands:
        if g >= best[0]:
            break
        d, p, q = dist_pts(A[na][0], A[nb][0])
        if d < best[0]:
            best = (d, na, nb, p, q)
    return best


# ---- pair groups -------------------------------------------------------------------------------
def x_extent(A, n):
    bb = A[n][0].bounding_box()
    return bb.min.X, bb.max.X


def build_pairs(leg, A0):
    """Pose-dependent pairs (x-overlapping leg-side parts, plus every body part) and the designed axial gaps."""
    body = list(leg.body)
    pod = [n for n in leg.pod]                                     # yaw flange, hub plates, standoffs, shafts, drum bearings
    coxa = list(leg.coxa)
    drums = [n for n in leg.trans]
    femur_struct = [n for n in leg.femur if "bearing" not in n]
    femur_sect = list(leg.fsect)
    crank = [n for n in leg.crank]
    tibia = [n for n in leg.tibia if not n.startswith("knee_pulley")]
    foot = list(leg.foot)
    knee_pulley = ["knee_pulley", "knee_pulley_hub"]
    ropes = [n for n in A0 if n.startswith("rope_") and "_drum_" not in n and "_arc" not in n]
    xe = {n: x_extent(A0, n) for n in A0}

    def overlap(a, b):
        return not (xe[a][1] <= xe[b][0] or xe[b][1] <= xe[a][0])

    def xgap(a, b):
        return max(xe[b][0] - xe[a][1], xe[a][0] - xe[b][1])

    bearings = [n for n in A0 if "bearing" in n]                   # in their seats; the shafts run in them
    body = [n for n in body if n not in bearings]
    pod, coxa, femur_struct, crank, tibia = ([n for n in lst if n not in bearings and n != "knee_pin"] for lst in (pod, coxa, femur_struct, crank, tibia))
    own = {  # a rope's own drum, sector and pulleys; the pulley hubs on their own shafts; bearings in their seats
        "rope_femur_A": {"drum_femur", "femur_sector_A", "femur_tensioner_A"} | {f"rope_femur_drum_{k}" for k in range(5)},
        "rope_femur_B": {"drum_femur", "femur_sector_B", "femur_tensioner_B"} | {f"rope_femur_drum_{k}" for k in range(5)},
        "rope_knee_A": {"drum_knee", "crank_sector_A", "crank_tensioner_A"} | {f"rope_knee_drum_{k}" for k in range(5)},
        "rope_knee_B": {"drum_knee", "crank_sector_B", "crank_tensioner_B"} | {f"rope_knee_drum_{k}" for k in range(5)},
        "rope_link_ext": {"drive_pulley", "knee_pulley", "link_tensioner_ext"},
        "rope_link_int": {"drive_pulley", "knee_pulley", "link_tensioner_int"},
    }
    # the capstan and link runs are fixed in the coxa frame: against the pod, coxa and drums they are checked once (at the
    # three departure heights); in the pose loop they meet only the body (yaw) and the moving parts whose x-extents overlap
    groups = {
        "femur + sectors": (femur_struct + femur_sect, body + pod + coxa + drums),
        "tibia + foot": (tibia + foot, body + pod + coxa + drums + femur_struct + femur_sect + crank),
        "knee pulley": (knee_pulley, body + pod + coxa + drums + crank),
        "capstan rope runs": (["rope_femur_A", "rope_femur_B", "rope_knee_A", "rope_knee_B"], body + femur_struct + femur_sect + crank + tibia + foot + knee_pulley),
        "link loop runs": (["rope_link_ext", "rope_link_int"], body + femur_struct + femur_sect + tibia + foot + crank),
    }
    fixed = {"rope runs vs pod / coxa / drums": (ropes, pod + coxa + drums)}
    pairs, axial = {}, {}
    for gname, (a_list, b_list) in list(groups.items()) + list(fixed.items()):
        keep_b = {}
        for a in a_list:
            kb = []
            for b in b_list:
                if b == a or b in own.get(a, ()):
                    continue
                if b in body or overlap(a, b):
                    kb.append(b)
                else:
                    g = xgap(a, b)
                    if g < 6.0 and A0[a][0].distance_to(A0[b][0]) < g + 0.05:   # face to face across the gap at the stance
                        axial[f"{a} vs {b}"] = round(g, 2)
            keep_b[a] = kb
        pairs[gname] = keep_b
    return pairs, axial, list(fixed)


def departure_cases():
    """Capstan run departure heights (mm from the drum mid-height): mid, and the ends of the band's walk."""
    z_cases = {"mid": None}
    for j in ("femur", "knee"):
        Ab, Bb, *_ = L.drum_band(j)
        z_cases[f"{j} band: A low / B low"] = {f"{j}_A": Ab[0], f"{j}_B": Bb[0]}
        z_cases[f"{j} band: A high / B high"] = {f"{j}_A": Ab[1], f"{j}_B": Bb[1]}
    return z_cases


def fixed_rope_checks(leg, pairs, fixed_names, z_cases):
    """Rope runs against the pod, coxa and drums, which turn with them: pose-independent, so once per departure case."""
    phi, tau = L.STANCE["femur_deg"], L.tau_of(L.STANCE["femur_deg"], L.STANCE["knee_deg"])
    out = {}
    for gname in fixed_names:
        for cname, zo in z_cases.items():
            A = leg.assemble(phi, tau, 0.0, ropes=False)
            R, _ = L.rope_solids(phi, tau, zo)
            for n, sld in R.items():
                A[n] = (sld, "transmission")
            cache = {}
            for a, b_list in pairs[gname].items():
                if zo is not None and not any(a.startswith(f"rope_{k}") for k in zo):
                    continue                      # this case moves other runs
                d, na, nb, p, q = group_min(A, [a], b_list, cache)
                key = f"{a} ({cname})"
                out[key] = dict(min_mm=round(d, 2), vs=nb, at=[round(c, 1) for c in p] if p else None)
    return out


def use_proxies(leg):
    """Swap the sector plates and link pulleys for conservative envelopes (no lightening holes, no groove: a superset
    of each part, built with leg.py's own helpers), which OCCT measures ten times faster.  Distances can only shrink."""
    for run, (x0, x1) in (("A", L.X_FSECT), ("B", (-L.X_FSECT[1], -L.X_FSECT[0]))):
        a0, a1 = L.GROOVE_F[run][0] - L.GROOVE_MARGIN, L.GROOVE_F[run][1] + L.GROOVE_MARGIN
        leg.fsect[f"femur_sector_{run}"] = L.x_arc_sector(x0, x1, 32.0, L.R_SECTOR["femur"] + 3, a0, a1) + L.x_disc(x0, x1, 40.0)
    for run, (x0, x1) in (("A", L.X_CRANK), ("B", (-L.X_CRANK[1], -L.X_CRANK[0]))):
        a0, a1 = L.GROOVE_C[run][0] - L.GROOVE_MARGIN, L.GROOVE_C[run][1] + L.GROOVE_MARGIN
        leg.crank[f"crank_sector_{run}"] = L.x_arc_sector(x0, x1, 32.0, L.R_SECTOR["knee"] + 3, a0, a1) + L.x_disc(x0, x1, 40.0)
    disc = L.x_disc(L.X_LINK[0], L.X_LINK[1], L.R_LINK + 3)
    leg.crank["drive_pulley"] = disc
    leg.tibia["knee_pulley"] = disc


# ---- the sweep ---------------------------------------------------------------------------------
def sweep(leg, profile=False):
    use_proxies(leg)
    phis = np.linspace(*L.FEMUR_RANGE, N_F)
    taus = np.linspace(*L.TAU_RANGE, N_T)
    A0 = leg.assemble(L.STANCE["femur_deg"], L.tau_of(L.STANCE["femur_deg"], L.STANCE["knee_deg"]), 0.0)
    pairs, axial, fixed_names = build_pairs(leg, A0)
    z_cases = departure_cases()
    fixed = fixed_rope_checks(leg, pairs, fixed_names, z_cases)
    pose_pairs = {g: kb for g, kb in pairs.items() if g not in fixed_names}
    t0 = time.time()
    result = {}
    n_done = 0
    for psi in YAWS:
        M = np.full((N_F, N_T), np.inf)
        who = {}
        worst_by_group = {g: (math.inf, None) for g in pose_pairs}
        for i, phi in enumerate(phis):
            for k, tau in enumerate(taus):
                th = L.theta_of(phi, tau)
                if th < 0 or th > 180:
                    M[i, k] = np.nan
                    continue
                A = leg.assemble(phi, tau, psi, ropes=True)
                cache = {}
                pose_min = (math.inf, None)
                for gname, kb in pose_pairs.items():
                    tg = time.time()
                    best = (math.inf, None, None, None, None)
                    for a, b_list in kb.items():
                        r = group_min(A, [a], b_list, cache)
                        if r[0] < best[0]:
                            best = r
                    if profile:
                        print(f"    {gname}: {best[0]:.1f} mm ({best[1]} vs {best[2]}) in {time.time() - tg:.1f} s")
                    if best[0] < worst_by_group[gname][0]:
                        worst_by_group[gname] = (best[0], dict(femur=float(phi), tau=float(tau), knee=round(th, 1), yaw=psi, a=best[1], b=best[2], p=best[3], q=best[4]))
                    if best[0] < pose_min[0]:
                        pose_min = (best[0], f"{best[1]} vs {best[2]} [{gname}]")
                M[i, k] = pose_min[0]
                who[(i, k)] = pose_min[1]
                n_done += 1
                if profile:
                    print(f"  pose femur {phi:.0f} tau {tau:.0f} yaw {psi:.0f}: {pose_min[0]:.1f} mm, {time.time() - t0:.1f} s")
                    return
        result[psi] = (M, who, worst_by_group)
        fin = M[np.isfinite(M)]
        print(f"  yaw {psi:+.0f}: min {fin.min():.1f} mm over {fin.size} poses, {int((fin < BLOCK_MM).sum())} blocked; {time.time() - t0:.0f} s, {n_done} poses")
    return phis, taus, result, axial, pairs, z_cases, fixed


# ---- record ------------------------------------------------------------------------------------
def summarise(phis, taus, result, axial, z_cases, fixed):
    rec = {"grid": dict(femur_deg=[float(p) for p in phis], tau_deg=[float(t) for t in taus], yaw_deg=list(YAWS), n_poses=int(N_F * N_T * len(YAWS)),
                        block_mm=BLOCK_MM, knee_limits_deg=list(L.KNEE_LIMITS)),
           "departure_cases": {k: ({r: round(v, 1) for r, v in v.items()} if v else "mid-height, as leg.py draws") for k, v in z_cases.items()},
           "by_yaw": {}, "worst_by_group": {}, "collisions": [], "designed_axial_gaps_mm": dict(sorted(axial.items(), key=lambda kv: kv[1])),
           "rope_runs_vs_coxa_frame_mm": dict(sorted(fixed.items(), key=lambda kv: kv[1]["min_mm"]))}
    gmin = {}
    for psi, (M, who, wbg) in result.items():
        valid = np.array([[L.KNEE_LIMITS[0] <= L.theta_of(p, t) <= L.KNEE_LIMITS[1] for t in taus] for p in phis])
        fin = np.isfinite(M)
        ok = fin & valid & (M >= BLOCK_MM)
        blocked = [dict(femur=float(phis[i]), tau=float(taus[k]), knee=round(L.theta_of(phis[i], taus[k]), 1), clearance=round(float(M[i, k]), 1), pair=who[(i, k)])
                   for i in range(len(phis)) for k in range(len(taus)) if fin[i, k] and valid[i, k] and M[i, k] < BLOCK_MM]
        feas = M[ok]
        rec["by_yaw"][f"{psi:+.0f}"] = dict(min_over_feasible_poses_mm=round(float(feas.min()), 1) if feas.size else None,
                                             feasible=int(ok.sum()), within_knee_limits=int((valid & fin).sum()), total=int(M.size),
                                             blocked=blocked, grid=[[None if not np.isfinite(v) else round(float(v), 1) for v in row] for row in M])
        for g, (d, info) in wbg.items():
            if info and (g not in gmin or d < gmin[g][0]):
                gmin[g] = (d, info)
        for b in blocked:
            if b["clearance"] <= 0.05:
                rec["collisions"].append(dict(yaw=psi, **b))
    for g, (d, info) in gmin.items():
        rec["worst_by_group"][g] = dict(min_mm=round(d, 2), **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in info.items() if k not in ("p", "q")},
                                        at=[round(c, 1) for c in info["p"]] if info["p"] else None)
    fleet = L.fleet_angles()
    rec["fleet_angle_deg"] = fleet
    rec["drum_bands"] = {j: dict(zip(("A_departure_from_mid", "B_departure_from_mid", "band_mm", "groove_mm", "walk_mm"), [list(np.round(v, 1)) if isinstance(v, tuple) else round(v, 1) for v in L.drum_band(j)])) for j in ("femur", "knee")}
    rec["summary"] = dict(
        min_clearance_mm=round(min(v[0] for v in gmin.values()), 2),
        min_pair=min(gmin.items(), key=lambda kv: kv[1][0])[0],
        rope_vs_coxa_frame_min_mm=min(v["min_mm"] for v in fixed.values()),
        rope_vs_coxa_frame_min_pair=min(fixed.items(), key=lambda kv: kv[1]["min_mm"])[0] + " vs " + min(fixed.values(), key=lambda v: v["min_mm"])["vs"],
        collisions=len(rec["collisions"]),
        poses_blocked={k: len(v["blocked"]) for k, v in rec["by_yaw"].items()},
        axial_gaps_under_1mm=[k for k, v in axial.items() if v < 1.0],
        fleet_angle_max_deg=max(fleet.values()),
    )
    return rec


# ---- figure ------------------------------------------------------------------------------------
def draw(phis, taus, result, rec, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ny = len(YAWS)
    fig = plt.figure(figsize=(17, 11.5))
    gs = fig.add_gridspec(2, ny, height_ratios=[1.9, 1], width_ratios=[1] * ny)
    ax = fig.add_subplot(gs[0, : max(1, ny - 2)])
    # slab, coxa, hub plates and drums in the leg plane at yaw 0
    ax.add_patch(plt.Rectangle((-L.BODY_HALF_W - 250, 0), L.BODY_HALF_W + 250 + L.Y_SLAB_EDGE, L.SLAB_H, color="#dfe3e6", ec="#999"))
    ax.add_patch(plt.Rectangle((L.Y_RAIL_IN, L.T_PLATE), L.Y_SLAB_EDGE - L.Y_RAIL_IN, L.SLAB_H - 2 * L.T_PLATE, color="#c9ced3", ec="#999"))
    ax.fill(*zip(*L.coxa_profile()), color=L.COLORS["coxa"], alpha=0.3)
    for (z0, z1), r in ((L.Z_HUB_TOP, 42), (L.Z_HUB_MID, 26), (L.Z_HUB_BOT, 22)):
        ax.add_patch(plt.Rectangle((L.HUB_Y_IN, z1), r - L.HUB_Y_IN, z0 - z1, color="#8a97a3", alpha=0.6))
    for j in ("knee", "femur"):
        z0, z1 = L.Z_DRUM[j]
        ax.add_patch(plt.Rectangle((-L.R_DRUM[j], z1), 2 * L.R_DRUM[j], z0 - z1, color=L.COLORS["drum"], alpha=0.6))
    gz = L.Z_PIVOT - L.HIP_HEIGHT
    ax.axhline(gz, color="#5a3e1b", lw=1.2)
    ax.text(-300, gz + 8, f"ground in the stance (femur axis {L.HIP_HEIGHT:.0f} mm up)", fontsize=8, color="#5a3e1b")
    M0 = result[0.0][0] if 0.0 in result else next(iter(result.values()))[0]
    for i, phi in enumerate(phis):
        for k, tau in enumerate(taus):
            th = L.theta_of(phi, tau)
            if not (L.KNEE_LIMITS[0] <= th <= L.KNEE_LIMITS[1]):
                continue
            kn = L.PIV + L.L_FEMUR * L.yz_dir(phi)[1:]
            ft = kn + L.L_TIBIA * L.yz_dir(tau)[1:]
            ok = np.isfinite(M0[i, k]) and M0[i, k] >= BLOCK_MM
            ax.plot([L.PIV[0], kn[0]], [L.PIV[1], kn[1]], color="#0f9b8e" if ok else "#d9a3a3", lw=0.7, alpha=0.7)
            ax.plot([kn[0], ft[0]], [kn[1], ft[1]], color="#2f6f9f" if ok else "#e5b8b8", lw=0.6, alpha=0.6)
            ax.plot(ft[0], ft[1], ".", color="#2f6f9f" if ok else "#c0392b", ms=3)
    # named poses with the rope paths and the sector arcs
    poses = ((L.STANCE["femur_deg"], L.tau_of(L.STANCE["femur_deg"], L.STANCE["knee_deg"]), "#c0392b", "sprawl stance"),
             (L.MAMMAL["femur_deg"], L.tau_of(L.MAMMAL["femur_deg"], L.MAMMAL["knee_deg"]), "#7d3c98", "mammal stance (yaw 90)"),
             (L.FEMUR_RANGE[1], L.TAU_RANGE[1], "#1f618d", "femur up / tibia back"),
             (L.FEMUR_RANGE[0], L.TAU_RANGE[0], "#117a65", "femur down / tibia forward"))
    for phi, tau, col, lab in poses:
        g = L.rope_geometry(phi, tau)
        kn = g["knee"]
        ft = kn + L.L_TIBIA * L.yz_dir(tau)[1:]
        th = L.theta_of(phi, tau)
        ax.plot([L.PIV[0], kn[0], ft[0]], [L.PIV[1], kn[1], ft[1]], color=col, lw=2.6, label=f"{lab}: femur {phi:.0f}°, tibia {tau:.0f}° (knee {th:.0f}°)")
        for name, (p, q) in g["runs"].items():
            ax.plot([p[1], q[1]], [p[2], q[2]], color=L.COLORS["rope"] if name.startswith("link") else "#b8860b", lw=1.4, alpha=0.9)
        for run in ("A", "B"):
            a0, a1 = L.GROOVE_F[run]
            t_ = np.radians(np.linspace(a0 + phi, a1 + phi, 40))
            ax.plot(L.PIV[0] + L.R_SECTOR["femur"] * np.cos(t_), L.PIV[1] + L.R_SECTOR["femur"] * np.sin(t_), color=L.COLORS["sector"], lw=1.6, alpha=0.7)
            a0, a1 = L.GROOVE_C[run]
            t_ = np.radians(np.linspace(a0 + tau, a1 + tau, 40))
            ax.plot(L.PIV[0] + (L.R_SECTOR["knee"] + 4) * np.cos(t_), L.PIV[1] + (L.R_SECTOR["knee"] + 4) * np.sin(t_), color=L.COLORS["crank"], lw=1.6, alpha=0.7)
        for c in (L.PIV, kn):
            ax.add_patch(plt.Circle((c[0], c[1]), L.R_LINK, fill=False, color=L.COLORS["pulley"], lw=0.8, ls=":"))
    # fleet angles at the drums
    g = L.rope_geometry(*poses[0][:2])
    for n, a in rec["fleet_angle_deg"].items():
        p, q = g["runs"][n]
        ax.annotate(f"{n} leaves the drum at {a:.0f}° to its plane of rotation", (0.35 * p[1] + 0.65 * q[1], 0.35 * p[2] + 0.65 * q[2]),
                    (-310, -60 - 22 * list(rec["fleet_angle_deg"]).index(n)), fontsize=7, color="#7a5c00", arrowprops=dict(arrowstyle="-", color="#b8860b", lw=0.6))
    # min-clearance callouts at the worst pose of each group (points rotated back into the leg plane)
    k = 0
    for gname, w in sorted(rec["worst_by_group"].items(), key=lambda kv: kv[1]["min_mm"]):
        if w["at"] is None:
            continue
        x, y, z = w["at"]
        psi = math.radians(w["yaw"])
        yy = -x * math.sin(psi) + y * math.cos(psi)        # back into the coxa frame (yaw undone); x dropped
        col = "#b03a2e" if w["min_mm"] < BLOCK_MM else "#2c3e50"
        ax.plot(yy, z, "o", ms=6, mfc="none", mec=col, mew=1.4)
        ax.annotate(f"{w['min_mm']:.1f} mm: {w['a']} vs {w['b']}\n({gname}; femur {w['femur']:.0f}°, tibia {w['tau']:.0f}°, yaw {w['yaw']:+.0f}°)",
                    (yy, z), (560, 200 - 46 * k), fontsize=7, color=col, ha="left", va="center",
                    bbox=dict(fc="white", ec=col, lw=0.6, alpha=0.9), arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
        k += 1
    ax.set_aspect("equal")
    ax.set_xlim(-330, 940)
    ax.set_ylim(gz - 60, L.SLAB_H + 40)
    ax.set_xlabel("y, mm (outboard →), leg plane at yaw 0")
    ax.set_ylabel("z, mm")
    ax.legend(fontsize=7.5, loc="lower right")
    s = rec["summary"]
    ax.set_title(f"Range of motion: femur {L.FEMUR_RANGE[0]:.0f}..{L.FEMUR_RANGE[1]:.0f}°, tibia absolute {L.TAU_RANGE[0]:.0f}..{L.TAU_RANGE[1]:.0f}°, knee {L.KNEE_LIMITS[0]:.0f}..{L.KNEE_LIMITS[1]:.0f}°, yaw {YAWS[0]:+.0f}..{YAWS[-1]:+.0f}°: "
                 f"{rec['grid']['n_poses']} poses, min clearance {s['min_clearance_mm']:.1f} mm ({s['min_pair']}), {s['collisions']} collisions\n"
                 "pink sticks = poses blocked at yaw 0 (< 3 mm); rope runs and sector arcs at four named poses; circles = where the minimum of each group occurs", fontsize=9)
    # the designed axial gaps and the blocked-pose count, as text
    axt = fig.add_subplot(gs[0, max(1, ny - 2):])
    axt.set_axis_off()
    lines = ["Designed axial gaps (x, constant over the range; < 1 mm flagged):"]
    for kname, v in list(rec["designed_axial_gaps_mm"].items())[:14]:
        lines.append(f"{'  !! ' if v < 1.0 else '     '}{v:4.1f} mm  {kname}")
    lines.append("")
    lines.append("Rope runs vs pod / coxa / drums (pose-independent; mid = as drawn, band = ends of the wrap's walk):")
    for kname, v in list(rec["rope_runs_vs_coxa_frame_mm"].items())[:10]:
        lines.append(f"{'  !! ' if v['min_mm'] < BLOCK_MM else '     '}{v['min_mm']:4.1f} mm  {kname} vs {v['vs']}")
    lines.append("")
    lines.append("Poses blocked (< 3 mm) per yaw, of those within the knee limits:")
    for y_, v in rec["by_yaw"].items():
        lines.append(f"     yaw {y_:>3s}: {len(v['blocked'])} of {v['within_knee_limits']}; min over feasible {v['min_over_feasible_poses_mm']} mm")
    lines.append("")
    lines.append("Capstan runs at the drum (inclination to the drum's plane of rotation):")
    for n, a in rec["fleet_angle_deg"].items():
        lines.append(f"     {n}: {a:.1f}°   (a grooved capstan wants < ~1.5°)")
    lines.append("")
    for j, b in rec["drum_bands"].items():
        lines.append(f"{j} drum: band {b['band_mm']:.0f} + walk {b['walk_mm']:.0f} mm in a {b['groove_mm']:.0f} mm groove{'' if b['band_mm'] + b['walk_mm'] <= b['groove_mm'] else '  !! does not fit'}")
    axt.text(0.0, 1.0, "\n".join(lines), fontsize=7.6, family="monospace", va="top", ha="left", transform=axt.transAxes)
    # heat maps per yaw
    for c, psi in enumerate(YAWS):
        axm = fig.add_subplot(gs[1, c])
        M = result[psi][0]
        Mp = np.where(np.isfinite(M), np.clip(M, 0, 40), np.nan)
        axm.imshow(Mp.T, origin="lower", extent=(phis[0], phis[-1], taus[0], taus[-1]), aspect="auto", cmap="RdYlGn", vmin=0, vmax=40)
        for i in range(len(phis)):
            for kk in range(len(taus)):
                if np.isfinite(M[i, kk]):
                    inside = L.KNEE_LIMITS[0] <= L.theta_of(phis[i], taus[kk]) <= L.KNEE_LIMITS[1]
                    axm.text(phis[i], taus[kk], f"{M[i, kk]:.0f}", ha="center", va="center", fontsize=6.5, color=("k" if M[i, kk] >= BLOCK_MM else "w") if inside else "#777")
        for th_lim in L.KNEE_LIMITS:
            axm.plot(phis, phis + 180 + th_lim, color="#555", lw=0.8, ls="--")
        axm.set_ylim(taus[0], taus[-1])
        axm.set_xlim(phis[0], phis[-1])
        axm.set_title(f"yaw {psi:+.0f}°: min clearance, mm", fontsize=8.5)
        axm.set_xlabel("femur, deg")
        if c == 0:
            axm.set_ylabel("tibia absolute angle, deg\n(dashed: knee 15° / 145°)")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    t_all = time.time()
    leg = L.Leg()
    if "--profile" in sys.argv:
        sweep(leg, profile=True)
        sys.exit(0)
    phis, taus, result, axial, pairs, z_cases, fixed = sweep(leg)
    rec = summarise(phis, taus, result, axial, z_cases, fixed)
    draw(phis, taus, result, rec, OUT_PNG)
    LEG = json.load(open(LEG_JSON))
    LEG["clearances"] = rec
    json.dump(LEG, open(LEG_JSON, "w"), indent=1)
    print(json.dumps(rec["summary"], indent=1))
    print("worst by group:", json.dumps(rec["worst_by_group"], indent=1))
    print("designed axial gaps:", json.dumps(rec["designed_axial_gaps_mm"], indent=1))
    print("rope runs vs the coxa frame:", json.dumps(rec["rope_runs_vs_coxa_frame_mm"], indent=1))
    print("collisions:", json.dumps(rec["collisions"], indent=1))
    print(f"done in {time.time() - t_all:.0f} s -> {OUT_PNG}")

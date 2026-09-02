#!/usr/bin/env python3
"""Generate the axial-flux stator PCB with KiCad's own pcbnew API.

    python3 hw/stator/make_stator.py          # writes hw/stator/stator.kicad_pcb + geometry.json

Winding (see docs/design/08-actuator-design.md):
  36 concentrated coils, 30 poles (12-slot/10-pole family x3), 12 copper layers.
  Each coil: an inward spiral on the six odd layers and an outward (mirrored)
  spiral on the six even layers, joined by a through via at the coil centre,
  so both terminals T1 (start) and T2 (end) sit in the outer band.  Odd layers
  are in parallel, even layers in parallel; series turns per coil = 2 x N_T.
  Phase: two coils with identical EMF phasor in parallel (a pair), two pairs in
  series, and the four 90-degree repeats in parallel.  Series turns per
  phase = 4 x N_T.

All geometry constants are at the top.  Copper is left on net 0: KiCad cannot
represent a coil (one conductor joining two nodes) as a net, so DRC is used
for width / clearance / edge / drill rules and the turn spacing is by
construction.
"""
import json
import math
import os
import sys

import pcbnew
from pcbnew import VECTOR2I

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stator.kicad_pcb")

# ---- design constants (mm, degrees) ------------------------------------------
N_COILS = 36
POLE_PAIRS = 15
N_LAYERS = 12
COPPER_OZ = 3               # per layer; JLCPCB multilayer stops at 2 oz, PCBWay-class houses do 3 oz
N_T = 10                    # turns per layer (the 8-turn variant in variants/8t trades eddy loss for the yaw joint's speed)
TRACE = 0.28                # coil trace width
SPACE = 0.15                # copper-to-copper
GAP = 0.60                  # coil-to-coil gap at the legs
R_IN = 53.0                 # innermost coil copper
R_OUT = 84.6                # outermost coil copper: to the magnets' outer edge (v2, round 9; was 80.2 with the interconnect inside the magnet span)
R_VIA_T = 85.3              # terminal vias
R_M = 86.15                 # M-node arcs (In9/In10/B.Cu), 0.7 wide
R_RING = {"A": 87.1, "B": 88.0, "C": 88.9}   # phase rings (3 layers each), 0.7 wide, 0.2 space, outside the magnets
R_N = 50.9                  # star ring, inner band (In9/In10/B.Cu)
R_PAD = 90.4                # phase terminal pads, at the clamp radius (wall notches at 88/208/328 deg)
R_BOARD = 91.9              # board edge (Ø184)
R_BORE = 50.0
RING_W = 0.7
M_W = 0.7
STUB_W = 0.4
JUMPER_W = 0.3
VIA_D, VIA_DRILL = 0.6, 0.3
PAD_D, PAD_DRILL = 2.0, 1.0

PITCH = TRACE + SPACE
SECTOR = 360.0 / N_COILS
# layers


def set_layers(n):
    """Layer roles for an n-layer board (n even, >= 12): coils on every layer;
    rings on the first 9, N ring / jumpers / M arcs on 10-12; extra layers coil only."""
    global CU, ODD, EVEN, RING_LAYERS, NM_LAYERS, M_LAYER, N_LAYERS
    N_LAYERS = n
    CU = [pcbnew.F_Cu] + [getattr(pcbnew, f"In{i}_Cu") for i in range(1, n - 1)] + [pcbnew.B_Cu]
    ODD = CU[0::2]              # inward spirals
    EVEN = CU[1::2]             # outward spirals (mirrored)
    RING_LAYERS = {"A": CU[0:3], "B": CU[3:6], "C": CU[6:9]}
    NM_LAYERS = CU[9:12]        # N ring + gap jumpers + M arcs
    M_LAYER = {"A": CU[9], "B": CU[10], "C": CU[11]}


set_layers(N_LAYERS)


def board_thickness(n, oz):
    """Finished thickness estimate: copper + ~0.09 mm dielectric per interface (12L 3 oz -> 2.25 mm)."""
    return round(n * oz * 0.035 + (n - 1) * 0.09, 2)

# 12-slot/10-pole star of slots, phasor angle 150*k: (phase, sign) per local coil
PHASES = {0: ("A", +1), 6: ("A", -1), 7: ("A", +1), 1: ("A", -1),
          8: ("B", +1), 2: ("B", -1), 3: ("B", +1), 9: ("B", -1),
          4: ("C", +1), 10: ("C", -1), 11: ("C", +1), 5: ("C", -1)}
PAIR1 = {"A": (0, 6), "B": (8, 2), "C": (4, 10)}     # ring -> pair1 -> M
PAIR2 = {"A": (7, 1), "B": (3, 9), "C": (11, 5)}     # M -> pair2 -> N


def mm(v):
    return int(round(v * 1e6))


def pt(r, deg):
    a = math.radians(deg)
    return VECTOR2I(mm(r * math.cos(a)), mm(-r * math.sin(a)))


class Builder:
    def __init__(self):
        self.b = pcbnew.BOARD()
        ds = self.b.GetDesignSettings()
        ds.SetCopperLayerCount(N_LAYERS)
        ds.m_TrackMinWidth = mm(0.15)
        ds.m_ViasMinSize = mm(0.5)
        ds.m_MinThroughDrill = mm(0.3)
        ds.m_MinClearance = mm(0.15)
        ds.m_CopperEdgeClearance = mm(0.3)
        self.length = {}        # (layer, kind) -> mm of track
        self.marc_len = {}      # phase -> mm of M arc per board

    def track(self, p, q, w, layer, kind="coil"):
        t = pcbnew.PCB_TRACK(self.b)
        t.SetStart(p); t.SetEnd(q); t.SetWidth(mm(w)); t.SetLayer(layer)
        self.b.Add(t)
        self.length[(layer, kind)] = self.length.get((layer, kind), 0.0) + math.hypot(p.x - q.x, p.y - q.y) / 1e6

    def arc(self, r, a0, a1, w, layer, kind="coil"):
        """Arc at radius r from angle a0 to a1 (deg); split into <= 90-degree pieces."""
        n = max(1, int(math.ceil(abs(a1 - a0) / 90.0)))
        for i in range(n):
            s = a0 + (a1 - a0) * i / n
            e = a0 + (a1 - a0) * (i + 1) / n
            t = pcbnew.PCB_ARC(self.b)
            t.SetStart(pt(r, s)); t.SetMid(pt(r, (s + e) / 2)); t.SetEnd(pt(r, e))
            t.SetWidth(mm(w)); t.SetLayer(layer)
            self.b.Add(t)
            self.length[(layer, kind)] = self.length.get((layer, kind), 0.0) + abs(math.radians(e - s)) * r

    def via(self, p):
        v = pcbnew.PCB_VIA(self.b)
        v.SetPosition(p); v.SetDrill(mm(VIA_DRILL)); v.SetWidth(mm(VIA_D))
        v.SetViaType(pcbnew.VIATYPE_THROUGH); v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        self.b.Add(v)

    def circle(self, r, layer, w=0.1):
        s = pcbnew.PCB_SHAPE(self.b)
        s.SetShape(pcbnew.SHAPE_T_CIRCLE)
        s.SetCenter(VECTOR2I(0, 0)); s.SetEnd(VECTOR2I(mm(r), 0))
        s.SetLayer(layer); s.SetWidth(mm(w))
        self.b.Add(s)

    def pads(self, positions):
        fp = pcbnew.FOOTPRINT(self.b)
        fp.SetReference("J1")
        fp.SetPosition(VECTOR2I(0, 0))
        for i, (name, p) in enumerate(positions, 1):
            pad = pcbnew.PAD(fp)
            pad.SetNumber(str(i)); pad.SetName(name)
            pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
            pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
            pad.SetSize(VECTOR2I(mm(PAD_D), mm(PAD_D)))
            pad.SetDrillSize(VECTOR2I(mm(PAD_DRILL), mm(PAD_DRILL)))
            pad.SetLayerSet(pcbnew.LSET.AllCuMask())
            pad.SetPosition(p)
            fp.Add(pad)
        self.b.Add(fp)

    # ---- one coil -------------------------------------------------------------
    def spiral(self, theta_c, layer, mirrored):
        """Spiral in a sector centred on theta_c.  Not mirrored: starts at the
        outer-LEFT corner (lower angle), winds inward clockwise-in-sector, ends
        at the sector centre.  Mirrored: starts at the centre, ends at the
        outer-RIGHT corner, same circulation sense."""
        half_in = (2 * math.pi * R_IN / N_COILS - GAP) / 2          # half leg-span at r_in, mm
        # angular half-width of the coil at radius r: (sector arc - gap)/2 / r
        def half_deg(r, i):
            return math.degrees(((2 * math.pi * r / N_COILS - GAP) / 2 - i * PITCH) / r)
        pts = []   # polyline of (r, theta) with arcs flagged
        for i in range(N_T):
            r_top = R_OUT - i * PITCH
            r_bot = R_IN + i * PITCH
            hl_top = half_deg(r_top, i)
            hl_bot = half_deg(r_bot, i)
            r_top_n = R_OUT - (i + 1) * PITCH
            hl_top_n = half_deg(r_top_n, i + 1)
            # left leg down, inner arc, right leg up, outer arc (to next turn's left corner)
            pts.append(("line", (r_top, theta_c - hl_top), (r_bot, theta_c - hl_bot)))
            pts.append(("arc", r_bot, theta_c - hl_bot, theta_c + hl_bot))
            pts.append(("line", (r_bot, theta_c + hl_bot), (r_top_n, theta_c + hl_top_n)))
            if i < N_T - 1:
                pts.append(("arc", r_top_n, theta_c + hl_top_n, theta_c - hl_top_n))
            else:
                # last turn: outer arc to the centre angle, then a stub to the centre point
                pts.append(("arc", r_top_n, theta_c + hl_top_n, theta_c))
                r_ctr = (r_top_n + R_IN + N_T * PITCH) / 2
                pts.append(("line", (r_top_n, theta_c), (r_ctr, theta_c)))
        if mirrored:
            def m(a):
                return 2 * theta_c - a
            out = []
            for e in pts:
                if e[0] == "line":
                    out.append(("line", (e[1][0], m(e[1][1])), (e[2][0], m(e[2][1]))))
                else:
                    out.append(("arc", e[1], m(e[2]), m(e[3])))
            pts = out
        for e in pts:
            if e[0] == "line":
                self.track(pt(*e[1]), pt(*e[2]), TRACE, layer)
            else:
                self.arc(e[1], e[2], e[3], TRACE, layer)
        r_ctr = (R_OUT - N_T * PITCH + R_IN + N_T * PITCH) / 2
        return {"start": (R_OUT, theta_c - half_deg(R_OUT, 0)), "end": (R_OUT, theta_c + half_deg(R_OUT, 0)),
                "centre": (r_ctr, theta_c)}

    def coil(self, g):
        theta_c = g * SECTOR
        geo = None
        for layer in ODD:
            geo = self.spiral(theta_c, layer, mirrored=False)
        for layer in EVEN:
            self.spiral(theta_c, layer, mirrored=True)
        # centre via joins odd and even layers
        self.via(pt(*geo["centre"]))
        # terminal vias just outside the outer corners, 0.25 mm inboard of the coil edge
        d_in = math.degrees(0.25 / R_VIA_T)
        t1 = (R_VIA_T, geo["start"][1] + d_in)
        t2 = (R_VIA_T, geo["end"][1] - d_in)
        for layer in ODD:
            self.track(pt(*geo["start"]), pt(*t1), TRACE, layer)
        for layer in EVEN:
            self.track(pt(*geo["end"]), pt(*t2), TRACE, layer)
        self.via(pt(*t1)); self.via(pt(*t2))
        return {"T1": t1, "T2": t2, "centre": geo["centre"]}

    # ---- interconnect ---------------------------------------------------------
    def stub_to_ring(self, t, ring):
        for layer in RING_LAYERS[ring]:
            self.track(pt(*t), pt(R_RING[ring], t[1]), STUB_W, layer, "stub")

    def stub_to_M(self, t, phase):
        self.track(pt(*t), pt(R_M, t[1]), STUB_W, M_LAYER[phase], "stub")

    def jumper_to_N(self, t, gap_angle):
        # radially inward through the coil gap on the N layers
        for layer in NM_LAYERS:
            self.track(pt(*t), pt(R_VIA_T, gap_angle), JUMPER_W, layer, "stub")
            self.track(pt(R_VIA_T, gap_angle), pt(R_N, gap_angle), JUMPER_W, layer, "jumper")

    def build(self):
        terms = [self.coil(g) for g in range(N_COILS)]
        # rings
        for ring, r in R_RING.items():
            for layer in RING_LAYERS[ring]:
                self.arc(r, 0, 360, RING_W, layer, "ring")
        for layer in NM_LAYERS:
            self.arc(R_N, 0, 360, RING_W, layer, "ring")
        # per repeat
        for rep in range(N_COILS // 12):
            base = 12 * rep
            for ph in "ABC":
                p1a, p1b = PAIR1[ph]           # (+, -) : ring -> pair1 -> M
                p2a, p2b = PAIR2[ph]           # (+, -) : M -> pair2 -> N
                T = lambda k: terms[base + k]
                # ring side
                self.stub_to_ring(T(p1a)["T1"], ph)
                self.stub_to_ring(T(p1b)["T2"], ph)
                # M node: pair1 exits, pair2 entries
                m_terms = [T(p1a)["T2"], T(p1b)["T1"], T(p2a)["T1"], T(p2b)["T2"]]
                for t in m_terms:
                    self.stub_to_M(t, ph)
                angs = [t[1] for t in m_terms]
                self.arc(R_M, min(angs), max(angs), M_W, M_LAYER[ph], "marc")
                self.marc_len[ph] = self.marc_len.get(ph, 0.0) + math.radians(max(angs) - min(angs)) * R_M
                # star: pair2 exits -> N ring through the nearest gap
                for k, t in ((p2a, T(p2a)["T2"]), (p2b, T(p2b)["T1"])):
                    g = base + k
                    gap = (g + 0.5) * SECTOR if t[1] > g * SECTOR else (g - 0.5) * SECTOR
                    self.jumper_to_N(t, gap)
        # phase pads at 88, 208, 328 deg, on the ring layers
        pads = []
        for ph, ang in zip("ABC", (88.0, 208.0, 328.0)):
            pads.append((ph, pt(R_PAD, ang)))
            for layer in RING_LAYERS[ph]:
                self.track(pt(R_RING[ph], ang), pt(R_PAD, ang), STUB_W, layer, "stub")
        self.pads(pads)
        # outline: outer circle and bore
        self.circle(R_BOARD, pcbnew.Edge_Cuts)
        self.circle(R_BORE, pcbnew.Edge_Cuts)
        return terms


def geometry_report(bld):
    per_layer_coil = {str(l): v for (l, k), v in bld.length.items() if k == "coil"}
    coil_len_total = sum(v for (l, k), v in bld.length.items() if k == "coil")
    other = sum(v for (l, k), v in bld.length.items() if k != "coil")
    return {
        "coils": N_COILS, "pole_pairs": POLE_PAIRS, "layers": N_LAYERS, "copper_oz": COPPER_OZ,
        "t_board_mm": board_thickness(N_LAYERS, COPPER_OZ), "turns_per_layer": N_T,
        "series_turns_per_coil": 2 * N_T, "series_turns_per_phase": 4 * N_T, "parallel_paths_per_phase": 6 * 2 * 4,
        "trace_mm": TRACE, "space_mm": SPACE, "r_in_mm": R_IN, "r_out_mm": R_OUT,
        "coil_track_mm_total": coil_len_total, "coil_track_mm_per_coil_layer": coil_len_total / (N_COILS * N_LAYERS),
        "interconnect_track_mm": other, "vias": 3 * N_COILS, "board_od_mm": 2 * R_BOARD,
        "repeats": N_COILS // 12, "gap_mm": GAP, "ring_w_mm": RING_W, "m_w_mm": M_W, "jumper_w_mm": JUMPER_W,
        "r_ring_mm": R_RING, "r_m_mm": R_M, "r_n_mm": R_N, "r_via_t_mm": R_VIA_T,
        "marc_mm_per_phase": {k: v for k, v in bld.marc_len.items()},
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coils", type=int, default=N_COILS)
    ap.add_argument("--pp", type=int, default=POLE_PAIRS)
    ap.add_argument("--turns", type=int, default=N_T)
    ap.add_argument("--trace", type=float, default=None, help="trace width; default fills the leg")
    ap.add_argument("--gap", type=float, default=GAP, help="coil-to-coil gap at the legs, mm")
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--oz", type=float, default=COPPER_OZ, help="copper weight per layer, oz")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    N_COILS, POLE_PAIRS, N_T, OUT, GAP, COPPER_OZ = a.coils, a.pp, a.turns, a.out, a.gap, a.oz
    set_layers(a.layers)
    SECTOR = 360.0 / N_COILS
    leg = (2 * math.pi * R_IN / N_COILS - GAP) / 2
    TRACE = a.trace if a.trace else math.floor((leg / N_T - SPACE) * 200) / 200
    PITCH = TRACE + SPACE
    assert N_T * PITCH <= leg + 1e-9, f"{N_T} turns of {PITCH:.3f} mm do not fit the {leg:.2f} mm leg"
    bld = Builder()
    bld.build()
    pcbnew.SaveBoard(OUT, bld.b)
    rep = geometry_report(bld)
    with open(os.path.join(os.path.dirname(os.path.abspath(OUT)), "geometry.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", OUT)
    for k, v in rep.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")

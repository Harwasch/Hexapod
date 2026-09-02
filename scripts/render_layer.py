#!/usr/bin/env python3
"""Raster render of one copper layer of a KiCad board, straight from the pcbnew
object model (no SVG rasteriser is installed in this environment).

    python3 scripts/render_layer.py hw/stator/variants/1s-opt/stator.kicad_pcb F.Cu docs/design/actuator/1s-opt-F_Cu.png [dpi]

Tracks, arcs, vias, pads and the board outline are drawn at their true widths
in millimetres; the colour is KiCad's default F.Cu red.  Read-only: the board
file is never written.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

HW_PY = "/opt/hw-py/bin/python"      # has matplotlib; the system python3 has pcbnew.  Stage 1 extracts, stage 2 draws.


def extract(board_path, layer_name):
    import pcbnew
    b = pcbnew.LoadBoard(board_path)
    layer = b.GetLayerID(layer_name)
    bb = b.GetBoardEdgesBoundingBox()
    prim = dict(front=(layer == pcbnew.F_Cu), bbox=[bb.GetX() / 1e6, bb.GetY() / 1e6, bb.GetWidth() / 1e6, bb.GetHeight() / 1e6], segs=[], circles=[], holes=[], edges=[])
    for t in b.GetTracks():
        if t.Type() == pcbnew.PCB_VIA_T:
            if t.IsOnLayer(layer):
                p = t.GetPosition()
                prim["circles"].append([p.x / 1e6, p.y / 1e6, t.GetWidth(layer) / 2e6])
                prim["holes"].append([p.x / 1e6, p.y / 1e6, t.GetDrillValue() / 2e6])
            continue
        if t.GetLayer() != layer:
            continue
        if t.Type() == pcbnew.PCB_ARC_T:
            c = t.GetCenter(); r = t.GetRadius() / 1e6
            a0 = math.atan2(t.GetStart().y - c.y, t.GetStart().x - c.x)
            sweep = math.radians(t.GetAngle().AsDegrees())
            n = max(2, int(abs(sweep) * r / 0.5) + 1)
            pts = [(c.x / 1e6 + r * math.cos(a0 + sweep * i / n), c.y / 1e6 + r * math.sin(a0 + sweep * i / n)) for i in range(n + 1)]
            prim["segs"] += [[pts[i], pts[i + 1], t.GetWidth() / 1e6] for i in range(n)]
        else:
            s, e = t.GetStart(), t.GetEnd()
            prim["segs"].append([[s.x / 1e6, s.y / 1e6], [e.x / 1e6, e.y / 1e6], t.GetWidth() / 1e6])
    for fp in b.GetFootprints():
        for pad in fp.Pads():
            if pad.IsOnLayer(layer):
                p = pad.GetPosition()
                prim["circles"].append([p.x / 1e6, p.y / 1e6, pad.GetSize().x / 2e6])
                prim["holes"].append([p.x / 1e6, p.y / 1e6, pad.GetDrillSize().x / 2e6])
    for d in b.GetDrawings():
        if d.GetLayer() == pcbnew.Edge_Cuts and d.GetShape() == pcbnew.SHAPE_T_CIRCLE:
            c = d.GetCenter()
            prim["edges"].append([c.x / 1e6, c.y / 1e6, d.GetRadius() / 1e6])
    return prim


def draw(prim, out, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle
    CU = "#C83434" if prim["front"] else "#4D7FC4"
    x0, y0, w, h = prim["bbox"]
    fig = plt.figure(figsize=(w / 25.4, h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x0 + w); ax.set_ylim(y0 + h, y0)          # KiCad y points down
    ax.set_aspect("equal"); ax.set_axis_off()
    mm_per_pt = 25.4 / 72.0                                    # the canvas is the board at 1:1, so a track's width in points is its mm / 0.353
    ax.add_collection(LineCollection([(a, b_) for a, b_, _ in prim["segs"]], linewidths=[wd / mm_per_pt for _, _, wd in prim["segs"]],
                                     colors=CU, capstyle="round", joinstyle="round"))
    for x, y, r in prim["circles"]:
        ax.add_patch(Circle((x, y), r, color=CU, lw=0))
    for x, y, r in prim["holes"]:
        ax.add_patch(Circle((x, y), r, color="white", lw=0))
    for x, y, r in prim["edges"]:
        ax.add_patch(Circle((x, y), r, fill=False, ec="#444", lw=0.6))
    fig.savefig(out, dpi=dpi, facecolor="white")
    from PIL import Image                                      # palette PNG: a quarter of the size, the review page refuses images over ~400 kB
    Image.open(out).convert("RGB").quantize(64, dither=Image.Dither.NONE).save(out, optimize=True)
    print("wrote", out, f"{w:.1f} x {h:.1f} mm at {dpi} dpi")


if __name__ == "__main__":
    if sys.argv[1] == "--draw":
        draw(json.load(open(sys.argv[2])), sys.argv[3], int(sys.argv[4]))
        sys.exit(0)
    board_path, layer_name, out = sys.argv[1], sys.argv[2], sys.argv[3]
    dpi = int(sys.argv[4]) if len(sys.argv) > 4 else 160
    prim = extract(board_path, layer_name)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(prim, f)
    subprocess.run([HW_PY, os.path.abspath(__file__), "--draw", f.name, out, str(dpi)], check=True)
    os.unlink(f.name)

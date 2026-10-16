"""Parametric CAD of the arm (build123d), from the same design the optimizer picked.

    uv run python -m armdesign.cad   # cad/: STEP assembly, STL per printed part, GLB per link

Every link is built in its MuJoCo body frame (see mjcf.py), so the viewer can drive the
CAD with the simulation's body poses directly. Units are millimetres here; the GLB files
are written in metres.

Construction: aluminium 25.4 x 1.5 mm tube links, printed housings (ASA or PETG-CF), the
actuators bought in. The yaw actuator is recessed into the hexapod's deck so the shoulder
axis can sit low; the shoulder and elbow motors sit face to face on the turret, one each
side, and the elbow motor drives the forearm through a crank and a rod (the
parallelogram). Bought-in parts are modelled as envelopes with their real mounting faces.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from build123d import (
    Align,
    Box,
    Color,
    Compound,
    Cylinder,
    Part,
    Polygon,
    Pos,
    Rot,
    SlotCenterToCenter,
    export_gltf,
    export_step,
    export_stl,
    extrude,
)

from armdesign.evaluate import load_tools
from armdesign.mjcf import CRANK, ArmActuators
from armdesign.model import ROOT, ArmDesign, Hexapod

OUT = ROOT / "cad"  # STEP + STL, committed
GLB = ROOT / "cad" / "glb"  # per-link meshes in the simulation's frames; the viewer embeds them

TUBE_OD = 25.4
TUBE_ID = 22.4
CHEEK_T = 8.0
PRINT_DENSITY = 1.30e-3  # g/mm^3, ASA / PETG-CF solid
PRINT_FILL = 0.6  # perimeters + 30% gyroid, effective
COLORS = {
    "printed": Color(0.93, 0.94, 0.95),
    "accent": Color(0.97, 0.55, 0.16),
    "actuator": Color(0.16, 0.18, 0.21),
    "actuator_face": Color(0.55, 0.58, 0.62),
    "tube": Color(0.72, 0.74, 0.77),
    "carbon": Color(0.08, 0.08, 0.09),
    "steel": Color(0.6, 0.62, 0.65),
    "tpu": Color(0.2, 0.62, 0.92),
    "glass": Color(0.05, 0.05, 0.08),
    "servo": Color(0.1, 0.1, 0.12),
}


def _c(shape: Part, key: str, label: str) -> Part:
    shape.color = COLORS[key]
    shape.label = label
    return shape


def cyl_y(r: float, y0: float, y1: float, x: float = 0, z: float = 0) -> Part:
    """A cylinder on an axis parallel to y, from y0 to y1."""
    lo, hi = min(y0, y1), max(y0, y1)
    return Pos(x, lo, z) * (Rot(-90, 0, 0) * Cylinder(r, hi - lo, align=(Align.CENTER, Align.CENTER, Align.MIN)))


def cyl_x(r: float, x0: float, x1: float, y: float = 0, z: float = 0) -> Part:
    lo, hi = min(x0, x1), max(x0, x1)
    return Pos(lo, y, z) * (Rot(0, 90, 0) * Cylinder(r, hi - lo, align=(Align.CENTER, Align.CENTER, Align.MIN)))


def cyl_z(r: float, z0: float, z1: float, x: float = 0, y: float = 0) -> Part:
    lo, hi = min(z0, z1), max(z0, z1)
    return Pos(x, y, lo) * Cylinder(r, hi - lo, align=(Align.CENTER, Align.CENTER, Align.MIN))


def bolt_circle(r_pcd: float, n: int, hole_r: float, make: callable) -> Part:
    holes = None
    for k in range(n):
        a = 2 * math.pi * k / n
        h = make(r_pcd * math.cos(a), r_pcd * math.sin(a), hole_r)
        holes = h if holes is None else holes + h
    return holes


# --- bought-in envelopes ----------------------------------------------------------------


def qdd_y(dia: float, length: float, y_face: float, toward: int) -> list[Part]:
    """A round actuator on the y axis whose output face is at y_face, facing `toward` (+1/-1).
    Returns the dark body and the bright output flange."""
    r = dia / 2
    y_back = y_face - toward * length
    body = cyl_y(r, y_back, y_face - toward * 3)
    body -= cyl_y(r * 0.35, y_back - toward * 1, y_back + toward * 2)  # rear cap recess
    face = cyl_y(r * 0.58, y_face - toward * 3, y_face)
    face -= bolt_circle(r * 0.42, 6, 1.7, lambda a, b, hr: cyl_y(hr, y_face - toward * 4, y_face + toward, a, b))
    boss = Pos(0, (y_back + y_face) / 2, -r) * Box(14, 10, 8)
    return [_c(body + boss, "actuator", "actuator"), _c(face, "actuator_face", "actuator output")]


def qdd_z(dia: float, length: float, z_face: float) -> list[Part]:
    r = dia / 2
    body = cyl_z(r, z_face - length, z_face - 3)
    face = cyl_z(r * 0.58, z_face - 3, z_face)
    return [_c(body, "actuator", "actuator"), _c(face, "actuator_face", "actuator output")]


def servo_box(center: tuple[float, float, float], axis: str) -> list[Part]:
    """A standard 45 x 24.7 x 35 mm bus servo with its horn on `axis` (+x or +y)."""
    x, y, z = center
    if axis == "x":
        body = Pos(x, y, z) * Box(35, 24.7, 45)
        horn = cyl_x(11, x + 17.5, x + 21.5, y, z + 11)
    else:
        body = Pos(x, y, z) * Box(45, 35, 24.7)
        horn = cyl_y(11, y + 17.5, y + 21.5, x + 11, z)
    return [_c(body, "servo", "servo"), _c(horn, "actuator_face", "servo horn")]


def actuator_parts(act, frame_y_face: float, toward: int) -> list[Part]:
    if act.kind == "servo":
        return servo_box((0, frame_y_face - toward * 17.5, 0), "y")
    return qdd_y(act.dia * 1000, act.length * 1000, frame_y_face, toward)


def tube_x(x0: float, x1: float, y: float = 0, z: float = 0) -> Part:
    return _c(cyl_x(TUBE_OD / 2, x0, x1, y, z) - cyl_x(TUBE_ID / 2, x0 - 1, x1 + 1, y, z), "tube", "aluminium tube")


# --- printed parts ------------------------------------------------------------------------


@dataclass
class Layout:
    """Derived dimensions, mm."""

    l1: float
    l2: float
    ped: float
    wl: float
    sh_dia: float
    sh_len: float
    el_dia: float
    el_len: float
    yaw_dia: float
    yaw_len: float
    wp_dia: float
    wp_len: float
    wp_servo: bool
    para: bool
    spring: bool
    cheek_in: float = 20.0  # inner faces of the turret cheeks at +-cheek_in

    @property
    def cheek_out(self) -> float:
        return self.cheek_in + CHEEK_T

    @property
    def head_r(self) -> float:
        return max(self.sh_dia, self.el_dia) / 2 + 4


def layout(design: ArmDesign, acts: ArmActuators) -> Layout:
    mm = 1000
    wp = acts.wrist_pitch
    return Layout(
        l1=design.l1 * mm,
        l2=design.l2 * mm,
        ped=design.pedestal * mm,
        wl=design.wrist_len * mm,
        sh_dia=acts.shoulder.dia * mm,
        sh_len=acts.shoulder.length * mm,
        el_dia=acts.elbow.dia * mm,
        el_len=acts.elbow.length * mm,
        yaw_dia=acts.yaw.dia * mm,
        yaw_len=acts.yaw.length * mm,
        wp_dia=wp.dia * mm,
        wp_len=wp.length * mm,
        wp_servo=wp.kind == "servo",
        para=design.arch == "parallelogram",
        spring=design.spring,
    )


def deck_ring(lo: Layout) -> Part:
    """Seal ring on the deck around the recessed yaw actuator (body frame at the mount)."""
    r_out = lo.yaw_dia / 2 + 14
    # Low (2 mm) so the turret's motors can sweep over it on a short pedestal.
    ring = cyl_z(r_out, 0, 2) - cyl_z(lo.yaw_dia / 2 * 0.62 + 1.5, -1, 3)
    ring -= bolt_circle(lo.yaw_dia / 2 + 7, 6, 2.2, lambda a, b, r: cyl_z(r, -1, 3, a, b))
    return ring


def turret_plate(lo: Layout) -> Part:
    w = 2 * lo.cheek_out
    plate = Pos(0, 0, 2.5) * Box(lo.head_r * 2 - 6, w, 7, align=(Align.CENTER, Align.CENTER, Align.MIN))
    plate -= bolt_circle(lo.yaw_dia / 2 * 0.42, 6, 1.7, lambda a, b, r: cyl_z(r, 0, 20, a, b))
    plate -= cyl_z(6, 0, 20)  # cable pass-through on the yaw axis
    return plate


def cheek(lo: Layout, side: int) -> Part:
    """One turret cheek (side +1 or -1): a plate in xz from the turret plate up round the
    shoulder axis, carrying one motor's flange."""
    y0 = side * lo.cheek_in
    y1 = side * lo.cheek_out
    r = lo.head_r
    prof = cyl_y(r, y0, y1, 0, lo.ped) + Pos(0, (y0 + y1) / 2, (lo.ped + 5.5) / 2 + 2) * Box(
        2 * r - 6, CHEEK_T, lo.ped - 5.5 + 6
    )
    dia = lo.sh_dia if side > 0 else lo.el_dia
    prof -= cyl_y(dia / 2 * 0.62 + 1.0, y0 - 1, y1 + 1, 0, lo.ped)
    prof -= bolt_circle(dia / 2 * 0.8, 6, 1.7, lambda a, b, hr: cyl_y(hr, y0 - 1, y1 + 1, a, lo.ped + b))
    # Lightening window below the axis.
    if lo.ped > 55:
        prof -= Pos(0, (y0 + y1) / 2, (lo.ped - r * 0.6 + 14) / 2) * Box(r, CHEEK_T + 2, max(lo.ped - r * 0.6 - 14, 4))
    return prof


def spring_mast(lo: Layout) -> Part:
    """Pulley mast above the shoulder axis (shoulder spring) and a lug below it (elbow
    spring), both at y = 0 between the cheeks... carried by a bridge across the cheeks."""
    a = 50.0
    bridge = Pos(-lo.head_r + 8, 0, lo.ped + a / 2) * Box(10, 2 * lo.cheek_in, a + 16)
    pulley = cyl_y(9, -4, 4, -lo.head_r + 8, lo.ped + a)
    return bridge + pulley


def upper_hub(lo: Layout) -> Part:
    """Root of the upper arm on the shoulder motor's output, with the tube socket."""
    hub = cyl_y(lo.sh_dia / 2 * 0.58, 2, lo.cheek_in - 1)
    sock = cyl_x(17, 8, 48, 10)
    web = Pos(18, 10, 0) * Box(24, 16, 30)
    part = hub + sock + web
    part -= cyl_x(TUBE_OD / 2 + 0.15, 14, 50, 10)
    part -= bolt_circle(lo.sh_dia / 2 * 0.42, 6, 1.7, lambda a, b, r: cyl_y(r, 0, lo.cheek_in + 1, a, b))
    part -= Pos(38, 10, 17) * Box(22, 2, 10)  # clamp slot
    return part


def elbow_housing(lo: Layout) -> Part:
    x = lo.l1
    hub = cyl_y(19, 2, 18, x, 0)
    sock = cyl_x(17, x - 48, x - 10, 10)
    part = hub + sock + Pos(x - 14, 10, 0) * Box(16, 16, 30)
    part -= cyl_y(12, 1, 19, x, 0)  # 6802 bearing seat
    part -= cyl_x(TUBE_OD / 2 + 0.15, x - 50, x - 14, 10)
    if lo.spring:
        part += Pos(110, 10, 18) * Box(14, 10, 8)  # spring cable lug on the tube clamp
    return part


def crank(lo: Layout) -> Part:
    """On the elbow motor's output, the crank arm points back along -x (the forearm's
    direction, reversed) to the rod pin."""
    hub = cyl_y(lo.el_dia / 2 * 0.58, -lo.cheek_in + 1, -6)
    arm = Pos(0, -12, 0) * (Rot(90, 0, 0) * extrude(SlotCenterToCenter(CRANK * 1000, 22), 6, both=True))
    arm = Pos(-CRANK * 500, 0, 0) * arm
    part = hub + arm
    part -= cyl_y(3.1, -20, 0, -CRANK * 1000, 0)
    part -= bolt_circle(lo.el_dia / 2 * 0.42, 6, 1.7, lambda a, b, r: cyl_y(r, -lo.cheek_in - 1, 0, a, b))
    return part


def rod_parts(lo: Layout) -> list[Part]:
    """Carbon rod with a rod-end eye at each pin (rod frame: from the crank pin along +x)."""
    rod = _c(cyl_x(5, 8, lo.l1 - 8, -12), "carbon", "carbon rod")
    eyes = []
    for x in (0.0, lo.l1):
        eye = cyl_y(8, -17, -7, x, 0) - cyl_y(3.1, -18, -6, x, 0)
        eyes.append(_c(eye, "steel", "rod end"))
    return [rod, *eyes]


def fore_hub(lo: Layout) -> Part:
    """Forearm root on the elbow axle, with the rear lever out to the rod pin."""
    hub = cyl_y(17, -16, 1.5)
    sock = cyl_x(17, 8, 46, -5)
    lever = Pos(0, -12, 0) * (Rot(90, 0, 0) * extrude(SlotCenterToCenter(CRANK * 1000, 20), 6, both=True))
    lever = Pos(-CRANK * 500, 0, 0) * lever
    part = hub + sock + Pos(14, -6, 0) * Box(20, 16, 28) + lever
    part -= cyl_y(5.1, -17, 3)
    part -= cyl_y(3.1, -20, 0, -CRANK * 1000, 0)
    part -= cyl_x(TUBE_OD / 2 + 0.15, 12, 48, -5)
    return part


def wrist_clamp(lo: Layout) -> Part:
    """End of the forearm holding the wrist-pitch motor (axis y, centred on W)."""
    x = lo.l2
    if lo.wp_servo:
        ring = Pos(x, 0, 0) * Box(52, 30, 42) - Pos(x, 0, 0) * Box(46, 32, 36)
    else:
        r = lo.wp_dia / 2
        ring = cyl_y(r + 4, -lo.wp_len / 2 + 4, lo.wp_len / 2 - 8, x, 0) - cyl_y(r + 0.2, -lo.wp_len, lo.wp_len, x, 0)
    sock = cyl_x(16.5, x - 70, x - 22, -5)
    part = ring + sock + Pos(x - 28, -5, 0) * Box(14, 24, 28)
    part -= cyl_x(TUBE_OD / 2 + 0.15, x - 72, x - 26, -5)
    return part


def wrist_bracket(lo: Layout, roll_servo: bool) -> Part:
    """On the wrist motor's output face (outboard, +y), wraps round to hold the tool-roll
    actuator on the tool axis."""
    yf = (17.5 + 4) if lo.wp_servo else lo.wp_len / 2
    plate = cyl_y(lo.wp_dia / 2 * 0.62 + 2, yf, yf + 5)
    arm = Pos(22, yf + 2.5, 0) * Box(40, 5, 34)
    bridge = Pos(40, yf / 2 + 1.5, 0) * Box(10, yf + 3, 34)
    cradle = Pos(46, 0, 0) * Box(40, 32, 52) - Pos(46, 0, 0) * Box(36, 26, 46)
    part = plate + arm + bridge + cradle
    part -= bolt_circle(lo.wp_dia / 2 * 0.42, 6, 1.7, lambda a, b, r: cyl_y(r, yf - 1, yf + 6, a, b))
    return part


def camera_housing(lo: Layout) -> list[Part]:
    x = lo.wl - 22
    shell = Pos(x, 0, 40) * Box(26, 30, 16) - Pos(x + 2, 0, 40) * Box(24, 26, 12)
    lens = _c(cyl_x(5, x + 13, x + 16, 0, 40), "glass", "camera lens")
    tof = _c(Pos(x + 13.5, 9, 40) * Box(2, 6, 4), "glass", "ToF window")
    return [_c(shell, "accent", "camera housing (printed)"), lens, tof]


def changer_arm_side(lo: Layout) -> Part:
    x0 = lo.wl - 20
    disk = cyl_x(31, x0, x0 + 10)
    disk -= bolt_circle(14, 6, 1.0, lambda a, b, r: cyl_x(r, x0 + 4, x0 + 11, a, b))  # pogo pins
    for k in range(3):
        a = 2 * math.pi * k / 3
        disk += Pos(x0 + 8, 33 * math.cos(a), 33 * math.sin(a)) * Box(4, 8, 8)  # bayonet lugs
    return disk


def changer_tool_side(lo: Layout) -> Part:
    x0 = lo.wl - 10
    disk = cyl_x(31, x0, x0 + 8) - cyl_x(25, x0 - 1, x0 + 3)
    return disk


def gripper_palm(lo: Layout, tool_len: float) -> list[Part]:
    x0 = lo.wl - 2
    palm = Pos(x0 + 22, 0, 0) * Box(44, 76, 42)
    palm -= Pos(x0 + 24, 0, 0) * Box(40, 50, 36)
    rail = Pos(x0 + 46, 0, 0) * Box(6, 120, 14)
    servo = servo_box((x0 + 22, 0, 0), "x")
    return [_c(palm + rail, "printed", "gripper palm (printed)"), *servo]


def finray_finger(side: int, length: float) -> Part:
    """A fin-ray finger in its slide frame: flexible TPU triangle with ribs. Its gripping
    face is the inner (towards y = 0) edge, which wraps round a fruit."""
    inner = 8.0 * side
    base_w = 24.0
    pts_outer = [(0, inner), (length, inner + side * 2), (0, inner + side * base_w)]
    tri = extrude(Polygon(*pts_outer, align=None), 15, both=True)
    hollow_pts = [(6, inner + side * 3), (length - 12, inner + side * 3.6), (6, inner + side * (base_w - 4))]
    tri -= extrude(Polygon(*hollow_pts, align=None), 16, both=True)
    for k in range(1, 5):
        x = 6 + k * (length - 18) / 5
        y_in = inner + side * 3
        y_out = inner + side * (base_w - (base_w - 2) * x / length)
        rib = Pos(x, (y_in + y_out) / 2, 0) * (Rot(0, 0, -side * 18) * Box(2.2, abs(y_out - y_in) + 2, 30))
        tri += rib
    return tri


def tool_shape(tool_id: str, lo: Layout, length: float) -> list[Part]:
    """A non-gripper tool behind its changer plate, in the flange frame."""
    x0 = lo.wl + 8
    tip = lo.wl + length * 1000
    if tool_id == "probe":
        body = Pos(x0 + 15, 0, 0) * Box(30, 40, 40)
        shaft = cyl_x(6, x0 + 30, tip - 8)
        point = cyl_x(3, tip - 8, tip)
        return [_c(body, "printed", "probe holder"), _c(shaft + point, "steel", "probe rod")]
    if tool_id == "sprayer":
        body = Pos(x0 + 12, 0, 0) * Box(24, 34, 34)
        nozzle = cyl_x(5, x0 + 24, tip - 2) + cyl_x(2.5, tip - 2, tip)
        return [_c(body, "printed", "spray head"), _c(nozzle, "accent", "nozzle")]
    body = Pos(x0 + 20, 0, 0) * Box(40, 50, 50)
    return [_c(body, "printed", tool_id)]


# --- assembly --------------------------------------------------------------------------------


def link_parts(design: ArmDesign, acts: ArmActuators, hx: Hexapod) -> dict[str, list[Part]]:
    """Every part, grouped by the MuJoCo body it rides on, in that body's frame (mm)."""
    lo = layout(design, acts)
    deck = hx.deck_z * 1000
    mx = design.mount_x * 1000
    tools = load_tools()
    gr = tools["gripper"]
    links: dict[str, list[Part]] = {}
    # Fixed to the hexapod body (body frame): recessed yaw actuator and the deck seal ring.
    yaw = qdd_z(lo.yaw_dia, lo.yaw_len, 0.0) if acts.yaw.kind != "servo" else servo_box((0, 0, -17.5), "x")
    links["body"] = [Pos(mx, 0, deck) * p for p in [*yaw, _c(deck_ring(lo), "printed", "deck seal ring (printed)")]]
    tur = [
        _c(turret_plate(lo), "printed", "turret plate (printed)"),
        _c(cheek(lo, 1), "printed", "turret cheek, shoulder side (printed)"),
        _c(cheek(lo, -1), "printed", "turret cheek, elbow side (printed)"),
    ]
    sh = [Pos(0, 0, lo.ped) * p for p in qdd_y(lo.sh_dia, lo.sh_len, lo.cheek_out, -1)]
    tur += sh
    if lo.para:
        tur += [Pos(0, 0, lo.ped) * p for p in qdd_y(lo.el_dia, lo.el_len, -lo.cheek_out, 1)]
    if lo.spring:
        tur.append(_c(spring_mast(lo), "accent", "spring pulley mast (printed)"))
    links["turret"] = tur
    up = [
        _c(upper_hub(lo), "printed", "upper arm hub (printed)"),
        tube_x(14, lo.l1 - 14, 10),
        _c(elbow_housing(lo), "printed", "elbow housing (printed)"),
        _c(cyl_y(5, -16, 18, lo.l1, 0), "steel", "elbow axle"),
    ]
    if not lo.para:
        up += [Pos(lo.l1, 0, 0) * p for p in qdd_y(lo.el_dia, lo.el_len, 20, 1)]
    links["upper"] = up
    if lo.para:
        links["crank"] = [_c(crank(lo), "accent", "crank (printed)")]
        links["rod"] = rod_parts(lo)
    fore = [
        _c(fore_hub(lo), "accent" if lo.para else "printed", "forearm hub and rear lever (printed)"),
        tube_x(12, lo.l2 - 28, -5),
        _c(wrist_clamp(lo), "printed", "wrist clamp (printed)"),
    ]
    if lo.wp_servo:
        fore += servo_box((lo.l2, 0, 0), "y")
    else:
        fore += [Pos(lo.l2, 0, 0) * p for p in qdd_y(lo.wp_dia, lo.wp_len, lo.wp_len / 2, 1)]
    links["fore"] = fore
    roll_servo = acts.tool_roll.kind == "servo"
    wr = [_c(wrist_bracket(lo, roll_servo), "printed", "wrist bracket (printed)"), *camera_housing(lo)]
    if roll_servo:
        wr += servo_box((46, 0, 0), "x")
    else:
        wr += [Rot(0, 0, -90) * p for p in qdd_y(acts.tool_roll.dia * 1000, acts.tool_roll.length * 1000, 66, 1)]
    links["wrist"] = wr
    links["flange"] = [_c(changer_arm_side(lo), "accent", "tool changer, arm side (printed)")]
    links["tool_gripper"] = [
        _c(changer_tool_side(lo), "printed", "tool changer, tool side (printed)"),
        *gripper_palm(lo, gr.length),
    ]
    fl = (gr.length - 0.04) * 1000
    links["finger_l"] = [_c(finray_finger(1, fl), "tpu", "fin-ray finger (TPU)")]
    links["finger_r"] = [_c(finray_finger(-1, fl), "tpu", "fin-ray finger (TPU)")]
    for tid in ("probe", "sprayer"):
        links[f"tool_{tid}"] = [
            _c(changer_tool_side(lo), "printed", "tool changer, tool side (printed)"),
            *tool_shape(tid, lo, tools[tid].length),
        ]
    return links


PRINTED_KEYS = ("printed", "accent", "tpu")


def is_printed(p: Part) -> bool:
    return "(printed)" in p.label or "(TPU)" in p.label


def build(design: ArmDesign, acts: ArmActuators, hx: Hexapod) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stl").mkdir(exist_ok=True)
    GLB.mkdir(parents=True, exist_ok=True)
    links = link_parts(design, acts, hx)
    report = {"links": {}, "printed": []}
    seen: dict[str, int] = {}
    for name, parts in links.items():
        comp = Compound(children=parts, label=name)
        export_gltf(comp, str(GLB / f"{name}.glb"), binary=True)
        report["links"][name] = [p.label for p in parts]
        for p in parts:
            if is_printed(p):
                base = p.label.replace(" (printed)", "").replace(" (TPU)", "").replace(",", "").replace(" ", "_")
                k = seen.get(base, 0)
                seen[base] = k + 1
                fname = f"{base}{'_' + str(k + 1) if k else ''}.stl"
                export_stl(p, str(OUT / "stl" / fname))
                grams = p.volume * PRINT_DENSITY * PRINT_FILL
                report["printed"].append(
                    {"part": p.label, "file": f"stl/{fname}", "grams": round(grams, 1), "link": name}
                )
    # A reference-pose STEP of the whole arm (gripper on), for checking fit in any CAD tool.
    report["step"] = "arm_assembly.step"
    export_step(assembly_at(design, links, hx), str(OUT / "arm_assembly.step"))
    report["printed_grams"] = round(sum(p["grams"] for p in report["printed"]), 1)
    (OUT / "cad.json").write_text(json.dumps(report, indent=1))
    return report


def assembly_at(
    design: ArmDesign, links: dict[str, list[Part]], hx: Hexapod, th2: float = 35.0, th3: float = -40.0
) -> Compound:
    """Place the links at a reference pose (degrees, absolute) in the body frame."""
    mx, deck, ped = design.mount_x * 1000, hx.deck_z * 1000, design.pedestal * 1000
    l1, l2 = design.l1 * 1000, design.l2 * 1000
    q4 = -50.0 - th3  # tool pointing 50 degrees down
    out = list(links["body"])
    t = Pos(mx, 0, deck)
    out += [t * p for p in links["turret"]]
    up = t * Pos(0, 0, ped) * Rot(0, -th2, 0)
    out += [up * p for p in links["upper"]]
    fore = t * Pos(0, 0, ped) * Rot(0, -th2, 0) * Pos(l1, 0, 0) * Rot(0, -(th3 - th2), 0)
    out += [fore * p for p in links["fore"]]
    if "crank" in links:
        cr = t * Pos(0, 0, ped) * Rot(0, -th3, 0)
        out += [cr * p for p in links["crank"]]
        rod = cr * Pos(-CRANK * 1000, 0, 0) * Rot(0, -(th2 - th3), 0)
        out += [rod * p for p in links["rod"]]
    wr = fore * Pos(l2, 0, 0) * Rot(0, -q4, 0)
    for k in ("wrist", "flange", "tool_gripper"):
        out += [wr * p for p in links[k]]
    gr = load_tools()["gripper"]
    for k in ("finger_l", "finger_r"):
        out += [wr * Pos(design.wrist_len * 1000 + 40, 0, 0) * p for p in links[k]]
    _ = gr
    return Compound(children=out, label="arm")


def main() -> None:
    from armdesign.sim import load_selected

    design, acts = load_selected()
    rep = build(design, acts, Hexapod.load())
    print(f"{len(rep['printed'])} printed parts, {rep['printed_grams']} g of filament")


if __name__ == "__main__":
    main()

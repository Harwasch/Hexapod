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
    Circle,
    Color,
    Compound,
    Cylinder,
    Helix,
    Part,
    Plane,
    Polygon,
    Pos,
    Rot,
    SlotCenterToCenter,
    export_gltf,
    export_step,
    export_stl,
    extrude,
    sweep,
)

from armdesign.evaluate import load_tools
from armdesign.mjcf import CRANK, ArmActuators
from armdesign.model import ROOT, ArmDesign, Hexapod
from armdesign.parts import Stage, load_stages

OUT = ROOT / "cad"  # STEP + STL, committed
GLB = ROOT / "cad" / "glb"  # per-link meshes in the simulation's frames; the viewer embeds them

TUBE_OD = 25.4
TUBE_ID = 22.4
CHEEK_T = 8.0
# Capstan stages (docs/research/reductions.md): Dyneema rope on a steel pinion with a
# helical groove, to a smooth flanged drum; sizes come from the stage in
# data/transmissions.yaml. The yaw belt is 3M x 9 mm from a 15T pinion (PD 14.3 mm).
ROPE_PITCH = 1.2  # helical groove pitch on the pinion, x rope diameter
DRUM_FLANGE = 2.0
BELT_PINION_R = 7.16
BELT_W = 9.0
BELT_T = 2.4
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


def capstan_drum(r_eff: float, rope: float, y0: float, y1: float) -> Part:
    """A smooth drum on the y axis, flanged so the rope cannot walk off. r_eff is the
    radius to the rope centre (the ratio is r_eff over the pinion's)."""
    lo, hi = min(y0, y1), max(y0, y1)
    r_out = r_eff + rope / 2 + DRUM_FLANGE
    return cyl_y(r_eff - rope / 2, lo + 1.5, hi - 1.5) + cyl_y(r_out, lo, lo + 1.5) + cyl_y(r_out, hi - 1.5, hi)


def capstan_pinion(x: float, z: float, r_eff: float, rope: float, y0: float, y1: float) -> Part:
    """A steel pinion on the y axis with one helical groove for the wraps."""
    lo, hi = min(y0, y1), max(y0, y1)
    core = r_eff - rope / 2
    length = hi - lo
    helix = Helix(pitch=rope * ROPE_PITCH, height=length - 3, radius=core + rope * 0.15)
    groove = sweep(Plane(origin=helix @ 0, z_dir=helix % 0) * Circle(rope * 0.55), path=helix)
    pin = Cylinder(core + rope * 0.4, length, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Pos(0, 0, 1.5) * groove
    return Pos(x, lo, z) * (Rot(-90, 0, 0) * pin)


def belt_loop(c: float, r1: float, r2: float, z0: float, z1: float) -> Part:
    """A closed belt round a ring of radius r1 at the origin and a pinion of radius r2 at
    (c, 0), as the band between the two circles' hull and the same hull grown by BELT_T."""

    def hull(a: float, b: float, n: int = 48) -> list[tuple[float, float]]:
        th = math.acos((a - b) / c)
        pts = [(a * math.cos(t), a * math.sin(t)) for t in (th + (2 * math.pi - 2 * th) * k / n for k in range(n + 1))]
        pts += [(c + b * math.cos(t), b * math.sin(t)) for t in (-th + 2 * th * k / n for k in range(n + 1))]
        return pts

    h = z1 - z0
    outer = extrude(Polygon(*hull(r1 + BELT_T, r2 + BELT_T), align=None), h)
    inner = extrude(Polygon(*hull(r1, r2), align=None), h + 2)
    return Pos(0, 0, z0) * (outer - Pos(0, 0, -1) * inner)


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
    sh_stage: Stage | None = None  # capstan at the shoulder, None for a coaxial motor
    el_stage: Stage | None = None
    yaw_stage: Stage | None = None  # belt round a ring under the turret
    cheek_in: float = 20.0  # inner faces of the turret cheeks at +-cheek_in

    @property
    def cheek_out(self) -> float:
        return self.cheek_in + CHEEK_T

    @property
    def head_r(self) -> float:
        drums = [drum_r(s) for s in (self.sh_stage, self.el_stage) if s]
        coax = [d / 2 + 4 for d, s in ((self.sh_dia, self.sh_stage), (self.el_dia, self.el_stage)) if not s]
        return max(drums + coax) if self.para else max(drums or [self.sh_dia / 2 + 4])

    def motor_x(self, side: int) -> float:
        """Where a turret motor's axis sits along x: on the shoulder axis, or behind it by
        the capstan's centre distance (drum radius plus pinion radius)."""
        s = self.sh_stage if side > 0 else self.el_stage
        return -(s.ratio + 1) * s.pinion_r * 1000 if s else 0.0


def stage_of(act: object) -> Stage | None:
    """The reduction stage a part drives its joint through, None when it drives it directly."""
    sid = getattr(act, "id", "").partition("+")[2]
    return {s.id: s for s in load_stages()}[sid] if sid else None


def rope_mm(s: Stage) -> float:
    return float(s.rope.split(" mm")[0])


def drum_r(s: Stage) -> float:
    """Outer radius of a capstan drum, mm."""
    return s.drum_r * 1000 + 1


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
        sh_stage=stage_of(acts.shoulder),
        el_stage=stage_of(acts.elbow) if design.arch == "parallelogram" else None,
        yaw_stage=stage_of(acts.yaw),
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
    mx = lo.motor_x(side)
    if mx:
        # The motor sits behind the axis; the cheek carries it and a bearing for the joint.
        prof += cyl_y(dia / 2 + 4, y0, y1, mx, lo.ped)
        prof += Pos(mx / 2, (y0 + y1) / 2, lo.ped) * Box(abs(mx), CHEEK_T, dia * 0.8)
        prof -= cyl_y(12, y0 - 1, y1 + 1, 0, lo.ped)  # 6802 bearing for the joint shaft
    prof -= cyl_y(dia / 2 * 0.62 + 1.0, y0 - 1, y1 + 1, mx, lo.ped)
    prof -= bolt_circle(dia / 2 * 0.8, 6, 1.7, lambda a, b, hr: cyl_y(hr, y0 - 1, y1 + 1, mx + a, lo.ped + b))
    # Lightening window below the axis.
    if lo.ped > 55:
        prof -= Pos(0, (y0 + y1) / 2, (lo.ped - r * 0.6 + 14) / 2) * Box(r, CHEEK_T + 2, max(lo.ped - r * 0.6 - 14, 4))
    return prof


def turret_box(lo: Layout) -> Part:
    """On a tall pedestal: a closed box between the cheeks, under the drums. It stiffens the
    column and is the sealed, vented housing for the turret electronics."""
    top = lo.ped - lo.head_r - 6
    x = 2 * lo.head_r - 6
    outer = Pos(0, 0, 9.5) * Box(x, 2 * lo.cheek_in, top - 9.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    inner = Pos(0, 0, 9.5) * Box(x - 6, 2 * lo.cheek_in + 2, top - 12.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return outer - inner


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
    if lo.sh_stage:
        st = lo.sh_stage
        part += capstan_drum(st.ratio * st.pinion_r * 1000, rope_mm(st), 3, 17)
        part -= cyl_y(5.1, 0, lo.cheek_in + 1)  # shaft into the cheek bearing
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
    if lo.el_stage:
        # Drum beside the crank arm, in its own plane so the crank pin clears the pinion.
        st = lo.el_stage
        part += capstan_drum(st.ratio * st.pinion_r * 1000, rope_mm(st), -lo.cheek_in + 0.5, -12.5)
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


FROLL = 70.0  # forearm-roll actuator centre behind the wrist pitch axis (MassModel)


def roll_clamp(lo: Layout) -> Part:
    """6-axis forearm: the tube's end clamp, a cradle for the forearm-roll servo whose horn
    sits on the forearm axis."""
    x = lo.l2 - FROLL
    cradle = Pos(x, 0, -11) * Box(39, 30, 50) - Pos(x + 2, 0, -11) * Box(37, 26, 46)
    sock = cyl_x(16.5, x - 62, x - 17, -5)
    part = cradle + sock
    part -= cyl_x(TUBE_OD / 2 + 0.15, x - 64, x - 21, -5)
    return part


def roll_yoke(lo: Layout) -> Part:
    """6-axis wrist: from the forearm-roll horn to the wrist-pitch servo's cradle (froll frame)."""
    plate = cyl_x(12, 21.5, 25.5)
    beam = Pos(35, 0, 0) * Box(20, 30, 16)
    ring = Pos(FROLL, 0, 0) * Box(52, 30, 42) - Pos(FROLL, 0, 0) * Box(46, 32, 36)
    part = plate + beam + ring
    part -= bolt_circle(7, 4, 1.2, lambda a, b, r: cyl_x(r, 20, 27, a, b))
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
    if acts.yaw.kind == "servo":
        yaw = servo_box((0, 0, -17.5), "x")
    elif lo.yaw_stage:
        # The yaw motor stands beside the axis in the deck. A closed belt runs from its
        # pinion round a toothed ring under the turret, which turns on its own bearing
        # round a fixed post. The belt's path does not change as the turret turns.
        ring_r = lo.yaw_stage.ratio * BELT_PINION_R
        cy = ring_r + max(BELT_PINION_R, lo.yaw_dia / 2) + 8
        yaw = [Pos(-cy, 0, -12) * p for p in qdd_z(lo.yaw_dia, lo.yaw_len, 0.0)]
        pin = cyl_z(BELT_PINION_R, -12, -1, -cy, 0) + cyl_z(BELT_PINION_R + 2, -2, -1, -cy, 0)
        belt = Rot(0, 0, 180) * belt_loop(cy, ring_r, BELT_PINION_R, -11, -11 + BELT_W)
        yaw += [
            _c(pin, "steel", "yaw belt pinion (15T)"),
            _c(belt, "carbon", "yaw timing belt (3M, 9 mm)"),
            _c(cyl_z(8, -30, 8), "steel", "turret post + bearing"),
        ]
    else:
        yaw = qdd_z(lo.yaw_dia, lo.yaw_len, 0.0)
    links["body"] = [Pos(mx, 0, deck) * p for p in [*yaw, _c(deck_ring(lo), "printed", "deck seal ring (printed)")]]
    tur = [
        _c(turret_plate(lo), "printed", "turret plate (printed)"),
        _c(cheek(lo, 1), "printed", "turret cheek, shoulder side (printed)"),
        _c(cheek(lo, -1), "printed", "turret cheek, elbow side (printed)"),
    ]
    if lo.ped - lo.head_r - 6 > 40:
        tur.append(_c(turret_box(lo), "printed", "turret box (printed)"))
    tur += [Pos(lo.motor_x(1), 0, lo.ped) * p for p in qdd_y(lo.sh_dia, lo.sh_len, lo.cheek_out, -1)]
    if lo.sh_stage:
        st = lo.sh_stage
        pin = capstan_pinion(lo.motor_x(1), lo.ped, st.pinion_r * 1000, rope_mm(st), 3, lo.cheek_out)
        tur.append(_c(pin, "steel", f"shoulder capstan pinion ({st.rope})"))
    if lo.para:
        tur += [Pos(lo.motor_x(-1), 0, lo.ped) * p for p in qdd_y(lo.el_dia, lo.el_len, -lo.cheek_out, 1)]
        if lo.el_stage:
            st = lo.el_stage
            pin = capstan_pinion(lo.motor_x(-1), lo.ped, st.pinion_r * 1000, rope_mm(st), -lo.cheek_out, -12.5)
            tur.append(_c(pin, "steel", f"elbow capstan pinion ({st.rope})"))
    if lo.yaw_stage:
        ring_r = lo.yaw_stage.ratio * BELT_PINION_R
        ring = cyl_z(ring_r, -12, -1) + cyl_z(ring_r + 2, -2, -1)
        tur.append(_c(ring - cyl_z(9, -13, 0), "accent", "yaw toothed ring (printed)"))
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
    hub = _c(fore_hub(lo), "accent" if lo.para else "printed", "forearm hub and rear lever (printed)")
    if lo.wp_servo:
        wrist_motor = servo_box((0, 0, 0), "y")
    else:
        wrist_motor = qdd_y(lo.wp_dia, lo.wp_len, lo.wp_len / 2, 1)
    fr = acts.forearm_roll if design.dof == 6 else None
    if fr is not None:
        # The forearm-roll servo ends the forearm; the roll yoke carries the wrist motor.
        x = lo.l2 - FROLL
        fore = [hub, tube_x(12, x - 22, -5), _c(roll_clamp(lo), "printed", "forearm roll clamp (printed)")]
        if fr.kind == "servo":
            fore += servo_box((x, 0, -11), "x")
        else:
            fore += [Pos(x, 0, 0) * (Rot(0, 0, -90) * p) for p in qdd_y(fr.dia * 1000, fr.length * 1000, 20, 1)]
        links["fore"] = fore
        links["froll"] = [_c(roll_yoke(lo), "printed", "forearm roll yoke (printed)")]
        links["froll"] += [Pos(FROLL, 0, 0) * p for p in wrist_motor]
    else:
        fore = [hub, tube_x(12, lo.l2 - 28, -5), _c(wrist_clamp(lo), "printed", "wrist clamp (printed)")]
        links["fore"] = fore + [Pos(lo.l2, 0, 0) * p for p in wrist_motor]
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

"""MuJoCo model of the reference hexapod carrying a given arm.

The parallelogram elbow is modelled as the real closed chain: the elbow motor turns a
crank on the turret, a rod runs from the crank pin to a pin behind the elbow, and a
`connect` equality closes the loop. The torques the simulation reports at the shoulder and
elbow motors are therefore an independent check on the statics in model.py.

Every link's frame matches model.py: at zero joint angles the arm points along the
turret's +x, pitch joints turn about -y (positive lifts), and the tool axis is the wrist
frame's +x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from xml.sax.saxutils import quoteattr

import numpy as np

from armdesign.evaluate import Tool
from armdesign.model import Actuator, ArmDesign, Hexapod, MassModel

CRANK = 0.045  # parallelogram crank radius (and the forearm's rear pin offset)


@dataclass(frozen=True)
class ArmActuators:
    yaw: Actuator
    shoulder: Actuator
    elbow: Actuator
    wrist_pitch: Actuator
    tool_roll: Actuator
    forearm_roll: Actuator | None = None


def _arm(act: Actuator) -> str:
    """Joint armature: the motor's rotor inertia seen at the joint, at least the default."""
    return f"{max(0.006, act.inertia):.5g}"


def _f(*xs: float) -> str:
    return " ".join(f"{x:.5g}" for x in xs)


def _cyl_inertia_geom(name: str, act: Actuator, pos: tuple[float, float, float], axis: str, rgba: str) -> str:
    """An actuator as a cylinder geom with its catalog mass."""
    r = act.dia / 2
    h = act.length / 2
    zaxis = {"y": "0 1 0", "z": "0 0 1", "x": "1 0 0"}[axis]
    return (
        f'<geom name="{name}" type="cylinder" size="{_f(r, h)}" pos="{_f(*pos)}" '
        f'zaxis="{zaxis}" mass="{act.mass:.4g}" rgba="{rgba}" class="armcol"/>'
    )


def leg_frames(hx: Hexapod) -> list[tuple[np.ndarray, float]]:
    """(hip position, heading) for each leg; the leg's local x points at its foot."""
    out = []
    for hip, foot in zip(hx.hips, hx.feet, strict=True):
        heading = math.atan2(foot[1] - hip[1], foot[0] - hip[0])
        out.append((hip, heading))
    return out


def leg_ik(hx: Hexapod, i: int, foot_body: np.ndarray, foot_r: float = 0.02) -> tuple[float, float, float]:
    """Coxa, femur, tibia angles that put leg i's foot centre at foot_body (body frame)."""
    hip, heading = leg_frames(hx)[i]
    c, s = math.cos(-heading), math.sin(-heading)
    d = foot_body - hip
    p = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
    q0 = math.atan2(p[1], p[0])
    rho = math.hypot(p[0], p[1]) - hx.coxa
    h = p[2]
    dist = min(math.hypot(rho, h), hx.femur + hx.tibia - 1e-4)
    b1 = math.atan2(h, rho) + math.acos(np.clip((hx.femur**2 + dist**2 - hx.tibia**2) / (2 * hx.femur * dist), -1, 1))
    kx, kz = hx.femur * math.cos(b1), hx.femur * math.sin(b1)
    b2 = math.atan2(h - kz, rho - kx)
    return q0, b1, b2 - b1


def build_mjcf(
    design: ArmDesign,
    acts: ArmActuators,
    hx: Hexapod,
    tool: Tool,
    mm: MassModel | None = None,
    scene: str = "",
    equality: str = "",
    leg_mass: tuple[float, float, float] = (0.40, 0.30, 0.13),
) -> str:
    """The MJCF text. `scene` is extra worldbody XML (objects for a scenario) and
    `equality` extra equality constraints (grasp welds)."""
    mm = mm or MassModel()
    body_mass = hx.mass - 6 * sum(leg_mass)
    blue, grey, dark, orange = "0.25 0.45 0.85 1", "0.75 0.77 0.8 1", "0.18 0.19 0.22 1", "0.95 0.55 0.15 1"
    legs = []
    for i, (hip, heading) in enumerate(leg_frames(hx)):
        q = f"0 0 {math.sin(heading / 2):.6g}"
        legs.append(
            f"""
      <body name="leg{i}_coxa" pos="{_f(*hip)}" quat="{math.cos(heading / 2):.6g} {q}">
        <joint name="leg{i}_coxa" axis="0 0 1" range="-1.2 1.2" class="leg"/>
        <geom type="capsule" fromto="0 0 0 {hx.coxa} 0 0" size="0.03" mass="{leg_mass[0]}" rgba="{dark}" class="legcol"/>
        <body name="leg{i}_femur" pos="{hx.coxa} 0 0">
          <joint name="leg{i}_femur" axis="0 -1 0" range="-1.6 1.6" class="leg"/>
          <geom type="capsule" fromto="0 0 0 {hx.femur} 0 0" size="0.025" mass="{leg_mass[1]}" rgba="{grey}" class="legcol"/>
          <body name="leg{i}_tibia" pos="{hx.femur} 0 0">
            <joint name="leg{i}_tibia" axis="0 -1 0" range="-2.8 0.3" class="leg"/>
            <geom type="capsule" fromto="0 0 0 {hx.tibia} 0 0" size="0.018" mass="{leg_mass[2]}" rgba="{grey}" class="legcol"/>
            <geom name="foot{i}" type="sphere" pos="{hx.tibia} 0 0" size="0.022" mass="0.01" rgba="{dark}" friction="1.2 0.02 0.001"/>
          </body>
        </body>
      </body>"""
        )
    # --- the arm --------------------------------------------------------------------
    l1, l2, wl = design.l1, design.l2, design.wrist_len
    tube_r = 0.0127
    t1_mass = mm.tube_kg_per_m * l1
    t2_mass = mm.tube_kg_per_m * l2
    para = design.arch == "parallelogram"
    elbow_motor_on_arm = not para
    elbow_geom = (
        _cyl_inertia_geom("act_elbow", acts.elbow, (0, 0, 0), "y", blue)
        if elbow_motor_on_arm
        else f'<geom type="cylinder" size="0.03 0.03" zaxis="0 1 0" mass="{mm.elbow_housing_kg + mm.rod_kg / 2}" rgba="{grey}" class="armcol"/>'
    )
    crank = ""
    rod_eq = ""
    if para:
        y_off = -(acts.shoulder.length / 2 + acts.elbow.length / 2 + 0.01)
        crank = f"""
        <body name="crank" pos="0 0 {design.pedestal}">
          <joint name="arm_crank" axis="0 -1 0" class="arm" armature="{_arm(acts.elbow)}"/>
          {_cyl_inertia_geom("act_elbow", acts.elbow, (0, y_off, 0), "y", blue)}
          <geom type="capsule" fromto="0 0 0 {-CRANK} 0 0" size="0.008" mass="0.02" rgba="{dark}" class="vis"/>
          <body name="rod" pos="{-CRANK} 0 0">
            <joint name="arm_rod" axis="0 -1 0" class="passive"/>
            <geom type="capsule" fromto="0 0 0 {l1} 0 0" size="0.006" mass="{mm.rod_kg / 2}" rgba="{dark}" class="vis"/>
          </body>
        </body>"""
        rod_eq = f'<connect name="parallelogram" body1="rod" body2="fore" anchor="{l1} 0 0" solref="0.002 1"/>'
    elbow_joint_class = "passive" if para else "arm"
    froll = ""
    froll_close = ""
    if design.dof == 6 and acts.forearm_roll is not None:
        froll = f"""<body name="froll" pos="{l2 - mm.forearm_roll_offset} 0 0">
              <joint name="arm_forearm_roll" axis="1 0 0" class="arm"/>
              {_cyl_inertia_geom("act_froll", acts.forearm_roll, (0, 0, 0), "x", blue)}
              <body name="wrist_mount" pos="{mm.forearm_roll_offset} 0 0">"""
        froll_close = "</body></body>"
        wrist_pos = "0 0 0"
    else:
        wrist_pos = f"{l2} 0 0"
    tool_geoms = tool_xml(tool, wl)
    spring_note = (
        "<!-- zero-length gravity springs are applied as joint torques by the controller -->" if design.spring else ""
    )
    yaw_h = acts.yaw.length
    return f"""<mujoco model="hexapod_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" impratio="5"/>
  <size memory="20M"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <default>
    <geom friction="0.9 0.02 0.001" solref="0.005 1" solimp="0.95 0.99 0.001"/>
    <default class="leg"><joint damping="0.6" armature="0.01"/></default>
    <default class="arm"><joint damping="0.08" armature="0.006"/></default>
    <default class="passive"><joint damping="0.01" armature="0.0005"/></default>
    <default class="vis"><geom contype="0" conaffinity="0" group="1"/></default>
    <default class="armcol"><geom contype="2" conaffinity="1" group="2"/></default>
    <default class="legcol"><geom contype="4" conaffinity="1" group="2"/></default>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="6 6 0.1" rgba="0.36 0.31 0.24 1" contype="1" conaffinity="7"/>
    <body name="body" pos="0 0 {hx.center_height}">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size="{_f(hx.length / 2, hx.width / 2, hx.thickness / 2)}" mass="{body_mass:.4g}" rgba="0.22 0.24 0.27 1" contype="4" conaffinity="2"/>
      <site name="deck" pos="0 0 {hx.deck_z}" size="0.01"/>
      <geom name="tray" type="box" size="0.09 0.09 0.004" pos="-0.12 0 {hx.deck_z + 0.004}" rgba="0.15 0.5 0.35 1" contype="4" conaffinity="0"/>
      <geom name="tray_wall_b" type="box" size="0.004 0.09 0.02" pos="-0.21 0 {hx.deck_z + 0.02}" rgba="0.15 0.5 0.35 1" contype="4" conaffinity="0"/>
      {"".join(legs)}
      <body name="turret" pos="{design.mount_x} 0 {hx.deck_z}">
        <joint name="arm_yaw" axis="0 0 1" range="-3.5 3.5" class="arm" armature="{_arm(acts.yaw)}"/>
        {_cyl_inertia_geom("act_yaw", acts.yaw, (0, 0, yaw_h / 2), "z", blue)}
        <geom type="box" size="0.035 0.05 {design.pedestal / 2:.4g}" pos="0 0 {design.pedestal / 2:.4g}" mass="{mm.turret_kg:.4g}" rgba="{grey}" class="vis"/>
        <body name="upper" pos="0 0 {design.pedestal}">
          <joint name="arm_shoulder" axis="0 -1 0" range="-0.7 2.7" class="arm" armature="{_arm(acts.shoulder)}"/>
          {_cyl_inertia_geom("act_shoulder", acts.shoulder, (0, acts.shoulder.length / 2 + 0.005, 0), "y", blue)}
          <geom type="capsule" fromto="0 0 0 {l1} 0 0" size="{tube_r}" mass="{t1_mass:.4g}" rgba="{grey}" class="armcol"/>
          <body name="fore" pos="{l1} 0 0">
            <joint name="arm_elbow" axis="0 -1 0" range="-2.95 0.1" class="{elbow_joint_class}"{"" if para else f' armature="{_arm(acts.elbow)}"'}/>
            {elbow_geom}
            <geom type="capsule" fromto="{-CRANK if para else 0} 0 0 {l2} 0 0" size="{tube_r}" mass="{t2_mass:.4g}" rgba="{grey}" class="armcol"/>
            {froll}
            <body name="wrist" pos="{wrist_pos}">
              <joint name="arm_wrist_pitch" axis="0 -1 0" range="-2.2 2.2" class="arm"/>
              {_cyl_inertia_geom("act_wrist", acts.wrist_pitch, (0, 0, 0), "y", blue)}
              <geom type="box" size="0.025 0.025 0.025" mass="{mm.wrist_housing_kg}" rgba="{grey}" class="vis"/>
              <body name="flange" pos="0 0 0">
                <joint name="arm_tool_roll" axis="1 0 0" class="arm"/>
                {_cyl_inertia_geom("act_roll", acts.tool_roll, (mm.roll_offset, 0, 0), "x", blue)}
                <geom type="cylinder" size="0.03 0.008" pos="{wl - 0.01} 0 0" zaxis="1 0 0" mass="{mm.flange_kg}" rgba="{orange}" class="vis"/>
                <site name="camera" pos="{wl - 0.02} 0 0.035" size="0.008" rgba="0 0 0 1"/>
                {tool_geoms}
              </body>
            </body>
            {froll_close}
          </body>
        </body>{crank}
      </body>
    </body>
    {scene}
  </worldbody>
  <equality>
    {rod_eq}
    {equality}
  </equality>
  {spring_note}
  <actuator>
    {"".join(f'<position name="leg{i}_{j}" joint="leg{i}_{j}" kp="400" kv="10" forcerange="-30 30"/>' for i in range(6) for j in ("coxa", "femur", "tibia"))}
    <motor name="m_yaw" joint="arm_yaw" ctrlrange="{_f(-acts.yaw.peak, acts.yaw.peak)}"/>
    <motor name="m_shoulder" joint="arm_shoulder" ctrlrange="{_f(-acts.shoulder.peak, acts.shoulder.peak)}"/>
    <motor name="m_elbow" joint="{"arm_crank" if para else "arm_elbow"}" ctrlrange="{_f(-acts.elbow.peak, acts.elbow.peak)}"/>
    {f'<motor name="m_forearm_roll" joint="arm_forearm_roll" ctrlrange="{_f(-acts.forearm_roll.peak, acts.forearm_roll.peak)}"/>' if froll else ""}
    <motor name="m_wrist_pitch" joint="arm_wrist_pitch" ctrlrange="{_f(-acts.wrist_pitch.peak, acts.wrist_pitch.peak)}"/>
    <motor name="m_tool_roll" joint="arm_tool_roll" ctrlrange="{_f(-acts.tool_roll.peak, acts.tool_roll.peak)}"/>
    {tool_actuators(tool)}
  </actuator>
  <sensor>
    <framepos name="tcp_pos" objtype="site" objname="tcp"/>
  </sensor>
</mujoco>
"""


def tool_xml(tool: Tool, wl: float) -> str:
    """Tool geometry in the flange frame (tool axis = +x)."""
    x0 = wl
    if tool.id == "gripper":
        fl = tool.length - 0.04
        return f"""<geom type="box" size="0.02 0.035 0.02" pos="{x0 + 0.02} 0 0" mass="{tool.mass - 0.04:.4g}" rgba="0.3 0.3 0.33 1" class="armcol"/>
                <site name="tcp" pos="{x0 + tool.length} 0 0" size="0.006" rgba="1 0.2 0.2 1"/>
                <body name="finger_l" pos="{x0 + 0.04} 0 0">
                  <joint name="finger_l" type="slide" axis="0 1 0" range="0 0.06" damping="2"/>
                  <geom name="pad_l" type="box" size="{fl / 2:.4g} 0.006 0.015" pos="{fl / 2:.4g} 0.008 0" mass="0.02" rgba="0.95 0.55 0.15 1" friction="1.5 0.02 0.001" class="armcol"/>
                </body>
                <body name="finger_r" pos="{x0 + 0.04} 0 0">
                  <joint name="finger_r" type="slide" axis="0 -1 0" range="0 0.06" damping="2"/>
                  <geom name="pad_r" type="box" size="{fl / 2:.4g} 0.006 0.015" pos="{fl / 2:.4g} -0.008 0" mass="0.02" rgba="0.95 0.55 0.15 1" friction="1.5 0.02 0.001" class="armcol"/>
                </body>"""
    r = 0.006 if tool.id == "probe" else 0.02
    return f"""<geom type="capsule" fromto="{x0} 0 0 {x0 + tool.length - r} 0 0" size="{r}" mass="{tool.mass:.4g}" rgba="0.8 0.8 0.82 1" class="vis"/>
                <site name="tcp" pos="{x0 + tool.length} 0 0" size="0.006" rgba="1 0.2 0.2 1"/>"""


def tool_actuators(tool: Tool) -> str:
    if tool.id != "gripper":
        return ""
    return (
        '<position name="grip_l" joint="finger_l" kp="300" forcerange="-25 25" ctrlrange="0 0.06"/>'
        '<position name="grip_r" joint="finger_r" kp="300" forcerange="-25 25" ctrlrange="0 0.06"/>'
    )


def body_xml(name: str, pos: tuple[float, float, float], geom: str, free: bool = True) -> str:
    j = "<freejoint/>" if free else ""
    return f'<body name={quoteattr(name)} pos="{_f(*pos)}">{j}{geom}</body>'

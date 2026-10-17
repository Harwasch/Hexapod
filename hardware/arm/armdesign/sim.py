"""Run the selected arm on the reference hexapod through real tasks in MuJoCo.

    uv run python -m armdesign.sim                 # every scenario, writes out/sim/*.json
    uv run python -m armdesign.sim weed_body rock  # just these

The arm is torque-controlled the way the real one will be: joint-space PD on the motor
angles, model-based gravity feed-forward, the gravity springs as physical torques, and
every motor clipped at its catalog peak torque. The legs are position servos that hold the
feet and move the body. Soil, roots and stems are force laws applied to the objects (a
breakable spring for a root or a stem, depth-dependent resistance for a probe), so a pull
or a push shows up in the motor torques exactly as it would on hardware.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np
import yaml

from armdesign.evaluate import Tool, load_tools
from armdesign.mjcf import ArmActuators, build_mjcf, leg_ik
from armdesign.model import ROOT, ArmDesign, BodyPose, Hexapod, planar_ik, wrap
from armdesign.parts import resolve

OUT = ROOT / "out" / "sim"
Array = np.ndarray
MOTORS = ("yaw", "shoulder", "elbow", "wrist_pitch", "tool_roll")
YAW_LIMIT = math.radians(200)  # cable wrap
LOG_HZ = 50


def min_jerk(s: float) -> tuple[float, float, float]:
    """Position, velocity and acceleration of the minimum-jerk profile on [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return (
        10 * s**3 - 15 * s**4 + 6 * s**5,
        30 * s**2 - 60 * s**3 + 30 * s**4,
        60 * s - 180 * s**2 + 120 * s**3,
    )


# --- objects and the forces of the world ---------------------------------------------


@dataclass
class Tether:
    """A root or a stem: a stiff spring to an anchor that lets go above `breaks_at` newtons,
    then drags for `tail` metres while the last roots tear."""

    body: str
    anchor: Array
    k: float
    breaks_at: float
    tail: float = 0.02
    damping: float = 30.0
    broken: bool = False
    broke_at: Array | None = None
    force: Array = field(default_factory=lambda: np.zeros(3))


@dataclass
class Soil:
    """Resistance to a probe tip below the surface: tip + shaft friction growing with depth,
    and a lateral spring that keeps the probe in its hole."""

    base: float = 35.0
    per_m: float = 600.0
    lateral_k: float = 3000.0
    entry: Array | None = None
    force: Array = field(default_factory=lambda: np.zeros(3))


@dataclass
class Obj:
    name: str
    xml: str
    tether: Tether | None = None
    graspable: bool = True


# Loose objects (fruit, rocks) land on the ground, the chassis, the legs and the deck tray
# (conaffinity 1 | 4) but never push on the arm's own collision shapes.
def strawberry(name: str, pos: tuple[float, float, float], breaks_at: float = 13.9) -> Obj:
    x, y, z = pos
    xml = f"""<body name="{name}" pos="{x} {y} {z}"><freejoint/>
      <geom type="ellipsoid" size="0.015 0.015 0.019" mass="0.03" rgba="0.85 0.08 0.1 1" contype="8" conaffinity="5"/>
      <geom type="cylinder" size="0.012 0.003" pos="0 0 0.019" mass="0.002" rgba="0.2 0.55 0.2 1" contype="0" conaffinity="0"/>
    </body>
    <body name="{name}_plant" pos="{x - 0.05} {y} 0"><geom type="cylinder" size="0.004 {z / 2 + 0.02:.3f}" pos="0 0 {z / 2 + 0.02:.3f}" rgba="0.25 0.5 0.2 1" contype="0" conaffinity="0"/>
      <geom type="ellipsoid" size="0.06 0.05 0.01" pos="0 0 {z + 0.05:.3f}" rgba="0.2 0.5 0.18 1" contype="0" conaffinity="0"/></body>"""
    return Obj(name, xml, Tether(name, np.array(pos, float), 4000.0, breaks_at, tail=0.004))


def weed(name: str, pos: tuple[float, float], breaks_at: float) -> Obj:
    x, y = pos
    leaves = "".join(
        f'<geom type="ellipsoid" size="0.035 0.012 0.003" pos="{0.03 * math.cos(a):.3f} {0.03 * math.sin(a):.3f} -0.025" '
        f'euler="0 0 {a:.3f}" rgba="0.3 0.6 0.15 1" mass="0.002" contype="0" conaffinity="0"/>'
        for a in np.linspace(0, 2 * math.pi, 7)[:-1]
    )
    xml = f"""<body name="{name}" pos="{x} {y} 0.035"><freejoint/>
      <geom type="capsule" size="0.006 0.03" mass="0.02" rgba="0.35 0.55 0.2 1" contype="0" conaffinity="0"/>
      <geom type="capsule" size="0.004 0.05" pos="0 0 -0.08" mass="0.01" rgba="0.85 0.8 0.65 1" contype="0" conaffinity="0"/>
      {leaves}
    </body>"""
    return Obj(name, xml, Tether(name, np.array([x, y, 0.035]), 3000.0, breaks_at, tail=0.03, damping=60.0))


def rock(name: str, pos: tuple[float, float], mass: float = 1.0) -> Obj:
    x, y = pos
    xml = f"""<body name="{name}" pos="{x} {y} 0.031"><freejoint/>
      <geom type="box" size="0.042 0.034 0.03" mass="{mass}" rgba="0.55 0.53 0.5 1" contype="1" conaffinity="5" friction="0.8 0.01 0.001"/>
    </body>"""
    return Obj(name, xml)


def tomato(name: str, pos: tuple[float, float, float], breaks_at: float = 16.3) -> Obj:
    x, y, z = pos
    xml = f"""<body name="{name}" pos="{x} {y} {z}"><freejoint/>
      <geom type="sphere" size="0.03" mass="0.2" rgba="0.9 0.2 0.1 1" contype="8" conaffinity="5"/>
    </body>
    <body name="{name}_plant" pos="{x + 0.12} {y} 0"><geom type="cylinder" size="0.008 {z / 2 + 0.15:.3f}" pos="0 0 {z / 2 + 0.15:.3f}" rgba="0.3 0.5 0.2 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 {z + 0.03:.3f} -0.12 0 {z + 0.03:.3f}" size="0.004" rgba="0.3 0.5 0.2 1" contype="0" conaffinity="0"/></body>"""
    return Obj(name, xml, Tether(name, np.array(pos, float), 3000.0, breaks_at, tail=0.006))


def marker(name: str, pos: tuple[float, float]) -> Obj:
    x, y = pos
    xml = f"""<body name="{name}" pos="{x} {y} 0.002">
      <geom type="cylinder" size="0.025 0.002" rgba="0.95 0.85 0.2 1" contype="0" conaffinity="0"/>
      <geom type="ellipsoid" size="0.02 0.008 0.004" pos="0.012 0 0.006" rgba="0.35 0.6 0.15 1" contype="0" conaffinity="0"/>
    </body>"""
    return Obj(name, xml, graspable=False)


# --- the rig ------------------------------------------------------------------------


class Rig:
    def __init__(
        self,
        design: ArmDesign,
        acts: ArmActuators,
        hx: Hexapod,
        tool: Tool,
        objects: list[Obj],
        standoff: float = 0.0,
    ) -> None:
        self.design, self.acts, self.hx, self.tool = design, acts, hx, tool
        self.objects = {o.name: o for o in objects}
        eq = "".join(
            f'<weld name="grasp_{o.name}" body1="flange" body2="{o.name}" active="false" solref="0.004 1"/>'
            for o in objects
            if o.graspable
        )
        self.xml = build_mjcf(design, acts, hx, tool, scene="".join(o.xml for o in objects), equality=eq)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        self.para = design.arch == "parallelogram"
        self.six = design.dof == 6 and acts.forearm_roll is not None
        self.tcp_len = design.wrist_len + tool.length + standoff
        jid = lambda n: self.m.joint(n).id  # noqa: E731
        self.dof = {n: self.m.jnt_dofadr[jid(n)] for n in self._arm_joint_names()}
        self.qadr = {n: self.m.jnt_qposadr[jid(n)] for n in self._arm_joint_names()}
        # A 6-axis arm's forearm roll is the sixth motor coordinate. The scenarios hold it
        # level: every task here has a 5-axis solution, so it only carries the wrist.
        self.motors = MOTORS + (("forearm_roll",) if self.six else ())
        self.n = len(self.motors)
        self.act = {n: self.m.actuator(f"m_{n}").id for n in self.motors}
        self.legs = [[self.m.actuator(f"leg{i}_{j}").id for j in ("coxa", "femur", "tibia")] for i in range(6)]
        self.body_id = self.m.body("body").id
        self.flange_id = self.m.body("flange").id
        self.tcp_site = self.m.site("tcp").id
        self.foot_geoms = {self.m.geom(f"foot{i}").id for i in range(6)}
        self.ground = self.m.geom("ground").id
        self.grip = [self.m.actuator(n).id for n in ("grip_l", "grip_r")] if tool.id == "gripper" else []
        self.peak = np.array([getattr(acts, n).peak for n in self.motors])
        self.rated = np.array([getattr(acts, n).rated for n in self.motors])
        extra = 6 - len(MOTORS) if self.six else 0
        self.kp = np.array([60.0, 160.0, 120.0, 25.0, 6.0] + [25.0] * extra)
        self.kd = np.array([2.5, 5.0, 4.0, 0.5, 0.15] + [0.5] * extra)
        self.ki = np.array([20.0, 400.0, 300.0, 40.0, 0.0] + [40.0] * extra)
        self.i_limit = 0.25 * self.peak
        self.standoff = standoff
        self.soil: Soil | None = None
        self.spring = np.zeros(self.n)
        self._place_initial()
        self._tune_springs()

    def _arm_joint_names(self) -> list[str]:
        names = ["arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch", "arm_tool_roll"]
        if self.design.arch == "parallelogram":
            names += ["arm_crank", "arm_rod"]
        if self.design.dof == 6 and self.acts.forearm_roll is not None:
            names.append("arm_forearm_roll")
        return names

    # motor coordinates: yaw, shoulder angle th2, forearm angle th3 (absolute), wrist, roll,
    # and on a 6-axis arm the forearm roll
    def set_motor_q(self, a: Array) -> None:
        d, q = self.d, self.qadr
        d.qpos[q["arm_yaw"]] = a[0]
        d.qpos[q["arm_shoulder"]] = a[1]
        d.qpos[q["arm_elbow"]] = a[2] - a[1]
        d.qpos[q["arm_wrist_pitch"]] = a[3]
        d.qpos[q["arm_tool_roll"]] = a[4]
        if self.para:
            d.qpos[q["arm_crank"]] = a[2]
            d.qpos[q["arm_rod"]] = a[1] - a[2]
        if self.six:
            d.qpos[q["arm_forearm_roll"]] = a[5]

    def motor_q(self) -> tuple[Array, Array]:
        d, q, v = self.d, self.qadr, self.dof
        pos = np.array(
            [
                d.qpos[q["arm_yaw"]],
                d.qpos[q["arm_shoulder"]],
                d.qpos[q["arm_shoulder"]] + d.qpos[q["arm_elbow"]],
                d.qpos[q["arm_wrist_pitch"]],
                d.qpos[q["arm_tool_roll"]],
                *([d.qpos[q["arm_forearm_roll"]]] if self.six else []),
            ]
        )
        vel = np.array(
            [
                d.qvel[v["arm_yaw"]],
                d.qvel[v["arm_shoulder"]],
                d.qvel[v["arm_shoulder"]] + d.qvel[v["arm_elbow"]],
                d.qvel[v["arm_wrist_pitch"]],
                d.qvel[v["arm_tool_roll"]],
                *([d.qvel[v["arm_forearm_roll"]]] if self.six else []),
            ]
        )
        return pos, vel

    def _gamma(self) -> tuple[list[int], Array]:
        """Joint dofs and d(joint)/d(motor coordinate)."""
        names = self._arm_joint_names()
        g = np.zeros((len(names), self.n))
        rows = {n: i for i, n in enumerate(names)}
        g[rows["arm_yaw"], 0] = 1
        g[rows["arm_shoulder"], 1] = 1
        g[rows["arm_elbow"], 2], g[rows["arm_elbow"], 1] = 1, -1
        g[rows["arm_wrist_pitch"], 3] = 1
        g[rows["arm_tool_roll"], 4] = 1
        if self.para:
            g[rows["arm_crank"], 2] = 1
            g[rows["arm_rod"], 1], g[rows["arm_rod"], 2] = 1, -1
        if self.six:
            g[rows["arm_forearm_roll"], 5] = 1
        return [self.dof[n] for n in names], g

    def gravity_motor(self) -> Array:
        dofs, g = self._gamma()
        return g.T @ self.d.qfrc_bias[dofs]

    def inertia_motor(self) -> Array:
        """The arm's mass matrix in motor coordinates (base held still)."""
        dofs, g = self._gamma()
        full = np.zeros((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, self.d, full)
        return g.T @ full[np.ix_(dofs, dofs)] @ g

    def motor_to_actuator(self, tau: Array) -> Array:
        """Motor-coordinate torques -> what each physical motor applies. A direct elbow's
        motor turns the relative joint, so the shoulder motor also carries the elbow's."""
        if self.para:
            return tau
        out = tau.copy()
        out[1] = tau[1] + tau[2]
        return out

    def actuator_to_motor(self, f: Array) -> Array:
        if self.para:
            return f
        out = f.copy()
        out[1] = f[1] - f[2]
        return out

    def _place_initial(self) -> None:
        d = self.d
        d.qpos[:3] = [0, 0, self.hx.center_height]
        d.qpos[3:7] = [1, 0, 0, 0]
        self.body_target = BodyPose()
        self.set_legs(self.body_target)
        for i in range(6):
            for a in self.legs[i]:
                jnt = self.m.actuator_trnid[a, 0]
                d.qpos[self.m.jnt_qposadr[jnt]] = d.ctrl[a]
        self.home = self.stow_pose()
        self.set_motor_q(self.home)
        mujoco.mj_forward(self.m, d)

    def stow_pose(self) -> Array:
        # Folded over the deck with the tool pointing down, clear of the chassis.
        r, t = self.hx.body_rt(BodyPose())
        target = r @ np.array([self.design.mount_x + 0.12, 0.0, self.hx.deck_z + 0.14]) + t
        return self.ik(target, np.array([0.0, 0.0, -1.0]), 0.0, BodyPose())

    def _tune_springs(self) -> None:
        if not self.design.spring:
            return
        saved = self.d.qpos.copy()
        level = np.zeros(self.n)
        level[2] = -math.pi / 2 if not self.para else 0.0
        self.set_motor_q(level)
        mujoco.mj_forward(self.m, self.d)
        g = self.gravity_motor()
        self.spring[1] = g[1]
        if self.para:
            self.spring[2] = g[2]
        self.d.qpos[:] = saved
        mujoco.mj_forward(self.m, self.d)

    # --- the legs ----------------------------------------------------------------
    def set_legs(self, pose: BodyPose) -> None:
        r, t = self.hx.body_rt(pose)
        for i in range(6):
            foot = self.hx.feet[i] + np.array([0, 0, 0.022])
            fb = r.T @ (foot - t)
            for a, q in zip(self.legs[i], leg_ik(self.hx, i, fb), strict=True):
                self.d.ctrl[a] = q

    # --- inverse kinematics -------------------------------------------------------
    def ik(self, p: Array, a: Array, roll: float, pose: BodyPose, near: Array | None = None) -> Array:
        r, t = self.hx.body_rt(pose)
        d = self.design
        s_body = np.array([d.mount_x, 0.0, self.hx.deck_z + d.pedestal])
        pb = r.T @ (np.asarray(p) - t) - s_body
        ab = r.T @ np.asarray(a, float)
        yaw = math.atan2(pb[1], pb[0])
        ref = near[0] if near is not None else 0.0
        # Of the equivalent angles inside the cable wrap, the one closest to where it is.
        options = [yaw + k * 2 * math.pi for k in (-1, 0, 1) if abs(yaw + k * 2 * math.pi) <= YAW_LIMIT]
        yaw = min(options, key=lambda y: abs(y - ref))
        radial = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        normal = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        ab = ab - (ab @ normal) * normal
        ab /= np.linalg.norm(ab)
        w = pb - self.tcp_len * ab
        th2, th3, ok = planar_ik(np.array([w @ radial]), np.array([w[2]]), d.l1, d.l2)
        if not ok[0]:
            raise ValueError(f"target out of reach: {p}")
        phi = math.atan2(ab[2], ab @ radial)
        q4 = float(wrap(np.array(phi - th3[0])))
        return np.array([yaw, th2[0], th3[0], q4, roll] + [0.0] * (self.n - 5))

    def fk(self, a: Array, pose: BodyPose) -> Array:
        """TCP position for motor coordinates a with the body at pose."""
        r, t = self.hx.body_rt(pose)
        d = self.design
        s = np.array([d.mount_x, 0.0, self.hx.deck_z + d.pedestal])
        radial = np.array([math.cos(a[0]), math.sin(a[0]), 0.0])
        z = np.array([0.0, 0.0, 1.0])
        w = (
            s
            + d.l1 * (math.cos(a[1]) * radial + math.sin(a[1]) * z)
            + d.l2 * (math.cos(a[2]) * radial + math.sin(a[2]) * z)
        )
        phi = a[2] + a[3]
        tcp = w + self.tcp_len * (math.cos(phi) * radial + math.sin(phi) * z)
        return r @ tcp + t

    def tcp(self) -> Array:
        return self.d.site_xpos[self.tcp_site].copy()

    def aim(self) -> Array:
        """The point the tool works on: the TCP, or for a non-contact tool the point its
        standoff away along the tool axis."""
        axis = self.d.xmat[self.flange_id].reshape(3, 3)[:, 0]
        return self.tcp() + self.standoff * axis

    # --- grasping -----------------------------------------------------------------
    def attach(self, name: str) -> None:
        eq = self.m.equality(f"grasp_{name}").id
        b1, b2 = self.flange_id, self.m.body(name).id
        p1, q1 = self.d.xpos[b1], self.d.xquat[b1]
        p2, q2 = self.d.xpos[b2], self.d.xquat[b2]
        q1i = np.zeros(4)
        mujoco.mju_negQuat(q1i, q1)
        relp = np.zeros(3)
        mujoco.mju_rotVecQuat(relp, p2 - p1, q1i)
        relq = np.zeros(4)
        mujoco.mju_mulQuat(relq, q1i, q2)
        self.m.eq_data[eq, 0:3] = 0.0
        self.m.eq_data[eq, 3:6] = relp
        self.m.eq_data[eq, 6:10] = relq
        self.d.eq_active[eq] = 1

    def release(self, name: str) -> None:
        self.d.eq_active[self.m.equality(f"grasp_{name}").id] = 0

    # --- the world's forces ---------------------------------------------------------
    def apply_world(self) -> Array:
        """Apply root/stem/soil forces; return the force the world puts on the tool."""
        self.d.xfrc_applied[:] = 0.0
        on_tool = np.zeros(3)
        for o in self.objects.values():
            t = o.tether
            if t is None:
                continue
            bid = self.m.body(o.name).id
            x = self.d.xpos[bid]
            v = self._body_vel(bid)
            # The force is applied explicitly, so damping on a light free object must stay
            # below m/dt or it rings; once gripped the arm's mass dominates anyway.
            c = min(t.damping, 0.4 * self.m.body_subtreemass[bid] / self.m.opt.timestep)
            if not t.broken:
                f = -t.k * (x - t.anchor) - c * v
                if np.linalg.norm(f) > t.breaks_at:
                    t.broken, t.broke_at = True, x.copy()
            if t.broken:
                gone = np.linalg.norm(x - t.broke_at)
                scale = max(0.0, 1.0 - gone / t.tail) if t.tail > 0 else 0.0
                f = -unit3(x - t.anchor) * t.breaks_at * 0.5 * scale - 2.0 * v * scale
            t.force = f
            self.d.xfrc_applied[bid, :3] = f
            if self.d.eq_active[self.m.equality(f"grasp_{o.name}").id]:
                on_tool += f
        if self.soil is not None:
            tip = self.tcp()
            s = self.soil
            if tip[2] < 0.0:
                if s.entry is None:
                    s.entry = tip.copy()
                depth = -tip[2]
                vz = self._tcp_vel()[2]
                fz = (s.base + s.per_m * depth) * (1.0 if vz <= 0 else 0.4) - 200.0 * min(vz, 0.0) * 0
                lateral = -s.lateral_k * (tip[:2] - s.entry[:2])
                f = np.array([lateral[0], lateral[1], fz])
                if vz > 0:
                    f[2] = -min(15.0 + 150.0 * depth, 40.0)  # friction holds it back on the way out
            else:
                s.entry = None
                f = np.zeros(3)
            s.force = f
            com = self.d.xipos[self.flange_id]
            self.d.xfrc_applied[self.flange_id, :3] = f
            self.d.xfrc_applied[self.flange_id, 3:] = np.cross(tip - com, f)
            on_tool += f
        return on_tool

    def _body_vel(self, bid: int) -> Array:
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
        return res[3:]

    def _tcp_vel(self) -> Array:
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jac, None, self.tcp_site)
        return jac @ self.d.qvel

    # --- stability from the contact forces -------------------------------------------
    def cop_margin(self) -> tuple[float, Array]:
        total, moment = 0.0, np.zeros(2)
        feet = []
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            pair = {c.geom1, c.geom2}
            if self.ground not in pair:
                continue
            foot = pair - {self.ground}
            if not foot or next(iter(foot)) not in self.foot_geoms:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f6)
            fn = f6[0]
            if fn <= 1e-6:
                continue
            total += fn
            moment += fn * c.pos[:2]
            feet.append(c.pos[:2].copy())
        if total <= 0 or len(feet) < 3:
            return -1.0, np.zeros(2)
        cop = moment / total
        hull = convex_hull(np.array(feet))
        return polygon_margin(hull, cop), cop


def unit3(v: Array) -> Array:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def convex_hull(pts: Array) -> Array:
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o: tuple, a: tuple, b: tuple) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_margin(hull: Array, p: Array) -> float:
    """Signed distance from p to the polygon's boundary, positive inside (CCW hull)."""
    best = np.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        e = b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        best = min(best, float((p - a) @ n))
    return best


# --- scripted motion -----------------------------------------------------------------


@dataclass
class Seg:
    """One phase of a scenario. kind: 'joint' (min-jerk in motor space to an IK pose),
    'line' (straight TCP line, IK every step), 'hold' (keep the arm's joints still while the
    body moves). grip: finger opening (m) at the end of the phase, None to keep it."""

    label: str
    duration: float
    kind: str = "joint"
    target: Array | None = None  # TCP position (ground frame), or body frame if in_body
    approach: Array | None = None
    roll: float = 0.0
    in_body: bool = False
    body: BodyPose | None = None
    grip: float | None = None
    at_start: Callable[[Rig], None] | None = None
    at_end: Callable[[Rig], None] | None = None
    motor: Array | None = None  # explicit motor target instead of a TCP target


def run(rig: Rig, segs: list[Seg], name: str, settle: float = 0.6) -> dict:
    m, d = rig.m, rig.d
    dt = m.opt.timestep
    every = round(1 / (LOG_HZ * dt))
    bodies = [m.body(i).name for i in range(1, m.nbody)]
    log: dict[str, list] = {k: [] for k in ("t", "pos", "quat", "tau", "tcp", "tool_force", "margin", "cop", "body")}
    events: list[dict] = []
    a_cmd = rig.home.copy()
    grip_cmd = 0.04
    pose = BodyPose()
    tracking: list[dict] = []
    last_tgt: Array | None = None

    integ = np.zeros(rig.n)
    zero = np.zeros(rig.n)

    def step(a_des: Array, ad_des: Array, t_now: float, add_des: Array | None = None) -> None:
        nonlocal grip_cmd, integ
        q, qd = rig.motor_q()
        on_tool = rig.apply_world()
        # Integral action stands in for payload estimation: gravity feed-forward knows the
        # arm, not what it is carrying.
        integ = np.clip(integ + rig.ki * (a_des - q) * dt, -rig.i_limit, rig.i_limit)
        tau = (
            rig.kp * (a_des - q)
            + rig.kd * (ad_des - qd)
            + integ
            + rig.gravity_motor()
            - rig.spring * np.r_[0, math.cos(q[1]), math.cos(q[2]), np.zeros(rig.n - 3)]
        )
        if add_des is not None:
            tau += rig.inertia_motor() @ add_des
        act = np.clip(rig.motor_to_actuator(tau), -rig.peak, rig.peak)
        for i, n in enumerate(rig.motors):
            d.ctrl[rig.act[n]] = act[i]
        # The springs are hardware: torque on the shoulder and the crank (or the shoulder only).
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[rig.dof["arm_shoulder"]] += rig.spring[1] * math.cos(q[1])
        if rig.para:
            d.qfrc_applied[rig.dof["arm_crank"]] += rig.spring[2] * math.cos(q[2])
        for gid in rig.grip:
            d.ctrl[gid] = grip_cmd
        mujoco.mj_step(m, d)
        if round(t_now / dt) % every == 0:
            margin, cop = rig.cop_margin()
            log["t"].append(round(t_now, 3))
            log["pos"].append(np.round(d.xpos[1:], 4).tolist())
            log["quat"].append(np.round(d.xquat[1:], 4).tolist())
            log["tau"].append(np.round(d.actuator_force[[rig.act[n] for n in rig.motors]], 3).tolist())
            log["tcp"].append(np.round(rig.tcp(), 4).tolist())
            log["tool_force"].append(np.round(on_tool, 2).tolist())
            log["margin"].append(round(margin, 4))
            log["cop"].append(np.round(cop, 4).tolist())
            log["body"].append(np.round(np.r_[d.qpos[:3], d.qpos[3:7]], 4).tolist())

    t = 0.0
    for _ in range(round(settle / dt)):
        step(a_cmd, zero, t)
        t += dt
    for seg in segs:
        if seg.at_start:
            seg.at_start(rig)
        events.append({"t": round(t, 3), "label": seg.label})
        a0, pose0, grip0 = a_cmd.copy(), pose, grip_cmd
        pose1 = seg.body if seg.body is not None else pose0
        if not rig.hx.pose_ok(pose1):
            raise ValueError(f"{name}: the legs cannot hold the body at {pose1} ({seg.label})")
        tgt = seg.target
        if tgt is not None and seg.in_body:
            r1, t1 = rig.hx.body_rt(pose1)
            tgt = r1 @ np.asarray(tgt) + t1
        app = np.asarray(seg.approach if seg.approach is not None else [0, 0, -1.0], float)
        app = app / np.linalg.norm(app)
        if seg.kind == "joint":
            a1 = seg.motor if seg.motor is not None else rig.ik(tgt, app, seg.roll, pose1, near=a0)
        # Lines start where the arm was commanded to be, not where it sagged to.
        p0 = rig.fk(a_cmd, pose0)
        n = max(1, round(seg.duration / dt))
        ad_prev = zero
        for k in range(n):
            s, ds, dds = min_jerk((k + 1) / n)
            ds /= seg.duration
            dds /= seg.duration**2
            pose = BodyPose(
                pose0.dz + s * (pose1.dz - pose0.dz),
                pose0.pitch + s * (pose1.pitch - pose0.pitch),
                pose0.yaw + s * (pose1.yaw - pose0.yaw),
                pose0.dx + s * (pose1.dx - pose0.dx),
            )
            rig.set_legs(pose)
            if seg.grip is not None:
                grip_cmd = grip0 + s * (seg.grip - grip0)
            if seg.kind == "joint":
                a_cmd = a0 + s * (a1 - a0)
                ad = ds * (a1 - a0)
                add = dds * (a1 - a0)
            elif seg.kind == "line":
                p = p0 + s * (tgt - p0)
                a_new = rig.ik(p, app, seg.roll, pose, near=a_cmd)
                ad = (a_new - a_cmd) / dt
                add = (ad - ad_prev) / dt if k else zero
                a_cmd = a_new
            else:
                ad = add = zero
            ad_prev = ad
            step(a_cmd, ad, t, add)
            t += dt
        if tgt is not None:
            last_tgt = tgt
            tracking.append({"label": seg.label, "error_mm": round(float(np.linalg.norm(rig.aim() - tgt)) * 1000, 2)})
        elif seg.kind == "hold" and seg.body is None and last_tgt is not None and seg.label.startswith("dwell"):
            # A dwell is when the dose goes out: how far from the weed it lands.
            tracking.append(
                {"label": seg.label, "error_mm": round(float(np.linalg.norm(rig.aim() - last_tgt)) * 1000, 2)}
            )
        if seg.at_end:
            seg.at_end(rig)
    for _ in range(round(0.4 / dt)):
        step(a_cmd, zero, t)
        t += dt
    tau = np.array(log["tau"])
    return {
        "name": name,
        "bodies": bodies,
        "events": events,
        "tracking": tracking,
        "motors": list(rig.motors),
        "rated": rig.rated.tolist(),
        "peak": rig.peak.tolist(),
        "spring": rig.spring.tolist(),
        "summary": {
            "duration_s": round(t, 2),
            "peak_torque": np.round(np.abs(tau).max(axis=0), 2).tolist(),
            "peak_util": np.round(np.abs(tau).max(axis=0) / rig.peak, 3).tolist(),
            "rms_util": np.round(np.sqrt((tau**2).mean(axis=0)) / rig.rated, 3).tolist(),
            "min_margin_m": round(float(np.min([x for x in log["margin"] if x > -1.0])), 4),
            "max_tool_force": round(float(np.max(np.linalg.norm(np.array(log["tool_force"]), axis=1))), 1),
        },
        "log": log,
    }


# --- scenarios ------------------------------------------------------------------------

DOWN = np.array([0.0, 0.0, -1.0])
TRAY = np.array([-0.12, 0.0, 0.0])  # body frame, above the deck tray


def tray_drop(rig: Rig, h: float = 0.12) -> Array:
    return TRAY + np.array([0.0, 0.0, rig.hx.deck_z + h])


def to_tray(
    rig: Rig, h: float = 0.12, body: BodyPose | None = None, roll: float = 0.0, swing: float = 1.0
) -> list[Seg]:
    """Lift clear above the mount, then swing round to the tray: the yaw turns half a turn
    with the arm folded high, so nothing sweeps past the chassis. A heavy payload gets a
    slower swing (`swing`, seconds)."""
    via = np.array([rig.design.mount_x + 0.04, 0.0, rig.hx.deck_z + 0.30])
    return [
        Seg("lift clear", 0.8, target=via, in_body=True, body=body, roll=roll),
        Seg("swing to the deck tray", swing, target=tray_drop(rig, h), in_body=True, roll=roll),
    ]


def from_tray(rig: Rig) -> list[Seg]:
    via = np.array([rig.design.mount_x + 0.04, 0.0, rig.hx.deck_z + 0.30])
    return [Seg("swing back", 1.0, target=via, in_body=True)]


def ready(rig: Rig) -> Array:
    return rig.ik(np.array([0.45, 0.0, 0.25]), DOWN, 0.0, BodyPose())


WEED = (0.58, 0.04)


def sc_weed(rig: Rig, whole_body: bool) -> list[Seg]:
    """Pull a 60 N taproot. Arm only: the body stays put and the arm lifts. Whole body: the
    legs first slide and tip the body forward over the weed, which shortens the lever from
    the shoulder to the root, then the legs do the lifting while the arm holds still."""
    w = np.array([WEED[0], WEED[1], 0.035])
    # Slid 12 cm forward and tipped 8 degrees, the legs can still lift 9 cm (dz -0.05 to
    # +0.04) with every foot planted; Hexapod.pose_ok checks both ends.
    lift = 0.09
    lean = BodyPose(dz=-0.05, pitch=math.radians(8), dx=0.12)
    if whole_body:
        prep = [Seg("lean the body over the weed", 1.2, "hold", body=lean)]
        pull = Seg("lift with the legs", 1.6, "hold", body=BodyPose(dz=-0.05 + lift, pitch=math.radians(8), dx=0.12))
        body = lean
    else:
        prep = []
        pull = Seg("lift with the arm", 1.6, "line", target=w + [0, 0, lift])
        body = None
    return [
        *prep,
        Seg("reach over the weed", 1.2, target=w + [0, 0, 0.12], grip=0.04, body=body),
        Seg("descend", 0.6, "line", target=w + [0, 0, 0.005]),
        Seg("grip", 0.3, "hold", grip=0.004, at_end=lambda r: r.attach("weed")),
        pull,
        *to_tray(rig, body=BodyPose()),
        Seg("drop", 0.4, "hold", grip=0.04, at_start=lambda r: r.release("weed")),
        *from_tray(rig),
        Seg("ready", 0.8, motor=ready(rig)),
    ]


def sc_strawberries(rig: Rig) -> list[Seg]:
    berries = [(0.46, -0.16, 0.035), (0.62, 0.08, 0.05), (0.50, 0.20, 0.03)]
    segs: list[Seg] = []
    for i, b in enumerate(berries):
        b = np.array(b)
        app = np.array([0.5, 0.0, -0.866])
        name = f"berry{i}"
        segs += [
            Seg(f"approach berry {i + 1}", 1.0, target=b - 0.08 * app, approach=app, grip=0.035),
            Seg("close in", 0.4, "line", target=b, approach=app),
            Seg("grip (10 N max)", 0.25, "hold", grip=0.012, at_end=lambda r, n=name: r.attach(n)),
            Seg("pull off the stem", 0.5, "line", target=b - 0.06 * app, approach=app),
            *to_tray(rig, 0.1),
            Seg("release", 0.3, "hold", grip=0.035, at_start=lambda r, n=name: r.release(n)),
            *from_tray(rig),
        ]
    segs.append(Seg("ready", 0.9, motor=ready(rig)))
    return segs


def sc_rock(rig: Rig) -> list[Seg]:
    p = np.array([0.62, -0.12, 0.031])
    return [
        Seg("reach over the rock", 1.3, target=p + [0, 0, 0.14], grip=0.058, roll=math.pi / 2),
        Seg("descend", 0.6, "line", target=p + [0, 0, 0.0], roll=math.pi / 2),
        Seg("grip", 0.3, "hold", grip=0.03, roll=math.pi / 2, at_end=lambda r: r.attach("rock")),
        Seg("lift 1 kg at full reach", 0.8, "line", target=p + [0, 0, 0.15], roll=math.pi / 2),
        Seg("hold out", 1.0, "hold"),
        *to_tray(rig, 0.1, roll=math.pi / 2, swing=1.6),
        Seg("set down", 0.5, "hold", grip=0.058, at_start=lambda r: r.release("rock")),
        *from_tray(rig),
        Seg("ready", 0.8, motor=ready(rig)),
    ]


def sc_probe(rig: Rig) -> list[Seg]:
    p = np.array([0.42, 0.0, 0.0])

    def soil_on(r: Rig) -> None:
        r.soil = Soil()

    return [
        Seg("over the sample point", 1.2, target=p + [0, 0, 0.10], at_start=soil_on),
        Seg("touch the surface", 0.6, "line", target=p + [0, 0, 0.003]),
        Seg("press with the body", 1.8, "hold", body=BodyPose(dz=-0.065)),
        Seg("hold for the reading", 0.8, "hold"),
        Seg("rise with the body", 1.4, "hold", body=BodyPose()),
        Seg("ready", 1.0, motor=ready(rig)),
    ]


def sc_spray(rig: Rig) -> list[Seg]:
    pts = [
        (0.40, -0.20),
        (0.52, -0.08),
        (0.66, -0.22),
        (0.72, 0.02),
        (0.58, 0.14),
        (0.44, 0.06),
        (0.36, 0.24),
        (0.66, 0.24),
    ]
    segs = []
    for i, (x, y) in enumerate(pts):
        segs.append(Seg(f"dose weed {i + 1}", 0.38, target=np.array([x, y, 0.0])))
        segs.append(Seg("dwell 0.1 s", 0.1, "hold"))
    segs.append(Seg("ready", 0.8, motor=ready(rig)))
    return segs


def sc_tomato(rig: Rig) -> list[Seg]:
    p = np.array([0.60, 0.05, 0.80])
    side = np.array([1.0, 0.0, 0.0])
    up = BodyPose(dz=0.10, pitch=math.radians(-8))
    return [
        Seg("raise the body", 1.0, "hold", body=up),
        Seg("reach up beside the truss", 1.6, target=p - 0.10 * side, approach=side, grip=0.045, body=up),
        Seg("close in", 0.5, "line", target=p, approach=side),
        Seg("grip (9 N max)", 0.3, "hold", grip=0.022, at_end=lambda r: r.attach("tomato")),
        Seg("pull and twist off", 0.6, "line", target=p - 0.07 * side, approach=side),
        *to_tray(rig, 0.1, body=BodyPose()),
        Seg("release", 0.3, "hold", grip=0.045, at_start=lambda r: r.release("tomato")),
        *from_tray(rig),
        Seg("ready", 0.8, motor=ready(rig)),
    ]


@dataclass
class Scenario:
    id: str
    title: str
    tool: str
    objects: Callable[[], list[Obj]]
    script: Callable[[Rig], list[Seg]]
    standoff: float = 0.0
    note: str = ""


SCENARIOS = [
    Scenario(
        "weed_body",
        "Pull a 60 N taproot: body leans in, legs lift",
        "gripper",
        lambda: [weed("weed", WEED, 60.0)],
        lambda r: sc_weed(r, True),
        note="The legs slide the body 12 cm forward and tip it 8 degrees over the weed, then lift 9 cm while the arm holds.",
    ),
    Scenario(
        "weed_arm",
        "The same 60 N taproot pulled by the arm alone",
        "gripper",
        lambda: [weed("weed", WEED, 60.0)],
        lambda r: sc_weed(r, False),
        note="Same weed, body still: the root's pull acts on the arm's full lever.",
    ),
    Scenario(
        "strawberries",
        "Pick three strawberries into the deck tray",
        "gripper",
        lambda: [
            strawberry(f"berry{i}", p)
            for i, p in enumerate([(0.46, -0.16, 0.035), (0.62, 0.08, 0.05), (0.50, 0.20, 0.03)])
        ],
        sc_strawberries,
    ),
    Scenario(
        "rock",
        "Lift a 1 kg rock at full reach and stow it on the deck",
        "gripper",
        lambda: [rock("rock", (0.62, -0.12))],
        sc_rock,
        note="With the rock in hand the swing to the tray takes 1.6 s. At 1.0 s the yaw saturates and the rock lands 37 mm off.",
    ),
    Scenario(
        "soil_probe",
        "Press a soil probe 6 cm in with the body's weight",
        "probe",
        lambda: [],
        sc_probe,
    ),
    Scenario(
        "spray",
        "Micro-dose eight weeds as fast as the arm can",
        "sprayer",
        lambda: [
            marker(f"w{i}", p)
            for i, p in enumerate(
                [
                    (0.40, -0.20),
                    (0.52, -0.08),
                    (0.66, -0.22),
                    (0.72, 0.02),
                    (0.58, 0.14),
                    (0.44, 0.06),
                    (0.36, 0.24),
                    (0.66, 0.24),
                ]
            )
        ],
        sc_spray,
        standoff=0.08,
    ),
    Scenario(
        "tomato",
        "Pick a tomato 0.8 m up (body raised and tilted back)",
        "gripper",
        lambda: [tomato("tomato", (0.60, 0.05, 0.80))],
        sc_tomato,
    ),
]


def load_selected() -> tuple[ArmDesign, ArmActuators]:
    sel = yaml.safe_load((ROOT / "design" / "selected.yaml").read_text())
    chosen = sel["picks"][sel["chosen"]] if "picks" in sel else sel
    design = ArmDesign(**chosen["design"])
    a = resolve(chosen["actuators"], chosen.get("lever", "retail"))
    acts = ArmActuators(a["yaw"], a["shoulder"], a["elbow"], a["wrist_pitch"], a["tool_roll"], a.get("forearm_roll"))
    return design, acts


def scene_description(rig: Rig) -> list[dict]:
    """Every visible geom, for the viewer to rebuild the scene without MuJoCo."""
    m = rig.m
    types = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"}
    out = []
    for g in range(m.ngeom):
        t = types.get(int(m.geom_type[g]))
        if t is None or m.geom_rgba[g][3] == 0:
            continue
        out.append(
            {
                "body": int(m.geom_bodyid[g]) - 1,
                "name": m.geom(g).name,
                "type": t,
                "size": np.round(m.geom_size[g], 4).tolist(),
                "pos": np.round(m.geom_pos[g], 4).tolist(),
                "quat": np.round(m.geom_quat[g], 4).tolist(),
                "rgba": np.round(m.geom_rgba[g], 3).tolist(),
            }
        )
    return out


def main(argv: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    design, acts = load_selected()
    hx = Hexapod.load()
    tools = load_tools()
    wanted = set(argv) or {s.id for s in SCENARIOS}
    index = []
    for sc in SCENARIOS:
        if sc.id not in wanted:
            continue
        rig = Rig(design, acts, hx, tools[sc.tool], sc.objects(), sc.standoff)
        res = run(rig, sc.script(rig), sc.id)
        res.update(title=sc.title, note=sc.note, tool=sc.tool, scene=scene_description(rig))
        (OUT / f"{sc.id}.json").write_text(json.dumps(res, separators=(",", ":")))
        s = res["summary"]
        print(
            f"{sc.id:12s} {s['duration_s']:5.1f}s  peak/peak {s['peak_util']}  rms/rated {s['rms_util']}  "
            f"margin {s['min_margin_m']:.3f} m  tool {s['max_tool_force']} N  track {[x['error_mm'] for x in res['tracking']]}"
        )
        index.append({"id": sc.id, "title": sc.title, "summary": s, "tracking": res["tracking"], "note": sc.note})
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])

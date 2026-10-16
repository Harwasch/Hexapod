"""The optimizer's statics against first principles and against MuJoCo's closed chain."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from armdesign.evaluate import load_tasks, load_tools
from armdesign.mjcf import ArmActuators, build_mjcf
from armdesign.model import (
    ELBOW,
    SHOULDER,
    WRIST_PITCH,
    ArmDesign,
    BodyPose,
    Hexapod,
    Load,
    load_torques,
    planar_ik,
    solve_poses,
    stability_margin,
)
from armdesign.optimize import joint_options, load_catalog


@pytest.fixture(scope="module")
def hx() -> Hexapod:
    return Hexapod.load()


def _design(arch: str) -> ArmDesign:
    return ArmDesign(arch, 5, 0.31, 0.28, 0.18, 0.10)


def _poses(design: ArmDesign, hx: Hexapod, target: np.ndarray, approach: np.ndarray, tcp: float):
    return solve_poses(design, hx, [BodyPose()], target[None], approach[None], np.array([tcp]), np.array([0]))


def test_planar_ik_round_trip() -> None:
    rng = np.random.default_rng(1)
    th2 = rng.uniform(-0.5, 2.0, 200)
    fold = rng.uniform(0.4, 2.6, 200)
    th3 = th2 - fold
    l1, l2 = 0.31, 0.28
    r = l1 * np.cos(th2) + l2 * np.cos(th3)
    z = l1 * np.sin(th2) + l2 * np.sin(th3)
    a, b, ok = planar_ik(r, z, l1, l2)
    assert ok.all()
    np.testing.assert_allclose(a, th2, atol=1e-9)
    np.testing.assert_allclose(b, th3, atol=1e-9)


def test_tcp_lands_on_target(hx: Hexapod) -> None:
    d = _design("parallelogram")
    target = np.array([0.6, 0.1, 0.05])
    p = _poses(d, hx, target, np.array([0.0, 0.0, -1.0]), 0.2)
    assert p.ok[0]
    np.testing.assert_allclose(p.T[0], target, atol=1e-9)


def test_parallelogram_splits_what_a_direct_elbow_sums(hx: Hexapod) -> None:
    """A tip force's moment about the shoulder is shared by the two motors in the
    parallelogram and carried by the shoulder alone with a direct elbow."""
    target = np.array([0.6, 0.0, 0.0])
    f = np.array([[0.0, 0.0, -60.0]])
    taus = {}
    for arch in ("parallelogram", "direct"):
        d = _design(arch)
        p = _poses(d, hx, target, np.array([0.0, 0.0, -1.0]), 0.2)
        taus[arch] = load_torques(d, p, Load("tool", 0.2, 0.0, f))[0]
    para, direct = taus["parallelogram"], taus["direct"]
    assert para[SHOULDER] + para[ELBOW] == pytest.approx(direct[SHOULDER], rel=1e-9)
    assert para[ELBOW] == pytest.approx(direct[ELBOW], rel=1e-9)
    assert abs(para[SHOULDER]) < abs(direct[SHOULDER])


@pytest.mark.parametrize("arch", ["parallelogram", "direct"])
def test_statics_match_mujoco(hx: Hexapod, arch: str) -> None:
    """Hold a pose against a tip force in MuJoCo (gravity off, body fixed) and read the
    motor torques; they must match load_torques."""
    cat = {a.id: a for a in load_catalog()}
    acts = ArmActuators(cat["rs_00"], cat["sw_gim8108_8"], cat["sw_gim8108_8"], cat["rs_edulite_05"], cat["sts3215"])
    d = _design(arch)
    tool = load_tools()["probe"]
    xml = build_mjcf(d, acts, hx, tool).replace('<freejoint name="root"/>', "")
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.gravity[:] = 0
    data = mujoco.MjData(m)
    th2, th3, q4 = math.radians(40), math.radians(-35), math.radians(-45)
    names = ["arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch", "arm_tool_roll"]
    qa = {n: m.jnt_qposadr[m.joint(n).id] for n in names}
    va = {n: m.jnt_dofadr[m.joint(n).id] for n in names}
    data.qpos[qa["arm_shoulder"]] = th2
    data.qpos[qa["arm_elbow"]] = th3 - th2
    data.qpos[qa["arm_wrist_pitch"]] = q4
    if arch == "parallelogram":
        data.qpos[m.jnt_qposadr[m.joint("arm_crank").id]] = th3
        data.qpos[m.jnt_qposadr[m.joint("arm_rod").id]] = th2 - th3
    mujoco.mj_forward(m, data)
    tcp = data.site_xpos[m.site("tcp").id].copy()
    force = np.array([0.0, 0.0, -30.0])  # keeps the direct shoulder under its 22 Nm peak
    flange = m.body("flange").id
    target = np.array([th2, th3, q4])
    ctrl = {n: m.actuator(f"m_{n}").id for n in ("shoulder", "elbow", "wrist_pitch")}
    crank_v = m.jnt_dofadr[m.joint("arm_crank").id] if arch == "parallelogram" else None
    for _ in range(4000):
        data.xfrc_applied[flange, :3] = force
        data.xfrc_applied[flange, 3:] = np.cross(tcp - data.xipos[flange], force)
        q = np.array(
            [
                data.qpos[qa["arm_shoulder"]],
                data.qpos[qa["arm_shoulder"]] + data.qpos[qa["arm_elbow"]],
                data.qpos[qa["arm_wrist_pitch"]],
            ]
        )
        qd = np.array(
            [
                data.qvel[va["arm_shoulder"]],
                data.qvel[va["arm_shoulder"]] + data.qvel[va["arm_elbow"]],
                data.qvel[va["arm_wrist_pitch"]],
            ]
        )
        tau = np.array([400.0, 300.0, 60.0]) * (target - q) - np.array([8.0, 6.0, 0.6]) * qd
        if arch == "direct":
            tau = np.array([tau[0] + tau[1], tau[1], tau[2]])
        data.ctrl[ctrl["shoulder"]], data.ctrl[ctrl["elbow"]], data.ctrl[ctrl["wrist_pitch"]] = tau
        mujoco.mj_step(m, data)
        tcp = data.site_xpos[m.site("tcp").id].copy()
    measured = np.array([data.actuator_force[ctrl[k]] for k in ("shoulder", "elbow", "wrist_pitch")])
    p = _poses(d, hx, tcp, data.xmat[flange].reshape(3, 3)[:, 0].copy(), d.wrist_len + tool.length)
    model = load_torques(d, p, Load("tool", d.wrist_len + tool.length, 0.0, force[None]))[0]
    expected = np.array([model[SHOULDER], model[ELBOW], model[WRIST_PITCH]])
    np.testing.assert_allclose(measured, expected, rtol=0.03, atol=0.15)
    _ = crank_v


def test_lone_robot_margin_is_distance_to_nearest_edge(hx: Hexapod) -> None:
    margin, on_feet = stability_margin(hx, [BodyPose()], np.array([0]), [])
    # Centre of mass at the origin; nearest edges are the two front-to-middle sides.
    feet = hx.feet[:, :2]
    best = min(
        abs(_cross2(feet[(i + 1) % 6] - feet[i], -feet[i])) / np.linalg.norm(feet[(i + 1) % 6] - feet[i])
        for i in range(6)
    )
    assert margin[0] == pytest.approx(best, rel=1e-9)
    assert on_feet[0] == pytest.approx(1.0)


def test_catalog_and_options() -> None:
    cat = load_catalog()
    assert len(cat) >= 20
    assert {a.backdrivable for a in cat} <= {"yes", "no", "partly"}
    field_shoulder = joint_options(cat, "shoulder", "field")
    assert field_shoulder and all(a.kind == "qdd" and a.force_sense == "yes" for a in field_shoulder)


def test_task_library_loads() -> None:
    tasks = load_tasks()
    assert len({t.id for t in tasks}) == len(tasks)
    assert all(t.tool in load_tools() for t in tasks)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])

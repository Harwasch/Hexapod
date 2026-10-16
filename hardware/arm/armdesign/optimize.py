"""Search the design space and write the cost-versus-capability study.

    uv run python -m armdesign.optimize            # full study (about 10 minutes, 4 cores)
    uv run python -m armdesign.optimize --quick    # a smaller grid for a smoke run

Stage A sweeps geometry (DOF, elbow drive, link lengths, mount position, pedestal) for
kinematic coverage: reach, approach, collisions and tip-over with reference masses.
Stage B takes the geometries on the coverage/size frontier, adds the gravity-spring
option, and enumerates every actuator assignment from data/actuators.yaml against the
joint torques each task point needs. Output: out/study.json (everything the viewer and
the docs read) and out/selected.yaml (the recommended design).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from armdesign.evaluate import (
    PEAK_DERATE,
    RATED_DERATE,
    SLOTS,
    TorqueTable,
    kinematic_table,
    load_tasks,
    load_tools,
    make_instances,
    torque_table,
)
from armdesign.model import (
    ELBOW,
    FOREARM_ROLL,
    JOINTS,
    ROOT,
    SHOULDER,
    TOOL_ROLL,
    WRIST_PITCH,
    YAW,
    Actuator,
    ArmDesign,
    Hexapod,
    MassModel,
    body_pose_grid,
    load_yaml,
)

OUT = ROOT / "out"
Array = np.ndarray


# --- catalog and costs -----------------------------------------------------------------


def _yn(v: object) -> str:
    # YAML 1.1 reads a bare yes/no as a boolean.
    return "yes" if v is True else "no" if v is False else str(v)


def load_catalog() -> list[Actuator]:
    out = []
    for a in load_yaml("actuators.yaml")["actuators"]:
        if a.get("exclude"):
            continue
        out.append(
            Actuator(
                id=a["id"],
                name=a["name"],
                kind=a["kind"],
                rated=float(a["rated_nm"]),
                peak=float(a["peak_nm"]),
                mass=float(a["mass_kg"]),
                speed=float(a["speed_rpm"]) * 2 * math.pi / 60,
                price=float(a["price_usd"]),
                dia=float(a["dia_mm"]) / 1000,
                length=float(a["len_mm"]) / 1000,
                backdrivable=_yn(a["backdrivable"]),
                interface=str(a["interface"]),
                backlash_deg=float(a["backlash_deg"]),
                force_sense=_yn(a["force_sense"]),
                output_encoder=_yn(a["output_encoder"]),
                ip=str(a["ip"]),
                notes=str(a.get("notes", "")),
            )
        )
    return out


def load_costs() -> dict[str, float]:
    return {k: float(v) for k, v in load_yaml("costs.yaml")["usd"].items()}


def fixed_cost(design: ArmDesign, costs: dict[str, float]) -> dict[str, float]:
    """Everything but the actuators, itemised."""
    c = {
        "aluminium tube": costs["tube_per_m"] * (design.l1 + design.l2 + 0.08),
        "printed parts": costs["printed_parts"],
        "bearings": costs["bearings"],
        "fasteners": costs["fasteners"],
        "tool changer": costs["tool_changer"],
        "gripper (servo + fin-ray fingers)": costs["gripper"],
        "wrist camera + ToF": costs["wrist_sensing"],
        "harness + connectors": costs["harness"],
        "seals + boots": costs["seals"],
    }
    if design.arch == "parallelogram":
        c["parallelogram rod + rod ends"] = costs["parallelogram_rod"]
    if design.spring:
        c["gravity springs"] = costs["spring_each"] * (2 if design.arch == "parallelogram" else 1)
    # The gripper is a bus servo, so the adapter is always there.
    c["servo bus adapter"] = costs["servo_bus_adapter"]
    return c


def fixed_mass(design: ArmDesign, mm: MassModel) -> float:
    m = mm.tube_kg_per_m * (design.l1 + design.l2) + mm.turret_kg + mm.elbow_housing_kg
    m += mm.wrist_housing_kg + mm.flange_kg
    if design.arch == "parallelogram":
        m += mm.rod_kg
    if design.spring:
        m += mm.spring_kg * (2 if design.arch == "parallelogram" else 1)
    return m


# What a joint must be in each tier. "field" is the outdoor-robot bar: the shoulder and
# elbow can feel contact through motor current and survive being back-driven by a fall or
# a branch, the yaw is backdrivable, and nothing at the base is a hobby part.
TIERS = {
    "any": {},
    "field": {
        "yaw": lambda a: a.kind == "qdd",
        "shoulder": lambda a: a.kind == "qdd" and a.force_sense == "yes",
        "elbow": lambda a: a.kind == "qdd" and a.force_sense == "yes",
    },
}
# Root-sum-square tip error from gear backlash, open loop. Micro-dosing and stamping want
# +-2-5 mm at the tool; the wrist camera closes the last few millimetres.
FIELD_MAX_BACKLASH_MM = 6.0
# The arm's own mass budget on a medium (8-20 kg) hexapod, tools excluded.
MAX_ARM_KG = 2.5


def joint_options(catalog: list[Actuator], joint: str, tier: str = "any") -> list[Actuator]:
    """Candidates for one joint, without the ones another candidate beats on every count."""
    rule = TIERS[tier].get(joint)
    if rule is not None:
        catalog = [a for a in catalog if rule(a)]
    if joint in ("wrist_pitch",):
        pool = [a for a in catalog if a.mass <= 0.45]
    elif joint in ("forearm_roll", "tool_roll"):
        pool = [a for a in catalog if a.mass <= 0.35]
    else:
        pool = list(catalog)

    def beats(x: Actuator, y: Actuator) -> bool:
        ge = x.price <= y.price and x.rated >= y.rated and x.peak >= y.peak
        ge &= x.mass <= y.mass and x.speed >= y.speed
        gt = x.price < y.price or x.rated > y.rated or x.peak > y.peak or x.mass < y.mass
        return ge and gt

    return [y for y in pool if not any(beats(x, y) for x in pool if x is not y)]


# --- stage A ----------------------------------------------------------------------------

_CTX: dict[str, object] = {}


def _init(coarse: bool) -> None:
    hx = Hexapod.load()
    tasks = load_tasks()
    _CTX.update(
        hx=hx,
        tasks=tasks,
        tools=load_tools(),
        inst=make_instances(tasks, hx, coarse=True),
        bp=body_pose_grid(hx, fine=not coarse),
    )


def _coverage(feasible: Array) -> tuple[float, dict[str, float]]:
    tasks = _CTX["tasks"]
    inst = _CTX["inst"]
    per = {}
    for i, t in enumerate(tasks):
        sel = inst.task == i
        per[t.id] = float(feasible[sel].mean()) if np.any(sel) else 0.0
    w = np.array([t.weight for t in tasks])
    score = float(sum(per[t.id] * t.weight for t in tasks) / w.sum())
    return score, per


def _kin_job(design: ArmDesign) -> dict:
    kin = kinematic_table(design, _CTX["tasks"], _CTX["tools"], _CTX["inst"], _CTX["hx"], _CTX["bp"])
    score, per = _coverage(kin.feasible)
    return {"design": asdict(design), "kin_score": score, "kin_tasks": per}


def stage_a(quick: bool, workers: int) -> list[dict]:
    lengths = np.round(np.arange(0.22, 0.41, 0.06 if quick else 0.03), 3)
    mounts = (0.12, 0.24) if quick else (0.10, 0.18, 0.26)
    peds = (0.06, 0.14) if quick else (0.05, 0.10, 0.16)
    designs = [
        ArmDesign(arch, dof, float(l1), float(l2), m, p)
        for dof in (5, 6)
        for arch in ("parallelogram", "direct")
        for l1 in lengths
        for l2 in lengths
        for m in mounts
        for p in peds
    ]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(True,)) as ex:
        return list(ex.map(_kin_job, designs, chunksize=4))


def frontier(rows: list[dict], keys: list[tuple[str, int]]) -> list[dict]:
    """Rows not dominated on keys; sign +1 maximises, -1 minimises."""
    vals = np.array([[s * r[k] for k, s in keys] for r in rows])
    keep = []
    for i, v in enumerate(vals):
        dom = np.all(vals >= v, axis=1) & np.any(vals > v, axis=1)
        if not np.any(dom):
            keep.append(rows[i])
    return keep


# --- stage B ----------------------------------------------------------------------------


@dataclass
class Assignment:
    yaw: Actuator
    shoulder: Actuator
    elbow: Actuator
    wrist_pitch: Actuator
    forearm_roll: Actuator | None
    tool_roll: Actuator

    def by_joint(self) -> dict[str, Actuator | None]:
        return {j: getattr(self, j) for j in JOINTS}


def _joint_ok(tt: TorqueTable, slot_mass: Array, j: int, rated: Array, peak: Array) -> Array:
    """(O, I, K) for O actuator options on joint j, with the given moving masses."""
    static = tt.base_static[..., j] + np.einsum("s,sik->ik", slot_mass, tt.slot_static[..., j])
    inert = tt.base_inert[..., j] + np.einsum("s,sik->ik", slot_mass, tt.slot_inert[..., j])
    with_ext = static + tt.ext[..., j]
    hold = tt.hold[:, None]
    sustained = np.where(hold, np.abs(with_ext), np.abs(static))
    burst = np.where(hold, np.abs(static), np.abs(with_ext)) + tt.ext_abs[..., j] + inert
    ok = (sustained[None] <= RATED_DERATE * rated[:, None, None]) & (burst[None] <= PEAK_DERATE * peak[:, None, None])
    return ok & tt.valid[None]


def enumerate_actuators(
    design: ArmDesign, tt: TorqueTable, weights: Array, catalog: list[Actuator], tier: str = "any"
) -> list[dict]:
    """Every actuator assignment's capability score; returns the cost/score frontier."""
    opts = {j: joint_options(catalog, j, tier) for j in JOINTS}
    cap = lambda xs: (np.array([a.rated for a in xs]), np.array([a.peak for a in xs]))  # noqa: E731
    results = []
    distal_rolls = opts["tool_roll"]
    for wp, tr in itertools.product(opts["wrist_pitch"], distal_rolls):
        # A 6-DOF wrist uses the same part for both rolls.
        fr = tr if design.dof == 6 else None
        base_m = {"elbow": 0.0, "wrist_pitch": wp.mass, "forearm_roll": fr.mass if fr else 0.0, "tool_roll": tr.mass}
        m0 = np.array([base_m[s] for s in SLOTS])
        ok = tt.valid.copy()
        for j, a in ((WRIST_PITCH, wp), (TOOL_ROLL, tr)) + (((FOREARM_ROLL, fr),) if fr else ()):
            r, p = cap([a])
            ok &= _joint_ok(tt, m0, j, r, p)[0]
        if not np.any(ok):
            continue
        el_r, el_p = cap(opts["elbow"])
        ok_el = _joint_ok(tt, m0, ELBOW, el_r, el_p)  # (E, I, K): the elbow motor never loads itself
        sh_r, sh_p = cap(opts["shoulder"])
        yw_r, yw_p = cap(opts["yaw"])
        if design.arch == "direct":
            ok_sh, ok_yw = [], []
            for e in opts["elbow"]:
                m = m0.copy()
                m[SLOTS.index("elbow")] = e.mass
                ok_sh.append(_joint_ok(tt, m, SHOULDER, sh_r, sh_p))
                ok_yw.append(_joint_ok(tt, m, YAW, yw_r, yw_p))
            ok_sh_a = np.stack(ok_sh)[:, :, None]  # (E, S, 1, I, K)
            ok_yw_a = np.stack(ok_yw)[:, None, :]  # (E, 1, Y, I, K)
        else:
            ok_sh_a = _joint_ok(tt, m0, SHOULDER, sh_r, sh_p)[None, :, None]
            ok_yw_a = _joint_ok(tt, m0, YAW, yw_r, yw_p)[None, None, :]
        full = ok[None, None, None] & ok_el[:, None, None] & ok_sh_a & ok_yw_a
        served = np.any(full, axis=-1)  # (E, S, Y, I)
        scores = served.astype(float) @ weights
        price = lambda xs: np.array([a.price for a in xs])  # noqa: E731
        cost = (
            price(opts["elbow"])[:, None, None]
            + price(opts["shoulder"])[None, :, None]
            + price(opts["yaw"])[None, None, :]
            + wp.price
            + tr.price
            + (fr.price if fr else 0.0)
        )
        # Only this distal set's own cost/score frontier can matter.
        flat_s, flat_c = scores.ravel(), cost.ravel()
        order = np.lexsort((-flat_s, flat_c))
        best = -1.0
        for k in order:
            if flat_s[k] > best + 1e-9:
                best = flat_s[k]
                ei, si, yi = np.unravel_index(k, scores.shape)
                asg = Assignment(opts["yaw"][yi], opts["shoulder"][si], opts["elbow"][ei], wp, fr, tr)
                results.append({"assignment": asg, "score": float(flat_s[k])})
    return results


def precision_speed(design: ArmDesign, acts: dict[str, Actuator]) -> tuple[float, float]:
    """Tip error from gear backlash (mm, root-sum-square over the joints: the joints'
    play does not all line up at once, and gravity preloads most of it) and the slowest
    proximal joint's tip speed (m/s), with the gripper on and the arm at 85% of stretch."""
    tool = 0.11
    reach = 0.85 * (design.l1 + design.l2) + design.wrist_len + tool
    levers = {
        "yaw": reach,
        "shoulder": reach,
        "elbow": design.l2 + design.wrist_len + tool,
        "wrist_pitch": design.wrist_len + tool,
        "forearm_roll": 0.03,
        "tool_roll": 0.03,
    }
    err = math.sqrt(sum((math.radians(a.backlash_deg) * levers[j]) ** 2 for j, a in acts.items()))
    speed = min(a.speed * levers[j] for j, a in acts.items() if j in ("yaw", "shoulder", "elbow"))
    return err * 1000, speed


def summarise(
    design: ArmDesign, asg: Assignment, score: float, costs: dict[str, float], mm: MassModel, tier: str = "any"
) -> dict:
    acts = {j: a for j, a in asg.by_joint().items() if a is not None}
    backlash_mm, tip_speed = precision_speed(design, acts)
    fixed = fixed_cost(design, costs)
    act_cost = sum(a.price for a in acts.values())
    mass = fixed_mass(design, mm) + sum(a.mass for a in acts.values())
    return {
        "design": asdict(design),
        "label": design.label,
        "actuators": {j: a.id for j, a in acts.items()},
        "score": score,
        "cost": round(act_cost + sum(fixed.values()), 2),
        "actuator_cost": round(act_cost, 2),
        "fixed_cost": {k: round(v, 2) for k, v in fixed.items()},
        "mass": round(mass, 3),
        "tier": tier,
        "backlash_mm": round(backlash_mm, 2),
        "tip_speed": round(tip_speed, 2),
        "force_sensing": all(acts[j].force_sense == "yes" for j in ("shoulder", "elbow")),
        "kinds": sorted({a.kind for a in acts.values()}),
    }


def _stage_b_job(args: tuple[ArmDesign, int]) -> list[dict]:
    design, _ = args
    tasks, tools, inst, hx, bp = (_CTX[k] for k in ("tasks", "tools", "inst", "hx", "bp"))
    kin = kinematic_table(design, tasks, tools, inst, hx, bp)
    tt = torque_table(design, tasks, tools, inst, kin)
    w_task = np.array([t.weight for t in tasks])
    counts = np.bincount(inst.task, minlength=len(tasks))
    weights = w_task[inst.task] / counts[inst.task] / w_task.sum()
    catalog = load_catalog()
    costs = load_costs()
    mm = MassModel()
    out = []
    for tier in TIERS:
        rows = [
            summarise(design, r["assignment"], r["score"], costs, mm, tier)
            for r in enumerate_actuators(design, tt, weights, catalog, tier)
        ]
        rows = [r for r in rows if r["mass"] <= MAX_ARM_KG]
        if tier == "field":
            rows = [r for r in rows if r["backlash_mm"] <= FIELD_MAX_BACKLASH_MM]
        out.extend(frontier(rows, [("score", 1), ("cost", -1)]))
    return out


def stage_b(finalists: list[ArmDesign], workers: int) -> list[dict]:
    jobs = [(d, i) for i, d in enumerate(finalists)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(False,)) as ex:
        out = []
        for rows in ex.map(_stage_b_job, jobs):
            out.extend(rows)
    return out


def pick_finalists(rows: list[dict], per_group: int) -> list[ArmDesign]:
    finalists = []
    for dof in (5, 6):
        for arch in ("parallelogram", "direct"):
            grp = [r for r in rows if r["design"]["dof"] == dof and r["design"]["arch"] == arch]
            for r in grp:
                d = r["design"]
                r["size"] = d["l1"] + d["l2"]
                r["ped"] = d["pedestal"]
            front = frontier(grp, [("kin_score", 1), ("size", -1), ("ped", -1)])
            front.sort(key=lambda r: -r["kin_score"])
            # Spread the picks along the frontier rather than taking the top few.
            if len(front) > per_group:
                idx = np.linspace(0, len(front) - 1, per_group).round().astype(int)
                front = [front[i] for i in sorted(set(idx))]
            for r in front:
                for spring in (False, True):
                    finalists.append(ArmDesign(**{**r["design"], "spring": spring}))
    return finalists


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-group", type=int, default=7)
    ap.add_argument("--reuse-a", action="store_true", help="take stage A from the last out/study.json")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    if args.reuse_a:
        rows_a = json.loads((OUT / "study.json").read_text())["stage_a"]
    else:
        rows_a = stage_a(args.quick, args.workers)
    print(f"stage A: {len(rows_a)} geometries in {time.time() - t0:.0f}s")
    finalists = pick_finalists(rows_a, 3 if args.quick else args.per_group)
    t1 = time.time()
    rows_b = stage_b(finalists, args.workers)
    print(f"stage B: {len(finalists)} designs, {len(rows_b)} frontier points in {time.time() - t1:.0f}s")
    fronts = {}
    for tier in TIERS:
        f = frontier([r for r in rows_b if r["tier"] == tier], [("score", 1), ("cost", -1)])
        f.sort(key=lambda r: r["cost"])
        fronts[tier] = f
        print(f"frontier [{tier}]: {len(f)} designs")
        for r in f:
            print(
                f"  ${r['cost']:7.0f}  score {r['score']:.3f}  {r['mass']:.2f} kg  "
                f"{r['backlash_mm']:.1f} mm  {r['label']}  {r['actuators']}"
            )
    study = {
        "generated": time.strftime("%Y-%m-%d"),
        "hexapod": load_yaml("hexapod.yaml")["name"],
        "stage_a": rows_a,
        "stage_b": rows_b,
        "frontier": fronts,
    }
    (OUT / "study.json").write_text(json.dumps(study, indent=1))


if __name__ == "__main__":
    main()

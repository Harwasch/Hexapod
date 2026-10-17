"""Choose the design from the study and package everything the viewer and the docs read.

    uv run python -m armdesign.report select   # out/study.json -> design/selected.yaml
    uv run python -m armdesign.report          # selected design -> viewer/data/*, viewer/assets/*

The pick: on the "field" frontier (backdrivable, force-sensing shoulder and elbow, within
the mass budget and the backlash limit), the cheapest design within 5% of the best score
any field design reaches. Past that knee each extra point of capability costs $25-100.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import sys

import numpy as np
import yaml

from armdesign.evaluate import (
    PEAK_DERATE,
    RATED_DERATE,
    SLOTS,
    feasible_with,
    fixed_loads,
    kinematic_table,
    load_tasks,
    load_tools,
    make_instances,
    slot_loads,
    torque_table,
)
from armdesign.model import (
    JOINTS,
    ROOT,
    Actuator,
    ArmDesign,
    BodyPose,
    Hexapod,
    Load,
    MassModel,
    body_pose_grid,
    load_torques,
    load_yaml,
    solve_poses,
    spring_torques,
)
from armdesign.optimize import fixed_cost, fixed_mass, load_costs, precision_speed

OUT = ROOT / "out"
VIEWER = ROOT / "viewer"
# Points of capability (percent of task points served) per $100 below which an upgrade
# is no longer worth it: where the frontier flattens.
RETURN_FLOOR = 2.0


FA1 = {
    "design": {
        "arch": "parallelogram",
        "dof": 5,
        "l1": 0.37,
        "l2": 0.28,
        "mount_x": 0.26,
        "pedestal": 0.05,
        "spring": True,
        "wrist_len": 0.09,
    },
    "actuators": {
        "yaw": "rs_edulite_05",
        "shoulder": "sw_gim8108_8",
        "elbow": "sw_gim6010_8",
        "wrist_pitch": "hx_35hm",
        "tool_roll": "hx_65hm",
    },
    "lever": "retail",
}
# What v1 priced FA-1 at: list prices, before duty and freight, with a $10 seals line.
FA1_LIST_COST = 607.22
V1_SEALS_USD = 10.0


def score_of(entry: dict) -> float:
    """A design's capability under this study's rules (body poses, leg reach, speed)."""
    from armdesign.parts import resolve

    design = ArmDesign(**entry["design"])
    acts = resolve(entry["actuators"], entry.get("lever", "retail"))
    return workspace(design, acts, Hexapod.load())["score"]


def stage_limit(a: Actuator) -> float | None:
    """The torque a part's reduction stage can carry, or None without a stage."""
    from armdesign.parts import load_stages

    sid = a.id.partition("+")[2]
    return round({s.id: s for s in load_stages()}[sid].max_torque, 2) if sid else None


def part_names(sel: dict) -> dict[str, str]:
    """Display names of every part the picks and FA-1 use, by base id."""
    from armdesign.parts import load_catalog

    cat = {a.id: a.name for a in load_catalog("all")}
    entries = [*sel.get("picks", {}).values(), sel.get("fa1") or {}]
    ids = {pid.partition("+")[0] for e in entries for pid in e.get("actuators", {}).values()}
    return {i: cat[i] for i in sorted(ids) if i in cat}


def cost_of(design: ArmDesign, acts: dict[str, Actuator]) -> float:
    """Landed cost of one arm, the same sum the study makes."""
    fixed = fixed_cost(design, load_costs(), any(a.needs_driver for a in acts.values()))
    return round(sum(a.price for a in acts.values()) + sum(fixed.values()), 2)


def choose(study: dict, fa1_score: float, fa1_cost: float) -> dict[str, dict]:
    """Three picks on the field-grade frontiers.

    lite: the cheapest that matches FA-1's capability (within half a point, the sampling
    noise of the coverage score). balanced (FA-2): where returns diminish, the cheapest
    design beyond which no upgrade buys RETURN_FLOOR points of capability per $100. Both
    come from bought parts only (no custom board, whose engineering 10 arms never repay).
    max: the most capability at FA-1's price, any lever."""
    bought = sorted(study["frontier"]["bought"]["field"], key=lambda r: r["cost"])
    every = study["frontier"]["all"]["field"]

    def worth_upgrading(i: int) -> bool:
        p = bought[i]
        return any(
            (q["score"] - p["score"]) * 100 / max(q["cost"] - p["cost"], 1e-6) * 100 >= RETURN_FLOOR
            for q in bought[i + 1 :]
        )

    knee = next(r for i, r in enumerate(bought) if not worth_upgrading(i))
    return {
        "lite": min((r for r in bought if r["score"] >= fa1_score - 0.005), key=lambda r: r["cost"]),
        "balanced": knee,
        "max": max((r for r in every if r["cost"] <= fa1_cost + 1e-6), key=lambda r: (r["score"], -r["cost"])),
    }


def select() -> None:
    from armdesign.parts import resolve

    study = json.loads((OUT / "study.json").read_text())
    fa1_score = score_of(FA1)
    fa1_acts = resolve(FA1["actuators"], "retail")
    fa1_design = ArmDesign(**FA1["design"])
    fa1_cost = cost_of(fa1_design, fa1_acts)
    fa1_sourced = cost_of(fa1_design, resolve(FA1["actuators"], "sourcing"))
    fa1_backlash, _ = precision_speed(fa1_design, fa1_acts)
    fa1_mass = fixed_mass(fa1_design, MassModel()) + sum(a.mass for a in fa1_acts.values())
    picks = choose(study, fa1_score, fa1_cost)
    keep = ("design", "actuators", "stages", "sources", "lever", "score", "cost", "mass", "backlash_mm", "label")
    sel = {
        "why": (
            "FA-2 is the balanced pick: on the field-grade frontier of bought parts (sourcing and "
            "printed reductions, no custom board), the cheapest design beyond which no upgrade buys "
            f"{RETURN_FLOOR:g} points of capability per $100. lite matches FA-1's capability for the "
            "least money; max is the most capability at FA-1's price with any lever."
        ),
        "chosen": "balanced",
        "fa1": {
            **FA1,
            "score_v2": round(fa1_score, 4),
            "cost": fa1_cost,
            "cost_list": FA1_LIST_COST,
            "cost_sourced": fa1_sourced,
            # v1 costed seals at $10; the field kit every design now carries is costs.seals.
            "cost_field_kit": round(load_costs()["seals"] - V1_SEALS_USD, 2),
            "parts": {
                j: {"name": a.name, "price": a.price, "rated": round(a.rated, 2), "peak": round(a.peak, 2)}
                for j, a in fa1_acts.items()
            },
            "mass": round(fa1_mass, 3),
            "backlash_mm": round(fa1_backlash, 2),
        },
        "picks": {k: {x: v[x] for x in keep if x in v} for k, v in picks.items()},
    }
    (ROOT / "design").mkdir(exist_ok=True)
    with (ROOT / "design" / "selected.yaml").open("w") as f:
        f.write("# Written by `python -m armdesign.report select` from out/study.json.\n")
        yaml.safe_dump(sel, f, sort_keys=False)
    print(
        f"FA-1 under these rules: score {fa1_score:.3f}, ${fa1_cost} landed (${fa1_sourced} same parts, best sources)"
    )
    for k, v in picks.items():
        print(f"{k:9s} ${v['cost']} score {v['score']:.3f} {v['label']} {v['actuators']}")


# --- metrics of the selected design -------------------------------------------------------


def payload_at_reach(design: ArmDesign, acts: dict, hx: Hexapod, mm: MassModel, frac: float = 0.95) -> dict:
    """Largest payload at the gripper with the arm stretched horizontally (wrist centre at
    frac of full reach, tool level), continuous and for a short lift."""
    tools = load_tools()
    gr = tools["gripper"]
    tcp = design.wrist_len + gr.length
    s = np.array([design.mount_x, 0.0, hx.center_height + hx.deck_z + design.pedestal])
    reach = frac * (design.l1 + design.l2)
    target = s + np.array([reach + tcp, 0.0, 0.0])
    p = solve_poses(design, hx, [BodyPose()], target[None], np.array([[1.0, 0.0, 0.0]]), np.array([tcp]), np.array([0]))
    arm = np.zeros(6)
    for ld in fixed_loads(design, mm):
        arm += load_torques(design, p, ld)[0] - spring_torques(design, p, ld)[0]
    arm += load_torques(design, p, Load("tool", design.wrist_len + gr.cg, gr.mass))[0]
    arm -= spring_torques(design, p, Load("tool", design.wrist_len + gr.cg, gr.mass))[0]
    for name, ld in slot_loads(design, mm).items():
        m = acts[name].mass
        arm += m * (load_torques(design, p, ld)[0] - spring_torques(design, p, ld)[0])
    per_kg = load_torques(design, p, Load("tool", tcp, 1.0))[0]
    out = {}
    for label, cap, derate in (("continuous_kg", "rated", RATED_DERATE), ("peak_kg", "peak", PEAK_DERATE)):
        best = math.inf
        for j, name in enumerate(JOINTS):
            if name not in acts or abs(per_kg[j]) < 1e-9:
                continue
            limit = derate * getattr(acts[name], cap)
            best = min(best, (limit - abs(arm[j])) / abs(per_kg[j]))
        out[label] = round(max(best, 0.0), 2)
    out["reach_m"] = round(reach + tcp, 3)
    return out


def leg_fit(acts: dict, hx: Hexapod) -> dict:
    """Could the hexapod's legs use the arm's shoulder actuator? Static femur torque with a
    tripod (3 feet) carrying robot + arm, foot 0.20 m out from the femur axis, x2 for gait."""
    a = acts["shoulder"]
    need_cont = (hx.mass + 2.5) * 9.81 / 3 * 0.20
    return {
        "actuator": a.name,
        "femur_need_nm": round(need_cont, 1),
        "femur_peak_need_nm": round(2 * need_cont, 1),
        "rated_nm": a.rated,
        "peak_nm": a.peak,
        "fits": a.rated >= need_cont and a.peak >= 2 * need_cont,
    }


def workspace(design: ArmDesign, acts: dict, hx: Hexapod) -> dict:
    """Every task point and whether the selected design (actuators included) serves it."""
    tasks, tools = load_tasks(), load_tools()
    inst = make_instances(tasks, hx, coarse=True)
    bp = body_pose_grid(hx, fine=True)
    kin = kinematic_table(design, tasks, tools, inst, hx, bp)
    tt = torque_table(design, tasks, tools, inst, kin)
    slot_mass = np.array([[acts[s].mass if s in acts else 0.0 for s in SLOTS]])
    rated = np.array([[acts[j].rated if j in acts else np.inf for j in JOINTS]])
    peak = np.array([[acts[j].peak if j in acts else np.inf for j in JOINTS]])
    served = feasible_with(tt, slot_mass, rated, peak)[0]
    per_task = []
    for i, t in enumerate(tasks):
        sel = inst.task == i
        per_task.append(
            {
                "id": t.id,
                "name": t.name,
                "category": t.category,
                "weight": t.weight,
                "reach": round(float(kin.feasible[sel].mean()), 3),
                "served": round(float(served[sel].mean()), 3),
                "points": int(sel.sum()),
            }
        )
    w = np.array([t.weight for t in tasks])
    score = float(sum(p["served"] * t.weight for p, t in zip(per_task, tasks, strict=True)) / w.sum())
    pts = []
    for k in range(len(inst.task)):
        t = tasks[inst.task[k]]
        p = inst.target[k]
        if t.frame == "body":
            r, tr = hx.body_rt(BodyPose())
            p = r @ p + tr
        pts.append([round(float(v), 3) for v in p] + [int(inst.task[k]), int(kin.feasible[k]) + int(served[k])])
    return {"tasks": per_task, "score": round(score, 4), "points": pts}


def bom(design: ArmDesign, acts: dict) -> list[dict]:
    from armdesign.parts import load_stages

    stages = {s.id: s for s in load_stages()}
    costs = load_costs()
    rows = []
    for j in JOINTS:
        if j not in acts:
            continue
        a = acts[j]
        where = j.replace("_", " ")
        stage = stages.get(a.id.partition("+")[2])
        driver = costs["custom_driver_per_axis"] if a.needs_driver else 0.0
        encoder = stage.joint_encoder if stage and not a.needs_driver else 0.0
        motor = a.price - (stage.cost if stage else 0.0) - driver - encoder
        name = a.name.split(" + ")[0] if stage else a.name
        rows.append(
            {"group": "Actuators", "item": name, "where": where, "qty": 1, "usd": round(motor, 2), "source": a.source}
        )
        if driver:
            rows.append(
                {"group": "Actuators", "item": "custom driver channel", "where": where, "qty": 1, "usd": driver}
            )
        if stage:
            rows.append(
                {
                    "group": "Reductions",
                    "item": f"{stage.ratio:g}:1 {stage.kind} (drum, pinion, cable, bearings)",
                    "where": where,
                    "qty": 1,
                    "usd": stage.cost,
                }
            )
        if encoder:
            rows.append(
                {
                    "group": "Reductions",
                    "item": "joint encoder (MT6701 on the drum, read on the servo-bus board)",
                    "where": where,
                    "qty": 1,
                    "usd": encoder,
                }
            )
    fixed = fixed_cost(design, costs, any(a.needs_driver for a in acts.values()))
    for k, v in fixed.items():
        group = "Sensing" if "camera" in k else "End effector" if k.startswith(("gripper", "tool")) else "Structure"
        if k in ("harness + connectors", "servo bus adapter") or k.startswith("field kit") or "driver board" in k:
            group = "Electrical + sealing"
        rows.append({"group": group, "item": k, "where": "", "qty": 1, "usd": round(v, 2)})
    return rows


def compact_sim(src: dict, hz: int = 25) -> dict:
    """Downsample a simulation log for the viewer."""
    t = np.array(src["log"]["t"])
    step = max(1, round((len(t) - 1) / (t[-1] - t[0]) / hz)) if len(t) > 1 else 1
    keep = slice(None, None, step)
    log = {k: v[keep] for k, v in src["log"].items()}
    log["pos"] = [[[round(x, 4) for x in p] for p in frame] for frame in log["pos"]]
    log["quat"] = [[[round(x, 3) for x in q] for q in frame] for frame in log["quat"]]
    return {**{k: v for k, v in src.items() if k != "log"}, "log": log}


def main(argv: list[str]) -> None:
    if argv[:1] == ["select"]:
        select()
        return
    from armdesign.sim import load_selected

    design, aa = load_selected()
    acts = {j: getattr(aa, j) for j in JOINTS if getattr(aa, j, None) is not None}
    hx = Hexapod.load()
    mm = MassModel()
    study = json.loads((OUT / "study.json").read_text())
    data = VIEWER / "data"
    data.mkdir(parents=True, exist_ok=True)
    ws = workspace(design, acts, hx)
    backlash, speed = precision_speed(design, acts)
    cad = json.loads((ROOT / "cad" / "cad.json").read_text())
    rows_bom = bom(design, acts)
    total_cost = sum(r["usd"] for r in rows_bom)
    selected = {
        "design": design.__dict__,
        "label": design.label,
        "actuators": {
            j: {
                k: getattr(a, k)
                for k in (
                    "id",
                    "name",
                    "kind",
                    "rated",
                    "peak",
                    "mass",
                    "speed",
                    "price",
                    "backlash_deg",
                    "force_sense",
                    "ip",
                    "interface",
                    "stage",
                    "ratio",
                    "source",
                    "retail",
                    "needs_driver",
                )
            }
            | {"stage_limit": stage_limit(a)}
            for j, a in acts.items()
        },
        "cost": round(total_cost, 2),
        "mass": round(fixed_mass(design, mm) + sum(a.mass for a in acts.values()), 3),
        "backlash_mm": round(backlash, 2),
        "tip_speed": round(speed, 2),
        "payload": payload_at_reach(design, acts, hx, mm),
        "legs": leg_fit(acts, hx),
        "printed_grams": cad["printed_grams"],
        "printed_parts": cad["printed"],
        "coverage": ws["tasks"],
        "score": ws["score"],
        "hexapod": load_yaml("hexapod.yaml"),
    }
    # The study, trimmed to what the charts draw.
    keep = (
        "cost",
        "score",
        "mass",
        "backlash_mm",
        "tip_speed",
        "label",
        "actuators",
        "tier",
        "design",
        "force_sensing",
        "kinds",
    )
    sel = yaml.safe_load((ROOT / "design" / "selected.yaml").read_text())
    keep += ("lever", "stages", "sources")
    trim = lambda r: {x: r[x] for x in keep if x in r}  # noqa: E731
    slim = {
        "frontier": {
            lv: {t: [trim(r) for r in rows] for t, rows in tiers.items()} for lv, tiers in study["frontier"].items()
        },
        # Every frontier point of every geometry, field grade, for the scatter and the
        # architecture chart; the any-tier points only for the cheapest setting.
        "stage_b": [
            trim(r)
            for r in study["stage_b"]
            if r["lever"] in ("all", "bought") or (r["lever"] == "retail" and r["tier"] == "field")
        ],
        "stage_a": [{"d": r["design"], "s": round(r["kin_score"], 4)} for r in study["stage_a"]],
        "picks": sel.get("picks", {}),
        "chosen": sel.get("chosen"),
        "fa1": sel.get("fa1"),
        "part_names": part_names(sel),
    }
    with (ROOT / "data" / "benchmarks.csv").open() as f:
        bench = [row for row in csv.DictReader(f)]
    slim["benchmarks"] = [
        {
            k: row[k]
            for k in (
                "name",
                "reach_mm",
                "payload_kg",
                "mass_kg",
                "price_usd",
                "price_kind",
                "actuators",
                "backdrivable",
                "ip",
            )
        }
        for row in bench
        if row["reach_mm"] and row["payload_kg"] and row["price_usd"]
    ]
    (data / "study.json").write_text(json.dumps(slim, separators=(",", ":")))
    (data / "selected.json").write_text(json.dumps(selected, indent=1))
    (data / "workspace.json").write_text(json.dumps(ws, separators=(",", ":")))
    (data / "bom.json").write_text(json.dumps(rows_bom, indent=1))
    sims = []
    for f in sorted((OUT / "sim").glob("*.json")):
        if f.name == "index.json":
            continue
        s = compact_sim(json.loads(f.read_text()))
        (data / f"sim_{f.stem}.json").write_text(json.dumps(s, separators=(",", ":")))
        entry = {
            "id": s["name"],
            "title": s["title"],
            "summary": s["summary"],
            "tool": s["tool"],
            "note": s.get("note", ""),
        }
        doses = [e["t"] for e in s["events"] if e["label"].startswith("dose weed")]
        if len(doses) > 1:
            entry["per_weed_s"] = round((doses[-1] - doses[0]) / (len(doses) - 1), 3)
        sims.append(entry)
    order = ["strawberries", "tomato", "weed_body", "weed_arm", "rock", "soil_probe", "spray"]
    sims.sort(key=lambda s: order.index(s["id"]) if s["id"] in order else 99)
    (data / "sims.json").write_text(json.dumps(sims, indent=1))
    # One JSON of base64 GLBs: artifact hosting serves JSON but not .glb.
    meshes = {g.stem: base64.b64encode(g.read_bytes()).decode() for g in sorted((ROOT / "cad" / "glb").glob("*.glb"))}
    (data / "meshes.json").write_text(json.dumps(meshes))
    print(f"viewer data written; coverage score {ws['score']:.3f}, payload {selected['payload']}")


if __name__ == "__main__":
    main(sys.argv[1:])

# FA-1 Field Arm

The manipulator for the fleet's medium hexapod: the arm that reads the landscape (samples,
close-range sensing) and writes to it (weeding, dosing, planting, picking, clearing). Its
geometry and every actuator were chosen by optimization against 17 field tasks, then
checked in MuJoCo with the hexapod's legs doing part of the work.

**Interactive viewer:** `viewer/index.html` shows the CAD on the hexapod, replays the seven
simulations with live motor torques, lets you pose the arm, maps where it can work, and
charts the trade study. Serve the folder and open it:

```bash
cd hardware/arm/viewer && python3 -m http.server 8000   # then open http://localhost:8000
```

The design, its numbers and the reasons behind it are in [docs/DESIGN.md](docs/DESIGN.md).
What it has to do is in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## Layout

```text
hardware/arm/
├── armdesign/            the study, as code
│   ├── model.py          kinematics, joint torques, tip-over stability, collisions
│   ├── evaluate.py       task points → candidate poses → torque tables
│   ├── optimize.py       geometry sweep, actuator enumeration, cost/capability frontier
│   ├── report.py         picks the design, writes the viewer's data
│   ├── mjcf.py           MuJoCo model: hexapod + arm, parallelogram as a closed chain
│   ├── sim.py            seven task scenarios with a torque-limited controller
│   └── cad.py            parametric build123d CAD in the simulation's link frames
├── data/                 inputs: host, tasks, tools, actuator catalog, costs, benchmarks
├── design/selected.yaml  the chosen design (written by `report select`)
├── cad/                  STEP assembly and one STL per printed part
├── viewer/               the interactive page, its data and the per-link GLB meshes
├── docs/                 design write-up, requirements, research notes
└── tests/                statics checked against first principles and MuJoCo
```

## Reproduce

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
cd hardware/arm
uv sync
uv run python -m armdesign.optimize          # ~35 min on 4 cores → out/study.json
uv run python -m armdesign.report select     # → design/selected.yaml
uv run python -m armdesign.sim               # seven MuJoCo scenarios → out/sim/
uv run python -m armdesign.cad               # → cad/ (STEP, STL) and out/glb/
uv run python -m armdesign.report            # → viewer/data, viewer/assets
uv run pytest -q && uv run ruff check .
```

Change the host in `data/hexapod.yaml`, the tasks in `data/tasks.yaml` or the parts in
`data/actuators.yaml` and rerun; nothing else is hard-coded. `optimize --reuse-a` skips the
geometry sweep when only actuators or costs changed.

## Status

A design study, not a released part. The CAD is detailed enough to check fit, mass and
print time, but it has not been printed or test-fitted. Bolt patterns follow each
actuator's published flange, and the tolerances still have to be checked against the
parts in hand. Actuator prices are single-unit US retail read on 2026-10-16/17.

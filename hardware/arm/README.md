# FA-2 Field Arm

> **Two known errors** (wrist statics, joint speed) are open from the 2026-10-19 review; see
> [Known errors in docs/DESIGN.md](docs/DESIGN.md#fa-2-design).

The manipulator for the fleet's medium hexapod: the arm that reads the landscape (samples,
close-range sensing) and writes to it (weeding, dosing, planting, picking, clearing). Its
geometry and every part were chosen by optimization against 17 field tasks at delivered
prices, then checked in MuJoCo with the hexapod's legs doing part of the work.

FA-2 is the second design. It replaces FA-1 with rope capstans on the turret motors,
duty-paid sourcing and a six-axis wrist. The trade study, every lever's worth and the
three picks are in [docs/DESIGN.md](docs/DESIGN.md).

**Interactive viewer:** `viewer/index.html` shows the CAD on the hexapod, replays the seven
simulations with live motor torques, lets you pose the arm, maps where it can work, and
charts the trade study. Serve the folder and open it:

```bash
cd hardware/arm/viewer && python3 -m http.server 8000   # then open http://localhost:8000
```

The design, its numbers and the reasons behind it are in [docs/DESIGN.md](docs/DESIGN.md).
What it has to do is in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md). The custom driver
board that the study priced but did not choose is specified in
[docs/ELECTRONICS.md](docs/ELECTRONICS.md). Sourcing, reduction, electronics and sealing
research, and what its verification changed, are in [docs/research/](docs/research/).

## Layout

```text
hardware/arm/
├── armdesign/            the study, as code
│   ├── model.py          kinematics, joint torques, tip-over stability, collisions
│   ├── evaluate.py       task points → candidate poses → torque tables
│   ├── parts.py          lever settings, delivered prices, capstan and belt stages
│   ├── optimize.py       geometry sweep, actuator enumeration, cost/capability frontier
│   ├── report.py         picks the design, writes the viewer's data
│   ├── mjcf.py           MuJoCo model: hexapod + arm, parallelogram as a closed chain
│   ├── sim.py            seven task scenarios with a torque-limited controller
│   └── cad.py            parametric build123d CAD in the simulation's link frames
├── data/                 inputs: host, tasks, tools, actuators, bare motors, reductions,
│                         costs, benchmarks
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
#   --stage-a-only writes the geometry sweep alone; --from-a out/stage_a.json reuses it
uv run python -m armdesign.report select     # → design/selected.yaml
uv run python -m armdesign.sim               # seven MuJoCo scenarios → out/sim/
uv run --group cad python -m armdesign.cad   # → cad/ (STEP, STL) and cad/glb/
uv run python -m armdesign.report            # → viewer/data, viewer/assets
uv run pytest -q && uv run ruff check .
```

Change the host in `data/hexapod.yaml`, the tasks in `data/tasks.yaml`, the parts in
`data/actuators.yaml` and `data/bare_units.yaml`, the reductions in
`data/transmissions.yaml` or duty and shipping in `data/costs.yaml`, then rerun. Nothing
else is hard-coded. `optimize --reuse-a` skips the
geometry sweep when only actuators or costs changed.

## Status

A design study, not a released part. The CAD is detailed enough to check fit, mass and
print time, but it has not been printed or test-fitted. Bolt patterns follow each
actuator's published flange, and the tolerances still have to be checked against the
parts in hand. The capstan and belt stages are sized from published design rules and still
need the bench test in [docs/research/reductions.md](docs/research/reductions.md). Prices are
delivered to the US (duty and freight included), checked against live carts on 2026-10-17.

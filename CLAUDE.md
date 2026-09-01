# Working on this hardware project

This repo drives hardware design. The cost of a confident wrong answer here is
a fabricated board or a moulded part, so the bias is toward checking.

The workflow, skills and tooling come from the **MakeHardware** plugin. Run
`hw-doctor` first if anything behaves unexpectedly — it reports what the
environment build actually managed to install.

## Stages

`hw-vision` → `hw-planning` → `hw-requirements` → `hw-block-diagram` → design
→ `hw-simulation` → `hw-verification`, with `hw-review`, `hw-sourcing`,
`hw-documentation`, `hw-imagegen` and `hw-retro` throughout.

The block diagram sits between requirements and schematic capture on purpose:
the architecture and the power budget are agreed before anything is wired.

Each of the first four stages ends in a **recorded human review**, and so does
each large design stage. `review-gate check --gate` is the gate;
`docs/review/reviews.yaml` is the record.

## Three skill sets, one rule each

Three plugins provide KiCad knowledge and they overlap. The division:

| Job | Use | Why |
|---|---|---|
| **Changing** any `.kicad_*` file | **Konnect** MCP tools, always | Direct edits corrupt these files. Konnect's own rules make this mandatory, and they win. |
| **Reviewing** a design | **kicad-happy** (`kicad`, `emc`, `bom`) | Deeper read-only analysers: EMC pre-compliance, thermal, voltage derating, datasheet cross-reference. |
| **Searching for parts** | **kicad-happy** (`digikey`, `mouser`, `lcsc`, `element14`) | Real distributor stock and pricing. |
| **Deciding** which part | **`hw-sourcing`** | The house philosophy and standards. kicad-happy finds candidates; hw-sourcing picks between them. |
| **Simulating** | **`hw-simulation`** + the `spice` MCP | Wired to the requirements evidence flow. kicad-happy's `spice` skill is fine for a quick sanity check. |
| **Fab output** | Either `kicad-manufacture` or `jlcpcb`/`pcbway` | Whichever matches the house you are using. |

When two skills would both fire, the table decides. If it does not cover the
case, prefer the one that reads over the one that writes, and say which you
chose.

## Commands

```bash
hw-doctor                 # what the toolchain can actually do right now
imagegen --list           # which image providers have keys
plan-render               # refresh docs/plan.{svg,md,drawio} and the README
plan-render --summary     # status, what is ready, what awaits review
plan-render --check       # exit 1 on a `done` chunk whose outputs or review are missing
req-trace --gate          # traceability gate; exit 1 while gaps remain
req-trace --map           # the requirements map, SVG + draw.io
block-diagram             # refresh hw/block-diagram.drawio and the review image
block-diagram --check     # architecture gate; exit 1 on an over-budget rail
vision-board concepts/*.py            # renders + docs/design/vision.md
review-gate list          # where every human review stands
review-gate check --gate  # exit 1 while a milestone is unsigned or stale
```

Python for CAD and analysis is `/opt/hw-py/bin/python`. Do not `pip install`
into the system Python.

## Rules that matter

**An artefact the human has not seen is not a deliverable.** This session is
running in a cloud VM; the human is looking at this repository on github.com in
a browser. Anything you want reviewed has to be committed, pushed, and in a
format that renders there — a PDF of the schematic, a PNG of the board, an SVG
of the diagram. A `.kicad_sch`, a `.step` or a `.drawio` is a download, not a
review, and a render under `build/` is a render nobody sees.

**Get the review, do not assume it.** At the vision, the plan, the
requirements and the architecture — and at each large design stage — build the
artefact, commit it, `review-gate open`, then **ask the human directly with
`AskUserQuestion`, with the github.com link, and block on the answer**. Record
what they said with `review-gate sign`, in their words. Do not sign on their
behalf, do not read approval into silence or into a message about something
else, and do not mark a chunk `done` while its review is open or stale. See
`hw-review`.

**Keep the plan current.** Update `status` in `plan.yaml` as work completes and
re-render in the same session. A stale plan is worse than no plan, because
people trust it.

**Do not start schematic capture without an agreed block diagram.** The major
ICs, the power tree and the buses are settled in `hw/block-diagram.yaml` first,
and `block-diagram --check` has to pass. A rail found short at layout time is a
respin; found here it is an edit.

**Never mark a requirement `Verified` without evidence.** All four of:
verification method, an `EVIDENCE` pointer someone can look at, a `File`
relation to the artefact, and evidence at the corners — not just nominal. An
unmet requirement reported plainly is a normal outcome. One marked Verified is
a hidden defect.

**Report what the gate says.** `req-trace --gate` exits 1 while gaps remain. If
coverage is 60%, say 60%. Do not call a design complete while it is failing.

**Cross-check simulations against closed form** wherever one exists, and read
the `observations` field — ngspice prints `singular matrix` and then writes
plausible numbers anyway.

**Requirements need a parent.** If you cannot name the parent for a new
requirement, nobody asked for it. Raise it with the human rather than
inventing one.

**Never take a number from memory when a datasheet exists.** If the datasheet
cannot be fetched, record it as blocked in `docs/reference/manifest.yaml` and
ask the human for it.

**Before concluding an approach cannot work, list the levers you did not
vary.** A negative result from one configuration is a result about that
configuration. Corner sweeps cover tolerance, temperature and supply; they do
not cover layer count, geometry or topology, which are usually the real levers.

**Log friction as it happens.** When the human corrects you, when something
takes far more loops than it should, or when you had to guess — append three
lines to `docs/design/friction-log.md` naming the MakeHardware file that should
change. See `hw-retro`. This is the only way the toolbox gets better; a
correction that lives only in a chat transcript is one you will make again.

**Numbers and units, always.** "Low power" is not a requirement; "<= 40 uA in
standby" is. Put the reasoning in `RATIONALE` — it is what gets re-read when
the number has to move.

## Gotchas that have already cost time

* StrictDoc multi-line fields need `>>>` / `<<<` blocks, and bodies render as
  RST — a bare `SYS-*` breaks the build on an unterminated emphasis span.
  Write ``` ``SYS-`` ```.
* ngspice `.meas` at deck top level rejects LTspice's `vdb()`. Put the analysis
  and measurement in a `.control` block.
* `ccx -v` exits 201 on success. Check its output, not its exit code.
* MLCC capacitance derates hard with DC bias — a 10 uF 0603 at 5 V can be a
  third of its marked value.
* The environment snapshot preserves files, not processes.
* `hw/block-diagram.drawio` and the SVG are generated. Edit
  `block-diagram.yaml`; rearranging blocks in draw.io is fine and is kept.
* Konnect's schematic tools are file-based; its **PCB tools need a live
  KiCad**. When `check_kicad_ui` says `ipc_responsive: false`, script the board
  with KiCad's own `pcbnew` API — same object model, so still not text editing.
* A net class track width wider than the pads on its nets is unroutable, and
  nothing warns you. Route at pad width and restore the width afterwards. Run
  DRC on the placed, unrouted board. See `hw-verification`.
* `_ThermalVias` footprint variants punch vias through to the other side and
  short whatever is opposite. Check before placing.
* KiCad's Specctra DSN export writes fractional coordinates into a file it
  declares as integers, and freerouting then routes nothing while blaming the
  maze search. Round-trip and integerise the DSN first.
* `WebFetch` cannot read most datasheet PDFs. Download and extract with
  `pypdf` or `pdftotext`, both installed.

## Publishing

The repository is the primary surface: commit the review artefacts under
`docs/` so they render on github.com, and put the link in front of the human.
Publishing a vision board, a plan or a traceability report as an Artifact as
well is good — the images and the numbers together — but it is in addition to
the committed version, not instead of it. A session ends and its Artifacts go
with it; the repo is what the human still has tomorrow.

Lead with gaps, not percentages.

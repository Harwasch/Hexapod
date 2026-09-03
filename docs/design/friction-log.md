
## 2026-09-02 — vision renders at robot scale

* `scripts/vision_board.py` `iso_svg`: stroke width is fixed at 0.083 units, sized for a
  pocket-scale part. For a 1.1 m robot the viewBox is ~1500 units wide and the lines are
  invisible — the human saw "a blank frame". Stroke width should scale with the bounding box
  (e.g. 0.15 % of the viewBox width). Worked around in `scripts/render-vision.sh`.
* `scripts/vision_board.py` writes `<details><summary>` around the isometric drawing, and
  `scripts/review_artifact.py` `markdown_to_html` escapes raw HTML, so the review page shows the
  tags as text. One of the two should give; stripped in `scripts/render-vision.sh` meanwhile.
* `scripts/vision_board.py` has no way to include a scale reference (a 6 ft person) without it
  counting toward the concept's envelope, volume and mass. A `CONTEXT` module attribute rendered
  but excluded from `measure()` would do it. Worked around by reporting the robot-only envelope in
  `NOTES`.
* The vision interview was skipped because the session runs unattended; the first render then
  missed a load-bearing preference (steep femur, long tibia, coxa outboard of a hip under the body)
  that one question would have caught. `skills/hw-vision/SKILL.md` could offer a non-blocking
  variant: state the assumed answers to the load-bearing questions in the vision document itself.
* `scripts/review_artifact.py` has `images:` and `docs:` per phase but no `models:`. A human
  reviewing a mechanical concept wants to orbit it, and the Artifact sandbox allows three.js
  from cdnjs, so a built-in STL/GLB viewer per phase is feasible. Done here by post-processing
  the page in `scripts/inject_viewer.py`; the vision phase should offer it natively.
* `WebFetch` returns 403 on kollmorgen.com; the TBM2G datasheet used to anchor the motor study's
  thermal resistance only came through a distributor mirror, page by page with `pdftotext -f/-l`.
  `skills/hw-documentation/SKILL.md` could list the mirror-and-pdftotext route for vendor sites
  that block fetches.

## 2026-09-02 — round 5: parts, CAD, thermal, BOM

* `analysis/motor_options.py` `PCB_STACKS` lists "PCB 12L 3oz" as if it were a stock
  order; JLCPCB's multilayer capability stops at 2 oz inner copper, so the canonical
  stator needs a PCBWay-class heavy-copper house. `skills/hw-sourcing/SKILL.md` should say:
  check the fab's copper-weight and layer-count capability before a stack-up becomes a
  design point, and record the fab it assumes.
* The Halbach segment order in a hand-written 2-D field model was flipped on the first
  run (zero fundamental, the field concentrated on the far side). A one-line sanity check —
  "the mid-plane fundamental must exceed the far-side field" — would have caught it before
  the numbers were read. `skills/hw-simulation/SKILL.md` could list "sign of a Halbach
  array" among the closed-form cross-checks.
* `build123d` MCP `render_view` returned a blank PNG for imported STL shells and timed out
  at 60 s on `quality: high` for the STEP assembly; `import_cad_file` loads STL as a shell
  it then cannot render. The cutaway in `cad/actuator/actuator.py` is a numpy z-buffer
  raster written from scratch. The server should either render shells or refuse them.
* `hw/stator/make_stator.py` wrote `geometry.json` next to itself regardless of `--out`,
  so generating a variant silently overwrote the canonical geometry. Fixed to write beside
  the board; a generator template in `skills/hw-verification` should keep every output
  next to its board file.
* The 42 mm actuator envelope from the skeleton left no room for a second cycloid disc
  once real bearings (RB5013 13 mm, HK2512 12 mm) were placed; the analytic stack-up in
  `analysis/actuator_section.py` had assumed 8 mm discs with no bearing widths. The
  section tool should take bearing catalogue widths as inputs, not a single "reducer
  axial" number.

## 2026-09-02 — round 6: the mass loop

* "Renegotiate the mass budget" is not a number, it is a fixed-point problem: joint torque
  scales with robot mass and the robot carries eighteen of the actuators being sized. The
  sizing skill (`skills/hw-sizing` / `analysis/sizing.py` pattern) should compute the fixed
  point whenever an actuator mass changes, rather than reporting margins at a stale mass.
  `analysis/closure.py` does it here; it took the CAD masses to notice the loop diverges.
* `analysis/actuator_section.py` had no place for bearing widths or for a second disc, so
  the 42 mm envelope was carried for three rounds before the CAD showed it was 50–62 mm.
  The section tool should take catalogue bearing widths and the disc count as inputs.
* The review's four answers arrived as one `AskUserQuestion`; one of them ("two discs")
  was a change request and the other three were approvals. `review-gate sign` records one
  decision per milestone, so the record says "changes requested" and the approvals live in
  the note. A per-question decision in `reviews.yaml` would keep the record honest.

## 2026-09-02 — round 7

* The review answered "lower the yaw swing requirement": there was no place in the model
  for a reviewed cap on a derived requirement, so `YAW_SWING_CAP` was added to
  `hexapod_model.py` with the decision as its rationale. Derived requirements that a review
  overrides need a first-class home (StrictDoc `RATIONALE` once the requirements stage runs).
* Bearing ratings differ by maker for the same drawn-cup size (NTN HK2512 11.8/16.3 kN,
  PTI 10.45/14.45). `hw-sourcing` should say which maker's number a BOM line is rated on.
* The human said the review artifact is the only surface they can see. Every "look at" the
  CAD stage asks for is therefore only as good as the page: the actuator model went into a
  second orbit viewer (`scripts/inject_actuator_viewer.py`, quarter-cut and whole STL groups
  exported by the CAD script). `review-artifact` should take a `models:` list per phase
  natively; two hand-injected viewers is the workaround.

## 2026-09-02 — round 8

* The human's two questions ("that many lobes?", "have we optimised for cost?") were both
  answerable from the record but neither had been asked by the workflow: no stage asks
  "what is the cheapest manufacturing route for each part" or "is the part count sensible".
  `hw-sourcing` should carry a cost-down checklist (process per part, quantity breaks, the
  two or three lines that are physics rather than machining) and `hw-review` should ask it
  before the CAD stage is signed.
* A datasheet fetched as a PDF was image-only (`pdftotext` returned nothing); the Read tool
  rendered the page and the numbers were transcribed by eye. `hw-documentation` should say
  so and ask for the transcription to be recorded in the manifest note, as done here.

## 2026-09-02 — round 9

* The largest free torque gain in the whole project (+14 %) came from moving the stator's
  interconnect rings out from under the magnets — a layout choice made in round 5 without a
  check of "how much of the magnet span do the legs cover". The stator generator should print
  that coverage ratio with its geometry report, and `hw-verification`'s pre-route checks should
  include "active conductor under active field" for any PCB motor.
* The N-S-on-iron rotor was rejected on back-plate mass, which only showed up when the yoke
  flux was checked (≥ 4.5 mm of steel at 1.6 T). `analysis/motor_options.py` already has that
  check for its iron-core candidates; the rotor field script did not, and needed it added.

## 2026-09-02 — round 10

* "Have we found the global minimum" needed a search over the design space (motor family,
  stator count, reduction, quantity, requirement), which no stage in the workflow asks for;
  the answer overturned the round-4 motor choice (an OTS iron-core outrunner is cheaper and
  lighter than the ironless PCB machine at these torques). `hw-sourcing` / `hw-planning`
  should require a cost-and-mass search across motor families before a custom machine is
  designed, with the closure loop (torque need grows with unit mass) built in.
* OTS motor listings quote current only with propeller airflow; the heat-sunk continuous
  rating that decides the design is an assumption until measured. `hw-documentation` should
  flag "rating conditions" as a field to capture with every motor datasheet.

## 2026-09-02 — round 13

* `hw/stator/rotor_field.json` was written at the v1 coil span (r_m 66.6, a 7.0 mm Halbach
  segment) and never re-run after layout v2 moved the coils to r 84.6; the same block model at
  v2's mean radius gives a field ratio 3 % lower than every rating since round 9 has used.
  A generated JSON that another script reads as a constant needs a stamp of the inputs it was
  computed from, and `hw-verification`'s pre-release check should diff those stamps against the
  current geometry. `analysis/rotor_field.py` should take the geometry it is run against as an
  argument instead of reading the canonical file.
* The stator generator's layer roles were hard-wired for 12 layers and up, so every JLCPCB
  stack below 12 layers had been rated only as a bonded pair of a 16-layer file; the sweep of
  6/8/10-layer single boards needed `set_layers` generalised. A generator whose CLI advertises
  `--layers` should assert the range it actually supports (`hw/stator/make_stator.py`).
* The femur swing speed (3.8 rad/s at 80:1 = 2900 rpm) has been tabled as "2.9 (need 3.8)" since
  round 8 without any stage asking whether the 48 V bus can reach it; the sweep shows that a
  board fast enough at 48 V loses over half its torque to eddy loss or copper. `hw-requirements`
  should carry the joint speed as a requirement with a parent, and `hw-verification` should
  flag a tabled shortfall that survives three rounds as an open item, not a footnote.
* No SVG rasteriser exists in the environment (no cairosvg, rsvg, inkscape or ImageMagick;
  LibreOffice refuses KiCad's SVG), `kicad-cli pcb export pdf` has no board-area page mode so
  the board falls off the page, and the review page refuses a PNG over ~400 kB. The layer
  render for the page is now `scripts/render_layer.py` (pcbnew object model, matplotlib, a
  palette PNG). `hw-documentation` should name the one raster route the environment build
  guarantees, and `review-artifact` should downscale rather than refuse.

## 2026-09-03 — round 14b (leg assembly: ROM sweep, keys, BOM, note)

* The capstan concept (`analysis/capstan.py`, round 8) puts the drum on the vertical yaw axis
  and the sector on the horizontal femur pivot, and the leg CAD (`cad/leg/leg.py`) declared
  "zero fleet angle" because the runs stay in the planes x = ±r_drum. Nobody checked the other
  component: the runs leave the drums at 5–53° to the drum's plane of rotation, and the wrapped
  band plus its walk does not fit the femur drum. Six rounds of costing sat on a rope drive
  that cannot be wound. `hw-verification` should carry a cable-drive pre-CAD checklist (fleet
  angle in *both* planes at each drum, band + walk vs groove, D/d, termination) and `hw-review`
  should ask "in which plane does each rope leave each pulley" before a transmission is costed.
* `cad/leg/leg.json` was committed from an older `leg.py` than the one beside it (femur shaft
  487 g in the JSON, 337 g from the code; `hub_bot` too), so the 15.72 kg the loads note quoted
  was never reproducible. Same failure as round 13's `rotor_field.json`. Generated JSON should
  carry the generator's git hash and `hw-verification` should re-run generators and diff.
* The clearance sweep as written in `leg.py` (compound-to-compound OCCT distances over 245 poses)
  was never run because it would take over an hour, and its minimum would have been a designed
  2 mm axial gap at every pose. A moving-group sweep needs an x-extent filter for parts that
  only rotate about parallel axes, bounding-box prefiltering with the closest pair named, and
  conservative proxies for parts with many faces (a pulley with holes costs 3 s per distance).
  The `build123d` skill (`build123d://skill/modeling`) should document that recipe; the
  `compare` tool measures one fit, not a sweep.
* `analysis/leg_loads.py` (round 14) checked a 20 mm key on a 10 mm plate and used h/2 for the
  hub seat depth instead of the DIN 6885 t2; two keys came out at SF 0.9 and one at 0.23 in hub
  bearing without the script flagging that the hub was aluminium. `hw-verification` should list
  "key length ≤ hub length, bearing checked on the softer member, DIN seat depths" as a check.
* Two agents were interrupted (container restart, API rate limit) with no note of what was
  finished and what was not; the third had to infer it from `leg_loads.json` saying "sweep not
  run". `hw-planning` should ask for a `docs/design/<chunk>-state.md` checkpoint (done / missing /
  known wrong) whenever a chunk spans sessions.

## 2026-09-03 — round 14c (general arrangement)

* The actuator can had three different sizes live at once: `analysis/frameless_motor.py`'s
  half-section draws motor + 6 mm (Ø172 × 37), the same file's sweep comment sizes the motor
  against a Ø192 can carried over from the PCB unit, and `cad/actuator/frameless.json` — written
  in the same session — measures Ø172 × 49.7. The GA had to pick one before it could dimension
  anything, and picked the CAD. A study that fixes an envelope should write that envelope to its
  own JSON as a named field, not leave it to be re-derived from a plotting call; `hw-verification`
  should check that every "the unit is X × Y" in a design note resolves to one machine-readable
  number.
* `hexapod_model.Actuator` still reads `cad/actuator/femur.json` (the round-6 PCB unit,
  Ø192 × 54, 5.33 kg), so `BODY.slab_width(ACT)` and everything drawn by `analysis/drawing.py`
  is two architectures out of date while reading as current. A model that auto-loads "the CAD"
  needs the variant in its filename or a `current` pointer; `hw-documentation` should say that a
  superseded generated file is deleted or renamed, not left where a loader will find it.
* There is no model that says where an actuator is *mounted*. Link lengths, joint windows and
  can sizes all exist, but the position of each of the eighteen units is implicit in prose across
  06, 08 and 09, and the round-14b decision to delete the capstan silently orphaned twelve of
  them. `hw-block-diagram` (or a mechanical sibling) should require a placement table — unit,
  frame, origin, axis — before a mass budget is quoted, so that "the units do not fit" is caught
  at the architecture gate rather than by the first person who draws the machine.

## 2026-09-03 — round 14b (frameless actuator CAD)

* `analysis/cycloid.py::profile()` offsets the epitrochoid **outward** by the pin radius, so the
  "disc" it returns is bigger than its own ring-pin circle (radii 59.6–64.1 mm on an R 59.3 pin
  circle) and cannot mesh; the discs in `cad/actuator/actuator.py` therefore intersect the pin
  cage by ~3 mm and every disc mass in `cad/actuator/*.json` is high. The sign was found only by
  writing a meshing check — rotate the disc through one input revolution and measure the minimum
  distance from each pin centre to the profile, which must equal the pin radius exactly.
  `hw-verification` should list "a generated profile is proved by a meshing/contact check, not by
  eye" alongside the DRC and clearance checks, and any skill that ships a gear/cam profile helper
  should ship that check with it.
* `analysis/frameless_motor.py` models the unit mass as `motor + M_REDUCER(1.25) + M_HOUSING(0.55)`,
  two constants lifted from the CAD table of a *different* architecture (Ø100 bore, Ø192 can). Both
  scale hard with this design — the cycloid pin circle moves from r 43.5 to r 59.3, so the two discs
  alone weigh 822 g — and the unit came out 4.09 kg against the 2.84 kg the closure assumed, which
  moves the robot from 80 kg to 95 kg and the knee margin from 1.21 to 1.03. A study that solves a
  mass fixed point should mark constants carried over from another architecture as such and refuse
  to close (or flag) when the new geometry moves them; `hw-verification` should require the CAD mass
  to be fed back into the closure before a margin is quoted.
* "37 mm tall" in §9.17 is `motor length + 12`, a plotting expression in the same file, and it was
  quoted in prose as if it were an envelope. The reducer's own axial stack — 13 mm of crossed roller,
  4 mm of flange, 24 mm of two discs on the HK2512 cup pitch, 10.5 mm of rotor carrier and cover —
  is 51 mm of it, and the modelled unit is 49.7 mm. Same failure mode as the Ø172/Ø192 confusion
  logged under round 14c: an envelope quoted in prose that no machine-readable field backs.
  `hw-documentation` should require every dimension in a design note to name the JSON field it came
  from.
* The `build123d` MCP `render_view` was unreliable in this environment, so the renders reuse the
  hand-rolled numpy z-buffer rasteriser in `cad/actuator/actuator.py`. That rasteriser has no way to
  place a label on a real 3D point, which is what makes an exploded or cutaway view readable, so a
  projector had to be written twice. The `build123d` skill should ship a small "raster + project"
  helper (orthographic camera, z-buffer, and a `project(point) -> pixel` closure) rather than leaving
  every project to re-derive it; and `render_view`'s failure mode should be named in
  `build123d://skill/modeling` so the next session does not spend a loop discovering it.
* Fitting a render needs the *silhouette* extent, not the bounding box: every part here is a body of
  revolution, and using bbox corners over-estimates the width by up to √2 and shrinks the object to
  60 % of the frame. Worked around with an exact radius/z extent in `auto_fit()`. Worth a line in the
  `build123d` skill.


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

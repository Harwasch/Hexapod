
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

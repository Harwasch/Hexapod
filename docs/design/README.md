# Design documentation

How and why this system was built, written as the work happens.

* `adr-NNNN-<slug>.md` — decision records. One per decision that constrains
  anything downstream.
* `architecture.md` — the block-level picture and how the pieces divide.
* `interfaces.md` — board outline, connector pinouts, mounting, anything two
  disciplines have to agree on.
* `verification-report.md` — the output of the verification stage.
* `friction-log.md` — three-line entries written during the work, whenever a
  correction or an avoidable loop happens, each naming the MakeHardware file
  that should change. See the `hw-retro` skill.
* `retro.md` — the synthesis, written by `/hw-retro` at a milestone.

An ADR has: Status, Context, Decision, Consequences, Alternatives considered.
Write the Consequences honestly, the bad ones included — that section is what a
future session reads when the number has to move.

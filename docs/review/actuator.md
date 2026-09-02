# Review — Actuator, round 9: layout v2, 6 mm magnets, cheap base — $597 per unit

`actuator` · requested 2026-09-02 · branch `claude/hexapod-robot-design-mt516g`

Pushing the unit's cost further. The stator was laid out again with the coils reaching the magnets' outer edge (r 84.6) and the interconnect moved out to the clamp rim: +14 % torque for the same copper, so the two 8-layer 2 oz JLCPCB boards per position now give 2.95 N·m (the 3 oz board gave 2.90 before). That paid for 6 mm magnets instead of 8 (unit 5.33 kg and 54 mm tall instead of 5.99 kg and 62 mm; robot 124 kg) with margins 1.16 femur / 1.10 knee / 1.11 yaw on the requirement as written. A conventional N-S rotor on steel back plates was simulated and rejected (same field, +0.7 kg per unit of yoke). The base became a laser-cut floor plate, a tube slice, a turned bearing carrier and a laser-cut steel pin cage; the flange a turned hub plus a plate. Housing Ø192. BOM $597 at 20 units, $396 at 100 ($1,690 in round 7).

## What you are agreeing to

**cad-femur-1s-cutaway.png**

![cad-femur-1s-cutaway.png](../design/actuator/cad-femur-1s-cutaway.png)

**cad-femur-1s-section.png**

![cad-femur-1s-section.png](../design/actuator/cad-femur-1s-section.png)

**cad-femur-cutaway.png**

![cad-femur-cutaway.png](../design/actuator/cad-femur-cutaway.png)

**cad-femur-iso.png**

![cad-femur-iso.png](../design/actuator/cad-femur-iso.png)

**cad-femur-section.png**

![cad-femur-section.png](../design/actuator/cad-femur-section.png)

**cad-knee-cutaway.png**

![cad-knee-cutaway.png](../design/actuator/cad-knee-cutaway.png)

**cad-knee-section.png**

![cad-knee-section.png](../design/actuator/cad-knee-section.png)

**cad-yaw-1s-cutaway.png**

![cad-yaw-1s-cutaway.png](../design/actuator/cad-yaw-1s-cutaway.png)

**cad-yaw-1s-section.png**

![cad-yaw-1s-section.png](../design/actuator/cad-yaw-1s-section.png)

**cad-yaw-cutaway.png**

![cad-yaw-cutaway.png](../design/actuator/cad-yaw-cutaway.png)

**cad-yaw-section.png**

![cad-yaw-section.png](../design/actuator/cad-yaw-section.png)

**capstan.png**

![capstan.png](../design/actuator/capstan.png)

**closure.png**

![closure.png](../design/actuator/closure.png)

**cycloid-profiles.png**

![cycloid-profiles.png](../design/actuator/cycloid-profiles.png)

**operating-envelopes.png**

![operating-envelopes.png](../design/actuator/operating-envelopes.png)

**rotor-field.png**

![rotor-field.png](../design/actuator/rotor-field.png)

**section-pcb-1.png**

![section-pcb-1.png](../design/actuator/section-pcb-1.png)

**section-pcb-2.png**

![section-pcb-2.png](../design/actuator/section-pcb-2.png)

**section-wound-1.png**

![section-wound-1.png](../design/actuator/section-wound-1.png)

**section-wound-2.png**

![section-wound-2.png](../design/actuator/section-wound-2.png)

**stator-F_Cu.svg**

![stator-F_Cu.svg](../design/actuator/stator-F_Cu.svg)

**stator-In9_Cu.svg**

![stator-In9_Cu.svg](../design/actuator/stator-In9_Cu.svg)

**stator-rotor-layout.png**

![stator-rotor-layout.png](../design/actuator/stator-rotor-layout.png)

**thermal-network.png**

![thermal-network.png](../design/actuator/thermal-network.png)

**front.png**

![front.png](../design/drawing/front.png)

**hip-stack.png**

![hip-stack.png](../design/drawing/hip-stack.png)

**side.png**

![side.png](../design/drawing/side.png)

**top.png**

![top.png](../design/drawing/top.png)

| File | Opens in |
|---|---|
| [docs/design/01-sizing.md](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/docs/design/01-sizing.md) | renders on GitHub |
| [docs/design/06-geometry.md](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/docs/design/06-geometry.md) | renders on GitHub |
| [docs/design/08-actuator-design.md](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/docs/design/08-actuator-design.md) | renders on GitHub |
| [docs/design/bom-actuator.csv](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/docs/design/bom-actuator.csv) | GitHub table view |

## For context

Sources and working files. Not part of the agreement — these change as work goes on, and changing them does not invalidate your sign-off.

**stator-F_Cu.svg**

![stator-F_Cu.svg](../../hw/stator/stator-F_Cu.svg)

**stator-In9_Cu.svg**

![stator-In9_Cu.svg](../../hw/stator/stator-In9_Cu.svg)

| File | Opens in |
|---|---|
| [analysis/capstan.py](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/analysis/capstan.py) | plain text on GitHub |
| [analysis/closure.py](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/analysis/closure.py) | plain text on GitHub |
| [analysis/rotor_field.py](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/analysis/rotor_field.py) | plain text on GitHub |
| [cad/actuator/actuator.py](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/cad/actuator/actuator.py) | plain text on GitHub |
| [hw/stator/asbuilt.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/asbuilt.json) | plain text on GitHub |
| [hw/stator/capstan.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/capstan.json) | plain text on GitHub |
| [hw/stator/closure.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/closure.json) | plain text on GitHub |
| [hw/stator/drc.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/drc.json) | plain text on GitHub |
| [hw/stator/geometry.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/geometry.json) | plain text on GitHub |
| [hw/stator/gerbers/stator-B_Adhesive.gba](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-B_Adhesive.gba) | download |
| [hw/stator/gerbers/stator-B_Courtyard.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-B_Courtyard.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-B_Cu.gbl](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-B_Cu.gbl) | download |
| [hw/stator/gerbers/stator-B_Fab.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-B_Fab.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-B_Mask.gbs](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-B_Mask.gbs) | download |
| [hw/stator/gerbers/stator-B_Paste.gbp](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-B_Paste.gbp) | download |
| [hw/stator/gerbers/stator-B_Silkscreen.gbo](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-B_Silkscreen.gbo) | download |
| [hw/stator/gerbers/stator-Edge_Cuts.gm1](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-Edge_Cuts.gm1) | download |
| [hw/stator/gerbers/stator-F_Adhesive.gta](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-F_Adhesive.gta) | download |
| [hw/stator/gerbers/stator-F_Courtyard.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-F_Courtyard.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-F_Cu.gtl](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-F_Cu.gtl) | download |
| [hw/stator/gerbers/stator-F_Fab.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-F_Fab.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-F_Mask.gts](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-F_Mask.gts) | download |
| [hw/stator/gerbers/stator-F_Paste.gtp](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-F_Paste.gtp) | download |
| [hw/stator/gerbers/stator-F_Silkscreen.gto](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-F_Silkscreen.gto) | download |
| [hw/stator/gerbers/stator-In10_Cu.g10](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-In10_Cu.g10) | download |
| [hw/stator/gerbers/stator-In1_Cu.g1](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-In1_Cu.g1) | download |
| [hw/stator/gerbers/stator-In2_Cu.g2](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-In2_Cu.g2) | download |
| [hw/stator/gerbers/stator-In3_Cu.g3](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-In3_Cu.g3) | download |
| [hw/stator/gerbers/stator-In4_Cu.g4](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-In4_Cu.g4) | download |
| [hw/stator/gerbers/stator-In5_Cu.g5](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-In5_Cu.g5) | download |
| [hw/stator/gerbers/stator-In6_Cu.g6](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-In6_Cu.g6) | download |
| [hw/stator/gerbers/stator-In7_Cu.g7](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-In7_Cu.g7) | download |
| [hw/stator/gerbers/stator-In8_Cu.g8](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-In8_Cu.g8) | download |
| [hw/stator/gerbers/stator-In9_Cu.g9](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-In9_Cu.g9) | download |
| [hw/stator/gerbers/stator-Margin.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-Margin.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-User_1.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-User_1.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-User_2.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-User_2.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-User_3.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-User_3.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-User_4.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-User_4.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-User_Comments.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-User_Comments.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-User_Drawings.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-User_Drawings.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-User_Eco1.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-User_Eco1.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-User_Eco2.gbr](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-User_Eco2.gbr) | a gerber viewer |
| [hw/stator/gerbers/stator-job.gbrjob](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator-job.gbrjob) | download |
| [hw/stator/gerbers/stator.drl](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/gerbers/stator.drl) | download |
| [hw/stator/make_stator.py](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/make_stator.py) | plain text on GitHub |
| [hw/stator/rotor_field.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/rotor_field.json) | plain text on GitHub |
| [hw/stator/stator.kicad_pcb](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/stator.kicad_pcb) | KiCad |
| [hw/stator/stator.kicad_prl](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/stator.kicad_prl) | download |
| [hw/stator/stator.kicad_pro](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/stator.kicad_pro) | KiCad |
| [hw/stator/thermal.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/thermal.json) | plain text on GitHub |
| [hw/stator/variants/16L-2oz/asbuilt.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/16L-2oz/asbuilt.json) | plain text on GitHub |
| [hw/stator/variants/16L-2oz/drc-16L.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/16L-2oz/drc-16L.json) | plain text on GitHub |
| [hw/stator/variants/16L-2oz/geometry.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/16L-2oz/geometry.json) | plain text on GitHub |
| [hw/stator/variants/16L-2oz/stator-16L-2oz.kicad_pcb](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/16L-2oz/stator-16L-2oz.kicad_pcb) | KiCad |
| [hw/stator/variants/16L-2oz/stator-16L-2oz.kicad_prl](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/16L-2oz/stator-16L-2oz.kicad_prl) | download |
| [hw/stator/variants/16L-2oz/stator-16L-2oz.kicad_pro](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/16L-2oz/stator-16L-2oz.kicad_pro) | KiCad |
| [hw/stator/variants/8t/asbuilt.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/8t/asbuilt.json) | plain text on GitHub |
| [hw/stator/variants/8t/drc.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/8t/drc.json) | plain text on GitHub |
| [hw/stator/variants/8t/geometry.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/8t/geometry.json) | plain text on GitHub |
| [hw/stator/variants/8t/stator-8t.kicad_pcb](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/8t/stator-8t.kicad_pcb) | KiCad |
| [hw/stator/variants/8t/stator-8t.kicad_prl](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/8t/stator-8t.kicad_prl) | download |
| [hw/stator/variants/8t/stator-8t.kicad_pro](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/8t/stator-8t.kicad_pro) | KiCad |
| [hw/stator/variants/v1-r80/asbuilt.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/v1-r80/asbuilt.json) | plain text on GitHub |
| [hw/stator/variants/v1-r80/geometry.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/v1-r80/geometry.json) | plain text on GitHub |
| [hw/stator/variants/v1-r80/stator.kicad_pcb](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/v1-r80/stator.kicad_pcb) | KiCad |

## What we need decided

1. Layout v2 puts the phase rings and pads at the clamp rim and grows the housing from Ø186 to Ø192 for +14 % torque. Accept?
2. 6 mm magnets everywhere (5.33 kg, 54 mm, 124 kg robot, margins 1.10-1.16) or 8 mm for margin (5.87 kg, 62 mm, 134 kg, 1.15-1.22)?
3. $597 per unit at 20 units, $396 at 100. The remaining lines that are physics are the boards ($80) and magnets ($60); the rest is machining route and the driver. Sign the actuator stage here, or name a target price to push against?

## Decision

**Not yet reviewed — this is waiting on you.**

Answer in the Claude session that sent you this link. There is nothing to run and nothing to sign here.

Say which option you want **and why**: the reason is worth more than the choice, and it is what gets re-read later when a number has to move. If something is wrong, say what would be right.

<details><summary>How the answer gets recorded</summary>

The agent writes your decision, in your words, into [`docs/review/reviews.yaml`](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/docs/review/reviews.yaml):

```bash
review-gate sign actuator --approve --by <name> --note "..."
review-gate sign actuator --changes "what to change"
```

Until that happens, any chunk of work depending on this review cannot be marked done.

</details>

---

<sub>Generated by `review-gate`. The sign-off record is [`docs/review/reviews.yaml`](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/docs/review/reviews.yaml).</sub>

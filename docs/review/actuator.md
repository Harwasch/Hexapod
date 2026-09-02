# Review — Actuator, round 8: two-stage reduction and cost-down

`actuator` · requested 2026-09-02 · branch `claude/hexapod-robot-design-mt516g`

Review feedback: too many cycloid lobes and pins; radically reduce cost. The reduction is now a 20-lobe cycloid (21 pins of Ø6, 1.66 mm eccentricity, HK2512 back on a straight Ø25 shaft) plus a 4:1 Dyneema capstan stage from a Ø60 drum on the unit's output to a Ø240 sector at the femur/knee pivot (Marlow D12 Max 78 datasheet: SF 4.6 at the knee peak); the yaw stays a direct 30-lobe cycloid. Every BOM line was re-sourced: two 8-layer 2 oz JLCPCB boards per stator position instead of a 12L 3 oz board, laser-cut hardened discs, 6061 and laser-cut housing parts, a Chinese crossed roller, an ODrive-compatible driver, volume magnets — $660 per unit against $1,690 (estimates at 20 units, verified lines marked). The closure re-check with the cheaper boards at 80:1 total closes at a 132 kg robot with 1.08 / 1.03 / 1.04 margins.

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

| File | Opens in |
|---|---|
| [docs/design/01-sizing.md](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/docs/design/01-sizing.md) | renders on GitHub |
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
| [analysis/cycloid.py](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/analysis/cycloid.py) | plain text on GitHub |
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

## What we need decided

1. Two-stage reduction: a 20-lobe cycloid plus a 4:1 Dyneema capstan at the femur and knee joints (yaw stays a direct 30-lobe cycloid). Accept the capstan stage, with its re-tension schedule and the yaw-to-pitch coupling the controller compensates?
2. Cost-down boards: two 8-layer 2 oz JLCPCB boards per stator position ($80 per unit instead of $300) cost 11 % torque, taken back by the 80:1 total ratio; margins 1.08 femur / 1.03 knee. Accept the cheaper boards, or keep the 3 oz board for margin (1.22 / 1.15)?
3. The re-sourced BOM lands at about $660 per unit at 20 units ($1,690 before). The remaining big lines are physics: magnets $72 and boards $80. Is that low enough, or do we also want the custom driver board and the cast base costed for a 100-unit run?

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

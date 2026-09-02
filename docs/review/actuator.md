# Review — Actuator, round 5: the parts as designed

`actuator` · requested 2026-09-02 · branch `claude/hexapod-robot-design-mt516g`

The A3 actuator designed as parts, not models: a 36-coil 12-layer 3 oz stator board laid out with KiCad's API (DRC clean, gerbers; a 16-layer 2 oz JLCPCB fallback rated alongside), the Halbach rotor field simulated with off-the-shelf 30x8x5 N48H blocks (0.95 T at the board, 2.3 kN pull between rotors), the unit modelled in build123d (fixed pin cylinder through an open-bottom rotor cup, one 10 mm cycloid disc on an HK2512, RB5013 crossed-roller output in the mounting face: 38.7 mm tall but Ø186, 3.41 kg), the stator thermal network from that geometry (0.80 K/W copper to housing; the body's cooling, not the motor, sets the sustained torque), and a bill of materials at about $1.3k per unit at one-off prices. Femur and knee both 60:1 on one disc blank (153 N·m continuous against 135 / 143 needed); yaw 30:1.

## What you are agreeing to

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

**cad-yaw-cutaway.png**

![cad-yaw-cutaway.png](../design/actuator/cad-yaw-cutaway.png)

**cad-yaw-section.png**

![cad-yaw-section.png](../design/actuator/cad-yaw-section.png)

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

| File | Opens in |
|---|---|
| [analysis/rotor_field.py](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/analysis/rotor_field.py) | plain text on GitHub |
| [analysis/stator_asbuilt.py](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/analysis/stator_asbuilt.py) | plain text on GitHub |
| [analysis/stator_thermal.py](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/analysis/stator_thermal.py) | plain text on GitHub |
| [cad/actuator/actuator.py](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/cad/actuator/actuator.py) | plain text on GitHub |
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
| [hw/stator/stator.kicad_pcb](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/stator.kicad_pcb) | KiCad |
| [hw/stator/variants/asbuilt.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/asbuilt.json) | plain text on GitHub |
| [hw/stator/variants/drc-16L.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/drc-16L.json) | plain text on GitHub |
| [hw/stator/variants/geometry.json](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/geometry.json) | plain text on GitHub |
| [hw/stator/variants/stator-16L-2oz.kicad_pcb](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/stator-16L-2oz.kicad_pcb) | KiCad |
| [hw/stator/variants/stator-16L-2oz.kicad_prl](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/stator-16L-2oz.kicad_prl) | download |
| [hw/stator/variants/stator-16L-2oz.kicad_pro](https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/hw/stator/variants/stator-16L-2oz.kicad_pro) | KiCad |

## What we need decided

1. The unit came out Ø186 instead of Ø170 because the board's phase rings and clamp sit outside the magnets. Accept Ø186, or spend a round moving the interconnect to the bore side?
2. A femur/knee unit is 3.4 kg in the CAD against the 1.1 kg the mass budget carried (18 units: 60 kg instead of 20). Renegotiate the mass budget, or cut the joint torque requirement?
3. One cycloid disc fits the 42 mm height but puts 12.8 kN on the needle bearing at the femur peak (about 100 h L10 at the continuous rating, 2000 h at walking loads). Keep one disc at 39 mm, or two discs at 52 mm?
4. The canonical stator needs a 3 oz 12-layer board (PCBWay-class); the 2 oz 16-layer JLCPCB fallback gives 11 % less torque and needs the knee at 65:1. Which board do we order first?
5. Sustained torque with all joints active is set by the body's ability to shed ~300 W, about half the per-joint rating. Is that acceptable for the terrain demo, or do the hip pods get forced air?

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

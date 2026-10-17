# FA-1: cheap field ruggedization that keeps the QDD joints backdrivable

Checked 2026-10-17. Quantity basis: about 10 arms. Prices are US-retail or AliExpress/factory-direct as stated.
**Estimate** marks every number I derived rather than read from a source, with the reasoning next to it.
Working files (downloaded datasheets, extracted text, rendered charts) are in
`/tmp/claude-0/-home-user-Hexapod/865b29de-20ae-56fc-81f7-885c9233101c/scratchpad/research2/dl_rugged/`.
The calculation script is at
`/tmp/claude-0/-home-user-Hexapod/865b29de-20ae-56fc-81f7-885c9233101c/scratchpad/research2/scripts_rugged/calc.py`.

---

## 0. Bottom line

1. **Seal friction scales with diameter squared.**
   - Seal where the rotating diameter is smallest, and only where the torque path leaves a sealed volume.
   - Use a **V-ring plus a printed labyrinth**.
   - Never put an O-ring or a lip seal on a QDD output.
   - Numbers at 40 mm (estimates from vendor data; §2):

     | Seal at 40 mm | Running drag   | Breakout drag                      |
     | ------------- | -------------- | ---------------------------------- |
     | V-ring        | 0.03–0.05 N·m  | –                                  |
     | Lip seal      | 0.06–0.14 N·m  | –                                  |
     | Rotary O-ring | about 0.31 N·m | up to 0.92 N·m (3× after standing) |

   - An 8:1 QDD backdrives at roughly **0.4 N·m** to begin with ([Yu et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7971415)).
2. **The parallelogram already gives FA-1 the best layout for sealing.**
   - Both big motors, the yaw motor, any capstan or belt stage and the FOC board fit in **one sealed, vented turret**.
   - Only two shafts leave the turret, at the shoulder axis: the upper-arm hub and the elbow crank.
   - Seal each with a 20–25 mm V-ring: **≈0.01–0.02 N·m each (estimate)**. That is 0.2–0.5% of the shoulder and elbow continuous need, and 3–5% of a QDD's own backdrive torque.
   - The yaw interface is large (~100 mm), where a V-ring would cost ≈0.17–0.30 N·m, or 11–20% of the 1.5 N·m yaw need. Use a **non-contact drip skirt and labyrinth there instead (≈0 drag)**.
3. **Heat in sun is the binding limit, not water.**
   - A black horizontal surface rises about **50 K above ambient** in full sun (ASTM D4803 black reference). That is about 95 °C at the 45 °C design ambient.
   - **PETG and PETG-CF (HDT 69–78 °C) must not carry load in sun.**
   - **Exterior parts:** white or light-grey ASA (HDT ~100–103 °C).
   - **Loaded parts near motors:** PA12-CF (HDT 131 °C; keeps ~93% of strength when wet). PA6-CF loses 50% of strength and 71% of modulus when wet.
   - A sealed printed box cannot get rid of the tens of watts the QDDs make at continuous torque. **The motor housings must bolt to shaded aluminium.**
4. **The servos (wrist, tool roll, gripper) are not backdrivable, so seal friction does not matter there.**
   - Put each in a sealed pod: static O-ring lid, X-ring on the output, cable gland, vent.
   - The real field risk for these servos is **stripped gears from impacts**. Protect them with a **magnetic breakaway at the tool interface** (the plan already has one) and a **printed single-index ball-detent clutch** at the wrist.
5. **Cost:**
   - The **full kit is about $40 per arm with AliExpress vents, or about $60 with genuine Gore vents**, and adds about 0.45–0.5 kg (≈0.35 kg net if the aluminium heat plate replaces printed turret structure).
   - It replaces the current `seals: $10` line in `data/costs.yaml`, so the **net change is about +$30 to +$50**.
   - A **minimum IP54 kit is about $20** (net +$10) and adds about 0.3 kg.
   - For comparison, sealing bought as a part costs more: RobStride sells an IP67 version of the RS02 for **+¥300 (+43%) and +110 g (+29%)** over the plain RS02 ([RobStride README](https://github.com/RobStride/Product_Information)).

---

## 1. Targets and what field robots actually do

FA-1 targets, from `docs/REQUIREMENTS.md`:

- **IP54 joints, with the wrist and tool sealed to IP65–67.**
- **Operating range:** −10 to +45 °C, plus sun load.
- **Exposure:** dust, rain and herbicide spray.

| Platform                               | Rating / practice                                                                                                                                                                                                                                                                                                                                                                                                                        | What FA-1 can copy                                                                                                        | Source                                                                                                                                                                                                                     |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ANYbotics ANYmal                       | Main body "sealed against water and dust ingress, complying to IP-67". Each ANYdrive is "a compact, tightly integrated and sealed actuator module". The housing also has foam side-impact protection and a Kevlar belly.                                                                                                                                                                                                                 | Seal at module level. Add sacrificial impact protection.                                                                  | [ANYbotics / HazardEx 2018](https://www.anybotics.com/hazardex_december_2018.pdf); [ANYmal spec](https://www.anybotics.com/anymal-technical-specifications.pdf)                                                            |
| Boston Dynamics Spot                   | IP54; −20 to 45 °C (2020 sheet), −20 to 55 °C (current page). Arm described only as "water and dust resistant".                                                                                                                                                                                                                                                                                                                          | IP54 is enough for a commercial field robot.                                                                              | [Spot spec 2020](https://bostondynamics.com/wp-content/uploads/2020/05/spot-spec-sheet.pdf); [Spot page](https://bostondynamics.com/products/spot/)                                                                        |
| Duatic DynaArm (QDD arm on ANYmal)     | IP66 at 48 V. The elbow actuator sits at the shoulder and drives the elbow through a belt, the same remote-drive idea as FA-1's parallelogram.                                                                                                                                                                                                                                                                                           | Shows a backdrivable QDD arm can be sealed.                                                                               | [Duatic flyer 2026](<https://www.hannovermesse.de/apollo/hannover_messe_2026/obs/Binary/A1508335/Duatic%20Flyer%202026%20(1)-compressed.pdf>) (cited in `docs/research/benchmarks.md`)                                     |
| Unitree B2 + Unitree joint-seal patent | B2 is IP67. Unitree's patent US12384056 seals a quadruped joint with a **dual-ring face seal**: a soft rubber or silicone ring preloads a hard, low-friction ring (ceramic, plastic or metal) with an oil groove. It adds static rings at the gland, end cover and connector, and "a dustproof layer" at the shaft–gland–housing junction. The patent states no IP rating.                                                               | The low-drag upgrade path if V-rings fail IPX5: an X-ring energizing a PTFE or POM washer (estimate of a DIY equivalent). | [RobotShop B2](https://ca.robotshop.com/products/unitree-b2-robotic-dog); [US12384056](https://patents.google.com/patent/US12384056)                                                                                       |
| Saga Robotics Thorvald                 | IP67 for low, submersible parts and IP56 higher up.                                                                                                                                                                                                                                                                                                                                                                                      | Split the rating by exposure: IP65–67 at the tool and wrist, which touch soil and spray; IP54 at the turret.              | [Develop3D](https://develop3d.com/cad/saga-robotics-fields-of-the-future/)                                                                                                                                                 |
| FarmBot Genesis                        | Outdoor gantry with no IP rating. The universal tool mount (UTM) holds tools with **3 neodymium ring magnets**. FarmBot moved from one central magnet to three "to keep the tools from easily being torqued out".                                                                                                                                                                                                                        | A magnetic tool interface works as a breakaway (§6).                                                                      | [FarmBot UTM v1.6](https://genesis.farm.bot/v1.6/assembly/tools/utm); [v0.5 changelog](https://genesis.farm.bot/v0.5/V0.5-Documentation/universal-tool-mount)                                                              |
| NASA VIPER rover                       | Combines labyrinth, Nomex felt and spring-energized PTFE seals to keep regolith out of bearings, motors and sensors.                                                                                                                                                                                                                                                                                                                     | Cheap non-contact or felt layers first; a contact seal only as the last stage.                                            | [McBryan et al. 2025, NTRS](https://ntrs.nasa.gov/citations/20250005201)                                                                                                                                                   |
| RobStride 02 vs 02-IP67                | Same 7 N·m @ 100 rpm and 17 N·m peak. Price ¥699 → ¥999; mass 380 g → 490 g. Both manuals list the same no-load current, 0.5 A rms. The IP67 manual also repeats the 380 g mass, so the equal no-load figure may be copied text (unverified). Both rate continuous torque **on a 260 × 280 mm heat sink**, at −20 to 50 °C ambient, insulation class B. The IP67 manual adds a power-off "anti-backdrive" damping that is on by default. | Buying sealing costs +43% per joint. Sealing also traps heat.                                                             | [RobStride Product_Information](https://github.com/RobStride/Product_Information) (README; `Product Literature/CAN2.0/RS02/` manuals); [OpenELAB RS02-IP67](https://openelab.com/products/robstride02-ip67-qdd-17nm-robot) |

Two warnings on IP claims, both from [humanoid.guide on sealed joints](https://humanoid.guide/?p=18933):

- **The water digits are not cumulative.** IP67 (1 m immersion for 30 min) does not imply IPX5 (6.3 mm nozzle, 12.5 L/min) or IPX6 (12.5 mm nozzle, 100 L/min).
- **"A grabby seal eats straight into the transparency the design paid for."** Seal breakout shows up as a dead band at low speed.

The Steadywin GIM parts carry only a vendor "IP54" claim ([aifitlab GIM8108-8](https://aifitlab.com/products/steadywin-gim8108-8-planetary-reducer-servo-motor)). Treat it as unverified.

---

## 2. Rotary seal options: drag, cost, mass, evidence

All torques are at the joint output. Drag on an output-side seal adds 1:1 to joint friction.

| Seal (per joint)                                   | Drag at 25 mm                                                          | Drag at 40 mm               | Drag at 100 mm            | Unit cost                                                                                                                                                                                                                           | Mass               | Ingress it buys                                                                                                          | Evidence and reasoning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Labyrinth / drip skirt (printed, non-contact)**  | ≈0                                                                     | ≈0                          | ≈0                        | ~$0.5–1 of ASA (estimate: 20–40 g at $25/kg)                                                                                                                                                                                        | 20–60 g (estimate) | Rain, splash and coarse dust. Fine dust needs grease in the gaps.                                                        | Non-contact by definition. Vendors recommend re-greasing labyrinths with a water-insoluble grease ([SKF TSN labyrinth listing](https://www.klium.be/en/skf-tsn-616-s-labyrinth-bearing-housing-seal-2084)). NASA VIPER used labyrinths as its first stage. Viscous drag of grease in a ~0.5 mm gap is negligible at joint speeds (estimate).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **V-ring, standard NBR (VA profile)**              | **0.019 N·m**                                                          | **0.048 N·m**               | **0.30 N·m**              | $2.64 at 1–49, $1.86 at 100+ (VA40 NBR, 2 g, [123bearing](https://www.123bearing.com/seals/seal/v-ring/va40-nbr)); AliExpress VA/VD 3–200 mm assortments about $3 ([listing](https://www.aliexpress.us/item/3256807328084378.html)) | 1–3 g              | Alone, rotating against a housing face: about IP54. With a labyrinth: about IP55–56 (secondary source; see last column). | **Estimate from [Trelleborg V-Ring catalogue, Fig. 54](https://www.allsealsinc.com/rotary_seals_vrings.pdf)**, read at 2 m/s on the dry, unhardened-steel test (rendered chart in `dl_rugged/vring_chart.png`). Power loss P divided by v gives the friction force; times r gives torque. All five sizes (25–190 mm) fit **T ≈ 3.0×10⁻⁵·d² N·m (d in mm)**. Trelleborg calls the chart "indicative". Breakaway is higher than running; take ×2 at robot joint speeds (estimate). Trelleborg ([V-Ring brochure](https://www.trelleborg.com/seals/-/media/tss-media-repository/tss_website/pdf-and-other-literature/brochures/v-ring_en.pdf?rev=-1)) says power loss is "always lower than a corresponding radial oil seal". IP mapping: a blog's version of a Schaeffler table puts a plain V-ring at IP54, V-ring + labyrinth at IP55–56, double lip at IP56–65 and a bearing isolator at IP66+ ([Industrial Monitor Direct](https://industrialmonitordirect.com/blogs/knowledgebase/resolving-motor-shaft-seal-ip-ratings-and-water-ingress-solutions)). It is unverified secondary material; test it (§9). |
| **V-ring, low-friction NBR (Trelleborg N6T5C)**    | **0.011 N·m**                                                          | **0.027 N·m**               | **0.17 N·m**              | Same as above if the compound can be sourced. Otherwise use standard NBR with the friction tricks below.                                                                                                                            | 1–3 g              | Same as above                                                                                                            | Same chart, solid curves: **T ≈ 1.7×10⁻⁵·d²** (estimate). Trelleborg's friction tricks: "coat the counter-face with a low friction agent, and do not apply grease to the lip"; a wider fitted width lowers lip pressure; "Grease lubrication gives higher power losses than oil lubrication or dry running".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Radial lip seal (spring-loaded TC/HMS type)**    | 0.03 N·m (0.02–0.06)                                                   | 0.08 N·m (0.06–0.14)        | 0.5 N·m (0.35–0.9)        | ~$1 on AliExpress (TC NBR assortments, e.g. [listing](https://www.aliexpress.us/item/2251832785792971.html))                                                                                                                        | 2–10 g             | IP56–65 as a double lip (secondary source, see V-ring row)                                                               | **Measured 0.17–0.43 N·m on a 70 × 110 × 12 FKM seal at 300–3100 rpm, oil-lubricated** ([Silvestri, Prati, Tasora](https://digitaldynamicslab.unipr.it/tasora/pubblicazioni/article-aitc-tenute.pdf), Fig. 2; rendered in `dl_rugged/tas_fig2b.png`). Scaled by d² (estimate). Dry or grease-only and low-speed running is likely higher (estimate). humanoid.guide notes an SKF wave-lip variant runs lower friction and more than 10 °C cooler than a plain lip.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **O-ring as a rotary seal**                        | 0.12 N·m running, 0.36 N·m breakout                                    | 0.31 running, 0.92 breakout | 1.9 running, 5.8 breakout | <$0.1                                                                                                                                                                                                                               | <1 g               | High (IP67 is possible)                                                                                                  | **[Parker O-Ring Handbook ORD 5700](https://mfc.arizona.edu/sites/default/files/2022-02/parker_o-ring_handbook.pdf)**, §5.11: F = f_c·L. The worked example uses f_c = 0.7 lb per inch at 10% squeeze and 70 Shore; Fig. 5-9 runs to about 4 lb/in. Breakout "is 3 times the running friction" after rest. T = f_c·πd·d/2 (estimate). §5.27 warns that a stretched rotary O-ring heats and contracts (Gow-Joule effect) and seizes, and that at low speed "a stretched O-ring tends to rotate with the shaft and leak". **Do not use on QDD outputs.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **X-ring / Quad-Ring (rotary)**                    | about 0.5–0.7× the O-ring (estimate)                                   | –                           | –                         | ~$1–2 per pack ([AliExpress CS 2.62 mm NBR X-rings](https://www.aliexpress.us/item/3256806328943425.html))                                                                                                                          | <1 g               | High                                                                                                                     | Parker: the deeper X-ring groove needs "less overall squeeze and therefore less friction"; rotary use below 6.5 ft/s, 150 psi, −30 to 100 °C ([Parker X-ring page](https://www.parker.com/us/en/divisions/o-ring-and-engineered-seals-division/resources/seal-solutions-guide/dynamic-sealing/x-rings.html); returned 403 to my fetch, content from the search index). Marco/Quad-Ring claims lower friction than O-rings with no number ([Marco Rubber](https://www.marcorubber.com/quad-ring-seals.htm)). No reliable percentage found, so the ×0.5–0.7 is an estimate. **Use only on servo outputs (6–10 mm), where drag is irrelevant.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Contact-sealed bearing (2RS) at passive pivots** | 6304-2RS (20 mm bore): **1.5–26 N·mm per bearing at 50 rpm, measured** | –                           | –                         | in the existing bearings line                                                                                                                                                                                                       | 0                  | Splash and dust at the bearing itself                                                                                    | [Frictional moment of 6304-2RS bearings, PMC12654103](https://pmc.ncbi.nlm.nih.gov/articles/PMC12654103/), Table 1. Ten bearings at 70–135 N axial load. ≤0.026 N·m each, so negligible for FA-1's pivots.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Spring-energized PTFE or face seal**             | low (no number found)                                                  | –                           | –                         | $10–40 commercial (estimate; not priced)                                                                                                                                                                                            | few g              | IP67 and up                                                                                                              | What NASA VIPER and the Unitree patent use. A fallback only if IPX5 testing fails.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

**Placement rule.**

- A seal at the joint output is felt 1:1 at the joint.
- A seal on the motor side of an N:1 stage is felt N× at the joint. It also runs N× faster, though V-ring friction falls with speed.
- So inside FA-1's turret, the motors and capstan drums need **no** dynamic seal at all. Only the output shafts that leave the box do.
- Keep those shafts small: a 20–25 mm hollow axle with the big bearings inside the box. At 25 mm a standard V-ring costs about 0.019 N·m, against 0.048 N·m at 40 mm and 0.30 N·m at 100 mm.

**Encoders.** The FTD-3 encoder satellites in `custom_electronics.md` read a magnet. They can read through a 1–2 mm printed wall, so encoders need no opening (estimate; magnetic-angle ICs are usually specified for a few mm air gap, so check the chosen part's gap budget).

---

## 3. FA-1 seal plan, joint by joint

Variant B is the recommended one: levers (2) and (3) adopted, with bare motors, printed capstan or belt stages and the FTD-3 board in one turret box. Variant A is the stock GIM and EduLite modules.

| Location                                                                         | Protection                                                                                                                                                                                                                                                                                                                                                                      | Added joint friction (estimate)                                              | % of continuous need                                        | vs ~0.4 N·m QDD backdrive | Cost per arm         | Mass             |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------- | -------------------- | ---------------- |
| **Yaw** (turret on base, ~80–120 mm interface)                                   | Printed ASA **drip skirt** from the turret over a raised lip on the base: ≥10 mm overlap, ~1 mm radial gap, a second labyrinth step, drain slots at the low point. **No contact seal.** Optional Nomex or wool felt ring if dust is severe; drag unknown, measure it.                                                                                                           | ≈0                                                                           | 0% of 1.5 N·m                                               | 0%                        | ~$1                  | 40–60 g          |
| _(for comparison: V-ring at 100 mm on yaw)_                                      | –                                                                                                                                                                                                                                                                                                                                                                               | 0.17–0.30 N·m running                                                        | 11–20% of 1.5 N·m; 9–17% of the EduLite 05's 1.8 N·m rating | 43–75%                    | –                    | –                |
| **Shoulder hub and elbow crank exits** (Variant B, 20–25 mm shafts)              | V-ring (VA20–VA25) running dry against a PTFE-sprayed or polished printed or aluminium counterface. Outside it, a printed labyrinth cap with a downward drip lip. Light silicone grease in the labyrinth, not on the lip.                                                                                                                                                       | 0.007–0.019 N·m running; ≤0.04 breakaway (×2, estimate)                      | 0.1–0.3% of 6 N·m; 0.2–0.5% of 4 N·m                        | 2–5% (≤10% at breakaway)  | 2 × $2.6 + ~$1 caps  | 2 × (2 g + 15 g) |
| **Shoulder and elbow** (Variant A: GIM8108 and GIM6010 output flange, ~40–50 mm) | Non-contact printed shroud overlapping the flange gap, plus a drip skirt above. Add a V-ring on the flange hub only if IPX4 testing fails. Grommet the module's cable exit. Do not use the housing-less GWK8108 outdoors without an enclosure: its rotor and magnets are exposed, and NdFeB will collect ferrous soil particles (estimate from physics; no field source found). | 0 (shroud); 0.027–0.075 N·m (V-ring)                                         | ≤1.3% / ≤1.9%                                               | ≤19%                      | ~$2–6                | 30–50 g          |
| **Passive pivots** (elbow joint, parallelogram rod ends, wrist bearings)         | 2RS deep-groove bearings, or iglidur-type dry polymer bushings with felt seals (igus markets these for dirty agricultural machinery: [igus agri](https://www.igus.eu/info/bearings-for-agricultural-engineering), vendor claim). Printed skirts over each pivot.                                                                                                                | ≤0.026 N·m per bearing (measured 6304-2RS)                                   | <1%                                                         | –                         | in the bearings line | 0                |
| **Wrist pitch (HX-35HM), tool roll (HX-65HM), gripper (STS3215)**                | Sealed printed pod per servo: static O-ring or printed TPU gasket on the lid, X-ring or small lip seal on the 6–10 mm output, PG7 gland on the cable, one breather vent. These servos are not backdrivable, so drag only costs a little torque.                                                                                                                                 | X-ring at 10 mm: ~0.02 N·m running, 0.06 breakout (Parker formula, estimate) | ≤2% of HX-35HM 3.4 N·m stall                                | n/a                       | ~$4–6 for three pods | 75–120 g         |
| **Turret box** (Variant B) or electronics box (Variant A)                        | Static O-ring cord or printed TPU gasket on the lid; one ePTFE breather vent; FTD-3 board conformal-coated; sealed connectors (SP13 or M8) or glands at every exit.                                                                                                                                                                                                             | 0                                                                            | –                                                           | –                         | see §8               | see §8           |

Gaiters and bellows:

- **Do not bridge a QDD revolute joint with a TPU bellows.**
  - A bellows of axial stiffness k_ax and mean diameter D has an angular stiffness of K_θ = k_ax·D²/8. That follows from the axial deflection at radius r being rθ·cosφ, integrated around the ring (derivation, estimate).
  - With k_ax = 100–1000 N/m (estimate for a printed TPU95 bellows; nothing published, so measure it with a hanging weight), a D = 40–60 mm bellows gives **0.02–0.47 N·m at ±60°**.
  - That torque is a spring and can be compensated in the model. But TPU hysteresis adds an uncompensated part (estimate 10–30% of the spring torque; no data found), and the folds trap grit and water.
  - No published torque-versus-angle data for printed TPU joint boots was found. The closest is printed-TPU flexure stiffness work, which shows strong dependence on print parameters ([Tec de Monterrey record](https://research.tec.mx/vivo-tec/display/AcademicArticleSCO_105021545164)).
  - A parametric TPU bellows generator exists if you want to try one ([MakerWorld](https://makerworld.com/en/models/2836996-parametric-generator-flex-tpu-bellows-sleeves)).
- **Where a gaiter is fine:**
  - Over the wrist or gripper, where the servos are not backdrivable.
  - As a cable sleeve where wiring crosses a joint.
- This changes the current plan in `DESIGN.md`, which calls for TPU gaiters over each joint.

---

## 4. Drip skirts, vents, coating, connectors

**Drip skirts and labyrinths.** The aim is to turn the IPX4 and IPX5 (rain and hose) cases into gravity drainage.

- **Turret top:** sloped or domed, with an overhanging drip edge (estimate: 3–5 mm drip groove under the edge so water does not creep back along the underside).
- **Every rotating gap:** opens downward.
- **Drain holes:** at the low points of each cavity.
- No published drip-edge geometry for printed robots was found, so these dimensions are engineering estimates. They are cheap to test with a hose (§9).

**Breather vents (ePTFE).** A sealed box breathes when its temperature changes, and it pulls water in when it cools.

- **Why (ideal gas, estimate):**
  - Heating from 25 to 70 °C in sun raises internal pressure by **+153 mbar**.
  - A rain shower cooling it from 60 to 25 °C gives **−106 mbar**.
  - That suction pulls water past V-ring lips and lid gaskets.
  - Gore names "cold water hitting electronics on a hot sunny day" as a trigger, and says vents "equalize pressures to protect housing seals" and reduce condensation ([Gore: what would you vent](https://www.gore.com/resources/what-would-you-vent)).
  - Kollmorgen makes the same point for washed-down hot servo motors ([Kollmorgen developer note](https://www.kollmorgen.com/en-us/developer-network/ingress-protection-servo-motors); returned 403 to my fetch, content from the search index).
- **Cost:**
  - **Gore PolyVent M12×1** (PMF100318): £5.89 at qty 1, £3.03 at 1000+ ([RS UK](https://uk.rs-online.com/web/p/enclosure-ventilation/2821500); returned 403 to my fetch, prices from the search index). That is about $7.8 / $4.0 (estimate at £1 ≈ $1.33).
  - **AliExpress "IP68 nylon breather" M12 plugs:** about $1–3 per 10 ([listing](https://www.aliexpress.us/item/3256806184604657.html)). A stainless "E-PTFE" version is listed at about $1 each ([listing](https://www.aliexpress.us/item/3256802979796502.html)). The membranes on these are unverified, and the shown prices may be new-buyer promotions.
  - **Plan:** 3 vents per arm (turret, wrist pod, gripper pod). AliExpress: ~$1.5 total. Gore at qty 1: ~$23.
- **Mass:** 2–5 g each (estimate).

**Conformal coating (FTD-3 board, servo boards where reachable).**

- **MG Chemicals 422C:**
  - Theoretical coverage is 58,900 cm²/L at 25 µm; recommended film 25–75 µm ([TDS](https://www.mouser.com/datasheet/3/293/1/tds_422c_l.pdf)).
  - Price: a case of five 55 mL bottles is $119.75, about $24 a bottle ([gotopac](https://gotopac.com/mg-chemical-422c-55mlca.html)).
  - One bottle covers about 3,200 cm² at 25 µm, enough for 10 boards of 100 × 80 mm both sides at 2–3 coats (estimate). That is **about $2.4 per arm**.
- **AliExpress acrylic or silicone "PCB conformal coating":** $3–11 per 60–100 mL; chemistry unverified ([listing](https://www.aliexpress.us/item/3256810344031333.html)).
- Mask the connectors.
- Coating adds no friction and about 1–2 g.

**Connectors and glands.**

- **XT30 and JST-GH are not sealed.**
  - **SP13 IP68 aviation connectors:** listed from $0.33 per pair as a promotion ([listing](https://www.aliexpress.us/item/3256810300903217.html)); realistic $1–2 per pair (estimate).
  - **PG7 nylon IP68 glands:** about $1–2 per 10 ([listing](https://www.aliexpress.us/item/3256811827631512.html)).
- Fill unsealed pins with silicone dielectric grease, about $2 a tube ([listing](https://www.aliexpress.us/item/3256811827576058.html)).
- **Budget:** about +$3 per arm over the existing harness line, which already includes glands.

---

## 5. Protecting capstan cables and belts

- **Put every capstan drum and belt inside the sealed turret.**
  - With the parallelogram, the shoulder and elbow drives both live at the shoulder axis, so nothing tension-critical has to be outside.
  - That one decision removes UV, grit and water from the rope at no added cost.
- **Dyneema in sun:**
  - Practical Sailor reports DSM's own claim that UV cuts strength "by about 40 percent in four or five years" on Dutch exposure racks.
  - Ten years on a Dutch rack equals about 2–3 years in Miami.
  - Dyneema stays failed in 3 years where steel or polyester lasted 15 or more.
  - Chafe is worse for Dyneema than for polyester: a 20% loss of cross-section costs it 35–40% of its strength.
  - Source: [Practical Sailor](https://www.practical-sailor.com/sails-rigging-deckgear/when-should-we-retire-dyneema-stays-and-running-rigging).
  - Grit on a drum is exactly a chafe load.
  - **Also shade the Dyneema line on the gravity-compensation springs** (`spring_each` in `costs.yaml`), or switch it to a covered double braid (estimate +$1–2).
- **Creep (pretension loss):**
  - Under 75 N for one hour, steel rope stretched less than 1% and Dyneema about 15%. This is as cited, from earlier work, in [Suzuki, Kawaharazuka, Okada 2025, arXiv:2509.10862](https://arxiv.org/abs/2509.10862).
  - Heat speeds creep (general polymer behaviour; estimate), so a capstan in a 60–70 °C turret needs DM20-grade line or a spring tensioner.
  - Aaron Newman reports steel snapping on small printed drums (he cites a D/d of at least 15:1 for steel) and uses Dyneema DM20, which he says "eliminates creep" ([substack](https://aaronnewmanb.substack.com/p/capstan-drives-affordable-precision); blog claim).
  - Sandia's synthetic-rope capstans reached 95% efficiency and a predicted ~247,000 cycles on a 90 kg biped ([Mazumdar et al., RA-L 2017](https://sandia.gov/research/publications/details/synthetic-fiber-capstan-drives-for-highly-efficient-torque-controlled-robot-2017-04-01)).
- **If a rope must leave the box:**
  - Use a felt or brush wiper at the exit, and a printed tube.
  - Accept a tiny wiper drag (estimate <0.01 N·m reflected through the drum).
- **Belts:**
  - Gates Carbon Drive bicycle belts (polyurethane with carbon cords) run unlubricated in mud and snow, with mud-shedding tooth profiles. Gates admits noise "when riding in dirty conditions" with grit ([Gates FAQ](https://gatescarbondrive.com/resources/faqs); manufacturer claims).
  - Polyurethane Synchro-Power belts use steel cords and are listed for −22 to +176 °F (−30 to 80 °C) ([Motion listing](https://www.motion.com/products/sku/02948985), from the search index).
  - Enclosed HTD-5M neoprene and fiberglass belts are fine. Prefer PU with steel or aramid cords only if a belt is outside the box (estimate).
- **Cost and mass:** if everything is inside the turret, $0 and 0 g beyond the box.

---

## 6. Overload and shock protection for the servo gears

The **QDD joints need no clutch.**

- They are backdrivable, so firmware current limits are the clutch. That is the QDD argument; see `benchmarks.md`, citing [Wensing et al. 2017](https://dspace.mit.edu/handle/1721.1/119863).
- The weak points are printed capstan drums and anchors. Reinforce them with through-bolts or rods; Newman (above) suggests embedded bolts plus epoxy.
- A capstan could in principle slip as a clutch: holding ratio e^(μθ) is 1.9–6.6 for 1 wrap at μ = 0.1–0.3. But anchored ropes are standard, and FA-1 has output encoders, so slip is not recommended (estimate).

The **servos are where gear damage happens.**

- An HBRobotics user reports stripped STS3215 gears in stalls on an SO-101, and burned out three servos. A respondent advises software limits, fuses and reduced voltage during development ([thread](https://groups.google.com/g/hbrobotics/c/wMPp7it97i4), from the search index; anecdote).
- HX-35HM is 35 kg·cm (3.43 N·m) at 11.1 V. HX-65HM is 65 kg·cm (6.37 N·m) at 12.6 V. Both have stall and overheat protection, but the gears are fixed to the output ([Hiwonder HX-35HM](https://www.hiwonder.com/products/hx-35hm), [HX-65HM](https://www.hiwonder.com/products/hx-65hm)).

| Option                                                                                             | Breakaway                                                                                                                                                                                                                    | Cost                                                                                                                                                                                                                            | Mass                       | Backdrive / accuracy effect                                                                                                                                            | Evidence                                                                                                                                                                                                                                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Magnetic breakaway at the tool interface** (already planned: 4 N52 magnets in `tool_changer`)    | n·(0.5·F_pull)·r. Four 12.7 × 3.2 mm N52 discs (7.97 lb = 35.5 N pull each) at r = 15 mm give **≈1.1 N·m**; six at r = 25 mm give ≈2.7 N·m (estimate). Off-axis bending capacity is much higher (pull, not shear).           | $1.49 each at KJ ([D82-N52](https://www.kjmagnetics.com/proddetail.asp?prod=D82-N52)); ~$0.8 on AliExpress ([N52 12×3](https://www.aliexpress.us/item/3256810522646275.html)). Grade upgrade (below): +$1–2 per arm (estimate). | ~3 g per magnet (estimate) | Rigid until it lets go. Self-resets if keyed. The tool must then be re-seated.                                                                                         | Shear is about 50% of pull; measured 47–70% ([KJ shear test](https://www.kjmagnetics.com/blog/magnet-shear-force)). FarmBot went to 3 magnets to resist torque-out (§1). **N52 max operating temperature is 80 °C**, below the ~95 °C black-in-sun case (§7). Use N42SH (SH-grade rating 150 °C, general knowledge; confirm with vendor) or keep the magnets in white parts. |
| **Printed single-index ball-detent clutch** on the HX-35HM output (wrist pitch) and on the gripper | Set to ~60–70% of stall, about 2–2.4 N·m (estimate). 6–8 steel balls in printed pockets on a wave spring. One pocket deeper, so the clutch re-seats only at the original angle and the servo's magnetic encoder stays valid. | ~$2–3 each (balls, spring, ~15 g PA12-CF; estimate)                                                                                                                                                                             | 20–30 g each (estimate)    | Zero backlash below breakaway if preloaded. Above it, the wrist "backdrives", which is the protection. Watch for creep of printed pockets at temperature: use PA12-CF. | Same principle as the KIST cam-and-spring "Spring-Clutch", "rigid… below threshold… free to rotate" above it ([KIST](https://pubs.kist.re.kr/handle/201004/116059)). Commercial friction torque limiters on AliExpress start at ~$21 and are too big ([BML listing](https://www.aliexpress.us/item/3256805506346275.html)).                                                  |
| Friction slip clutch (O-ring or Belleville-preloaded disc)                                         | adjustable                                                                                                                                                                                                                   | ~$1–2                                                                                                                                                                                                                           | 15–25 g                    | Loses position after every slip, so the wrist must re-home (camera or IMU).                                                                                            | Engineering estimate                                                                                                                                                                                                                                                                                                                                                         |
| RC "servo saver" (spring-loaded horn)                                                              | compliance, not a breakaway                                                                                                                                                                                                  | $4–13 (Traxxas 1844X / 7538X; e.g. [Walmart 7538X](https://www.walmart.com/ip/46160937), from the search index)                                                                                                                 | ~10 g                      | Adds wobble under load. Poor for wrist accuracy.                                                                                                                       | RC practice                                                                                                                                                                                                                                                                                                                                                                  |
| Magnetic coupling through the pod wall (seal-less output plus breakaway)                           | ~1–2 N·m with 6–8 pairs of 12 × 3 N52 at r ≈ 22 mm across a 3–4 mm gap (estimate)                                                                                                                                            | ~$10                                                                                                                                                                                                                            | 40–60 g                    | Low torsional stiffness, so several degrees of wind-up (estimate). Acceptable for tool roll, not for wrist pitch. Prototype only.                                      | Physics estimate                                                                                                                                                                                                                                                                                                                                                             |
| Firmware                                                                                           | –                                                                                                                                                                                                                            | $0                                                                                                                                                                                                                              | 0                          | Hiwonder and Feetech torque limits and stall protection. Lower bus voltage during development.                                                                         | Hiwonder pages; HBRobotics thread                                                                                                                                                                                                                                                                                                                                            |

Keep the TPU fin-ray gripper fingers. They are the gripper's compliant overload path.

---

## 7. Thermal limits in sun

**Solar heating.**

- ASTM D4803's black reference reaches about **+50 °C above ambient** on a horizontal or 45° surface, and about **+41 °C** on a vertical one. Those are worst-case conditions: cloudless, windless, insulated backing (values via the standard's scope as indexed; the [ANSI page](https://webstore.ansi.org/standards/astm/astmd480318) returned 403 to my fetch).
- At the 45 °C design ambient, black upper surfaces can reach **~90–95 °C**.
- Solar absorptance is about 0.88 for black, 0.25 for white paint and 0.08 for bare aluminium ([Wankhede et al., THERMINIC 2007, arXiv:0801.1043](https://arxiv.org/abs/0801.1043)).
- Scaling the rise by absorptance, a **white part rises ~14 K** (estimate).
- The same paper found that a white coating cuts internal ΔT by about 25%, and that a radiation shield or double wall cuts it by about 25%. Internal circulation fans cut it 50–55%. Vents alone helped little.

**Limits that this temperature hits:**

| Item                                                            | Limit                                                                                                                            | Source                                                                                                                                                                                                                    |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PETG / PETG-CF                                                  | HDT (0.45 MPa) 76–78 °C (Polymaker PETG); 68.6 °C (PETG-rCF08); 74 °C (Bambu PETG-CF)                                            | [Polymaker data at a glance](https://wiki.polymaker.com/polymaker-products/polymaker-filaments/technical-data-at-a-glance); [Bambu forum](https://forum.bambulab.com/t/bambu-petg-cf-low-hdt-compared-to-prusament/11609) |
| ASA                                                             | HDT 103 °C (Polymaker ASA, ASA-CF08); ~100 °C (Bambu ASA, secondary chart)                                                       | Polymaker as above                                                                                                                                                                                                        |
| PA12-CF10                                                       | HDT 131 °C at 0.45 MPa, 105 °C at 1.8 MPa                                                                                        | [Polymaker TDS](https://fiberon.polymaker.com/wp-content/uploads/TDS_FIBERON-PA12-CF10_V1.1_EN.pdf)                                                                                                                       |
| NdFeB N52 magnets (tool changer, motor rotors of unknown grade) | 80 °C max operating                                                                                                              | [KJ D82-N52](https://www.kjmagnetics.com/proddetail.asp?prod=D82-N52)                                                                                                                                                     |
| RobStride RS02 / RS02-IP67 / EduLite 05                         | Ambient −20 to 50 °C; winding limit 145 °C (RS02) or 135 °C (EL05); class B insulation; rated torque on a 260 × 280 mm heat sink | RobStride manuals (repo above)                                                                                                                                                                                            |
| Steadywin GIM8108-8 / GIM6010-8                                 | −20 to +80 °C operating                                                                                                          | [aifitlab GIM8108-8](https://aifitlab.com/products/steadywin-gim8108-8-planetary-reducer-servo-motor)                                                                                                                     |
| Steadywin GWK8108-8 (housing-less)                              | −20 to 65 °C                                                                                                                     | [aifitlab GWK8108-8](https://aifitlab.com/products/steadywin-gwk8108-8-motor) (from the search index)                                                                                                                     |

**The QDDs' own heat (estimate).**

- GIM8108-8 at 48 V: nominal 7.5 N·m at 7 A, torque constant 1 N·m/A, "phase resistance" 0.67 Ω (aifitlab).
- Holding the shoulder's 6 N·m needs about 6 A. That gives **18–36 W** of copper loss (1.5·I²·R, depending on whether 0.67 Ω is per phase or line-to-line).
- The elbow GIM6010-8 at 4 N·m adds about 10–20 W (estimate by the same method).
- A sealed printed box of about 0.1 m² sheds about 0.5–1 W/K (estimate: h ≈ 5–10 W/m²K including radiation).
- So **30–50 W inside a sealed plastic turret means +30 to +100 K before any sun.** It cannot work.
- RobStride's continuous rating already assumes a large heat sink.
- **Fixes:**
  - Bolt the motor housings or stators to a 2–3 mm aluminium turret plate, exposed to air but shaded.
  - Add a **white ASA sun shield** spaced 5–10 mm off it, with chimney gaps (a radiation shield, about −25% per Wankhede).
  - Keep the gravity-compensation springs. Copper loss goes as torque², so halving holding torque cuts heat by 75%.
  - Stow in the body's shade.
  - Derate in firmware from the motor thermistors. RobStride already trips at 135 °C.
  - Sealing does trap heat around the stator, as [humanoid.guide](https://humanoid.guide/?p=18933) also notes.
- **Cost:** ~$6–8 aluminium plate (estimate) plus ~$2.5 of white ASA for the shield. Mass ~100 g plate (it can replace printed turret structure) plus ~100 g shield.

---

## 8. UV-stable print materials

Prices are from the Polymaker US store, read 2026-10-17 from its product catalogue (`us.polymaker.com/products.json`): [ASA](https://us.polymaker.com/products/asa), [PETG](https://us.polymaker.com/products/petg), [PETG-rCF08](https://us.polymaker.com/products/fiberon-petg-rcf08), [ASA-CF08](https://us.polymaker.com/products/fiberon-asa-cf08), [PA12-CF10](https://us.polymaker.com/products/fiberon-pa12-cf10), [PA6-CF20](https://us.polymaker.com/products/fiberon-pa6-cf20), [PA612-CF15](https://us.polymaker.com/products/fiberon-pa612-cf), [TPU95](https://us.polymaker.com/products/polyflex-tpu95).

"Wet" in the TDS means annealed, then soaked 48 h in 60 °C water. That is a severe condition.

| Material                     | $/kg                                                                                                                | HDT 0.45 MPa                   | Wet vs dry (X-Y)                                                                           | UV evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Use on FA-1                                                                                                                                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **ASA**                      | $24.99 (1 kg)                                                                                                       | 103 °C                         | low uptake (not tabulated)                                                                 | In QUV with UV and condensation cycles, ASA "maintained its ductility and even showed improved strength"; PETG showed "a moderate decrease in strength" ([Kohutiar et al., Manufacturing Technology 2026](https://mt.ujep.cz/artkey/mft-202601-0007_effect-of-accelerated-aging-on-the-mechanical-properties-of-fff-manufactured-polymers.php)). A desert outdoor swatch test of ASA, PETG and PLA started 2026-06-25, with no results yet ([Bambu forum](https://forum.bambulab.com/t/asa-vs-the-desert-southwest-hellscape/255737)). | **All exterior covers, skirts, labyrinths and the sun shield, in white or light grey.**                                                                                                                            |
| ASA-CF08                     | $31.7 (3 kg $94.99)                                                                                                 | 103 °C                         | –                                                                                          | as ASA                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Stiffer exterior structure, but black, so it gets hotter.                                                                                                                                                          |
| PETG                         | $18.99                                                                                                              | 76–78 °C                       | –                                                                                          | **−36% tensile strength after 24 h UV-B** at 310 nm and 50 °C, with condensation cycles; the authors estimate several months outdoors ([Amza et al., Polymers 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659210/)).                                                                                                                                                                                                                                                                                                               | Shaded interior parts only.                                                                                                                                                                                        |
| PETG-CF (Bambu) / PETG-rCF08 | ~$30                                                                                                                | 74 / 68.6 °C                   | –                                                                                          | as PETG                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | **Remove from sun-loaded structure.** `costs.yaml` currently allows "ASA / PETG-CF".                                                                                                                               |
| **PA12-CF10**                | $53.3 (3 kg $159.99)                                                                                                | 131 °C (105 at 1.8 MPa)        | **Strength 77.4 → 71.7 MPa (−7%), modulus 3311 → 3132 MPa (−5%)**; equilibrium uptake 1.5% | No UV data. Nylon is UV-sensitive, but carbon fibre and black pigment shield the surface (estimate). Paint it or put it under the white shield.                                                                                                                                                                                                                                                                                                                                                                                        | **Loaded parts near motors and in the turret:** capstan drums, clutch pockets, shoulder hub.                                                                                                                       |
| PA612-CF15                   | $49.99 (1 kg)                                                                                                       | 175 °C                         | 91.9 → 83.1 MPa (−10%), 5137 → 3991 MPa (−22%); uptake 2.2%                                | –                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Option if PA12-CF is out of stock ([TDS](https://fiberon.polymaker.com/wp-content/uploads/TDS_FIBERON-PA612-CF15_V1.1_EN.pdf)).                                                                                    |
| PA6-CF20                     | $52.99 (1 kg)                                                                                                       | 215 °C                         | **109.3 → 54.7 MPa (−50%), 8637 → 2508 MPa (−71%)**; uptake 3.3%                           | –                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | **Avoid outdoors** ([TDS](https://fiberon.polymaker.com/wp-content/uploads/TDS_FIBERON-PA6-CF20_V1.1_EN.pdf)).                                                                                                     |
| Bambu PAHT-CF                | –                                                                                                                   | 194 °C; saturated uptake 0.88% | –                                                                                          | –                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Alternative to PA12-CF (retailer copy of Bambu data: [shop3duniverse](https://shop3duniverse.com/collections/carbon-fiber-filaments/products/bambu-lab-paht-cf-high-temperature-nylon-carbon-fiber-1-75mm-black)). |
| TPU95 (gaskets, sleeves)     | $40 (0.75 kg $29.99); AliExpress 95A 1 kg ~$20–45 ([listing](https://www.aliexpress.us/item/3256812781478157.html)) | –                              | –                                                                                          | Aromatic (MDI) TPU yellows and loses properties under UV; aliphatic TPU does not ([patent US 2022/0213253](https://patents.justia.com/patent/20220213253), from the search index). "TPU is not known for its UV resistance" ([MatterHackers](https://matterhackers.com/articles/the-best-3d-printing-filament-for-outdoor-use)).                                                                                                                                                                                                       | Shaded gaskets and sleeves only. Treat as yearly consumables (estimate).                                                                                                                                           |

**Cost delta (estimate).**

- The current line is 0.75 kg at ~$30/kg = $24.
- New split: 0.45 kg ASA at $25 plus 0.30 kg PA12-CF at $53 = ~$27, plus waste. That is **about +$3–5 per arm**.
- Mass is unchanged: densities are 1.06 (PA12-CF) versus 1.07 for ASA (estimate).

---

## 9. Cost and mass roll-up, and how to verify

**Per arm, at about 10 units.** AliExpress or factory-direct where a credible listing exists.

| Item                                                       | Full kit                        | Minimum IP54 kit   | Mass                                                    |
| ---------------------------------------------------------- | ------------------------------- | ------------------ | ------------------------------------------------------- |
| 2 V-rings on the shoulder and elbow exits                  | $5.3                            | $5.3               | 4 g                                                     |
| Printed labyrinth caps and yaw drip skirt (ASA)            | $2                              | $2                 | 70–90 g                                                 |
| Static gaskets (O-ring cord / TPU) for turret lid and pods | $2                              | $1                 | 10 g                                                    |
| Breather vents ×3 (AliExpress; Gore: ~$23)                 | $1.5                            | $1.5               | 6–15 g                                                  |
| Conformal coating                                          | $2.4                            | $2.4               | 2 g                                                     |
| Sealed connectors, over the harness line                   | $3                              | $0                 | 15 g                                                    |
| Servo pods ×3 (wrist, roll, gripper)                       | $5                              | $5                 | 75–120 g                                                |
| Wrist and gripper detent clutches                          | $4                              | $0                 | 40–60 g                                                 |
| SH-grade tool-changer magnets                              | $1.5                            | $1.5               | 0                                                       |
| Sun shield (white ASA) plus aluminium heat plate           | $9                              | $2.5 (shield only) | 200 g (plate can replace printed structure)             |
| Filament change (ASA plus PA12-CF)                         | $4                              | $0                 | 0                                                       |
| Shade or cover the gravity-spring Dyneema                  | $1.5                            | $0                 | 5 g                                                     |
| **Total**                                                  | **≈$41 (≈$63 with Gore vents)** | **≈$21**           | **≈0.45–0.5 kg / ≈0.3 kg** (estimates; sum of the rows) |
| **Net, after the existing `seals: $10` line**              | **+$31 (+$53)**                 | **+$11**           | –                                                       |

**Backdrive cost of the whole kit (estimate).**

- **Yaw:** 0.
- **Shoulder and elbow:** about +0.01–0.02 N·m each running and ≤0.04 N·m at breakaway (25 mm V-rings).
- **Passive pivots:** ≤0.026 N·m per bearing.
- **Total:** below 10% of an 8:1 QDD's own ~0.4 N·m backdrive torque. Current-based force sensing should still work after re-identifying friction.

**Cheap verification before committing.** The method is field practice; the thresholds are mine.

1. **Seal drag.**
   - Motor unpowered (disable RobStride's power-off damping), output encoder reading.
   - Spring scale on a 100 mm lever: breakaway and running torque, with and without the V-ring.
   - Or sweep slowly at constant velocity and log current × Kt.
   - Pass if it adds ≤0.05 N·m per joint.
2. **Water.**
   - IPX4: oscillating sprinkler for 10 min.
   - IPX5: garden hose with a 6.3 mm nozzle at about 12.5 L/min for at least 3 min, from 2.5–3 m. The nozzle and flow are from humanoid.guide; the duration and distance are IEC 60529 as I remember it, so verify.
   - Do it with the turret **hot (sun-soaked) and then hosed**, the worst case for suction.
   - Put paper towel or a humidity card inside.
3. **Sun soak.**
   - Thermocouples on the turret top, a motor stator and the board, at midday in full sun.
   - Compare the black prototype with the white shield.
   - Pass if parts stay ≤80 °C, motor ≤100 °C, board within the FTD-3 limits.
4. **Grit.**
   - Sprinkle dry sand or soil on the yaw skirt and shoulder labyrinth during 1,000 motion cycles.
   - Re-measure drag and inspect.
5. **Breakaway.**
   - Pull-test the tool changer and wrist clutch with a luggage scale at a known lever.
   - Confirm it re-indexes to the same angle.

---

## Sources (all URLs used)

**Seals and friction**

- Trelleborg V-Ring catalogue section (Edition April 2007, hosted by All Seals Inc.), Fig. 54 power loss: https://www.allsealsinc.com/rotary_seals_vrings.pdf
- Trelleborg Forsheda V-Ring brochure: https://www.trelleborg.com/seals/-/media/tss-media-repository/tss_website/pdf-and-other-literature/brochures/v-ring_en.pdf?rev=-1
- Parker O-Ring Handbook ORD 5700 (friction §5.11, rotary §5.27): https://mfc.arizona.edu/sites/default/files/2022-02/parker_o-ring_handbook.pdf
- Parker X-rings page (403 to fetch; content from the search index): https://www.parker.com/us/en/divisions/o-ring-and-engineered-seals-division/resources/seal-solutions-guide/dynamic-sealing/x-rings.html
- Marco Rubber, Quad-Ring / X-ring: https://www.marcorubber.com/quad-ring-seals.htm
- Silvestri, Prati, Tasora, lip-seal friction torque (70 × 110 × 12 FKM): https://digitaldynamicslab.unipr.it/tasora/pubblicazioni/article-aitc-tenute.pdf
- Contact-seal geometry and frictional moment of 6304-2RS bearings (PMC12654103): https://pmc.ncbi.nlm.nih.gov/articles/PMC12654103/
- Seal-to-IP table attributed to Schaeffler (secondary blog, unverified): https://industrialmonitordirect.com/blogs/knowledgebase/resolving-motor-shaft-seal-ip-ratings-and-water-ingress-solutions
- SKF TSN labyrinth seal listing (grease re-lubrication of labyrinths): https://www.klium.be/en/skf-tsn-616-s-labyrinth-bearing-housing-seal-2084
- Sealed joints in humanoids (IP test definitions, seal friction and transparency): https://humanoid.guide/?p=18933
- Kollmorgen, ingress protection of servo motors (403 to fetch; content from the search index): https://www.kollmorgen.com/en-us/developer-network/ingress-protection-servo-motors
- 123bearing VA40 NBR V-ring: https://www.123bearing.com/seals/seal/v-ring/va40-nbr
- AliExpress VA/VD V-rings: https://www.aliexpress.us/item/3256807328084378.html
- AliExpress NBR X-rings: https://www.aliexpress.us/item/3256806328943425.html
- AliExpress TC lip seals: https://www.aliexpress.us/item/2251832785792971.html
- igus iglidur for agricultural engineering (vendor claim): https://www.igus.eu/info/bearings-for-agricultural-engineering
- Printed TPU flexure stiffness (Tec de Monterrey record): https://research.tec.mx/vivo-tec/display/AcademicArticleSCO_105021545164
- MakerWorld parametric TPU bellows generator: https://makerworld.com/en/models/2836996-parametric-generator-flex-tpu-bellows-sleeves

**Field-robot practice**

- ANYbotics / HazardEx Dec 2018 (IP-67 body, sealed ANYdrive): https://www.anybotics.com/hazardex_december_2018.pdf
- ANYmal technical specifications: https://www.anybotics.com/anymal-technical-specifications.pdf
- Boston Dynamics Spot spec sheet 2020: https://bostondynamics.com/wp-content/uploads/2020/05/spot-spec-sheet.pdf
- Boston Dynamics Spot page: https://bostondynamics.com/products/spot/
- Duatic DynaArm flyer 2026: https://www.hannovermesse.de/apollo/hannover_messe_2026/obs/Binary/A1508335/Duatic%20Flyer%202026%20(1)-compressed.pdf
- Saga Robotics Thorvald (Develop3D): https://develop3d.com/cad/saga-robotics-fields-of-the-future/
- Unitree B2 (RobotShop): https://ca.robotshop.com/products/unitree-b2-robotic-dog
- Unitree (Hangzhou Yushu) joint dynamic-seal patent US12384056: https://patents.google.com/patent/US12384056
- FarmBot UTM v1.6: https://genesis.farm.bot/v1.6/assembly/tools/utm
- FarmBot UTM v0.5 changelog: https://genesis.farm.bot/v0.5/V0.5-Documentation/universal-tool-mount
- NASA VIPER dust mitigation (McBryan, Francis, Sobey 2025): https://ntrs.nasa.gov/citations/20250005201
- RobStride Product_Information: README plus RS02, RS02-IP67 and EL05 manuals in `Product Literature/CAN2.0/` (commit 384cc56, 2026-10-16): https://github.com/RobStride/Product_Information
- OpenELAB RS02-IP67: https://openelab.com/products/robstride02-ip67-qdd-17nm-robot
- Yu et al. 2020, QDD hip exoskeleton, ~0.4 N·m backdrive: https://pmc.ncbi.nlm.nih.gov/articles/PMC7971415
- Wensing et al. 2017, MIT Cheetah actuators: https://dspace.mit.edu/handle/1721.1/119863
- Steadywin GIM8108-8 (aifitlab): https://aifitlab.com/products/steadywin-gim8108-8-planetary-reducer-servo-motor
- Steadywin GWK8108-8 (aifitlab): https://aifitlab.com/products/steadywin-gwk8108-8-motor

**Vents, coating, connectors**

- Gore, "What would you vent": https://www.gore.com/resources/what-would-you-vent
- RS UK, Gore PolyVent M12×1 (403 to fetch; prices from the search index): https://uk.rs-online.com/web/p/enclosure-ventilation/2821500
- AliExpress nylon breather vents: https://www.aliexpress.us/item/3256806184604657.html
- AliExpress stainless ePTFE vent: https://www.aliexpress.us/item/3256802979796502.html
- MG Chemicals 422C TDS: https://www.mouser.com/datasheet/3/293/1/tds_422c_l.pdf
- MG Chemicals 422C, 55 mL ×5 (gotopac): https://gotopac.com/mg-chemical-422c-55mlca.html
- AliExpress conformal coating: https://www.aliexpress.us/item/3256810344031333.html
- AliExpress SP13 IP68 connectors: https://www.aliexpress.us/item/3256810300903217.html
- AliExpress PG7 glands: https://www.aliexpress.us/item/3256811827631512.html
- AliExpress dielectric grease: https://www.aliexpress.us/item/3256811827576058.html

**Cables and belts**

- Practical Sailor, Dyneema retirement (UV, chafe, creep): https://www.practical-sailor.com/sails-rigging-deckgear/when-should-we-retire-dyneema-stays-and-running-rigging
- Suzuki, Kawaharazuka, Okada 2025, Universal Wire Testing Machine (steel vs Dyneema creep): https://arxiv.org/abs/2509.10862
- Mazumdar et al. 2017, synthetic fiber capstan drives (Sandia): https://sandia.gov/research/publications/details/synthetic-fiber-capstan-drives-for-highly-efficient-torque-controlled-robot-2017-04-01
- Aaron Newman, capstan drives (blog): https://aaronnewmanb.substack.com/p/capstan-drives-affordable-precision
- Gates Carbon Drive FAQ: https://gatescarbondrive.com/resources/faqs
- Gates Synchro-Power PU belt listing (from the search index): https://www.motion.com/products/sku/02948985

**Materials**

- Polymaker technical data at a glance: https://wiki.polymaker.com/polymaker-products/polymaker-filaments/technical-data-at-a-glance
- Polymaker Fiberon PA12-CF10 TDS v1.1: https://fiberon.polymaker.com/wp-content/uploads/TDS_FIBERON-PA12-CF10_V1.1_EN.pdf
- Polymaker Fiberon PA6-CF20 TDS v1.1: https://fiberon.polymaker.com/wp-content/uploads/TDS_FIBERON-PA6-CF20_V1.1_EN.pdf
- Polymaker Fiberon PA612-CF15 TDS v1.1: https://fiberon.polymaker.com/wp-content/uploads/TDS_FIBERON-PA612-CF15_V1.1_EN.pdf
- Polymaker store: https://us.polymaker.com/products/asa
- Polymaker store: https://us.polymaker.com/products/petg
- Polymaker store: https://us.polymaker.com/products/fiberon-petg-rcf08
- Polymaker store: https://us.polymaker.com/products/fiberon-asa-cf08
- Polymaker store: https://us.polymaker.com/products/fiberon-pa12-cf10
- Polymaker store: https://us.polymaker.com/products/fiberon-pa6-cf20
- Polymaker store: https://us.polymaker.com/products/fiberon-pa612-cf
- Polymaker store: https://us.polymaker.com/products/polyflex-tpu95
- Bambu PETG-CF HDT thread: https://forum.bambulab.com/t/bambu-petg-cf-low-hdt-compared-to-prusament/11609
- Bambu PAHT-CF data (retailer): https://shop3duniverse.com/collections/carbon-fiber-filaments/products/bambu-lab-paht-cf-high-temperature-nylon-carbon-fiber-1-75mm-black
- Amza et al. 2021, accelerated aging of PLA and PETG: https://pmc.ncbi.nlm.nih.gov/articles/PMC8659210/
- Kohutiar et al. 2026, accelerated aging of FFF ASA, PETG, PLA: https://mt.ujep.cz/artkey/mft-202601-0007_effect-of-accelerated-aging-on-the-mechanical-properties-of-fff-manufactured-polymers.php
- Bambu forum desert UV test: https://forum.bambulab.com/t/asa-vs-the-desert-southwest-hellscape/255737
- TPU yellowing patent (from the search index): https://patents.justia.com/patent/20220213253
- MatterHackers, outdoor filaments: https://matterhackers.com/articles/the-best-3d-printing-filament-for-outdoor-use
- AliExpress TPU 95A: https://www.aliexpress.us/item/3256812781478157.html

**Overload protection**

- KJ Magnetics D82-N52: https://www.kjmagnetics.com/proddetail.asp?prod=D82-N52
- KJ Magnetics shear-force test: https://www.kjmagnetics.com/blog/magnet-shear-force
- AliExpress N52 12×3 discs: https://www.aliexpress.us/item/3256810522646275.html
- KIST Spring-Clutch: https://pubs.kist.re.kr/handle/201004/116059
- AliExpress BML friction torque limiter: https://www.aliexpress.us/item/3256805506346275.html
- HBRobotics, STS3215 gears stripped (from the search index): https://groups.google.com/g/hbrobotics/c/wMPp7it97i4
- Traxxas 7538X servo saver (Walmart, from the search index): https://www.walmart.com/ip/46160937
- Hiwonder HX-35HM: https://www.hiwonder.com/products/hx-35hm
- Hiwonder HX-65HM: https://www.hiwonder.com/products/hx-65hm

**Thermal**

- ASTM D4803-18, heat buildup (403 to fetch; scope values from the search index): https://webstore.ansi.org/standards/astm/astmd480318
- Wankhede et al., Evaluation of Cooling Solutions for Outdoor Electronics (THERMINIC 2007): https://arxiv.org/abs/0801.1043

**Repo context read (not changed)**

- `docs/REQUIREMENTS.md`
- `docs/DESIGN.md`
- `docs/research/benchmarks.md`
- `docs/research/task_forces.md`
- `data/costs.yaml`
- `data/actuators.yaml`
- `data/transmissions.yaml`

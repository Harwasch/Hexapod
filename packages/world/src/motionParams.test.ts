/**
 * The motion sidecar: its format, its derivation, and the contract between the two languages.
 *
 * `tools/captures/motion_params.py` writes the committed sidecar; `deriveMotionSidecar` here is
 * its twin. A silent divergence would not throw — the runtime would just move the tree with the
 * other language's numbers — so the committed file is re-derived here and compared field by
 * field, the way `fixture.test.ts` pins the checksum.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  ALLOMETRIC_PROVENANCE,
  branchFrequencyHz,
  branchStructure,
  createLivingMotion,
  deriveMotionSidecar,
  limbModes,
  MAX_BRANCH_ORDER,
  MODE_BAND,
  MOTION_EVIDENCE_LADDER,
  parseMotionSidecar,
  parseRig,
  serializeRig,
  syntheticTreeRig,
  treeFrequencyHz,
  turbulenceLengthScaleM,
  validateMotionSidecar,
  type MotionSidecar,
} from "./index";

function fixture(name: string): string {
  return readFileSync(
    fileURLToPath(new URL(`../../../data/tiles/synthetic-tree/source/${name}`, import.meta.url)),
    "utf8",
  );
}

const rigText = fixture("rig.json");
const rig = parseRig(rigText);
const committed = parseMotionSidecar(fixture("motion.json"), rig);

describe("the committed synthetic-tree sidecar", () => {
  it("is pointed at by its rig, and belongs to it", () => {
    expect(rig.motionPath).toBe("motion.json");
    expect(committed.rigChecksum).toBe(rig.canonicalChecksum);
    expect(committed.nodeCount).toBe(rig.nodes.length);
    expect(committed.motionEvidence).toBe("allometric");
    expect(validateMotionSidecar(committed, rig)).toEqual([]);
  });

  it("is what the TypeScript derivation gives for the same height and leaf size", () => {
    const derived = deriveMotionSidecar(rig, {
      treeHeightM: committed.treeHeightM,
      leafSizeM: committed.leafSizeM,
      seed: committed.seed,
      generator: committed.generator,
    });
    const { nodes: a, ...restA } = derived;
    const { nodes: b, ...restB } = committed;
    expect(restA).toEqual(restB);
    for (const column of Object.keys(a) as (keyof MotionSidecar["nodes"])[]) {
      const left = a[column];
      const right = b[column];
      expect(left.length).toBe(right.length);
      left.forEach((value, i) => {
        // Rounded to 4–7 decimals in both languages; a last-ulp difference in `pow` could flip
        // one rounding step, and nothing larger.
        expect({ column, i, delta: Math.abs(value - (right[i] ?? Number.NaN)) <= 2e-4 }).toEqual({
          column,
          i,
          delta: true,
        });
      });
    }
    expect(derived.provenance).toEqual(ALLOMETRIC_PROVENANCE);
  });

  it("is tens of bytes per node", () => {
    const bytes = fixture("motion.json").length;
    console.info(`MEASURED sidecar size: ${bytes} bytes for ${rig.nodes.length} nodes`);
    expect(bytes / rig.nodes.length).toBeLessThan(80);
  });

  it("round-trips the rig's pointer through serializeRig", () => {
    expect(parseRig(serializeRig(rig))).toEqual(rig);
    expect((JSON.parse(serializeRig(rig)) as { motion?: string }).motion).toBe("motion.json");
  });
});

describe("the allometric rules", () => {
  it("uses Coder's law for branches and the pendulum law for the tree", () => {
    expect(branchFrequencyHz(1)).toBeCloseTo(2.55, 10);
    expect(branchFrequencyHz(10)).toBeCloseTo(2.55 * 10 ** -0.59, 10);
    expect(treeFrequencyHz(16)).toBeCloseTo(0.6, 10);
    const sidecar = deriveMotionSidecar(syntheticTreeRig());
    sidecar.nodes.mode.forEach((mode, i) => {
      if (i === 0) return;
      if (mode === 0)
        expect(sidecar.nodes.frequencyHz[i]).toBeCloseTo(treeFrequencyHz(sidecar.treeHeightM), 3);
    });
  });

  it("splits the synthetic tree into a trunk and one limb per axis, twigs riding their limb", () => {
    const structure = branchStructure(syntheticTreeRig());
    const limbs = new Set(structure.limb.slice(1));
    expect(structure.treeBranch).toBe(1);
    // 12 primaries, 36 secondaries, 108 twigs; the middle child of each fork (equal tips, the
    // straightest) continues its parent, so 12 + 24 + 72 limbs start, plus the trunk.
    expect(limbs.size).toBe(108);
    const f0 = treeFrequencyHz(6);
    const { branch, twig } = limbModes(syntheticTreeRig(), structure, f0);
    const modes = new Set(branch.slice(1));
    // Limbs too short to ring inside the tree's band, or past the third order, fold into the
    // mode of the limb they hang from.
    expect(modes.size).toBeLessThan(limbs.size);
    for (const base of modes) {
      if (base === structure.treeBranch) continue;
      expect(structure.order[base]).toBeLessThanOrEqual(MAX_BRANCH_ORDER);
      expect(branchFrequencyHz(structure.branchLengthM.get(base) ?? 0)).toBeLessThanOrEqual(
        MODE_BAND * f0,
      );
    }
    twig.forEach((isTwig, i) => {
      if (isTwig) expect(branch[i]).not.toBe(structure.limb[i]);
    });
  });

  it("spreads each branch's bend over its joints, summing to one", () => {
    const sidecar = deriveMotionSidecar(syntheticTreeRig());
    const sums = new Map<number, number>();
    sidecar.nodes.branch.forEach((base, i) => {
      if (i > 0) sums.set(base, (sums.get(base) ?? 0) + (sidecar.nodes.share[i] ?? 0));
    });
    for (const sum of sums.values()) expect(sum).toBeCloseTo(1, 4);
  });

  it("damps the tree at 8.6 %, limbs in the measured branch range, bigger limbs never less", () => {
    const tree = syntheticTreeRig();
    const sidecar = deriveMotionSidecar(tree);
    const { tips } = branchStructure(tree);
    sidecar.nodes.damping.forEach((zeta, i) => {
      if (sidecar.nodes.mode[i] === 0) {
        expect(zeta).toBe(0.086);
        return;
      }
      // James & Haritos 2010: single branches 3.5-4.5 %, the tree with its branches 10.6 %.
      expect(zeta).toBeGreaterThanOrEqual(0.045);
      expect(zeta).toBeLessThanOrEqual(0.106);
    });
    // A limb carrying more is damped no less (its sub-branches are tuned mass dampers on it).
    const limbs = [...new Set(sidecar.nodes.branch.slice(1))].filter(
      (b) => sidecar.nodes.mode[b] === 1,
    );
    for (const a of limbs)
      for (const b of limbs)
        if ((tips[a] ?? 0) > (tips[b] ?? 0))
          expect(sidecar.nodes.damping[a] ?? 0).toBeGreaterThanOrEqual(
            sidecar.nodes.damping[b] ?? 0,
          );
    // Quantised, so branches share motion textures.
    expect(new Set(sidecar.nodes.damping).size).toBeLessThanOrEqual(4);
  });

  it("derives the wind from the tree's height (EN 1991-1-4, terrain category III)", () => {
    const sidecar = deriveMotionSidecar(syntheticTreeRig(), { treeHeightM: 6 });
    // I = 1/ln(6/0.3) = 0.334; L = 300 (6/200)^(0.67 + 0.05 ln 0.3) = 35.4 m.
    expect(sidecar.wind.turbulence.along).toBeCloseTo(2 / Math.log(20), 4);
    expect(sidecar.wind.turbulence.across).toBeCloseTo(0.75 / Math.log(20), 4);
    expect(sidecar.wind.lengthScaleM).toBeCloseTo(35.36, 2);
    expect(sidecar.wind.gust.strength).toBe(0);
    // Below z_min = 5 m the intensity and scale are held at 5 m.
    const low = deriveMotionSidecar(syntheticTreeRig(), { treeHeightM: 3 });
    expect(low.wind.turbulence.along).toBeCloseTo(2 / Math.log(5 / 0.3), 4);
  });

  it("does not read a single radius", () => {
    const base = syntheticTreeRig();
    const fat = {
      ...base,
      nodes: base.nodes.map((node) => ({ ...node, radius: node.radius * 7 })),
    };
    expect(deriveMotionSidecar(fat).nodes).toEqual(deriveMotionSidecar(base).nodes);
  });
});

describe("validation", () => {
  it("refuses a sidecar for another capture", () => {
    const wrong = { ...committed, rigChecksum: "fnv1a32:1:00000000" };
    expect(() => createLivingMotion(rig, wrong)).toThrow(/rigChecksum/);
  });

  it("refuses a sidecar with the wrong number of nodes, or off the ladder", () => {
    const short = {
      ...committed,
      nodes: { ...committed.nodes, share: committed.nodes.share.slice(1) },
    };
    expect(validateMotionSidecar(short).join()).toMatch(/share/);
    const offLadder = { ...committed, motionEvidence: "vibes" } as unknown as MotionSidecar;
    expect(validateMotionSidecar(offLadder).join()).toMatch(/ladder/);
    expect(() => parseMotionSidecar(JSON.stringify({ ...committed, version: 2 }))).toThrow(
      /version/,
    );
  });

  it("loads a sidecar written before the length scale was recorded, and refuses a bad one", () => {
    // The format before this field: wind without `lengthScaleM`, old sway ratios and gusts.
    const { lengthScaleM: _dropped, ...oldWind } = committed.wind;
    const old = {
      ...committed,
      wind: {
        ...oldWind,
        gust: { strength: 0.35, variance: 0.3, frequencyPerMin: 3, durationS: 5 },
        turbulence: { along: 0.8, across: 0.6 },
      },
    };
    const motion = createLivingMotion(rig, parseMotionSidecar(JSON.stringify(old), rig));
    expect(motion.lengthScaleM).toBeCloseTo(turbulenceLengthScaleM(committed.treeHeightM), 10);
    const bad = { ...committed, wind: { ...committed.wind, lengthScaleM: -1 } };
    expect(validateMotionSidecar(bad).join()).toMatch(/lengthScaleM/);
  });

  it("orders the evidence ladder strongest first", () => {
    expect(MOTION_EVIDENCE_LADDER).toEqual([
      "recorded",
      "fitted-real",
      "fitted-generated",
      "allometric",
    ]);
  });

  it("refuses a rig pointer that escapes the rig's folder", () => {
    const text = JSON.stringify({ ...JSON.parse(rigText), motion: "../elsewhere/motion.json" });
    expect(() => parseRig(text)).toThrow(/relative path/);
  });
});

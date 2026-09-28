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
  MODE_MAX_HZ,
  MOTION_EVIDENCE_LADDER,
  parseMotionSidecar,
  parseRig,
  serializeRig,
  syntheticTreeRig,
  treeFrequencyHz,
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
    const modes = new Set(structure.branch.slice(1));
    // The shortest twigs (chord < 0.47 m) ring above 4 Hz and fold into their limb's mode.
    expect(modes.size).toBeLessThan(limbs.size);
    for (const base of modes) {
      if (base === structure.treeBranch) continue;
      expect(branchFrequencyHz(structure.branchLengthM.get(base) ?? 0)).toBeLessThanOrEqual(
        MODE_MAX_HZ,
      );
    }
    structure.twig.forEach((twig, i) => {
      if (twig) expect(structure.branch[i]).not.toBe(structure.limb[i]);
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

  it("damps the tree at 8.6 % and big leafy limbs more, never less", () => {
    const sidecar = deriveMotionSidecar(syntheticTreeRig());
    sidecar.nodes.damping.forEach((zeta, i) => {
      expect(zeta).toBeGreaterThanOrEqual(0.086);
      expect(zeta).toBeLessThanOrEqual(0.15);
      if (sidecar.nodes.mode[i] === 0) expect(zeta).toBe(0.086);
    });
    expect(new Set(sidecar.nodes.damping).size).toBeLessThanOrEqual(4);
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

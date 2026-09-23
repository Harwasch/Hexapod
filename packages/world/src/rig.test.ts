import { describe, expect, it } from "vitest";

import {
  ancestorsOf,
  assertValidRig,
  checksumPositions,
  DEFAULT_BAND_ANGLE_LIMIT_RAD,
  heightAboveRoot,
  nodeAngleLimit,
  parseRig,
  positionsMatchChecksum,
  serializeRig,
  syntheticTreeRig,
  validateRig,
  type MotionRig,
  type SkeletonNode,
} from "./index";

const rig = syntheticTreeRig();

describe("syntheticTreeRig", () => {
  it("is a skeleton of a couple of hundred nodes, not an index list", () => {
    expect(rig.nodes.length).toBe(214);
    // Still a skeleton and not one node per splat: 214 nodes against 12,000 splats. The count
    // went up from 33 because nine leaf clusters cannot rustle — there are 108 now.
    expect(rig.nodes.length).toBeGreaterThanOrEqual(150);
    expect(rig.nodes.length).toBeLessThanOrEqual(300);
    expect(rig.nodes.filter((n) => n.band === "leaf").length).toBe(108);
    expect(validateRig(rig)).toEqual([]);
  });

  it("gives every limb a physically plausible slenderness", () => {
    // Length ÷ diameter, the number the first fixture got wrong by 5–15×: it gave a 1.15 m
    // branch a 12 cm radius, so `radius / length²` correctly called it stiff, put it outside
    // the forcing band, and left the crown riding rigid while the trunk did all the work.
    const slenderness = (index: number): number => {
      const node = rig.nodes[index];
      const parent = node === undefined ? undefined : rig.nodes[node.parent];
      if (node === undefined || parent === undefined) return Number.NaN;
      const dz = node.position[2] - parent.position[2];
      const dx = node.position[0] - parent.position[0];
      const dy = node.position[1] - parent.position[1];
      return Math.hypot(dx, dy, dz) / (2 * node.radius);
    };
    const worst = new Map<string, number>();
    rig.nodes.forEach((node, i) => {
      if (node.parent < 0) return;
      worst.set(node.band, Math.min(worst.get(node.band) ?? Infinity, slenderness(i)));
    });
    // Per segment rather than per limb, so these are lower than the whole-limb figures quoted
    // in the docs (trunk 15, branch 24, twig 41). The point is the floor, not the exact value:
    // the fixture this replaced scored 2.0, 4.8 and 8.1 on the same measure.
    expect(worst.get("trunk") ?? 0).toBeGreaterThan(1.5);
    expect(worst.get("branch") ?? 0).toBeGreaterThan(5);
    expect(worst.get("leaf") ?? 0).toBeGreaterThan(15);
  });

  it("obeys da Vinci's rule: a node's cross-section is the sum of its children's", () => {
    // Exactly, up to the per-segment taper. This is what ties the twig radius to the trunk
    // radius, and it is why the two cannot both be realistic at 214 nodes — a real tree carries
    // thousands of twigs, not a hundred. See TWIG_RADIUS_M.
    const childArea = new Array<number>(rig.nodes.length).fill(0);
    rig.nodes.forEach((node) => {
      if (node.parent >= 0)
        childArea[node.parent] = (childArea[node.parent] ?? 0) + node.radius ** 2;
    });
    let checked = 0;
    rig.nodes.forEach((node, i) => {
      const area = childArea[i] ?? 0;
      if (area === 0) return;
      const ratio = node.radius / Math.sqrt(area);
      // The taper constants are 1.03 (trunk) and 1.06 (limb).
      expect(ratio).toBeGreaterThan(1.02);
      expect(ratio).toBeLessThan(1.07);
      checked += 1;
    });
    expect(checked).toBeGreaterThan(100);
    // And the trunk really is thicker than a twig by the square root of the tips it carries.
    const tips = rig.nodes.filter((n) => n.band === "leaf").length;
    expect((rig.nodes[0]?.radius ?? 0) / (rig.nodes.at(-1)?.radius ?? 1)).toBeGreaterThan(
      Math.sqrt(tips) * 0.8,
    );
  });

  it("is deterministic: the same options give an identical rig", () => {
    expect(syntheticTreeRig()).toEqual(syntheticTreeRig());
    expect(serializeRig(syntheticTreeRig({ heightM: 9 }))).toBe(
      serializeRig(syntheticTreeRig({ heightM: 9 })),
    );
  });

  it("roots the trunk at the origin with the tree along +Z", () => {
    expect(rig.nodes[0]?.position).toEqual([0, 0, 0]);
    expect(rig.nodes[0]?.parent).toBe(-1);
    const top = Math.max(...rig.nodes.map((n) => n.position[2]));
    expect(top).toBeGreaterThan(6);
    expect(Math.min(...rig.nodes.map((n) => n.position[2]))).toBe(0);
  });

  it("is in topological order, so parents are resolved before their children", () => {
    rig.nodes.forEach((node, i) => {
      if (i === 0) expect(node.parent).toBe(-1);
      else expect(node.parent).toBeLessThan(i);
    });
  });

  it("covers all three bands", () => {
    const bands = new Set(rig.nodes.map((n) => n.band));
    expect([...bands].sort()).toEqual(["branch", "leaf", "trunk"]);
  });

  it("scales with its options", () => {
    const tall = syntheticTreeRig({
      heightM: 12,
      trunkSegments: 8,
      whorls: 2,
      branchesPerWhorl: 4,
      secondariesPerBranch: 2,
      twigsPerSecondary: 2,
    });
    // trunk + primaries(2 nodes) + secondaries(2 nodes each) + twigs, per primary limb.
    expect(tall.nodes.length).toBe(8 + 2 * 4 * (2 + 2 * (2 + 2)));
    expect(Math.max(...tall.nodes.map((n) => n.position[2]))).toBeGreaterThan(12);
  });
});

describe("rig serialisation", () => {
  it("round-trips losslessly through JSON", () => {
    expect(parseRig(serializeRig(rig))).toEqual(rig);
    expect(JSON.parse(JSON.stringify(rig)) as MotionRig).toEqual(rig);
  });

  it("round-trips a per-node angular limit override", () => {
    const withOverride: MotionRig = {
      ...rig,
      nodes: rig.nodes.map((node, i) => (i === 4 ? { ...node, maxAngleRad: 0.02 } : node)),
    };
    const back = parseRig(serializeRig(withOverride));
    expect(back).toEqual(withOverride);
    expect(back.nodes[4]?.maxAngleRad).toBe(0.02);
    expect(back.nodes[5]).not.toHaveProperty("maxAngleRad");
  });

  it("produces byte-stable text for an unchanged rig", () => {
    expect(serializeRig(parseRig(serializeRig(rig)))).toBe(serializeRig(rig));
  });

  it("refuses malformed documents rather than returning a half-trusted rig", () => {
    expect(() => parseRig("[]")).toThrow(/top level/);
    expect(() => parseRig(JSON.stringify({ units: "meters" }))).toThrow(/nodes/);
    expect(() => parseRig(serializeRig(rig).replace('"meters"', '"feet"'))).toThrow(/units/);
    expect(() => parseRig(serializeRig(rig).replace('"trunk"', '"root"'))).toThrow(/band/);
  });
});

describe("validateRig", () => {
  const node = (over: Partial<SkeletonNode> = {}): SkeletonNode => ({
    id: "n",
    parent: -1,
    position: [0, 0, 0],
    radius: 0.1,
    stiffness: 1,
    band: "trunk",
    ...over,
  });
  const of = (nodes: SkeletonNode[]): MotionRig => ({
    nodes,
    canonicalChecksum: "fnv1a32:0:00000000",
    units: "meters",
    sourceNote: "test",
  });

  it("reports every problem, not just the first", () => {
    const issues = validateRig(of([node({ id: "a", stiffness: 0 }), node({ id: "a", parent: 5 })]));
    expect(issues.length).toBeGreaterThanOrEqual(3);
    expect(issues.join("\n")).toMatch(/stiffness/);
    expect(issues.join("\n")).toMatch(/duplicate id/);
    expect(issues.join("\n")).toMatch(/topological order/);
  });

  it("rejects an empty rig and a non-root first node", () => {
    expect(validateRig(of([]))).toEqual(["rig has no nodes"]);
    expect(validateRig(of([node({ parent: 0 })])).join()).toMatch(/must be the root/);
  });

  it("rejects a forward reference, which would break parent composition", () => {
    const issues = validateRig(of([node({ id: "root" }), node({ id: "child", parent: 1 })]));
    expect(issues.join()).toMatch(/topological order/);
  });

  it("assertValidRig throws with the problems listed", () => {
    expect(() => assertValidRig(of([node({ parent: 0 })]))).toThrow(/invalid motion rig/);
    expect(assertValidRig(rig)).toBe(rig);
  });
});

describe("angular limits", () => {
  it("falls back to the band default and honours an override", () => {
    const trunk = rig.nodes[2];
    expect(trunk).toBeDefined();
    if (!trunk) return;
    expect(nodeAngleLimit(trunk)).toBe(DEFAULT_BAND_ANGLE_LIMIT_RAD.trunk);
    expect(nodeAngleLimit({ ...trunk, maxAngleRad: 0.05 })).toBe(0.05);
    expect(nodeAngleLimit({ ...trunk, maxAngleRad: 0 })).toBe(DEFAULT_BAND_ANGLE_LIMIT_RAD.trunk);
    expect(nodeAngleLimit({ ...trunk, maxAngleRad: Number.NaN })).toBe(
      DEFAULT_BAND_ANGLE_LIMIT_RAD.trunk,
    );
  });

  it("gives leaves more freedom than the trunk", () => {
    expect(DEFAULT_BAND_ANGLE_LIMIT_RAD.leaf).toBeGreaterThan(DEFAULT_BAND_ANGLE_LIMIT_RAD.branch);
    expect(DEFAULT_BAND_ANGLE_LIMIT_RAD.branch).toBeGreaterThan(DEFAULT_BAND_ANGLE_LIMIT_RAD.trunk);
  });
});

describe("tree walking", () => {
  it("walks ancestors from a node up to the root", () => {
    expect(ancestorsOf(rig, 0)).toEqual([0]);
    const leaf = rig.nodes.length - 1;
    const chain = ancestorsOf(rig, leaf);
    expect(chain[0]).toBe(leaf);
    expect(chain.at(-1)).toBe(0);
    for (let i = 1; i < chain.length; i += 1) {
      expect(chain[i]).toBeLessThan(chain[i - 1] ?? -1);
    }
  });

  it("measures height above the trunk base", () => {
    expect(heightAboveRoot(rig, 0)).toBe(0);
    // Node 9 is the top of the bole, at 70 % of the tree's 6 m; the crown carries the rest.
    expect(heightAboveRoot(rig, 9)).toBeCloseTo(4.2, 10);
  });
});

describe("checksumPositions", () => {
  const positions = new Float32Array([0, 0, 0, 1, 2, 3, -4.5, 0.25, 9]);

  it("is deterministic and records the splat count", () => {
    expect(checksumPositions(positions)).toBe(checksumPositions(positions.slice()));
    expect(checksumPositions(positions)).toMatch(/^fnv1a32:3:[0-9a-f]{8}$/);
  });

  it("changes when a single value changes", () => {
    const nudged = positions.slice();
    nudged[4] = 2.0000002;
    expect(checksumPositions(nudged)).not.toBe(checksumPositions(positions));
  });

  it("backs an exact refusal check", () => {
    const keyed: MotionRig = { ...rig, canonicalChecksum: checksumPositions(positions) };
    expect(positionsMatchChecksum(keyed, positions)).toBe(true);
    expect(positionsMatchChecksum(keyed, positions.slice(0, 6))).toBe(false);
    expect(positionsMatchChecksum(rig, positions)).toBe(false);
  });

  it("reads through a view's byte offset, not the whole buffer", () => {
    const backing = new Float32Array(20);
    backing.set(positions, 5);
    const view = backing.subarray(5, 14);
    expect(checksumPositions(view)).toBe(checksumPositions(positions));
  });
});

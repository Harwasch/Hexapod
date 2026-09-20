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
  it("is a skeleton of a few dozen nodes, not an index list", () => {
    expect(rig.nodes.length).toBe(33);
    expect(rig.nodes.length).toBeGreaterThanOrEqual(20);
    expect(rig.nodes.length).toBeLessThanOrEqual(60);
    expect(validateRig(rig)).toEqual([]);
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
    });
    expect(tall.nodes.length).toBe(8 + 2 * 4 * 3);
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
    expect(heightAboveRoot(rig, 5)).toBeCloseTo(6, 10);
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

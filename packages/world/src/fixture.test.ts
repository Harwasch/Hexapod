/**
 * The synthetic tree fixture, read from disk, as the contract between the two languages.
 *
 * `tools/captures/synthetic_tree.py` generates the tree and computes `canonicalChecksum` with
 * its own transcription of `checksumPositions`. The runtime refuses to deform when that
 * checksum does not match the splats it decoded, so a silent divergence between the Python
 * and TypeScript implementations would not throw — it would just leave the tree frozen, or,
 * worse, agree by accident on the wrong bytes. These tests pin the two to each other in CI.
 *
 * The matching Python half is `tools/captures/tests/test_synthetic_tree.py`.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  assignmentHistogram,
  assignSplatsToNodes,
  checksumPositions,
  parseRig,
  positionsMatchChecksum,
  syntheticTreeRig,
  validateRig,
} from "./index";

/** `data/tiles/synthetic-tree/source/`, from `packages/world/src/`. */
function fixturePath(name: string): string {
  return fileURLToPath(
    new URL(`../../../data/tiles/synthetic-tree/source/${name}`, import.meta.url),
  );
}

/**
 * Node's `readFileSync` returns a `Buffer` that may be a view into a shared, unaligned pool,
 * so the bytes are copied into a fresh buffer before being read as float32.
 */
function readFloat32(name: string): Float32Array {
  const bytes = Uint8Array.from(readFileSync(fixturePath(name)));
  return new Float32Array(bytes.buffer);
}

function readJson(name: string): unknown {
  return JSON.parse(readFileSync(fixturePath(name), "utf8"));
}

interface ChecksumCase {
  readonly name: string;
  readonly positionsHex: string;
  readonly checksum: string;
}

const checksumCases = (readJson("checksum_vectors.json") as { cases: ChecksumCase[] }).cases;
const rigText = readFileSync(fixturePath("rig.json"), "utf8");
const rig = parseRig(rigText);
const positions = readFloat32("positions.f32");
const truth = (readJson("labels.json") as { nodes: number[] }).nodes;

function positionsFromHex(hex: string): Float32Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i += 1) {
    bytes[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return new Float32Array(bytes.buffer);
}

describe("checksumPositions agrees with tools/captures/synthetic_tree.py", () => {
  it("has cases to check", () => {
    expect(checksumCases.length).toBeGreaterThanOrEqual(6);
    expect(checksumCases.map((c) => c.name)).toContain("negative-zero");
  });

  it.each(checksumCases)("matches Python on $name", ({ positionsHex, checksum }) => {
    expect(checksumPositions(positionsFromHex(positionsHex))).toBe(checksum);
  });

  it("distinguishes -0 from +0, so the digest is over bytes and not values", () => {
    const zero = checksumCases.find((c) => c.name === "origin");
    const negativeZero = checksumCases.find((c) => c.name === "negative-zero");
    expect(zero?.checksum).toBeDefined();
    expect(negativeZero?.checksum).not.toBe(zero?.checksum);
  });

  it("matches Python on the fixture's own 12,000 positions", () => {
    expect(checksumPositions(positions)).toBe(rig.canonicalChecksum);
    expect(positionsMatchChecksum(rig, positions)).toBe(true);
  });

  it("rejects the same positions with one bit changed", () => {
    const tampered = Float32Array.from(positions);
    tampered[1] = (tampered[1] ?? 0) + 2 ** -12;
    expect(positionsMatchChecksum(rig, tampered)).toBe(false);
  });
});

describe("the fixture rig", () => {
  it("parses and validates against the schema this package defines", () => {
    expect(validateRig(rig)).toEqual([]);
    expect(rig.units).toBe("meters");
    expect(rig.canonicalChecksum).toMatch(/^fnv1a32:12000:[0-9a-f]{8}$/);
  });

  it("is the same skeleton syntheticTreeRig() builds", () => {
    const mirror = syntheticTreeRig();
    expect(rig.nodes.length).toBe(mirror.nodes.length);
    rig.nodes.forEach((node, i) => {
      const expected = mirror.nodes[i];
      expect(node.id).toBe(expected?.id);
      expect(node.parent).toBe(expected?.parent);
      expect(node.band).toBe(expected?.band);
      expect(node.radius).toBeCloseTo(expected?.radius ?? Number.NaN, 12);
      expect(node.stiffness).toBeCloseTo(expected?.stiffness ?? Number.NaN, 12);
      node.position.forEach((value, axis) => {
        expect(value).toBeCloseTo(expected?.position[axis] ?? Number.NaN, 9);
      });
    });
  });
});

describe("assignSplatsToNodes against known ground truth", () => {
  const assignment = assignSplatsToNodes(positions, rig);

  it("labels every splat", () => {
    expect(assignment.length).toBe(positions.length / 3);
    expect(assignment.length).toBe(truth.length);
  });

  it("recovers the node each splat was generated from, for the large majority", () => {
    // Not all of them, honestly: a bark splat on the far side of the trunk is genuinely
    // nearer its neighbour's node than its own. This is the number to watch if the
    // assignment algorithm changes — it is scored against truth, not against itself.
    let agree = 0;
    for (let i = 0; i < truth.length; i += 1) if (assignment[i] === truth[i]) agree += 1;
    expect(agree / truth.length).toBeGreaterThan(0.85);
  });

  it("leaves no node of the rig without splats", () => {
    const histogram = assignmentHistogram(assignment, rig.nodes.length);
    expect([...histogram].filter((count) => count === 0)).toEqual([]);
  });

  it("does not depend on the order the splats arrive in", () => {
    const reversed = new Float32Array(positions.length);
    const count = positions.length / 3;
    for (let i = 0; i < count; i += 1) {
      const from = (count - 1 - i) * 3;
      reversed[i * 3] = positions[from] ?? 0;
      reversed[i * 3 + 1] = positions[from + 1] ?? 0;
      reversed[i * 3 + 2] = positions[from + 2] ?? 0;
    }
    const other = assignSplatsToNodes(reversed, rig);
    for (let i = 0; i < count; i += 1) {
      expect(other[count - 1 - i]).toBe(assignment[i]);
    }
  });
});

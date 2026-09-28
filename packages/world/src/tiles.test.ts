/**
 * What a level-of-detail tileset needs from the motion model: a gaussian's motion must be a
 * function of where it stands, not of which tile carries it or where that tile lands in the
 * viewer's aggregate — and the rig must be able to say which tiles it was built for.
 */

import { describe, expect, it } from "vitest";

import {
  applyFlutter,
  assignSplatsToNodes,
  deform,
  deformPositions,
  flutterField,
  flutterHash,
  FLUTTER_GPU_LAYOUT,
  hash32,
  parseRig,
  positionKey,
  positionKeys,
  rigTileChecksums,
  serializeRig,
  splatFlutter,
  syntheticTreeRig,
  validateRig,
  type MotionRig,
} from "./index";

const rig = syntheticTreeRig();
const GALE = { strength: 1, bearingDeg: 250 };

/** A deterministic cloud on the 1/4096 grid, spread through the crown. */
function cloud(count: number, seed: number): Float32Array {
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const h = hash32(i, seed);
    out[i * 3] = Math.round(((h & 0xff) / 255 - 0.5) * 4 * 4096) / 4096;
    out[i * 3 + 1] = Math.round((((h >>> 8) & 0xff) / 255 - 0.5) * 4 * 4096) / 4096;
    out[i * 3 + 2] = Math.round(((((h >>> 16) & 0xff) / 255) * 6 + 1) * 4096) / 4096;
  }
  return out;
}

describe("position keys", () => {
  it("are a function of the snapped position alone", () => {
    expect(positionKey(1.25, -2.5, 3)).toBe(positionKey(1.25, -2.5, 3));
    // Anything within half a grid step lands on the same key; one step does not.
    expect(positionKey(1.25 + 1e-6, -2.5, 3)).toBe(positionKey(1.25, -2.5, 3));
    expect(positionKey(1.25 + 1 / 4096, -2.5, 3)).not.toBe(positionKey(1.25, -2.5, 3));
    // Axis order matters: a key is not symmetric in its coordinates.
    expect(positionKey(1, 2, 3)).not.toBe(positionKey(3, 2, 1));
  });

  it("do not collide on a crown's worth of gaussians", () => {
    const keys = positionKeys(cloud(20000, 3));
    expect(new Set(keys).size).toBeGreaterThan(19900);
  });

  it("are what applyFlutter hashes when given, and the index when not", () => {
    const field = flutterField(rig, 4.5, GALE);
    const positions = cloud(300, 11);
    const assignment = assignSplatsToNodes(positions, rig);
    const keys = positionKeys(positions);
    const target = new Float32Array(positions.length);
    applyFlutter(target, assignment, field, 300, keys);
    for (let i = 0; i < 300; i += 1) {
      const expected = splatFlutter(keys[i] ?? 0, assignment[i] ?? 0, field);
      for (let k = 0; k < 3; k += 1) expect(target[i * 3 + k]).toBe(Math.fround(expected[k] ?? 0));
    }
  });
});

describe("a gaussian moves the same whichever tile carries it", () => {
  it("gives bit-identical displacement to one canonical position in two aggregates", () => {
    // Tile A holds the first half, tile B the second; a leaf-like view aggregates [A, B], a
    // reordered selection [B, A]. Every gaussian's output must not care.
    const positions = cloud(4000, 5);
    const half = positions.length / 2;
    const a = positions.subarray(0, half);
    const b = positions.subarray(half);
    const swapped = new Float32Array(positions.length);
    swapped.set(b, 0);
    swapped.set(a, b.length);

    const t = 7.25;
    const transforms = deform(rig, t, GALE);
    const field = flutterField(rig, t, GALE);
    const run = (input: Float32Array): Float32Array =>
      deformPositions(
        input,
        assignSplatsToNodes(input, rig),
        transforms,
        undefined,
        field,
        positionKeys(input),
      );
    const first = run(positions);
    const second = run(swapped);
    const offset = b.length;
    for (let i = 0; i < a.length; i += 1) expect(second[offset + i]).toBe(first[i]);
    for (let i = 0; i < b.length; i += 1) expect(second[i]).toBe(first[half + i]);

    // And by index, which is what a tiled capture must not use, they would not agree.
    const byIndex = (input: Float32Array): Float32Array =>
      deformPositions(input, assignSplatsToNodes(input, rig), transforms, undefined, field);
    const indexed = byIndex(positions);
    const reindexed = byIndex(swapped);
    let differs = 0;
    for (let i = 0; i < a.length; i += 1) if (reindexed[offset + i] !== indexed[i]) differs += 1;
    expect(differs).toBeGreaterThan(0);
  });

  it("moves a merged parent with the node its own position binds to", () => {
    // A parent gaussian at the centroid of a leaf cluster: bound by the same nearest-node rule,
    // so it rides the cluster rather than staying behind or following the trunk.
    const leaf = rig.nodes.findIndex((node) => node.band === "leaf");
    const node = rig.nodes[leaf];
    expect(node).toBeDefined();
    if (node === undefined) return;
    const parent = new Float32Array(node.position.map((v) => Math.round(v * 4096) / 4096));
    expect(assignSplatsToNodes(parent, rig)[0]).toBe(leaf);
  });
});

describe("the GPU layout is the CPU's", () => {
  it("hashes a key exactly as applyFlutter does", () => {
    for (const key of [0, 1, 77, 0xffffffff, positionKey(1, 2, 3)]) {
      expect(flutterHash(key)).toBe(hash32(key, 0x5eedf1c7));
    }
    expect(FLUTTER_GPU_LAYOUT.phaseSteps).toBe(1024);
    expect(FLUTTER_GPU_LAYOUT.axes.length).toBe(FLUTTER_GPU_LAYOUT.axisCount * 6);
    // Unit, orthogonal pairs: what the shader reads is a plane, not a skew.
    const axes = FLUTTER_GPU_LAYOUT.axes;
    for (let k = 0; k < FLUTTER_GPU_LAYOUT.axisCount; k += 1) {
      const [ax, ay, az, bx, by, bz] = [0, 1, 2, 3, 4, 5].map((j) => axes[k * 6 + j] ?? 0);
      expect(Math.hypot(ax ?? 0, ay ?? 0, az ?? 0)).toBeCloseTo(1, 12);
      expect(Math.hypot(bx ?? 0, by ?? 0, bz ?? 0)).toBeCloseTo(1, 12);
      expect((ax ?? 0) * (bx ?? 0) + (ay ?? 0) * (by ?? 0) + (az ?? 0) * (bz ?? 0)).toBeCloseTo(
        0,
        12,
      );
    }
  });
});

describe("tileChecksums", () => {
  const stamped: MotionRig = {
    ...rig,
    tileChecksums: ["fnv1a32:114:f34d4e26", "fnv1a32:1157:9d336a44"],
  };

  it("round-trips through JSON and is accepted by the validator", () => {
    expect(validateRig(stamped)).toEqual([]);
    const parsed = parseRig(serializeRig(stamped));
    expect(parsed.tileChecksums).toEqual(stamped.tileChecksums);
    // A rig without it serialises as before, so single-tile rigs are byte-stable.
    expect(serializeRig(rig)).not.toContain("tileChecksums");
    expect(parseRig(serializeRig(rig)).tileChecksums).toBeUndefined();
  });

  it("is the accepted set when present, the one canonical checksum when not", () => {
    expect([...rigTileChecksums(stamped)]).toEqual(stamped.tileChecksums);
    expect([...rigTileChecksums(rig)]).toEqual([rig.canonicalChecksum]);
  });

  it("rejects an empty list or something that is not a digest", () => {
    expect(validateRig({ ...rig, tileChecksums: [] }).join()).toContain("non-empty");
    expect(validateRig({ ...rig, tileChecksums: ["sha256:abc"] }).join()).toContain("digests");
    const text = JSON.parse(serializeRig(rig)) as Record<string, unknown>;
    expect(() => parseRig(JSON.stringify({ ...text, tileChecksums: "fnv" }))).toThrow(/array/);
  });
});

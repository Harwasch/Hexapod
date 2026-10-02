/**
 * Smooth skinning (`skin.ts`): the search is exact, the weights are what the formula says, and
 * the rigid skin is the old binding.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  assignSplatsToNodes,
  deformPositions,
  IDENTITY_TRANSFORM,
  parseRig,
  quantiseSkinWeights,
  rigidSkin,
  SKIN_INFLUENCES,
  SKIN_WEIGHT_TOTAL,
  skinPrimary,
  skinSplatsToNodes,
  syntheticTreeRig,
  type MotionRig,
  type NodeTransform,
} from "./index";

const minnetonka = parseRig(
  readFileSync(fileURLToPath(new URL("../fixtures/minnetonka/rig.json", import.meta.url)), "utf8"),
);
const synthetic = syntheticTreeRig();

/** Deterministic points: around the nodes, between them, far outside, and exactly on some. */
function probes(rig: MotionRig, count: number): Float32Array {
  const out = new Float32Array(count * 3);
  let s = 12345;
  const rand = (): number => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return s / 2 ** 32;
  };
  for (let i = 0; i < count; i += 1) {
    const a = rig.nodes[Math.floor(rand() * rig.nodes.length)]?.position ?? [0, 0, 0];
    const b = rig.nodes[Math.floor(rand() * rig.nodes.length)]?.position ?? [0, 0, 0];
    const t = rand();
    const kind = i % 10;
    const spread = kind === 9 ? 20 : 0.3;
    for (let c = 0; c < 3; c += 1) {
      const onNode = kind === 8;
      out[i * 3 + c] = onNode
        ? (a[c] ?? 0)
        : (a[c] ?? 0) + ((b[c] ?? 0) - (a[c] ?? 0)) * t * 0.3 + (rand() - 0.5) * spread;
    }
  }
  return out;
}

/** The brute-force five nearest, ties to the lower index. */
function bruteForce(rig: MotionRig, x: number, y: number, z: number): [number[], number[]] {
  const all = rig.nodes.map((node, n) => {
    const dx = x - node.position[0];
    const dy = y - node.position[1];
    const dz = z - node.position[2];
    return { n, d2: dx * dx + dy * dy + dz * dz };
  });
  all.sort((p, q) => p.d2 - q.d2 || p.n - q.n);
  const five = all.slice(0, SKIN_INFLUENCES + 1);
  return [five.map((e) => e.n), five.map((e) => e.d2)];
}

describe("the search", () => {
  it.each([
    ["synthetic", synthetic],
    ["minnetonka", minnetonka],
  ])("finds exactly the brute-force four on the %s rig, slot 0 = assignSplatsToNodes", (_, rig) => {
    const positions = probes(rig, 4000);
    const skin = skinSplatsToNodes(positions, rig);
    const assignment = assignSplatsToNodes(positions, rig);
    expect(Array.from(skinPrimary(skin))).toEqual(Array.from(assignment));
    const expected = new Uint16Array(SKIN_INFLUENCES);
    for (let i = 0; i < 4000; i += 1) {
      const [nodes, d2] = bruteForce(
        rig,
        positions[i * 3] ?? 0,
        positions[i * 3 + 1] ?? 0,
        positions[i * 3 + 2] ?? 0,
      );
      quantiseSkinWeights(d2, expected, 0);
      for (let k = 0; k < SKIN_INFLUENCES; k += 1) {
        expect(skin.weights[i * SKIN_INFLUENCES + k]).toBe(expected[k]);
        if ((expected[k] ?? 0) > 0) expect(skin.nodes[i * SKIN_INFLUENCES + k]).toBe(nodes[k]);
      }
    }
  });

  it("measures binding cost against the nearest-node assignment", () => {
    const positions = probes(minnetonka, 150_000);
    const start = process.hrtime.bigint();
    assignSplatsToNodes(positions, minnetonka);
    const assignMs = Number(process.hrtime.bigint() - start) / 1e6;
    const mid = process.hrtime.bigint();
    skinSplatsToNodes(positions, minnetonka);
    const skinMs = Number(process.hrtime.bigint() - mid) / 1e6;
    console.info(
      `MEASURED binding 150,000 splats to the 200-node Minnetonka rig: nearest node ${assignMs.toFixed(1)} ms, four-node skin ${skinMs.toFixed(1)} ms`,
    );
    expect(skinMs).toBeLessThan(5000);
  });
});

describe("the weights", () => {
  it("sum to 1023, nearest heaviest, all on a node when the splat is on it", () => {
    const positions = probes(synthetic, 3000);
    const skin = skinSplatsToNodes(positions, synthetic);
    for (let i = 0; i < 3000; i += 1) {
      let sum = 0;
      for (let k = 0; k < SKIN_INFLUENCES; k += 1) {
        sum += skin.weights[i * SKIN_INFLUENCES + k] ?? 0;
        if (k > 0)
          expect(skin.weights[i * SKIN_INFLUENCES + k]).toBeLessThanOrEqual(
            (skin.weights[i * SKIN_INFLUENCES] ?? 0) + 1,
          );
      }
      expect(sum).toBe(SKIN_WEIGHT_TOTAL);
      if (i % 10 === 8) expect(skin.weights[i * SKIN_INFLUENCES]).toBe(SKIN_WEIGHT_TOTAL);
    }
  });

  it("follow the modified Shepard formula, and vanish at the fifth node's distance", () => {
    const weights = new Uint16Array(SKIN_INFLUENCES);
    quantiseSkinWeights([1, 4, 9, 16, 25], weights, 0);
    // (1/d − 1/5)²: 0.64, 0.09, 0.0178, 0.0025
    const raw = [0.64, 0.09, (1 / 3 - 0.2) ** 2, 0.05 ** 2];
    const total = raw.reduce((a, b) => a + b, 0);
    for (let k = 1; k < 4; k += 1)
      expect(weights[k]).toBe(Math.round(((raw[k] ?? 0) / total) * SKIN_WEIGHT_TOTAL));
    quantiseSkinWeights([1, 4, 9, 25, 25], weights, 0);
    expect(weights[3]).toBe(0);
  });
});

describe("the rigid skin", () => {
  it("is the nearest-node binding", () => {
    const positions = probes(synthetic, 500);
    const assignment = assignSplatsToNodes(positions, synthetic);
    const transforms: NodeTransform[] = synthetic.nodes.map((_, n) =>
      n === 0
        ? IDENTITY_TRANSFORM
        : {
            rotation: [0.01 * Math.sin(n), 0.02 * Math.cos(n), 0.005, 1] as const,
            translation: [0.001 * n, -0.002, 0.0005] as const,
          },
    );
    // Normalise the quaternions so both paths see a rotation.
    const normalised = transforms.map((t) => {
      const [x, y, z, w] = t.rotation;
      const l = Math.hypot(x, y, z, w);
      return { rotation: [x / l, y / l, z / l, w / l] as const, translation: t.translation };
    });
    const rigid = deformPositions(positions, assignment, normalised);
    const skinned = deformPositions(
      positions,
      assignment,
      normalised,
      undefined,
      undefined,
      undefined,
      rigidSkin(assignment),
    );
    for (let i = 0; i < rigid.length; i += 1)
      expect(Math.abs((rigid[i] ?? 0) - (skinned[i] ?? 0))).toBeLessThan(2e-6);
  });
});

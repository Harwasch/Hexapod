import { describe, expect, it } from "vitest";

import {
  applyTransform,
  cross,
  deform,
  deformPositions,
  DEG_TO_RAD,
  dot,
  IDENTITY_TRANSFORM,
  magnitude,
  maxDeformSpeed,
  maxDisplacement,
  maxNodeAngle,
  maxNodeDisplacements,
  nodeAngleLimit,
  nodeDisplacement,
  nodeModes,
  normalize,
  parseRig,
  quatAngle,
  quatConjugate,
  quatMultiply,
  QUAT_IDENTITY,
  serializeRig,
  subtract,
  syntheticTreeRig,
  VEC3_UP,
  VEC3_ZERO,
  wind,
  type MotionRig,
  type NodeTransform,
  type WindSettings,
} from "./index";

const rig = syntheticTreeRig();
const GALE: WindSettings = { strength: 1, bearingDeg: 90 };
/** 0..120 s at 1/60, the sweep the boundedness and continuity properties are asserted over. */
const SWEEP_FRAMES = 120 * 60;
const DT = 1 / 60;

function frame(bearingDeg: number): { down: [number, number]; across: [number, number] } {
  const b = bearingDeg * DEG_TO_RAD;
  return { down: [Math.sin(b), Math.cos(b)], across: [Math.cos(b), -Math.sin(b)] };
}

describe("determinism", () => {
  it("returns bit-identical transforms across two calls", () => {
    for (let k = 0; k < 400; k += 1) {
      const t = k * 0.29;
      expect(deform(rig, t, GALE)).toEqual(deform(rig, t, GALE));
    }
  });

  it("returns bit-identical transforms across a rig JSON round-trip", () => {
    const reloaded: MotionRig = parseRig(serializeRig(rig));
    for (let k = 0; k < 200; k += 1) {
      const t = k * 0.61;
      expect(deform(reloaded, t, GALE)).toEqual(deform(rig, t, GALE));
    }
  });

  it("holds no state between calls: replaying out of order changes nothing", () => {
    const forward = [0, 1, 2, 3, 4].map((t) => deform(rig, t, GALE));
    const backward = [4, 3, 2, 1, 0].map((t) => deform(rig, t, GALE)).reverse();
    expect(backward).toEqual(forward);
  });
});

describe("zero-wind identity", () => {
  it("gives exactly identity transforms, not approximately", () => {
    for (const bearingDeg of [0, 90, 180, 270, 45.5]) {
      for (let k = 0; k < 200; k += 1) {
        const transforms = deform(rig, k * 0.53, { strength: 0, bearingDeg });
        for (const transform of transforms) {
          expect(transform.rotation).toEqual(QUAT_IDENTITY);
          expect(transform.translation).toEqual(VEC3_ZERO);
        }
      }
    }
  });

  it("recovers the canonical pose bit-for-bit through deformPositions", () => {
    const canonical = new Float32Array([0, 0, 0, 1.25, -3.5, 4, -0.5, 0.5, 6.25]);
    const assignment = new Uint16Array([0, 7, 20]);
    const out = deformPositions(
      canonical,
      assignment,
      deform(rig, 91.7, { strength: 0, bearingDeg: 12 }),
    );
    expect([...out]).toEqual([...canonical]);
  });

  it("leaves the canonical positions untouched, always", () => {
    const canonical = new Float32Array([0.5, 0.25, 3, -1, 2, 5]);
    const before = [...canonical];
    deformPositions(canonical, new Uint16Array([9, 14]), deform(rig, 3.5, GALE));
    expect([...canonical]).toEqual(before);
  });

  it("refuses to write its output over the canonical buffer", () => {
    const canonical = new Float32Array([0, 0, 1]);
    expect(() =>
      deformPositions(canonical, new Uint16Array([1]), deform(rig, 1, GALE), canonical),
    ).toThrow(/canonical/);
  });
});

describe("anchoring", () => {
  it("fixes the trunk base: the root transform is exactly identity", () => {
    for (let k = 0; k < 300; k += 1) {
      const root = deform(rig, k * 0.41, GALE)[0];
      expect(root).toEqual(IDENTITY_TRANSFORM);
      expect(root?.translation).toEqual([0, 0, 0]);
    }
  });

  it("holds for a root that is not at the origin", () => {
    const shifted: MotionRig = {
      ...rig,
      nodes: rig.nodes.map((n) => ({
        ...n,
        position: [n.position[0] + 13.5, n.position[1] - 4.25, n.position[2] + 2] as const,
      })),
    };
    const base = shifted.nodes[0];
    expect(base).toBeDefined();
    if (!base) return;
    for (let k = 0; k < 200; k += 1) {
      const transforms = deform(shifted, k * 0.37, GALE);
      const root = transforms[0];
      expect(root?.translation).toEqual([0, 0, 0]);
      expect(root && applyTransform(root, base.position)).toEqual([...base.position]);
      expect(magnitude(nodeDisplacement(shifted, transforms, 0))).toBe(0);
    }
  });

  it("never lets the tree slide: the base is fixed at every strength", () => {
    for (const strength of [0, 0.01, 0.3, 0.9, 1]) {
      const transforms = deform(rig, 17.25, { strength, bearingDeg: 200 });
      expect(magnitude(nodeDisplacement(rig, transforms, 0))).toBe(0);
    }
  });
});

describe("parent composition", () => {
  it("joins child to parent at the child's pivot: no gaps in the chain", () => {
    for (let k = 0; k < 120; k += 1) {
      const transforms = deform(rig, k * 0.83, GALE);
      rig.nodes.forEach((node, i) => {
        if (node.parent < 0) return;
        const child = transforms[i];
        const parent = transforms[node.parent];
        expect(child).toBeDefined();
        expect(parent).toBeDefined();
        if (!child || !parent) return;
        const viaChild = applyTransform(child, node.position);
        const viaParent = applyTransform(parent, node.position);
        expect(magnitude(subtract(viaChild, viaParent))).toBeLessThan(1e-12);
      });
    }
  });

  it("accumulates rotation down the chain: a leaf turns more than its trunk", () => {
    const transforms = deform(rig, 8.5, GALE);
    const leaf = transforms.at(-1);
    const trunkMid = transforms[3];
    expect(leaf).toBeDefined();
    expect(trunkMid).toBeDefined();
    if (!leaf || !trunkMid) return;
    expect(quatAngle(leaf.rotation)).toBeGreaterThan(quatAngle(trunkMid.rotation));
  });
});

describe("per-node angular limits", () => {
  it("keeps every local bend inside its node's cap over the sweep", () => {
    for (let k = 0; k <= SWEEP_FRAMES; k += 13) {
      const transforms = deform(rig, k * DT, GALE);
      rig.nodes.forEach((node, i) => {
        if (node.parent < 0) return;
        const child = transforms[i];
        const parent = transforms[node.parent];
        if (!child || !parent) return;
        const local = quatMultiply(quatConjugate(parent.rotation), child.rotation);
        expect(quatAngle(local)).toBeLessThanOrEqual(nodeAngleLimit(node) + 1e-9);
      });
    }
  });

  it("saturates smoothly: the cap is approached but never reached", () => {
    const strong = { strength: 1, bearingDeg: 0 };
    const modes = nodeModes(rig);
    rig.nodes.forEach((node, i) => {
      if (node.parent < 0) return;
      const mode = modes[i];
      expect(mode).toBeDefined();
      if (!mode) return;
      expect(maxNodeAngle(node, mode, strong)).toBeLessThan(nodeAngleLimit(node));
    });
    const leaf = rig.nodes.at(-1);
    const leafMode = modes.at(-1);
    expect(leaf).toBeDefined();
    expect(leafMode).toBeDefined();
    if (!leaf || !leafMode) return;
    // A leaf at full strength is genuinely on its limiter, not merely under the cap by luck.
    expect(maxNodeAngle(leaf, leafMode, strong)).toBeGreaterThan(0.7 * nodeAngleLimit(leaf));
  });

  it("honours a per-node override, which binds that node's descendants", () => {
    // A node rotates about its own rest position, so its own limit does not move itself: it
    // caps everything hanging off it. Pinning the branches must therefore quieten the leaves.
    const pinned: MotionRig = {
      ...rig,
      nodes: rig.nodes.map((n) => (n.band === "branch" ? { ...n, maxAngleRad: 0.001 } : n)),
    };
    const bounds = maxNodeDisplacements(pinned, GALE);
    const loose = maxNodeDisplacements(rig, GALE);
    let checked = 0;
    pinned.nodes.forEach((node, i) => {
      if (node.band !== "leaf") return;
      expect(bounds[i] ?? 0).toBeLessThan(loose[i] ?? 0);
      checked += 1;
    });
    expect(checked).toBeGreaterThan(0);
    expect(maxDisplacement(pinned, GALE)).toBeLessThan(maxDisplacement(rig, GALE));
  });
});

describe("boundedness", () => {
  it("stays inside the per-node displacement cap over a dense 0..120 s sweep", () => {
    const bounds = maxNodeDisplacements(rig, GALE);
    let worstRatio = 0;
    for (let k = 0; k <= SWEEP_FRAMES; k += 1) {
      const transforms = deform(rig, k * DT, GALE);
      for (let i = 0; i < rig.nodes.length; i += 1) {
        const moved = magnitude(nodeDisplacement(rig, transforms, i));
        const cap = bounds[i] ?? 0;
        expect(moved).toBeLessThanOrEqual(cap + 1e-12);
        if (cap > 0) worstRatio = Math.max(worstRatio, moved / cap);
      }
    }
    // The cap is an upper bound, but a useful one: the tree actually gets near it.
    expect(worstRatio).toBeGreaterThan(0.5);
    expect(maxDisplacement(rig, GALE)).toBeLessThan(2);
  });

  it("is monotone in strength", () => {
    const strengths = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 1];
    let previousBound = -1;
    for (const strength of strengths) {
      const settings = { strength, bearingDeg: 90 };
      const bound = maxDisplacement(rig, settings);
      expect(bound).toBeGreaterThan(previousBound);
      previousBound = bound;
    }
    expect(maxDisplacement(rig, { strength: 0, bearingDeg: 90 })).toBe(0);

    // And the actual motion, at a frozen instant, is monotone too.
    for (const t of [1.5, 33.25, 97]) {
      let previous = -1;
      for (const strength of strengths) {
        const transforms = deform(rig, t, { strength, bearingDeg: 90 });
        const moved = magnitude(nodeDisplacement(rig, transforms, rig.nodes.length - 1));
        expect(moved).toBeGreaterThan(previous);
        previous = moved;
      }
    }
  });

  it("is monotone in height above the trunk base", () => {
    const trunk = rig.nodes
      .map((node, index) => ({ node, index }))
      .filter((entry) => entry.node.band === "trunk")
      .sort((a, b) => a.node.position[2] - b.node.position[2]);
    expect(trunk.length).toBeGreaterThanOrEqual(5);

    // The *bound* is monotone by construction — it is a sum of non-negative terms over an
    // ancestor chain, and a higher node's chain contains a lower one's — so that part is exact.
    const bounds = maxNodeDisplacements(rig, GALE);
    let previousBound = -Number.EPSILON;
    for (const { index } of trunk) {
      const bound = bounds[index] ?? 0;
      expect(bound).toBeGreaterThanOrEqual(previousBound);
      previousBound = bound;
    }

    // The motion itself is monotone on average and not at every instant, and that is a fact
    // about the model rather than a weakness of the test. Each joint bends in its own plane, so
    // two joints of the chain can be momentarily opposed and a node can sit nearer its rest
    // position than the one below it — by 6 % at the worst moment measured. Asserting the
    // frozen instant passed for a long time by luck and started failing when the wind became a
    // field; the mean over a sweep is the claim that was always meant.
    const frames = 40 * 60;
    const means = trunk.map(({ index }) => {
      let total = 0;
      for (let k = 0; k < frames; k += 1) {
        total += magnitude(nodeDisplacement(rig, deform(rig, k * DT, GALE), index));
      }
      return total / frames;
    });
    let previousMean = -Number.EPSILON;
    for (const mean of means) {
      expect(mean).toBeGreaterThanOrEqual(previousMean);
      previousMean = mean;
    }
    expect(means.at(-1)).toBeGreaterThan(0.05);
  });
});

describe("continuity", () => {
  it("bounds the deformation speed: no jumps where the gust noise wraps", () => {
    const speedCap = maxDeformSpeed(rig, GALE);
    let worst = 0;
    const posed = (t: number) => {
      const transforms = deform(rig, t, GALE);
      return rig.nodes.map((node, i) =>
        applyTransform(transforms[i] ?? IDENTITY_TRANSFORM, node.position),
      );
    };
    let previous = posed(0);
    for (let k = 1; k <= SWEEP_FRAMES; k += 1) {
      const current = posed(k * DT);
      for (let i = 0; i < current.length; i += 1) {
        const a = previous[i];
        const b = current[i];
        if (!a || !b) continue;
        worst = Math.max(worst, magnitude(subtract(b, a)) / DT);
      }
      previous = current;
    }
    expect(worst).toBeLessThanOrEqual(speedCap);
    expect(worst).toBeGreaterThan(0.01);
  });

  it("converges as the step shrinks, so the motion is differentiable, not merely bounded", () => {
    const node = rig.nodes.length - 1;
    const at = (t: number) => nodeDisplacement(rig, deform(rig, t, GALE), node);
    const speedAt = (h: number) => magnitude(subtract(at(30 + h), at(30))) / h;
    // The steps are two decades apart and both far below a frame. A frame is no longer a small
    // step for this model: the fastest forcing component turns over a tenth of a cycle in 1/60 s,
    // so a difference quotient taken across a whole frame measures the chord, not the tangent.
    // Differentiability is a statement about the limit, and this is the limit taken properly.
    const coarse = speedAt(1 / 6000);
    const fine = speedAt(1 / 600000);
    expect(Math.abs(coarse - fine)).toBeLessThan(0.2 * Math.max(fine, 1e-6));
  });
});

describe("direction", () => {
  it("pushes downwind on average, with a cross-wind mean of about zero", () => {
    for (const bearingDeg of [0, 90, 180, 270, 143]) {
      const { down, across } = frame(bearingDeg);
      const node = rig.nodes.length - 1;
      let along = 0;
      let cross = 0;
      for (let k = 0; k <= SWEEP_FRAMES; k += 1) {
        const d = nodeDisplacement(rig, deform(rig, k * DT, { strength: 1, bearingDeg }), node);
        along += d[0] * down[0] + d[1] * down[1];
        cross += d[0] * across[0] + d[1] * across[1];
      }
      const n = SWEEP_FRAMES + 1;
      expect(along / n).toBeGreaterThan(0.05);
      expect(Math.abs(cross / n)).toBeLessThan(0.05 * (along / n));
    }
  });

  it("gives every node its own bending plane: the crown is not one flat sheet", () => {
    // This is the exact inverse of the assertion it replaces, and the inversion is deliberate.
    //
    // The old test read: "bends every node in one plane: no displacement along the rotation
    // axis", and it passed, because every local rotation shared the single axis
    // `up x downwind` computed once for the whole rig. That is not a safety property. It is a
    // statement that the tree differs from a rigid plate only in amplitude, which is why no
    // amount of tuning could have made the old model read as a tree, and the test that proved
    // it was proving the defect.
    //
    // What is asserted instead: a real fraction of the motion lies along that old shared axis.
    // Each node now sways about its own plane — mostly downwind, swung and twisted by its own
    // azimuth — while the steady lean stays exactly downwind for everyone, which is what steady
    // drag is. The measured fraction is about 0.13 of the RMS displacement; 0.05 is asserted, a
    // margin below it, and the old model would have scored exactly 0.
    let acrossPlane = 0;
    let total = 0;
    for (let k = 0; k <= SWEEP_FRAMES; k += 29) {
      const t = k * DT;
      const axis = normalize(cross(VEC3_UP, wind(GALE.strength, GALE.bearingDeg, t)), [1, 0, 0]);
      const transforms = deform(rig, t, GALE);
      for (let i = 0; i < rig.nodes.length; i += 1) {
        const moved = nodeDisplacement(rig, transforms, i);
        const along = dot(moved, axis);
        acrossPlane += along * along;
        total += dot(moved, moved);
      }
    }
    expect(total).toBeGreaterThan(0);
    expect(Math.sqrt(acrossPlane / total)).toBeGreaterThan(0.05);
  });

  it("keeps the steady lean exactly downwind, whatever plane a node sways in", () => {
    // The per-node planes are given to the oscillation only. Averaged over a long sweep the
    // oscillation cancels and what is left is the lean, which must be downwind for every node —
    // a branch does not acquire a permanent sideways set because of how it happens to shake.
    const { down, across } = frame(GALE.bearingDeg);
    const posed: NodeTransform[][] = [];
    for (let k = 0; k <= SWEEP_FRAMES; k += 1) posed.push(deform(rig, k * DT, GALE));
    let checked = 0;
    rig.nodes.forEach((node, i) => {
      let along = 0;
      let sideways = 0;
      let travelled = 0;
      for (const transforms of posed) {
        const d = nodeDisplacement(rig, transforms, i);
        along += d[0] * down[0] + d[1] * down[1];
        sideways += d[0] * across[0] + d[1] * across[1];
        travelled += magnitude(d);
      }
      // The root and the first trunk joint rotate about themselves with no rotating ancestor,
      // so they are fixed by construction and have no direction to speak of.
      if (travelled === 0) return;
      const n = posed.length;
      expect(along / n).toBeGreaterThan(0);
      expect(Math.abs(sideways / n)).toBeLessThan(0.25 * (along / n));
      checked += 1;
    });
    expect(checked).toBeGreaterThan(rig.nodes.length - 4);
  });

  it("rotates with the bearing rather than ignoring it", () => {
    const east = nodeDisplacement(rig, deform(rig, 9, { strength: 1, bearingDeg: 90 }), 32);
    const north = nodeDisplacement(rig, deform(rig, 9, { strength: 1, bearingDeg: 0 }), 32);
    expect(east[0]).toBeGreaterThan(Math.abs(east[1]));
    expect(north[1]).toBeGreaterThan(Math.abs(north[0]));
    // A node on the trunk axis is rotationally symmetric, so only its direction may change.
    const trunkEast = nodeDisplacement(rig, deform(rig, 9, { strength: 1, bearingDeg: 90 }), 5);
    const trunkNorth = nodeDisplacement(rig, deform(rig, 9, { strength: 1, bearingDeg: 0 }), 5);
    expect(magnitude(trunkEast)).toBeCloseTo(magnitude(trunkNorth), 9);
    expect(trunkEast[0]).toBeCloseTo(trunkNorth[1], 9);
  });
});

describe("deformPositions", () => {
  it("moves each splat by its assigned node's transform", () => {
    const canonical = new Float32Array([0, 0, 0, 0.5, 0.5, 5.5, -0.25, 0.75, 4]);
    const assignment = new Uint16Array([0, 30, 11]);
    const transforms = deform(rig, 22.5, GALE);
    const out = deformPositions(canonical, assignment, transforms);
    for (let i = 0; i < 3; i += 1) {
      const transform = transforms[assignment[i] ?? 0] ?? IDENTITY_TRANSFORM;
      const expected = applyTransform(transform, [
        canonical[i * 3] ?? 0,
        canonical[i * 3 + 1] ?? 0,
        canonical[i * 3 + 2] ?? 0,
      ]);
      expect(out[i * 3]).toBeCloseTo(expected[0], 6);
      expect(out[i * 3 + 1]).toBeCloseTo(expected[1], 6);
      expect(out[i * 3 + 2]).toBeCloseTo(expected[2], 6);
    }
  });

  it("reuses a caller-supplied output buffer", () => {
    const canonical = new Float32Array([1, 2, 3]);
    const out = new Float32Array(3);
    expect(deformPositions(canonical, new Uint16Array([5]), deform(rig, 1, GALE), out)).toBe(out);
  });
});

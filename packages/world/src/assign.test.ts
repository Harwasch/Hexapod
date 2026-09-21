import { describe, expect, it } from "vitest";

import {
  assignmentHistogram,
  assignSplatsToNodes,
  deform,
  hash32,
  nodeDisplacement,
  magnitude,
  syntheticTreeRig,
  UNASSIGNED_NODE,
  type MotionRig,
  type SkeletonBand,
} from "./index";

const rig = syntheticTreeRig();

/** A seeded shuffle, so the "order does not matter" test is itself reproducible. */
function shuffledOrder(count: number, seed: number): number[] {
  const order = Array.from({ length: count }, (_, i) => i);
  for (let i = count - 1; i > 0; i -= 1) {
    const j = hash32(i, seed) % (i + 1);
    const a = order[i];
    const b = order[j];
    if (a === undefined || b === undefined) continue;
    order[i] = b;
    order[j] = a;
  }
  return order;
}

/** Splats hugging the trunk, from the base to the top of the bole. */
function trunkSplats(count: number): Float32Array {
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const f = i / (count - 1);
    const angle = i * 2.399963;
    const radius = 0.3 * (1 - 0.6 * f);
    out[i * 3] = Math.cos(angle) * radius;
    out[i * 3 + 1] = Math.sin(angle) * radius;
    out[i * 3 + 2] = f * 5.4;
  }
  return out;
}

function bandsOf(assignment: Uint16Array, from: MotionRig): Set<SkeletonBand | undefined> {
  return new Set([...assignment].map((index) => from.nodes[index]?.band));
}

describe("assignSplatsToNodes", () => {
  it("returns one node index per splat", () => {
    const positions = new Float32Array(30);
    const assignment = assignSplatsToNodes(positions, rig);
    expect(assignment).toBeInstanceOf(Uint16Array);
    expect(assignment.length).toBe(10);
  });

  it("picks the nearest node", () => {
    for (let n = 0; n < rig.nodes.length; n += 1) {
      const node = rig.nodes[n];
      if (!node) continue;
      const nudged = new Float32Array([
        node.position[0] + 0.001,
        node.position[1] - 0.001,
        node.position[2] + 0.001,
      ]);
      expect(assignSplatsToNodes(nudged, rig)[0]).toBe(n);
    }
  });

  it("never gives a trunk splat a canopy transform", () => {
    const positions = trunkSplats(500);
    const assignment = assignSplatsToNodes(positions, rig);
    expect([...bandsOf(assignment, rig)]).toEqual(["trunk"]);

    // And the consequence that actually matters: the motion they get is trunk motion.
    const gale = { strength: 1, bearingDeg: 90 };
    const transforms = deform(rig, 14.75, gale);
    const trunkCap = Math.max(
      ...rig.nodes.map((node, i) =>
        node.band === "trunk" ? magnitude(nodeDisplacement(rig, transforms, i)) : 0,
      ),
    );
    for (const index of assignment) {
      expect(magnitude(nodeDisplacement(rig, transforms, index))).toBeLessThanOrEqual(trunkCap);
    }

    // And the canopy reaches half again as far as any trunk node does. Two numbers moved here
    // and both for the same reason. It is measured as a reach over a sweep rather than at one
    // instant, because nodes now ring at their own frequencies and which of them happens to be
    // near its extreme at a frozen `t` is a coincidence. And the factor is 1.5 where it was 2:
    // a chain of nodes that all bent in phase about one shared axis summed its contributions
    // coherently, which is precisely the defect that made the crown move as a single sheet. Out
    // of phase they partly cancel, so the canopy's lead over the trunk is smaller and the
    // motion is a tree rather than a plate. The separation the test exists to prove — a trunk
    // splat never receives canopy motion — is the assertion above, and it is unchanged.
    let trunkReach = 0;
    let canopyReach = 0;
    for (let k = 0; k <= 20 * 60; k += 1) {
      const posed = deform(rig, k / 60, gale);
      rig.nodes.forEach((node, i) => {
        const moved = magnitude(nodeDisplacement(rig, posed, i));
        if (node.band === "trunk") trunkReach = Math.max(trunkReach, moved);
        if (node.band === "leaf") canopyReach = Math.max(canopyReach, moved);
      });
    }
    expect(canopyReach).toBeGreaterThan(1.5 * trunkReach);
  });

  it("is independent of the order the splats arrive in", () => {
    const positions = new Float32Array(3000);
    for (let i = 0; i < 1000; i += 1) {
      positions[i * 3] = ((hash32(i, 11) / 0x1_0000_0000) * 2 - 1) * 3;
      positions[i * 3 + 1] = ((hash32(i, 22) / 0x1_0000_0000) * 2 - 1) * 3;
      positions[i * 3 + 2] = (hash32(i, 33) / 0x1_0000_0000) * 7;
    }
    const direct = assignSplatsToNodes(positions, rig);

    const order = shuffledOrder(1000, 99);
    expect(order.slice(0, 20)).not.toEqual(Array.from({ length: 20 }, (_, i) => i));
    const shuffled = new Float32Array(3000);
    order.forEach((source, target) => {
      shuffled[target * 3] = positions[source * 3] ?? 0;
      shuffled[target * 3 + 1] = positions[source * 3 + 1] ?? 0;
      shuffled[target * 3 + 2] = positions[source * 3 + 2] ?? 0;
    });
    const fromShuffled = assignSplatsToNodes(shuffled, rig);
    order.forEach((source, target) => {
      expect(fromShuffled[target]).toBe(direct[source]);
    });
  });

  it("breaks ties towards the lower node index, so ties are order-free too", () => {
    const a = rig.nodes[6];
    const b = rig.nodes[9];
    expect(a).toBeDefined();
    expect(b).toBeDefined();
    if (!a || !b) return;
    const midpoint = new Float32Array([
      (a.position[0] + b.position[0]) / 2,
      (a.position[1] + b.position[1]) / 2,
      (a.position[2] + b.position[2]) / 2,
    ]);
    const twoNodes: MotionRig = {
      ...rig,
      nodes: [rig.nodes[0] ?? a, a, b].map((n, i) => ({ ...n, parent: i === 0 ? -1 : 0 })),
    };
    expect(assignSplatsToNodes(midpoint, twoNodes)[0]).toBe(1);
  });

  it("parks non-finite splats on the root, which never moves", () => {
    const positions = new Float32Array([Number.NaN, 0, 0, 0, Number.POSITIVE_INFINITY, 0, 0, 0, 5]);
    const assignment = assignSplatsToNodes(positions, rig);
    expect(assignment[0]).toBe(UNASSIGNED_NODE);
    expect(assignment[1]).toBe(UNASSIGNED_NODE);
    expect(assignment[2]).not.toBe(UNASSIGNED_NODE);
    const transforms = deform(rig, 5, { strength: 1, bearingDeg: 0 });
    expect(magnitude(nodeDisplacement(rig, transforms, UNASSIGNED_NODE))).toBe(0);
  });

  it("ignores a trailing partial splat rather than reading past the end", () => {
    expect(assignSplatsToNodes(new Float32Array([0, 0, 0, 1, 2]), rig).length).toBe(1);
    expect(assignSplatsToNodes(new Float32Array(0), rig).length).toBe(0);
  });

  it("handles a capture-sized cloud", () => {
    const count = 150_000;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      positions[i * 3] = ((hash32(i, 5) / 0x1_0000_0000) * 2 - 1) * 3;
      positions[i * 3 + 1] = ((hash32(i, 6) / 0x1_0000_0000) * 2 - 1) * 3;
      positions[i * 3 + 2] = (hash32(i, 7) / 0x1_0000_0000) * 7;
    }
    const assignment = assignSplatsToNodes(positions, rig);
    expect(assignment.length).toBe(count);
    const histogram = assignmentHistogram(assignment, rig.nodes.length);
    expect(histogram.reduce((a, b) => a + b, 0)).toBe(count);
    // Every node owns some splats, so no group is silently dead.
    expect([...histogram].every((n) => n > 0)).toBe(true);
  });
});

describe("assignmentHistogram", () => {
  it("counts splats per node and ignores out-of-range indices", () => {
    expect([...assignmentHistogram(new Uint16Array([0, 0, 2, 9]), 3)]).toEqual([2, 0, 1]);
  });
});

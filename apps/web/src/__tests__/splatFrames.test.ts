/**
 * The frame arithmetic, and the checksum gap it exists to close.
 *
 * The gap, from S2's fourth finding: `checksumPositions` is bit-exact and was taken over the
 * capture's **local ENU** positions, but `transformTile` bakes the root transform into the
 * positions the runtime sees. So decoded runtime positions legitimately do not match
 * `canonicalChecksum`, and a naive comparison would refuse every valid tree forever — while
 * dropping the comparison would deform whatever happened to be in the variable.
 *
 * These tests run the real committed fixture through the real bake matrix shape and assert that
 * un-baking recovers the capture's own float32s bit for bit, that a changed model matrix is
 * invisible to the digest, and that a different tree is not.
 */

import { describe, expect, it } from "vitest";

import { checksumPositions } from "@twin/world";

import {
  GEODETIC_ALIGNMENT_MIN,
  geodeticFrameAlignment,
  invertAffine,
  maxAbsDifference,
  resolveBakedPositions,
  snapToGrid,
  SPZ_POSITION_QUANTUM_M,
  transformPositions,
  treeUprightness,
  unbakePositions,
} from "@/cesium/splatFrames";

import {
  bakeFixture as bake,
  bakeMatrix,
  canonicalPositions as canonical,
  fixtureRig as rig,
  multiplyMat4 as multiply,
  rootPlacement,
  rootTransform,
} from "./splatFixture";

describe("invertAffine", () => {
  it("inverts the bake matrix back to the identity", () => {
    const inverse = invertAffine(bakeMatrix);
    expect(inverse).toBeDefined();
    const product = multiply(bakeMatrix, inverse ?? []);
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    product.forEach((value, i) => {
      expect(value).toBeCloseTo(identity[i] ?? 0, 9);
    });
  });

  it("returns undefined rather than NaNs for a singular or non-finite matrix", () => {
    expect(invertAffine(new Array<number>(16).fill(0))).toBeUndefined();
    const nan = [...bakeMatrix];
    nan[5] = Number.NaN;
    expect(invertAffine(nan)).toBeUndefined();
  });
});

describe("snapToGrid", () => {
  it("normalises -0 to +0, which the byte digest can tell apart", () => {
    expect(Object.is(snapToGrid(-1e-9, SPZ_POSITION_QUANTUM_M), 0)).toBe(true);
    expect(Object.is(snapToGrid(-0, SPZ_POSITION_QUANTUM_M), -0)).toBe(false);
  });

  it("lands on exact multiples of the quantum", () => {
    for (const raw of [0.5, -3.14159, 7.6125488281, 1e-5]) {
      const snapped = snapToGrid(raw, SPZ_POSITION_QUANTUM_M);
      expect(Number.isInteger(snapped * 4096)).toBe(true);
      expect(Math.abs(snapped - raw)).toBeLessThanOrEqual(SPZ_POSITION_QUANTUM_M / 2 + 1e-12);
    }
  });
});

describe("the checksum gap", () => {
  const baked = bake(canonical, bakeMatrix);

  it("baked positions do not match the rig's checksum — that is the gap", () => {
    expect(checksumPositions(canonical)).toBe(rig.canonicalChecksum);
    expect(checksumPositions(baked)).not.toBe(rig.canonicalChecksum);
  });

  it("un-baking is numerically tiny against the grid it snaps to", () => {
    const raw = transformPositions(
      baked,
      invertAffine(bakeMatrix) ?? [],
      new Float32Array(baked.length),
    );
    // Measured 2.3e-8 m against a 1.2e-4 m half-step: the snap has four orders of margin.
    expect(maxAbsDifference(raw, canonical)).toBeLessThan(SPZ_POSITION_QUANTUM_M / 100);
  });

  it("un-baking and snapping recovers the capture's own float32s bit for bit", () => {
    const recovered = unbakePositions(baked, invertAffine(bakeMatrix) ?? []);
    for (let i = 0; i < canonical.length; i += 1) {
      expect(Object.is(recovered[i], canonical[i])).toBe(true);
    }
    expect(checksumPositions(recovered)).toBe(rig.canonicalChecksum);
  });

  it("survives a model matrix change — clamp-to-ground re-derives, it does not refuse", () => {
    // `SiteManager.clampToGround` sets `tileset.modelMatrix` after a terrain sample, seconds
    // after load. Every baked position changes; the tree does not. The digest must not notice.
    const raised = [...rootPlacement];
    raised[12] = (raised[12] ?? 0) + 1.5;
    raised[13] = (raised[13] ?? 0) - 2.25;
    raised[14] = (raised[14] ?? 0) + 0.75;
    const clamped = multiply(invertAffine(rootTransform) ?? [], raised);
    const rebaked = bake(canonical, clamped);

    expect(maxAbsDifference(rebaked, baked)).toBeGreaterThan(1);
    const recovered = unbakePositions(rebaked, invertAffine(clamped) ?? []);
    expect(checksumPositions(recovered)).toBe(rig.canonicalChecksum);
  });

  it("does not survive a different tree — a wrong tileset is refused", () => {
    const other = Float32Array.from(canonical);
    other[3] = (other[3] ?? 0) + SPZ_POSITION_QUANTUM_M;
    const recovered = unbakePositions(bake(other, bakeMatrix), invertAffine(bakeMatrix) ?? []);
    expect(checksumPositions(recovered)).not.toBe(rig.canonicalChecksum);
  });

  it("does not survive a splat count change", () => {
    const fewer = canonical.slice(0, canonical.length - 3);
    const recovered = unbakePositions(bake(fewer, bakeMatrix), invertAffine(bakeMatrix) ?? []);
    expect(checksumPositions(recovered)).not.toBe(rig.canonicalChecksum);
  });
});

describe("the frame assertion", () => {
  it("reads 1 for the root transform the engine actually builds", () => {
    expect(geodeticFrameAlignment(rootTransform)).toBeGreaterThanOrEqual(GEODETIC_ALIGNMENT_MIN);
  });

  it("catches a frame whose Z column is not the geodetic normal", () => {
    const tilted = [...rootTransform];
    // Swap the up and north columns: the classic axis mistake, which would sway the tree
    // sideways into the ground.
    for (let r = 0; r < 3; r += 1) {
      const up = tilted[8 + r] ?? 0;
      tilted[8 + r] = tilted[4 + r] ?? 0;
      tilted[4 + r] = up;
    }
    expect(geodeticFrameAlignment(tilted)).toBeLessThan(GEODETIC_ALIGNMENT_MIN);
  });

  it("is not the principal-axis test, which a correct capture fails", () => {
    // S0 measured mygla's principal axis at [0.316, 0.948, 0.026] — nearly horizontal on a
    // correct capture — and the synthetic tree's crown makes its cloud almost isotropic. This
    // pins the reason the frame check is on the transform, not on the points.
    const up = treeUprightness(canonical);
    expect(up.verticalExtentM).toBeGreaterThan(7);
    expect(up.horizontalExtentM).toBeGreaterThan(7);
  });
});

describe("treeUprightness", () => {
  it("recognises the fixture standing along +Z", () => {
    const up = treeUprightness(canonical);
    expect(up.upright).toBe(true);
    expect(up.crownToBase).toBeGreaterThan(5);
  });

  it.each([
    ["x", 0],
    ["y", 1],
  ])("rejects the same tree lying along %s", (_axis, index) => {
    const swapped = new Float32Array(canonical.length);
    for (let i = 0; i < canonical.length; i += 3) {
      swapped[i] = canonical[i] ?? 0;
      swapped[i + 1] = canonical[i + 1] ?? 0;
      swapped[i + 2] = canonical[i + 2] ?? 0;
      const up = swapped[i + 2] ?? 0;
      swapped[i + 2] = swapped[i + index] ?? 0;
      swapped[i + index] = up;
    }
    expect(treeUprightness(swapped).upright).toBe(false);
  });

  it("says nothing confident about too few points", () => {
    expect(treeUprightness(new Float32Array(9)).upright).toBe(false);
  });
});

describe("resolveBakedPositions", () => {
  const baked = bake(canonical, bakeMatrix);
  const assignment = new Uint16Array(canonical.length / 3);
  for (let i = 0; i < assignment.length; i += 1) assignment[i] = i % 4;

  it("copies the engine's own bytes for a node at rest", () => {
    // Not "re-bakes them faithfully": byte-identical. This is what makes wind → 0 return the
    // tree to exactly the pose it was measured in, with no epsilon in the claim.
    const displaced = Float32Array.from(canonical, (v) => v + 5);
    const out = new Float32Array(canonical.length);
    resolveBakedPositions(displaced, baked, assignment, new Uint8Array(4), bakeMatrix, out);
    for (let i = 0; i < baked.length; i += 1) expect(Object.is(out[i], baked[i])).toBe(true);
  });

  it("bakes only the splats whose node moves", () => {
    const displaced = Float32Array.from(canonical);
    for (let i = 0; i < displaced.length; i += 3) displaced[i + 2] = (displaced[i + 2] ?? 0) + 1;
    const moves = Uint8Array.from([0, 1, 0, 0]);
    const out = new Float32Array(canonical.length);
    resolveBakedPositions(displaced, baked, assignment, moves, bakeMatrix, out);

    const wholeTreeMoved = bake(displaced, bakeMatrix);
    for (let i = 0; i < assignment.length; i += 1) {
      const source = assignment[i] === 1 ? wholeTreeMoved : baked;
      for (let c = 0; c < 3; c += 1) {
        expect(Object.is(out[i * 3 + c], source[i * 3 + c])).toBe(true);
      }
    }
  });

  it("refuses to alias its output onto an input", () => {
    const out = new Float32Array(canonical.length);
    expect(() =>
      resolveBakedPositions(out, baked, assignment, new Uint8Array(4), bakeMatrix, out),
    ).toThrow(/alias/);
  });
});

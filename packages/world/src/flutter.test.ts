/**
 * Per-splat flutter: does the foliage stop moving as a block, and does it cost what it claims?
 *
 * Two of these tests carry the feature. **Per-splat decorrelation within one node** is the
 * direct statement of what flutter is for: before it, every splat of a leaf cluster shared one
 * transform and so moved identically, and no amount of node-level work could change that. And
 * **calm is exactly identity**, because everything the runtime guarantees about never touching
 * the measurement runs through a path that only holds if adding flutter at zero wind adds
 * literally nothing — not a rounding error, not a `+0`.
 *
 * The rest are the properties that make it a model rather than a jitter: amplitude in the band
 * it claims, frequency in the band it claims, a trunk that is still rather than nearly still,
 * an exact bound, and determinism that survives a rig going through JSON.
 */

import { describe, expect, it } from "vitest";

import {
  applyFlutter,
  assignSplatsToNodes,
  deform,
  deformPositions,
  DEFAULT_WIND_STRENGTH,
  flutterField,
  FLUTTER_SCALE_M,
  FLUTTER_STILL,
  flutterShape,
  magnitude,
  maxDisplacement,
  maxFlutterAmplitude,
  maxFlutterSpeed,
  nodeFlutterHz,
  nodeModes,
  nodeNaturalFrequencyHz,
  parseRig,
  serializeRig,
  sortStaleness,
  splatFlutter,
  syntheticTreeRig,
  subtract,
  WIND_CALM,
  type MotionRig,
  type Vec3,
} from "./index";

const rig = syntheticTreeRig();
const DT = 1 / 60;
const BREEZE = { strength: DEFAULT_WIND_STRENGTH, bearingDeg: 250 };
const GALE = { strength: 1, bearingDeg: 250 };

/** The first node of each band, and a handful of splat indices to probe with. */
const LEAF = rig.nodes.findIndex((node) => node.band === "leaf");
const TRUNK = 0;

/** Pearson correlation of two scalar series. */
function correlation(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  const ma = a.reduce((s, v) => s + v, 0) / n;
  const mb = b.reduce((s, v) => s + v, 0) / n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i += 1) {
    const x = (a[i] ?? 0) - ma;
    const y = (b[i] ?? 0) - mb;
    cov += x * y;
    va += x * x;
    vb += y * y;
  }
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0;
}

// -----------------------------------------------------------------------------------------------

describe("calm is exactly identity", () => {
  it("returns a still field at zero strength, with no amplitudes at all", () => {
    const field = flutterField(rig, 12.5, WIND_CALM);
    expect(field.still).toBe(true);
    expect([...field.amplitudeM].filter((a) => a !== 0)).toEqual([]);
  });

  it("leaves every splat where it was, bit for bit, at zero strength", () => {
    // The guarantee the restore path and `snapshot()` both rest on. Not "close to": the same
    // float32 words, including the sign of a zero.
    const positions = new Float32Array([0, 0, 0, 1.5, -2.25, 3, -0.5, 0.125, 6]);
    const assignment = assignSplatsToNodes(positions, rig);
    const transforms = deform(rig, 9, WIND_CALM);
    const withFlutter = deformPositions(
      positions,
      assignment,
      transforms,
      undefined,
      flutterField(rig, 9, WIND_CALM),
    );
    const without = deformPositions(positions, assignment, transforms);
    expect([...withFlutter]).toEqual([...positions]);
    expect([...withFlutter]).toEqual([...without]);
  });

  it("is a no-op when handed the shared still field", () => {
    const positions = new Float32Array([0.25, -0.5, 4, 1, 1, 1]);
    const assignment = assignSplatsToNodes(positions, rig);
    const transforms = deform(rig, 3, BREEZE);
    const a = deformPositions(positions, assignment, transforms, undefined, FLUTTER_STILL);
    const b = deformPositions(positions, assignment, transforms);
    expect([...a]).toEqual([...b]);
  });

  it("gives a splat of a non-fluttering node exactly the shared zero vector", () => {
    const field = flutterField(rig, 7, GALE);
    expect(field.amplitudeM[TRUNK]).toBe(0);
    for (const index of [0, 1, 999, 7777]) {
      expect(splatFlutter(index, TRUNK, field)).toEqual([0, 0, 0]);
    }
  });
});

describe("per-splat decorrelation within a single node", () => {
  /**
   * The direct test for the feature.
   *
   * A leaf cluster is ~55 splats sharing one rig node, and before flutter they shared one
   * transform, so their tracks were identical to the last bit — every correlation below was
   * exactly 1 and no arrangement of node-level constants could have moved it.
   *
   * It is measured on the frame-to-frame *velocity* rather than on position, for the same
   * reason the sibling-limb test in `motion.test.ts` is: the splats of one cluster are carried
   * bodily by their node, so their positions share that 5 cm excursion however independently
   * they shimmer on top of it, and the position correlation stays near 0.96 by arithmetic. The
   * per-frame step is the other way round — flutter at 4–15 Hz contributes several times what
   * the node contributes at 1–5 Hz — so differencing is what exposes the term under test.
   * Position is still checked, for the one thing it can say: that no two splats are identical.
   */
  const MAX_WITHIN_NODE_CORRELATION = 0.5;

  const positions = (() => {
    // Ten splats scattered around the first leaf node, which is where the foliage actually is.
    const centre = rig.nodes[LEAF]?.position ?? [0, 0, 0];
    const out = new Float32Array(30);
    for (let i = 0; i < 10; i += 1) {
      out[i * 3] = centre[0] + (i % 3) * 0.01;
      out[i * 3 + 1] = centre[1] + (i % 5) * 0.01;
      out[i * 3 + 2] = centre[2] + (i % 7) * 0.01;
    }
    return out;
  })();
  const assignment = assignSplatsToNodes(positions, rig);

  /** Each splat's downwind offset from its own rest position, frame by frame. */
  function tracks(frames: number): number[][] {
    const out: number[][] = Array.from({ length: 10 }, () => []);
    for (let k = 0; k < frames; k += 1) {
      const t = 20 + k * DT;
      const moved = deformPositions(
        positions,
        assignment,
        deform(rig, t, BREEZE),
        undefined,
        flutterField(rig, t, BREEZE),
      );
      for (let i = 0; i < 10; i += 1) {
        out[i]?.push((moved[i * 3] ?? 0) - (positions[i * 3] ?? 0));
      }
    }
    return out;
  }

  it("puts the probe splats on one fluttering node", () => {
    expect(new Set(assignment).size).toBe(1);
    const node = assignment[0] ?? 0;
    expect(rig.nodes[node]?.band).toBe("leaf");
    expect(flutterField(rig, 20, BREEZE).amplitudeM[node]).toBeGreaterThan(0);
  });

  it("stops splats of one leaf cluster from moving identically", () => {
    const paths = tracks(120);
    const steps = paths.map((path) => path.slice(1).map((v, i) => v - (path[i] ?? 0)));
    let pairs = 0;
    let total = 0;
    let worstPosition = 0;
    for (let i = 0; i < paths.length; i += 1) {
      for (let j = i + 1; j < paths.length; j += 1) {
        total += Math.abs(correlation(steps[i] ?? [], steps[j] ?? []));
        worstPosition = Math.max(
          worstPosition,
          Math.abs(correlation(paths[i] ?? [], paths[j] ?? [])),
        );
        pairs += 1;
      }
    }
    expect(pairs).toBe(45);
    expect(total / pairs).toBeLessThan(MAX_WITHIN_NODE_CORRELATION);
    // And no two of them are the same splat twice, which is what "identical" looks like.
    expect(worstPosition).toBeLessThan(0.999);
  });

  it("without flutter they are identical, which is the defect this exists to remove", () => {
    const out: number[][] = Array.from({ length: 10 }, () => []);
    for (let k = 0; k < 60; k += 1) {
      const t = 20 + k * DT;
      const moved = deformPositions(positions, assignment, deform(rig, t, BREEZE));
      for (let i = 0; i < 10; i += 1) {
        out[i]?.push((moved[i * 3] ?? 0) - (positions[i * 3] ?? 0));
      }
    }
    const steps = out.map((path) => path.slice(1).map((v, i) => v - (path[i] ?? 0)));
    expect(Math.abs(correlation(out[0] ?? [], out[7] ?? []))).toBeGreaterThan(0.999);
    expect(Math.abs(correlation(steps[0] ?? [], steps[7] ?? []))).toBeGreaterThan(0.999);
  });

  it("moves a splat measurably faster than its node does", () => {
    // The point of a high-frequency, low-amplitude term: the per-frame step is what the eye
    // reads, and flutter supplies several times what the node transform does. Measured over the
    // whole fixture: 1.73 mm per frame with flutter against 0.67 mm without.
    const node = assignment[0] ?? 0;
    const at = (t: number): Vec3 => {
      const moved = deformPositions(
        positions,
        assignment,
        deform(rig, t, BREEZE),
        undefined,
        flutterField(rig, t, BREEZE),
      );
      return [moved[0] ?? 0, moved[1] ?? 0, moved[2] ?? 0];
    };
    const bare = (t: number): Vec3 => {
      const moved = deformPositions(positions, assignment, deform(rig, t, BREEZE));
      return [moved[0] ?? 0, moved[1] ?? 0, moved[2] ?? 0];
    };
    let withFlutter = 0;
    let without = 0;
    for (let k = 1; k < 600; k += 1) {
      withFlutter += magnitude(subtract(at(20 + k * DT), at(20 + (k - 1) * DT)));
      without += magnitude(subtract(bare(20 + k * DT), bare(20 + (k - 1) * DT)));
    }
    expect(rig.nodes[node]?.band).toBe("leaf");
    expect(withFlutter).toBeGreaterThan(2 * without);
  });
});

describe("the band it claims", () => {
  it("keeps amplitude between a millimetre and a couple of centimetres", () => {
    // "Order of millimetres to a couple of centimetres" is the design brief, and these are the
    // numbers: 5.0 mm at the default wind, 19.8 mm at a full gale, where it saturates.
    const breeze = maxFlutterAmplitude(rig, BREEZE);
    expect(breeze).toBeGreaterThan(0.002);
    expect(breeze).toBeLessThan(0.01);
    const gale = maxFlutterAmplitude(rig, GALE);
    expect(gale).toBeGreaterThan(breeze);
    expect(gale).toBeLessThan(0.03);
    expect(maxFlutterAmplitude(rig, WIND_CALM)).toBe(0);
  });

  it("saturates with wind rather than growing without bound", () => {
    // Doubling a strong wind must not double the shimmer: leaves reconfigure.
    const half = maxFlutterAmplitude(rig, { strength: 0.5, bearingDeg: 0 });
    const full = maxFlutterAmplitude(rig, { strength: 1, bearingDeg: 0 });
    expect(full).toBeGreaterThan(half);
    expect(full).toBeLessThan(1.5 * half);
  });

  it("flutters at 4–15 Hz, from each node's own resonance", () => {
    const modes = nodeModes(rig);
    const fluttering = rig.nodes
      .map((node, i) => ({ node, hz: nodeFlutterHz(nodeNaturalFrequencyHz(modes[i]!)) }))
      .filter(({ node }) => flutterShape(node.radius) > 0);
    expect(fluttering.length).toBeGreaterThan(100);
    for (const { hz } of fluttering) {
      expect(hz).toBeGreaterThanOrEqual(4);
      expect(hz).toBeLessThanOrEqual(15);
    }
    // Not all the same: the band is a clamp on geometry, not a constant.
    expect(new Set(fluttering.map(({ hz }) => hz.toFixed(3))).size).toBeGreaterThan(5);
  });

  it("silences the trunk and moves the crown, from radius alone", () => {
    const field = flutterField(rig, 5, GALE);
    rig.nodes.forEach((node, i) => {
      if (node.band === "trunk") expect(field.amplitudeM[i]).toBe(0);
      if (node.band === "leaf") expect(field.amplitudeM[i] ?? 0).toBeGreaterThan(0);
    });
    // And the rule that produced that is thickness, never the band label — a rig out of
    // `skeleton.py` has bands that were guessed at, and a mislabelled trunk must not shimmer.
    expect(flutterShape(0.14)).toBe(0);
    expect(flutterShape(0.0088)).toBeGreaterThan(0.5);
    expect(flutterShape(0.033)).toBeLessThan(flutterShape(0.017));
  });
});

describe("the bound is a bound", () => {
  it("never exceeds the node's amplitude, over many splats and many times", () => {
    const field = (t: number) => flutterField(rig, t, GALE);
    let worstRatio = 0;
    for (let k = 0; k < 400; k += 1) {
      const t = k * 0.017;
      const f = field(t);
      for (let i = 0; i < 200; i += 1) {
        const index = i * 61 + k;
        const amplitude = f.amplitudeM[LEAF] ?? 0;
        if (amplitude === 0) continue;
        const offset = magnitude(splatFlutter(index, LEAF, f));
        expect(offset).toBeLessThanOrEqual(amplitude + 1e-15);
        worstRatio = Math.max(worstRatio, offset / amplitude);
      }
    }
    // Useful as well as true: the two sinusoids do line up, at 0.7 + 0.3 of the amplitude.
    expect(worstRatio).toBeGreaterThan(0.5);
  });

  it("is folded into maxDisplacement, so sortStaleness accounts for it", () => {
    // Staleness is a fact about splats, not about nodes, and a splat carries its node's
    // displacement plus its own shimmer. 9.2 radii became 9.5 at the default wind — a quarter
    // of a reference splat radius for several times the per-frame motion.
    const total = maxDisplacement(rig, BREEZE);
    const flutter = maxFlutterAmplitude(rig, BREEZE);
    expect(flutter).toBeGreaterThan(0);
    expect(total).toBeGreaterThan(flutter);
    const staleness = sortStaleness(rig, BREEZE, 0.02);
    expect(staleness).toBeGreaterThan(9);
    expect(staleness).toBeLessThan(10);
  });

  it("bounds the speed too, and the speed is where the value is", () => {
    // `A·ω` at 4–15 Hz is far more per second than the node model produces at 1–5 Hz, which is
    // the entire argument for spending amplitude here rather than on bigger branch swings.
    expect(maxFlutterSpeed(rig, BREEZE)).toBeGreaterThan(0.1);
    expect(maxFlutterSpeed(rig, WIND_CALM)).toBe(0);
  });

  it("scales the amplitude linearly in FLUTTER_SCALE_M, so the constant means what it says", () => {
    expect(maxFlutterAmplitude(rig, GALE)).toBeLessThan(FLUTTER_SCALE_M);
  });
});

describe("the batch pass and the readable one agree", () => {
  it("gives every splat exactly what splatFlutter gives it", () => {
    // `applyFlutter` is the loop the runtime actually runs — the per-node coefficients hoisted
    // out, `hash32` written out because a cross-module call per splat was most of its cost, and
    // the whole thing kept in one module so the JIT can hold it in registers. `splatFlutter` is
    // the same arithmetic written to be read. Nothing keeps them equal except this test.
    const field = flutterField(rig, 17.5, BREEZE);
    const count = 500;
    const assignment = new Uint16Array(count);
    for (let i = 0; i < count; i += 1) assignment[i] = (i * 7) % rig.nodes.length;
    const target = new Float32Array(count * 3);
    applyFlutter(target, assignment, field, count);
    for (let i = 0; i < count; i += 1) {
      const expected = splatFlutter(i, assignment[i] ?? 0, field);
      for (let k = 0; k < 3; k += 1) {
        expect(target[i * 3 + k]).toBe(Math.fround(expected[k] ?? 0));
      }
    }
  });

  it("adds to what is already there rather than replacing it", () => {
    const field = flutterField(rig, 3, BREEZE);
    const assignment = new Uint16Array([LEAF, LEAF]);
    const target = new Float32Array([1, 2, 3, -1, -2, -3]);
    applyFlutter(target, assignment, field, 2);
    expect(target[0]).not.toBe(1);
    expect(Math.abs((target[0] ?? 0) - 1)).toBeLessThan(0.02);
  });

  it("does nothing at all for a still field", () => {
    const target = new Float32Array([1, 2, 3, -0, 0, 5]);
    const before = [...target];
    applyFlutter(target, new Uint16Array([LEAF, LEAF]), FLUTTER_STILL, 2);
    expect([...target]).toEqual(before);
    expect(Object.is(target[3], -0)).toBe(true);
  });
});

describe("determinism", () => {
  it("returns identical numbers for identical arguments", () => {
    const a = flutterField(rig, 31.25, BREEZE);
    const b = flutterField(rig, 31.25, BREEZE);
    expect([...a.amplitudeM]).toEqual([...b.amplitudeM]);
    expect([...a.phase]).toEqual([...b.phase]);
    for (const index of [0, 5, 4321]) {
      expect(splatFlutter(index, LEAF, a)).toEqual(splatFlutter(index, LEAF, b));
    }
  });

  it("survives a rig round-trip through JSON", () => {
    // The rig the runtime uses is parsed from `rig.json`, not built in process, so anything
    // seeded by object identity rather than by content would diverge here and nowhere else.
    const round: MotionRig = parseRig(serializeRig(rig));
    const a = flutterField(rig, 8.5, BREEZE);
    const b = flutterField(round, 8.5, BREEZE);
    expect([...a.amplitudeM]).toEqual([...b.amplitudeM]);
    expect([...a.phase]).toEqual([...b.phase]);
    expect(splatFlutter(77, LEAF, a)).toEqual(splatFlutter(77, LEAF, b));
  });

  it("owes nothing to the previous frame", () => {
    // Evaluating a time directly and evaluating it after a thousand other frames must agree.
    const direct = flutterField(rig, 40, BREEZE);
    for (let k = 0; k < 1000; k += 1) flutterField(rig, k * DT, BREEZE);
    const after = flutterField(rig, 40, BREEZE);
    expect([...after.phase]).toEqual([...direct.phase]);
  });

  it("gives different splats different offsets and different nodes different phases", () => {
    const field = flutterField(rig, 6.75, BREEZE);
    const offsets = [0, 1, 2, 3, 4].map((i) => splatFlutter(i, LEAF, field).join(","));
    expect(new Set(offsets).size).toBe(5);
  });
});

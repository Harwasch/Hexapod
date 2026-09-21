/**
 * Does the tree actually move?
 *
 * Every other test in this package proves *safety*: bounded, monotone, continuous, deterministic,
 * anchored, canonical-bytes-immutable. All of them passed against a model whose highest tip moved
 * 0.15 mm per frame at the default wind — and several of them passed *because* the motion was
 * tiny. A bounds test is happiest when nothing moves.
 *
 * These are the tests that would have caught that. They assert the four things a person means by
 * "it looks like a tree": that the motion is large enough to see, that its energy sits where a
 * tree's does, that neighbouring limbs are not marching in step, and that a node responds to the
 * wind at its own frequency rather than at the wind's.
 *
 * Each threshold is stated with the assumption it rests on, in the comment above it, so it cannot
 * decay into a number nobody can defend.
 */

import { describe, expect, it } from "vitest";

import {
  deform,
  DEFAULT_WIND_STRENGTH,
  dot,
  magnitude,
  nodeDisplacement,
  nodeModes,
  nodeNaturalFrequencyHz,
  responseGain,
  subtract,
  syntheticTreeRig,
  TURBULENCE_MODES,
  type MotionRig,
  type Vec3,
} from "./index";

const rig = syntheticTreeRig();
const DT = 1 / 60;

/** The highest node in the rig: the tip a person watching the canopy edge is actually watching. */
const TIP = (() => {
  let best = 0;
  rig.nodes.forEach((node, i) => {
    if (node.position[2] > (rig.nodes[best]?.position[2] ?? -Infinity)) best = i;
  });
  return best;
})();

/** Displacement of one node, sampled at 60 fps from `t = 0`. */
function track(node: number, frames: number, bearingDeg: number, strength: number): Vec3[] {
  const path: Vec3[] = [];
  for (let k = 0; k < frames; k += 1) {
    path.push(nodeDisplacement(rig, deform(rig, k * DT, { strength, bearingDeg }), node));
  }
  return path;
}

/** Mean and largest frame-to-frame step of a track, metres. */
function stepStats(path: readonly Vec3[]): { mean: number; max: number; peak: number } {
  let sum = 0;
  let max = 0;
  let peak = 0;
  for (let i = 1; i < path.length; i += 1) {
    const a = path[i - 1];
    const b = path[i];
    if (a === undefined || b === undefined) continue;
    const step = magnitude(subtract(b, a));
    sum += step;
    if (step > max) max = step;
    const reach = magnitude(b);
    if (reach > peak) peak = reach;
  }
  return { mean: sum / Math.max(1, path.length - 1), max, peak };
}

/**
 * Power of a real signal at one frequency, by direct evaluation of the DFT sum.
 *
 * Deliberately the naive `O(n)` form rather than an FFT: the signals here are a few thousand
 * samples and a handful of frequencies, and a transform nobody can read is a poor thing to put
 * in a test that exists to be argued with.
 */
function powerAt(signal: readonly number[], hz: number): number {
  const n = signal.length;
  const mean = signal.reduce((a, b) => a + b, 0) / n;
  let re = 0;
  let im = 0;
  for (let i = 0; i < n; i += 1) {
    const phase = 2 * Math.PI * hz * i * DT;
    const value = (signal[i] ?? 0) - mean;
    re += value * Math.cos(phase);
    im -= value * Math.sin(phase);
  }
  return (re * re + im * im) / (n * n);
}

/** Fraction of a signal's alternating power that falls between `lo` and `hi` hertz. */
function bandFraction(signal: readonly number[], lo: number, hi: number, nyquist = 12): number {
  const step = 0.05;
  let band = 0;
  let total = 0;
  for (let hz = step; hz <= nyquist; hz += step) {
    const p = powerAt(signal, hz);
    total += p;
    if (hz >= lo && hz <= hi) band += p;
  }
  return total > 0 ? band / total : 0;
}

/** Pearson correlation of two vector tracks, each detrended by its own mean. */
function correlation(a: readonly Vec3[], b: readonly Vec3[]): number {
  const n = Math.min(a.length, b.length);
  const meanOf = (path: readonly Vec3[]): Vec3 => [
    path.reduce((s, v) => s + v[0], 0) / n,
    path.reduce((s, v) => s + v[1], 0) / n,
    path.reduce((s, v) => s + v[2], 0) / n,
  ];
  const ma = meanOf(a);
  const mb = meanOf(b);
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i += 1) {
    const x = subtract(a[i] ?? ma, ma);
    const y = subtract(b[i] ?? mb, mb);
    cov += dot(x, y);
    va += dot(x, x);
    vb += dot(y, y);
  }
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0;
}

// -----------------------------------------------------------------------------------------------

describe("visibility at the default wind", () => {
  /**
   * The viewing assumption, written out so the millimetre figures below are derived rather than
   * chosen: **a 6 m tree that fills 800 px of the viewport is 133 px per metre.** That is a
   * person who has flown close enough to see individual branches, which is the condition the
   * feature is for.
   */
  const TREE_HEIGHT_M = 6;
  const TREE_PX = 800;
  const PX_PER_M = TREE_PX / TREE_HEIGHT_M;

  /**
   * 0.12 px per frame is about where a high-contrast edge stops reading as still on a 60 Hz
   * display. The model it replaced managed **0.02 px** — a quarter of a pixel per second — which
   * is why the tree bent once and then appeared frozen. 0.12 px is 0.9 mm per frame here.
   */
  const VISIBLE_PX_PER_FRAME = 0.12;
  const VISIBLE_M_PER_FRAME = VISIBLE_PX_PER_FRAME / PX_PER_M;

  /** A gust's peak has to be several times the floor, or the motion is a shimmer and not a sway. */
  const GUST_PX_PER_FRAME = 0.3;

  it("moves the highest tip by at least 0.12 px per frame on a 6 m tree filling 800 px", () => {
    // Checked at several bearings: the wind's compass direction must not decide whether the
    // feature is visible at all.
    for (const bearingDeg of [0, 37, 90, 143, 250, 315]) {
      const stats = stepStats(track(TIP, 60 * 60, bearingDeg, DEFAULT_WIND_STRENGTH));
      expect(stats.mean * PX_PER_M).toBeGreaterThan(VISIBLE_PX_PER_FRAME);
      expect(stats.max * PX_PER_M).toBeGreaterThan(GUST_PX_PER_FRAME);
      expect(stats.mean).toBeGreaterThan(VISIBLE_M_PER_FRAME);
    }
  });

  it("holds its amplitude near where it was: the change is speed, not reach", () => {
    // The constraint that makes this safe. Draw-order staleness is driven by displacement, not
    // by how fast that displacement changes, and `DEFAULT_WIND_STRENGTH` was derived from the
    // staleness bound. So the tip's reach must stay in the band a light breeze gives a real
    // branch — 5 to 20 cm — which is also where the quasi-static model sat.
    const stats = stepStats(track(TIP, 60 * 60, 250, DEFAULT_WIND_STRENGTH));
    expect(stats.peak).toBeGreaterThan(0.05);
    expect(stats.peak).toBeLessThan(0.2);
    // And the resulting peak speed is a branch tip's, not a flag's: 0.1–0.5 m/s.
    expect(stats.max / DT).toBeGreaterThan(0.1);
    expect(stats.max / DT).toBeLessThan(0.5);
  });

  it("still stops dead at zero strength", () => {
    const stats = stepStats(track(TIP, 600, 250, 0));
    expect(stats.mean).toBe(0);
    expect(stats.max).toBe(0);
    expect(stats.peak).toBe(0);
  });
});

describe("frequency content", () => {
  /**
   * A 6 m tree's fundamental is measured at 0.5–1.5 Hz, its branches at 1–3 Hz. The old model
   * had essentially no energy above 1 Hz at all: its only time-varying term was a gust envelope
   * at 0.22 noise-units per second, so a whole cycle took 5–15 s and the tree read as a
   * photograph of a bent tree.
   */
  const SWAY_LO_HZ = 0.5;
  const SWAY_HI_HZ = 3;

  const downwindTrack = track(TIP, 60 * 60, 250, DEFAULT_WIND_STRENGTH).map((v) => v[0]);

  it("puts most of the tip's alternating energy in 0.5–3 Hz", () => {
    expect(bandFraction(downwindTrack, SWAY_LO_HZ, SWAY_HI_HZ)).toBeGreaterThan(0.4);
  });

  it("does not simply sit at one frequency: it has a spectrum", () => {
    // Two well-separated bands both carry real power, which is what distinguishes a tree from a
    // metronome. The sway band dominates — 57 % of the tip's alternating power sits in
    // 0.5–1.5 Hz — while 1.5–3.5 Hz still carries a few per cent, which is the branch tremor
    // riding on the trunk's swing. Measured, with margin: 0.3 and 0.02 are asserted.
    expect(bandFraction(downwindTrack, 0.5, 1.5)).toBeGreaterThan(0.3);
    expect(bandFraction(downwindTrack, 1.5, 3.5)).toBeGreaterThan(0.02);
    // And the slow gust envelope, which is all the old model had, is now the minority partner.
    expect(bandFraction(downwindTrack, 0, 0.4)).toBeLessThan(0.45);
    expect(powerAt(downwindTrack, 1.1)).toBeGreaterThan(powerAt(downwindTrack, 0.2));
  });

  it("crosses its own mean often enough to read as a sway, not a drift", () => {
    // A zero-crossing count is the crudest possible frequency estimate and needs no transform:
    // over 60 s a 1 Hz sway crosses its mean about 120 times. The old model crossed 6 times in
    // 10 s, and those crossings were gust envelope, not sway.
    const mean = downwindTrack.reduce((a, b) => a + b, 0) / downwindTrack.length;
    let crossings = 0;
    for (let i = 1; i < downwindTrack.length; i += 1) {
      const a = (downwindTrack[i - 1] ?? 0) - mean;
      const b = (downwindTrack[i] ?? 0) - mean;
      if (a * b < 0) crossings += 1;
    }
    const hz = crossings / 2 / (downwindTrack.length * DT);
    expect(hz).toBeGreaterThan(SWAY_LO_HZ);
    expect(hz).toBeLessThan(SWAY_HI_HZ);
  });
});

describe("decorrelation", () => {
  /**
   * The direct test for "not one shared bending plane" — the defect that no amount of tuning
   * could have fixed, because every node rotated about a single axis computed once for the whole
   * rig and differed from its neighbours only in amplitude.
   *
   * Sibling *tips* legitimately correlate: they hang off one trunk and the trunk's own sway is
   * common to all of them, exactly as in a real tree. What must not correlate is each limb's own
   * motion, so each tip is measured **relative to the trunk joint it hangs from**. Under a single
   * shared axis those relative tracks are collinear and in phase, and this number is ≈ 1.
   */
  const MAX_SIBLING_CORRELATION = 0.75;

  /** The tip's track minus its limb root's: what the limb itself is doing. */
  function limbTrack(tip: number, limbRoot: number, frames: number): Vec3[] {
    const a = track(tip, frames, 250, 1);
    const b = track(limbRoot, frames, 250, 1);
    return a.map((v, i) => subtract(v, b[i] ?? v));
  }

  it("keeps sibling limbs out of step with each other", () => {
    // leaf-0, leaf-1 and leaf-2 are the three leaves of the topmost whorl; node 5 is the trunk
    // joint all three hang from.
    const trunkJoint = 5;
    const leaves = [8, 11, 14];
    expect(leaves.every((i) => rig.nodes[i]?.band === "leaf")).toBe(true);
    const tracks = leaves.map((i) => limbTrack(i, trunkJoint, 60 * 60));
    let checked = 0;
    for (let i = 0; i < tracks.length; i += 1) {
      for (let j = i + 1; j < tracks.length; j += 1) {
        const a = tracks[i];
        const b = tracks[j];
        if (a === undefined || b === undefined) continue;
        expect(Math.abs(correlation(a, b))).toBeLessThan(MAX_SIBLING_CORRELATION);
        checked += 1;
      }
    }
    expect(checked).toBe(3);
  });

  it("gives sibling limbs different bending planes, not just different phases", () => {
    // Two limbs could be uncorrelated in time and still sweep the same plane. The planes
    // themselves must differ, which is what the per-node azimuth and twist are for.
    const modes = nodeModes(rig);
    const azimuths = [8, 11, 14].map((i) => modes[i]?.azimuthRad ?? 0);
    const spread = Math.max(...azimuths) - Math.min(...azimuths);
    expect(spread).toBeGreaterThan(0.2);
    const twists = [8, 11, 14].map((i) => modes[i]?.twist ?? 0);
    expect(Math.max(...twists) - Math.min(...twists)).toBeGreaterThan(0.05);
  });

  it("is a property of the rig's ids, so it survives a rig round-trip", () => {
    // The decorrelation comes from hashed node ids, not from a random seed drawn at load: two
    // copies of the same rig must move identically or the snapshot guarantees are worthless.
    const modes = nodeModes(rig);
    const again = nodeModes(rig);
    expect(again.map((m) => m.azimuthRad)).toEqual(modes.map((m) => m.azimuthRad));
  });
});

describe("resonance", () => {
  it("peaks at a node's own natural frequency and falls away either side", () => {
    const zeta = 0.08;
    const natural = 2 * Math.PI * 1.2;
    const atResonance = responseGain(natural, natural, zeta);
    // The closed form: `H(ω₀) = 1 / (2ζ)`, exactly.
    expect(atResonance).toBeCloseTo(1 / (2 * zeta), 12);
    expect(atResonance).toBeGreaterThan(4 * responseGain(natural / 4, natural, zeta));
    expect(atResonance).toBeGreaterThan(4 * responseGain(natural * 4, natural, zeta));
    // Far below resonance a node simply follows the wind; far above it ignores it.
    expect(responseGain(natural / 1000, natural, zeta)).toBeCloseTo(1, 4);
    expect(responseGain(natural * 1000, natural, zeta)).toBeLessThan(1e-5);
  });

  it("derives that frequency from geometry, not from the band label", () => {
    const modes = nodeModes(rig);
    const hzOf = (i: number) => nodeNaturalFrequencyHz(modes[i] ?? modes[0]!);
    // The trunk base carries the whole tree and rings slowest; a twig rings fastest. Both are
    // consequences of `radius / length²`, and nothing consults `band`.
    expect(hzOf(0)).toBeLessThan(hzOf(5));
    expect(hzOf(5)).toBeLessThan(hzOf(8));
    expect(hzOf(0)).toBeGreaterThan(0.5);
    expect(hzOf(0)).toBeLessThan(1.5);
    // Two nodes with the same band and different geometry get different frequencies, which is
    // what has to be true for a rig that came out of `skeleton.py` with inferred bands.
    expect(hzOf(1)).not.toBeCloseTo(hzOf(5), 3);
    expect(rig.nodes[1]?.band).toBe(rig.nodes[5]?.band);
  });

  it("makes a node respond at its own frequency, not at the wind's", () => {
    // Two rigs identical but for one node's radius, so they differ only in natural frequency,
    // driven by the same forcing. Each tip's motion should concentrate near its own node's
    // frequency — the definition of resonant selection, and the thing the old model could not
    // express because nothing in it had a frequency at all.
    const chain = (radius: number): MotionRig => ({
      nodes: [
        { id: "root", parent: -1, position: [0, 0, 0], radius: 0.3, stiffness: 1, band: "trunk" },
        {
          id: "joint",
          parent: 0,
          position: [0, 0, 2],
          radius,
          stiffness: 1,
          band: "branch",
          maxAngleRad: 1.2,
        },
        {
          id: "tip",
          parent: 1,
          position: [0, 0, 4],
          radius,
          stiffness: 1,
          band: "branch",
          maxAngleRad: 1.2,
        },
      ],
      canonicalChecksum: "fnv1a32:0:00000000",
      units: "meters",
      sourceNote: "resonance probe",
    });

    const tipTrack = (rigUnderTest: MotionRig): number[] => {
      const out: number[] = [];
      for (let k = 0; k < 60 * 60; k += 1) {
        const transforms = deform(rigUnderTest, k * DT, { strength: 0.2, bearingDeg: 0 });
        out.push(nodeDisplacement(rigUnderTest, transforms, 2)[1]);
      }
      return out;
    };

    const slow = chain(0.0205);
    const fast = chain(0.0764);
    const slowHz = nodeNaturalFrequencyHz(nodeModes(slow)[1]!);
    const fastHz = nodeNaturalFrequencyHz(nodeModes(fast)[1]!);
    expect(slowHz).toBeGreaterThan(1);
    expect(slowHz).toBeLessThan(1.5);
    expect(fastHz).toBeGreaterThan(3.5);

    const slowSignal = tipTrack(slow);
    const fastSignal = tipTrack(fast);
    // Each responds more, relative to itself, in the band containing its own frequency.
    const slowInSlowBand = bandFraction(slowSignal, 0.8, 1.6);
    const slowInFastBand = bandFraction(slowSignal, 3.2, 4.8);
    const fastInSlowBand = bandFraction(fastSignal, 0.8, 1.6);
    const fastInFastBand = bandFraction(fastSignal, 3.2, 4.8);
    expect(slowInSlowBand).toBeGreaterThan(slowInFastBand);
    expect(fastInFastBand).toBeGreaterThan(fastInSlowBand);
    // And the comparison across the pair, which is the one that rules out a shared envelope.
    expect(slowInSlowBand).toBeGreaterThan(fastInSlowBand);
    expect(fastInFastBand).toBeGreaterThan(slowInFastBand);
  });

  it("forces the tree across the band a tree actually feels", () => {
    const hz = TURBULENCE_MODES.map((mode) => mode.hz);
    expect(Math.min(...hz)).toBeGreaterThan(0.2);
    expect(Math.max(...hz)).toBeLessThan(8);
    // The amplitudes are a share of a whole, which is what makes every bound in `deform` exact.
    const total = TURBULENCE_MODES.reduce((sum, mode) => sum + mode.amplitude, 0);
    expect(total).toBeCloseTo(1, 12);
    // Slow components dominate, as a wind spectrum's do, without the fast ones vanishing.
    const sorted = [...TURBULENCE_MODES].sort((a, b) => a.hz - b.hz);
    expect(sorted[0]!.amplitude).toBeGreaterThan(sorted.at(-1)!.amplitude);
    expect(sorted.at(-1)!.amplitude).toBeGreaterThan(0.02);
  });
});

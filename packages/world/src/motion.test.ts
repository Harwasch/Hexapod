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
  gustDelaySeconds,
  syntheticTreeRig,
  TURBULENCE_MODES,
  windAt,
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
    // metronome. The sway band dominates — 65 % of the tip's alternating power sits in
    // 0.5–1.5 Hz, where the 1.33 Hz trunk rings — while 3.5–8 Hz carries 11 % as branch tremor
    // riding on that swing. That upper band is what the fixture's proportions bought: with
    // branches 24 cm thick it had nothing there to excite, because they rang at 7–10 Hz where
    // the forcing stopped at 6.
    expect(bandFraction(downwindTrack, 0.5, 1.5)).toBeGreaterThan(0.3);
    expect(bandFraction(downwindTrack, 3.5, 8)).toBeGreaterThan(0.06);
    // And the slow gust envelope, which is all the old model had, is now the minority partner.
    expect(bandFraction(downwindTrack, 0, 0.4)).toBeLessThan(0.45);
  });

  it("rings at the forcing lines, not at a smear", () => {
    // The forcing is nine fixed sinusoids, so a node that resonates shows it as a *line*. The
    // 4.50 Hz component lands on the crown's primary limbs and is amplified into a peak four
    // orders of magnitude above the all-but-empty spectrum either side of it. That is the
    // direct evidence that the crown has dynamics of its own rather than following the trunk:
    // on the fixture this replaced there was no such line, because nothing in the crown had a
    // natural frequency inside the forcing band.
    const line = powerAt(downwindTrack, 4.5);
    expect(line).toBeGreaterThan(1000 * powerAt(downwindTrack, 3.5));
    expect(line).toBeGreaterThan(1000 * powerAt(downwindTrack, 5.25));
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
   *
   * On positions the threshold is now 0.9 where it was 0.75, and that is honest rather than
   * convenient: the pairwise numbers are 0.86, 0.68, 0.63. What is left in common is the gust
   * field's own coherence — two limbs 1.5 m apart in a crown genuinely do see nearly the same
   * gust at nearly the same instant, and after `travellingGust` that is a physical statement
   * rather than an artefact. The per-node lag it replaced was a hash, which decorrelated
   * neighbours *more* than physics warrants and made this number look better than the model
   * deserved. The velocity test below is the one that discriminates.
   */
  const MAX_SIBLING_CORRELATION = 0.9;

  /**
   * The sharper measure, and the one that carries the claim.
   *
   * Two limbs share a wind field, so their *positions* share its slow envelope however
   * independent their dynamics are — that common mode survives subtracting each track's mean
   * and dominates the correlation of the positions. Differencing the tracks frame to frame is a
   * one-line high-pass that removes it and leaves what the limbs are actually doing. Under a
   * single shared bending axis this number is ≈ 1 too, so nothing is given away by using it.
   */
  const MAX_SIBLING_VELOCITY_CORRELATION = 0.7;

  /** Frame-to-frame differences of a track: its velocity, up to the constant `DT`. */
  function velocity(path: readonly Vec3[]): Vec3[] {
    return path.slice(1).map((v, i) => subtract(v, path[i] ?? v));
  }

  /** The tip's track minus its limb root's: what the limb itself is doing. */
  function limbTrack(tip: number, limbRoot: number, frames: number): Vec3[] {
    const a = track(tip, frames, 250, 1);
    const b = track(limbRoot, frames, 250, 1);
    return a.map((v, i) => subtract(v, b[i] ?? v));
  }

  it("keeps sibling limbs out of step with each other", () => {
    // The first leaf of each of the three primary limbs of the topmost whorl; node 9 is the
    // trunk joint all three hang from. Each primary limb is 17 nodes, so they are 17 apart.
    const trunkJoint = 9;
    const leaves = [14, 31, 48];
    expect(leaves.every((i) => rig.nodes[i]?.band === "leaf")).toBe(true);
    const tracks = leaves.map((i) => limbTrack(i, trunkJoint, 60 * 60));
    let checked = 0;
    for (let i = 0; i < tracks.length; i += 1) {
      for (let j = i + 1; j < tracks.length; j += 1) {
        const a = tracks[i];
        const b = tracks[j];
        if (a === undefined || b === undefined) continue;
        expect(Math.abs(correlation(a, b))).toBeLessThan(MAX_SIBLING_CORRELATION);
        // The one that matters: with the shared gust envelope differenced away, sibling limbs
        // agree 0.24–0.59 of the time. Measured; 0.7 is asserted.
        expect(Math.abs(correlation(velocity(a), velocity(b)))).toBeLessThan(
          MAX_SIBLING_VELOCITY_CORRELATION,
        );
        checked += 1;
      }
    }
    expect(checked).toBe(3);
  });

  it("gives sibling limbs different bending planes, not just different phases", () => {
    // Two limbs could be uncorrelated in time and still sweep the same plane. The planes
    // themselves must differ, which is what the per-node azimuth and twist are for.
    const modes = nodeModes(rig);
    const azimuths = [14, 31, 48].map((i) => modes[i]?.azimuthRad ?? 0);
    const spread = Math.max(...azimuths) - Math.min(...azimuths);
    expect(spread).toBeGreaterThan(0.2);
    const twists = [14, 31, 48].map((i) => modes[i]?.twist ?? 0);
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

describe("a gust crosses the crown", () => {
  /**
   * The property the wind field exists for: a gust **arrives**, rather than switching on
   * everywhere at once.
   *
   * Two things are true here and they are not the same size, which is worth writing down rather
   * than quietly asserting the flattering one.
   *
   * The **field** crosses the crown unambiguously. The gust magnitude at the most upwind leaf
   * cluster leads the most downwind one by well over a second, and at any instant the wind
   * across the crown varies by a third between its windiest and calmest limb.
   *
   * The **motion** lags by far less — about 0.2 s where the field's arrival differs by 0.9 s.
   * That is not a bug and no constant will fix it. A limb's displacement is mostly its
   * ancestors' displacement, and its ancestors are nearer the trunk where the delays are
   * smaller; and what is left is a resonant response at 1–5 Hz, where a delay of most of a
   * second is most of a cycle and reads as a small one. The gust field buys decorrelation and a
   * visible unevenness across the crown, and it buys a modest, real lag. It does not make a
   * wave of motion roll across the tree, and it would take a convection speed far slower than
   * any real gust to make it do so on a crown this small.
   */
  const BEARING = 250;

  /** The crown limbs furthest upwind and furthest downwind, and their separation in metres. */
  const pair = (() => {
    const bearing = BEARING * (Math.PI / 180);
    const de = Math.sin(bearing);
    const dn = Math.cos(bearing);
    let first = 0;
    let last = 0;
    let least = Infinity;
    let most = -Infinity;
    rig.nodes.forEach((node, i) => {
      if (node.band !== "leaf") return;
      const along = node.position[0] * de + node.position[1] * dn;
      if (along < least) {
        least = along;
        first = i;
      }
      if (along > most) {
        most = along;
        last = i;
      }
    });
    return { first, last, separationM: most - least };
  })();

  /** Along-wind gust magnitude at a node, sampled at 60 fps and mean-removed. */
  function gustAt(node: number, frames: number): number[] {
    const out: number[] = [];
    const position = rig.nodes[node]?.position ?? [0, 0, 0];
    for (let k = 0; k < frames; k += 1) {
      const g = windAt(1, BEARING, k * DT, position);
      out.push(Math.hypot(g[0], g[1]));
    }
    const mean = out.reduce((a, b) => a + b, 0) / out.length;
    return out.map((v) => v - mean);
  }

  /** The lag in seconds at which `later` best matches `earlier`. Positive means later lags. */
  function bestLagSeconds(earlier: readonly number[], later: readonly number[]): number {
    const maxLag = 150;
    let best = 0;
    let bestScore = -Infinity;
    for (let lag = -maxLag; lag <= maxLag; lag += 1) {
      let score = 0;
      for (let i = maxLag; i < earlier.length - maxLag; i += 1) {
        score += (earlier[i] ?? 0) * (later[i + lag] ?? 0);
      }
      if (score > bestScore) {
        bestScore = score;
        best = lag;
      }
    }
    return best * DT;
  }

  it("has two leaf clusters a few metres apart along the wind", () => {
    expect(rig.nodes[pair.first]?.band).toBe("leaf");
    expect(rig.nodes[pair.last]?.band).toBe("leaf");
    expect(pair.separationM).toBeGreaterThan(2);
    // And the field says so: the two are most of a second apart in gust arrival.
    const upwind = gustDelaySeconds(BEARING, rig.nodes[pair.first]?.position ?? [0, 0, 0]);
    const downwind = gustDelaySeconds(BEARING, rig.nodes[pair.last]?.position ?? [0, 0, 0]);
    expect(downwind - upwind).toBeGreaterThan(0.5);
  });

  it("delivers the gust to the upwind cluster first", () => {
    expect(bestLagSeconds(gustAt(pair.first, 3600), gustAt(pair.last, 3600))).toBeGreaterThan(0.5);
  });

  it("does not blow equally hard on every limb at any instant", () => {
    // Under one wind vector for the whole tree this ratio was exactly 1 at every instant.
    for (const t of [3, 17.5, 40, 91.25]) {
      const magnitudes = rig.nodes
        .filter((node) => node.band === "leaf")
        .map((node) => {
          const g = windAt(1, BEARING, t, node.position);
          return Math.hypot(g[0], g[1]);
        });
      expect(Math.max(...magnitudes) / Math.min(...magnitudes)).toBeGreaterThan(1.1);
    }
  });

  it("moves the downwind cluster after the upwind one, if only by a little", () => {
    // Measured on each limb's own bend — its displacement minus its parent's — because a tip's
    // absolute track is mostly the trunk's, and the trunk stands on the wind's axis where the
    // delay is zero by construction.
    const frames = 60 * 60;
    const own = (node: number): number[] => {
      const parent = rig.nodes[node]?.parent ?? 0;
      const tip = track(node, frames, BEARING, 1);
      const root = track(parent, frames, BEARING, 1);
      return tip.map((v, i) => v[0] - (root[i]?.[0] ?? 0));
    };
    const lag = bestLagSeconds(own(pair.first), own(pair.last));
    expect(lag).toBeGreaterThan(0.05);
    expect(lag).toBeLessThan(1);
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
    expect(hzOf(5)).toBeLessThan(hzOf(14));
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

    // `nodeNaturalHz` is `FREQ_SCALE_HZ · radius / length²` with a 2 m cantilever, so at the
    // recalibrated scale of 450 Hz these two radii are 1.20 Hz and 4.19 Hz.
    const slow = chain(0.0107);
    const fast = chain(0.0373);
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
    // The top of the band went from 6 Hz to 10 Hz with the fixture's proportions: a 0.5 m twig
    // with a physical 9 mm radius rings near 20 Hz, and a forcing that stopped at 6 Hz had
    // nothing to offer it. It is still short of a real twig's band; per-splat flutter covers
    // the rest.
    expect(Math.max(...hz)).toBeLessThan(12);
    // The amplitudes are a share of a whole, which is what makes every bound in `deform` exact.
    const total = TURBULENCE_MODES.reduce((sum, mode) => sum + mode.amplitude, 0);
    expect(total).toBeCloseTo(1, 12);
    // Slow components dominate, as a wind spectrum's do, without the fast ones vanishing.
    const sorted = [...TURBULENCE_MODES].sort((a, b) => a.hz - b.hz);
    expect(sorted[0]!.amplitude).toBeGreaterThan(sorted.at(-1)!.amplitude);
    expect(sorted.at(-1)!.amplitude).toBeGreaterThan(0.02);
  });
});

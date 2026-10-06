/**
 * The limbs skin's wind (`limbWind.ts`, docs/SCENE_OBJECTS.md §9 "Limbs"), on the Minnetonka
 * tree's extracted rig (packages/world/fixtures, CC BY 4.0) with its sidecar derived as
 * living.test.ts derives it.
 *
 * What must hold: every limb sways exactly as the rig's oscillator of the same key (the joints'
 * local rotations are the limb's bend times their gains, to rounding); a handle turns about its
 * pivot, which stays; a child limb's pivot moves as its parent limb carries it; each limb rings
 * at its own frequency; calm is rest, by value; the flutter is the rig's band, unit RMS per
 * component at share 1 before its amplitude, carried downwind; and the per-limb record the
 * Python converter writes is the one built here (`fixtures/minnetonka/limbs.json`).
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  createLivingMotion,
  deriveMotionSidecar,
  fft,
  LIMB_FLUTTER_FLOATS,
  LIMB_FLUTTER_WAVES,
  limbBends,
  limbFlutterOffset,
  limbFlutterWavelengths,
  limbHandleFloats,
  limbHandles,
  limbHandlesFromRig,
  limbSourceFromRig,
  limbWindFromSettings,
  limbWindModel,
  livingTransforms,
  nodeAngleLimit,
  NO_GUSTS,
  parseRig,
  quatConjugate,
  quatMultiply,
  cappedRotation,
  speedFromStrength,
  type LivingWind,
  type Vec3,
} from "./index";

const rig = parseRig(
  readFileSync(fileURLToPath(new URL("../fixtures/minnetonka/rig.json", import.meta.url)), "utf8"),
);
const sidecar = deriveMotionSidecar(rig, { treeHeightM: 6 });
const motion = createLivingMotion(rig, sidecar);
const source = limbSourceFromRig(motion, [0, 0, 0]);
const model = limbWindModel(source);

function wind(speedMps: number, bearingDeg = 0): LivingWind {
  return { speedMps, bearingDeg, gust: sidecar.wind.gust, season: "summer" };
}

/** `Z [x − o; 1]` of handle `j` from a `limbHandles` buffer. */
function handleDisplacement(z: Float64Array, j: number, x: Vec3, o: Vec3): Vec3 {
  const at = j * 12;
  const l = [x[0] - o[0], x[1] - o[1], x[2] - o[2]];
  return [0, 1, 2].map(
    (r) =>
      (z[at + r * 4] ?? 0) * (l[0] ?? 0) +
      (z[at + r * 4 + 1] ?? 0) * (l[1] ?? 0) +
      (z[at + r * 4 + 2] ?? 0) * (l[2] ?? 0) +
      (z[at + r * 4 + 3] ?? 0),
  ) as unknown as Vec3;
}

describe("a limbs skin sways as today's rig", () => {
  it("has one handle per oscillator, trunk first, as the converter writes them", () => {
    const handles = limbHandlesFromRig(motion);
    expect(handles.length).toBe(24);
    expect(handles[0]?.tree).toBe(true);
    const fixture = JSON.parse(
      readFileSync(
        fileURLToPath(new URL("../fixtures/minnetonka/limbs.json", import.meta.url)),
        "utf8",
      ),
    ) as { handles: Record<string, unknown>[] };
    expect(fixture.handles.length).toBe(handles.length);
    handles.forEach((h, j) => {
      const py = fixture.handles[j] ?? {};
      for (const key of ["key", "parent", "level", "tree"] as const) expect(h[key]).toBe(py[key]);
      for (const key of [
        "spanM",
        "frequencyHz",
        "damping",
        "widthM",
        "heightM",
        "staticTipM",
        "flutterM",
      ] as const)
        expect(h[key]).toBeCloseTo(py[key] as number, 4);
      for (const key of ["pivot", "direction", "samplePoint"] as const)
        (py[key] as number[]).forEach((v, c) => {
          expect(h[key][c]).toBeCloseTo(v, 5);
        });
    });
  });

  it("gives every joint of a limb the limb's bend times its gain, at any wind", () => {
    let checked = 0;
    for (const [speed, bearing] of [
      [speedFromStrength(0.1), 0],
      [speedFromStrength(0.5), 0],
      [speedFromStrength(0.3), 135],
    ] as const) {
      const w = wind(speed, bearing);
      for (const t of [0, 3.7, 41.25, 602.5]) {
        const today = livingTransforms(motion, t, w);
        const bends = limbBends(model, t, w);
        source.handles.forEach((limb, j) => {
          rig.nodes.forEach((node, i) => {
            const gain = sidecar.nodes.gainRad[i] ?? 0;
            if (sidecar.nodes.branch[i] !== limb.key || gain === 0 || node.parent < 0) return;
            const parent = today[node.parent]?.rotation ?? [0, 0, 0, 1];
            const local = quatMultiply(quatConjugate(parent), today[i]?.rotation ?? [0, 0, 0, 1]);
            const expected = cappedRotation(
              [
                gain * (bends.theta[j * 3] ?? 0),
                gain * (bends.theta[j * 3 + 1] ?? 0),
                gain * (bends.theta[j * 3 + 2] ?? 0),
              ],
              nodeAngleLimit(node),
            );
            for (let c = 0; c < 4; c += 1) expect(local[c]).toBeCloseTo(expected[c] ?? 0, 12);
            checked += 1;
          });
        });
      }
    }
    expect(checked).toBeGreaterThan(1000);
  });

  it("turns each handle about its pivot, which stays, and leaves the constant handle still", () => {
    const origin: Vec3 = [0.3, -0.4, 0.2];
    // A limb's gain as the converter finds it on a real tree (0.01–0.05 rad).
    const shifted = limbWindModel({
      ...source,
      origin,
      handles: source.handles.map((h) => ({ ...h, gain: 0.04, limitRad: 0.5 })),
    });
    const z = limbHandles(shifted, 12.5, wind(speedFromStrength(0.4), 30));
    expect(z.length).toBe(limbHandleFloats(source.handles.length + 1));
    for (let k = 0; k < 12; k += 1) expect(z[k]).toBe(0);
    source.handles.forEach((limb, j) => {
      const at = handleDisplacement(z, j + 1, limb.pivot, origin);
      for (const v of at) expect(Math.abs(v)).toBeLessThan(1e-12);
      // A point a metre along the limb moves, across it.
      const tip: Vec3 = [
        limb.pivot[0] + limb.direction[0],
        limb.pivot[1] + limb.direction[1],
        limb.pivot[2] + limb.direction[2],
      ];
      const moved = handleDisplacement(z, j + 1, tip, origin);
      const size = Math.hypot(...moved);
      expect(size).toBeGreaterThan(0);
      const along = moved[0] * limb.direction[0] + moved[1] * limb.direction[1] + moved[2] * limb.direction[2];
      expect(Math.abs(along)).toBeLessThan(0.05 * size);
    });
  });

  it("carries a child limb's pivot with its parent's bend: the child adds none there", () => {
    const z = limbHandles(model, 30, wind(speedFromStrength(0.5), 0));
    let children = 0;
    source.handles.forEach((limb, j) => {
      if (limb.parent === 0) return;
      const own = handleDisplacement(z, j + 1, limb.pivot, source.origin);
      expect(Math.hypot(...own)).toBeLessThan(1e-12);
      // With its parent's weight there, the pivot moves as the parent's handle moves it.
      const carried = handleDisplacement(z, limb.parent, limb.pivot, source.origin);
      const parentPivot = source.handles[limb.parent - 1]?.pivot ?? [0, 0, 0];
      if (Math.hypot(...limb.pivot.map((v, c) => v - (parentPivot[c] ?? 0))) > 0.1)
        expect(Math.hypot(...carried)).toBeGreaterThan(0);
      children += 1;
    });
    expect(children).toBeGreaterThan(15);
  });

  it("rings each limb at its own frequency", () => {
    // 12 m/s, where the resonance stands out of the gusts: the peak of the bend's velocity
    // spectrum within [0.5, 2]·f (living.test.ts's criterion for the rig's branches).
    const w = wind(12);
    const fs = 20;
    const n = 1 << 13; // 410 s, one Hann window
    const picks = [0, 1, 5, 9, source.handles.length - 1];
    const signals = picks.map(() => [0, 1, 2].map(() => new Float64Array(n)));
    for (let s = 0; s < n; s += 1) {
      const theta = limbBends(model, 300 + s / fs, w).theta;
      picks.forEach((j, p) => {
        for (let c = 0; c < 3; c += 1)
          (signals[p]?.[c] as Float64Array)[s] = theta[j * 3 + c] ?? 0;
      });
    }
    const df = fs / n;
    picks.forEach((j, p) => {
      const limb = source.handles[j];
      const f = limb?.frequencyHz ?? 1;
      const psd = new Float64Array(n / 2);
      for (const signal of signals[p] ?? []) {
        const re = new Float64Array(n);
        const im = new Float64Array(n);
        let mean = 0;
        for (const v of signal) mean += v / n;
        for (let i = 0; i < n; i += 1)
          re[i] = ((signal[i] ?? 0) - mean) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
        fft(re, im);
        for (let k = 0; k < n / 2; k += 1)
          psd[k] = (psd[k] ?? 0) + (re[k] ?? 0) ** 2 + (im[k] ?? 0) ** 2;
      }
      // The velocity spectrum, smoothed over ±0.03 Hz.
      const half = Math.round(0.03 / df);
      let best = 0;
      let peak = 0;
      for (let k = Math.ceil((0.5 * f) / df); k <= Math.floor((2 * f) / df); k += 1) {
        let sum = 0;
        for (let q = k - half; q <= k + half; q += 1) sum += (psd[q] ?? 0) * (q * df) ** 2;
        if (sum > best) {
          best = sum;
          peak = k * df;
        }
      }
      expect(
        Math.abs(peak / f - 1),
        `limb ${String(limb?.key)} at ${String(f)} Hz peaked at ${peak.toFixed(3)} Hz`,
      ).toBeLessThan(0.1);
    });
    expect(new Set(source.handles.map((h) => h.frequencyHz)).size).toBeGreaterThan(15);
  });

  it("is still at calm, by value, and the same at the same time", () => {
    const calm = limbHandles(model, 10, wind(0));
    expect(calm.every((v) => v === 0)).toBe(true);
    const a = limbHandles(model, 77.7, wind(8, 210));
    const b = limbHandles(limbWindModel(source), 77.7, wind(8, 210));
    expect(Array.from(a)).toEqual(Array.from(b));
    const settings = limbWindFromSettings({ strength: 0.1, bearingDeg: 45 }, source);
    expect(settings.speedMps).toBeCloseTo(speedFromStrength(0.1), 12);
    expect(settings.bearingDeg).toBe(45);
    expect(settings.gust).toBe(source.gust);
  });
});

describe("a limbs skin's leaf flutter", () => {
  const m = source.handles.length + 1;
  const at = m * 12;

  it("is the rig's band: 4 to 10 leaf sizes, unit RMS at share 1 before its amplitude", () => {
    const lengths = limbFlutterWavelengths();
    expect(lengths.length).toBe(LIMB_FLUTTER_WAVES);
    expect(Math.min(...lengths)).toBeGreaterThan(4);
    expect(Math.max(...lengths)).toBeLessThan(10);
    const w = wind(10);
    const z = limbHandles(model, 5, { ...w, gust: NO_GUSTS });
    expect(z[at]).toBe(1);
    expect(z[at + 1]).toBe(source.flutterByte);
    // At the reference speed a share of 1 moves 2·ref·tanh(1/2) RMS per component.
    const amplitude = 2 * source.flutterReferenceM * Math.tanh(0.5);
    const samples = 4000;
    const sums = [0, 0, 0];
    for (let s = 0; s < samples; s += 1) {
      const x: Vec3 = [Math.sin(s * 1.7) * 3, Math.cos(s * 2.3) * 3, 1 + (s % 97) * 0.05];
      const offset = limbFlutterOffset(z, at, 1, x);
      // Rest frame to the wind frame (bearing 0: along is north).
      sums[0] = (sums[0] ?? 0) + offset[1] ** 2;
      sums[1] = (sums[1] ?? 0) + offset[0] ** 2;
      sums[2] = (sums[2] ?? 0) + offset[2] ** 2;
    }
    for (const sum of sums) {
      const rms = Math.sqrt(sum / samples);
      expect(rms / amplitude).toBeGreaterThan(0.7);
      expect(rms / amplitude).toBeLessThan(1.3);
    }
    expect(limbFlutterOffset(z, at, 0, [1, 1, 1])).toEqual([0, 0, 0]);
  });

  it("is carried downwind at the canopy's speed: 2 to 10 Hz at the default wind", () => {
    const u = speedFromStrength(0.1);
    const fs = 60;
    const z0 = limbHandles(model, 0, wind(u));
    const z1 = limbHandles(model, 1 / fs, wind(u));
    const advected = u * source.canopyAdvection;
    for (let k = 0; k < 3 * LIMB_FLUTTER_WAVES; k += 1) {
      const o = at + 4 + k * 4;
      // The phase falls by κ·(advection) per second along the wind (north at bearing 0).
      const ka = z0[o + 1] ?? 0;
      let dPhase = (z0[o + 3] ?? 0) - (z1[o + 3] ?? 0);
      dPhase -= 2 * Math.PI * Math.round(dPhase / (2 * Math.PI));
      expect(dPhase * fs).toBeCloseTo(ka * advected, 6);
      const hz = Math.abs(ka * advected) / (2 * Math.PI);
      expect(hz).toBeGreaterThan(1.5);
      expect(hz).toBeLessThan(10);
    }
  });

  it("stops at calm and leafless, and takes no space in an older skin's handles", () => {
    expect(limbHandles(model, 3, wind(0))[at]).toBe(0);
    expect(limbHandles(model, 3, { ...wind(8), season: "winter" })[at]).toBe(0);
    expect(LIMB_FLUTTER_FLOATS % 4).toBe(0);
    expect(LIMB_FLUTTER_FLOATS / 4 + 1 + 3 * 32).toBeLessThanOrEqual(128);
  });
});

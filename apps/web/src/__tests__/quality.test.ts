import type { CaptureQuality } from "@twin/contracts";
import { describe, expect, it } from "vitest";

import { DEFAULTS, paramsFor, parsePlace, previewParamsFor } from "@/upload/options";
import {
  COVERAGE_TIP_IDS,
  forecast,
  qualityOf,
  REFINE_MIN_KEEP_PCT,
  REFINE_WITH_GAPS_MIN_KEEP_PCT,
  refineAdvice,
  summary,
} from "@/upload/quality";
import { parseCoverage } from "@/view/coverage";

/** What tools/pipeline/quality.py `write_coverage` writes, byte for byte. */
function coveragePly(points: [number, number, number, number][]): ArrayBuffer {
  const colours: Record<number, [number, number, number]> = {
    0: [132, 136, 130],
    1: [238, 170, 52],
    2: [64, 196, 108],
    3: [127, 216, 192],
  };
  const header = [
    "ply",
    "format binary_little_endian 1.0",
    "comment tier: 0 drop, 1 context, 2 keep, 3 camera (in capture order)",
    `element vertex ${String(points.length)}`,
    "property float x",
    "property float y",
    "property float z",
    "property uchar red",
    "property uchar green",
    "property uchar blue",
    "property uchar tier",
    "end_header",
    "",
  ].join("\n");
  const head = new TextEncoder().encode(header);
  const buffer = new ArrayBuffer(head.length + points.length * 16);
  new Uint8Array(buffer).set(head);
  const view = new DataView(buffer, head.length);
  points.forEach(([x, y, z, tier], i) => {
    view.setFloat32(i * 16, x, true);
    view.setFloat32(i * 16 + 4, y, true);
    view.setFloat32(i * 16 + 8, z, true);
    const [r, g, b] = colours[tier] ?? [0, 0, 0];
    view.setUint8(i * 16 + 12, r);
    view.setUint8(i * 16 + 13, g);
    view.setUint8(i * 16 + 14, b);
    view.setUint8(i * 16 + 15, tier);
  });
  return buffer;
}

describe("the coverage cloud", () => {
  it("separates the tier-coloured points from the camera path", () => {
    const coverage = parseCoverage(
      coveragePly([
        [0, 0, 0, 2],
        [1, 2, 3, 1],
        [4, 5, 6, 0],
        [10, 0, 1, 3],
        [0, 10, 1, 3],
      ]),
    );
    expect(coverage.counts).toEqual({ keep: 1, context: 1, drop: 1, cameras: 2 });
    expect(Array.from(coverage.positions)).toEqual([0, 0, 0, 1, 2, 3, 4, 5, 6]);
    expect(Array.from(coverage.cameraPath)).toEqual([10, 0, 1, 0, 10, 1]);
    expect(coverage.colors[0]).toBeCloseTo(64 / 255);
    expect(coverage.colors[4]).toBeCloseTo(170 / 255);
  });

  it("refuses what is not one", () => {
    expect(() => parseCoverage(new TextEncoder().encode("hello").buffer)).toThrow("not a PLY");
    const cut = coveragePly([[0, 0, 0, 2]]).slice(0, -4);
    expect(() => parseCoverage(cut)).toThrow("truncated");
  });
});

const verdict = (overrides: Partial<CaptureQuality> = {}): CaptureQuality => ({
  jobId: "job-1",
  mode: "preview",
  bar: "balanced",
  barApplied: "balanced",
  keepPct: 62.4,
  contextPct: 90,
  heldOutPsnr: 23.04,
  gaussians: { total: 10, kept: 9, keep: 6, context: 3, drop: 1 },
  roi: { center: [0, 0, 0], radius: 1 },
  tips: [],
  gsdMm: null,
  medianViews: 12,
  coverageUrl: null,
  ...overrides,
});

describe("the quality forecast", () => {
  it("reads as the phone shows it", () => {
    expect(forecast(verdict())).toBe(
      "62% of the scene met the high-quality bar · held-out 23.0 dB",
    );
    expect(forecast(verdict({ heldOutPsnr: null, keepPct: null }))).toBe(
      "nothing met the high-quality bar",
    );
    expect(forecast(verdict({ keepVerifiedPct: 48.4 }))).toBe(
      "62% of the scene met the high-quality bar (48% verified by held-out frames) · " +
        "held-out 23.0 dB",
    );
    expect(forecast(verdict({ keepVerifiedPct: null }))).toBe(forecast(verdict()));
    expect(summary(verdict({ mode: "refine", bar: "strict", barApplied: "balanced" }))).toBe(
      "Refined · 62% of the scene met the high-quality bar · held-out 23.0 dB · " +
        "Balanced bar (the bar asked for kept too little)",
    );
  });

  it("belongs to the run that measured it and no other", () => {
    expect(qualityOf(verdict(), { id: "job-1" })).not.toBeNull();
    expect(qualityOf(verdict(), { id: "job-2" })).toBeNull();
    expect(qualityOf(verdict(), undefined)).toBeNull();
    expect(qualityOf(null, { id: "job-1" })).toBeNull();
  });
});

describe("whether Refine is the next step", () => {
  const tip = (id: string) => ({ id, text: `tip ${id}` });

  it("leads with Refine when enough of the scene met the bar", () => {
    expect(refineAdvice(verdict())).toEqual({ worthIt: true, reason: null });
    // Gaps are worth fixing next time, but with most of the scene kept Refine still helps.
    expect(refineAdvice(verdict({ keepPct: 62, tips: [tip("from-above")] })).worthIt).toBe(true);
    expect(refineAdvice(verdict({ keepPct: REFINE_MIN_KEEP_PCT })).worthIt).toBe(true);
  });

  it("does not when too little of the scene met the bar, whatever the tips", () => {
    const advice = refineAdvice(verdict({ keepPct: 8, tips: [] }));
    expect(advice.worthIt).toBe(false);
    expect(advice.reason).toMatch(/^Only 8% of the scene met the high-quality bar\./);
    expect(refineAdvice(verdict({ keepPct: null })).reason).toMatch(/^None of the scene/);
  });

  it("does not when a middling share goes with missing views", () => {
    const gaps = refineAdvice(verdict({ keepPct: 31, tips: [tip("all-around")] }));
    expect(gaps.worthIt).toBe(false);
    expect(gaps.reason).toContain("missing views");
    expect(refineAdvice(verdict({ keepPct: REFINE_WITH_GAPS_MIN_KEEP_PCT - 1 })).worthIt).toBe(
      true,
    );
    // "Well covered" is not a gap.
    expect(refineAdvice(verdict({ keepPct: 31, tips: [tip("good")] })).worthIt).toBe(true);
    for (const id of COVERAGE_TIP_IDS) {
      expect(refineAdvice(verdict({ keepPct: 31, tips: [tip(id)] })).worthIt).toBe(false);
    }
  });

  it("has nothing to say about a run that was already refined", () => {
    expect(refineAdvice(verdict({ mode: "refine" }))).toEqual({ worthIt: false, reason: null });
  });
});

describe("preview and refine parameters", () => {
  it("a preview trains briefly and keeps the phone's frames and bar", () => {
    const options = { ...DEFAULTS, quality: "best" as const, bar: "strict" as const };
    expect(previewParamsFor("photo-reconstruct", options)).toEqual({
      normalize: { max_side: "auto" },
      train: { schedule_scale: 0.1, cap_max: 200000, train_max_side: 800 },
      quality: { mode: "preview", bar: "strict" },
    });
    expect(paramsFor("photo-reconstruct", options)).toEqual({
      normalize: { max_side: "auto" },
      train: { schedule_floor: 1, density_scale: 2 },
      quality: { bar: "strict" },
    });
  });

  it("an explicit photo size and frame rate are sent as numbers", () => {
    const options = { ...DEFAULTS, photoSize: "2400" as const, videoFps: "4" as const };
    expect(paramsFor("photo-reconstruct", options).normalize).toEqual({ max_side: 2400, fps: 4 });
  });

  it("sends nothing for packaging: Detail is this phone's viewing budget, not a cut", () => {
    for (const detail of ["200000", "800000"] as const) {
      expect(paramsFor("photo-reconstruct", { ...DEFAULTS, detail }).package).toBeUndefined();
      expect(paramsFor("splat-ingest", { ...DEFAULTS, detail })).toEqual({
        normalize: { up_axis: "", heading_deg: 0 },
      });
    }
  });

  it("a typed place becomes the splat's georeference; an empty or bad one sends none", () => {
    expect(paramsFor("splat-ingest", { ...DEFAULTS, place: "46.134, -123.881" })).toEqual({
      normalize: { up_axis: "", heading_deg: 0 },
      georeference: { lat: 46.134, lon: -123.881 },
    });
    for (const place of ["", "46.134", "north, west", "91, 0", "0, 181", "1, 2, 3"]) {
      expect(paramsFor("splat-ingest", { ...DEFAULTS, place }).georeference).toBeUndefined();
    }
    expect(
      paramsFor("photo-reconstruct", { ...DEFAULTS, place: "46, -123" }).georeference,
    ).toBeUndefined();
  });

  it("reads a place with commas, spaces or semicolons, in decimal degrees", () => {
    expect(parsePlace(" 46.134 -123.881 ")).toEqual({ lat: 46.134, lon: -123.881 });
    expect(parsePlace("-33.9;151.2")).toEqual({ lat: -33.9, lon: 151.2 });
    expect(parsePlace("46°N 123°W")).toBeNull();
  });

  it("a splat file has no preview and no quality bar", () => {
    const params = previewParamsFor("splat-ingest", DEFAULTS);
    expect(params).toEqual(paramsFor("splat-ingest", DEFAULTS));
    expect(params.quality).toBeUndefined();
  });
});

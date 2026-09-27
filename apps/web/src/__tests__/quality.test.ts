import type { CaptureQuality } from "@twin/contracts";
import { describe, expect, it } from "vitest";

import { DEFAULTS, paramsFor, previewParamsFor } from "@/upload/options";
import { forecast, qualityOf, summary } from "@/upload/quality";
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

describe("preview and refine parameters", () => {
  it("a preview trains briefly and keeps the phone's frames and bar", () => {
    const options = { ...DEFAULTS, quality: "best" as const, bar: "strict" as const };
    expect(previewParamsFor("photo-reconstruct", options)).toEqual({
      normalize: { max_side: 1600, fps: 4 },
      package: { max_gaussians: 400000 },
      train: { schedule_scale: 0.1, cap_max: 200000, train_max_side: 800 },
      quality: { mode: "preview", bar: "strict" },
    });
    expect(paramsFor("photo-reconstruct", options)).toEqual({
      normalize: { max_side: 1600, fps: 4 },
      package: { max_gaussians: 400000 },
      train: { schedule_floor: 1, cap_max: 1000000 },
      quality: { bar: "strict" },
    });
  });

  it("a splat file has no preview and no quality bar", () => {
    const params = previewParamsFor("splat-ingest", DEFAULTS);
    expect(params).toEqual(paramsFor("splat-ingest", DEFAULTS));
    expect(params.quality).toBeUndefined();
  });
});

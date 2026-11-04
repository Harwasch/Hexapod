import { describe, expect, it } from "vitest";
import type { LiveState, LiveTrain, LiveTrainPlan } from "@twin/contracts";
import {
  blockCountText,
  isNewAttempt,
  qualityReadouts,
  chartDomain,
  chartPoints,
  ceilingText,
  cost,
  distanceColour,
  eventsDisplay,
  gainBarFraction,
  lastStep,
  nearestCameraDistance,
  nearestIndex,
  phaseOf,
  pillText,
  snapshotAt,
  stopRule,
  timeText,
} from "@/view/liveTrain";

const plan: LiveTrainPlan = {
  stepsMax: 60000,
  refineStart: 1000,
  refineStop: 50000,
  scheduleFactor: 2,
  rule: { every: 1000, window: 4000, minGainDb: 0.05 },
};
const train: LiveTrain = {
  plan,
  series: {
    step: [1, 50000, 54000],
    t: [1, 400, 500],
    gs: [10, 100, 100],
    lr: [0.01, 0.001, 0.0001],
    its: [20, 10, 10],
    mem: [2, 3, 4],
    sh: [0, 3, 3],
    loss: { total: [0.5, 0.1, 0.09], future: [null, 0.03, 0.02] },
  },
  evals: [
    { step: 50000, psnr: 25, gain: null, decision: "warming up" },
    { step: 54000, psnr: 25.02, gain: 0.02, decision: "converged" },
  ],
};
const state = (overrides: Partial<LiveState> = {}): LiveState => ({
  jobId: "j",
  captureId: "c",
  captureName: "Tree",
  recipe: "photo",
  status: "in-progress",
  error: null,
  siteId: null,
  stage: null,
  stepsDone: 1,
  stepsStarted: 2,
  cameras: null,
  splat: null,
  train,
  ...overrides,
});

describe("live training presentation", () => {
  it("uses the resolved schedule and rule", () => {
    expect(phaseOf(999, plan)).toBe("Warm-up");
    expect(phaseOf(1000, plan)).toBe("Densifying");
    expect(phaseOf(50000, plan)).toBe("Settling");
    expect(pillText(state())).toBe("Training · settling");
    expect(
      pillText(state({ train: { ...train, stop: { step: 54001, maxSteps: 60000, gain: 0.02 } } })),
    ).toBe("Finished · converged");
    expect(pillText(state({ status: "complete" }))).toBe("Finished");
    expect(pillText(state({ status: "error" }))).toBe("Stopped");
    const rule = stopRule(train);
    expect(rule.line).toBe("stop if Δ best PSNR (last 4k steps) < 0.05 dB");
    expect(rule.checks.every((c) => c.ok)).toBe(true);
    expect(rule.checks[2]?.value).toBe("0.020 dB");
    expect(ceilingText(plan)).toBe("30k × 2");
    expect(lastStep(train)).toBe(54000);
  });
  it("handles missing telemetry, unknown cost and real evaluation windows", () => {
    expect(chartPoints(null, "gs")).toEqual([]);
    expect(stopRule(null).line).toBe("–");
    expect(cost(3600, null)).toBeNull();
    expect(cost(3600, 1.1)).toBe(1.1);
    expect(timeText(3660)).toBe("1 h 1 min");
    expect(timeText(null)).toBe("–");
    expect(stopRule({ ...train, evals: [] }).checks[1]?.ok).toBe(false);
    expect(gainBarFraction(-0.1, 0.05)).toBe(0);
    expect(gainBarFraction(5, 0.05)).toBe(1);
  });
  it("keeps plot domains finite and nearest points synchronized", () => {
    expect(chartDomain([], true)).toEqual([-6, -2]);
    const [lo, hi] = chartDomain([{ step: 1, value: 0.001 }], true);
    expect(lo).toBeLessThan(-3);
    expect(hi).toBeGreaterThan(-3);
    expect(chartDomain([{ step: 1, value: 2 }], false, 24, true)[1]).toBeGreaterThan(24);
    expect(nearestIndex([1, 50000, 54000], 53500)).toBe(2);
    expect(nearestIndex([], 0)).toBe(-1);
    expect(chartPoints(train, "loss").at(-1)?.value).toBe(0.09);
    expect(chartPoints(train, "psnr")).toHaveLength(2);
  });
  it("scrubs to the previous available snapshot, and formats derived events", () => {
    const snapshots = [
      { step: 3000, key: "a", count: 100 },
      { step: 7500, key: "b", count: 100 },
    ];
    expect(snapshotAt(snapshots, 7000)?.key).toBe("a");
    expect(snapshotAt(snapshots, 2999)).toBeNull();
    expect(snapshotAt(snapshots, 9000)?.key).toBe("b");
    expect(eventsDisplay({ events: [{ step: 1000, kind: "sh", text: "SH 1" }] })).toEqual([
      { step: "1,000", kind: "sh", text: "SH 1" },
    ]);
  });
  it("measures nearest cameras in the same frame and maps metric distance", () => {
    expect(nearestCameraDistance(0, 0, 0, [3, 4, 0, 0, 0, 2])).toBe(2);
    expect(nearestCameraDistance(0, 0, 0, [])).toBeNull();
    expect(distanceColour(0)).toEqual([127 / 255, 216 / 255, 192 / 255]);
    expect(distanceColour(40)).toEqual([216 / 255, 195 / 255, 154 / 255]);
    expect(distanceColour(1000)).toEqual(distanceColour(120));
  });
});

describe("review regressions", () => {
  it("uses the singular for one block", () => {
    expect(blockCountText(1)).toBe("1 block");
    expect(blockCountText(2)).toBe("2 blocks");
    expect(blockCountText(null)).toBe("– blocks");
  });
  it("keeps the displayed mesh and scrub on block changes, but resets on retries", () => {
    const previous = { run: "a", block: "b0", planId: "plan" };
    expect(isNewAttempt(previous, { ...previous, run: "b", block: "b1" })).toBe(false);
    expect(isNewAttempt(previous, { ...previous, run: "b" })).toBe(true);
    expect(isNewAttempt(previous, { ...previous, planId: "new", block: "b1" })).toBe(true);
    expect(isNewAttempt(previous, previous)).toBe(false);
    expect(isNewAttempt(null, previous)).toBe(false);
    expect(isNewAttempt({ run: "a" }, { run: "b", block: null })).toBe(true);
  });
  it("prefers shown snapshot quality, falling back per value to training metrics", () => {
    const quality = { step: 500, farFieldShare: 0.2, elongationShare: 0.03 };
    expect(qualityReadouts(null, quality)).toEqual(["Far-field 20.0%", "Elongated 3.0%"]);
    expect(qualityReadouts({ step: 100, key: "a", count: 10, farFieldShare: 0 }, quality)).toEqual([
      "Far-field 0.0%",
      "Elongated 3.0%",
    ]);
    expect(
      qualityReadouts({ step: 100, key: "a", count: 10, elongationShare: 0.1 }, quality),
    ).toEqual(["Far-field 20.0%", "Elongated 10.0%"]);
    expect(qualityReadouts(null, null)).toEqual([]);
  });
});

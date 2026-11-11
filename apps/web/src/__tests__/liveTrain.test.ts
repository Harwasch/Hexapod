import { describe, expect, it, vi } from "vitest";
import type { LiveState, LiveTrain, LiveTrainPlan } from "@twin/contracts";
import {
  tilesAt,
  blockCountText,
  effectiveCap,
  snapshotLag,
  scrubPosition,
  seedsLabel,
  decodeSeeds,
  lossLines,
  finalReadouts,
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
  it.each([
    ["preempted", "Pre-empted"],
    ["oom", "Out of memory"],
    ["failed", "Failed"],
  ] as const)("freezes a run that ended: %s", (reason, label) => {
    const ended = { reason, step: 50000 };
    const frozen = { ...train, ended };
    expect(pillText(state({ status: "error", train: frozen }))).toBe(`${label} · step 50,000`);
    expect(lastStep(frozen)).toBe(50000);
    expect(phaseOf(lastStep(frozen), plan, 100, ended)).toBe(label);
    expect(scrubPosition(60000, lastStep(frozen), 60000, 600, true, true)).toBe(50000);
    expect(scrubPosition(40000, lastStep(frozen), 60000, 600, false, true)).toBe(40000);
    expect(chartPoints(frozen, "gs").at(-1)?.step).toBe(50000);
    expect(chartPoints(frozen, "psnr").at(-1)?.step).toBe(50000);
    expect(stopRule(frozen).gain).toBeNull();
  });

  it("uses the resolved schedule and rule", () => {
    expect(phaseOf(999, plan)).toBe("Warm-up");
    expect(phaseOf(1000, plan)).toBe("Densifying");
    expect(phaseOf(50000, plan)).toBe("Settling");
    expect(pillText(state())).toBe("Training · settling");
    expect(
      pillText(state({ train: { ...train, stop: { step: 54001, maxSteps: 60000, gain: 0.02 } } })),
    ).toBe("Finished · converged");
    expect(pillText(state({ status: "complete" }))).toBe("Finished");
    expect(pillText(state({ status: "error" }))).toBe("Failed");
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

describe("real-run console", () => {
  it("enters relocating at equality with the lower cap, then settles", () => {
    const plan = { cap: 3456214, capMax: 3455958, refineStart: 930, refineStop: 46500 };
    expect(effectiveCap(plan)).toBe(3455958);
    expect(phaseOf(5000, plan, 3455958)).toBe("Relocating");
    expect(phaseOf(5000, plan, 3456214)).toBe("Relocating");
    expect(phaseOf(5000, plan, 3455957)).toBe("Densifying");
    expect(phaseOf(46500, plan, 3456214)).toBe("Settling");
    expect(
      pillText(
        state({ train: { plan, series: { ...train.series, step: [5000], gs: [3456214] } } }),
      ),
    ).toBe("Relocating · at cap");
    const domain = chartDomain([{ step: 5000, value: 3455958 }], false, 3455958, true);
    expect(domain[1]).toBeGreaterThan(3455958);
    expect(chartDomain([], false, 0.05, true)[1]).toBeGreaterThan(0.05);
  });
  it("shows snapshot lag and snaps a released drag back within six pixels of live", () => {
    expect(snapshotLag(7500, 15800)).toBe("−8.3k");
    expect(snapshotLag(-1, 15800)).toBeNull();
    expect(snapshotLag(15800, 15800)).toBeNull();
    expect(scrubPosition(15500, 15600, 55800, 600)).toBe(15500);
    expect(scrubPosition(15500, 15600, 55800, 600, true)).toBeNull();
    expect(scrubPosition(14000, 15600, 55800, 600, true)).toBe(14000);
    expect(
      snapshotAt(
        [
          { step: 5580, key: "a", count: 1 },
          { step: 13950, key: "b", count: 1 },
        ],
        13949,
      )?.key,
    ).toBe("a");
  });
  it("decodes ten-byte seed records and labels full source counts", () => {
    const seeds = {
      key: "seeds.bin",
      count: 1,
      of: 640000,
      sources: { sfm: 640000, dense: 0, seed: 0 },
      origin: [1, 2, 3],
      scale: 2,
      bytes: 10,
    };
    const data = new ArrayBuffer(10),
      view = new DataView(data);
    view.setInt16(0, 32767, true);
    view.setUint8(9, 1);
    const decoded = decodeSeeds(data, seeds);
    expect([...decoded.positions]).toEqual([3, 2, 3]);
    expect(decoded.colors[0]).toBeCloseTo(216 / 255);
    expect(seedsLabel(seeds)).toBe("640k seeds · 640k SfM · 0 dense");
    expect(() => decodeSeeds(new ArrayBuffer(9), seeds)).toThrow("Invalid seeds");
  });
  it("reads quality losses and memory only when measured, and formats final scores", () => {
    const enriched = {
      ...train,
      quality: {
        series: {
          step: [100],
          farFieldShare: [0.14],
          elongationShare: [0.06],
          peakAllocatedGb: [5.9],
          loss: { depth: [0.05], normal: [0.2], anisotropy: [0.02], gradient: [0] },
        },
      },
    };
    expect(lossLines(enriched).map((l) => l.name)).toEqual([
      "total",
      "depth",
      "normal",
      "anisotropy",
    ]);
    enriched.quality.series.loss.gradient = [0.01];
    expect(lossLines(enriched).at(-1)?.name).toBe("gradient");
    expect(chartPoints(enriched, "mem")).toEqual([{ step: 100, value: 5.9 }]);
    expect(chartPoints(train, "mem").at(-1)?.value).toBe(4);
    expect(chartPoints(enriched, "far")[0]?.value).toBeCloseTo(14);
    expect(finalReadouts({ final: { psnr: 25.123, ssim: 0.9, lpips: 0.1, ccPsnr: 26 } })).toEqual([
      "PSNR 25.12",
      "SSIM 0.900",
      "LPIPS 0.100",
      "cc_PSNR 26.00",
    ]);
  });
  it("draws the cap above the real series and synchronizes all eleven crosshairs", async () => {
    const { benchTrain } = await import("../../e2e/liveTrainFixture");
    const { liveConsole } = await import("@/view/liveConsole");
    const markup = await import("../../view.html?raw");
    document.body.innerHTML = markup.default;
    const context = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const consoleView = liveConsole({
      scrub: vi.fn(),
      color: vi.fn(),
      camera: vi.fn(),
      view: vi.fn(),
      layers: vi.fn(),
    });
    try {
      consoleView.update(state({ train: benchTrain("https://bucket") }));
      consoleView.snapshot(13950);
      const svg = document.getElementById("lc-chart-gs");
      expect(svg?.innerHTML.indexOf('data-reference="cap"')).toBeGreaterThan(
        svg?.innerHTML.indexOf('data-series="gs"') ?? 0,
      );
      expect(document.getElementById("lc-pill")?.textContent).toBe("Relocating · at cap");
      if (!svg) throw new Error("Missing chart");
      vi.spyOn(svg, "getBoundingClientRect").mockReturnValue(new DOMRect(0, 0, 300, 150));
      svg.dispatchEvent(new MouseEvent("pointermove", { bubbles: true, clientX: 150 }));
      expect(document.querySelectorAll(".lc-crosshair")).toHaveLength(11);
      expect(
        new Set(
          [...document.querySelectorAll("[data-training-chart]")].map((node) =>
            node.getAttribute("data-crosshair"),
          ),
        ).size,
      ).toBe(1);
      expect(document.querySelector("#lc-chart-gain [data-reference] text")?.textContent).toBe(
        "0.05 dB stop",
      );
    } finally {
      consoleView.stop();
      context.mockRestore();
      document.body.innerHTML = "";
    }
  });
  it("links the replacement run and clears the chip when no replacement is set", async () => {
    const { liveConsole } = await import("@/view/liveConsole");
    const markup = await import("../../view.html?raw");
    document.body.innerHTML = markup.default;
    const context = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const consoleView = liveConsole({
      scrub: vi.fn(),
      color: vi.fn(),
      camera: vi.fn(),
      view: vi.fn(),
      layers: vi.fn(),
    });
    try {
      const replacement = document.getElementById("lc-replacement") as HTMLAnchorElement;
      consoleView.update(state());
      expect(replacement.hidden).toBe(true);
      const replacedBy = "4769ee65e33c44f2aacd87a45dfa3811";
      consoleView.update(
        state({
          status: "error",
          train: { ...train, ended: { reason: "failed", step: 50000 }, replacedBy },
        }),
      );
      expect(replacement.hidden).toBe(false);
      expect(replacement.textContent).toBe("→ 4769ee65");
      expect(replacement.getAttribute("href")).toBe(`view.html#live/${replacedBy}`);
      consoleView.update(state());
      expect(replacement.hidden).toBe(true);
      expect(replacement.hasAttribute("href")).toBe(false);
      expect(replacement.textContent).toBe("");
    } finally {
      consoleView.stop();
      context.mockRestore();
      document.body.innerHTML = "";
    }
  });
  it("pins an ended console and its scrubber without showing pending work", async () => {
    const { liveConsole } = await import("@/view/liveConsole");
    const markup = await import("../../view.html?raw");
    document.body.innerHTML = markup.default;
    const fillRect = vi.fn();
    const context = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      setTransform: vi.fn(),
      fillRect,
      fillText: vi.fn(),
    } as unknown as CanvasRenderingContext2D);
    const scrub = vi.fn();
    const consoleView = liveConsole({
      camera: vi.fn(),
      scrub,
      color: vi.fn(),
      view: vi.fn(),
      layers: vi.fn(),
    });
    try {
      const error = "Original <provider> error";
      consoleView.update(
        state({
          status: "error",
          error,
          train: { ...train, ended: { reason: "preempted", step: 50000 } },
        }),
      );
      const timeline = document.getElementById("lc-timeline");
      expect(document.getElementById("lc-pill")?.textContent).toBe("Pre-empted · step 50,000");
      expect(document.getElementById("lc-kpi-0")?.textContent).toBe("50,000");
      expect(document.getElementById("lc-kpi-2")?.textContent).toBe("25.00 dB");
      expect(document.getElementById("lc-kpi-3")?.textContent).toBe("Pre-empted");
      expect(document.getElementById("lc-kpi-4")?.textContent).toBe("7 min");
      expect(document.getElementById("lc-follow")?.textContent).toBe("step 50,000");
      expect(document.getElementById("lc-error")?.textContent).toBe(error);
      expect(document.getElementById("lc-error")?.hidden).toBe(false);
      expect(document.getElementById("lc-now-gain")?.textContent).toBe("–");
      expect(document.getElementById("lc-now-ceiling")?.textContent).not.toContain("…");
      expect(timeline?.getAttribute("aria-valuemax")).toBe("50000");
      expect(timeline?.getAttribute("aria-valuenow")).toBe("50000");
      expect(document.querySelectorAll('[data-stop="50000"]')).toHaveLength(9);
      expect(fillRect).toHaveBeenLastCalledWith(expect.any(Number), 16, 2, 22);
      timeline?.dispatchEvent(new KeyboardEvent("keydown", { key: "Home" }));
      expect(scrub).toHaveBeenLastCalledWith(0);
      timeline?.dispatchEvent(new KeyboardEvent("keydown", { key: "End" }));
      expect(scrub).toHaveBeenLastCalledWith(50000);
    } finally {
      consoleView.stop();
      context.mockRestore();
      document.body.innerHTML = "";
    }
  });
});

describe("chartDomain floor", () => {
  it("never gives a non-negative series a negative axis", () => {
    const [lo] = chartDomain([
      { step: 0, value: 0 },
      { step: 1, value: 2 },
    ]);
    expect(lo).toBe(0);
    expect(
      chartDomain([
        { step: 0, value: -1 },
        { step: 1, value: 2 },
      ])[0],
    ).toBeLessThan(-1);
  });
});

describe("tiles versus history thumbnails", () => {
  const tiles = { url: "/bench/tileset.json", step: 50000, final: false };
  const full = { ...train, tiles };
  it("streams at live, at the tiles' step and at the latest chart step", () => {
    expect(tilesAt(full, null)).toEqual(tiles);
    expect(tilesAt(full, 50000)).toEqual(tiles);
    expect(tilesAt(full, 54000)).toEqual(tiles);
    expect(tilesAt(full, 52000)).toBeNull();
    expect(tilesAt(full, 3000)).toBeNull();
    expect(tilesAt(train, null)).toBeNull();
  });
  it("never pretends a published ENU scan is in the solve frame", () => {
    expect(tilesAt({ ...full, tiles: { ...tiles, frame: "enu" } }, null)).toBeNull();
  });
});

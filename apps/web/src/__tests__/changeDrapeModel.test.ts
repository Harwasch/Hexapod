/**
 * What the change drape says and draws, from a change map (lib/changeDrape.ts, ADR 0018): the
 * dates it is offered for, its label and the floor under which no change can be seen, the
 * placement it is only as good as, the reason a request was refused, and its areas grouped
 * into one draw per class and confidence.
 */
import { describe, expect, it } from "vitest";

import type { ChangeArea, ChangeMap, Epoch, EpochSurface } from "@twin/contracts";

import {
  changeBuckets,
  changeError,
  changeLabel,
  changeOffer,
  calibrationNote,
  legendRows,
  vegetationNote,
  verdictNote,
  VEGETATION_TITLE,
  NEEDS_TOKEN,
} from "@/lib/changeDrape";

import { E1, E2 } from "./epochFixtures";

const surface = (extra: Partial<EpochSurface> = {}): EpochSurface => ({
  status: "measured",
  assetId: "a",
  url: "https://tiles.example/surface.bin",
  cellM: 1,
  level: 2,
  policy: "surface-v1",
  calibration: "uncalibrated",
  cells: 1280,
  surfacedCells: 1272,
  medianSigmaM: 0.02,
  originHeightM: 0,
  groundWindowM: 5,
  ...extra,
});

const measured = (e: Epoch, extra: Partial<EpochSurface> = {}): Epoch => ({
  ...e,
  surface: surface(extra),
});

const square = (x: number, y: number, d = 0.0001): number[][][] => [
  [
    [x, y],
    [x + d, y],
    [x + d, y + d],
    [x, y + d],
    [x, y],
  ],
];

function area(cls: ChangeArea["class"], lowConfidence: boolean, x: number): ChangeArea {
  return {
    class: cls,
    footprint: { type: "MultiPolygon", coordinates: [square(x, 0), square(x, 1)] },
    cells: 4,
    areaM2: 4,
    lowConfidence,
    lowConfidenceCells: lowConfidence ? 4 : 0,
    meanDeltaM: cls === "insufficient" ? null : 1.5,
    maxAbsDeltaM: cls === "insufficient" ? null : 1.5,
  };
}

type Layer = ChangeMap["ground"];

function layer(name: Layer["layer"], extra: Partial<Layer> = {}): Layer {
  return {
    layer: name,
    label: name === "ground" ? "Ground: the bare earth" : "Vegetation and objects",
    heights: name === "ground" ? "ground" : "dsm",
    minDetectableM: name === "ground" ? 0.489 : 1.089,
    motionM: name === "ground" ? 0 : 0.25,
    horizontalRadiusM: name === "ground" ? 1 : 2,
    driftReachM: name === "ground" ? 3.536 : 0,
    counts: { unchanged: 1137, raised: 47, lowered: 24, insufficient: 72 },
    lowConfidence: { unchanged: 3, raised: 23, lowered: 0 },
    areaLevel: 2,
    areaCellM: 1,
    areas: [
      area("insufficient", false, 5),
      area("raised", true, 1),
      area("lowered", false, 3),
      area("raised", false, 2),
      area("raised", false, 4),
    ],
    ...extra,
  };
}

function map(extra: Partial<ChangeMap> = {}): ChangeMap {
  return {
    policy: "change-v1",
    derived: true,
    label: "Derived from height surfaces, not observed",
    calibration: "uncalibrated",
    siteId: "s",
    fromEpochId: E1.id,
    toEpochId: E2.id,
    fromAssetId: "a1",
    toAssetId: "a2",
    frame: "tileset",
    level: 2,
    nativeLevel: 2,
    cellM: 1,
    thresholdSigma: 3,
    uncertainty: {
      fromM: 0.03,
      toM: 0.062,
      fromVerdict: "survey",
      toVerdict: "survey",
      fromFrameM: 0,
      toFrameM: 0.062,
    },
    ground: layer("ground"),
    vegetation: layer("vegetation"),
    appearance: { status: "not-computed", reason: "needs the new photos" },
    ...extra,
  };
}

describe("the offer", () => {
  it("is made for two dates that both measured their heights, older first", () => {
    const epochs = [measured(E1), measured(E2)];
    expect(changeOffer(epochs, E2.id, E1.id)).toEqual({ offered: true, from: E1.id, to: E2.id });
    expect(changeOffer(epochs, E1.id, E2.id)).toEqual({ offered: true, from: E1.id, to: E2.id });
  });

  it("says which date has no height surface, rather than offering a blank drape", () => {
    const one = changeOffer([measured(E1), E2], E1.id, E2.id);
    expect(one).toEqual({
      offered: false,
      why: "Not offered: Jun 1, 2026 has no height surface measured",
    });
    const none = changeOffer([E1, { ...E2, surface: null }], E1.id, E2.id);
    expect(none.offered).toBe(false);
    expect(!none.offered && none.why).toBe(
      "Not offered: neither Mar 1, 2026 nor Jun 1, 2026 has a height surface measured",
    );
  });

  it("is not made for one date twice, or a date the site does not list", () => {
    const epochs = [measured(E1), measured(E2)];
    expect(changeOffer(epochs, E1.id, E1.id).offered).toBe(false);
    expect(changeOffer(epochs, E1.id, "gone").offered).toBe(false);
  });
});

describe("the label", () => {
  it("says it is derived, and the least change each layer can show", () => {
    expect(changeLabel(map().ground)).toBe(
      "Derived from height surfaces — not observed · changes under ±0.49 m cannot be seen between these dates",
    );
    expect(changeLabel(map().vegetation)).toContain("changes under ±1.1 m cannot");
    const ground = (minDetectableM: number | null) => layer("ground", { minDetectableM });
    expect(changeLabel(ground(30))).toContain("changes under ±30 m cannot");
    // Rounded up, never down: the floor is not understated.
    expect(changeLabel(ground(0.124))).toContain("changes under ±0.13 m cannot");
    expect(changeLabel(ground(0.191))).toContain("changes under ±0.2 m cannot");
    expect(changeLabel(ground(0.12))).toContain("changes under ±0.12 m cannot");
  });

  it("says so when no cell could be compared", () => {
    expect(changeLabel(layer("ground", { minDetectableM: null }))).toBe(
      "Derived from height surfaces — not observed · no cell could be compared between these dates",
    );
  });

  it("calls the rule and the ground window uncalibrated", () => {
    const epochs = [measured(E1), measured(E2)];
    expect(calibrationNote(map(), epochs)).toBe(
      "Uncalibrated: the 3σ rule and the 5 m ground window are first principles until a site scanned twice measures them",
    );
    const noWindow = [measured(E1, { groundWindowM: null }), measured(E2, { groundWindowM: null })];
    expect(calibrationNote(map(), noWindow)).toBe(
      "Uncalibrated: the 3σ rule is first principles until a site scanned twice measures it",
    );
  });

  it("names vegetation for what it is, and its sway as uncalibrated", () => {
    expect(VEGETATION_TITLE).toBe(
      "Vegetation: canopy height change, higher threshold, wind and growth not ground change",
    );
    expect(vegetationNote(map(), map().vegetation)).toBe(
      "Uncalibrated: the 3σ rule and ±0.25 m of wind sway per date are first principles until a site scanned twice measures them",
    );
  });
});

describe("the placement it is only as good as", () => {
  it("says nothing when both dates are survey-grade", () => {
    expect(verdictNote(map().uncertainty, E1, E2)).toBeNull();
  });

  it("says not survey-grade, with each date's verdict, when either is coarse", () => {
    const coarse = { ...map().uncertainty, toM: 5, toVerdict: "coarse" as const };
    expect(verdictNote(coarse, E1, E2)).toBe(
      "Not survey-grade: Mar 1, 2026 survey (±0.03 m) · Jun 1, 2026 coarse (±5 m)",
    );
    const both = { ...coarse, fromM: 2, fromVerdict: "coarse" as const };
    expect(verdictNote(both, E1, E2)).toBe(
      "Not survey-grade: Mar 1, 2026 coarse (±2 m) · Jun 1, 2026 coarse (±5 m)",
    );
  });
});

describe("a refusal", () => {
  it("asks for the pilot's write token on a 401 or 403", () => {
    expect(changeError(401)).toBe(NEEDS_TOKEN);
    expect(changeError(403)).toBe(NEEDS_TOKEN);
    expect(NEEDS_TOKEN).toMatch(/needs the pilot's write token.*Settings › API access/);
  });

  it("says each 409 in words", () => {
    const codes = [
      "epoch-not-published",
      "no-site-frame",
      "surface-not-measured",
      "confidence-not-measured",
      "frames-differ",
      "surface-unreadable",
      "confidence-unreadable",
      "epoch-unaligned",
      "not-metric",
      "scale-corrected",
    ];
    const said = codes.map((code) => changeError(409, code));
    expect(new Set(said).size).toBe(codes.length);
    expect(changeError(409, "confidence-not-measured")).toMatch(/no confidence map/);
    expect(changeError(409, "epoch-unaligned")).toMatch(/not aligned/);
    expect(changeError(409, "not-metric")).toMatch(/not in metres/);
    expect(changeError(409, "scale-corrected")).toMatch(/scale/);
    expect(changeError(409, "something-new")).toMatch(/409/);
  });

  it("says a 404, a 422, a 429 and anything else", () => {
    expect(changeError(404)).toMatch(/not found/);
    expect(changeError(422)).toMatch(/two different dates/);
    expect(changeError(429)).toMatch(/try again/);
    expect(changeError(500)).toMatch(/500/);
  });
});

describe("the drawing", () => {
  it("is one draw per class and confidence, raised first, each with every polygon of its areas", () => {
    const buckets = changeBuckets(map().ground);
    expect(
      buckets.map((b) => [b.cls, b.lowConfidence, b.areas, b.cells, b.polygons.length]),
    ).toEqual([
      ["raised", false, 2, 8, 4],
      ["raised", true, 1, 4, 2],
      ["lowered", false, 1, 4, 2],
      ["insufficient", false, 1, 4, 2],
    ]);
    // A Polygon footprint counts as one polygon.
    const polygon = {
      ...area("lowered", true, 9),
      footprint: { type: "Polygon", coordinates: square(9, 0) },
    };
    expect(
      changeBuckets(layer("ground", { areas: [polygon as unknown as ChangeArea] }))[0]?.polygons,
    ).toHaveLength(1);
  });

  it("draws nothing for no areas, and never an unchanged one", () => {
    expect(changeBuckets(layer("ground", { areas: [] }))).toEqual([]);
    const unchanged = { ...area("raised", false, 1), class: "unchanged" } as unknown as ChangeArea;
    expect(changeBuckets(layer("ground", { areas: [unchanged] }))).toEqual([]);
  });
});

describe("the legend", () => {
  it("counts every class, low confidence among them, and unchanged as counted only", () => {
    expect(legendRows(map().ground)).toEqual([
      { key: "raised", label: "Raised", cells: 47, lowConfidence: 23, drawn: true },
      { key: "lowered", label: "Lowered", cells: 24, lowConfidence: 0, drawn: true },
      { key: "insufficient", label: "Not comparable", cells: 72, lowConfidence: 0, drawn: true },
      { key: "unchanged", label: "Unchanged", cells: 1137, lowConfidence: 3, drawn: false },
    ]);
  });
});

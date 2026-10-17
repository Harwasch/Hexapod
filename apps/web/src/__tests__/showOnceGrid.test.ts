/**
 * Show once, everywhere's grid on the client: the slider's rule (the same quantized compare
 * as the server's areas and the drape's uniform), the tally, the patch a protect tap takes,
 * and the rectangles that carry protected ground to the server.
 */

import { expect, it } from "vitest";

import type { SignatureSimilarity } from "@twin/contracts";

import {
  decodeScores,
  formatAcres,
  isLit,
  litStats,
  NO_SCORE,
  patchAt,
  protectedRectangles,
  rampImage,
  rampThreshold,
  rawThreshold,
  toggleProtected,
  type ScoreGrid,
} from "@/features/showOnce/grid";

const _ = NO_SCORE;

/** A 4 x 3 grid, row 0 at the north, 100 m² cells. */
function grid(scores: number[], columns = 4): ScoreGrid {
  const rows = scores.length / columns;
  return {
    west: -120.05,
    south: 34.69,
    east: -120.05 + columns * 0.001,
    north: 34.69 + rows * 0.001,
    columns,
    rows,
    cellM: 10,
    cellAreaM2: 100,
    scores: Uint8Array.from(scores),
  };
}

it("decodes the API's u8 grid", () => {
  const values = [250, 200, 0, 255];
  const read = {
    cellAreaM2: 99.5,
    grid: {
      west: -120.05,
      south: 34.69,
      east: -120.03,
      north: 34.7,
      columns: 2,
      rows: 2,
      cellM: 10,
      encoding: "u8",
      scale: 0.004,
      offset: 0,
      nodata: 255,
      data: btoa(String.fromCharCode(...values)),
    },
  } as Pick<SignatureSimilarity, "grid" | "cellAreaM2">;
  const decoded = decodeScores(read);
  expect([...decoded.scores]).toEqual(values);
  expect(decoded.cellAreaM2).toBe(99.5);
  expect(decoded.columns).toBe(2);
});

it("lights a cell when its quantized score clears the slider, as the server and the GPU do", () => {
  expect(rawThreshold(0.8)).toBe(200);
  expect(rawThreshold(0.81)).toBe(203);
  expect(rawThreshold(0.3)).toBe(75);
  for (const threshold of [0.3, 0.55, 0.8, 0.81, 0.99]) {
    const raw = rawThreshold(threshold);
    const edge = rampThreshold(threshold);
    for (let score = 0; score <= 250; score++) {
      // The shader's step(threshold, texel) with the texel at score / 255.
      expect(score / 255 >= edge).toBe(isLit(score, raw));
    }
  }
  expect(isLit(NO_SCORE, 0)).toBe(false);
});

it("tallies the lit cells and acres, protected ones apart", () => {
  const g = grid([250, 210, 160, _, 200, 199, 150, 0, _, 250, 240, 100]);
  expect(litStats(g, 0.8, null)).toMatchObject({ cells: 5, protectedCells: 0 });
  expect(litStats(g, 0.6, null).cells).toBe(8);
  expect(litStats(g, 0.8, null).acres).toBeCloseTo((5 * 100) / 4046.8564224, 6);
  const guarded = new Uint8Array(12);
  guarded[0] = 1;
  guarded[1] = 1;
  const stats = litStats(g, 0.8, guarded);
  expect(stats).toMatchObject({ cells: 3, protectedCells: 2 });
  expect(stats.protectedAcres).toBeCloseTo((2 * 100) / 4046.8564224, 6);
});

// Two lit patches at 80%, 5 x 3: the top-left pair with a diagonal step to (1, 2), and (2, 4)
// on its own; (2, 0) has no score.
const PATCHES = [250, 250, 0, 0, 0, 0, 0, 230, 0, 0, _, 0, 0, 0, 220];

it("floods the lit patch through diagonals, stopping at no-score and unlit cells", () => {
  const g = grid(PATCHES, 5);
  expect(patchAt(g, 0, 0.8).sort((a, b) => a - b)).toEqual([0, 1, 7]);
  expect(patchAt(g, 14, 0.8)).toEqual([14]);
  expect(patchAt(g, 2, 0.8)).toEqual([]);
  expect(patchAt(g, 10, 0.1)).toEqual([]);
  // Below the slider the diagonal step is unlit: the patch stops short of it.
  expect(patchAt(g, 0, 0.95).sort((a, b) => a - b)).toEqual([0, 1]);
});

it("protects a lit patch with one tap and frees it with another", () => {
  const g = grid(PATCHES, 5);
  const first = toggleProtected(g, null, 1, 0.8);
  expect(first?.change).toBe("protected");
  expect([...first!.mask]).toEqual([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
  // A second patch joins it; tapping the first again frees only the first.
  const second = toggleProtected(g, first!.mask, 14, 0.8);
  expect(second!.mask[14]).toBe(1);
  const freed = toggleProtected(g, second!.mask, 7, 0.8);
  expect(freed?.change).toBe("freed");
  expect([...freed!.mask]).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
  // Unlit ground is already out of the job: nothing to protect there.
  expect(toggleProtected(g, null, 2, 0.8)).toBeNull();
});

it("sends protected ground as row runs that cover exactly the protected cells", () => {
  const g = grid([1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0].map(() => 250));
  const guarded = Uint8Array.from([1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]);
  const rectangles = protectedRectangles(g, guarded);
  expect(rectangles).toHaveLength(3);
  const dlon = 0.001;
  const dlat = 0.001;
  const covered = new Uint8Array(12);
  for (const [west, south, east, north] of rectangles) {
    for (let row = 0; row < g.rows; row++) {
      for (let col = 0; col < g.columns; col++) {
        const lon = g.west + (col + 0.5) * dlon;
        const lat = g.north - (row + 0.5) * dlat;
        if (lon > west! && lon < east! && lat > south! && lat < north!)
          covered[row * g.columns + col]! += 1;
      }
    }
  }
  expect([...covered]).toEqual([...guarded]);
  expect(protectedRectangles(g, null)).toEqual([]);
});

it("packs the drape's texture: score, scored, protected", () => {
  const g = grid([250, 120, _, 0], 2);
  const guarded = Uint8Array.from([0, 1, 1, 0]);
  const image = rampImage(g, guarded);
  expect(image.width).toBe(2);
  expect(image.height).toBe(2);
  expect([...image.data]).toEqual([
    250, 255, 0, 255, 120, 255, 255, 255, 0, 0, 0, 255, 0, 255, 0, 255,
  ]);
});

it("says acres plainly", () => {
  expect(formatAcres(0.04)).toBe("<0.1");
  expect(formatAcres(6.24)).toBe("6.2");
  expect(formatAcres(78.8)).toBe("79");
  expect(formatAcres(1349.4)).toBe("1,349");
});

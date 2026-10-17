import { describe, expect, it } from "vitest";

import type { KnowledgeGrid, LonLatGrid } from "@twin/contracts";

import {
  cellAt,
  cellRing,
  decodeGrid,
  densityAt,
  densityImage,
  fogLifted,
  NO_RECORD,
  OUTSIDE,
  rawValues,
  seedOf,
} from "@/features/knowledge/grid";

const frame = { west: -120.05, south: 34.69, east: -120.03, north: 34.7, columns: 4, rows: 2 };

function base64(bytes: Uint8Array): string {
  let text = "";
  for (const byte of bytes) text += String.fromCharCode(byte);
  return btoa(text);
}

function lonLatGrid(encoding: "u8" | "u16", values: number[]): LonLatGrid {
  const bytes =
    encoding === "u8"
      ? Uint8Array.from(values)
      : (() => {
          const out = new Uint8Array(values.length * 2);
          const view = new DataView(out.buffer);
          values.forEach((v, i) => view.setUint16(i * 2, v, true));
          return out;
        })();
  return {
    ...frame,
    cellM: 20,
    encoding,
    scale: encoding === "u8" ? 0.004 : 1,
    offset: 0,
    nodata: encoding === "u8" ? 255 : 65535,
    data: base64(bytes),
  };
}

// Row 0 (north): outside, 250, 125, 0. Row 1 (south): 250, 250, outside, 39.
const density = [OUTSIDE, 250, 125, 0, 250, 250, OUTSIDE, 39];
const best = [NO_RECORD, NO_RECORD, 1, 0, NO_RECORD, NO_RECORD, NO_RECORD, 300];
const read = {
  density: lonLatGrid("u8", density),
  best: lonLatGrid("u16", best),
} as Pick<KnowledgeGrid, "density" | "best">;

describe("the fog's grid", () => {
  it("decodes the base64 density bytes and little-endian best indices", () => {
    const grid = decodeGrid(read);
    expect(Array.from(grid.density)).toEqual(density);
    expect(Array.from(grid.best)).toEqual(best);
    expect(grid.columns).toBe(4);
    expect(grid.cellM).toBe(20);
    expect(Array.from(rawValues(read.best))).toEqual(best);
  });

  it("finds the cell under a point with row 0 at the north, and nothing outside", () => {
    const dlon = (frame.east - frame.west) / 4;
    const dlat = (frame.north - frame.south) / 2;
    expect(cellAt(frame, frame.west + 1e-9, frame.north - 1e-9)).toEqual({ col: 0, row: 0 });
    expect(cellAt(frame, frame.east - 1e-9, frame.south + 1e-9)).toEqual({ col: 3, row: 1 });
    expect(cellAt(frame, frame.west + dlon * 2.5, frame.north - dlat * 0.5)).toEqual({
      col: 2,
      row: 0,
    });
    expect(cellAt(frame, frame.west + dlon * 1.01, frame.north - dlat * 1.01)).toEqual({
      col: 1,
      row: 1,
    });
    expect(cellAt(frame, frame.west - 1e-6, frame.north - 1e-6)).toBeNull();
    expect(cellAt(frame, frame.east + 1e-6, frame.south + 1e-6)).toBeNull();
    expect(cellAt(frame, frame.west + 1e-6, frame.north + 1e-6)).toBeNull();
    const ring = cellRing(frame, 2, 1);
    expect(ring).toHaveLength(5);
    expect(ring[0]).toEqual(ring[4]);
    expect(ring[0]?.longitude).toBeCloseTo(frame.west + 2 * dlon, 12);
    expect(ring[2]?.latitude).toBeCloseTo(frame.north - dlat, 12);
  });

  it("reads density in 0..1 and null outside the land", () => {
    const grid = decodeGrid(read);
    expect(densityAt(grid, 0, 0)).toBeNull();
    expect(densityAt(grid, 1, 0)).toBe(1);
    expect(densityAt(grid, 2, 0)).toBe(0.5);
    expect(densityAt(grid, 3, 1)).toBeCloseTo(0.156, 3);
  });

  it("paints the texture with row 0 at the north: R density, G inside", () => {
    const image = densityImage(decodeGrid(read));
    expect(image.width).toBe(4);
    expect(image.height).toBe(2);
    const px = (col: number, row: number) =>
      Array.from(image.data.slice((row * 4 + col) * 4, (row * 4 + col) * 4 + 4));
    expect(px(0, 0)).toEqual([0, 0, 0, 255]); // outside: G 0
    expect(px(1, 0)).toEqual([255, 255, 0, 255]); // full fog
    expect(px(2, 0)).toEqual([128, 255, 0, 255]);
    expect(px(3, 0)).toEqual([0, 255, 0, 255]); // inside, no fog left
    expect(px(3, 1)).toEqual([40, 255, 0, 255]);
  });

  it("says when fog lifted somewhere, not when it thickened or the grid changed", () => {
    const before = decodeGrid(read);
    const lifted = {
      ...before,
      density: Uint8Array.from(density.map((v, i) => (i === 1 ? 100 : v))),
    };
    const aged = { ...before, density: Uint8Array.from(density.map((v, i) => (i === 3 ? 10 : v))) };
    expect(fogLifted(before, lifted)).toBe(true);
    expect(fogLifted(before, aged)).toBe(false);
    expect(fogLifted(null, lifted)).toBe(false);
    expect(fogLifted({ ...before, columns: 5 }, lifted)).toBe(false);
    expect(seedOf("a")).toBe(seedOf("a"));
    expect(seedOf("a")).not.toBe(seedOf("b"));
  });
});

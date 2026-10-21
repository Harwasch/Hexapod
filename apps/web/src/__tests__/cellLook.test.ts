/**
 * The one colour code of Confidence and Data age (lib/cellLook.ts): its GLSL carries the
 * legend's own ramps, CesiumJS's splat primitive draws through it rather than a copy, and the
 * level rule a dedicated renderer applies per tile is CesiumJS's per-splat rule at one point.
 * That the GLSL draws what `cellViewColor` says, input for input, is checked on a real GPU in
 * e2e/confidenceViewScene.spec.ts.
 */
import { describe, expect, it } from "vitest";

import { cellViewGlsl } from "@/cesium/splatCellView";
import { encodeConfidenceGrid, REASON_BITS } from "@/dev/cellGridFixture";
import { cellLevelFor, cellTexelLayout, decodeCellGrid } from "@/lib/cellGrid";
import {
  CELL_LOOK_GLSL,
  CELL_MIN_PX,
  cellLevelAgreement,
  cellLevelIndex,
  cellRuleOf,
  cellTexelFor,
  cellTexelOn,
  cellTileUniforms,
  pxPerRadianOf,
} from "@/lib/cellLook";
import { AGE_RAMP, CONFIDENCE_RAMP } from "@/lib/cellRamps";

/** The yard's 1 m grid with a 2 m margin, as the e2e routes serve it. */
async function grid() {
  const { meta, planes } = await encodeConfidenceGrid({
    level: 2,
    i0: -2,
    j0: -2,
    ni: 36,
    nj: 28,
    cell: (i) => (i < 16 ? { score: 60, reasons: REASON_BITS.views } : { score: 240 }),
    coarser: 8,
  });
  return decodeCellGrid(meta, planes);
}

describe("the GLSL", () => {
  it("carries the legend's ramps, Confidence's then Data age's, as written in cellRamps", () => {
    const declared = /HEXAPOD_CELL_RAMP\[12\] = vec3\[12\]\(([^;]*)\);/.exec(CELL_LOOK_GLSL)?.[1];
    expect(declared).toBeDefined();
    const values = [...(declared ?? "").matchAll(/vec3\(([^)]*)\)/g)].map((m) =>
      (m[1] ?? "").split(",").map(Number),
    );
    expect(values).toEqual([...CONFIDENCE_RAMP, ...AGE_RAMP].map((c) => [...c]));
  });

  it("is what CesiumJS's splat primitive draws through, not a copy of it", () => {
    const cesium = cellViewGlsl();
    expect(cesium).toContain(CELL_LOOK_GLSL);
    expect(cesium).toContain("hexapodCellLook(mode, t, p.xy, cell, u_cellRule, color)");
    expect(cesium).toContain("hexapodCellTexel(u_cellGrid, p.xy, cell");
    // The ramps are the shared constants now, not a uniform of its own.
    expect(cesium).not.toContain("u_cellRamp");
  });
});

describe("the level a tile reads", () => {
  it("is CesiumJS's per-splat rule, at the tile's distance, within the grid's levels", async () => {
    const g = await grid();
    for (const distance of [0, 3, 10, 40, 160, 900, 50_000]) {
      const k = cellLevelIndex(g.meta, distance, 1000);
      expect(k).toBe(
        Math.min(g.meta.levels.length - 1, cellLevelFor(distance, 1000, 1, CELL_MIN_PX)),
      );
      expect(cellTexelFor(g.meta, 5, 5, distance, 1000).level).toBe(k);
    }
    // A model drawn twice its size, its distance in metres on the globe: one level finer.
    expect(cellLevelIndex(g.meta, 1600, 1000, CELL_MIN_PX, 2)).toBe(3);
    expect(cellLevelIndex(g.meta, 1600, 1000)).toBe(4);
  });

  it("is handed as the plane's extent, its cell size, its first texel and the mode", async () => {
    const g = await grid();
    const view = {
      mode: "confidence" as const,
      rows: { data: new Uint8Array(0), height: 1 },
      meta: g.meta,
    };
    const at = cellTileUniforms(view, 2);
    const l = g.meta.levels[2];
    expect(Array.from(at.level)).toEqual([l?.i0, l?.j0, l?.ni, l?.nj]);
    expect(Array.from(at.tile)).toEqual([1, 4, cellTexelLayout(g.meta).offsets[2], 2]);
    expect(Array.from(cellTileUniforms({ ...view, mode: "data-age" }, 0).tile)).toEqual([
      2, 1, 0, 0,
    ]);
    // No view: the photo.
    expect(Array.from(cellTileUniforms(null, 2).tile)).toEqual([0, 0, 0, 0]);
    expect(Array.from(cellRuleOf(g.meta))).toEqual([128, REASON_BITS.unseen, Math.fround(0.6), 0]);
    // The shader's lookup and the reference agree on the texel.
    const ci = Math.floor(20.5 / 4) - (l?.i0 ?? 0);
    const cj = Math.floor(3.2 / 4) - (l?.j0 ?? 0);
    expect(cellTexelOn(g.meta, 20.5, 3.2, 2)).toBe(
      (cellTexelLayout(g.meta).offsets[2] ?? 0) + cj * (l?.ni ?? 0) + ci,
    );
    expect(cellTexelOn(g.meta, -40, 3.2, 2)).toBeNull();
  });

  it("counts how many of a tile's splats CesiumJS would read at the same level", async () => {
    const g = await grid();
    const px = pxPerRadianOf(64, Math.PI / 3);
    // Splats 10 m to 70 m from the eye, along x.
    const positions = Float32Array.from(
      Array.from({ length: 61 }, (_, i) => [10 + i, 0, 0]).flat(),
    );
    const tileLevel = cellLevelIndex(g.meta, 40, px);
    const { splats, agree } = cellLevelAgreement(
      g.meta,
      positions,
      [0, 0, 0],
      [0, 0, 0],
      px,
      tileLevel,
    );
    expect(splats).toBe(61);
    const expected = Array.from({ length: 61 }, (_, i) =>
      cellLevelIndex(g.meta, 10 + i, px),
    ).filter((k) => k === tileLevel).length;
    expect(agree).toBe(expected);
    expect(agree).toBeGreaterThan(0);
    expect(agree).toBeLessThan(61);
  });
});

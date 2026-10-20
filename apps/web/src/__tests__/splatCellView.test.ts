/**
 * The Confidence / Data age colour part (cesium/splatCellView.ts): what it declares to the
 * patched engine, the frame it reads cells in, the level and texel each splat reads, the tint
 * each verdict gets, and that it joins and leaves the colour chain without the others noticing.
 * The GLSL itself compiles in a real engine in e2e/confidenceViewScene.spec.ts.
 */
import { describe, expect, it } from "vitest";

import {
  AGE_RAMP,
  CELL_MIN_PX,
  cellFromModel,
  cellTexelFor,
  cellViewColor,
  CONFIDENCE_RAMP,
  SplatCellView,
  type CellViewGpu,
  type CellViewPrimitive,
} from "@/cesium/splatCellView";
import { colorChainOf, type SplatColorPart } from "@/cesium/splatColor";
import { inCompanionSlots } from "@/cesium/splatSlots";
import type { SplatShaderBuilder } from "@/cesium/splatInternals";
import { encodeConfidenceGrid, REASON_BITS } from "@/dev/cellGridFixture";
import {
  CELL_TEXTURE_WIDTH,
  cellAt,
  cellLevelFor,
  confidenceTexels,
  decodeCellGrid,
  type CellGrid,
} from "@/lib/cellGrid";

const created: { width: number; height: number; data: Uint8Array }[] = [];
const gpu: CellViewGpu = {
  createRG: (_context, width, height, data) => {
    created.push({ width, height, data });
    let destroyed = false;
    return {
      copyFrom: () => undefined,
      isDestroyed: () => destroyed,
      destroy: () => {
        destroyed = true;
      },
    };
  },
  mat4: (values) => ({ mat4: [...values] }),
  vec4: (x, y, z, w) => ({ vec4: [x, y, z, w] }),
};

/** 1 m cells over x 0-32, y 0-24 (the yard): weak west of x = 16, strong east. */
async function yardGrid(): Promise<CellGrid> {
  const { meta, planes } = await encodeConfidenceGrid({
    level: 2,
    i0: 0,
    j0: 0,
    ni: 32,
    nj: 24,
    cell: (i) => (i < 16 ? { score: 60, reasons: REASON_BITS.views } : { score: 240 }),
    coarser: 8,
  });
  return decodeCellGrid(meta, planes);
}

function build(part: SplatCellView) {
  const out: string[] = [];
  const uniforms: Record<string, () => unknown> = {};
  const builder: SplatShaderBuilder = {
    addUniform: (type, name) => out.push(`uniform ${type} ${name};`),
    addVertexLines: (text) => out.push(...(typeof text === "string" ? [text] : text)),
  };
  part.addToShader(builder, uniforms, { fake: "context" });
  return { lines: out.join("\n"), uniforms };
}

const translation = (x: number, y: number, z: number) => [
  1,
  0,
  0,
  0,
  0,
  1,
  0,
  0,
  0,
  0,
  1,
  0,
  x,
  y,
  z,
  1,
];

function primitive(): CellViewPrimitive & { _rootTransform?: number[] } {
  return { vertexColor: undefined, isDestroyed: () => false };
}

describe("what the part declares", () => {
  it("defines the chain's function and every uniform it reads", async () => {
    const part = new SplatCellView(await yardGrid(), gpu, () => translation(0, 0, 0));
    const { lines, uniforms } = build(part);
    expect(part.colorFunction).toBe("splatCellViewColor");
    expect(lines).toContain("vec4 splatCellViewColor(uint splatIndex, vec3 position, vec4 color)");
    for (const name of [
      "u_cellGrid",
      "u_cellFromModel",
      "u_cellLevels",
      "u_cellOffsets",
      "u_cellParams",
      "u_cellRule",
      "u_cellRamp",
      "u_cellFills",
    ]) {
      expect(lines).toContain(name);
      expect(uniforms[name]).toBeTypeOf("function");
    }
    // Per level (i0, j0, ni, nj): 32 x 24 at 1 m, then 16 x 12, ... down to 1 x 1.
    const levels = uniforms.u_cellLevels?.() as { vec4: number[] }[];
    expect(levels.slice(0, 3).map((v) => v.vec4)).toEqual([
      [0, 0, 32, 24],
      [0, 0, 16, 12],
      [0, 0, 8, 6],
    ]);
    const offsets = uniforms.u_cellOffsets?.() as { vec4: number[] }[];
    expect(offsets[0]?.vec4).toEqual([0, 768, 960, 1008]);
    expect(uniforms.u_cellRule?.()).toEqual({ vec4: [128, REASON_BITS.unseen, 0.6, 1] });
  });

  it("uploads (score, reasons) per cell, every level, in whole texture rows", async () => {
    const grid = await yardGrid();
    const part = new SplatCellView(grid, gpu, () => translation(0, 0, 0));
    const { uniforms } = build(part);
    uniforms.u_cellGrid?.();
    const texture = created.at(-1);
    expect(texture?.width).toBe(CELL_TEXTURE_WIDTH);
    expect(texture?.height).toBe(1);
    const texels = confidenceTexels(grid);
    expect(Array.from(texture?.data.subarray(0, texels.length) ?? [])).toEqual(Array.from(texels));
    expect(Array.from(texels.subarray(0, 2))).toEqual([60, REASON_BITS.views]);
  });
});

describe("the frame cells are read in", () => {
  it("is the root's frame: T⁻¹ · R, whatever frame the engine baked into", () => {
    // The root placed 100 m east of the bake frame's origin, turned 90° about up.
    const root = [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 100, 0, 0, 1];
    const bake = translation(0, 0, 0);
    const m = cellFromModel(root, bake);
    // The bake frame's origin is 100 m west of the root, i.e. at root-frame y = +100.
    const p = [0, 0, 0, 1];
    const out = [0, 1, 2].map((r) => p.reduce((s, v, c) => s + (m?.[c * 4 + r] ?? 0) * v, 0));
    out.forEach((v, i) => expect(v).toBeCloseTo([0, 100, 0][i] ?? Number.NaN, 9));
    expect(cellFromModel([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], bake)).toBeUndefined();
  });

  it("draws the photo until both frames are known, then reads cells through them", async () => {
    let root: number[] | undefined;
    const part = new SplatCellView(await yardGrid(), gpu, () => root);
    const { uniforms } = build(part);
    const target = primitive();
    expect(part.install(target)).toBe(true);
    const mode = () => (uniforms.u_cellParams?.() as { vec4: number[] }).vec4[0];
    expect(mode()).toBe(0);
    root = translation(1000, 2000, 0);
    expect(mode()).toBe(0);
    target._rootTransform = translation(1010, 2005, 3);
    expect(mode()).toBe(1);
    const value = uniforms.u_cellFromModel?.() as { mat4: number[] };
    expect(value.mat4.slice(12, 15)).toEqual([10, 5, 3]);
    // Cached while neither moves; rebuilt when the placement does.
    expect(uniforms.u_cellFromModel?.()).toBe(value);
    root = translation(1000, 2001, 0);
    expect((uniforms.u_cellFromModel?.() as { mat4: number[] }).mat4.slice(12, 15)).toEqual([
      10, 4, 3,
    ]);
    const params = uniforms.u_cellParams?.() as { vec4: number[] };
    expect(params.vec4).toEqual([1, 1, CELL_MIN_PX, 6]);
  });
});

describe("a model drawn at a runtime scale", () => {
  it("picks levels by the cell's size on the globe, not in the grid's metres", async () => {
    const grid = await yardGrid();
    // renderConfig.scale 2: the root's computed transform doubles lengths (tilesetScale.ts).
    let root = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1];
    const part = new SplatCellView(grid, gpu, () => root);
    const { uniforms } = build(part);
    const target = primitive();
    target._rootTransform = translation(0, 0, 0);
    part.install(target);
    const rule = () => (uniforms.u_cellRule?.() as { vec4: number[] }).vec4;
    expect(rule()[3]).toBeCloseTo(2, 12);
    root = translation(0, 0, 0);
    expect(rule()[3]).toBeCloseTo(1, 12);
    // The reference: at 1600 m and 1000 px a radian, a 1 m cell needs level 4 (6 px); drawn
    // twice as large, level 3. The cell indexed is still the grid's own.
    expect(cellTexelFor(grid.meta, 20, 10, 1600, 1000).level).toBe(4);
    const scaled = cellTexelFor(grid.meta, 20, 10, 1600, 1000, CELL_MIN_PX, 2);
    expect(scaled.level).toBe(3);
    const cell = cellAt(grid, 20, 10, grid.meta.levels[3]?.level ?? 0);
    expect(confidenceTexels(grid)[(scaled.texel ?? 0) * 2]).toBe(cell?.score);
  });
});

describe("the cell a splat reads", () => {
  it("is the finest level legible at its distance, at the texel the grid's layout puts it", async () => {
    const grid = await yardGrid();
    const texels = confidenceTexels(grid);
    for (const [x, y, distance] of [
      [3.5, 2.5, 10],
      [20.2, 13.9, 10],
      [31.99, 23.99, 400],
      [0.01, 0.01, 3000],
      [17, 9, 50_000],
    ] as const) {
      const at = cellTexelFor(grid.meta, x, y, distance, 1000);
      expect(at.level).toBe(Math.min(5, cellLevelFor(distance, 1000, 1, CELL_MIN_PX)));
      const level = grid.meta.levels[at.level]?.level ?? -1;
      const cell = cellAt(grid, x, y, level);
      expect(at.texel).not.toBeNull();
      expect(texels[(at.texel ?? 0) * 2]).toBe(cell?.score);
      expect(texels[(at.texel ?? 0) * 2 + 1]).toBe(cell?.reasons);
    }
    // Outside the grid, no texel: the splat is drawn as "no evidence".
    expect(cellTexelFor(grid.meta, -0.5, 3, 10, 1000).texel).toBeNull();
    expect(cellTexelFor(grid.meta, 32.5, 3, 10, 1000).texel).toBeNull();
  });
});

/** A white splat's tint is the ramp's colour (its luminance, and so its shade, is 1). */
function close(actual: readonly number[], expected: readonly number[] | undefined): void {
  expect(expected).toBeDefined();
  expected?.forEach((v, i) => expect(actual[i]).toBeCloseTo(v, 9));
}

describe("the tint", () => {
  const leaf = [0.3, 0.5, 0.2, 0.8] as const;
  const red = (c: number[]) => (c[0] ?? 0) > 2 * Math.max(c[1] ?? 0, c[2] ?? 0);
  const green = (c: number[]) => (c[1] ?? 0) > 1.5 * Math.max(c[0] ?? 0, c[2] ?? 0);

  it("is red where weak or unseen, green where strong, grey where nothing was measured", () => {
    expect(red(cellViewColor("confidence", [60, REASON_BITS.views], leaf))).toBe(true);
    expect(red(cellViewColor("confidence", [0, REASON_BITS.unseen], leaf))).toBe(true);
    expect(red(cellViewColor("confidence", [250, REASON_BITS.unseen], leaf))).toBe(true);
    expect(red(cellViewColor("confidence", [127, 0], leaf))).toBe(true);
    expect(red(cellViewColor("confidence", [128, 0], leaf))).toBe(false);
    expect(green(cellViewColor("confidence", [250, 0], leaf))).toBe(true);
    const none = cellViewColor("confidence", [0, 0], leaf);
    expect(none[0]).toBeCloseTo(none[1] ?? Number.NaN, 12);
    expect(none[1]).toBeCloseTo(none[2] ?? Number.NaN, 12);
    expect(cellViewColor("confidence", null, leaf)).toEqual(none);
    // An inferred layer's splat has no evidence, whatever its cell says.
    expect(cellViewColor("confidence", [250, 0], leaf, { fill: true })).toEqual(none);
    // Opacity is the splat's own.
    expect(cellViewColor("confidence", [60, 1], leaf)[3]).toBe(0.8);
  });

  it("steps from amber to green above the threshold", () => {
    const tintOf = (score: number) => cellViewColor("confidence", [score, 0], [1, 1, 1, 1]);
    close(tintOf(128), CONFIDENCE_RAMP[2]);
    close(tintOf(160), CONFIDENCE_RAMP[3]);
    close(tintOf(192), CONFIDENCE_RAMP[4]);
    close(tintOf(255), CONFIDENCE_RAMP[5]);
  });

  it("hatches weak cells and dots fair ones, so each reads without its colour", () => {
    const white = [1, 1, 1, 1] as const;
    const at = (score: number, p: readonly [number, number]) =>
      cellViewColor("confidence", [score, 0], white, { p, cell: 1 });
    // Weak: dark diagonal bands, a band pair every two cells.
    expect(at(60, [0.2, 0])[0]).toBeGreaterThan(at(60, [1.2, 0])[0]);
    expect(at(60, [0.2, 0])).toEqual(at(60, [2.2, 0]));
    // Fair: a dark dot every two cells, the rest plain.
    expect(at(140, [1.5, 1.5])[0]).toBeLessThan(at(140, [0.5, 0.5])[0]);
    expect(at(140, [1.5, 0.5])).toEqual(at(140, [0.5, 0.5]));
    // Strong: solid.
    expect(at(250, [1.5, 1.5])).toEqual(at(250, [0.5, 0.5]));
    expect(at(250, [1.2, 0])).toEqual(at(250, [0.2, 0]));
    // Still the weak colour where darkened: red, not some other class's colour.
    const dark = at(60, [1.2, 0]);
    expect(dark[0]).toBeGreaterThan(2 * Math.max(dark[1], dark[2]));
  });

  it("colours data age by bucket and stripes where a later date has evidence", () => {
    const white = [1, 1, 1, 1] as const;
    close(cellViewColor("data-age", [1, 0], white), AGE_RAMP[0]);
    close(cellViewColor("data-age", [6, 0], white), AGE_RAMP[5]);
    const plain = cellViewColor("data-age", [3, 0], white, { p: [0.1, 0] });
    const striped = cellViewColor("data-age", [3, 255], white, { p: [0.1, 0] });
    const between = cellViewColor("data-age", [3, 255], white, { p: [0.4, 0] });
    expect(striped[0]).toBeGreaterThan(plain[0]);
    expect(between).toEqual(plain);
    // Not observed by the shown date: grey.
    const grey = cellViewColor("data-age", [0, 0], white);
    expect(grey[0]).toBeCloseTo(grey[2] ?? Number.NaN, 12);
  });
});

describe("in the colour chain", () => {
  it("runs below the inferred highlight, leaves the slot as it found it, and swaps modes by uniform", async () => {
    const part = new SplatCellView(await yardGrid(), gpu, () => translation(0, 0, 0));
    const highlight: SplatColorPart = {
      colorFunction: "splatInferredColor",
      colorOrder: 10,
      addToShader: () => undefined,
    };
    const target = primitive();
    target._rootTransform = translation(0, 0, 0);
    expect(part.install(target)).toBe(true);
    colorChainOf(target)?.add(highlight);
    expect(colorChainOf(target)?.parts).toEqual([part, highlight]);
    const { uniforms } = build(part);
    const mode = () => (uniforms.u_cellParams?.() as { vec4: number[] }).vec4[0];
    expect(mode()).toBe(1);
    // Data age without its bytes yet draws the photo, not the confidence texture.
    part.mode = "data-age";
    expect(mode()).toBe(0);
    part.setDataAge(new Uint8Array([3, 255]));
    expect(mode()).toBe(2);
    expect(uniforms.u_cellRamp?.()).toHaveLength(6);
    uniforms.u_cellGrid?.();
    expect(Array.from(created.at(-1)?.data.subarray(0, 2) ?? [])).toEqual([3, 255]);
    part.uninstall();
    expect(colorChainOf(target)?.parts).toEqual([highlight]);
    expect(part.installed).toBe(false);
  });

  it("greys the slots of the layers the scan's primitive draws in its sort", async () => {
    const part = new SplatCellView(await yardGrid(), gpu, () => translation(0, 0, 0));
    const scan = {};
    const layer = {};
    const target = primitive();
    Object.assign(target, {
      _tileset: scan,
      companions: [layer],
      _tileSlots: new Map([
        [{ tileset: scan }, { start: 0, count: 100 }],
        [{ tileset: layer }, { start: 100, count: 40 }],
      ]),
    });
    part.install(target);
    const { uniforms } = build(part);
    const fills = uniforms.u_cellFills?.() as { vec4: number[] }[];
    expect(fills[0]?.vec4).toEqual([100, 140, 0, 0]);
    // Without companions, nothing is greyed.
    Object.assign(target, { companions: [], _tileSlots: new Map(target._tileSlots) });
    expect((uniforms.u_cellFills?.() as { vec4: number[] }[])[0]?.vec4).toEqual([0, 0, 0, 0]);
  });

  it("greys every splat of a layer with more ranges than the uniform holds, never colours one", async () => {
    const part = new SplatCellView(await yardGrid(), gpu, () => translation(0, 0, 0));
    const scan = {};
    const layer = {};
    const target = primitive();
    // Nine disjoint layer ranges (100-110, 200-210, ... 900-910) between the scan's tiles: one
    // more than the eight the shader can tell apart.
    const slots = new Map<{ tileset: object }, { start: number; count: number }>();
    for (let k = 0; k < 10; k += 1) slots.set({ tileset: scan }, { start: k * 100, count: 100 });
    for (let k = 1; k <= 9; k += 1) slots.set({ tileset: layer }, { start: k * 100, count: 10 });
    Object.assign(target, { _tileset: scan, companions: [layer], _tileSlots: slots });
    part.install(target);
    const { uniforms } = build(part);
    const vecs = (uniforms.u_cellFills?.() as { vec4: [number, number, number, number] }[]).map(
      (v) => v.vec4,
    );
    // Every inferred splat is grey, the ninth range's too: no measured colour on inferred
    // content. What that costs is scan splats past the eighth range drawn grey (no evidence
    // claimed), never the other way round.
    for (let k = 1; k <= 9; k += 1) {
      for (const index of [k * 100, k * 100 + 9]) expect(inCompanionSlots(vecs, index)).toBe(true);
    }
    expect(inCompanionSlots(vecs, 50)).toBe(false);
    expect(inCompanionSlots(vecs, 150)).toBe(false);
  });

  it("under a view rule greys only the inferred layers, not the other dates it draws", async () => {
    // A rule's host draws another date and that date's inferred layer as companions; Data
    // age's part is told which are inferred (`fills`): the date is coloured by its age.
    const scan = {};
    const date = {};
    const layer = {};
    const inferred = new Set<unknown>([layer]);
    const part = new SplatCellView(
      await yardGrid(),
      gpu,
      () => translation(0, 0, 0),
      () => inferred,
    );
    const target = primitive();
    Object.assign(target, {
      _tileset: scan,
      companions: [date, layer],
      _tileSlots: new Map([
        [{ tileset: scan }, { start: 0, count: 100 }],
        [{ tileset: date }, { start: 100, count: 50 }],
        [{ tileset: layer }, { start: 150, count: 20 }],
      ]),
    });
    part.install(target);
    const { uniforms } = build(part);
    const fills = (uniforms.u_cellFills?.() as { vec4: number[] }[]).map((v) => v.vec4);
    expect(fills[0]).toEqual([150, 170, 0, 0]);
    expect(fills.slice(1).every((v) => v.every((n) => n === 0))).toBe(true);
  });

  it("is one a primitive: a second cell view is refused, and the first stays", async () => {
    // Two would declare the same uniforms (a date's own view and a rule's Data age).
    const first = new SplatCellView(await yardGrid(), gpu, () => translation(0, 0, 0));
    const second = new SplatCellView(await yardGrid(), gpu, () => translation(0, 0, 0));
    const target = primitive();
    expect(first.install(target)).toBe(true);
    expect(second.install(target)).toBe(false);
    expect(second.installed).toBe(false);
    expect(colorChainOf(target)?.parts).toEqual([first]);
    first.uninstall();
    expect(second.install(target)).toBe(true);
  });

  it("is refused by an engine without the colour slot", async () => {
    const part = new SplatCellView(await yardGrid(), gpu, () => translation(0, 0, 0));
    expect(part.install({ isDestroyed: () => false })).toBe(false);
  });
});

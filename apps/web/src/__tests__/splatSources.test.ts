/**
 * The per-slot source texture (cesium/splatSources.ts): each slot of an incremental splat
 * primitive says which listed tileset it holds, rewritten from the patch's `_tileSlots` as
 * tiles come and go, only the rows touched uploaded; a reset of the slots (a model-matrix
 * change, the texture growing) is just every tile having moved.
 */
import { describe, expect, it } from "vitest";

import {
  NO_SOURCE,
  SOURCE_TEXTURE_WIDTH,
  SplatSources,
  sourceGlsl,
  type SourceTexture,
} from "@/cesium/splatSources";

interface Upload {
  height: number;
  yOffset: number;
}

function gpu() {
  const made: { width: number; height: number; data: Uint8Array }[] = [];
  const uploads: Upload[] = [];
  return {
    made,
    uploads,
    createR8(_context: unknown, width: number, height: number, data: Uint8Array): SourceTexture {
      made.push({ width, height, data: data.slice() });
      let destroyed = false;
      return {
        copyFrom: ({ source, yOffset }) =>
          void uploads.push({ height: source.height, yOffset: yOffset ?? 0 }),
        isDestroyed: () => destroyed,
        destroy: () => {
          destroyed = true;
        },
      };
    },
  };
}

const host = { name: "host" };
const later = { name: "later date" };
const layer = { name: "inferred layer" };

type Slots = Map<{ tileset: unknown }, { start: number; count: number }>;
const primitive = (entries: [unknown, number, number][]) => ({
  _tileSlots: new Map(entries.map(([tileset, start, count]) => [{ tileset }, { start, count }])),
});

describe("each slot's source", () => {
  it("writes each tile's range with its tileset's index, and NO_SOURCE elsewhere", () => {
    const sources = new SplatSources(gpu());
    sources.setSources([host, later, layer]);
    expect(
      sources.sync(
        primitive([
          [host, 0, 100],
          [later, 150, 50],
          [layer, 100, 20],
        ]),
      ),
    ).toBe(true);
    expect(sources.sourceAt(0)).toBe(0);
    expect(sources.sourceAt(99)).toBe(0);
    expect(sources.sourceAt(100)).toBe(2);
    expect(sources.sourceAt(119)).toBe(2);
    // A gap between ranges holds nothing.
    expect(sources.sourceAt(120)).toBe(NO_SOURCE);
    expect(sources.sourceAt(150)).toBe(1);
    expect(sources.sourceAt(199)).toBe(1);
    expect(sources.covered).toBe(200);
    expect(sources.sourceAt(200)).toBe(NO_SOURCE);
    // A tileset not listed holds no source.
    const other = new SplatSources(gpu());
    other.setSources([host]);
    other.sync(primitive([[later, 0, 10]]));
    expect(other.sourceAt(5)).toBe(NO_SOURCE);
  });

  it("does nothing while the slot map is the same object", () => {
    const sources = new SplatSources(gpu());
    sources.setSources([host]);
    const p = primitive([[host, 0, 10]]);
    expect(sources.sync(p)).toBe(true);
    const version = sources.version;
    expect(sources.sync(p)).toBe(false);
    expect(sources.version).toBe(version);
  });

  it("clears a range a tile left and writes the one that took it, in the same batch", () => {
    const sources = new SplatSources(gpu());
    sources.setSources([host, later]);
    const a = { tileset: host };
    const b = { tileset: later };
    const p: { _tileSlots: Slots } = {
      _tileSlots: new Map([[a, { start: 0, count: 10 }]]),
    };
    sources.sync(p);
    // `a` leaves, `b` takes its slot range (the patch reuses freed ranges).
    p._tileSlots = new Map([[b, { start: 0, count: 10 }]]);
    sources.sync(p);
    expect(sources.sourceAt(0)).toBe(1);
    expect(sources.sourceAt(9)).toBe(1);
    // `b` moves away and nothing replaces it.
    p._tileSlots = new Map([[b, { start: 20, count: 10 }]]);
    sources.sync(p);
    expect(sources.sourceAt(0)).toBe(NO_SOURCE);
    expect(sources.sourceAt(25)).toBe(1);
  });

  it("starts over when the slots reset (model matrix, texture growth)", () => {
    const sources = new SplatSources(gpu());
    sources.setSources([host]);
    const p: { _tileSlots: Slots | undefined } = primitive([[host, 0, 30]]);
    sources.sync(p);
    p._tileSlots = undefined;
    sources.sync(p);
    expect(sources.sourceAt(0)).toBe(NO_SOURCE);
    expect(sources.covered).toBe(0);
    p._tileSlots = primitive([[host, 5, 10]])._tileSlots;
    sources.sync(p);
    expect(sources.sourceAt(4)).toBe(NO_SOURCE);
    expect(sources.sourceAt(5)).toBe(0);
  });

  it("writes every slot again when the list of sources changes", () => {
    const sources = new SplatSources(gpu());
    sources.setSources([host, later]);
    const p = primitive([
      [host, 0, 10],
      [later, 10, 10],
    ]);
    sources.sync(p);
    expect(sources.setSources([later, host])).toBe(true);
    expect(sources.setSources([later, host])).toBe(false);
    expect(sources.sync(p)).toBe(true);
    expect(sources.sourceAt(0)).toBe(1);
    expect(sources.sourceAt(10)).toBe(0);
  });

  it("tells sixteen sources apart; the rest hold none", () => {
    const sources = new SplatSources(gpu());
    const many = Array.from({ length: 18 }, (_, i) => ({ i }));
    sources.setSources(many);
    sources.sync(primitive(many.map((t, i) => [t, i * 2, 2])));
    expect(sources.sourceAt(30)).toBe(15);
    expect(sources.sourceAt(32)).toBe(NO_SOURCE);
  });

  it("uploads only the rows touched, and makes the texture again when it grows", () => {
    const g = gpu();
    const sources = new SplatSources(g);
    sources.setSources([host, later]);
    const row = SOURCE_TEXTURE_WIDTH;
    // The engine keeps a tile's object while it stays: the same keys, batch after batch.
    const a = { tileset: host };
    const b = { tileset: later };
    const p: { _tileSlots: Slots } = { _tileSlots: new Map([[a, { start: 0, count: 10 }]]) };
    sources.sync(p);
    sources.texture({});
    expect(g.made).toHaveLength(1);
    expect(g.made[0]?.height).toBe(1);
    // Grows past one row: a new texture, whole.
    p._tileSlots = new Map([
      [a, { start: 0, count: 10 }],
      [b, { start: row * 2 + 5, count: 100 }],
    ]);
    sources.sync(p);
    sources.texture({});
    expect(g.made).toHaveLength(2);
    expect(g.made[1]?.height).toBe(3);
    expect(g.made[1]?.data[row * 2 + 5]).toBe(1);
    expect(g.uploads).toEqual([]);
    // A change in the third row alone uploads the third row alone.
    p._tileSlots = new Map([
      [a, { start: 0, count: 10 }],
      [b, { start: row * 2 + 50, count: 100 }],
    ]);
    sources.sync(p);
    expect(sources.dirtyRows).toEqual([2, 2]);
    sources.texture({});
    expect(g.uploads).toEqual([{ height: 1, yOffset: 2 }]);
    expect(sources.dirtyRows).toBeUndefined();
  });

  it("reads its byte in GLSL as an integer, and nothing past what it covers", () => {
    const glsl = sourceGlsl("splatSource", "u_sources", "u_params.w");
    expect(glsl).toContain("int splatSource(uint splatIndex)");
    expect(glsl).toContain(`return ${String(NO_SOURCE)};`);
    expect(glsl).toContain(`i & ${String(SOURCE_TEXTURE_WIDTH - 1)}`);
    expect(glsl).toContain("* 255.0 + 0.5");
  });
});

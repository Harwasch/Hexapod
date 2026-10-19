/**
 * "More like this" below the card (lib/instances.ts `loadEmbeddings`, `describedMatrix`,
 * `rankSimilar`; state/instances.ts `showSimilar`, `similarSummary`;
 * cesium/splatInstances.ts `embeddingOf`): the scan's object embeddings read once, centred so
 * that the cosines tell objects apart, ranked against the chosen object without what holds it
 * or is part of it, and lit through the objects store at a threshold.
 */

import type { Cesium3DTileset, Scene } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { attachInstances, embeddingOf, embeddingUrlOf } from "@/cesium/splatInstances";
import type { InstanceGpu } from "@/cesium/splatInstances";
import {
  describedMatrix,
  embeddingModelName,
  formatBytes,
  halfToFloat,
  loadEmbeddings,
  parseInstances,
  rankByVector,
  rankSimilar,
  type InstancesDoc,
} from "@/lib/instances";
import { scanPayloads } from "@/lib/payloadCache";
import {
  DEFAULT_SIMILAR_THRESHOLD,
  similarMatches,
  similarSummary,
  useInstances,
} from "@/state/instances";

/** A number as IEEE 754 half bits (round to nearest), for writing an `instances.emb`. */
function toHalf(value: number): number {
  const bits = new Uint32Array(new Float32Array([value]).buffer)[0] ?? 0;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  const mantissa = bits & 0x7fffff;
  if (exponent <= 0) return sign;
  if (exponent >= 31) return sign | 0x7c00;
  // `+`, not `|`: a mantissa that rounds up carries into the exponent.
  return sign | ((exponent << 10) + ((mantissa + 0x1000) >> 13));
}

const DIM = 4;

/**
 * Rows of the synthetic scan, by id (row `id - 1`): every vector shares a large first
 * component, as every crop of one scan shares its light and lens, so all raw cosines are
 * above 0.9. Ids 5, 8, 9 and 10 were never described (zero rows).
 */
const VECTORS: Record<number, number[]> = {
  1: [4, 1, 0, 0], // shrub A, the seed
  2: [4, 0.9, 0.2, 0], // shrub B: its neighbour
  3: [4, 0, 0, 1], // rock
  4: [4, -0.3, 1, 0.2], // conifer
  6: [4, 0.3, 0.3, 0.3], // the forest floor shrub A stands in
  7: [4, 1, 0.1, 0], // a part of shrub A: closest of all, and not a match
};

const box = { min: [0, 0, 0], max: [1, 1, 1] };
const tagged = (id: number, label: string, parent: number | null = null) => ({
  id,
  parent,
  bounds: box,
  splats: 100,
  tags: [{ label, score: 0.6 }],
});
const untagged = (id: number, parent: number | null = null) => ({
  id,
  parent,
  bounds: box,
  splats: 10,
  tags: [],
});

const DOC: InstancesDoc = parseInstances({
  format: "hexapod.instances",
  version: 1,
  instances: [
    tagged(1, "shrub", 6),
    tagged(2, "shrub"),
    tagged(3, "rock"),
    tagged(4, "conifer"),
    untagged(5),
    tagged(6, "forest floor"),
    tagged(7, "bush", 1),
    untagged(8, 1),
    untagged(9),
    untagged(10, 9),
  ],
  tiles: {},
  embedding: { file: "instances.emb", model: "siglip:google/siglip2-base-patch16-224", dim: DIM },
})!;

/** The file's bytes: 10 rows of `DIM` halves, each described row unit length. */
function embeddings(rows = 10): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(rows * DIM);
  for (const [id, vector] of Object.entries(VECTORS)) {
    const norm = Math.hypot(...vector);
    vector.forEach((v, j) => (out[(Number(id) - 1) * DIM + j] = toHalf(v / norm)));
  }
  return out;
}

function rawCosine(raw: Uint16Array, a: number, b: number): number {
  let dot = 0;
  for (let j = 0; j < DIM; j++)
    dot += halfToFloat(raw[(a - 1) * DIM + j] ?? 0) * halfToFloat(raw[(b - 1) * DIM + j] ?? 0);
  return dot;
}

describe("the centred embeddings", () => {
  it("spreads cosines that raw all sit above 0.9, the intended neighbour first", () => {
    const raw = embeddings();
    const described = Object.keys(VECTORS).map(Number);
    for (const a of described)
      for (const b of described) expect(rawCosine(raw, a, b)).toBeGreaterThan(0.9);
    const matrix = describedMatrix(raw, DIM);
    const ranked = rankSimilar(DOC, matrix, 1);
    expect(ranked?.seedUsed).toBe(1);
    const scores = [...(ranked?.scores ?? [])].sort((a, b) => b[1] - a[1]);
    expect(scores[0]?.[0]).toBe(2);
    expect(scores[0]?.[1]).toBeGreaterThan(0.9);
    // From near 1 down to well below 0: what raw cosine squeezed into 0.9..1.
    expect(Math.min(...scores.map(([, s]) => s))).toBeLessThan(-0.5);
  });

  it("keeps only described rows, each unit length, and is made once per file", () => {
    const raw = embeddings();
    const matrix = describedMatrix(raw, DIM);
    expect([...matrix.ids]).toEqual([1, 2, 3, 4, 6, 7]);
    for (let r = 0; r < matrix.ids.length; r++) {
      const row = matrix.m.subarray(r * DIM, (r + 1) * DIM);
      expect(Math.hypot(...row)).toBeCloseTo(1, 5);
    }
    expect(matrix.rowOf.get(5)).toBeUndefined();
    expect(describedMatrix(raw, DIM)).toBe(matrix);
    const scores = rankSimilar(DOC, matrix, 2)?.scores;
    for (const id of [5, 8, 9, 10]) expect(scores?.has(id)).toBe(false);
  });

  it("scores neither what holds the seed nor what is part of it", () => {
    const scores = rankSimilar(DOC, describedMatrix(embeddings(), DIM), 1)?.scores;
    // 6 holds it, 7 is part of it (and closer than any other object).
    expect(scores?.has(6)).toBe(false);
    expect(scores?.has(7)).toBe(false);
    expect(scores?.has(1)).toBe(false);
    expect([...(scores?.keys() ?? [])].sort()).toEqual([2, 3, 4]);
  });

  it("compares an undescribed part by the nearest described instance it is part of", () => {
    const matrix = describedMatrix(embeddings(), DIM);
    const ranked = rankSimilar(DOC, matrix, 8);
    expect(ranked?.seedUsed).toBe(1);
    expect(ranked?.scores.get(2)).toBeCloseTo(rankSimilar(DOC, matrix, 1)?.scores.get(2) ?? 0, 6);
    // Nothing above 10 was described either.
    expect(rankSimilar(DOC, matrix, 10)).toBeNull();
  });

  it("ranks by a vector over the decoded matrix", () => {
    const m = new Float32Array([1, 0, 0, 1, 0.6, 0.8]);
    expect([...rankByVector(new Float32Array([2, 0]), m, 2)].map((v) => +v.toFixed(3))).toEqual([
      1, 0, 0.6,
    ]);
    expect(() => rankByVector(new Float32Array(3), m, 2)).toThrow(/dimension/);
  });

  it("names the model and the download as a person reads them", () => {
    expect(embeddingModelName("siglip:google/siglip2-base-patch16-224")).toBe("SigLIP 2");
    expect(embeddingModelName("fake-colour-histogram")).toBe("fake-colour-histogram");
    expect(formatBytes(28_400 * 768 * 2)).toBe("43.6 MB");
    expect(formatBytes(4_000)).toBe("4 kB");
  });
});

describe("loadEmbeddings", () => {
  beforeEach(() => scanPayloads.clear());
  afterEach(() => vi.unstubAllGlobals());

  it("reads the file once, saying how far along it is", async () => {
    const raw = embeddings();
    const fetcher = vi.fn(() => Promise.resolve(new Response(raw.slice().buffer)));
    vi.stubGlobal("fetch", fetcher);
    const progress: number[] = [];
    const read = await loadEmbeddings("https://x.test/a/instances.emb", 10, DIM, (received) =>
      progress.push(received),
    );
    expect([...read]).toEqual([...raw]);
    expect(progress.at(-1)).toBe(raw.byteLength);
    await loadEmbeddings("https://x.test/a/instances.emb", 10, DIM);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("refuses a file whose size is not count × dim × 2, and a missing one", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(embeddings(9).buffer))),
    );
    await expect(loadEmbeddings("https://x.test/b/instances.emb", 10, DIM)).rejects.toThrow(
      /72 bytes, not 80/,
    );
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(embeddings(11).buffer))),
    );
    await expect(loadEmbeddings("https://x.test/c/instances.emb", 10, DIM)).rejects.toThrow(
      /88 bytes, not 80/,
    );
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response("gone", { status: 404 }))),
    );
    await expect(loadEmbeddings("https://x.test/d/instances.emb", 10, DIM)).rejects.toThrow(
      /answered 404/,
    );
    // A failed read is not kept: the next asks again.
    expect(scanPayloads.keys).toEqual([]);
  });
});

describe("the similar highlight in the objects store", () => {
  const ASSET = "scan";
  // Shrub B, a third shrub, the rock, the conifer, at falling cosines.
  const SCORES = new Map([
    [2, 0.92],
    [11, 0.7],
    [3, 0.5],
    [4, 0.35],
  ]);
  const doc = parseInstances({
    format: "hexapod.instances",
    version: 1,
    instances: [
      ...[1, 2, 3, 4, 6, 7].map((id) =>
        id === 1
          ? tagged(1, "shrub", 6)
          : id === 7
            ? tagged(7, "bush", 1)
            : tagged(id, { 2: "shrub", 3: "rock", 4: "conifer", 6: "forest floor" }[id] ?? ""),
      ),
      tagged(11, "bush"),
      tagged(12, "bush", 2),
    ],
    tiles: {},
  })!;

  beforeEach(() => {
    useInstances.setState({ assets: {}, dimOthers: true });
    useInstances.getState().setTable(ASSET, doc);
  });

  const lit = (): number[] => [...(useInstances.getState().assets[ASSET]?.highlighted ?? [])];

  it("lights the matches above the threshold, what was compared, and all below them", () => {
    useInstances.getState().showSimilar(ASSET, 1, 1, SCORES);
    const entry = useInstances.getState().assets[ASSET];
    expect(entry?.focus).toEqual({ kind: "similar", seedId: 1 });
    expect(entry?.similar?.threshold).toBe(DEFAULT_SIMILAR_THRESHOLD);
    // 2 with its part 12, 11, and the seed 1 with its part 7.
    expect(lit().sort((a, b) => a - b)).toEqual([1, 2, 7, 11, 12]);
    expect(similarMatches(entry!.similar!)).toEqual([2, 11]);
    expect(similarSummary(entry)).toMatchObject({
      objects: 2,
      categories: [{ category: { id: "shrubs" }, count: 2 }],
    });

    useInstances.getState().setSimilarThreshold(ASSET, 0.3);
    expect(lit().sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 7, 11, 12]);
    const wide = similarSummary(useInstances.getState().assets[ASSET]);
    expect(wide?.objects).toBe(4);
    // The most first; the rest one each.
    expect(wide?.categories[0]).toMatchObject({ category: { id: "shrubs" }, count: 2 });
    expect(wide?.categories.slice(1).map((c) => c.count)).toEqual([1, 1]);
    expect(wide?.categories.map((c) => c.category.id).sort()).toEqual(["rock", "shrubs", "trees"]);

    useInstances.getState().setSimilarThreshold(ASSET, 0.95);
    expect(lit().sort((a, b) => a - b)).toEqual([1, 7]);
    expect(similarSummary(useInstances.getState().assets[ASSET])?.objects).toBe(0);
  });

  it("does not count a match inside the object compared", () => {
    // 7 is part of shrub A's object: lit, but not "another object like it".
    useInstances.getState().showSimilar(ASSET, 1, 1, new Map([[7, 0.9]]));
    expect(similarSummary(useInstances.getState().assets[ASSET])?.objects).toBe(0);
  });

  it("gives way to any other highlight, and to reset", () => {
    const store = useInstances.getState();
    store.showSimilar(ASSET, 1, 1, SCORES);
    store.highlight(ASSET, [3]);
    expect(useInstances.getState().assets[ASSET]?.similar).toBeNull();
    expect(useInstances.getState().assets[ASSET]?.focus).toEqual({ kind: "ids" });
    // The threshold then has nothing to move.
    const before = useInstances.getState().assets;
    store.setSimilarThreshold(ASSET, 0.4);
    expect(useInstances.getState().assets).toBe(before);

    store.showSimilar(ASSET, 1, 1, SCORES);
    store.toggleFocus(ASSET, { kind: "category", id: "rock" });
    expect(useInstances.getState().assets[ASSET]?.similar).toBeNull();

    store.showSimilar(ASSET, 1, 1, SCORES);
    // The same focus again clears it, as every highlight's does.
    store.toggleFocus(ASSET, { kind: "similar", seedId: 1 });
    expect(useInstances.getState().assets[ASSET]?.similar).toBeNull();
    expect(lit()).toEqual([]);

    store.showSimilar(ASSET, 1, 1, SCORES);
    store.reset(ASSET);
    expect(useInstances.getState().assets[ASSET]?.similar).toBeNull();
    expect(lit()).toEqual([]);
  });
});

describe("embeddingOf", () => {
  beforeEach(() => useInstances.setState({ assets: {}, dimOthers: true }));

  const gpu: InstanceGpu = {
    createUintQuads: () => ({
      copyFrom: () => undefined,
      isDestroyed: () => false,
      destroy: () => undefined,
    }),
    createBytes: () => ({
      copyFrom: () => undefined,
      isDestroyed: () => false,
      destroy: () => undefined,
    }),
    vec4: (x, y, z, w) => [x, y, z, w],
  };
  const scene = {
    preUpdate: { addEventListener: () => () => undefined },
    requestRender: () => undefined,
  } as unknown as Pick<Scene, "preUpdate" | "requestRender">;

  it("is beside the instances.json drawn, sized by its declaration", async () => {
    const tileset = {
      root: { extras: { instances: { uri: "objects/instances.json?v=2", count: 10 } } },
      resource: { url: "https://x.test/scan/tileset.json?sig=abc" },
    } as unknown as Cesium3DTileset;
    const dispose = attachInstances(tileset, scene, "scan", gpu, () => Promise.resolve(DOC));
    await Promise.resolve();
    await Promise.resolve();
    expect(embeddingOf("scan")).toEqual({
      url: "https://x.test/scan/objects/instances.emb?v=2",
      rows: 10,
      dim: DIM,
      model: "siglip:google/siglip2-base-patch16-224",
    });
    expect(embeddingUrlOf("scan")).toBe("https://x.test/scan/objects/instances.emb?v=2");
    dispose();
    expect(embeddingUrlOf("scan")).toBeNull();
  });

  it("is null for a file that declares no embedding", async () => {
    const plain = parseInstances({
      format: "hexapod.instances",
      version: 1,
      instances: [tagged(1, "shrub")],
      tiles: {},
    })!;
    const tileset = {
      root: { extras: { instances: { uri: "instances.json", count: 1 } } },
      resource: { url: "https://x.test/plain/tileset.json" },
    } as unknown as Cesium3DTileset;
    const dispose = attachInstances(tileset, scene, "plain", gpu, () => Promise.resolve(plain));
    await Promise.resolve();
    await Promise.resolve();
    expect(useInstances.getState().assets.plain).toBeDefined();
    expect(embeddingUrlOf("plain")).toBeNull();
    dispose();
  });
});

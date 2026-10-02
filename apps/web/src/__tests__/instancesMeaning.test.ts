/**
 * Search by meaning (lib/instances.ts, state/instances.ts): `instances.emb` read and ranked
 * against a query's text embedding, blended with the tag match, and loaded only when a
 * search needs it -- with tags carrying the search while it loads or when it cannot.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { QueryEmbedding } from "@/api/textEmbedding";
import {
  ANCESTOR_DISCOUNT,
  cosineById,
  loadEmbeddings,
  MEANING_FLOOR,
  MEANING_SPAN,
  meaningScores,
  parseEmbeddings,
  PROMINENCE_FLOOR,
  searchInstances,
  type EmbeddingRef,
  type Instance,
} from "@/lib/instances";
import { configureMeaning, useInstances, type EmbeddingSource } from "@/state/instances";

/** float32 to half bits (round toward zero; exact for the values used here). */
function half(v: number): number {
  if (v === 0) return Object.is(v, -0) ? 0x8000 : 0;
  const u = new Uint32Array(new Float32Array([v]).buffer)[0] ?? 0;
  const sign = (u >>> 16) & 0x8000;
  const exponent = ((u >>> 23) & 0xff) - 127 + 15;
  return sign | (exponent << 10) | ((u >>> 13) & 0x3ff);
}

const DIM = 4;
const REF: EmbeddingRef = {
  file: "instances.emb",
  model: "siglip:test",
  dim: DIM,
  dtype: "float16",
};

function embBuffer(rows: number[][]): ArrayBuffer {
  return Uint16Array.from(rows.flat().map(half)).buffer;
}

function inst(id: number, extra: Partial<Instance> = {}): Instance {
  return {
    id,
    parent: null,
    level: 0,
    splats: 100,
    bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    centroid: [0.5, 0.5, 0.5],
    tags: [],
    properties: {},
    behaviour: "static",
    views: 3,
    ...extra,
  };
}

describe("instances.emb", () => {
  it("reads float16 rows, row k as id k + 1, and knows which are zero", () => {
    const emb = parseEmbeddings(
      embBuffer([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [-0, -0, 0, -0],
        [0.5, 0.5, 0.5, 0.5],
      ]),
      REF,
    );
    expect(emb.count).toBe(4);
    expect(emb.model).toBe("siglip:test");
    expect([...emb.described]).toEqual([1, 0, 0, 1]);
    expect(() => parseEmbeddings(new ArrayBuffer(10), REF)).toThrow(/whole number/);
    expect(() => parseEmbeddings(new ArrayBuffer(0), REF)).toThrow(/whole number/);
  });

  it("is fetched beside instances.json, keeping a signed URL's query", async () => {
    const asked: string[] = [];
    vi.stubGlobal("fetch", (url: string) => {
      asked.push(url);
      return Promise.resolve(new Response(embBuffer([[1, 0, 0, 0]])));
    });
    const emb = await loadEmbeddings("https://x.test/scan/instances.json?sig=1", REF);
    expect(asked).toEqual(["https://x.test/scan/instances.emb?sig=1"]);
    expect(emb.count).toBe(1);
    vi.stubGlobal("fetch", () => Promise.resolve(new Response("", { status: 404 })));
    await expect(loadEmbeddings("https://x.test/a.json", REF)).rejects.toThrow(/404/);
    vi.unstubAllGlobals();
  });

  it("scores rows by cosine, by id; NaN where there is no row or a zero row", () => {
    const emb = parseEmbeddings(
      embBuffer([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0.5, 0.5, 0.5, 0.5],
      ]),
      REF,
    );
    const cos = cosineById(new Float32Array([3, 0, 0, 0]), emb);
    expect(cos).toHaveLength(4);
    expect(cos[0]).toBeNaN();
    expect(cos[1]).toBeCloseTo(1, 6);
    expect(cos[2]).toBeNaN();
    expect(cos[3]).toBeCloseTo(0.5, 6);
    expect(() => cosineById(new Float32Array(3), emb)).toThrow(/dimension/);
  });
});

describe("meaning scores", () => {
  it("measures the margin over the scan's median cosine, and inherits for undescribed", () => {
    const instances = [
      inst(1),
      inst(2),
      inst(3),
      inst(4, { parent: 3, level: 1 }), // no embedding: takes 3's
      inst(5, { parent: 4, level: 2 }), // nor its parent: takes 3's too
      inst(6), // none, and no ancestor
    ];
    const cos = Float32Array.from([NaN, 0.05, 0.06, 0.1, NaN, NaN, NaN]);
    const m = meaningScores(instances, cos);
    // Median of the described (0.05, 0.06, 0.1) is 0.06.
    const expected = (c: number): number =>
      Math.max(0, Math.min(1, (c - 0.06 - MEANING_FLOOR) / MEANING_SPAN));
    expect(m.score[1]).toBe(0);
    expect(m.score[2]).toBe(0);
    expect(m.score[3]).toBeCloseTo(expected(0.1), 6);
    expect(m.score[4]).toBeCloseTo(expected(0.1) * ANCESTOR_DISCOUNT, 6);
    expect(m.score[5]).toBeCloseTo(expected(0.1) * ANCESTOR_DISCOUNT, 6);
    expect([...m.inherited]).toEqual([0, 0, 0, 0, 1, 1, 0]);
    expect(m.score[6]).toBe(0);
  });

  it("survives a parent cycle in a malformed file", () => {
    const m = meaningScores(
      [inst(1, { parent: 2 }), inst(2, { parent: 1 })],
      Float32Array.from([NaN, NaN, NaN]),
    );
    expect([...m.score]).toEqual([0, 0, 0]);
  });
});

describe("search blends tags and meaning", () => {
  const spoolScan = (): Instance[] => [
    // The spool: tagged as something else (no "spool" in the vocabulary).
    inst(1, { splats: 220, tags: [{ label: "milestone", score: 0.08 }] }),
    inst(2, { parent: 1, level: 1, splats: 95_000, tags: [{ label: "stone", score: 0.08 }] }),
    // Too small to describe: no tags, no row.
    inst(3, { parent: 1, level: 1, splats: 50 }),
    // A fragment the tags call a spool.
    inst(4, { splats: 40, tags: [{ label: "cable spool", score: 0.3 }] }),
    inst(5, { splats: 5_000, tags: [{ label: "grass", score: 0.6 }] }),
  ];
  const scores = (by: Record<number, number>, inherited: number[] = []) => {
    const score = new Float32Array(6);
    const flags = new Uint8Array(6);
    for (const [id, s] of Object.entries(by)) score[Number(id)] = s;
    for (const id of inherited) flags[id] = 1;
    return { score, inherited: flags };
  };

  it("without meaning, is the tag search", () => {
    const results = searchInstances(spoolScan(), "spool");
    expect(results.map((r) => [r.id, r.via])).toEqual([[4, "tags"]]);
  });

  it("finds what the tags missed, and blends both by noisy-OR under the prominence prior", () => {
    const instances = spoolScan();
    const meaning = scores({ 1: 0.6, 2: 0.2, 3: 0.54, 4: 0.5 }, [3]);
    const results = searchInstances(instances, "spool", 50, meaning);
    // The spool first; its undescribed part is covered by it and left out.
    expect(results.map((r) => r.id)).toEqual([1, 4, 2]);
    expect(results[0]).toMatchObject({ id: 1, via: "meaning", label: "milestone" });
    // 1 is the largest (with its parts): full prominence.
    expect(results[0]?.score).toBeCloseTo(0.6, 6);
    const fragment = results.find((r) => r.id === 4);
    const tag = 0.95 * 0.3;
    const prior = PROMINENCE_FLOOR + (1 - PROMINENCE_FLOOR) * (Math.log1p(40) / Math.log1p(95_270));
    expect(fragment?.score).toBeCloseTo((1 - (1 - tag) * (1 - 0.5)) * prior, 6);
    expect(fragment?.via).toBe("meaning");
    expect(fragment?.label).toBe("cable spool"); // the top tag, which is what it is shown by
  });

  it("lists an inherited match when its ancestor is not listed", () => {
    const meaning = scores({ 1: 0.6, 3: 0.54 }, [3]);
    // A filter keeps the parent out; its undescribed part then stands for it.
    const results = searchInstances(spoolScan(), "spool behaviour:in-place", 50, {
      ...meaning,
    });
    expect(results).toEqual([]);
    const parts = spoolScan().map((i) =>
      i.id === 3 ? { ...i, behaviour: "in-place" as const } : i,
    );
    expect(
      searchInstances(parts, "spool behaviour:in-place", 50, meaning).map((r) => r.id),
    ).toEqual([3]);
    expect(searchInstances(parts, "spool behaviour:in-place", 50, meaning)[0]?.label).toBe(
      "Object 3",
    );
  });

  it("lets meaning lift a tag match, and leaves filters-only queries alone", () => {
    const instances = spoolScan();
    const meaning = scores({ 5: 0.9 });
    const tagOnly = searchInstances(instances, "grass")[0]?.score ?? 0;
    const both = searchInstances(instances, "grass", 50, meaning)[0];
    expect(both?.id).toBe(5);
    expect(both?.score).toBeGreaterThan(tagOnly);
    expect(both?.via).toBe("meaning");
    expect(searchInstances(instances, "behaviour:static", 50, meaning).map((r) => r.via)).toEqual(
      Array(5).fill("tags"),
    );
  });
});

describe("the store, searching by meaning", () => {
  const SOURCE: EmbeddingSource = { instancesUrl: "https://x.test/scan/instances.json", ref: REF };
  // Row 1: the spool; row 2: grass; row 3: undescribed; 4 and 5 neutral.
  const ROWS = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  const table = () => ({
    instances: [
      inst(1, { tags: [{ label: "milestone", score: 0.1 }], splats: 1000 }),
      inst(2, { tags: [{ label: "grass", score: 0.5 }] }),
      inst(3, { parent: 1, level: 1, splats: 10 }),
      inst(4, { tags: [{ label: "rock", score: 0.5 }] }),
      inst(5, { tags: [{ label: "dirt", score: 0.5 }] }),
    ],
    propertyNames: [],
    embeddingSource: SOURCE,
  });
  let loads: string[];
  let encoded: string[];
  const encode = (text: string): Promise<QueryEmbedding> => {
    encoded.push(text);
    const vector = text.includes("spool")
      ? new Float32Array([1, 0, 0, 0])
      : new Float32Array([0, 0.1, 0.1, 0.1]);
    return Promise.resolve({ model: "siglip:test", prompt: `a photo of a ${text}.`, vector });
  };
  const settle = () => new Promise((resolve) => setTimeout(resolve, 10));
  const entry = () => useInstances.getState().assets.scan;

  beforeEach(() => {
    loads = [];
    encoded = [];
    useInstances.setState({ assets: {}, dimOthers: true });
    configureMeaning({
      debounceMs: 0,
      retryMs: 60_000,
      encode,
      loadEmbeddings: (url, ref) => {
        loads.push(`${url} ${ref.file}`);
        return Promise.resolve(parseEmbeddings(embBuffer(ROWS), ref));
      },
    });
  });
  afterEach(() => configureMeaning(null));

  it("answers by tags at once, then by meaning, loading the rows once", async () => {
    useInstances.getState().setTable("scan", table());
    expect(entry()?.meaning).toBe("idle");
    expect(loads).toEqual([]); // nothing until a search
    useInstances.getState().setQuery("scan", "spool");
    expect(entry()?.results).toEqual([]); // no tag says spool
    expect(entry()?.meaning).toBe("loading");
    await settle();
    expect(entry()?.meaning).toBe("ready");
    expect(entry()?.results.map((r) => [r.id, r.via])).toEqual([[1, "meaning"]]);
    useInstances.getState().setQuery("scan", "a cable spool");
    await settle();
    expect(entry()?.results[0]?.id).toBe(1);
    expect(loads).toEqual(["https://x.test/scan/instances.json instances.emb"]);
    expect(encoded).toEqual(["spool", "a cable spool"]);
    // Asked again: from the cache, at once, without the encoder.
    useInstances.getState().setQuery("scan", "spool");
    expect(entry()?.meaning).toBe("ready");
    expect(entry()?.results[0]?.id).toBe(1);
    expect(encoded).toHaveLength(2);
  });

  it("sends only the words, and nothing for filters alone", async () => {
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "behaviour:static");
    await settle();
    expect(encoded).toEqual([]);
    expect(entry()?.results).toHaveLength(5);
    useInstances.getState().setQuery("scan", "Spool, behaviour:static");
    await settle();
    expect(encoded).toEqual(["spool"]);
  });

  it("drops an answer for a query that is no longer the query", async () => {
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "spool");
    useInstances.getState().setQuery("scan", "grass");
    await settle();
    expect(encoded).toEqual(["grass"]); // the debounce never asked for "spool"
    expect(entry()?.query).toBe("grass");
    expect(entry()?.results[0]?.id).toBe(2);
  });

  it("stays on tags when the encoder is not there, without asking on every keystroke", async () => {
    configureMeaning({
      encode: (text) => {
        encoded.push(text);
        return Promise.reject(new Error("503 Service Unavailable"));
      },
    });
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "grass");
    expect(entry()?.results.map((r) => r.id)).toEqual([2]);
    await settle();
    expect(entry()?.meaning).toBe("unavailable");
    expect(entry()?.results.map((r) => [r.id, r.via])).toEqual([[2, "tags"]]);
    useInstances.getState().setQuery("scan", "grassy");
    await settle();
    expect(encoded).toEqual(["grass"]); // backed off
    expect(entry()?.meaning).toBe("unavailable");
  });

  it("stays on tags when instances.emb does not load", async () => {
    configureMeaning({ loadEmbeddings: () => Promise.reject(new Error("404")) });
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "grass");
    await settle();
    expect(entry()?.meaning).toBe("unavailable");
    expect(entry()?.results.map((r) => r.id)).toEqual([2]);
  });

  it("refuses a query embedding from another model", async () => {
    configureMeaning({
      encode: (text) => encode(text).then((e) => ({ ...e, model: "siglip:other" })),
    });
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "spool");
    await settle();
    expect(entry()?.meaning).toBe("unavailable");
    expect(entry()?.results).toEqual([]);
  });

  it("never asks for a scan without instances.emb", async () => {
    useInstances.getState().setTable("scan", { ...table(), embeddingSource: null });
    expect(entry()?.meaning).toBe("none");
    useInstances.getState().setQuery("scan", "grass");
    await settle();
    expect(encoded).toEqual([]);
    expect(loads).toEqual([]);
    expect(entry()?.meaning).toBe("none");
  });
});

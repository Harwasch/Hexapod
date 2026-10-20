/**
 * Search by meaning (lib/instances.ts `textCosines`, `meaningScores`, `searchInstances`;
 * state/instances.ts): a query's text embedding ranked against the scan's described rows,
 * blended with the tag match, and loaded only when a search needs it -- with tags carrying
 * the search while it loads or when it cannot.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { ApiError } from "@/api/error";
import { embeddable, queryLabel, type QueryEmbedding } from "@/api/textEmbedding";
import {
  ANCESTOR_DISCOUNT,
  describedMatrix,
  MEANING_FLOOR,
  MEANING_MIN_DESCRIBED,
  MEANING_SPAN,
  meaningScores,
  PROMINENCE_FLOOR,
  searchInstances,
  textCosines,
  type EmbeddingSource,
  type Instance,
} from "@/lib/instances";
import { configureMeaning, meaningRowsHeld, useInstances } from "@/state/instances";

/** float32 to half bits (round toward zero; exact for the values used here). */
function half(v: number): number {
  if (v === 0) return Object.is(v, -0) ? 0x8000 : 0;
  const u = new Uint32Array(new Float32Array([v]).buffer)[0] ?? 0;
  const sign = (u >>> 16) & 0x8000;
  const exponent = ((u >>> 23) & 0xff) - 127 + 15;
  return sign | (exponent << 10) | ((u >>> 13) & 0x3ff);
}

const DIM = 4;

function rows(list: number[][]): Uint16Array {
  return Uint16Array.from(list.flat().map(half));
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

describe("text cosines", () => {
  it("are the raw cosines with each described row, by id, from the centred matrix", () => {
    const raw = [
      [1, 0, 0, 0],
      [0, 0, 0, 0], // not described
      [0.5, 0.5, 0.5, 0.5],
      [0, 0.6, 0.8, 0],
    ];
    const matrix = describedMatrix(rows(raw), DIM);
    const query = new Float32Array([3, 1, -2, 0.5]);
    const cos = textCosines(query, matrix);
    expect(cos).toHaveLength(5);
    expect(cos[0]).toBeNaN();
    expect(cos[2]).toBeNaN();
    const norm = Math.hypot(...query);
    for (const id of [1, 3, 4]) {
      const row = raw[id - 1] ?? [];
      const want = row.reduce((sum, v, j) => sum + v * (query[j] ?? 0), 0) / norm;
      expect(cos[id]).toBeCloseTo(want, 5);
    }
    expect(() => textCosines(new Float32Array(3), matrix)).toThrow(/dimension/);
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

  it("is not measurable with fewer than three described instances", () => {
    expect(MEANING_MIN_DESCRIBED).toBe(3);
    // One: its margin over the median is 0 whatever the query. Two: the median is their mean,
    // so the nearer always clears it by half their difference. Neither says "this matches".
    for (const cos of [Float32Array.from([NaN, 0.2, NaN]), Float32Array.from([NaN, 0.02, 0.2])]) {
      const m = meaningScores([inst(1), inst(2, { parent: 1 })], cos);
      expect(m.measurable).toBe(false);
      expect([...m.score]).toEqual([0, 0, 0]);
    }
    const three = meaningScores(
      [inst(1), inst(2), inst(3)],
      Float32Array.from([NaN, 0.02, 0.03, 0.2]),
    );
    expect(three.measurable).toBe(true);
    expect(three.score[3]).toBe(1);
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
    return { score, inherited: flags, measurable: true };
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
    expect(searchInstances(spoolScan(), "spool behaviour:in-place", 50, meaning)).toEqual([]);
    const parts = spoolScan().map((i) =>
      i.id === 3 ? { ...i, behaviour: "in-place" as const } : i,
    );
    const results = searchInstances(parts, "spool behaviour:in-place", 50, meaning);
    expect(results.map((r) => r.id)).toEqual([3]);
    expect(results[0]?.label).toBe("untagged");
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

describe("what is sent to the encoder", () => {
  it("is what the API would embed, and not what it would refuse", () => {
    expect(queryLabel("  A Cable   SPOOL. ")).toBe("cable spool");
    expect(queryLabel("a photo of a log cabin")).toBe("log cabin");
    expect(queryLabel("antenna")).toBe("antenna");
    for (const refused of ["", "  ", "a", "The", "an", "x".repeat(201)])
      expect(embeddable(refused)).toBe(false);
    for (const fine of ["spool", "a spool", "x".repeat(200)]) expect(embeddable(fine)).toBe(true);
  });
});

describe("the store, searching by meaning", () => {
  const SOURCE: EmbeddingSource = {
    url: "https://x.test/scan/instances.emb",
    rows: 5,
    dim: DIM,
    model: "siglip:test",
  };
  // Row 1: the spool; row 2: moss; row 3: undescribed; 4 and 5 neutral.
  const ROWS = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  // No tag or category says "spool"; objects are the roots, 1, 2, 4 and 5.
  const table = () => ({
    instances: [
      inst(1, { tags: [{ label: "milestone", score: 0.1 }], splats: 1000 }),
      inst(2, { tags: [{ label: "moss", score: 0.5 }] }),
      inst(3, { parent: 1, level: 1, splats: 10 }),
      inst(4, { tags: [{ label: "pebble", score: 0.5 }] }),
      inst(5, { tags: [{ label: "puddle", score: 0.5 }] }),
    ],
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
      loadEmbeddings: (source) => {
        loads.push(source.url);
        return Promise.resolve(rows(ROWS));
      },
    });
  });
  afterEach(() => configureMeaning(null));

  it("answers by tags at once, then by meaning, loading the rows once", async () => {
    useInstances.getState().setTable("scan", table());
    expect(entry()?.meaning).toBe("idle");
    expect(loads).toEqual([]); // nothing until a search
    useInstances.getState().setQuery("scan", "spool");
    expect(entry()?.matches).toEqual([]); // no tag says spool
    expect(entry()?.meaning).toBe("loading");
    await settle();
    expect(entry()?.meaning).toBe("ready");
    expect(entry()?.matches).toEqual([1]);
    expect([...(entry()?.byMeaning ?? [])]).toEqual([1]);
    useInstances.getState().setQuery("scan", "a cable spool");
    await settle();
    expect(entry()?.matches[0]).toBe(1);
    expect(loads).toEqual(["https://x.test/scan/instances.emb"]);
    expect(encoded).toEqual(["spool", "a cable spool"]);
    // Asked again: from the cache, at once, without the encoder.
    useInstances.getState().setQuery("scan", "spool");
    expect(entry()?.meaning).toBe("ready");
    expect(entry()?.matches).toEqual([1]);
    expect(encoded).toHaveLength(2);
  });

  it("lights the new matches when the matches were lit", async () => {
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "moss");
    useInstances.getState().toggleFocus("scan", { kind: "matches" });
    expect([...(entry()?.highlighted ?? [])]).toEqual([2]);
    useInstances.getState().setQuery("scan", "spool");
    await settle();
    // The spool object: the instance and its undescribed part.
    expect([...(entry()?.highlighted ?? [])].sort()).toEqual([1, 3]);
  });

  it("sends only the words, and nothing for filters alone", async () => {
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "behaviour:static");
    await settle();
    expect(encoded).toEqual([]);
    expect(entry()?.matches).toHaveLength(4);
    useInstances.getState().setQuery("scan", "Spool, behaviour:static");
    await settle();
    expect(encoded).toEqual(["spool"]);
  });

  it("drops an answer for a query that is no longer the query", async () => {
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "spool");
    useInstances.getState().setQuery("scan", "moss");
    await settle();
    expect(encoded).toEqual(["moss"]); // the debounce never asked for "spool"
    expect(entry()?.query).toBe("moss");
    expect(entry()?.matches[0]).toBe(2);
  });

  it("stays on tags when the encoder is not there, without asking on every keystroke", async () => {
    configureMeaning({
      encode: (text) => {
        encoded.push(text);
        return Promise.reject(new Error("503 Service Unavailable"));
      },
    });
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "moss");
    expect(entry()?.matches).toEqual([2]);
    await settle();
    expect(entry()?.meaning).toBe("unavailable");
    expect(entry()?.matches).toEqual([2]);
    expect(entry()?.byMeaning.size).toBe(0);
    useInstances.getState().setQuery("scan", "mossy");
    await settle();
    expect(encoded).toEqual(["moss"]); // backed off
    expect(entry()?.meaning).toBe("unavailable");
  });

  it("stays on tags when instances.emb does not load", async () => {
    configureMeaning({ loadEmbeddings: () => Promise.reject(new Error("404")) });
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "moss");
    await settle();
    expect(entry()?.meaning).toBe("unavailable");
    expect(entry()?.matches).toEqual([2]);
  });

  it("asks nothing for an article alone, then embeds the query it grows into", async () => {
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "a");
    await settle();
    expect(encoded).toEqual([]);
    expect(loads).toEqual([]);
    expect(entry()?.meaning).toBe("idle");
    useInstances.getState().setQuery("scan", "a spool");
    expect(entry()?.meaning).toBe("loading");
    await settle();
    expect(encoded).toEqual(["a spool"]);
    expect(entry()?.meaning).toBe("ready");
    expect(entry()?.matches).toEqual([1]);
  });

  it("does not back off when the API refuses one query (422)", async () => {
    configureMeaning({
      encode: (text) => {
        encoded.push(text);
        return text === "zzz"
          ? Promise.reject(new ApiError(422, undefined, "422 Unprocessable Content"))
          : encode(text).then((e) => (encoded.pop(), e));
      },
    });
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "zzz");
    await settle();
    expect(entry()?.meaning).toBe("unavailable");
    useInstances.getState().setQuery("scan", "spool");
    await settle();
    expect(encoded).toEqual(["zzz", "spool"]);
    expect(entry()?.meaning).toBe("ready");
    expect(entry()?.matches).toEqual([1]);
  });

  it("backs off after a 5xx or no answer, not after a 4xx", async () => {
    for (const [error, backsOff] of [
      [new ApiError(503, undefined, "503"), true],
      [new ApiError(500, undefined, "500"), true],
      [new TypeError("Failed to fetch"), true],
      [new ApiError(422, undefined, "422"), false],
      [new ApiError(400, undefined, "400"), false],
    ] as const) {
      encoded = [];
      configureMeaning({
        encode: (text) => {
          encoded.push(text);
          return Promise.reject(error);
        },
      });
      useInstances.getState().setTable("scan", table());
      useInstances.getState().setQuery("scan", "moss");
      await settle();
      useInstances.getState().setQuery("scan", "mossy");
      await settle();
      expect(encoded).toEqual(backsOff ? ["moss"] : ["moss", "mossy"]);
    }
  });

  it("never lets a late answer for an older query overwrite a newer one", async () => {
    const pending = new Map<string, (e: QueryEmbedding) => void>();
    configureMeaning({
      encode: (text) => {
        encoded.push(text);
        return new Promise((resolve) => pending.set(text, resolve));
      },
    });
    const answer = (text: string, vector: number[]) =>
      pending.get(text)?.({
        model: "siglip:test",
        prompt: `a photo of a ${text}.`,
        vector: Float32Array.from(vector),
      });
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "spool");
    await settle(); // "spool" is asked
    useInstances.getState().setQuery("scan", "moss");
    await settle(); // and "moss"
    expect(encoded).toEqual(["spool", "moss"]);
    answer("moss", [0, 1, 0, 0]);
    await settle();
    expect(entry()?.matches).toEqual([2]);
    answer("spool", [1, 0, 0, 0]); // late, for a query no longer typed
    await settle();
    expect(entry()?.query).toBe("moss");
    expect(entry()?.matches).toEqual([2]);
    expect(entry()?.meaning).toBe("ready");
  });

  it("keeps the rows of the last few scans searched, not of every scan", async () => {
    configureMeaning({ rowsKept: 2 });
    const scan = (n: number) => ({
      ...table(),
      embeddingSource: { ...SOURCE, url: `https://x.test/scan${String(n)}/instances.emb` },
    });
    for (const n of [1, 2, 3, 1]) {
      useInstances.getState().setTable("scan", scan(n));
      useInstances.getState().setQuery("scan", "spool");
      await settle();
      expect(entry()?.matches).toEqual([1]);
      expect(meaningRowsHeld()).toBeLessThanOrEqual(2);
    }
    // Scan 1 was dropped for 3, so coming back to it reads it again; 2 and 3 once each.
    expect(loads).toEqual([1, 2, 3, 1].map((n) => `https://x.test/scan${String(n)}/instances.emb`));
    // Scan 3 is still held: searched again, it is not read again.
    useInstances.getState().setTable("scan", scan(3));
    useInstances.getState().setQuery("scan", "spool");
    await settle();
    expect(loads).toHaveLength(4);
  });

  it("is unavailable on a scan with fewer than three described objects", async () => {
    configureMeaning({
      loadEmbeddings: (source) => {
        loads.push(source.url);
        return Promise.resolve(
          rows([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
          ]),
        );
      },
    });
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "spool");
    await settle();
    expect(entry()?.meaning).toBe("unavailable");
    expect(entry()?.matches).toEqual([]);
    expect(entry()?.byMeaning.size).toBe(0);
  });

  it("refuses a query embedding from another model", async () => {
    configureMeaning({
      encode: (text) => encode(text).then((e) => ({ ...e, model: "siglip:other" })),
    });
    useInstances.getState().setTable("scan", table());
    useInstances.getState().setQuery("scan", "spool");
    await settle();
    expect(entry()?.meaning).toBe("unavailable");
    expect(entry()?.matches).toEqual([]);
  });

  it("never asks for a scan without instances.emb", async () => {
    useInstances.getState().setTable("scan", { ...table(), embeddingSource: null });
    expect(entry()?.meaning).toBe("none");
    useInstances.getState().setQuery("scan", "moss");
    await settle();
    expect(encoded).toEqual([]);
    expect(loads).toEqual([]);
    expect(entry()?.meaning).toBe("none");
  });
});

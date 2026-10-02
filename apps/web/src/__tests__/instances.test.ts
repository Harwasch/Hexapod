import { checksumPositions, decodeRuns, plantLabels, runsLength, tileRunsIssue } from "@twin/world";
import type { Cesium3DTileset, Scene } from "cesium";
import { beforeEach, describe, expect, it } from "vitest";

import type { SplatShaderBuilder, SplatTilesetLike } from "@/cesium/splatInternals";
import {
  attachInstances,
  evaluateInstanceShading,
  idTextureRows,
  INSTANCE_TEXTURE_WIDTH,
  SplatInstances,
  SPLATS_PER_ID_ROW,
  stateTextureRows,
  writeStateTexels,
  type InstanceGpu,
  type InstancePrimitive,
  type SplatVertexColor,
} from "@/cesium/splatInstances";
import { SplatViewCones, type ViewConeGpu } from "@/cesium/splatViewCones";
import {
  addVisibilityPart,
  composeVisibilityGlsl,
  removeVisibilityPart,
  visibilityChainOf,
  type SplatVertexVisibility,
  type SplatVisibilityPart,
} from "@/cesium/splatVisibility";
import {
  halfToFloat,
  hiddenForOnly,
  instanceLabel,
  instancesRefOf,
  matchLabel,
  parseInstances,
  parseQuery,
  PROMINENCE_FLOOR,
  prominence,
  quickFilters,
  rankByEmbedding,
  resolveBeside,
  searchInstances,
  tileInstanceIds,
  withDescendants,
  type InstancesDoc,
} from "@/lib/instances";
import { cellCount, type ViewConesMeta } from "@/lib/viewCones";
import { useInstances } from "@/state/instances";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** A tile's positions on the SPZ grid (integers are), and their checksum. */
function tilePositions(count: number, offset: number): { positions: Float32Array; key: string } {
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    positions[i * 3] = offset + i;
    positions[i * 3 + 1] = offset * 2 + i;
    positions[i * 3 + 2] = 1;
  }
  return { positions, key: checksumPositions(positions) };
}

const TILE_A = tilePositions(6, 0);
const TILE_B = tilePositions(4, 100);

function instance(
  id: number,
  tags: [string, number][],
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    id,
    parent: null,
    level: 0,
    splats: 10 * id,
    bounds: { min: [0, 0, 0], max: [id, id, id] },
    centroid: [id / 2, id / 2, id / 2],
    tags: tags.map(([label, score]) => ({ label, score })),
    properties: {},
    behaviour: "static",
    views: 3,
    ...extra,
  };
}

const RAW = {
  format: "hexapod.instances",
  version: 1,
  embedding: { file: "instances.emb", model: "siglip2", dim: 4, dtype: "float16" },
  instances: [
    instance(
      1,
      [
        ["pickup truck", 0.6],
        ["car", 0.3],
      ],
      {
        properties: { movable: 0.9, vehicle: 0.95, vegetation: 0 },
        behaviour: "movable",
      },
    ),
    instance(2, [["oak tree", 0.8]], {
      properties: { movable: 0.05, vegetation: 0.97 },
      behaviour: "in-place",
    }),
    instance(3, [["tree trunk", 0.5]], {
      parent: 2,
      level: 1,
      properties: { vegetation: 0.7 },
      behaviour: "in-place",
    }),
    instance(4, [["picnic table", 0.4]], { properties: { movable: 0.6 }, behaviour: "movable" }),
  ],
  tiles: {
    [TILE_A.key]: [1, 2, 0, 1, 3, 3],
    [TILE_B.key]: [4, 4],
  },
};

function doc(): InstancesDoc {
  const parsed = parseInstances(structuredClone(RAW));
  if (!parsed) throw new Error("fixture did not parse");
  return parsed;
}

describe("instances.json", () => {
  it("reads the root's declaration", () => {
    expect(instancesRefOf({ instances: { uri: "instances.json", count: 4 } })).toEqual({
      uri: "instances.json",
      count: 4,
    });
    expect(instancesRefOf({ instances: { uri: "" } })).toBeNull();
    expect(instancesRefOf({ viewCones: {} })).toBeNull();
    expect(instancesRefOf(undefined)).toBeNull();
  });

  it("reads a well-formed document", () => {
    const d = doc();
    expect(d.instances.map((i) => i.id)).toEqual([1, 2, 3, 4]);
    expect(d.maxId).toBe(4);
    expect(d.byId.get(3)?.parent).toBe(2);
    expect(d.tiles.size).toBe(2);
    expect(d.embedding).toEqual({
      file: "instances.emb",
      model: "siglip2",
      dim: 4,
      dtype: "float16",
    });
    expect(d.propertyNames).toEqual(["movable", "vegetation", "vehicle"]);
    expect(d.issues).toEqual([]);
  });

  it("refuses what is not one, and costs a malformed entry only that entry", () => {
    expect(parseInstances({ ...RAW, version: 2 })).toBeNull();
    expect(parseInstances({ ...RAW, format: "hexapod.plants" })).toBeNull();
    expect(parseInstances({ ...RAW, instances: "nope" })).toBeNull();
    expect(parseInstances(null)).toBeNull();
    const d = parseInstances({
      ...structuredClone(RAW),
      embedding: { file: "x.emb" },
      instances: [
        ...structuredClone(RAW.instances),
        { id: 0, bounds: { min: [0, 0, 0], max: [1, 1, 1] } },
        { id: 7 },
        instance(2, [["duplicate", 1]]),
        instance(
          5,
          [
            ["  ", 1],
            ["lamp", Number.NaN],
          ],
          {
            parent: 99,
            behaviour: "flying",
            properties: { glow: "bright", lit: 0.4 },
            centroid: "middle",
          },
        ),
      ],
      tiles: {
        ...RAW.tiles,
        "not-a-checksum": [1, 1],
        "fnv1a32:3:0000abcd": [1, 2],
        "fnv1a32:2:0000abce": [9, 2],
        "fnv1a32:2:0000abcf": [1, 2, 3],
      },
    });
    expect(d).not.toBeNull();
    if (!d) return;
    expect(d.instances.map((i) => i.id)).toEqual([1, 2, 3, 4, 5]);
    const lamp = d.byId.get(5);
    expect(lamp?.parent).toBeNull();
    expect(lamp?.behaviour).toBe("static");
    expect(lamp?.tags).toEqual([{ label: "lamp", score: 0 }]);
    expect(lamp?.properties).toEqual({ lit: 0.4 });
    expect(lamp?.centroid).toEqual([2.5, 2.5, 2.5]);
    expect(d.tiles.size).toBe(2);
    expect(d.embedding).toBeNull();
    expect(d.issues.length).toBe(8);
    expect(d.issues.join("\n")).toMatch(/listed twice/);
    expect(d.issues.join("\n")).toMatch(/runs cover 2 gaussians, the tile holds 3/);
  });

  it("resolves the file beside the tileset, keeping a signed URL's query", () => {
    expect(resolveBeside("https://x.test/a/tileset.json?sig=1", "instances.json")).toBe(
      "https://x.test/a/instances.json?sig=1",
    );
    expect(resolveBeside("https://x.test/a/tileset.json", "https://y.test/i.json")).toBe(
      "https://y.test/i.json",
    );
  });
});

describe("per-tile ids (the plants.json encoding, shared)", () => {
  it("decodes a tile's runs in its own gaussian order", () => {
    const d = doc();
    expect([...(tileInstanceIds(d, TILE_A.key) ?? [])]).toEqual([1, 1, 0, 3, 3, 3]);
    expect([...(tileInstanceIds(d, TILE_B.key) ?? [])]).toEqual([4, 4, 4, 4]);
    expect(tileInstanceIds(d, "fnv1a32:1:00000000")).toBeUndefined();
  });

  it("is the same reader plants.json uses", () => {
    const runs = Int32Array.from([2, 3, 0, 1, 1, 2]);
    expect(runsLength(runs)).toBe(6);
    expect([...decodeRuns(runs, new Uint32Array(6))]).toEqual([2, 2, 2, 0, 1, 1]);
    const binding = { plantCount: 2, tiles: new Map([["k", runs]]) };
    expect([...(plantLabels(binding, "k") ?? [])]).toEqual([2, 2, 2, 0, 1, 1]);
    expect(tileRunsIssue("fnv1a32:6:0000abcd", [...runs], 2)).toBeUndefined();
    expect(tileRunsIssue("fnv1a32:6:0000abcd", [...runs], 1, "a plant")).toMatch(
      /label 2 is not 0 or a plant/,
    );
  });
});

describe("search", () => {
  it("splits words from property and behaviour filters", () => {
    expect(parseQuery("Oak  tree vegetation > 0.5 behaviour:movable")).toEqual({
      terms: ["oak", "tree"],
      filters: [
        { name: "vegetation", op: ">", value: 0.5 },
        { name: "behaviour", op: "=", value: "movable" },
      ],
    });
    expect(parseQuery("movable>=0.3").filters).toEqual([{ name: "movable", op: ">=", value: 0.3 }]);
    // Not a number, not a behaviour: words.
    expect(parseQuery("colour:red")).toEqual({ terms: ["colour", "red"], filters: [] });
  });

  it("scores the whole phrase above words, words above prefixes, prefixes above substrings", () => {
    expect(matchLabel(["oak", "tree"], "oak tree")).toBe(1);
    expect(matchLabel(["tree"], "oak tree")).toBeCloseTo(0.95, 9);
    expect(matchLabel(["tre"], "oak tree")).toBeCloseTo(0.75 * 0.95, 9);
    expect(matchLabel(["ree"], "oak tree")).toBeCloseTo(0.5 * 0.95, 9);
    expect(matchLabel(["car"], "oak tree")).toBe(0);
    expect(matchLabel([], "oak tree")).toBe(0);
  });

  it("ranks by tag match times tag score, with no class list", () => {
    const d = doc();
    const results = searchInstances(d.instances, "tree");
    expect(results.map((r) => r.id)).toEqual([2, 3]);
    expect(results[0]?.label).toBe("oak tree");
    const largest = Math.max(...d.instances.map((i) => i.splats));
    const oak = d.instances.find((i) => i.id === 2);
    expect(results[0]?.score).toBeCloseTo(
      0.95 *
        0.8 *
        (PROMINENCE_FLOOR + (1 - PROMINENCE_FLOOR) * prominence(oak?.splats ?? 0, largest)),
      9,
    );
    expect(results[0]?.behaviour).toBe("in-place");
    // A lower tag matches too, by its own score.
    expect(searchInstances(d.instances, "car").map((r) => [r.id, r.label])).toEqual([[1, "car"]]);
    expect(searchInstances(d.instances, "")).toEqual([]);
    expect(searchInstances(d.instances, "submarine")).toEqual([]);
  });

  it("puts the object above a fragment that matches a little better", () => {
    const base = doc().instances[0];
    if (!base) throw new Error("fixture");
    const tag = (label: string, score: number) => [{ label, score }];
    const object = { ...base, id: 1, splats: 30000, tags: tag("cable spool", 0.5) };
    const fragment = { ...base, id: 2, splats: 40, tags: tag("cable spool", 0.6) };
    expect(searchInstances([fragment, object], "spool").map((r) => r.id)).toEqual([1, 2]);
    expect(prominence(0, 10)).toBe(0);
    expect(prominence(10, 10)).toBe(1);
  });

  it("finds by property name, and filters by value", () => {
    const d = doc();
    const byName = searchInstances(d.instances, "vegetation");
    expect(byName.map((r) => r.id)).toEqual([2, 3]);
    expect(byName[0]?.label).toBe("vegetation 0.97");
    expect(searchInstances(d.instances, "movable > 0.5").map((r) => r.id)).toEqual([1, 4]);
    expect(searchInstances(d.instances, "behaviour:in-place").map((r) => r.id)).toEqual([3, 2]);
    expect(searchInstances(d.instances, "tree vegetation >= 0.9").map((r) => r.id)).toEqual([2]);
    // A property an instance does not carry fails the filter.
    expect(searchInstances(d.instances, "vehicle < 0.5")).toEqual([]);
    expect(searchInstances(d.instances, "tree", 1)).toHaveLength(1);
  });

  it("offers quick filters from what the file holds", () => {
    const filters = quickFilters(doc());
    expect(filters.map((f) => [f.kind, f.label, f.query, f.count])).toEqual([
      ["property", "movable", "movable > 0.5", 2],
      ["property", "vegetation", "vegetation > 0.5", 2],
      ["property", "vehicle", "vehicle > 0.5", 1],
      ["behaviour", "behaviour: movable", "behaviour:movable", 2],
      ["behaviour", "behaviour: in-place", "behaviour:in-place", 2],
    ]);
    // A property and a behaviour that share a name ("movable") never read the same.
    expect(new Set(filters.map((f) => f.label)).size).toBe(filters.length);
  });

  it("returns every match when asked for no limit, with each one's size", () => {
    const many = Array.from({ length: 120 }, (_, k) => ({
      ...(doc().instances[1] ?? ({} as never)),
      id: k + 1,
      splats: 1000 + k,
    }));
    expect(searchInstances(many, "vegetation")).toHaveLength(50);
    const all = searchInstances(many, "vegetation", Number.POSITIVE_INFINITY);
    expect(all).toHaveLength(120);
    // Ties go to the larger instance; the size comes with the result.
    expect(all[0]).toMatchObject({ id: 120, splats: 1119 });
  });

  it("hides everything but the matches, keeping what holds them and what they hold", () => {
    const d = doc();
    // 3 is inside 2: showing only 3 keeps its parent 2 (hiding 2 would hide 3 with it).
    expect(hiddenForOnly(d, [3]).sort()).toEqual([1, 4]);
    // Showing only 2 keeps 3, inside it.
    expect(hiddenForOnly(d, [2]).sort()).toEqual([1, 4]);
    expect(hiddenForOnly(d, [1, 4]).sort()).toEqual([2, 3]);
    expect(hiddenForOnly(d, []).sort()).toEqual([1, 2, 3, 4]);
  });

  it("walks the hierarchy down from a coarse instance", () => {
    const d = doc();
    expect([...withDescendants(d, [2])].sort()).toEqual([2, 3]);
    expect([...withDescendants(d, [1])]).toEqual([1]);
    expect(withDescendants(d, []).size).toBe(0);
    expect(instanceLabel(d.instances[0] ?? ({} as never))).toBe("pickup truck");
    expect(instanceLabel({ ...(d.instances[0] ?? ({} as never)), tags: [], id: 9 })).toBe(
      "Object 9",
    );
  });
});

describe("search by meaning (the seam)", () => {
  /** float32 to half bits, for values exactly representable. */
  function half(v: number): number {
    const f = new Float32Array([v]);
    const u = new Uint32Array(f.buffer)[0] ?? 0;
    const sign = (u >>> 16) & 0x8000;
    const exponent = ((u >>> 23) & 0xff) - 127 + 15;
    if (v === 0) return sign;
    return sign | (exponent << 10) | ((u >>> 13) & 0x3ff);
  }

  it("reads float16", () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x3800)).toBe(0.5);
    expect(halfToFloat(0x0001)).toBeCloseTo(2 ** -24, 30);
    expect(halfToFloat(0x7c00)).toBe(Number.POSITIVE_INFINITY);
    expect(halfToFloat(0x7e00)).toBeNaN();
    expect(halfToFloat(half(0.25))).toBe(0.25);
  });

  it("ranks rows by cosine against the query, row k as id k + 1", () => {
    const rows = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0.5, 0.5, 0.5, 0.5],
    ];
    const emb = Uint16Array.from(rows.flat().map(half));
    const ranked = rankByEmbedding(new Float32Array([0, 3, 0, 0]), emb, 4);
    expect(ranked.map((r) => r.id)).toEqual([2, 3, 1]);
    expect(ranked[0]?.score).toBeCloseTo(1, 6);
    expect(ranked[1]?.score).toBeCloseTo(0.5, 6);
    expect(rankByEmbedding(new Float32Array([0, 3, 0, 0]), emb, 4, 1)).toHaveLength(1);
    expect(() => rankByEmbedding(new Float32Array(3), emb, 4)).toThrow(/dimension/);
  });
});

describe("the store", () => {
  beforeEach(() => useInstances.setState({ assets: {}, dimOthers: true, gaps: {} }));

  it("hides all of a query's matches, or shows only them, past the listed ones", () => {
    const s = (): ReturnType<typeof useInstances.getState> => useInstances.getState();
    const base = doc();
    const many = Array.from({ length: 120 }, (_, k) => ({
      ...(base.instances[1] ?? ({} as never)),
      id: k + 1,
      properties: { vegetation: k < 90 ? 0.9 : 0.1 },
    }));
    s().setTable("a", { instances: many, propertyNames: ["vegetation"] });
    s().setQuery("a", "vegetation > 0.5");
    expect(s().assets.a?.results).toHaveLength(50);
    expect(s().assets.a?.matches).toHaveLength(90);
    s().hideMatches("a");
    expect(s().assets.a?.hidden.size).toBe(90);
    expect(s().assets.a?.hidden.has(90)).toBe(true);
    expect(s().assets.a?.hidden.has(91)).toBe(false);
    s().showOnlyMatches("a");
    expect([...(s().assets.a?.hidden ?? [])].sort((x, y) => x - y)).toEqual(
      Array.from({ length: 30 }, (_, k) => 91 + k),
    );
    // No query, no matches: nothing changes.
    s().setQuery("a", "");
    const hidden = s().assets.a?.hidden;
    s().hideMatches("a");
    s().showOnlyMatches("a");
    expect(s().assets.a?.hidden).toBe(hidden);
  });

  it("records a renderer that cannot draw a scan's objects", () => {
    const s = (): ReturnType<typeof useInstances.getState> => useInstances.getState();
    s().setGap("a", { renderer: "playcanvas", reason: "native" });
    const gaps = s().gaps;
    s().setGap("a", { renderer: "playcanvas", reason: "native" });
    expect(s().gaps).toBe(gaps);
    s().setGap("a", null);
    expect(s().gaps.a).toBeUndefined();
  });

  it("holds a scan's table, searches it, and hides and highlights", () => {
    const s = (): ReturnType<typeof useInstances.getState> => useInstances.getState();
    s().setTable("a", doc());
    expect(s().assets.a?.filters.length).toBeGreaterThan(0);
    s().setQuery("a", "tree");
    expect(s().assets.a?.results.map((r) => r.id)).toEqual([2, 3]);
    s().toggleHidden("a", 2);
    expect([...(s().assets.a?.hidden ?? [])]).toEqual([2]);
    s().toggleHidden("a", 2);
    expect(s().assets.a?.hidden.size).toBe(0);
    s().setHidden("a", [1, 4], true);
    s().setHidden("a", [4], false);
    expect([...(s().assets.a?.hidden ?? [])]).toEqual([1]);
    s().showAll("a");
    expect(s().assets.a?.hidden.size).toBe(0);
    s().highlight("a", [3]);
    expect([...(s().assets.a?.highlighted ?? [])]).toEqual([3]);
    s().highlight("a", []);
    expect(s().assets.a?.highlighted.size).toBe(0);
    s().setDimOthers(false);
    expect(s().dimOthers).toBe(false);
    // An unknown asset is left alone; clearing removes the table.
    const before = s().assets;
    s().toggleHidden("b", 1);
    expect(s().assets).toBe(before);
    s().setTable("a", null);
    expect(s().assets.a).toBeUndefined();
  });

  it("keeps no table for a scan with no instances", () => {
    useInstances.getState().setTable("a", { instances: [], propertyNames: [] });
    expect(useInstances.getState().assets.a).toBeUndefined();
  });
});

// ---- GPU ---------------------------------------------------------------------------------

interface FakeTexture {
  width: number;
  height: number;
  data: ArrayBufferView;
  copies: { yOffset?: number; height: number }[];
  isDestroyed(): boolean;
  destroy(): void;
  copyFrom(options: {
    source: { width: number; height: number; arrayBufferView: ArrayBufferView };
    yOffset?: number;
  }): void;
}

function fakeGpu(): InstanceGpu & { textures: FakeTexture[] } {
  const textures: FakeTexture[] = [];
  const make = (width: number, height: number, data: ArrayBufferView): FakeTexture => {
    let destroyed = false;
    const texture: FakeTexture = {
      width,
      height,
      data: data instanceof Uint32Array ? data.slice() : (data as Uint8Array).slice(),
      copies: [],
      isDestroyed: () => destroyed,
      destroy: () => {
        destroyed = true;
      },
      copyFrom: ({ source, yOffset }) => texture.copies.push({ yOffset, height: source.height }),
    };
    textures.push(texture);
    return texture;
  };
  return {
    textures,
    createUintQuads: (_c, w, h, d) => make(w, h, d),
    createBytes: (_c, w, h, d) => make(w, h, d),
    vec4: (x, y, z, w) => ({ vec4: [x, y, z, w] }),
  };
}

function build(
  hook: { addToShader: SplatVertexVisibility["addToShader"] },
  uniforms: Record<string, () => unknown> = {},
): { lines: string; uniforms: Record<string, () => unknown> } {
  const out: string[] = [];
  const builder: SplatShaderBuilder = {
    addUniform: (type, name) => out.push(`uniform ${type} ${name};`),
    addVertexLines: (text) => out.push(...(typeof text === "string" ? [text] : text)),
  };
  hook.addToShader(builder, uniforms, { fake: "context" });
  return { lines: out.join("\n"), uniforms };
}

/** A patched-engine primitive double in incremental mode, holding `slots`. */
function fakeScene(
  slots: { positions: Float32Array; start: number }[],
  highWater: number,
): {
  primitive: InstancePrimitive & {
    _positions: Float32Array;
    _numSplats: number;
    _snapshot: { generation: number };
    _tileSlots: Map<object, { start: number; count: number }>;
  };
  tileset: SplatTilesetLike;
  contents: object[];
} {
  const positions = new Float32Array(highWater * 3);
  const tileSlots = new Map<object, { start: number; count: number }>();
  const contents: object[] = [];
  for (const slot of slots) {
    positions.set(slot.positions, slot.start * 3);
    const content = { _lastSplatTransform: IDENTITY, positions: slot.positions };
    contents.push(content);
    tileSlots.set({ content }, { start: slot.start, count: slot.positions.length / 3 });
  }
  const primitive = {
    vertexVisibility: undefined as SplatVertexVisibility | undefined,
    vertexColor: undefined as SplatVertexColor | undefined,
    _positions: positions,
    _numSplats: highWater,
    _snapshot: { generation: 1 },
    _tileSlots: tileSlots,
    isDestroyed: () => false,
  };
  return {
    primitive,
    tileset: { gaussianSplatPrimitive: primitive },
    contents,
  };
}

describe("the GPU state", () => {
  it("sizes the textures", () => {
    expect(idTextureRows(0)).toBe(1);
    expect(idTextureRows(SPLATS_PER_ID_ROW + 1)).toBe(2);
    expect(stateTextureRows(INSTANCE_TEXTURE_WIDTH - 1)).toBe(1);
    expect(stateTextureRows(INSTANCE_TEXTURE_WIDTH)).toBe(2);
  });

  it("writes hidden to r and highlighted to g, one texel an id", () => {
    const state = writeStateTexels(new Uint8Array(4 * 8), 4, [2, 9], [3]);
    expect([...state.subarray(8, 16)]).toEqual([255, 0, 0, 0, 0, 255, 0, 0]);
    expect(state.reduce((a, b) => a + b, 0)).toBe(510);
  });

  it("hides, tints and dims as the shader does", () => {
    const state = writeStateTexels(new Uint8Array(4 * 8), 4, [1], [2]);
    const color = [0.5, 0.5, 0.5, 0.8] as const;
    const on = { active: true, highlightActive: true };
    expect(evaluateInstanceShading(1, state, 4, on, color).visibility).toBe(0);
    const lit = evaluateInstanceShading(2, state, 4, on, color);
    expect(lit.visibility).toBe(1);
    expect(lit.color[0]).toBeGreaterThan(0.5);
    expect(lit.color[3]).toBe(0.8);
    const dim = evaluateInstanceShading(3, state, 4, on, color);
    expect(dim.color).toEqual([0.175, 0.175, 0.175, 0.4]);
    expect(evaluateInstanceShading(0, state, 4, on, color).color).toEqual(dim.color);
    expect(
      evaluateInstanceShading(3, state, 4, { active: true, highlightActive: false }, color).color,
    ).toEqual([...color]);
    expect(
      evaluateInstanceShading(1, state, 4, { active: false, highlightActive: true }, color),
    ).toEqual({ visibility: 1, color: [...color] });
  });
});

describe("the hooks", () => {
  it("declares the hide function for the chain, and the colour function for vertexColor", () => {
    const { tileset } = fakeScene([], 1);
    const hook = new SplatInstances(doc(), fakeGpu(), tileset);
    const vis = build(hook);
    expect(vis.lines).toContain("float splatInstanceVisibility(uint splatIndex, vec3 position)");
    expect(vis.lines).not.toContain("splatVertexColor");
    const col = build(hook.colorHook);
    expect(col.lines).toContain(
      "vec4 splatVertexColor(uint splatIndex, vec3 position, vec4 color)",
    );
    expect(col.lines).not.toContain("splatInstanceVisibility");
    // Shared helpers are guarded, so both hooks in one shader declare them once.
    for (const lines of [vis.lines, col.lines])
      expect(lines).toContain("#ifndef HEXAPOD_SPLAT_INSTANCES");
    for (const name of [
      "u_instanceIds",
      "u_instanceState",
      "u_instanceParams",
      "u_instanceTint",
      "u_instanceDim",
    ]) {
      expect(vis.uniforms[name]).toBeTypeOf("function");
    }
  });

  it("puts each tile's ids at its slots, and acts only once something is hidden", () => {
    const { primitive, tileset } = fakeScene(
      [
        { positions: TILE_A.positions, start: 0 },
        { positions: TILE_B.positions, start: 8 },
      ],
      12,
    );
    const gpu = fakeGpu();
    const hook = new SplatInstances(doc(), gpu, tileset);
    expect(hook.install(primitive)).toBe(true);
    expect(hook.installed).toBe(true);
    expect(visibilityChainOf(primitive)?.parts).toEqual([hook]);
    expect(primitive.vertexColor).toBe(hook.colorHook);
    const { uniforms } = build(primitive.vertexVisibility ?? hook);

    const result = hook.sync();
    expect(result).toEqual({ changed: true, tiles: 2, matched: 2 });
    const ids = Array.from({ length: 12 }, (_, i) => hook.idAt(i));
    expect(ids).toEqual([1, 1, 0, 3, 3, 3, 0, 0, 4, 4, 4, 4]);
    expect(hook.active).toBe(false);
    expect(uniforms.u_instanceParams?.()).toEqual({ vec4: [0, 4, 12, 0] });

    hook.setState(new Set([2]), new Set(), true);
    expect(hook.active).toBe(true);
    // Hiding 2 hides its child 3.
    const state = gpu.textures.at(-1)?.data as Uint8Array;
    expect([state[2 * 4], state[3 * 4], state[1 * 4]]).toEqual([255, 255, 0]);
    expect(uniforms.u_instanceParams?.()).toEqual({ vec4: [1, 4, 12, 0] });
    hook.setState(new Set(), new Set([4]), false);
    expect(uniforms.u_instanceParams?.()).toEqual({ vec4: [1, 4, 12, 1] });
    expect(uniforms.u_instanceDim?.()).toEqual({ vec4: [1, 1, 0, 0] });

    // Nothing moved: nothing to do.
    expect(hook.sync().changed).toBe(false);
    hook.enabled = false;
    expect(hook.active).toBe(false);

    hook.destroy();
    expect(primitive.vertexVisibility).toBeUndefined();
    expect(primitive.vertexColor).toBeUndefined();
    expect(gpu.textures.every((t) => t.isDestroyed())).toBe(true);
  });

  it("zeroes a tile that leaves and a tile the file does not list", () => {
    const unlisted = tilePositions(3, 500);
    const scene = fakeScene(
      [
        { positions: TILE_A.positions, start: 0 },
        { positions: unlisted.positions, start: 6 },
      ],
      9,
    );
    const hook = new SplatInstances(doc(), fakeGpu(), scene.tileset);
    build(hook);
    expect(hook.sync()).toEqual({ changed: true, tiles: 2, matched: 1 });
    expect(Array.from({ length: 9 }, (_, i) => hook.idAt(i))).toEqual([1, 1, 0, 3, 3, 3, 0, 0, 0]);
    // Tile A leaves; its slots are freed.
    const [first] = scene.primitive._tileSlots.keys();
    if (first) scene.primitive._tileSlots.delete(first);
    scene.primitive._snapshot = { generation: 2 };
    hook.sync();
    expect(Array.from({ length: 6 }, (_, i) => hook.idAt(i))).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it("hiding composes with the view cones in one visibility chain", () => {
    const meta: ViewConesMeta = {
      format: "hexapod.viewcones",
      version: 1,
      uri: "viewcones.bin",
      origin: [0, 0, 0],
      cell: 1,
      dims: [2, 2, 2],
      fadeDeg: 20,
    };
    const coneGpu: ViewConeGpu = {
      createBytes: () => ({
        copyFrom: () => undefined,
        isDestroyed: () => false,
        destroy: () => undefined,
      }),
      mat4: (values) => ({ mat4: [...values] }),
      vec4: (x, y, z, w) => ({ vec4: [x, y, z, w] }),
      vertexDestination: 1,
    };
    const { primitive, tileset } = fakeScene([{ positions: TILE_A.positions, start: 0 }], 6);
    const cones = new SplatViewCones(meta, new Uint8Array(cellCount(meta) * 4), coneGpu);
    const objects = new SplatInstances(doc(), fakeGpu(), tileset);
    expect(cones.install(primitive)).toBe(true);
    expect(objects.install(primitive)).toBe(true);
    const chain = visibilityChainOf(primitive);
    // Cheapest first, whatever the install order.
    expect(chain?.parts).toEqual([objects, cones]);
    const { lines } = build(primitive.vertexVisibility ?? objects);
    expect(lines).toContain("float splatInstanceVisibility(");
    expect(lines).toContain("float splatViewConeVisibility(");
    const composed = lines.slice(lines.indexOf("float splatVertexVisibility("));
    expect(composed.indexOf("splatInstanceVisibility")).toBeLessThan(
      composed.indexOf("splatViewConeVisibility"),
    );
    expect(lines.match(/float splatVertexVisibility\(/g)).toHaveLength(1);

    // Either leaves without taking the other with it.
    objects.destroy();
    expect(visibilityChainOf(primitive)?.parts).toEqual([cones]);
    expect(cones.installed).toBe(true);
    expect(objects.install(primitive)).toBe(true);
    cones.destroy();
    expect(visibilityChainOf(primitive)?.parts).toEqual([objects]);
    objects.destroy();
    expect(primitive.vertexVisibility).toBeUndefined();
  });

  it("multiplies parts in order, and leaves a slot it does not own alone", () => {
    expect(composeVisibilityGlsl(["a", "b"])).toMatch(
      /weight \*= a\(splatIndex, position\);[\s\S]*weight \*= b\(splatIndex, position\);/,
    );
    const foreign: SplatVertexVisibility = { addToShader: () => undefined };
    const primitive = { vertexVisibility: foreign, isDestroyed: () => false };
    const part: SplatVisibilityPart = {
      visibilityFunction: "f",
      visibilityOrder: 0,
      addToShader: () => undefined,
    };
    expect(addVisibilityPart(primitive, part)).toBe(false);
    removeVisibilityPart(primitive, part);
    expect(primitive.vertexVisibility).toBe(foreign);
    expect(addVisibilityPart({ isDestroyed: () => false }, part)).toBe(false);
  });
});

describe("attachInstances", () => {
  beforeEach(() => useInstances.setState({ assets: {}, dimOthers: true }));

  it("loads what the root declares into the store, and clears it on dispose", async () => {
    const { primitive } = fakeScene([{ positions: TILE_A.positions, start: 0 }], 6);
    const tileset = {
      root: { extras: { instances: { uri: "instances.json", count: 4 } } },
      resource: { url: "https://x.test/scan/tileset.json" },
      gaussianSplatPrimitive: primitive,
    } as unknown as Cesium3DTileset;
    const listeners: (() => void)[] = [];
    let renders = 0;
    const scene = {
      preUpdate: {
        addEventListener: (f: () => void) => {
          listeners.push(f);
          return () => listeners.splice(listeners.indexOf(f), 1);
        },
      },
      requestRender: () => {
        renders += 1;
      },
    } as unknown as Pick<Scene, "preUpdate" | "requestRender">;
    const asked: string[] = [];
    const dispose = attachInstances(tileset, scene, "scan", fakeGpu(), (url, ref) => {
      asked.push(`${url} ${ref.uri}`);
      return Promise.resolve(doc());
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(asked).toEqual(["https://x.test/scan/tileset.json instances.json"]);
    expect(useInstances.getState().assets.scan?.instances).toHaveLength(4);
    for (const f of listeners) f();
    expect(primitive.vertexVisibility).toBeDefined();
    const before = renders;
    useInstances.getState().toggleHidden("scan", 1);
    expect(renders).toBe(before + 1);
    dispose();
    expect(useInstances.getState().assets.scan).toBeUndefined();
    expect(listeners).toHaveLength(0);
    expect(primitive.vertexVisibility).toBeUndefined();
  });

  it("costs nothing without a declaration", () => {
    const tileset = { root: { extras: {} }, resource: { url: "x" } } as unknown as Cesium3DTileset;
    const scene = {} as Pick<Scene, "preUpdate" | "requestRender">;
    expect(attachInstances(tileset, scene, "x", fakeGpu())()).toBeUndefined();
  });
});

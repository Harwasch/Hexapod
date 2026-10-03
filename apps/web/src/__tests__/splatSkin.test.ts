/**
 * Scene-object skins on the GPU path (`lib/skin.ts`, `cesium/splatSkin.ts`): the committed
 * yard fixture parses and fits the yard's tiles, the handles fold into the baked frame
 * exactly, the per-splat ids and weight rows land where the snapshot draws each tile, the
 * shader's arithmetic (restated in `evaluateSkinMotion`) is the format's, and the skin
 * composes with the rig in the motion chain with the Jacobian declared only when wanted.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { checksumPositions } from "@twin/world";
import { beforeEach, describe, expect, it } from "vitest";

import { invertAffine, transformPoint, type MutableVec3 } from "@/cesium/splatFrames";
import { SplatGpuMotion } from "@/cesium/splatGpuMotion";
import {
  composeMotionGlsl,
  hasMotionPart,
  motionChainOf,
  type SplatMotionPart,
} from "@/cesium/splatMotionChain";
import {
  evaluateSkinMotion,
  foldHandle,
  skinGlsl,
  SplatSkinning,
  TEXELS_PER_SKIN,
} from "@/cesium/splatSkin";
import { parseInstances, tileInstanceIds, withDescendants } from "@/lib/instances";
import {
  HANDLE_FLOATS,
  parseSkin,
  rigidHandle,
  rowWeight,
  skinDisplacement,
  skinRefOf,
  tileSkin,
  type SkinDoc,
} from "@/lib/skin";

import { buildDrawCommand, fakeFactory, FakeHookedPrimitive } from "./splatGpuFixture";
import { FakeTile, FakeTiledTileset } from "./splatTilesFixture";
import { yardBake, yardTiles } from "./splatYardFixture";

const YARD = resolve(process.cwd(), "../../data/tiles/synthetic-yard");

function fixture(): { raw: unknown; blob: ArrayBuffer } {
  const raw: unknown = JSON.parse(readFileSync(resolve(YARD, "skin/skin.json"), "utf8"));
  const bytes = Uint8Array.from(readFileSync(resolve(YARD, "skin/skin.bin")));
  return { raw, blob: bytes.buffer };
}

function yardDoc(): SkinDoc {
  const { raw, blob } = fixture();
  const doc = parseSkin(raw, blob);
  if (!doc) throw new Error("the fixture is not a skin");
  return doc;
}

const instances = parseInstances(
  JSON.parse(readFileSync(resolve(YARD, "instances/instances.json"), "utf8")),
);

/** A deterministic generator in [-1, 1). */
function random(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 31 - 1;
  };
}

describe("skin.json and skin.bin", () => {
  it("reads the committed yard skin cleanly, one skin per object", () => {
    const doc = yardDoc();
    expect(doc.issues).toEqual([]);
    expect(doc.skins.map((s) => s.instance)).toEqual([1, 9, 10, 12]);
    expect(doc.tiles.size).toBe(yardTiles.size);
    expect(doc.rows * 16).toBe(fixture().blob.byteLength);
    for (const skin of doc.skins) {
      expect(skin.handles).toBeGreaterThanOrEqual(8);
      expect(skin.handles).toBeLessThanOrEqual(16);
      expect(skin.eigenvalues).toHaveLength(skin.handles - 1);
      expect(skin.support).toHaveLength(skin.handles - 1);
    }
    expect(skinRefOf({ skin: { uri: "skin.json", count: 4 } })).toEqual({
      uri: "skin.json",
      count: 4,
    });
    expect(skinRefOf({})).toBeNull();
  });

  it("skins exactly the splats of each object's subtree, every field peaking at ±1", () => {
    const doc = yardDoc();
    if (!instances) throw new Error("no instances");
    const members = new Map(
      doc.skins.map((s) => [s.id, withDescendants(instances, [s.instance])] as const),
    );
    const peaks = new Map(doc.skins.map((s) => [s.id, new Float64Array(s.handles - 1)] as const));
    for (const tile of yardTiles.values()) {
      const checksum = checksumPositions(tile.local);
      const decoded = tileSkin(doc, checksum);
      const ids = tileInstanceIds(instances, checksum);
      if (!decoded || !ids) throw new Error(`${tile.uri} is not listed`);
      expect(decoded.skins.length).toBe(tile.local.length / 3);
      for (let i = 0; i < ids.length; i += 1) {
        const skin = decoded.skins[i] ?? 0;
        const owner = [...members].find(([, set]) => set.has(ids[i] ?? 0))?.[0] ?? 0;
        expect(skin).toBe(owner);
        if (skin === 0) {
          expect(Array.from(decoded.words.subarray(i * 4, i * 4 + 4))).toEqual([0, 0, 0, 0]);
          continue;
        }
        const peak = peaks.get(skin);
        if (!peak || !tile.leaf) continue;
        for (let k = 0; k < peak.length; k += 1) {
          peak[k] = Math.max(peak[k] ?? 0, rowWeight(decoded.words, i, k, doc.scale));
        }
      }
    }
    for (const peak of peaks.values()) for (const p of peak) expect(p).toBeCloseTo(1, 6);
  });

  it("decodes signed bytes as the shader does", () => {
    const words = new Uint32Array([0x80ff7f01, 0, 0, 0x000000fe]);
    expect(rowWeight(words, 0, 0, 1)).toBe(1);
    expect(rowWeight(words, 0, 1, 1)).toBe(127);
    expect(rowWeight(words, 0, 2, 1)).toBe(-1);
    expect(rowWeight(words, 0, 3, 1)).toBe(-128);
    expect(rowWeight(words, 0, 12, 1)).toBe(-2);
    // The GLSL is the same expression: shift the byte to the top, sign-extend down.
    expect(skinGlsl(1 / 127)).toContain("int(word << uint(24 - 8 * (k & 3))) >> 24");
  });

  it("refuses what is not a skin and drops tiles whose rows are missing", () => {
    const { raw, blob } = fixture();
    expect(parseSkin({ ...(raw as object), format: "hexapod.instances" }, blob)).toBeNull();
    expect(parseSkin({ ...(raw as object), version: 2 }, blob)).toBeNull();
    const short = parseSkin(raw, blob.slice(0, 16 * 100));
    expect(short?.tiles.size).toBeLessThan(yardTiles.size);
    expect(short?.issues.some((i) => i.includes("rows are not in skin.bin"))).toBe(true);
    const bad = parseSkin(
      {
        ...(raw as object),
        skins: [
          ...(raw as { skins: unknown[] }).skins,
          { id: 9, instance: 3, handles: 17, origin: [0, 0, 0] },
        ],
      },
      blob,
    );
    expect(bad?.skins).toHaveLength(4);
    expect(bad?.issues).toHaveLength(1);
  });
});

describe("handles", () => {
  it("fold into the baked frame: B·(x_c + Z[x_c − o; 1]) = x_b + Z_b[x_b; 1]", () => {
    const rnd = random(3);
    const inverse = invertAffine(yardBake);
    if (!inverse) throw new Error("singular bake");
    const origin: [number, number, number] = [12, 5, -0.2];
    const z = Array.from({ length: 12 }, () => rnd() * 0.3);
    const rows = new Float32Array(12);
    foldHandle(z, 0, origin, yardBake, inverse, rows, 0);
    for (let n = 0; n < 5; n += 1) {
      const xc: [number, number, number] = [rnd() * 20, rnd() * 20, rnd() * 5];
      const d = skinDisplacement(z, 1, origin, [], xc);
      const want = transformPoint(yardBake, xc[0] + d[0], xc[1] + d[1], xc[2] + d[2], [0, 0, 0]);
      const xb = transformPoint(yardBake, xc[0], xc[1], xc[2], [0, 0, 0]);
      for (let r = 0; r < 3; r += 1) {
        const got =
          (xb[r] ?? 0) +
          (rows[r * 4] ?? 0) * xb[0] +
          (rows[r * 4 + 1] ?? 0) * xb[1] +
          (rows[r * 4 + 2] ?? 0) * xb[2] +
          (rows[r * 4 + 3] ?? 0);
        expect(got).toBeCloseTo(want[r] ?? 0, 4);
      }
    }
  });

  it("the constant handle as [R − I | t] is a rigid motion", () => {
    // rigidHandle expects a unit quaternion.
    const q: [number, number, number, number] = [0.3, -0.1, 0.5, 0.8];
    const norm = Math.hypot(...q);
    const z = rigidHandle([q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm], [1, 2, 3]);
    const o: [number, number, number] = [1, 1, 0];
    const points: [number, number, number][] = [
      [0, 0, 0],
      [3, 1, 2],
      [-2, 4, 1],
    ];
    const moved = points.map((p) => {
      const d = skinDisplacement(z, 1, o, [], p);
      return [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
    });
    const dist = (a: number[], b: number[]): number =>
      Math.hypot((a[0] ?? 0) - (b[0] ?? 0), (a[1] ?? 0) - (b[1] ?? 0), (a[2] ?? 0) - (b[2] ?? 0));
    for (const [i, j] of [
      [0, 1],
      [1, 2],
      [0, 2],
    ] as const) {
      expect(dist(moved[i] ?? [], moved[j] ?? [])).toBeCloseTo(
        dist(points[i] ?? [], points[j] ?? []),
        12,
      );
    }
  });
});

describe("the skin part", () => {
  let doc: SkinDoc;
  let factory: ReturnType<typeof fakeFactory>;
  let primitive: FakeHookedPrimitive;
  let skinning: SplatSkinning;
  const uris = [...yardTiles.keys()].filter((uri) => yardTiles.get(uri)?.leaf);
  const tiles = uris.map(
    (uri) => new FakeTile(uri, yardTiles.get(uri)?.local ?? new Float32Array(0), yardBake),
  );

  beforeEach(() => {
    doc = yardDoc();
    factory = fakeFactory();
    primitive = new FakeHookedPrimitive();
    skinning = new SplatSkinning(doc, factory, new FakeTiledTileset(primitive));
    primitive.commit(tiles);
    expect(skinning.sync().matched).toBe(tiles.length);
    buildDrawCommand(primitive);
  });

  it("writes each tile's skins and rows at its range", () => {
    let start = 0;
    for (const tile of tiles) {
      const decoded = tileSkin(
        doc,
        checksumPositions(yardTiles.get(tile.uri)?.local ?? new Float32Array(0)),
      );
      if (!decoded) throw new Error("unlisted");
      for (let i = 0; i < decoded.skins.length; i += 97) {
        expect(skinning.skinAt(start + i)).toBe(decoded.skins[i]);
        expect(Array.from(skinning.wordsAt(start + i))).toEqual(
          Array.from(decoded.words.subarray(i * 4, i * 4 + 4)),
        );
      }
      start += decoded.skins.length;
    }
    // Ids, weights and handles: three textures, the handles one 1024-texel row per 16 skins.
    expect(factory.made).toHaveLength(3);
    expect(factory.made[2]?.width).toBe(1024);
    expect(factory.made[2]?.height).toBe(1);
  });

  it("is still at rest, and acts only while a skin is driven", () => {
    expect(skinning.active).toBe(false);
    const skin = doc.byInstance.get(1);
    if (!skin) throw new Error("no tree skin");
    const z = new Float64Array(skin.handles * HANDLE_FLOATS);
    z[HANDLE_FLOATS + 3] = 0.5;
    skinning.setInstanceHandles(1, z);
    expect(skinning.active).toBe(true);
    // Only the driven skin's header says it moves.
    const head = (id: number): number => skinning.handleData[id * TEXELS_PER_SKIN * 4] ?? 0;
    expect(head(skin.id)).toBe(1);
    expect(doc.skins.filter((s) => s.id !== skin.id).map((s) => head(s.id))).toEqual([0, 0, 0]);
    skinning.rest();
    expect(skinning.active).toBe(false);
    expect(head(skin.id)).toBe(0);
  });

  it("displaces each splat as the format says, in the baked frame", () => {
    const rnd = random(7);
    const skin = doc.byInstance.get(1);
    if (!skin) throw new Error("no tree skin");
    const z = Float64Array.from({ length: skin.handles * HANDLE_FLOATS }, () => rnd() * 0.05);
    skinning.setHandles(skin.id, z);
    const local = new Float32Array(tiles.reduce((n, t) => n + t.content.pointsLength * 3, 0));
    let at = 0;
    for (const tile of tiles) {
      local.set(yardTiles.get(tile.uri)?.local ?? [], at);
      at += tile.content.pointsLength * 3;
    }
    let checked = 0;
    const l = (r: number, c: number): number => yardBake[c * 4 + r] ?? 0;
    for (let i = 0; i < primitive._numSplats; i += 7) {
      if (skinning.skinAt(i) !== skin.id) continue;
      const words = skinning.wordsAt(i);
      const learned = Array.from({ length: skin.handles - 1 }, (_, k) =>
        rowWeight(words, 0, k, doc.scale),
      );
      const xc: [number, number, number] = [
        local[i * 3] ?? 0,
        local[i * 3 + 1] ?? 0,
        local[i * 3 + 2] ?? 0,
      ];
      const dc = skinDisplacement(z, skin.handles, skin.origin, learned, xc);
      const xb: MutableVec3 = [
        primitive._positions[i * 3] ?? 0,
        primitive._positions[i * 3 + 1] ?? 0,
        primitive._positions[i * 3 + 2] ?? 0,
      ];
      const got = evaluateSkinMotion(skinning.handleData, skin.id, words, doc.scale, xb);
      for (let r = 0; r < 3; r += 1) {
        const want = l(r, 0) * dc[0] + l(r, 1) * dc[1] + l(r, 2) * dc[2];
        expect(Math.abs((got.displacement[r] ?? 0) - want)).toBeLessThan(2e-4);
      }
      checked += 1;
    }
    expect(checked).toBeGreaterThan(500);
    // A splat of another skin, or none, does not move.
    const other = [...Array(primitive._numSplats).keys()].find(
      (i) => skinning.skinAt(i) !== skin.id,
    );
    const still = evaluateSkinMotion(
      skinning.handleData,
      skinning.skinAt(other ?? 0),
      skinning.wordsAt(other ?? 0),
      doc.scale,
      [0, 0, 0],
    );
    expect(still.displacement).toEqual([0, 0, 0]);
  });

  it("re-uploads only the driven skin's row of handles", () => {
    const handles = factory.made[2];
    if (!handles) throw new Error("no handle texture");
    const writes = handles.uploads.length;
    skinning.setInstanceHandles(10, new Float64Array(12).fill(0.01));
    expect(handles.uploads.length).toBe(writes + 1);
    expect(handles.uploads.at(-1)?.height).toBe(1);
  });

  it("does nothing for the same handles again (a driver at a held clock)", () => {
    const handles = factory.made[2];
    if (!handles) throw new Error("no handle texture");
    const z = new Float64Array(12).fill(0.01);
    skinning.setInstanceHandles(10, z);
    const version = skinning.motionVersion;
    const writes = handles.uploads.length;
    skinning.setInstanceHandles(10, Float64Array.from(z));
    expect(skinning.motionVersion).toBe(version);
    expect(handles.uploads.length).toBe(writes);
    skinning.setInstanceHandles(10, null);
    const rested = skinning.motionVersion;
    expect(rested).toBe(version + 1);
    skinning.setInstanceHandles(10, null);
    expect(skinning.motionVersion).toBe(rested);
    z[3] = 0.02;
    skinning.setInstanceHandles(10, z);
    expect(skinning.motionVersion).toBe(rested + 1);
  });
});

describe("the motion chain", () => {
  const part = (name: string, jacobian?: string): SplatMotionPart => ({
    motionFunction: name,
    jacobianFunction: jacobian,
    motionOrder: 0,
    addToShader: () => undefined,
  });

  it("adds every part's displacement to the rest position", () => {
    const glsl = composeMotionGlsl(["splatRigMotion", "splatSkinMotion"]);
    expect(glsl).toContain("vec3 splatVertexMotion(uint splatIndex, vec3 position)");
    expect(glsl).toContain("displacement += splatRigMotion(splatIndex, position);");
    expect(glsl).toContain("displacement += splatSkinMotion(splatIndex, position);");
    expect(glsl).toContain("return position + displacement;");
    expect(glsl).not.toContain("HAS_SPLAT_VERTEX_JACOBIAN");
    const withJacobian = composeMotionGlsl(["a"], ["aJ"]);
    expect(withJacobian).toContain("#define HAS_SPLAT_VERTEX_JACOBIAN");
    expect(withJacobian).toContain("mat3 splatVertexJacobian(uint splatIndex, vec3 position)");
    expect(withJacobian).toContain("jacobian += aJ(splatIndex, position);");
    expect(part("x").jacobianFunction).toBeUndefined();
  });

  it("holds the rig and a skin together, and the Jacobian follows `covariance`", () => {
    const doc = yardDoc();
    const factory = fakeFactory();
    const primitive = new FakeHookedPrimitive();
    const rig = new SplatGpuMotion(factory, 4);
    rig.install(primitive);
    const skinning = new SplatSkinning(doc, factory, new FakeTiledTileset(primitive));
    expect(skinning.install(primitive)).toBe(true);
    expect(motionChainOf(primitive)?.parts.map((p) => p.motionFunction)).toEqual([
      "splatRigMotion",
      "splatSkinMotion",
    ]);
    const lines = (): string => {
      const out: string[] = [];
      primitive.vertexMotion?.addToShader(
        {
          addUniform: (type, name) => out.push(`uniform ${type} ${name};`),
          addVertexLines: (text) => out.push(...(typeof text === "string" ? [text] : text)),
        },
        {},
        { fake: "context" },
      );
      return out.join("\n");
    };
    const both = lines();
    for (const needle of [
      "vec3 splatRigMotion(uint splatIndex, vec3 position)",
      "vec3 splatSkinMotion(uint splatIndex, vec3 position)",
      "uniform highp usampler2D u_skinWeights;",
      "uniform highp sampler2D u_skinHandles;",
      "#define HAS_SPLAT_VERTEX_JACOBIAN",
      "jacobian += splatSkinJacobian(splatIndex, position);",
    ]) {
      expect(both).toContain(needle);
    }
    // Function definitions come before the chain that calls them.
    expect(both.indexOf("vec3 splatSkinMotion(")).toBeLessThan(
      both.indexOf("vec3 splatVertexMotion("),
    );
    const slot = primitive.vertexMotion;
    skinning.covariance = false;
    expect(lines()).not.toContain("HAS_SPLAT_VERTEX_JACOBIAN");
    expect(primitive.vertexMotion).toBe(slot);
    skinning.uninstall();
    expect(hasMotionPart(primitive, skinning)).toBe(false);
    expect(hasMotionPart(primitive, rig)).toBe(true);
    rig.destroy();
    expect(primitive.vertexMotion).toBeUndefined();
  });
});

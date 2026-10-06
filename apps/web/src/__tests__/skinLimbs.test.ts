/**
 * Limbs skins in the viewer (docs/SCENE_OBJECTS.md §9, "Limbs"): `skin.json`'s `limbs` block is
 * read only for a document whose `method.name` is `limbs` and only when whole; the wind drives
 * such a skin with the plant's own per-limb model (`limbWind.ts`), never as eigenmodes; the
 * leaf flutter that follows its handles is drawn by CesiumJS's part (folded into the baked
 * frame) and by the dedicated renderers' shared tables, splat for splat what the format says;
 * and every older skin parses, drives and draws exactly as before.
 *
 * The case is the committed yard's skin, read as a limbs skin: its tree's 13 learned handles
 * given limb records and every row's last byte (which no weight uses) a flutter share.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  LIMB_FLUTTER_FLOATS,
  limbFlutterOffset,
  limbHandles,
  limbWindFromSettings,
  limbWindModel,
  SkinWindField,
  type WindSettings,
} from "@twin/world";
import { beforeEach, describe, expect, it } from "vitest";

import { buildScanMotion, evaluateScanMotion } from "@/cesium/scanView/scanMotion";
import { SkinWindDriver, type SkinWindTarget } from "@/cesium/skinWind";
import { type MutableVec3 } from "@/cesium/splatFrames";
import {
  carriesFlutter,
  evaluateSkinMotion,
  FLUTTER_TEXEL,
  skinGlsl,
  SplatSkinning,
  TEXELS_PER_SKIN,
} from "@/cesium/splatSkin";
import {
  HANDLE_FLOATS,
  parseSkin,
  rowWeight,
  skinDisplacement,
  skinFloats,
  type SkinDoc,
} from "@/lib/skin";
import type { MaterialTable } from "@/lib/skinMaterials";

import { buildDrawCommand, fakeFactory, FakeHookedPrimitive } from "./splatGpuFixture";
import { FakeTile, FakeTiledTileset } from "./splatTilesFixture";
import { yardBake, yardTiles } from "./splatYardFixture";

const YARD = resolve(process.cwd(), "../../data/tiles/synthetic-yard");
const SHARE = 100; // a row's last byte: 100/127 of the flutter
const BREEZE: WindSettings = { strength: 0.1, bearingDeg: 30 };

interface RawSkin {
  id: number;
  instance: number;
  handles: number;
  origin: [number, number, number];
  limbs?: unknown;
  [key: string]: unknown;
}

function yardRaw(): { method: unknown; skins: RawSkin[]; [key: string]: unknown } {
  return JSON.parse(readFileSync(resolve(YARD, "skin/skin.json"), "utf8")) as {
    method: unknown;
    skins: RawSkin[];
  };
}

/** The yard tree's limb records: a chain up from its origin, each a metre higher. */
function limbBlock(skin: RawSkin): Record<string, unknown> {
  const [x, y, z] = skin.origin;
  return {
    seed: 1,
    referenceSpeedMps: 10,
    treeHeightM: 9.7,
    leafSizeM: 0.05,
    wind: {
      turbulence: { along: 0.6, across: 0.25 },
      lengthScaleM: 40,
      gust: { strength: 0, variance: 0.3, frequencyPerMin: 3, durationS: 5 },
      canopyAdvection: 0.3,
    },
    seasons: { winter: { dampingScale: 0.45, branchFrequencyScale: 2.5, flutterScale: 0 } },
    flutter: { referenceM: 0.006 },
    handles: Array.from({ length: skin.handles - 1 }, (_, k) => ({
      key: k + 1,
      pivot: [x, y, z + k * 0.6],
      parent: k === 0 ? 0 : k,
      level: k === 0 ? 0 : 1,
      spanM: 2,
      frequencyHz: 1 + 0.1 * k,
      damping: k === 0 ? 0.086 : 0.0755,
      tree: k === 0,
      gain: 0.03,
      limitRad: 0.5,
      direction: k === 0 ? [0, 0, 1] : [Math.cos(k), Math.sin(k), 0.3],
      samplePoint: [x, y, z + 3],
      widthM: 2,
      heightM: 3,
      staticTipM: 0.05,
      flutterM: 0.006,
    })),
  };
}

/** The yard's skin as a limbs skin: the tree's limbs, every row's last byte a share. */
function limbsDoc(options: { method?: string; broken?: boolean } = {}): SkinDoc {
  const raw = yardRaw();
  raw.method = { name: options.method ?? "limbs" };
  const tree = raw.skins.find((s) => s.instance === 1);
  if (!tree) throw new Error("no tree");
  const block = limbBlock(tree);
  if (options.broken) (block.handles as unknown[]).pop();
  tree.limbs = block;
  const bytes = Uint8Array.from(readFileSync(resolve(YARD, "skin/skin.bin")));
  for (let row = 15; row < bytes.length; row += 16) bytes[row] = SHARE;
  const doc = parseSkin(raw, bytes.buffer);
  if (!doc) throw new Error("not a skin");
  return doc;
}

function yardDoc(): SkinDoc {
  const doc = parseSkin(
    yardRaw(),
    Uint8Array.from(readFileSync(resolve(YARD, "skin/skin.bin"))).buffer,
  );
  if (!doc) throw new Error("not a skin");
  return doc;
}

describe("a limbs skin's document", () => {
  it("carries its limbs, the plant's wind and the row byte of the flutter share", () => {
    const doc = limbsDoc();
    expect(doc.method).toBe("limbs");
    const tree = doc.byInstance.get(1);
    const limbs = tree?.limbs;
    expect(limbs?.handles).toHaveLength((tree?.handles ?? 0) - 1);
    expect(limbs?.flutterByte).toBe(15);
    expect(limbs?.seed).toBe(1);
    expect(limbs?.origin).toEqual(tree?.origin);
    expect(limbs?.handles[0]?.tree).toBe(true);
    expect(limbs?.handles[3]?.parent).toBe(3);
    expect(tree ? skinFloats(tree) : 0).toBe((tree?.handles ?? 0) * 12 + LIMB_FLUTTER_FLOATS);
    // The other skins of the file have no limbs block: none is made up for them.
    for (const skin of doc.skins.filter((s) => s.instance !== 1)) {
      expect(skin.limbs).toBeUndefined();
      expect(skinFloats(skin)).toBe(skin.handles * 12);
    }
  });

  it("is read only under method limbs, and only when whole", () => {
    expect(limbsDoc({ method: "simplicits-rkpm" }).byInstance.get(1)?.limbs).toBeUndefined();
    const broken = limbsDoc({ broken: true });
    expect(broken.byInstance.get(1)?.limbs).toBeUndefined();
    expect(broken.issues.some((i) => /limbs are incomplete/.test(i))).toBe(true);
    // An older skin reads as before.
    const yard = yardDoc();
    expect(yard.method).toBe("simplicits-rkpm");
    expect(yard.skins.every((s) => s.limbs === undefined)).toBe(true);
  });
});

class FakePart implements SkinWindTarget {
  readonly calls = new Map<number, number[] | null>();
  materials: MaterialTable = new Map();
  constructor(readonly doc: SkinDoc) {}
  setInstanceHandles(instanceId: number, handles: ArrayLike<number> | null): boolean {
    this.calls.set(instanceId, handles === null ? null : Array.from(handles));
    return true;
  }
}

const PLANT = {
  properties: { vegetation: 1, elastic: 0.7, rigid: 0.2 },
  behaviour: "in-place" as const,
};

describe("the wind on a limbs skin", () => {
  it("is the plant's own model, flutter and all; calm hands it null", () => {
    const doc = limbsDoc();
    const part = new FakePart(doc);
    const driver = new SkinWindDriver(part, () => PLANT, new SkinWindField(7));
    const t = 42.5;
    // Only the tree: in a limbs document the yard's other skins have no limbs to sway by.
    expect(driver.tick(t, BREEZE).driven).toBe(1);
    const tree = doc.byInstance.get(1);
    const limbs = tree?.limbs;
    if (!tree || !limbs) throw new Error("no limbs");
    const want = limbHandles(limbWindModel(limbs), t, limbWindFromSettings(BREEZE, limbs));
    const got = part.calls.get(1) ?? [];
    expect(got).toHaveLength(skinFloats(tree));
    got.forEach((v, i) => {
      expect(v).toBeCloseTo(want[i] ?? 0, 15);
    });
    // Rotations about pivots, not translations: its linear parts move; flutter is on.
    expect(Math.abs(got[12] ?? 0) + Math.abs(got[13] ?? 0)).toBeGreaterThan(0);
    expect(got[tree.handles * 12]).toBe(1);
    expect(part.calls.has(9)).toBe(false);
    driver.tick(t + 1, { strength: 0, bearingDeg: 30 });
    expect(part.calls.get(1)).toBeNull();
  });

  it("leaves an older skin's sway as it was: translations from its eigenmodes", () => {
    const doc = yardDoc();
    const part = new FakePart(doc);
    const driver = new SkinWindDriver(part, () => PLANT, new SkinWindField(7));
    expect(driver.tick(42.5, BREEZE).driven).toBe(doc.skins.length);
    for (const skin of doc.skins) {
      const got = part.calls.get(skin.instance) ?? [];
      expect(got).toHaveLength(skin.handles * 12);
      // Z_j = [0 | q_j]: no linear part.
      for (let j = 0; j < skin.handles; j += 1)
        for (const k of [0, 1, 2, 4, 5, 6, 8, 9, 10]) expect(got[j * 12 + k]).toBe(0);
    }
  });

  it("does not sway a limbs skin whose limbs are incomplete, though it has dynamics", () => {
    const doc = limbsDoc({ broken: true });
    expect(doc.byInstance.get(1)?.dynamics).toBeDefined();
    const part = new FakePart(doc);
    const driver = new SkinWindDriver(part, () => PLANT, new SkinWindField(7));
    driver.tick(3, BREEZE);
    expect(part.calls.has(1)).toBe(false);
  });
});

describe("a limbs skin drawn", () => {
  let doc: SkinDoc;
  let primitive: FakeHookedPrimitive;
  let skinning: SplatSkinning;
  const uris = [...yardTiles.keys()].filter((uri) => yardTiles.get(uri)?.leaf);
  const tiles = uris.map(
    (uri) => new FakeTile(uri, yardTiles.get(uri)?.local ?? new Float32Array(0), yardBake),
  );
  const local = (): Float32Array => {
    const out = new Float32Array(tiles.reduce((n, t) => n + t.content.pointsLength * 3, 0));
    let at = 0;
    for (const tile of tiles) {
      out.set(yardTiles.get(tile.uri)?.local ?? [], at);
      at += tile.content.pointsLength * 3;
    }
    return out;
  };

  beforeEach(() => {
    doc = limbsDoc();
    primitive = new FakeHookedPrimitive();
    skinning = new SplatSkinning(doc, fakeFactory(), new FakeTiledTileset(primitive));
    primitive.commit(tiles);
    expect(skinning.sync().matched).toBe(tiles.length);
    buildDrawCommand(primitive);
  });

  it("moves every splat by its limbs and its leaves' flutter, folded into the baked frame", () => {
    const tree = doc.byInstance.get(1);
    const limbs = tree?.limbs;
    if (!tree || !limbs) throw new Error("no limbs");
    const z = limbHandles(limbWindModel(limbs), 61.25, limbWindFromSettings(BREEZE, limbs));
    expect(carriesFlutter(z, tree.handles)).toBe(true);
    skinning.setHandles(tree.id, z);
    expect(skinning.drivenSkins.get(tree.id)).toHaveLength(skinFloats(tree));
    const rest = local();
    const l = (r: number, c: number): number => yardBake[c * 4 + r] ?? 0;
    let checked = 0;
    let fluttered = 0;
    for (let i = 0; i < primitive._numSplats; i += 5) {
      if (skinning.skinAt(i) !== tree.id) continue;
      const words = skinning.wordsAt(i);
      const learned = Array.from({ length: tree.handles - 1 }, (_, k) =>
        rowWeight(words, 0, k, doc.scale),
      );
      const share = rowWeight(words, 0, 15, doc.scale);
      expect(share).toBeCloseTo(SHARE / 127, 12);
      const xc: [number, number, number] = [rest[i * 3] ?? 0, rest[i * 3 + 1] ?? 0, rest[i * 3 + 2] ?? 0];
      const moved = skinDisplacement(z, tree.handles, tree.origin, learned, xc);
      const leaves = limbFlutterOffset(z, tree.handles * HANDLE_FLOATS, share, xc);
      const xb: MutableVec3 = [
        primitive._positions[i * 3] ?? 0,
        primitive._positions[i * 3 + 1] ?? 0,
        primitive._positions[i * 3 + 2] ?? 0,
      ];
      const got = evaluateSkinMotion(skinning.handleData, tree.id, words, doc.scale, xb);
      for (let r = 0; r < 3; r += 1) {
        const want = [0, 1, 2].reduce(
          (sum, c) => sum + l(r, c) * ((moved[c] ?? 0) + (leaves[c] ?? 0)),
          0,
        );
        expect(Math.abs((got.displacement[r] ?? 0) - want)).toBeLessThan(2e-5);
      }
      if (Math.hypot(...leaves) > 1e-4) fluttered += 1;
      checked += 1;
    }
    expect(checked).toBeGreaterThan(300);
    expect(fluttered).toBeGreaterThan(checked / 2);
  });

  it("draws an older skin as before: no flutter texels, the same displacement", () => {
    const skin = doc.byInstance.get(9);
    if (!skin) throw new Error("no snag");
    const z = new Float64Array(skin.handles * HANDLE_FLOATS);
    z[HANDLE_FLOATS + 3] = 0.2;
    skinning.setHandles(skin.id, z);
    const base = skin.id * TEXELS_PER_SKIN * 4;
    const flutter = skinning.handleData.subarray(
      base + FLUTTER_TEXEL * 4,
      base + TEXELS_PER_SKIN * 4,
    );
    expect(flutter.every((v) => v === 0)).toBe(true);
    // Its rows' last byte is a share now, and is never read for it.
    const i = [...Array(primitive._numSplats).keys()].find((k) => skinning.skinAt(k) === skin.id);
    if (i === undefined) throw new Error("no snag splat");
    const words = skinning.wordsAt(i);
    const rest = local();
    const xc: [number, number, number] = [rest[i * 3] ?? 0, rest[i * 3 + 1] ?? 0, rest[i * 3 + 2] ?? 0];
    const learned = Array.from({ length: skin.handles - 1 }, (_, k) => rowWeight(words, 0, k, doc.scale));
    const moved = skinDisplacement(z, skin.handles, skin.origin, learned, xc);
    const xb: MutableVec3 = [
      primitive._positions[i * 3] ?? 0,
      primitive._positions[i * 3 + 1] ?? 0,
      primitive._positions[i * 3 + 2] ?? 0,
    ];
    const got = evaluateSkinMotion(skinning.handleData, skin.id, words, doc.scale, xb);
    const l = (r: number, c: number): number => yardBake[c * 4 + r] ?? 0;
    for (let r = 0; r < 3; r += 1) {
      const want = [0, 1, 2].reduce((sum, c) => sum + l(r, c) * (moved[c] ?? 0), 0);
      expect(Math.abs((got.displacement[r] ?? 0) - want)).toBeLessThan(2e-5);
    }
  });

  it("keeps the flutter under a poke's overlay, and drops it with the wind", () => {
    const tree = doc.byInstance.get(1);
    const limbs = tree?.limbs;
    if (!tree || !limbs) throw new Error("no limbs");
    const z = limbHandles(limbWindModel(limbs), 5, limbWindFromSettings(BREEZE, limbs));
    skinning.setHandles(tree.id, z);
    const poke = new Float64Array(tree.handles * HANDLE_FLOATS);
    poke[HANDLE_FLOATS + 3] = 0.01;
    skinning.setOverlay(tree.id, poke);
    const drawn = skinning.drivenSkins.get(tree.id);
    expect(drawn?.[tree.handles * 12]).toBe(1);
    expect(drawn?.[HANDLE_FLOATS + 3]).toBeCloseTo((z[HANDLE_FLOATS + 3] ?? 0) + 0.01, 15);
    skinning.setHandles(tree.id, null);
    const held = skinning.drivenSkins.get(tree.id);
    expect(held?.[tree.handles * 12]).toBe(0);
    const base = tree.id * TEXELS_PER_SKIN * 4;
    expect(skinning.handleData[base + FLUTTER_TEXEL * 4]).toBe(0);
  });

  it("is the same in the dedicated renderers' tables, in the scan's own frame", () => {
    const tree = doc.byInstance.get(1);
    const limbs = tree?.limbs;
    if (!tree || !limbs) throw new Error("no limbs");
    const z = limbHandles(limbWindModel(limbs), 17, limbWindFromSettings(BREEZE, limbs));
    const motion = buildScanMotion(
      null,
      { doc, drivenSkins: new Map([[tree.id, z]]), covariance: true },
      null,
      new Set([tree.id]),
      new Set(),
    );
    const rest = local();
    let checked = 0;
    for (let i = 0; i < primitive._numSplats; i += 11) {
      if (skinning.skinAt(i) !== tree.id) continue;
      const words = skinning.wordsAt(i);
      const xc: [number, number, number] = [rest[i * 3] ?? 0, rest[i * 3 + 1] ?? 0, rest[i * 3 + 2] ?? 0];
      const learned = Array.from({ length: tree.handles - 1 }, (_, k) =>
        rowWeight(words, 0, k, doc.scale),
      );
      const moved = skinDisplacement(z, tree.handles, tree.origin, learned, xc);
      const leaves = limbFlutterOffset(z, tree.handles * HANDLE_FLOATS, SHARE / 127, xc);
      const got = evaluateScanMotion(motion, { skin: tree.id, words, id: 0 }, xc);
      for (let r = 0; r < 3; r += 1)
        expect(Math.abs((got.delta[r] ?? 0) - (moved[r] ?? 0) - (leaves[r] ?? 0))).toBeLessThan(
          1e-5,
        );
      checked += 1;
    }
    expect(checked).toBeGreaterThan(100);
  });

  it("puts the flutter in the shader of every renderer", () => {
    const glsl = skinGlsl(doc.scale, true);
    expect(glsl).toContain(`base + ${String(FLUTTER_TEXEL)}`);
    expect(glsl).toMatch(/cos\(dot\(wave\.xyz, position\) \+ wave\.w\)/);
  });
});

/**
 * The skin wind driver (`cesium/skinWind.ts`) against a fake skin part: which instances it
 * drives (behaviour priors, `materials.json` overrides), that calm hands every driven skin
 * `null` in the same tick, and that what it writes is the anchored model's handles. Plus the
 * `dynamics` block of `skin.json` and the `materials.json` reader (`lib/skinMaterials.ts`).
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { HANDLE_REACH, SkinWindField, WIND_CALM, type WindSettings } from "@twin/world";
import { describe, expect, it } from "vitest";

import { SkinWindDriver, type InstanceTraits, type SkinWindTarget } from "@/cesium/skinWind";
import { parseSkin, type SkinDoc } from "@/lib/skin";
import { materialsRefOf, parseMaterials, type MaterialTable } from "@/lib/skinMaterials";

const YARD = resolve(process.cwd(), "../../data/tiles/synthetic-yard");
const BREEZE: WindSettings = { strength: 0.1, bearingDeg: 45 };

function yardDoc(): SkinDoc {
  const raw: unknown = JSON.parse(readFileSync(resolve(YARD, "skin/skin.json"), "utf8"));
  const bytes = Uint8Array.from(readFileSync(resolve(YARD, "skin/skin.bin")));
  const doc = parseSkin(raw, bytes.buffer);
  if (!doc) throw new Error("the fixture is not a skin");
  return doc;
}

function yardMaterials(): MaterialTable {
  const table = parseMaterials(
    JSON.parse(readFileSync(resolve(YARD, "skin/materials.json"), "utf8")) as unknown,
  );
  if (!table) throw new Error("the fixture is not a materials document");
  return table;
}

class FakePart implements SkinWindTarget {
  readonly doc: SkinDoc;
  materials: MaterialTable;
  readonly calls: [number, number[] | null][] = [];
  readonly current = new Map<number, number[] | null>();

  constructor(doc: SkinDoc, materials: MaterialTable = new Map()) {
    this.doc = doc;
    this.materials = materials;
  }

  setInstanceHandles(instanceId: number, handles: ArrayLike<number> | null): boolean {
    const copy = handles === null ? null : Array.from(handles);
    this.calls.push([instanceId, copy]);
    this.current.set(instanceId, copy);
    return true;
  }
}

const PLANT: InstanceTraits = {
  properties: { vegetation: 0.98, elastic: 0.95, rigid: 0.1 },
  behaviour: "in-place",
};
const CAR: InstanceTraits = {
  properties: { vehicle: 0.9, rigid: 0.9, movable: 0.95 },
  behaviour: "movable",
};

function traits(table: Record<number, InstanceTraits>): (id: number) => InstanceTraits | undefined {
  return (id) => table[id];
}

function runFor(driver: SkinWindDriver, wind: WindSettings, from: number, frames: number): void {
  for (let k = 0; k <= frames; k += 1) driver.tick(from + k / 60, wind);
}

describe("skin.json dynamics", () => {
  it("are read for every yard skin, m × m as upper triangles", () => {
    for (const skin of yardDoc().skins) {
      const m = skin.handles;
      expect(skin.dynamics?.mass).toHaveLength((m * (m + 1)) / 2);
      expect(skin.dynamics?.anchorGram).toHaveLength((m * (m + 1)) / 2);
      expect(skin.dynamics?.mass[0]).toBeCloseTo(1, 6);
      expect(skin.dynamics?.anchorSplats).toBeGreaterThan(0);
    }
  });

  it("are left out when malformed, and the skin still parses", () => {
    const raw = JSON.parse(readFileSync(resolve(YARD, "skin/skin.json"), "utf8")) as {
      skins: { dynamics?: unknown }[];
    };
    const first = raw.skins[0];
    if (first) first.dynamics = { mass: [1, 2, 3], anchor: { gram: [] } };
    const doc = parseSkin(
      raw,
      Uint8Array.from(readFileSync(resolve(YARD, "skin/skin.bin"))).buffer,
    );
    expect(doc?.skins[0]?.dynamics).toBeUndefined();
    expect(doc?.skins[1]?.dynamics).toBeDefined();
  });
});

describe("materials.json", () => {
  it("keeps only well-formed fields, per instance", () => {
    const table = parseMaterials({
      format: "hexapod.materials",
      version: 1,
      materials: [
        {
          instance: 3,
          stiffness: 5,
          damping: 0.2,
          drag: 0.01,
          wind: true,
          evidence: "fitted-real",
        },
        { instance: 4, stiffness: -1, damping: "x", wind: "yes", evidence: "guess" },
        { instance: 0, stiffness: 2 },
        { instance: 3, stiffness: 9 },
      ],
    });
    expect(table?.get(3)).toEqual({
      stiffness: 5,
      damping: 0.2,
      drag: 0.01,
      wind: true,
      evidence: "fitted-real",
    });
    expect(table?.get(4)).toEqual({});
    expect(table?.has(0)).toBe(false);
    expect(parseMaterials({ format: "hexapod.skin", version: 1, materials: [] })).toBeNull();
    expect(materialsRefOf({ materials: { uri: "materials.json", count: 3 } })).toEqual({
      uri: "materials.json",
      count: 3,
    });
    expect(materialsRefOf({})).toBeNull();
  });

  it("parses the yard's fixture", () => {
    const table = yardMaterials();
    expect([...table.keys()]).toEqual([1, 9, 10]);
    expect(table.get(1)?.wind).toBe(true);
  });
});

describe("the driver", () => {
  it("does nothing at all while the wind is calm", () => {
    const part = new FakePart(yardDoc());
    const driver = new SkinWindDriver(part, traits({ 1: PLANT, 9: PLANT }), new SkinWindField(1));
    for (let k = 0; k < 120; k += 1) {
      const tick = driver.tick(k / 60, WIND_CALM);
      expect(tick.moving).toBe(0);
      expect(tick.changed).toBe(false);
    }
    expect(part.calls).toHaveLength(0);
  });

  it("sways in-place instances, not movable ones nor ones it knows nothing about", () => {
    const part = new FakePart(yardDoc());
    const driver = new SkinWindDriver(
      part,
      traits({ 1: PLANT, 9: CAR, 10: PLANT }),
      new SkinWindField(1),
    );
    runFor(driver, BREEZE, 10, 120);
    expect(driver.material(1)?.wind).toBe(true);
    expect(driver.material(9)?.wind).toBe(false);
    expect(driver.material(12)).toBeUndefined();
    const touched = new Set(part.calls.map(([id]) => id));
    expect([...touched].sort((a, b) => a - b)).toEqual([1, 10]);
    const tree = part.current.get(1);
    expect(tree?.some((v) => v !== 0)).toBe(true);
  });

  it("lets materials.json drive a movable instance, or still an in-place one", () => {
    const part = new FakePart(yardDoc(), yardMaterials());
    // The yard reads every instance movable; the fixture's records turn 1, 9 and 10 on.
    const driver = new SkinWindDriver(
      part,
      traits({ 1: CAR, 9: CAR, 10: CAR, 12: CAR }),
      new SkinWindField(1),
    );
    const tick = driver.tick(5, BREEZE);
    expect(tick.driven).toBe(3);
    runFor(driver, BREEZE, 5, 60);
    expect([...new Set(part.calls.map(([id]) => id))].sort((a, b) => a - b)).toEqual([1, 9, 10]);
    expect(driver.material(1)?.evidence).toBe("prior");
  });

  it("follows a materials table that arrives late: a record turning the wind off wins", () => {
    const part = new FakePart(yardDoc());
    const driver = new SkinWindDriver(part, traits({ 1: PLANT }), new SkinWindField(1));
    runFor(driver, BREEZE, 5, 30);
    expect(part.current.get(1)?.some((v) => v !== 0)).toBe(true);
    part.materials = new Map([[1, { wind: false }]]);
    part.calls.length = 0;
    runFor(driver, BREEZE, 6, 10);
    expect(part.calls).toEqual([[1, null]]);
    expect(driver.material(1)?.wind).toBe(false);
  });

  it("hands every driven skin null the tick the wind drops, then stays quiet", () => {
    const part = new FakePart(yardDoc());
    const driver = new SkinWindDriver(part, traits({ 1: PLANT, 10: PLANT }), new SkinWindField(1));
    runFor(driver, BREEZE, 0, 60);
    part.calls.length = 0;
    const tick = driver.tick(1.1, { ...BREEZE, strength: 0 });
    expect(tick.changed).toBe(true);
    expect(part.calls.sort(([a], [b]) => a - b)).toEqual([
      [1, null],
      [10, null],
    ]);
    part.calls.length = 0;
    expect(driver.tick(1.2, WIND_CALM).changed).toBe(false);
    expect(part.calls).toHaveLength(0);
  });

  it("writes bounded translations only: the linear parts and the vertical stay zero", () => {
    const doc = yardDoc();
    const part = new FakePart(doc);
    const driver = new SkinWindDriver(part, traits({ 1: PLANT }), new SkinWindField(2));
    runFor(driver, { strength: 1, bearingDeg: 270 }, 0, 600);
    const tree = doc.byInstance.get(1);
    const z = part.current.get(1);
    if (!tree || !z) throw new Error("the tree was not driven");
    for (let j = 0; j < tree.handles; j += 1) {
      for (const k of [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]) expect(z[j * 12 + k]).toBe(0);
      const reach = Math.hypot(z[j * 12 + 3] ?? 0, z[j * 12 + 7] ?? 0);
      const radius = j === 0 ? tree.scale : (tree.support[j - 1]?.radius ?? 0);
      expect(reach).toBeLessThanOrEqual(HANDLE_REACH * radius + 1e-12);
    }
  });

  it("is a function of the scene clock: two drivers on the same field agree", () => {
    const run = (): number[] | null | undefined => {
      const part = new FakePart(yardDoc());
      const driver = new SkinWindDriver(part, traits({ 1: PLANT }), new SkinWindField(9));
      runFor(driver, BREEZE, 42, 90);
      return part.current.get(1);
    };
    expect(run()).toEqual(run());
  });
});

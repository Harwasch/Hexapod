/**
 * The change drape on the ground (cesium/changeDrape.ts): one ground primitive per class and
 * confidence, classified onto terrain and 3D tiles alike, never picked; low-confidence areas
 * hatched in their class's colour, confident ones solid; a new map replaces the old, and
 * clearing or destroying leaves nothing behind.
 */
import { ClassificationType, GroundPrimitive, type Scene } from "cesium";
import { beforeAll, describe, expect, it, vi } from "vitest";

import type { ChangeArea, ChangeMap } from "@twin/contracts";

import { CHANGE_DRAPE_TAG, ChangeDrapeLayer, type ChangeDrapeTag } from "@/cesium/changeDrape";

// A material's uniforms are typed against the DOM's image kinds, two of which jsdom lacks.
beforeAll(() => {
  if (!("ImageBitmap" in globalThis))
    vi.stubGlobal(
      "ImageBitmap",
      class ImageBitmap {
        readonly stub = true;
      },
    );
  if (!("OffscreenCanvas" in globalThis))
    vi.stubGlobal(
      "OffscreenCanvas",
      class OffscreenCanvas {
        readonly stub = true;
      },
    );
});

function fakeScene() {
  const added: GroundPrimitive[] = [];
  const scene = {
    groundPrimitives: {
      add: vi.fn((p: GroundPrimitive) => {
        added.push(p);
        return p;
      }),
      remove: vi.fn((p: GroundPrimitive) => {
        const at = added.indexOf(p);
        if (at >= 0) added.splice(at, 1);
        return at >= 0;
      }),
    },
    requestRender: vi.fn(),
    isDestroyed: () => false,
  };
  return { scene: scene as unknown as Scene, added, raw: scene };
}

const square = (x: number): number[][][] => [
  [
    [-123.88 + x * 1e-5, 46.13],
    [-123.88 + (x + 1) * 1e-5, 46.13],
    [-123.88 + (x + 1) * 1e-5, 46.13001],
    [-123.88 + x * 1e-5, 46.13001],
    [-123.88 + x * 1e-5, 46.13],
  ],
];

const area = (cls: ChangeArea["class"], lowConfidence: boolean, x: number) =>
  ({
    class: cls,
    footprint: { type: "MultiPolygon", coordinates: [square(x), square(x + 2)] },
    cells: 2,
    lowConfidence,
  }) as unknown as ChangeArea;

const map = (areas: ChangeArea[]) => ({ areas }) as unknown as ChangeMap;

const tags = (added: GroundPrimitive[]) =>
  added.map((p) => (p as unknown as Record<string, ChangeDrapeTag>)[CHANGE_DRAPE_TAG]);

describe("the change drape", () => {
  it("draws one ground primitive per class and confidence, onto terrain and tiles, unpicked", () => {
    const { scene, added, raw } = fakeScene();
    const layer = new ChangeDrapeLayer(scene);
    layer.show(
      map([
        area("raised", false, 0),
        area("raised", true, 4),
        area("lowered", false, 8),
        area("insufficient", false, 12),
        area("raised", false, 16),
      ]),
    );
    expect(added).toHaveLength(4);
    expect(added.every((p) => p instanceof GroundPrimitive)).toBe(true);
    expect(added.every((p) => p.classificationType === ClassificationType.BOTH)).toBe(true);
    expect(added.every((p) => !p.allowPicking)).toBe(true);
    expect(tags(added)).toEqual([
      { cls: "raised", lowConfidence: false, polygons: 4, hatched: false },
      { cls: "raised", lowConfidence: true, polygons: 2, hatched: true },
      { cls: "lowered", lowConfidence: false, polygons: 2, hatched: false },
      { cls: "insufficient", lowConfidence: false, polygons: 2, hatched: false },
    ]);
    expect(raw.requestRender).toHaveBeenCalled();
    expect(layer.drawn()).toEqual(tags(added));
  });

  it("hatches low-confidence areas in their class's colour, and fills confident ones", () => {
    const { scene, added } = fakeScene();
    new ChangeDrapeLayer(scene).show(map([area("lowered", true, 0), area("lowered", false, 4)]));
    const [solid, hatched] = added.map(
      (p) =>
        (
          p.appearance as unknown as {
            material: { type: string; uniforms: Record<string, unknown> };
          }
        ).material,
    );
    expect(solid?.type).toBe("Color");
    expect(hatched?.type).toBe("ChangeHatch");
    const color = (m: typeof solid) =>
      m?.uniforms.color as { red: number; green: number; blue: number; alpha: number };
    // One hue, the hatched one drawn lighter between its stripes.
    expect([color(hatched)?.red, color(hatched)?.green, color(hatched)?.blue]).toEqual([
      color(solid)?.red,
      color(solid)?.green,
      color(solid)?.blue,
    ]);
  });

  it("replaces what it drew with the next map, and clears and destroys to nothing", () => {
    const { scene, added } = fakeScene();
    const layer = new ChangeDrapeLayer(scene);
    layer.show(map([area("raised", false, 0), area("lowered", false, 4)]));
    layer.show(map([area("insufficient", false, 0)]));
    expect(tags(added).map((t) => t?.cls)).toEqual(["insufficient"]);
    layer.show(map([]));
    expect(added).toHaveLength(0);
    layer.show(map([area("raised", true, 0)]));
    layer.clear();
    expect(added).toHaveLength(0);
    expect(layer.drawn()).toEqual([]);
    layer.show(map([area("raised", true, 0)]));
    layer.destroy();
    expect(added).toHaveLength(0);
    // Destroyed: nothing more is drawn, and a second destroy is harmless.
    layer.show(map([area("raised", true, 0)]));
    expect(added).toHaveLength(0);
    layer.destroy();
  });

  it("keeps a polygon's holes open", () => {
    const { scene, added } = fakeScene();
    const [outer] = square(0);
    const hole = [
      [-123.88 + 0.2e-5, 46.130002],
      [-123.88 + 0.8e-5, 46.130002],
      [-123.88 + 0.8e-5, 46.130008],
      [-123.88 + 0.2e-5, 46.130008],
      [-123.88 + 0.2e-5, 46.130002],
    ];
    const ringed = {
      class: "lowered",
      footprint: { type: "MultiPolygon", coordinates: [[outer, hole]] },
      cells: 1,
      lowConfidence: false,
    } as unknown as ChangeArea;
    new ChangeDrapeLayer(scene).show(map([ringed]));
    const instances = added[0]?.geometryInstances as unknown as {
      geometry: { _polygonHierarchy: { positions: unknown[]; holes: { positions: unknown[] }[] } };
    }[];
    const hierarchy = instances[0]?.geometry._polygonHierarchy;
    expect(hierarchy?.positions).toHaveLength(5);
    expect(hierarchy?.holes).toHaveLength(1);
    expect(hierarchy?.holes[0]?.positions).toHaveLength(5);
  });

  it("leaves a destroyed scene alone when it is cleared after the viewer went", () => {
    const { scene, raw } = fakeScene();
    const layer = new ChangeDrapeLayer(scene);
    layer.show(map([area("raised", false, 0)]));
    Object.assign(raw, { isDestroyed: () => true });
    raw.requestRender.mockClear();
    expect(() => layer.destroy()).not.toThrow();
    expect(raw.groundPrimitives.remove).not.toHaveBeenCalled();
    expect(raw.requestRender).not.toHaveBeenCalled();
  });

  it("skips a ring too short to be a polygon rather than failing the whole class", () => {
    const { scene, added } = fakeScene();
    const broken = {
      class: "raised",
      footprint: {
        type: "MultiPolygon",
        coordinates: [
          [
            [
              [0, 0],
              [1, 1],
            ],
          ],
          square(0),
        ],
      },
      cells: 1,
      lowConfidence: false,
    } as unknown as ChangeArea;
    new ChangeDrapeLayer(scene).show(map([broken]));
    expect(tags(added)).toEqual([
      { cls: "raised", lowConfidence: false, polygons: 1, hatched: false },
    ]);
  });
});

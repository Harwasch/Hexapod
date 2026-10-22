/**
 * One companion list per splat primitive, several owners (cesium/companions.ts): each owner
 * says only its own tilesets, the primitive's array is them all, merged in the order the
 * owners joined, and it is reassigned only when it changes.
 */
import type { Cesium3DTileset } from "cesium";
import { describe, expect, it } from "vitest";

import { companionGroups, setCompanions, type CompanionHost } from "@/cesium/companions";

const tileset = (name: string) => ({ name }) as unknown as Cesium3DTileset;

function host(): CompanionHost & { assigned: number } {
  let list: Cesium3DTileset[] = [];
  const h = {
    assigned: 0,
    get companions() {
      return list;
    },
    set companions(next: Cesium3DTileset[]) {
      h.assigned += 1;
      list = next;
    },
    isDestroyed: () => false,
  };
  return h;
}

describe("a primitive's companions, from several owners", () => {
  const layers = Symbol("layers");
  const dates = Symbol("dates");
  const [a, b, c] = [tileset("a"), tileset("b"), tileset("c")];

  it("merges each owner's list in the order the owners joined", () => {
    const h = host();
    setCompanions(h, layers, [a]);
    setCompanions(h, dates, [b, c], "from-dates");
    expect(h.companions).toEqual([a, b, c]);
    // Another owner's change keeps the first's tilesets, and its place.
    setCompanions(h, dates, [c]);
    expect(h.companions).toEqual([a, c]);
    expect(companionGroups(h).map((g) => [g.owner, g.tilesets, g.from])).toEqual([
      [layers, [a], undefined],
      [dates, [c], undefined],
    ]);
  });

  it("assigns the array only when the merged list changes", () => {
    const h = host();
    setCompanions(h, layers, [a, b]);
    expect(h.assigned).toBe(1);
    setCompanions(h, layers, [a, b]);
    setCompanions(h, dates, []);
    expect(h.assigned).toBe(1);
    setCompanions(h, layers, []);
    expect(h.companions).toEqual([]);
    expect(h.assigned).toBe(2);
    expect(companionGroups(h)).toEqual([]);
  });

  it("leaves a destroyed primitive alone, and knows nothing of an unknown one", () => {
    const h = host();
    h.isDestroyed = () => true;
    setCompanions(h, layers, [a]);
    expect(h.assigned).toBe(0);
    expect(companionGroups(undefined)).toEqual([]);
    expect(companionGroups(host())).toEqual([]);
  });
});

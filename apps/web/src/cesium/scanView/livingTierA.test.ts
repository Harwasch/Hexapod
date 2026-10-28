import { describe, expect, it } from "vitest";
import { tierAFromInstances } from "./livingTierA";
import { motionTier } from "./livingDetail";
const doc = {
  format: "hexapod.instances",
  version: 1,
  tilesEncoding: "rle-u32",
  instances: [
    { id: 1, category: "fixtures", bounds: { min: [0, 0, 0], max: [1, 1, 1] } },
    { id: 2, category: "grass", bounds: { min: [0, 0, -0.4], max: [1, 1, 0.2] } },
  ],
  tiles: { checksum: [1, 2, 2, 3, 0, 1] },
};
describe("automatic Tier A from published segmentation", () => {
  it("binds only vegetation in canonical tile order and pins unknown and static objects", () => {
    const a = tierAFromInstances(doc);
    expect(a.header.nodes).toHaveLength(2);
    expect(motionTier(a.header.detail!)).toBe("A");
    expect(a.header.tiles.checksum).toEqual({ offset: 0, count: 6 });
    expect(Array.from(a.bindings.slice(0, 8))).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    expect(Array.from(a.bindings.slice(8, 12))).toEqual([2, 1, 65535, 0]);
    expect(Array.from(a.bindings.slice(-4))).toEqual([0, 0, 0, 0]);
    expect(a.header.groundM).toBe(-0.4);
    expect(a.header.nodes[1]?.groundM).toBe(-0.4);
  });
  it("rejects malformed or unbounded streams and selects off without vegetation", () => {
    expect(() => tierAFromInstances({ ...doc, tiles: { bad: [2] } })).toThrow();
    expect(() => tierAFromInstances({ ...doc, tiles: { bad: [2, 50000001] } })).toThrow();
    const a = tierAFromInstances({ ...doc, instances: [] });
    expect(motionTier(a.header.detail!)).toBe("off");
    expect(a.bindings.every((n) => n === 0)).toBe(true);
  });
  it("pins ground and wood despite an ambiguous vegetation score", () => {
    const a = tierAFromInstances({
      ...doc,
      instances: [
        ...doc.instances,
        {
          id: 3,
          category: "ground",
          properties: { vegetation: 0.99, rigid: 0.9 },
          bounds: { min: [0, 0, 0], max: [1, 1, 0.1] },
        },
        {
          id: 4,
          category: "wood",
          properties: { vegetation: 0.99, rigid: 0.2 },
          bounds: { min: [0, 0, 0], max: [1, 1, 1] },
        },
      ],
      tiles: { checksum: [3, 2, 4, 2, 2, 3, 0, 1] },
    });
    expect(Array.from(a.bindings.slice(0, 16))).toEqual(new Array(16).fill(0));
    expect(a.bindings[16]).toBeGreaterThan(0);
  });
});

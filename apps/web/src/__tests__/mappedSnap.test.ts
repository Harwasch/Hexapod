import { expect, it, vi } from "vitest";
import { MappedSnapIndex } from "@/features/land/mappedSnap";
import type { LandContextLayer } from "@/state/landContext";
import type { LandPoint } from "@/state/land";

const line: LandContextLayer = {
  id: "drawing-guides",
  title: "Guides",
  features: [
    {
      id: "line",
      label: "Mapped line",
      geometry: {
        type: "LineString",
        coordinates: [
          [-77.052, 38.888],
          [-77.048, 38.888],
        ],
      },
    },
  ],
};
const bounds = { minX: -77.054, minY: 38.886, maxX: -77.046, maxY: 38.89 };
const distance = (target: LandPoint) =>
  Math.hypot((target[0] + 77.05) * 100000, (target[1] - 38.88805) * 100000);

it("snaps to mapped line segments with a screen-space tolerance and removes hidden guides", () => {
  const index = new MappedSnapIndex();
  expect(index.empty).toBe(true);
  expect(index.sync({ [line.id]: line })).toBe(true);
  const hit = index.nearest([-77.05, 38.88805], bounds, distance);
  expect(hit?.point).toEqual([-77.05, 38.888]);
  expect(hit?.label).toBe("Guides: Mapped line");
  expect(index.nearest([-77.05, 38.88805], bounds, () => 13)).toBeNull();
  expect(index.sync({})).toBe(true);
  expect(index.nearest([-77.05, 38.88805], bounds, distance)).toBeNull();
});

it("keeps cached geometry when selection changes and ignores editable previews", () => {
  const index = new MappedSnapIndex();
  expect(index.sync({ [line.id]: line })).toBe(true);
  expect(index.sync({ [line.id]: { ...line, selectedIds: ["line"] } })).toBe(false);
  const draft = { ...line, id: "inventory-draft" };
  index.sync({ [draft.id]: draft });
  expect(index.empty).toBe(true);
  expect(index.sync({ map: { ...line, id: "map", researchArtifactId: "map" } })).toBe(true);
  expect(index.empty).toBe(false);
});

it("retains hole rings, disconnected polygons and point targets", () => {
  const index = new MappedSnapIndex();
  index.sync({
    inventory: {
      id: "inventory",
      title: "Assets",
      features: [
        {
          id: "hole",
          label: "Reserve",
          geometry: {
            type: "MultiPolygon",
            coordinates: [
              [
                [
                  [0, 0],
                  [2, 0],
                  [2, 2],
                  [0, 2],
                  [0, 0],
                ],
                [
                  [0.4, 0.4],
                  [0.6, 0.4],
                  [0.6, 0.6],
                  [0.4, 0.6],
                  [0.4, 0.4],
                ],
              ],
              [
                [
                  [3, 3],
                  [4, 3],
                  [4, 4],
                  [3, 4],
                  [3, 3],
                ],
              ],
            ],
          },
        },
        { id: "point", label: "Marker", geometry: { type: "Point", coordinates: [5, 5] } },
      ],
    },
  });
  const b = { minX: 0, minY: 0, maxX: 6, maxY: 6 };
  const near = (point: LandPoint) => (target: LandPoint) =>
    Math.hypot(target[0] - point[0], target[1] - point[1]) * 100;
  expect(index.nearest([0.5, 0.39], b, near([0.5, 0.39]))?.point).toEqual([0.5, 0.4]);
  expect(index.nearest([3.4, 3.01], b, near([3.4, 3.01]))?.point).toEqual([3.4, 3]);
  expect(index.nearest([5.01, 5], b, near([5.01, 5]))?.point).toEqual([5, 5]);
});

it("indexes large layers so a pointer move projects only nearby segments", () => {
  const index = new MappedSnapIndex();
  index.sync({
    inventory: {
      id: "inventory",
      title: "Assets",
      features: Array.from({ length: 20000 }, (_, i) => ({
        id: String(i),
        label: `Point ${i}`,
        geometry: { type: "Point" as const, coordinates: [i / 1000, 0] },
      })),
    },
  });
  const project = vi.fn(() => 0);
  index.nearest([1, 0], { minX: 0.9999, minY: -0.0001, maxX: 1.0001, maxY: 0.0001 }, project);
  expect(project).toHaveBeenCalledOnce();
});

it("does not invent a cross-world edge for an unsplit dateline segment", () => {
  const index = new MappedSnapIndex();
  index.sync({
    [line.id]: {
      ...line,
      features: [
        {
          id: "dateline",
          label: "Dateline",
          geometry: {
            type: "LineString",
            coordinates: [
              [179, 0],
              [-179, 0],
            ],
          },
        },
      ],
    },
  });
  expect(index.nearest([0, 0], { minX: -1, minY: -1, maxX: 1, maxY: 1 }, () => 0)).toBeNull();
});

import { beforeEach, describe, expect, it } from "vitest";

import type { LandArea, LandCreate } from "@twin/contracts";

import { importBoundary } from "@/features/land/geometry";
import { clearHistory, undo } from "@/state/history";
import { useLand } from "@/state/land";
import { useSelection } from "@/state/selection";

const polygon = {
  type: "Polygon" as const,
  coordinates: [
    [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [0, 0],
    ],
  ],
};
const draft: LandCreate = {
  name: "Land",
  description: "",
  boundary: polygon,
  source: { method: "drawn", label: "Drawn" },
};
const area: LandArea = {
  ...draft,
  id: "area-1",
  description: "",
  revision: 1,
  areaM2: 123,
  perimeterM: 45,
  createdAt: "2026-10-09T00:00:00Z",
  updatedAt: "2026-10-09T00:00:00Z",
};

beforeEach(() => {
  useLand.getState().clear();
  clearHistory();
});

describe("land scope", () => {
  it("persists while a different map feature is inspected", () => {
    useLand.getState().select(area);
    useSelection
      .getState()
      .setSelection({
        kind: "ground",
        title: "Creek",
        longitude: 0,
        latitude: 0,
        height: 0,
        terrainHeight: 0,
        at: 1,
      });
    expect(useLand.getState().active).toEqual(area);
  });
  it("does not undo an old drawing into a new drawing", () => {
    useLand.getState().begin("draw");
    useLand.getState().addPoint([1, 1]);
    useLand.getState().cancel();
    useLand.getState().begin("draw");
    useLand.getState().addPoint([2, 2]);
    undo();
    undo();
    expect(useLand.getState().points).toEqual([]);
  });
  it("preserves the saved boundary while editing and discarding a draft", () => {
    useLand.getState().select(area);
    useLand.getState().propose(draft);
    useLand.getState().updateBoundary({
      type: "Polygon",
      coordinates: [
        [
          [0, 0],
          [2, 0],
          [2, 2],
          [0, 0],
        ],
      ],
    });
    expect(useLand.getState().active?.boundary).toEqual(polygon);
    useLand.getState().cancel();
    expect(useLand.getState().active).toEqual(area);
    expect(useLand.getState().draft).toBeNull();
  });
  it("does not resurrect a discarded draft when another area is selected", () => {
    useLand.getState().propose(draft);
    useLand.getState().updateBoundary(polygon);
    useLand.getState().select(area);
    useLand.getState().propose({ ...draft, name: "Another area" });
    undo();
    expect(useLand.getState().draft?.name).toBe("Another area");
  });
});

describe("boundary imports", () => {
  it("preserves holes and all polygon features", () => {
    const hole = [
      [0.2, 0.2],
      [0.8, 0.2],
      [0.8, 0.8],
      [0.2, 0.2],
    ];
    const result = importBoundary(
      JSON.stringify({
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            geometry: { ...polygon, coordinates: [...polygon.coordinates, hole] },
          },
          { type: "Feature", geometry: polygon },
        ],
      }),
    );
    expect(result.type).toBe("MultiPolygon");
    expect(result.coordinates).toEqual([[...polygon.coordinates, hole], polygon.coordinates]);
  });
  it("refuses malformed coordinates rather than silently dropping them", () => {
    expect(() =>
      importBoundary(
        JSON.stringify({
          ...polygon,
          coordinates: [
            [
              [200, 0],
              [1, 0],
              [1, 1],
              [200, 0],
            ],
          ],
        }),
      ),
    ).toThrow("WGS 84");
    expect(() =>
      importBoundary(JSON.stringify({ type: "FeatureCollection", features: [] })),
    ).toThrow("no land areas");
    expect(() =>
      importBoundary(
        JSON.stringify({
          ...polygon,
          coordinates: [
            [
              [0, 0],
              [1, 0],
              [1, 1],
              [0, 1],
            ],
          ],
        }),
      ),
    ).toThrow("closed");
  });
});

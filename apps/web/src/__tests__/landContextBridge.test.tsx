import { cleanup, render } from "@testing-library/react";
import { EntityCollection, JulianDate, type PolygonHierarchy, type Color } from "cesium";
import { afterEach, expect, it, vi } from "vitest";
import { LandContextBridge } from "@/cesium/LandContextBridge";
import { useLandContext } from "@/state/landContext";
const fixture = vi.hoisted(() => ({
  entities: null as EntityCollection | null,
  render: vi.fn(),
  listeners: new Map<string, (value: unknown) => void>(),
}));
vi.mock("@/cesium/SceneContext", () => ({
  useScene: () => ({
    isDestroyed: false,
    viewer: { entities: fixture.entities },
    scene: { requestRender: fixture.render },
    events: {
      on: (name: string, listener: (value: unknown) => void) => {
        fixture.listeners.set(name, listener);
        return () => fixture.listeners.delete(name);
      },
    },
  }),
}));
afterEach(() => {
  cleanup();
  useLandContext.getState().clear();
  fixture.listeners.clear();
  vi.clearAllMocks();
});
it("highlights points in place, ignores unrelated changes and handles map picks", () => {
  const entities = new EntityCollection();
  fixture.entities = entities;
  const add = vi.spyOn(entities, "add"),
    remove = vi.spyOn(entities, "remove");
  useLandContext.getState().setLayer({
    id: "map",
    researchArtifactId: "map",
    title: "Samples",
    features: [
      { id: "0", label: "Zero", value: 0, geometry: { type: "Point", coordinates: [-77, 38.9] } },
      { id: "1", label: "One", geometry: { type: "Point", coordinates: [-77.001, 38.9] } },
    ],
  });
  const mounted = render(<LandContextBridge />);
  expect(add).toHaveBeenCalledTimes(2);
  const original = entities.getById("land-context:map/0")!;
  expect(original.properties?.getValue(JulianDate.now())).toMatchObject({
    landResearchArtifactId: "map",
    landResearchFeatureId: "0",
  });
  useLandContext.getState().selectMapFeature("map", "0");
  expect(entities.getById(original.id)).toBe(original);
  expect(original.point?.pixelSize?.getValue()).toBe(16);
  expect(add).toHaveBeenCalledTimes(2);
  expect(remove).not.toHaveBeenCalled();
  const renders = fixture.render.mock.calls.length;
  useLandContext.getState().setResearchQuestion("Unrelated typing");
  expect(fixture.render.mock.calls.length).toBe(renders);
  fixture.listeners.get("land-research-feature-select")?.({ layerId: "map", featureId: "1" });
  expect(useLandContext.getState().selectedMapFeature).toEqual({
    layerId: "map",
    featureId: "1",
    origin: "map",
  });
  expect(original.point?.pixelSize?.getValue()).toBe(12);
  expect(add).toHaveBeenCalledTimes(2);
  mounted.unmount();
  expect(entities.values).toHaveLength(0);
});
it("keeps polygon holes and multipart geometry intact when highlighting", () => {
  const entities = new EntityCollection();
  fixture.entities = entities;
  const outer = [
    [-77, 38],
    [-76, 38],
    [-76, 39],
    [-77, 39],
    [-77, 38],
  ];
  const hole = [
    [-76.8, 38.2],
    [-76.8, 38.4],
    [-76.6, 38.4],
    [-76.6, 38.2],
    [-76.8, 38.2],
  ];
  useLandContext.getState().setLayer({
    id: "map",
    researchArtifactId: "map",
    title: "Synthetic polygons",
    features: [
      {
        id: "0",
        label: "Multipart",
        geometry: {
          type: "MultiPolygon",
          coordinates: [[outer, hole], [outer.map(([x, y]) => [x! + 2, y!])]],
        },
      },
    ],
  });
  render(<LandContextBridge />);
  const originals = [...entities.values];
  expect(originals).toHaveLength(2);
  const hierarchy = originals[0]!.polygon!.hierarchy!.getValue(
    JulianDate.now(),
  ) as PolygonHierarchy;
  expect(hierarchy.holes).toHaveLength(1);
  useLandContext.getState().selectMapFeature("map", "0");
  for (const entity of originals) {
    expect(entities.getById(entity.id)).toBe(entity);
    expect(entity.polyline?.width?.getValue()).toBe(4);
    const material = entity.polygon?.material?.getValue(JulianDate.now()) as { color: Color };
    expect(material.color.alpha).toBeCloseTo(0.28);
  }
  expect(originals[0]!.polygon!.hierarchy!.getValue(JulianDate.now())).toBe(hierarchy);
});

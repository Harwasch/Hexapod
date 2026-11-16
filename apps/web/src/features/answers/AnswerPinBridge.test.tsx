import { act, cleanup, render } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { EntityCollection } from "cesium";
import { afterEach, expect, it, vi } from "vitest";
import { AnswerPinBridge } from "./AnswerPinBridge";
import { useSites } from "@/state/sites";
import { useAsCaptured } from "@/state/asCaptured";

const fixture = vi.hoisted(() => ({
  entities: null as EntityCollection | null,
  canvas: null as HTMLCanvasElement | null,
  render: vi.fn(),
}));
vi.mock("@/cesium/SceneContext", () => ({
  useScene: () => ({
    isDestroyed: false,
    viewer: { entities: fixture.entities, dataSourceDisplay: { ready: true } },
    scene: { canvas: fixture.canvas, requestRender: fixture.render, pick: () => undefined },
  }),
}));
vi.mock("./propertyPins", () => ({
  loadAnswerPins: vi.fn().mockResolvedValue([]),
}));
afterEach(() => {
  cleanup();
  useSites.getState().setActiveSite(null);
  useAsCaptured.setState({ on: false });
});
it("removes the old property's pins on site change, hides inferred answers and disposes its layer", () => {
  const entities = new EntityCollection();
  entities.add({ id: "scan" });
  fixture.entities = entities;
  fixture.canvas = document.createElement("canvas");
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const pin = {
    id: "e",
    siteId: "a",
    scanId: "c",
    label: "Oak",
    observedAt: "2026-01-01T00:00:00Z",
    uncertaintyM: 1,
    geo: { lat: 0, lon: 0, alt: 0 },
  };
  client.setQueryData(["property-answer-pins", "a"], [pin]);
  client.setQueryData(["property-answer-pins", "b"], [{ ...pin, siteId: "b" }]);
  useSites.getState().setActiveSite("a");
  const mounted = render(
    <QueryClientProvider client={client}>
      <AnswerPinBridge />
    </QueryClientProvider>,
  );
  expect(entities.getById("property-answer:a:e")).toBeDefined();
  act(() => {
    useSites.getState().setActiveSite("b");
  });
  expect(entities.getById("property-answer:a:e")).toBeUndefined();
  expect(entities.getById("property-answer:b:e")).toBeDefined();
  act(() => {
    useAsCaptured.setState({ on: true });
  });
  expect(entities.values.map((entity) => entity.id)).toEqual(["scan"]);
  mounted.unmount();
  client.clear();
  expect(entities.values.map((entity) => entity.id)).toEqual(["scan"]);
});

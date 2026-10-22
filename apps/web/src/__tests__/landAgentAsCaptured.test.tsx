/**
 * As captured (ADR 0019) draws none of the Land agent's work on the map: its findings' pins and
 * outlines, where it is looking, a research map it made, and the outline and leader a card in
 * the agent's tone points with. What the land's own records put there (a soil sample, the
 * Land's lenses, plans) stays. Turning it off draws the same again; nothing is cleared.
 */
import { act, cleanup, render, screen } from "@testing-library/react";
import { EntityCollection } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LandContextBridge } from "@/cesium/LandContextBridge";
import { LandFocusBridge } from "@/cesium/LandFocusBridge";
import { LandOverlays } from "@/features/land/surface/LandOverlays";
import { useAsCaptured } from "@/state/asCaptured";
import { useLandContext, type LandContextLayer } from "@/state/landContext";
import { registerLeader, useLandMap } from "@/state/landMap";

const fixture = vi.hoisted(() => ({
  entities: null as EntityCollection | null,
  frame: null as (() => void) | null,
}));

vi.mock("@/cesium/SceneContext", () => ({
  useScene: () => ({
    isDestroyed: false,
    viewer: { entities: fixture.entities, dataSourceDisplay: { ready: true } },
    scene: {
      requestRender: () => undefined,
      canvas: { getBoundingClientRect: () => ({ left: 0, top: 0, right: 800, bottom: 600 }) },
    },
    events: { on: () => () => undefined },
  }),
}));

// Every place lands at (600, 300) on screen, right of a card at (0..200, 0..400).
vi.mock("@/cesium/LandAnchorProjector", () => ({
  LandAnchorProjector: class {
    onFrame(listener: () => void) {
      fixture.frame = listener;
      return () => undefined;
    }
    requestFrame = () => undefined;
    destroy = () => undefined;
    project() {
      return { x: 600, y: 300, visible: true };
    }
  },
}));

const square = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-120.05, 34.69],
      [-120.03, 34.69],
      [-120.03, 34.7],
      [-120.05, 34.69],
    ],
  ],
};
const anchor: [number, number] = [-120.04, 34.695];

const layer = (id: string, extra: Partial<LandContextLayer> = {}): LandContextLayer => ({
  id,
  title: id,
  features: [{ id: "a", label: id, geometry: square }],
  ...extra,
});

const drawnLayers = () =>
  new Set(
    (fixture.entities?.values ?? [])
      .map((entity) => /^land-context:([^/]+)\//.exec(entity.id)?.[1])
      .filter(Boolean),
  );

function card(): HTMLElement {
  const element = document.createElement("div");
  element.getBoundingClientRect = () =>
    ({ left: 0, top: 0, right: 200, bottom: 400, width: 200, height: 400 }) as DOMRect;
  document.body.appendChild(element);
  return element;
}

beforeEach(() => {
  fixture.entities = new EntityCollection();
  vi.stubGlobal("requestAnimationFrame", () => 0);
  vi.stubGlobal("cancelAnimationFrame", () => undefined);
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe = () => undefined;
      disconnect = () => undefined;
    },
  );
});

afterEach(() => {
  cleanup();
  useAsCaptured.getState().set(false);
  useLandContext.getState().clear();
  useLandMap.setState({ pins: {}, focus: null });
  document.body.innerHTML = "";
  vi.unstubAllGlobals();
});

describe("the Land's context layers", () => {
  it("draw none the agent made while As captured is on, and the same layers after", () => {
    useLandContext.getState().setLayer(layer("agent-findings", { agent: true }));
    useLandContext.getState().setLayer(layer("agent-looking", { agent: true }));
    useLandContext.getState().setLayer(layer("research-1", { researchArtifactId: "research-1" }));
    useLandContext.getState().setLayer(layer("soil"));
    render(<LandContextBridge />);
    expect(drawnLayers()).toEqual(
      new Set(["agent-findings", "agent-looking", "research-1", "soil"]),
    );
    act(() => useAsCaptured.getState().set(true));
    expect(drawnLayers()).toEqual(new Set(["soil"]));
    // A research map shown meanwhile waits too.
    act(() =>
      useLandContext.getState().setLayer(layer("research-2", { researchArtifactId: "research-2" })),
    );
    expect(drawnLayers()).toEqual(new Set(["soil"]));
    // Nothing was cleared: the store still holds every layer.
    expect(Object.keys(useLandContext.getState().layers)).toHaveLength(5);
    act(() => useAsCaptured.getState().set(false));
    expect(drawnLayers()).toEqual(
      new Set(["agent-findings", "agent-looking", "research-1", "research-2", "soil"]),
    );
  });
});

describe("the Land's pins and leaders", () => {
  it("draw none in the agent's tone while As captured is on", () => {
    useLandMap
      .getState()
      .setPins("agent", [{ id: "agent-f1", position: anchor, label: "Erosion", tone: "agent" }]);
    useLandMap
      .getState()
      .setPins("soil", [{ id: "soil-sample", position: anchor, label: "Sampled", tone: "accent" }]);
    render(<LandOverlays />);
    const leaders = () => document.querySelectorAll(".land-leader").length;
    const from = card();
    const offAgent = registerLeader({ id: "f1", element: from, anchor, tone: "agent" });
    act(() => fixture.frame?.());
    expect(screen.queryByTestId("land-pin-agent-f1")).not.toBeNull();
    expect(leaders()).toBe(1);
    act(() => useAsCaptured.getState().set(true));
    act(() => fixture.frame?.());
    expect(screen.queryByTestId("land-pin-agent-f1")).toBeNull();
    expect(screen.queryByTestId("land-pin-soil-sample")).not.toBeNull();
    expect(leaders()).toBe(0);
    // A card of the land's own facts still points at its place.
    const offOwn = registerLeader({ id: "own", element: from, anchor, tone: "accent" });
    act(() => fixture.frame?.());
    expect(leaders()).toBe(1);
    act(() => useAsCaptured.getState().set(false));
    act(() => fixture.frame?.());
    expect(screen.queryByTestId("land-pin-agent-f1")).not.toBeNull();
    expect(leaders()).toBe(2);
    offAgent();
    offOwn();
  });

  it("outline no place a card in the agent's tone points at while As captured is on", () => {
    render(<LandFocusBridge />);
    const drawn = () => fixture.entities?.values.length ?? 0;
    const focus = (key: string, tone: "agent" | "accent") =>
      act(() => useLandMap.getState().setFocus({ key, geometry: square, anchor, tone }));
    focus("f1", "agent");
    expect(drawn()).toBeGreaterThan(0);
    act(() => useAsCaptured.getState().set(true));
    expect(drawn()).toBe(0);
    focus("f2", "agent");
    expect(drawn()).toBe(0);
    focus("own", "accent");
    expect(drawn()).toBeGreaterThan(0);
    focus("f2", "agent");
    act(() => useAsCaptured.getState().set(false));
    expect(drawn()).toBeGreaterThan(0);
  });
});

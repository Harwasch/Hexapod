/**
 * The Land panel on a phone (and a window as narrow): a touch screen is told to tap, not click;
 * a boundary going to review is framed above the sheet that would otherwise cover it; and
 * reopening a saved land shows its overview rather than the list it was opened from.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Cartographic, Math as CesiumMath, type Cartesian3 } from "cesium";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { LandArea } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { LandPanel } from "@/features/land/LandPanel";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useUi } from "@/state/ui";

/** A phone held upright: the map is the whole screen, the Land sheet its lower part. */
const SCREEN = { left: 0, top: 0, right: 390, bottom: 844, width: 390, height: 844 };
const SHEET = { left: 0, top: 300, right: 390, bottom: 844, width: 390, height: 544 };

/** What the camera shows now: a top-down view of this rectangle, filling the screen. */
const shown = vi.hoisted(() => ({ west: -120.1, south: 34.6, east: -120.0, north: 34.72 }));

const scene = vi.hoisted(() => ({
  camera: { flyTo: vi.fn(), groundHeightAt: () => Promise.resolve(0) },
  areas: { cancelPick: vi.fn(), edit: vi.fn() },
  isDestroyed: false,
  viewer: {
    canvas: null as unknown as HTMLCanvasElement,
    camera: { frustum: { fovy: 1 } },
  },
  scene: {
    sampleHeightSupported: false,
    sampleHeight: () => undefined,
    globe: { getHeight: () => 250 },
    cartesianToCanvasCoordinates: (position: Cartesian3) => {
      const at = Cartographic.fromCartesian(position);
      const lon = CesiumMath.toDegrees(at.longitude);
      const lat = CesiumMath.toDegrees(at.latitude);
      return {
        x: ((lon - shown.west) / (shown.east - shown.west)) * 390,
        y: ((shown.north - lat) / (shown.north - shown.south)) * 844,
      };
    },
  },
}));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));

/** A square in the lower part of the view, where the sheet sits. */
const LOW = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-120.06, 34.62],
      [-120.04, 34.62],
      [-120.04, 34.64],
      [-120.06, 34.64],
      [-120.06, 34.62],
    ],
  ],
};

const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Oak Bench Ranch",
  description: "",
  boundary: LOW,
  source: { method: "drawn", label: "Drawn on the map", meaning: "study-area" },
  areaM2: 4_000_000,
  perimeterM: 8_000,
  revision: 1,
  createdAt: "2026-10-17T00:00:00Z",
  updatedAt: "2026-10-17T00:00:00Z",
};

/**
 * What a camera flown by `flyTo(lon, lat, height, { pitch: -90 })` shows of a flat world under
 * the mocked frustum: the ground is 250 m up (`globe.getHeight`), so the view's half-height is
 * (height - 250) tan(fovy / 2).
 */
function seen(call: unknown[]) {
  const [lon, lat, height, options] = call as [number, number, number, { pitch: number }];
  expect(options.pitch).toBe(-90);
  const halfH = (height - 250) * Math.tan(0.5);
  const halfW = halfH * (SCREEN.width / SCREEN.height);
  const dLat = halfH / 111_320;
  const dLon = halfW / (111_320 * Math.cos((lat * Math.PI) / 180));
  return { west: lon - dLon, south: lat - dLat, east: lon + dLon, north: lat + dLat };
}

function touchScreen(coarse: boolean) {
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query: string) =>
      ({
        matches: coarse && query === "(pointer: coarse)",
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false,
      }) as MediaQueryList,
  );
}

function renderPanel() {
  useUi.getState().setPanel("land");
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <GlassTooltipProvider>
        <LandPanel />
      </GlassTooltipProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  const canvas = document.createElement("canvas");
  canvas.getBoundingClientRect = () => SCREEN as DOMRect;
  scene.viewer.canvas = canvas;
  scene.camera.flyTo.mockClear();
  // Only the Land panel has a layout here: the bottom sheet. Everything else measures nothing.
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (
    this: HTMLElement,
  ) {
    return (this.dataset.testid === "land-panel" ? SHEET : new DOMRect()) as DOMRect;
  });
});

afterEach(() => {
  cleanup();
  useLand.getState().clear();
  useLandContext.getState().clear();
  useUi.getState().setPanel(null);
  vi.restoreAllMocks();
});

it("tells a touch screen to tap, and a mouse to click", () => {
  vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
  touchScreen(true);
  renderPanel();
  expect(screen.getByText("Tap around your land on the map")).toBeInTheDocument();
  act(() => useLand.getState().begin("draw"));
  expect(
    screen.getByText("0 points placed. Tap the map to add the first point."),
  ).toBeInTheDocument();
  cleanup();
  touchScreen(false);
  renderPanel();
  expect(
    screen.getByText("0 points placed. Click the map to add the first point."),
  ).toBeInTheDocument();
});

it("frames a boundary going to review above the sheet that covers it", async () => {
  vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
  touchScreen(true);
  renderPanel();
  act(() =>
    useLand.getState().propose({
      name: "Untitled land",
      description: "",
      boundary: LOW,
      source: { method: "drawn", label: "Drawn on the map", meaning: "study-area" },
    }),
  );
  await waitFor(() => expect(scene.camera.flyTo).toHaveBeenCalledTimes(1));
  const { west, south, east, north } = seen(scene.camera.flyTo.mock.calls[0]!);
  // Seen through that rectangle, the whole square sits above the sheet's top edge.
  for (const [lon, lat] of LOW.coordinates[0]!) {
    const y = ((north - lat!) / (north - south)) * SCREEN.height;
    const x = ((lon! - west) / (east - west)) * SCREEN.width;
    expect(y).toBeLessThan(SHEET.top);
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x).toBeLessThanOrEqual(SCREEN.width);
  }
  // Renaming the draft does not move the camera again.
  act(() => useLand.getState().updateDraft({ name: "West field" }));
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(scene.camera.flyTo).toHaveBeenCalledTimes(1);
});

it("leaves the camera alone when the reviewed boundary is already in sight", async () => {
  vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
  renderPanel();
  const high = {
    ...LOW,
    coordinates: [LOW.coordinates[0]!.map(([lon, lat]) => [lon!, lat! + 0.07])],
  };
  act(() =>
    useLand.getState().propose({
      name: "Untitled land",
      description: "",
      boundary: high,
      source: { method: "drawn", label: "Drawn on the map", meaning: "study-area" },
    }),
  );
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(scene.camera.flyTo).not.toHaveBeenCalled();
});

it("opens a saved land at its overview, framed above the sheet", async () => {
  vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    path === "/api/v1/land"
      ? Promise.resolve({ data: [land], response: new Response(null, { status: 200 }) })
      : Promise.reject(new TypeError("offline"))) as never);
  const scrolled = vi.spyOn(Element.prototype, "scrollIntoView");
  renderPanel();
  fireEvent.click(await screen.findByRole("button", { name: /Oak Bench Ranch/ }));
  expect(useLand.getState().active?.id).toBe(land.id);
  await waitFor(() =>
    expect(
      scrolled.mock.contexts.some((el) => (el as Element).matches(".land-place-heading")),
    ).toBe(true),
  );
  const heading = scrolled.mock.contexts.find((el) =>
    (el as Element).matches(".land-place-heading"),
  );
  expect(heading).toHaveTextContent("Land overview");
  expect(scene.camera.flyTo).toHaveBeenCalled();
  const { south, north } = seen(scene.camera.flyTo.mock.calls.at(-1)!);
  // The land's centre is above the screen's centre: the sheet covers the lower part.
  expect((south + north) / 2).toBeLessThan(34.63);
});

/**
 * The Land surface on a phone (and a window as narrow): a touch screen is told to tap, not
 * click; a boundary going to review is framed above the sheet that would otherwise cover it;
 * and opening a saved land shows its chip over the Overview lens, framed above the sheet, rather
 * than the list it was opened from or whichever lens was open before.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Cartographic, Math as CesiumMath, type Cartesian3 } from "cesium";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { LandArea } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { frameLand } from "@/features/land/landFraming";
import { LandStrip } from "@/features/land/surface/LandStrip";
import { LandSurface } from "@/features/land/surface/LandSurface";
import { useLens } from "@/features/land/surface/lenses";
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
    camera: { frustum: { fovy: 1 }, positionCartographic: { height: 2000 } },
  },
  scene: {
    sampleHeightSupported: false,
    sampleHeight: (): number | undefined => undefined,
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

// The lens cards are each tested on their own; here a card only says which lens it is.
vi.mock("@/features/land/surface/lensCards", () => {
  const card = (lens: string) =>
    function LensStub({ land }: { land: LandArea }) {
      return <section data-testid={`land-lens-card-${lens}`} aria-label={`${lens} ${land.name}`} />;
    };
  const lenses = ["overview", "water", "soil", "ecology", "solar", "records", "assets", "plans"];
  return { LENS_CARDS: Object.fromEntries(lenses.map((lens) => [lens, card(lens)])) };
});

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

/** Every corner of LOW, seen through that flight's view, is on screen above the sheet. */
function expectAboveSheet(call: unknown[]) {
  const { west, south, east, north } = seen(call);
  for (const [lon, lat] of LOW.coordinates[0]!) {
    const y = ((north - lat!) / (north - south)) * SCREEN.height;
    const x = ((lon! - west) / (east - west)) * SCREEN.width;
    expect(y).toBeGreaterThanOrEqual(0);
    expect(y).toBeLessThan(SHEET.top);
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x).toBeLessThanOrEqual(SCREEN.width);
  }
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

function offline() {
  vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
}

/** The Land surface in the sheet, and its strip (the draw bar) below it. */
function renderSurface() {
  useUi.getState().setPanel("land");
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <GlassTooltipProvider>
        <LandSurface />
        <LandStrip />
      </GlassTooltipProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  const canvas = document.createElement("canvas");
  canvas.getBoundingClientRect = () => SCREEN as DOMRect;
  scene.viewer.canvas = canvas;
  scene.camera.flyTo.mockClear();
  // Only the Land surface has a layout here: the bottom sheet. Everything else measures nothing.
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
  useLens.getState().setLens("overview");
  useUi.getState().setPanel(null);
  vi.restoreAllMocks();
});

it("tells a touch screen to tap, and a mouse to click", () => {
  offline();
  touchScreen(true);
  renderSurface();
  act(() => useLand.getState().begin("draw"));
  expect(screen.getByTestId("land-drawbar")).toHaveTextContent("Tap the corners of your land");
  act(() => useLand.getState().begin("corridor"));
  expect(screen.getByTestId("land-drawbar")).toHaveTextContent("Tap along the centre line");
  act(() => useLand.getState().begin("pick"));
  expect(screen.getByTestId("land-drawbar")).toHaveTextContent(
    "Tap a field, woodland, park or lake",
  );
  act(() => useLand.getState().begin("candidates"));
  expect(screen.getByText("Tap near your land on the map.")).toBeInTheDocument();
  cleanup();
  useLand.getState().clear();
  touchScreen(false);
  renderSurface();
  act(() => useLand.getState().begin("draw"));
  expect(screen.getByTestId("land-drawbar")).toHaveTextContent("Click the corners of your land");
  act(() => useLand.getState().begin("candidates"));
  expect(screen.getByText("Click near your land on the map.")).toBeInTheDocument();
});

it("frames a boundary going to review above the sheet that covers it", async () => {
  offline();
  touchScreen(true);
  renderSurface();
  act(() =>
    useLand.getState().propose({
      name: "Untitled land",
      description: "",
      boundary: LOW,
      source: { method: "drawn", label: "Drawn on the map", meaning: "study-area" },
    }),
  );
  expect(screen.getByTestId("land-review")).toBeInTheDocument();
  await waitFor(() => expect(scene.camera.flyTo).toHaveBeenCalledTimes(1));
  // Seen through that view, the whole square sits above the sheet's top edge.
  expectAboveSheet(scene.camera.flyTo.mock.calls[0]!);
  // Naming the draft does not move the camera again.
  fireEvent.change(screen.getByLabelText("Name this land"), { target: { value: "West field" } });
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(scene.camera.flyTo).toHaveBeenCalledTimes(1);
  // Its own "Show boundary on map" frames it the same way.
  fireEvent.click(screen.getByRole("button", { name: "Show boundary on map" }));
  expect(scene.camera.flyTo).toHaveBeenCalledTimes(2);
  expectAboveSheet(scene.camera.flyTo.mock.calls[1]!);
});

it("leaves the camera alone when the reviewed boundary is already in sight", async () => {
  offline();
  renderSurface();
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

it("opens a saved land on its chip and Overview lens, framed above the sheet", async () => {
  vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    path === "/api/v1/land"
      ? Promise.resolve({ data: [land], response: new Response(null, { status: 200 }) })
      : Promise.reject(new TypeError("offline"))) as never);
  touchScreen(true);
  // The last land was left on its Plans lens, and the start card scrolled down to its list.
  useLens.getState().setLens("plans");
  renderSurface();
  const row = await screen.findByRole("button", { name: /Oak Bench Ranch/ });
  const sheet = screen.getByTestId("land-panel");
  sheet.scrollTop = 160;
  fireEvent.click(row);
  expect(useLand.getState().active?.id).toBe(land.id);
  expect(screen.queryByTestId("land-start")).not.toBeInTheDocument();
  expect(screen.getByTestId("land-chip")).toHaveTextContent("Oak Bench Ranch");
  expect(screen.getByTestId("land-lens-card-overview")).toBeInTheDocument();
  expect(screen.queryByTestId("land-lens-card-plans")).not.toBeInTheDocument();
  expect(sheet.scrollTop).toBe(0);
  // Framed once the chip and the lens have laid out, above the sheet they make.
  expect(scene.camera.flyTo).not.toHaveBeenCalled();
  await waitFor(() => expect(scene.camera.flyTo).toHaveBeenCalledTimes(1));
  expectAboveSheet(scene.camera.flyTo.mock.calls[0]!);
  // The chip's "Show the land on the map" does the same, straight away.
  fireEvent.click(screen.getByRole("button", { name: "Show the land on the map" }));
  expect(scene.camera.flyTo).toHaveBeenCalledTimes(2);
  expectAboveSheet(scene.camera.flyTo.mock.calls[1]!);
});

it("says where saved land will be, and where plans on it go, before any is saved", async () => {
  vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    path === "/api/v1/land"
      ? Promise.resolve({ data: [], response: new Response(null, { status: 200 }) })
      : Promise.reject(new TypeError("offline"))) as never);
  renderSurface();
  expect(await screen.findByTestId("land-saved-empty")).toHaveTextContent(
    "Your saved areas stay here; plans you make on them appear in Plan.",
  );
});

it("measures the flight from the ground only where the ground reads true", async () => {
  // Close to the land, a sample of the drawn world 40 km underground is a coarse tile, not the
  // ground: the terrain (250 m here) is used instead.
  scene.scene.sampleHeightSupported = true;
  scene.scene.sampleHeight = () => -42_000;
  frameLand(scene as never, LOW);
  const [, , closeHeight] = scene.camera.flyTo.mock.calls.at(-1) as [number, number, number];
  expect(closeHeight).toBeGreaterThan(250);
  // From orbit nothing under the land is detailed: the terrain is asked, and the flight waits.
  scene.viewer.camera.positionCartographic.height = 18_000_000;
  scene.camera.flyTo.mockClear();
  frameLand(scene as never, LOW);
  expect(scene.camera.flyTo).not.toHaveBeenCalled();
  await waitFor(() => expect(scene.camera.flyTo).toHaveBeenCalledTimes(1));
  const [, , farHeight] = scene.camera.flyTo.mock.calls[0] as [number, number, number];
  expect(farHeight).toBeCloseTo(closeHeight - 250, 0);
  scene.viewer.camera.positionCartographic.height = 2000;
  scene.scene.sampleHeightSupported = false;
});

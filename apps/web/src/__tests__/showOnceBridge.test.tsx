/**
 * Show once, everywhere on the map: a tap on the Land compares that place and drapes the
 * answer; the slider moves only the drape's uniform; a protect tap swaps the texture; hiding
 * the card lifts the drape and the taps; the fog taking the ground or Escape ends it.
 */

import { act, cleanup, render } from "@testing-library/react";
import { EntityCollection } from "cesium";
import type * as Cesium from "cesium";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { LandArea, SignatureSimilarity } from "@twin/contracts";

import { ShowOnceBridge } from "@/cesium/ShowOnceBridge";
import { rampThreshold } from "@/features/showOnce/grid";
import { useGridDrape } from "@/state/gridDrape";
import { useLand } from "@/state/land";
import { useShowOnce } from "@/state/showOnce";

const fixture = vi.hoisted(() => ({
  primitives: [] as { options: Record<string, unknown> }[],
  removed: [] as unknown[],
  materials: [] as { uniforms: Record<string, unknown> }[],
  listeners: new Map<string, (value: unknown) => void>(),
  released: 0,
  claims: null as ((longitude: number, latitude: number) => boolean) | null,
  render: vi.fn(),
  entities: null as EntityCollection | null,
  similar: vi.fn((..._args: unknown[]): Promise<unknown> => Promise.resolve(null)),
  canvas: { style: { cursor: "" } },
}));

vi.mock("cesium", async (original) => {
  const real = await original<typeof Cesium>();
  return {
    ...real,
    GroundPrimitive: class {
      constructor(public options: Record<string, unknown>) {}
    },
    Material: class {
      static DefaultImageId = "czm_defaultImage";
      uniforms: Record<string, unknown>;
      constructor(options: { fabric: { uniforms: Record<string, unknown> } }) {
        this.uniforms = { ...options.fabric.uniforms };
        fixture.materials.push(this);
      }
    },
    MaterialAppearance: Object.assign(
      class {
        constructor(public options: unknown) {}
      },
      { MaterialSupport: { TEXTURED: { vertexFormat: "textured" } } },
    ),
    RectangleGeometry: class {
      constructor(public options: unknown) {}
    },
    GeometryInstance: class {
      constructor(public options: unknown) {}
    },
  };
});

vi.mock("@/cesium/SceneContext", () => {
  let host: unknown = null;
  return {
    useScene: () =>
      (host ??= {
        viewer: {
          get entities() {
            return fixture.entities;
          },
          canvas: fixture.canvas,
          scene: {
            requestRender: fixture.render,
            isDestroyed: () => false,
            groundPrimitives: {
              add: (primitive: { options: Record<string, unknown> }) => {
                fixture.primitives.push(primitive);
                return primitive;
              },
              remove: (primitive: unknown) => fixture.removed.push(primitive),
            },
          },
        },
        selection: {
          wantGroundTaps: (claims: (longitude: number, latitude: number) => boolean) => {
            fixture.claims = claims;
            return () => {
              fixture.released++;
            };
          },
        },
        events: {
          on: (name: string, listener: (value: unknown) => void) => {
            fixture.listeners.set(name, listener);
            return () => fixture.listeners.delete(name);
          },
        },
      }),
  };
});

vi.mock("@/features/showOnce/api", () => ({
  fetchSimilar: (...args: unknown[]): unknown => fixture.similar(...args),
}));

const LAND: LandArea = {
  id: "land-a",
  name: "Oak Bench",
  revision: 1,
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-120.05, 34.69],
        [-120.03, 34.69],
        [-120.03, 34.7],
        [-120.05, 34.7],
        [-120.05, 34.69],
      ],
    ],
  },
} as LandArea;

/** 2 x 2 over the land: north-west 250, north-east 210, south-west none, south-east 120. */
function similarity(): SignatureSimilarity {
  return {
    rasterId: "raster-1",
    landId: LAND.id,
    boundaryRevision: 1,
    stale: false,
    seed: {
      longitude: -120.045,
      latitude: 34.698,
      cellLongitude: -120.045,
      cellLatitude: 34.6975,
      windows: [],
    },
    grid: {
      west: -120.05,
      south: 34.69,
      east: -120.03,
      north: 34.7,
      columns: 2,
      rows: 2,
      cellM: 10,
      encoding: "u8",
      scale: 0.004,
      offset: 0,
      nodata: 255,
      data: btoa(String.fromCharCode(250, 210, 255, 120)),
    },
    cellAreaM2: 100,
    nativeResolutionM: 10,
    comparedCells: 3,
    boundaryCells: 4,
    scenes: [],
    method: "cosine",
    interpretation: "Not a species ID.",
    warnings: [],
    attribution: "Copernicus",
  };
}

const tap = (longitude: number, latitude: number) =>
  act(async () => {
    fixture.listeners.get("ground-tap")?.({ longitude, latitude, x: 1, y: 1 });
    await Promise.resolve();
    await Promise.resolve();
  });

beforeEach(() => {
  fixture.entities = new EntityCollection();
  fixture.similar.mockResolvedValue(similarity());
  useLand.getState().select(LAND);
  useShowOnce.getState().stop();
  useShowOnce.getState().setShown(true);
  useShowOnce.getState().start({ landId: LAND.id, revision: 1, rasterId: "raster-1" });
});

afterEach(() => {
  cleanup();
  useShowOnce.getState().stop();
  useShowOnce.getState().setShown(false);
  useLand.getState().clear();
  useGridDrape.setState({ owner: null });
  fixture.primitives.length = 0;
  fixture.removed.length = 0;
  fixture.materials.length = 0;
  fixture.listeners.clear();
  fixture.released = 0;
  fixture.claims = null;
  fixture.canvas.style.cursor = "";
  vi.clearAllMocks();
});

it("takes taps on the land, compares the place and drapes the answer", async () => {
  render(<ShowOnceBridge />);
  expect(fixture.canvas.style.cursor).toBe("crosshair");
  // The Land's ground is claimed before any answer; past its boundary nothing is.
  expect(fixture.claims!(-120.04, 34.695)).toBe(true);
  expect(fixture.claims!(-120.2, 34.695)).toBe(false);
  await tap(-120.045, 34.698);
  expect(fixture.similar).toHaveBeenCalledWith("raster-1", -120.045, 34.698);
  expect(useShowOnce.getState().result?.grid.columns).toBe(2);
  expect(fixture.primitives).toHaveLength(1);
  expect(useGridDrape.getState().owner).toBe("signature");
  const material = fixture.materials[0]!;
  expect(material.uniforms.threshold).toBe(rampThreshold(0.8));
  expect(fixture.entities?.getById("grid-outline:signature")).toBeDefined();
  // A cell with no signature is not the answer's ground; the boundary still is.
  expect(fixture.claims!(-120.045, 34.692)).toBe(true);
});

it("moves only the uniform with the slider, and swaps the texture to protect", async () => {
  render(<ShowOnceBridge />);
  await tap(-120.045, 34.698);
  const material = fixture.materials[0]!;
  const texture = material.uniforms.values;
  act(() => useShowOnce.getState().setThreshold(0.9));
  expect(material.uniforms.threshold).toBe(rampThreshold(0.9));
  expect(material.uniforms.values).toBe(texture);
  expect(fixture.primitives).toHaveLength(1);
  // Protect the north-east cell (lit at 80%).
  act(() => useShowOnce.getState().setThreshold(0.8));
  act(() => useShowOnce.getState().setMode("protect"));
  await tap(-120.035, 34.698);
  expect([...(useShowOnce.getState().guarded ?? [])]).toEqual([1, 1, 0, 0]);
  expect(material.uniforms.values).not.toBe(texture);
  expect(fixture.primitives).toHaveLength(1);
  expect(fixture.similar).toHaveBeenCalledTimes(1);
  // Unlit ground: nothing to protect, and the card says so.
  await tap(-120.035, 34.692);
  expect(useShowOnce.getState().message).toMatch(/not lit/);
});

it("says when a tap is off the land", async () => {
  render(<ShowOnceBridge />);
  await tap(-120.2, 34.695);
  expect(fixture.similar).not.toHaveBeenCalled();
  expect(useShowOnce.getState().message).toBe(
    "That spot is outside Oak Bench. Choose a place inside its boundary.",
  );
});

it("lifts the drape and the taps while the card is hidden, and keeps the answer", async () => {
  render(<ShowOnceBridge />);
  await tap(-120.045, 34.698);
  act(() => useShowOnce.getState().setShown(false));
  expect(fixture.released).toBe(1);
  expect(fixture.removed).toHaveLength(1);
  expect(fixture.canvas.style.cursor).toBe("");
  expect(useShowOnce.getState().result).not.toBeNull();
  act(() => useShowOnce.getState().setShown(true));
  expect(fixture.primitives).toHaveLength(2);
});

it("ends when the fog takes the ground, on Escape, and with another land", async () => {
  render(<ShowOnceBridge />);
  await tap(-120.045, 34.698);
  act(() => useGridDrape.getState().claim("fog"));
  expect(useShowOnce.getState().session).toBeNull();

  act(() => useShowOnce.getState().start({ landId: LAND.id, revision: 1, rasterId: "raster-1" }));
  const escape = new KeyboardEvent("keydown", { key: "Escape", cancelable: true });
  act(() => {
    window.dispatchEvent(escape);
  });
  // Consumed: the shell does not also close the panel.
  expect(escape.defaultPrevented).toBe(true);
  expect(useShowOnce.getState().session).toBeNull();

  act(() => useShowOnce.getState().start({ landId: LAND.id, revision: 1, rasterId: "raster-1" }));
  act(() => useLand.getState().select({ ...LAND, id: "land-b" }));
  expect(useShowOnce.getState().session).toBeNull();
});

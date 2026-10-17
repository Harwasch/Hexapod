import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render } from "@testing-library/react";
import { EntityCollection } from "cesium";
import type * as Cesium from "cesium";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { KnowledgeGrid, LandArea, LonLatGrid } from "@twin/contracts";

import { FOG_SHADER, RAMP_SHADER } from "@/cesium/GroundGridLayer";
import { KnowledgeFogBridge } from "@/cesium/KnowledgeFogBridge";
import { useGridDrape } from "@/state/gridDrape";
import { useKnowledge } from "@/state/knowledge";
import { useLand } from "@/state/land";

const fixture = vi.hoisted(() => ({
  primitives: [] as { options: Record<string, unknown>; index: number | undefined }[],
  removed: [] as unknown[],
  materials: [] as { uniforms: Record<string, unknown> }[],
  listeners: new Map<string, (value: unknown) => void>(),
  released: 0,
  wanted: 0,
  claims: null as ((longitude: number, latitude: number) => boolean) | null,
  render: vi.fn(),
  entities: null as EntityCollection | null,
  data: undefined as KnowledgeGrid | undefined,
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
  // One host for the whole test, as the app's SceneContext gives.
  let host: unknown = null;
  return {
    useScene: () =>
      (host ??= {
        isDestroyed: false,
        viewer: {
          get entities() {
            return fixture.entities;
          },
          scene: {
            requestRender: fixture.render,
            isDestroyed: () => false,
            groundPrimitives: {
              add: (primitive: { options: Record<string, unknown> }, index?: number) => {
                fixture.primitives.push({ options: primitive.options, index });
                return primitive;
              },
              remove: (primitive: unknown) => fixture.removed.push(primitive),
            },
          },
        },
        scene: { requestRender: fixture.render },
        selection: {
          wantGroundTaps: (claims: (longitude: number, latitude: number) => boolean) => {
            fixture.wanted++;
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

vi.mock("@/features/knowledge/api", () => ({
  useKnowledgeGrid: (land: { id: string; revision: number } | null, enabled: boolean) => ({
    data: enabled && fixture.data?.landId === land?.id ? fixture.data : undefined,
    isLoading: false,
  }),
}));

const frame = { west: -120.05, south: 34.69, east: -120.03, north: 34.7, columns: 2, rows: 2 };

function grid(landId: string, density: number[]): KnowledgeGrid {
  const u8: LonLatGrid = {
    ...frame,
    cellM: 20,
    encoding: "u8",
    scale: 0.004,
    offset: 0,
    nodata: 255,
    data: btoa(String.fromCharCode(...density)),
  };
  const best: LonLatGrid = {
    ...u8,
    encoding: "u16",
    scale: 1,
    nodata: 65535,
    data: btoa(String.fromCharCode(...density.flatMap(() => [255, 255]))),
  };
  return {
    landId,
    boundaryRevision: 1,
    asOf: "2026-10-16T00:00:00Z",
    density: u8,
    best,
    records: [],
    summary: { insideCells: 3, byBestClass: { none: 1 }, foggyFraction: 1, medianDensity: 1 },
    policy: {
      version: "fog-v1",
      classes: [],
      undatedFactor: 0.25,
      fieldReachM: {},
      foggyThreshold: 0.5,
      formula: "",
    },
    notCounted: [],
  };
}

function land(id: string, revision = 1): LandArea {
  return { id, name: id, revision } as LandArea;
}

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  fixture.entities = new EntityCollection();
  fixture.data = grid("land-a", [250, 250, 255, 120]);
  useLand.getState().select(land("land-a"));
  useKnowledge.getState().setVisible(true);
});

afterEach(() => {
  cleanup();
  useKnowledge.getState().setVisible(false);
  useKnowledge.getState().clear();
  useLand.getState().clear();
  useGridDrape.setState({ owner: null });
  fixture.primitives.length = 0;
  fixture.removed.length = 0;
  fixture.materials.length = 0;
  fixture.listeners.clear();
  fixture.released = 0;
  fixture.wanted = 0;
  fixture.claims = null;
  vi.clearAllMocks();
});

it("drapes one unpickable ground primitive first, and a tap inside opens that cell", () => {
  render(<KnowledgeFogBridge />, { wrapper });
  expect(fixture.primitives).toHaveLength(1);
  const [added] = fixture.primitives;
  expect(added?.index).toBe(0);
  expect(added?.options.allowPicking).toBe(false);
  expect(useGridDrape.getState().owner).toBe("fog");
  expect(fixture.wanted).toBe(1);
  // North-east cell (col 1, row 0) is inside.
  act(() =>
    fixture.listeners.get("ground-tap")?.({ longitude: -120.035, latitude: 34.698, x: 1, y: 1 }),
  );
  expect(useKnowledge.getState().selected).toMatchObject({ landId: "land-a", col: 1, row: 0 });
  expect(fixture.entities?.getById("grid-outline:fog")).toBeDefined();
  // South-west (col 0, row 1) is outside the land: the card closes.
  act(() =>
    fixture.listeners.get("ground-tap")?.({ longitude: -120.045, latitude: 34.692, x: 1, y: 1 }),
  );
  expect(useKnowledge.getState().selected).toBeNull();
  expect(fixture.entities?.getById("grid-outline:fog")).toBeUndefined();
});

it("claims a zone's fill as ground only inside the shown grid's land", () => {
  const view = render(<KnowledgeFogBridge />, { wrapper });
  const claims = fixture.claims!;
  // North-east (col 1, row 0) is inside: a click on a zone there opens the cell.
  expect(claims(-120.035, 34.698)).toBe(true);
  // South-west is outside the land, and past the grid is no land at all: the zone selects.
  expect(claims(-120.045, 34.692)).toBe(false);
  expect(claims(-120.2, 34.698)).toBe(false);
  // No grid shown (still loading, or failed): nothing is claimed.
  fixture.data = undefined;
  view.rerender(<KnowledgeFogBridge />);
  expect(fixture.claims!(-120.035, 34.698)).toBe(false);
});

it("a revised boundary closes the card and never draws the older grid", () => {
  const view = render(<KnowledgeFogBridge />, { wrapper });
  act(() =>
    useKnowledge.getState().select({ landId: "land-a", longitude: 0, latitude: 0, col: 1, row: 0 }),
  );
  // The boundary was revised: the grid in hand is of revision 1.
  act(() => useLand.getState().select(land("land-a", 2)));
  view.rerender(<KnowledgeFogBridge />);
  expect(useKnowledge.getState().selected).toBeNull();
  expect(fixture.removed).toHaveLength(1);
  expect(useGridDrape.getState().owner).toBeNull();
  // Its own grid arrives: drawn again.
  fixture.data = { ...grid("land-a", [250, 250, 255, 120]), boundaryRevision: 2 };
  view.rerender(<KnowledgeFogBridge />);
  expect(fixture.primitives).toHaveLength(2);
});

it("hiding releases the ground taps and the drape", () => {
  render(<KnowledgeFogBridge />, { wrapper });
  act(() => useKnowledge.getState().setVisible(false));
  expect(fixture.released).toBe(1);
  expect(fixture.removed).toHaveLength(1);
  expect(useGridDrape.getState().owner).toBeNull();
});

it("changing the land disposes the drape and closes the card", () => {
  render(<KnowledgeFogBridge />, { wrapper });
  act(() =>
    useKnowledge.getState().select({ landId: "land-a", longitude: 0, latitude: 0, col: 1, row: 0 }),
  );
  act(() => useLand.getState().select(land("land-b")));
  expect(fixture.removed).toHaveLength(1);
  expect(useKnowledge.getState().selected).toBeNull();
});

it("dissolves towards new evidence, and resolves a map-menu point into a cell", () => {
  const view = render(<KnowledgeFogBridge />, { wrapper });
  const material = fixture.materials[0]!;
  const shown = material.uniforms.densityFrom;
  fixture.data = grid("land-a", [100, 250, 255, 120]);
  view.rerender(<KnowledgeFogBridge />);
  expect(fixture.primitives).toHaveLength(1);
  expect(material.uniforms.lift).toBe(0);
  expect(material.uniforms.densityTo).not.toBe(shown);
  act(() => useKnowledge.getState().selectAt("land-a", -120.045, 34.698));
  expect(useKnowledge.getState().selected).toMatchObject({ col: 0, row: 0 });
});

it("gives way when another drape takes the ground", () => {
  render(<KnowledgeFogBridge />, { wrapper });
  act(() => useGridDrape.getState().claim("signature"));
  expect(fixture.removed).toHaveLength(1);
  expect(useKnowledge.getState().visible).toBe(false);
});

it("exports shaders that sample the texture, fog through fbm noise", () => {
  expect(FOG_SHADER).toContain("texture(densityFrom, st)");
  expect(FOG_SHADER).toContain("knowledgeFogFbm");
  expect(FOG_SHADER).toContain("smoothstep(n - 0.08, n + 0.08, lift)");
  expect(RAMP_SHADER).toContain("step(threshold, cell.r)");
});

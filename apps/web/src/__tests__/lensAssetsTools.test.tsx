import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { LandArea, LandMapGeometry } from "@twin/contracts";

import { api } from "@/api/client";
import { AssetsGeometryEditor } from "@/features/land/surface/lenses/assets/AssetsGeometryEditor";
import { AssetsImport } from "@/features/land/surface/lenses/assets/AssetsImport";
import { AssetsInspectionForm } from "@/features/land/surface/lenses/assets/AssetsInspectionForm";
import {
  mergeAssetDraft,
  parseInventoryDraft,
  type AssetDraft,
} from "@/features/land/surface/lenses/assets/assetsDraft";
import {
  closeInventoryShape,
  keepGeometryHistory,
  openInventoryShape,
} from "@/features/land/surface/lenses/assets/assetsGeometry";
import {
  csvRows,
  defaultImportMapping,
  parseInventoryImport,
} from "@/features/land/surface/lenses/assets/assetsImport";
import { landScope } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";

// The full tools behind the Assets lens: file import, exact coordinates, inspections and the
// draft helpers that keep unsaved work recoverable.

const scene = vi.hoisted(() => ({ areas: { pickGround: vi.fn(), cancelPick: vi.fn() } }));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));

const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Public software fixture",
  description: "",
  revision: 1,
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-77.055, 38.887],
        [-77.05, 38.887],
        [-77.05, 38.889],
        [-77.055, 38.887],
      ],
    ],
  },
  source: { method: "drawn", label: "Synthetic" },
  areaM2: 10000,
  perimeterM: 500,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const file = JSON.stringify({
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      id: "pole-a",
      properties: { name: "Pole A", nested: { accuracy: 2 } },
      geometry: { type: "Point", coordinates: [-77.05, 38.888, 10] },
    },
  ],
});
const geometry: LandMapGeometry = {
  type: "MultiPolygon",
  coordinates: [
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4],
        [0, 0],
      ],
      [
        [1, 1],
        [1, 2],
        [2, 2],
        [2, 1],
        [1, 1],
      ],
    ],
    [
      [
        [6, 0],
        [8, 0],
        [8, 2],
        [6, 0],
      ],
    ],
  ],
};
const draft: AssetDraft = {
  requestKey: "22222222-2222-4222-8222-222222222222",
  name: "Pole",
  category: "other",
  geometry: { type: "Point", coordinates: [0.1, 0.1] },
  source: { method: "drawn", label: "Synthetic fixture" },
};
const snapshot = {
  version: 1,
  landId: land.id,
  boundaryRevision: 1,
  draft,
  editing: null,
  revisionNote: "Recovered change",
  working: null,
};

function withClient(children: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

afterEach(() => {
  cleanup();
  localStorage.clear();
  useLandContext.getState().clear();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

it("parses CSV quoting without swapping coordinates or treating empty coordinates as zero", () => {
  expect(csvRows('name,longitude,latitude\r\n"A, ""quoted""\nname",-77.05,38.888\r\n')).toEqual([
    ["name", "longitude", "latitude"],
    ['A, "quoted"\nname', "-77.05", "38.888"],
  ]);
  const rows = parseInventoryImport(
    "name,longitude,latitude\nA,-77.05,38.888\nB,,38.888",
    "csv",
    defaultImportMapping,
  );
  expect(rows[0]?.geometry).toEqual({ type: "Point", coordinates: [-77.05, 38.888] });
  expect(rows[1]?.error).toContain("finite WGS 84");
  expect(() => csvRows('a,b\n"unclosed')).toThrow("unclosed");
  expect(() =>
    parseInventoryImport("longitude,longitude,latitude\n1,2,3", "csv", defaultImportMapping),
  ).toThrow("unique");
});
it("preserves holes and multipart IDs, flags dropped attributes, and rejects foreign CRS", () => {
  const rows = parseInventoryImport(file, "geojson", defaultImportMapping);
  expect(rows[0]).toMatchObject({
    recordId: "pole-a",
    generatedId: false,
    geometry: { type: "Point", coordinates: [-77.05, 38.888] },
  });
  expect(rows[0]?.warnings[0]).toContain("nested values");
  const polygon = {
    type: "Polygon",
    coordinates: [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 0],
      ],
      [
        [1, 0.2],
        [1.2, 0.2],
        [1.2, 0.4],
        [1, 0.2],
      ],
    ],
  };
  expect(
    parseInventoryImport(JSON.stringify(polygon), "geojson", defaultImportMapping)[0]?.geometry,
  ).toEqual(polygon);
  const parts = parseInventoryImport(
    JSON.stringify({
      type: "Feature",
      id: "set",
      properties: {},
      geometry: {
        type: "MultiPoint",
        coordinates: [
          [1, 2],
          [3, 4],
        ],
      },
    }),
    "geojson",
    defaultImportMapping,
  );
  expect(parts.map((r) => r.recordId)).toEqual(["set:part:1", "set:part:2"]);
  expect(() =>
    parseInventoryImport(
      JSON.stringify({
        type: "Point",
        coordinates: [1, 2],
        crs: { type: "name", properties: { name: "EPSG:3857" } },
      }),
      "geojson",
      defaultImportMapping,
    ),
  ).toThrow("Reproject");
  expect(
    parseInventoryImport(
      JSON.stringify({ type: "GeometryCollection", geometries: [] }),
      "geojson",
      defaultImportMapping,
    )[0]?.error,
  ).toContain("Unsupported");
  expect(() =>
    parseInventoryImport(
      JSON.stringify({
        type: "MultiPoint",
        coordinates: Array.from({ length: 201 }, () => [1, 2]),
      }),
      "geojson",
      defaultImportMapping,
    ),
  ).toThrow("200");
});
it("recovers the reviewed input and uses the same request after a failed save", async () => {
  const draft = {
    version: 1,
    text: file,
    format: "geojson",
    fileName: "test.geojson",
    sha256: "a".repeat(64),
    namespace: "file-sha256:" + "a".repeat(64),
    mapping: defaultImportMapping,
    excluded: [],
    names: {},
    categories: {},
    requestKey: "22222222-2222-4222-8222-222222222222",
    boundaryRevision: 1,
  };
  const key = `living-world-land-draft:${encodeURIComponent(landScope())}:inventory-import:${land.id}`;
  localStorage.setItem(key, JSON.stringify(draft));
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  const rows = [
    {
      rowId: "1",
      featureId: "asset",
      name: "Pole A",
      disposition: "created",
      geometryPreview: {
        geometry: { type: "Point", coordinates: [-77.05, 38.888] },
        intersectsLand: true,
        distanceM: 0,
        areaM2: null,
        lengthM: null,
        perimeterM: null,
      },
    },
  ];
  const post = vi
    .spyOn(api, "POST")
    .mockResolvedValueOnce({ data: { boundaryRevision: 1, rows }, response: new Response() })
    .mockRejectedValueOnce(new Error("Connection lost"))
    .mockResolvedValueOnce({
      data: {
        id: "receipt",
        landId: land.id,
        sourceLabel: draft.fileName,
        sourceFileSha256: draft.sha256,
        boundaryRevision: 1,
        requestKey: draft.requestKey,
        rows,
        createdAt: "2026-10-09",
      },
      response: new Response(),
    });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  // The land's fog, computed from its records (the assets among them).
  const fog = ["land-knowledge", landScope(), land.id, 20];
  client.setQueryData(fog, { cells: [] });
  render(
    <QueryClientProvider client={client}>
      <AssetsImport land={land} active />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Recover inventory import" }));
  expect(screen.getByLabelText("Row 1 name")).toHaveValue("Pole A");
  fireEvent.click(screen.getByRole("button", { name: "Preview selected assets" }));
  await screen.findByText(/1 new assets/);
  fireEvent.click(screen.getByRole("button", { name: "Save reviewed import" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Connection lost");
  expect(client.getQueryState(fog)?.isInvalidated).toBe(false);
  fireEvent.click(screen.getByRole("button", { name: "Save reviewed import" }));
  await screen.findByRole("heading", { name: "Import saved" });
  // Imported assets are the land's records: its fog is asked again.
  expect(client.getQueryState(fog)?.isInvalidated).toBe(true);
  const calls = post.mock.calls as unknown as [string, { body: unknown }][];
  expect(calls[1]?.[1].body).toEqual(calls[2]?.[1].body);
  expect(calls[1]?.[1].body).toMatchObject({
    requestKey: draft.requestKey,
    rows: [{ feature: { status: "candidate", externalRef: { recordId: "pole-a" } } }],
  });
  await waitFor(() => expect(localStorage.getItem(key)).toBeNull());
});

it("preserves every part and hole without exposing a duplicate closing vertex", () => {
  const shape = openInventoryShape(geometry);
  expect(shape.parts[0]?.[0]).toHaveLength(4);
  expect(closeInventoryShape(shape)).toEqual(geometry);
  expect(() =>
    closeInventoryShape({
      type: "Polygon",
      parts: [
        [
          [
            [1, 1],
            [2, 2],
            [NaN, 3],
          ],
        ],
      ],
    }),
  ).toThrow("every vertex");
  expect(() =>
    closeInventoryShape({
      type: "Polygon",
      parts: [
        [
          [
            [1, 1],
            [1, 1],
            [2, 2],
          ],
        ],
      ],
    }),
  ).toThrow("three distinct");
  const big = {
    type: "LineString" as const,
    parts: [[Array.from({ length: 20000 }, () => [1, 1] as [number, number])]],
  };
  expect(keepGeometryHistory(Array.from({ length: 30 }, () => big))).toHaveLength(5);
});
it("keeps holes through edits, invalidates preview, and restores removed parts with undo", async () => {
  const post = vi.spyOn(api, "POST").mockResolvedValue({
    data: {
      geometry,
      intersectsLand: true,
      distanceM: 0,
      areaM2: 100,
      lengthM: null,
      perimeterM: 50,
    },
    response: new Response(),
  });
  const apply = vi.fn();
  render(
    <AssetsGeometryEditor
      landId="land"
      active
      boundaryRevision={1}
      geometry={geometry}
      onApply={apply}
      onCancel={vi.fn()}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Remove area part" }));
  expect(screen.getByLabelText("Area part").querySelectorAll("option")).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Undo geometry" }));
  expect(screen.getByLabelText("Area part").querySelectorAll("option")).toHaveLength(2);
  fireEvent.change(screen.getByLabelText("Boundary ring"), { target: { value: "1" } });
  fireEvent.change(screen.getByLabelText("Vertex 1 longitude"), { target: { value: "1.1" } });
  fireEvent.click(screen.getByRole("button", { name: "Preview geometry" }));
  await screen.findByText("Geometry is valid.");
  const calls = post.mock.calls as unknown as [string, { body: { geometry: typeof geometry } }][];
  const body = calls[0]![1].body;
  expect(body.geometry.coordinates).toEqual([
    [
      geometry.coordinates[0]![0],
      [
        [1.1, 1],
        [1, 2],
        [2, 2],
        [2, 1],
        [1.1, 1],
      ],
    ],
    geometry.coordinates[1],
  ]);
  fireEvent.change(screen.getByLabelText("Vertex 1 latitude"), { target: { value: "1.2" } });
  expect(screen.queryByText("Geometry is valid.")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Apply geometry to draft" })).toBeDisabled();
  expect(apply).not.toHaveBeenCalled();
});
it("ignores stale preview results and invalidates review after a boundary revision", async () => {
  let resolve!: (value: unknown) => void;
  vi.spyOn(api, "POST").mockImplementation(
    () =>
      new Promise((r) => {
        resolve = r;
      }) as never,
  );
  const props = {
    landId: "land",
    active: true,
    boundaryRevision: 1,
    geometry,
    onApply: vi.fn(),
    onCancel: vi.fn(),
  };
  const view = render(<AssetsGeometryEditor {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Preview geometry" }));
  fireEvent.change(screen.getByLabelText("Vertex 1 longitude"), { target: { value: "0.1" } });
  await act(
    async () =>
      await Promise.resolve(
        resolve({
          data: {
            geometry,
            intersectsLand: true,
            distanceM: 0,
            areaM2: 100,
            lengthM: null,
            perimeterM: 50,
          },
          response: new Response(),
        }),
      ),
  );
  expect(screen.getByRole("button", { name: "Apply geometry to draft" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Preview geometry" }));
  await act(
    async () =>
      await Promise.resolve(
        resolve({
          data: {
            geometry,
            intersectsLand: true,
            distanceM: 0,
            areaM2: 100,
            lengthM: null,
            perimeterM: 50,
          },
          response: new Response(),
        }),
      ),
  );
  expect(screen.getByRole("button", { name: "Apply geometry to draft" })).toBeEnabled();
  view.rerender(<AssetsGeometryEditor {...props} boundaryRevision={2} />);
  expect(screen.getByRole("button", { name: "Apply geometry to draft" })).toBeDisabled();
});
it("cancels map picking when hidden and does not claim another picker's selection", async () => {
  const hide = vi.fn();
  let resolve!: (value: unknown) => void;
  scene.areas.pickGround.mockImplementation(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  const props = {
    landId: "land",
    boundaryRevision: 1,
    geometry: { type: "Point" as const, coordinates: [1, 2] },
    onApply: vi.fn(),
    onCancel: hide,
  };
  const view = render(<AssetsGeometryEditor {...props} active />);
  fireEvent.click(screen.getByRole("button", { name: "Pick vertex 1 on map" }));
  expect(useLandContext.getState().pointPicker).toBe("assets-geometry:land");
  view.rerender(<AssetsGeometryEditor {...props} active={false} />);
  await waitFor(() => expect(scene.areas.cancelPick).toHaveBeenCalledOnce());
  act(() => {
    useLandContext.getState().setPointPicker("other-tool");
  });
  await act(async () => {
    resolve({ longitude: 3, latitude: 4 });
    await Promise.resolve();
  });
  expect(screen.getByLabelText("Vertex 1 longitude")).toHaveValue(1);
  expect(useLandContext.getState().pointPicker).toBe("other-tool");
});

it("recovers unfinished coordinates without changing blanks to zero or crossing land scope", () => {
  const text = JSON.stringify({
    ...snapshot,
    working: {
      type: "LineString",
      parts: [
        [
          [
            [0.1, 0.1],
            [NaN, 0.2],
          ],
        ],
      ],
    },
  });
  expect(parseInventoryDraft(text, land.id).working?.parts[0]?.[0]?.[1]?.[0]).toBeNaN();
  expect(() => parseInventoryDraft(text, "other-land")).toThrow("current land");
  expect(() =>
    parseInventoryDraft(
      JSON.stringify({ ...snapshot, working: { type: "Point", parts: [[[["bad", 0.2]]]] } }),
      land.id,
    ),
  ).toThrow("vertices");
});
it("merges independent changes and requires explicit choices for conflicts, including unfinished geometry", () => {
  const baseline = { ...draft, attributes: { a: 1, b: 2 } };
  const mine = { ...baseline, name: "My name" };
  const current = { ...baseline, category: "power" as const, attributes: { b: 2, a: 1 } };
  const merged = mergeAssetDraft(baseline, mine, current);
  expect(merged.conflicts).toEqual([]);
  expect(merged.merged).toMatchObject({
    name: "My name",
    category: "power",
    requestKey: draft.requestKey,
  });
  expect(mergeAssetDraft(baseline, mine, { ...current, name: "Another name" }).conflicts).toEqual([
    "name",
  ]);
  expect(
    mergeAssetDraft(
      baseline,
      mine,
      { ...current, geometry: { type: "Point", coordinates: [0.2, 0.2] } },
      true,
    ).conflicts,
  ).toEqual(["geometry"]);
});

it("records the condition, local time, measurement units and a stable retry key", async () => {
  const post = vi
    .spyOn(api, "POST")
    .mockRejectedValueOnce(new Error("Network unavailable"))
    .mockResolvedValue({ data: {}, response: new Response() });
  const saved = vi.fn().mockResolvedValue(undefined);
  render(
    withClient(
      <AssetsInspectionForm landId="land" featureId="pole" onSaved={saved} onCancel={vi.fn()} />,
    ),
  );
  fireEvent.change(screen.getByLabelText("What did you see?"), {
    target: { value: "Corrosion at base" },
  });
  expect(screen.getByRole("button", { name: "Save inspection" })).toBeDisabled();
  fireEvent.click(screen.getByRole("radio", { name: "Poor" }));
  fireEvent.change(screen.getByLabelText("When"), { target: { value: "2026-10-09T12:30" } });
  fireEvent.click(screen.getByText(/Measurements and sources/));
  fireEvent.click(screen.getByRole("button", { name: "Add measurement" }));
  for (const [field, value] of Object.entries({ name: "height", value: "21.5", unit: "m" }))
    fireEvent.change(screen.getByLabelText(`Measurement 1 ${field}`), { target: { value } });
  fireEvent.click(screen.getByRole("button", { name: "Save inspection" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Network unavailable");
  fireEvent.click(screen.getByRole("button", { name: "Save inspection" }));
  await waitFor(() => expect(saved).toHaveBeenCalledOnce());
  const calls = post.mock.calls as unknown as [string, { body: unknown }][];
  expect(calls[0]?.[1].body).toMatchObject({
    condition: "poor",
    notes: "Corrosion at base",
    observedAt: new Date("2026-10-09T12:30").toISOString(),
    measurements: { height: 21.5 },
    measurementUnits: { height: "m" },
  });
  expect(calls[0]?.[1].body).toEqual(calls[1]?.[1].body);
});

it("rejects duplicate measurement names without submitting", async () => {
  const post = vi.spyOn(api, "POST");
  render(
    withClient(
      <AssetsInspectionForm
        landId="land"
        featureId="pole"
        onSaved={() => Promise.resolve()}
        onCancel={vi.fn()}
      />,
    ),
  );
  fireEvent.click(screen.getByRole("radio", { name: "Good" }));
  fireEvent.change(screen.getByLabelText("What did you see?"), { target: { value: "Checked" } });
  fireEvent.click(screen.getByText(/Measurements and sources/));
  for (const index of [1, 2]) {
    fireEvent.click(screen.getByRole("button", { name: "Add measurement" }));
    for (const [field, value] of Object.entries({ name: "height", value: "20", unit: "m" }))
      fireEvent.change(screen.getByLabelText(`Measurement ${index} ${field}`), {
        target: { value },
      });
  }
  fireEvent.click(screen.getByRole("button", { name: "Save inspection" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("different name");
  expect(post).not.toHaveBeenCalled();
});

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { InventoryImport } from "@/features/land/InventoryImport";
import {
  csvRows,
  defaultImportMapping,
  parseInventoryImport,
} from "@/features/land/inventoryImport";
import { landScope } from "@/state/landIdentity";
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
afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
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
  render(
    <QueryClientProvider client={client}>
      <InventoryImport land={land} />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Recover inventory import" }));
  expect(screen.getByLabelText("Row 1 name")).toHaveValue("Pole A");
  fireEvent.click(screen.getByRole("button", { name: "Preview selected assets" }));
  await screen.findByText(/1 new candidates/);
  fireEvent.click(screen.getByRole("button", { name: "Save reviewed import" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Connection lost");
  fireEvent.click(screen.getByRole("button", { name: "Save reviewed import" }));
  await screen.findByRole("heading", { name: "Import saved" });
  const calls = post.mock.calls as unknown as [string, { body: unknown }][];
  expect(calls[1]?.[1].body).toEqual(calls[2]?.[1].body);
  expect(calls[1]?.[1].body).toMatchObject({
    requestKey: draft.requestKey,
    rows: [{ feature: { status: "candidate", externalRef: { recordId: "pole-a" } } }],
  });
  await waitFor(() => expect(localStorage.getItem(key)).toBeNull());
});

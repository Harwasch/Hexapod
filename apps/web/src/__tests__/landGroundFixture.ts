/**
 * A saved Land and its ground as the API returns them, for the land-planning tests: a 1 km
 * square near Santa Ynez with a fence along its north edge and a stock pond.
 */

import type { LandArea, LandGround } from "@twin/contracts";
import { destination } from "@twin/geo";

export const origin = { longitude: -120.04, latitude: 34.7 };
const at = (east: number, north: number): [number, number] => {
  const p = destination(destination(origin, 90, east), 0, north);
  return [p.longitude, p.latitude];
};
const square = (west: number, south: number, size: number) => ({
  type: "Polygon" as const,
  coordinates: [
    [
      at(west, south),
      at(west + size, south),
      at(west + size, south + size),
      at(west, south + size),
      at(west, south),
    ],
  ],
});

export const LAND_ID = "0f1e2d3c-4b5a-4968-8776-655443322110";
export const land: LandArea = {
  id: LAND_ID,
  name: "Oak Bench",
  description: "",
  boundary: square(0, 0, 1000),
  source: { method: "drawn", label: "Drawn on map", meaning: "study-area", references: [] },
  revision: 2,
  areaM2: 1_000_000,
  perimeterM: 4000,
  createdAt: "2026-10-01T00:00:00Z",
  updatedAt: "2026-10-02T00:00:00Z",
};

export const ground: LandGround = {
  landId: LAND_ID,
  name: "Oak Bench",
  boundaryRevision: 2,
  skipped: 1,
  items: [
    {
      zoneId: "L-0f1e2d3c",
      kind: "boundary",
      featureId: null,
      revision: 2,
      name: "Oak Bench",
      category: null,
      status: null,
      footprint: square(0, 0, 1000),
      line: null,
      areaM2: 1_000_000,
      lengthM: null,
      bufferM: null,
      source:
        "Saved boundary, revision 2 · Drawn on map · study area, not a verified property boundary",
    },
    {
      zoneId: "L-aaaa0001",
      kind: "feature",
      featureId: "aaaa0001-0000-4000-8000-000000000001",
      revision: 3,
      name: "North fence",
      category: "other",
      status: "confirmed",
      footprint: {
        type: "Polygon",
        coordinates: [[at(0, 990), at(1000, 990), at(1000, 1000), at(0, 1000), at(0, 990)]],
      },
      line: { type: "LineString", coordinates: [at(0, 995), at(1000, 995)] },
      areaM2: 10_000,
      lengthM: 1000,
      bufferM: 10,
      source: "Inventory · confirmed · revision 3 · Walked with a GPS",
    },
    {
      zoneId: "L-aaaa0002",
      kind: "feature",
      featureId: "aaaa0002-0000-4000-8000-000000000002",
      revision: 1,
      name: "Stock pond",
      category: "water",
      status: "candidate",
      footprint: square(400, 400, 40),
      line: null,
      areaM2: 1600,
      lengthM: null,
      bufferM: null,
      source: "Inventory · candidate · revision 1 · OpenStreetMap",
    },
  ],
};

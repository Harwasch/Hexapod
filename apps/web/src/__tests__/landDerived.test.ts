/**
 * What a change to a Land's records asks again (features/land/landDerived.ts): a research run
 * finishing refreshes the fog and the rasters everywhere, but never the workable ground, whose
 * key names everything it is read from.
 */
import { QueryClient } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";

import type { LandArea } from "@twin/contracts";

import { api } from "@/api/client";
import { invalidateLandDerived } from "@/features/land/landDerived";
import { saveDraft } from "@/features/land/surface/landFlow";
import { useLand } from "@/state/land";

const KEYS = {
  fog: ["land-knowledge", "pilot", "land-1", 20],
  otherFog: ["land-knowledge", "pilot", "land-2", 20],
  rasters: ["land-rasters", "pilot", "raster-9"],
  workable: ["land-workable", "pilot", "land-1", "Z-14", "812:abc", "mower", "raster-1", false],
  otherWorkable: [
    "land-workable",
    "pilot",
    "land-2",
    "A-L01",
    "90:def",
    "mower",
    "raster-2",
    false,
  ],
  plans: ["plans", "demo-ranch"],
};

function seeded(): QueryClient {
  const client = new QueryClient();
  for (const key of Object.values(KEYS)) client.setQueryData(key, { answer: true });
  return client;
}

const stale = (client: QueryClient) =>
  Object.entries(KEYS)
    .filter(([, key]) => client.getQueryState(key)?.isInvalidated)
    .map(([name]) => name)
    .sort();

it("refreshes every land's fog and rasters when a run finishes, never its workable ground", async () => {
  const client = seeded();
  await invalidateLandDerived(client, null);
  expect(stale(client)).toEqual(["fog", "otherFog", "rasters"]);
});

it("refreshes one land's workable ground when that land's records change", async () => {
  const client = seeded();
  await invalidateLandDerived(client, "land-1");
  expect(stale(client)).toEqual(["fog", "rasters", "workable"]);
});

const LAND: LandArea = {
  id: "land-1",
  name: "West field",
  description: "",
  revision: 3,
  areaM2: 1,
  perimeterM: 4,
  createdAt: "2026-10-16",
  updatedAt: "2026-10-16",
  source: { method: "drawn", label: "Drawn on the map", meaning: "study-area" },
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-120.72, 35.53],
        [-120.71, 35.53],
        [-120.71, 35.54],
        [-120.72, 35.53],
      ],
    ],
  },
};

afterEach(() => {
  useLand.getState().clear();
  vi.restoreAllMocks();
});

it("asks a land's derived answers again when a revised boundary is saved", async () => {
  const client = seeded();
  const put = vi
    .spyOn(api, "PUT")
    .mockResolvedValue({ data: { ...LAND, revision: 4 }, response: new Response() });
  useLand.getState().select(LAND);
  useLand.getState().propose({ ...LAND, description: "Fence moved east" });
  await saveDraft(client);
  expect(put).toHaveBeenCalledOnce();
  expect(useLand.getState().active?.revision).toBe(4);
  expect(stale(client)).toEqual(["fog", "rasters", "workable"]);
});

it("leaves every land's derived answers alone when a new land is saved", async () => {
  const client = seeded();
  vi.spyOn(api, "POST").mockResolvedValue({
    data: { ...LAND, id: "land-3" },
    response: new Response(),
  });
  useLand.getState().propose({ ...LAND, name: "New field" });
  await saveDraft(client);
  expect(useLand.getState().active?.id).toBe("land-3");
  expect(stale(client)).toEqual([]);
});

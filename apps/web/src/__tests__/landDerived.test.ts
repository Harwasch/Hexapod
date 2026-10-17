/**
 * What a change to a Land's records asks again (features/land/landDerived.ts): a research run
 * finishing refreshes the fog and the rasters everywhere, but never the workable ground, whose
 * key names everything it is read from.
 */
import { QueryClient } from "@tanstack/react-query";
import { expect, it } from "vitest";

import { invalidateLandDerived } from "@/features/land/landDerived";

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

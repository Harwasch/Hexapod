import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { LandInventory } from "@/features/land/LandInventory";
import {
  mergeAssetDraft,
  parseInventoryDraft,
  type AssetDraft,
  type AssetRecord,
} from "@/features/land/inventoryDraft";
import { landScope } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Synthetic land",
  description: "",
  revision: 1,
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 0],
      ],
    ],
  },
  source: { method: "drawn", label: "Synthetic" },
  areaM2: 1,
  perimeterM: 1,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
const draft: AssetDraft = {
  requestKey: "22222222-2222-4222-8222-222222222222",
  name: "Pole",
  category: "other",
  geometry: { type: "Point", coordinates: [0.1, 0.1] },
  source: { method: "drawn", label: "Synthetic fixture" },
};
const record: AssetRecord = {
  ...draft,
  source: {
    ...draft.source,
    url: null,
    recordId: null,
    observedAt: null,
    attribution: null,
    meaning: "study-area",
  },
  id: "33333333-3333-4333-8333-333333333333",
  landId: land.id,
  revision: 1,
  boundaryRevision: 1,
  intersectsLand: true,
  distanceM: 0,
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
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
const key = () => `living-world-land-draft:${encodeURIComponent(landScope())}:inventory:${land.id}`;
function show() {
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={cache}>
      <LandInventory land={land} />
    </QueryClientProvider>,
  );
}
afterEach(() => {
  cleanup();
  localStorage.clear();
  useLandContext.getState().clear();
  vi.restoreAllMocks();
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
it("recognizes an already saved creation after a lost response without another write", async () => {
  localStorage.setItem(key(), JSON.stringify(snapshot));
  vi.spyOn(api, "GET").mockImplementation(
    (path: string) =>
      Promise.resolve({
        data: path.includes("/requests/")
          ? {
              current: {
                ...record,
                revision: 2,
                name: "Later saved name",
                requestKey: "55555555-5555-4555-8555-555555555555",
              },
              original: { ...draft, source: record.source },
            }
          : path.endsWith("/{feature_id}")
            ? record
            : [],
        response: new Response(),
      }) as never,
  );
  const post = vi.spyOn(api, "POST");
  show();
  fireEvent.click(screen.getByRole("button", { name: "Recover asset draft" }));
  await screen.findByText(/This draft was already saved/);
  expect(localStorage.getItem(key())).toBeNull();
  expect(screen.queryByLabelText("Feature name")).not.toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();
});
it("blocks revision saves until a recovered conflict is reviewed and retains concurrent unrelated fields", async () => {
  const mine = { ...draft, requestKey: "44444444-4444-4444-8444-444444444444", name: "My name" };
  const current: AssetRecord = { ...record, revision: 2, name: "Another name", category: "power" };
  localStorage.setItem(key(), JSON.stringify({ ...snapshot, draft: mine, editing: record }));
  vi.spyOn(api, "GET").mockImplementation(
    (path: string) =>
      Promise.resolve({
        data: path.endsWith("/{feature_id}") ? current : [],
        response: new Response(),
      }) as never,
  );
  const put = vi.spyOn(api, "PUT").mockResolvedValue({
    data: { ...current, ...mine, revision: 3, category: "power" },
    response: new Response(),
  });
  show();
  fireEvent.click(screen.getByRole("button", { name: "Recover asset draft" }));
  await screen.findByRole("region", { name: "Review newer asset revision" });
  expect(screen.getByRole("button", { name: "Save feature revision" })).toBeDisabled();
  fireEvent.click(screen.getByRole("radio", { name: "Keep my name" }));
  fireEvent.click(screen.getByRole("button", { name: "Apply reviewed merge to draft" }));
  expect(screen.getByLabelText("Category")).toHaveValue("power");
  fireEvent.click(screen.getByRole("button", { name: "Save feature revision" }));
  await waitFor(() => expect(put).toHaveBeenCalledOnce());
  const calls = put.mock.calls as unknown as [string, { body: unknown }][];
  expect(calls[0]?.[1].body).toMatchObject({
    expectedRevision: 2,
    name: "My name",
    category: "power",
  });
  await waitFor(() => expect(localStorage.getItem(key())).toBeNull());
});

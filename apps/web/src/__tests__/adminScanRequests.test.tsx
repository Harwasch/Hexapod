import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import type { ScanRequestRead } from "@twin/contracts";

import { api, ApiError } from "@/api/client";
import { areaLabel } from "@/admin/format";
import { RequestsView } from "@/admin/RequestsView";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function renderView() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <RequestsView />
    </QueryClientProvider>,
  );
}

const rows: ScanRequestRead[] = [
  {
    id: "2a1f0c9e-0000-4000-8000-000000000002",
    createdAt: "2026-10-20T17:05:00Z",
    name: "Ada Lovelace",
    email: "ada@example.com",
    notes: "Organization: Oak Bench Ranch\nBroom on the north slope.",
    boundary: {
      type: "MultiPolygon",
      coordinates: [
        [
          [
            [-122.272, 37.803],
            [-122.27, 37.803],
            [-122.27, 37.805],
            [-122.272, 37.803],
          ],
        ],
      ],
    },
    areaM2: 50_342,
    parcelIds: ["ca-parcel/06001/2-51-1"],
  },
  {
    id: "2a1f0c9e-0000-4000-8000-000000000001",
    createdAt: "2026-10-20T16:00:00Z",
    name: "Grace Hopper",
    email: "grace@example.com",
    notes: "Area (acres): 40",
    boundary: null,
    areaM2: null,
    parcelIds: [],
  },
];

it("lists the requests, newest first, with their land and notes", async () => {
  const get = vi.spyOn(api, "GET").mockResolvedValue({
    data: rows,
    response: new Response(),
  });
  renderView();
  const table = await screen.findByTestId("requests-table");
  const [first, second] = within(table).getAllByTestId("request-row");
  expect(get).toHaveBeenCalledWith("/api/v1/scan-requests", {
    params: { query: { limit: 200 } },
  });
  expect(first).toHaveTextContent("Ada Lovelace");
  expect(within(first!).getByRole("link", { name: "ada@example.com" })).toHaveAttribute(
    "href",
    "mailto:ada@example.com",
  );
  expect(first).toHaveTextContent("12.4 ac · 5.0 ha");
  expect(first).toHaveTextContent("ca-parcel/06001/2-51-1");
  expect(first).toHaveTextContent("Broom on the north slope.");
  expect(second).toHaveTextContent("no outline");
  expect(second).toHaveTextContent("Area (acres): 40");
});

it("asks for the write token when the API refuses without it", async () => {
  vi.spyOn(api, "GET").mockRejectedValue(
    new ApiError(401, { title: "Unauthorized", status: 401 }, "Unauthorized"),
  );
  renderView();
  expect(await screen.findByRole("alert")).toHaveTextContent("enter the write token");
});

it("labels an area in acres and hectares", () => {
  expect(areaLabel(null)).toBe("—");
  expect(areaLabel(4046.856 * 250)).toBe("250 ac · 101.2 ha");
  expect(areaLabel(1800)).toBe("0.44 ac · 0.2 ha");
});

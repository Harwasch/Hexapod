import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { components } from "@twin/contracts";
import type * as Identity from "@/state/landIdentity";

import { api } from "@/api/client";
import { useAgentBus } from "@/features/land/surface/agentBus";
import { useLens } from "@/features/land/surface/lenses";
import { RecordsLens } from "@/features/land/surface/lenses/RecordsLens";
import {
  archiveItems,
  boundaryPhrase,
  namesParcel,
  parcelIds,
  recordsHeadline,
  sheetGroups,
  shortTitle,
  yearOf,
} from "@/features/land/surface/lenses/records/recordsModel";
import { useRecords } from "@/features/land/surface/lenses/records/recordsStore";
import { useLandContext } from "@/state/landContext";
import { useLandMap } from "@/state/landMap";
import { useUi } from "@/state/ui";

import {
  archiveInvestigation,
  deed,
  detail,
  easement,
  land,
  laterSheet,
  photo,
  photoMedia,
  registration,
  sheet,
} from "./lensRecordsFixtures";

type Registration = components["schemas"]["ImageRegistrationRead"];

const identity = vi.hoisted(() => ({ canEdit: true }));
vi.mock("@/state/landIdentity", async (original) => ({
  ...(await original<typeof Identity>()),
  useLandCanEdit: () => identity.canEdit,
}));
const scene = vi.hoisted(() => ({
  camera: { flyToRectangle: vi.fn() },
  areas: { pickGround: vi.fn(), cancelPick: vi.fn() },
}));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));

interface Server {
  investigations: (typeof archiveInvestigation)[];
  detail: ReturnType<typeof detail>;
  documents: (typeof deed)[];
  registrations: Record<string, Registration[]>;
}
let server: Server;
interface Call {
  path: string;
  options?: { params?: { path?: Record<string, string> }; body?: Record<string, unknown> };
}
let posts: Call[] = [];

function route(path: string, options?: Call["options"]): unknown {
  const id = options?.params?.path?.evidence_id ?? "";
  switch (path) {
    case "/api/v1/land/{land_id}/investigations":
      return server.investigations;
    case "/api/v1/research/investigations/{investigation_id}":
      return server.detail;
    case "/api/v1/land/{land_id}/documents":
      return server.documents;
    case "/api/v1/land/{land_id}/documents/links":
      return [];
    case "/api/v1/research/evidence/{evidence_id}/image/registrations":
      return server.registrations[id] ?? [];
    default:
      return null;
  }
}

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={client}>
      <RecordsLens land={land} />
    </QueryClientProvider>,
  );
}

const pins = () => useLandMap.getState().pins.records ?? [];

beforeEach(() => {
  identity.canEdit = true;
  server = {
    investigations: [archiveInvestigation],
    detail: detail([sheet, laterSheet, photo]),
    documents: [easement, deed],
    registrations: {},
  };
  posts = [];
  useLens.getState().setLens("records");
  useUi.getState().setPanel("land");
  useRecords.setState({
    landId: null,
    view: { kind: "summary" },
    overlay: null,
    overlayHidden: false,
    opacity: 0.6,
  });
  useAgentBus.setState({ request: null });
  vi.spyOn(api, "GET").mockImplementation(((path: string, options?: Call["options"]) =>
    Promise.resolve({ data: route(path, options), response: new Response() })) as typeof api.GET);
  vi.spyOn(api, "POST").mockImplementation(((path: string, options?: Call["options"]) => {
    posts.push({ path, options });
    const data = path.endsWith("/investigations")
      ? archiveInvestigation
      : path.endsWith("/cancel")
        ? { ...detail([]).runs[0], status: "cancelled" }
        : { ...detail([]).runs[0], status: "queued" };
    return Promise.resolve({ data, response: new Response() });
  }) as typeof api.POST);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.clearAllMocks();
  useLandContext.getState().clear();
  useLandMap.setState({ pins: {}, focus: null });
});

describe("records words", () => {
  it("says where the outline came from in plain words", () => {
    expect(boundaryPhrase(land.source)).toBe("From a county parcel record (parcel 019-210-034).");
    expect(boundaryPhrase({ method: "drawn", label: "Drawn on the map" })).toBe(
      "Drawn by you on the map.",
    );
    expect(
      boundaryPhrase({
        method: "parcel",
        label: "Combined parcels",
        references: [
          { method: "parcel", label: "A", recordId: "1" },
          { method: "parcel", label: "B", recordId: "2" },
        ],
      }),
    ).toBe("Combined from 2 county parcel records.");
    expect(boundaryPhrase({ method: "imported", label: "farm.kml" })).toBe(
      "Imported from farm.kml.",
    );
    expect(recordsHeadline(land.source, 3, 4, 5)).toBe(
      "From a county parcel record (parcel 019-210-034). 3 documents and 9 old maps and photos.",
    );
    expect(recordsHeadline(land.source, 1, 0, 2)).toMatch(/1 document and 2 old photos\.$/);
    expect(recordsHeadline(land.source, 0, 0, 0)).toBe(
      "From a county parcel record (parcel 019-210-034).",
    );
  });

  it("reads catalog years and titles, matches parcel numbers and groups sheet editions", () => {
    expect(yearOf("circa 1890–1895")).toBe(1890);
    expect(yearOf("2012-05-03 10:12:00")).toBe(2012);
    expect(yearOf("undated")).toBeNull();
    expect(shortTitle("File:Main_Street barn.jpg")).toBe("Main Street barn");
    expect(shortTitle("USGS 1:62500-scale Quadrangle for Mount Vernon, WA 1911")).toBe(
      "Mount Vernon, WA",
    );
    expect(parcelIds(land.source)).toEqual(["019-210-034"]);
    expect(namesParcel(deed, ["019-210-034"])).toBe(true);
    expect(namesParcel(easement, ["019-210-034"])).toBe(false);
    const items = archiveItems([photo, laterSheet, sheet]);
    expect(items.map((item) => item.id)).toEqual(["map", "map-1956", "photo"]);
    expect(sheetGroups(items)).toHaveLength(1);
  });
});

describe("Records lens", () => {
  it("summarises the boundary, documents and archives, and draws them on the map", async () => {
    const view = mount();
    expect(
      await screen.findByRole(
        "heading",
        {
          name: "From a county parcel record (parcel 019-210-034). 2 documents and 3 old maps and photos.",
        },
        { timeout: 5000 },
      ),
    ).toBeVisible();
    expect(screen.getAllByText("Washington county tax-parcel record")[0]).toBeVisible();
    expect(screen.getAllByText(/not a survey/)[0]).toBeVisible();
    expect(screen.getByText("One names this parcel")).toBeVisible();
    expect(screen.getByRole("button", { name: "County deed" })).toBeVisible();
    expect(screen.getByText("Oldest of 2 maps")).toBeVisible();
    expect(screen.getAllByText("1894 · Mount Vernon, WA")[0]).toBeVisible();
    expect(screen.getByText("1 on the land · oldest 1905")).toBeVisible();

    await waitFor(() => expect(pins()).toHaveLength(2), { timeout: 5000 });
    expect(pins().map((pin) => pin.label)).toEqual([
      "1894–1956 · Mount Vernon, WA (2)",
      "1905 · Main Street barn",
    ]);
    expect(useLandContext.getState().layers["records-sheets"]?.features).toHaveLength(1);
    expect(posts).toHaveLength(0);

    view.unmount();
    expect(pins()).toHaveLength(0);
    expect(useLandContext.getState().layers["records-sheets"]).toBeUndefined();
  });

  it("opens an item from its pin, with its licence, and hands the agent a question", async () => {
    mount();
    await waitFor(() => expect(pins()).toHaveLength(2), { timeout: 5000 });
    act(() => pins()[1]?.onSelect?.());
    const item = await screen.findByRole("article", { name: "Main Street barn" });
    expect(within(item).getByRole("link", { name: "CC BY-SA 4.0" })).toHaveAttribute(
      "href",
      photoMedia.licenseUrl,
    );
    expect(within(item).getByText("3 of 3")).toBeVisible();
    expect(pins()[1]?.tone).toBe("warn");
    expect(screen.queryByRole("heading", { name: /county parcel record/ })).toBeNull();

    fireEvent.click(within(item).getByRole("button", { name: "Ask about this" }));
    expect(useAgentBus.getState().request?.question).toContain("(archive source photo)");

    fireEvent.click(within(item).getByRole("button", { name: "Previous source" }));
    const map = await screen.findByRole("article", { name: "Mount Vernon, WA" });
    expect(within(map).getByRole("button", { name: "Line it up with the land" })).toBeVisible();
    expect(useLandContext.getState().layers["records-selected"]?.features).toHaveLength(1);

    fireEvent.click(screen.getByRole("button", { name: "Records" }));
    expect(await screen.findByRole("heading", { name: /county parcel record/ })).toBeVisible();
    expect(useLandContext.getState().layers["records-selected"]).toBeUndefined();
  });

  it("opens the newest document from its fact", async () => {
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "County deed" }));
    expect(useRecords.getState().view).toMatchObject({ kind: "document", id: "deed", page: 1 });
  });

  it("starts the archive search only when asked, and not for viewers", async () => {
    server.investigations = [];
    const view = mount();
    const start = await screen.findByRole("button", { name: "Find old maps and photos" });
    expect(posts).toHaveLength(0);
    fireEvent.click(start);
    await waitFor(() => expect(posts).toHaveLength(2));
    expect(posts[0]?.path).toBe("/api/v1/land/{land_id}/investigations");
    expect(posts[0]?.options?.body).toMatchObject({
      title: "Photographs and historical maps",
      boundaryRevision: 2,
    });
    expect(posts[1]?.options?.body).toMatchObject({ kind: "archive" });
    view.unmount();

    identity.canEdit = false;
    mount();
    expect(await screen.findByText(/An editor can start the search/)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Find old maps and photos" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Add a document" })).toBeNull();
  });

  it("shows a running search that can be stopped, and a failed one that can be retried", async () => {
    server.detail = detail([], "running");
    const view = mount();
    expect(await screen.findByText("Searching photo and map catalogs…")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() => expect(posts[0]?.path).toBe("/api/v1/research/runs/{run_id}/cancel"));
    view.unmount();

    server.detail = {
      ...detail([], "failed"),
      runs: [{ ...detail([]).runs[0]!, status: "failed", error: "Catalog timed out" }],
    };
    mount();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The archive search failed: Catalog timed out",
    );
    expect(screen.getByRole("button", { name: "Try again" })).toBeVisible();
  });

  it("says plainly when nothing turned up, and when results are for an earlier boundary", async () => {
    server.detail = detail([]);
    const view = mount();
    expect(await screen.findByText(/No openly licensed old maps or photos/)).toBeVisible();
    view.unmount();

    server.investigations = [{ ...archiveInvestigation, boundaryRevision: 1 }];
    server.detail = detail([photo]);
    mount();
    expect(await screen.findByText(/Found for an earlier boundary/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Search again" })).toBeVisible();
  });

  it("lays the newest saved alignment over the land and fades it between then and now", async () => {
    server.registrations = {
      map: [
        registration("alignment", {
          name: "1894 sheet",
          notes: "",
          points: [],
          requestKey: crypto.randomUUID(),
          imageSha256: "a".repeat(64),
        }),
      ],
    };
    const view = mount();
    await waitFor(() =>
      expect(useLandContext.getState().rasters.alignment).toMatchObject({
        kind: "archive-alignment",
        opacity: 0.6,
      }),
    );
    const group = await screen.findByRole("group", { name: "Then and now" });
    expect(within(group).getByText("1894 · Mount Vernon, WA")).toBeVisible();
    fireEvent.keyDown(within(group).getByRole("slider", { name: "Aligned map opacity" }), {
      key: "ArrowLeft",
    });
    expect(useLandContext.getState().rasters.alignment?.opacity).toBeCloseTo(0.55);

    fireEvent.click(within(group).getByRole("button", { name: "Hide old map" }));
    expect(useLandContext.getState().rasters.alignment).toBeUndefined();
    fireEvent.click(within(group).getByRole("button", { name: "Show old map" }));
    expect(useLandContext.getState().rasters.alignment?.opacity).toBeCloseTo(0.55);

    view.unmount();
    expect(useLandContext.getState().rasters.alignment).toBeUndefined();
  });
});

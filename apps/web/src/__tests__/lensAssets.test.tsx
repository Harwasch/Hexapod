import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LandArea } from "@twin/contracts";

import { api } from "@/api/client";
import { AssetsLens } from "@/features/land/surface/lenses/AssetsLens";
import {
  assetsHeadline,
  nextName,
  shapeOf,
  shortLabel,
  summarizeAssets,
  type Asset,
  type Inspection,
} from "@/features/land/surface/lenses/assets/assetsModel";
import { useLens } from "@/features/land/surface/lenses";
import type * as identity from "@/state/landIdentity";
import { landScope } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import { useLandMap } from "@/state/landMap";
import { useUi } from "@/state/ui";

const scene = vi.hoisted(() => ({
  areas: { pickGround: vi.fn(), cancelPick: vi.fn() },
  camera: { flyToRectangle: vi.fn() },
}));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));
const role = vi.hoisted(() => ({ canEdit: true }));
vi.mock("@/state/landIdentity", async (original) => ({
  ...(await original<typeof identity>()),
  useLandCanEdit: () => role.canEdit,
}));

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
        [0.01, 0],
        [0.01, 0.01],
        [0, 0.01],
        [0, 0],
      ],
    ],
  },
  source: { method: "drawn", label: "Synthetic" },
  areaM2: 1_000_000,
  perimeterM: 4000,
  createdAt: "2026-10-01",
  updatedAt: "2026-10-01",
};

function asset(id: string, name: string, extra: Partial<Asset> = {}): Asset {
  return {
    id,
    landId: land.id,
    name,
    category: "water",
    status: "confirmed",
    geometry: { type: "Point", coordinates: [0.005, 0.005] },
    source: { method: "drawn", label: "Drawn on the map" },
    revision: 1,
    boundaryRevision: 1,
    intersectsLand: true,
    distanceM: 0,
    createdAt: "2026-10-01T00:00:00Z",
    updatedAt: "2026-10-01T00:00:00Z",
    ...extra,
  };
}

function inspection(featureId: string, condition: Inspection["condition"], observedAt: string) {
  return {
    id: `inspection-${featureId}`,
    featureId,
    featureRevision: 1,
    condition,
    notes: `${condition} on the day`,
    observedAt,
    recordedBy: "pilot",
    createdAt: observedAt,
  } satisfies Inspection;
}

const well = asset("a1", "North well");
const fence = asset("a2", "East fence", {
  category: "other",
  geometry: {
    type: "LineString",
    coordinates: [
      [0.001, 0.001],
      [0.009, 0.001],
    ],
  },
});
const barn = asset("a3", "Barn", {
  category: "building",
  status: "candidate",
  geometry: {
    type: "Polygon",
    coordinates: [
      [
        [0.002, 0.006],
        [0.003, 0.006],
        [0.003, 0.007],
        [0.002, 0.006],
      ],
    ],
  },
});
const shed = asset("a4", "Old shed", { status: "retired", category: "building" });
const inspections: Record<string, Inspection[]> = {
  a1: [inspection("a1", "good", "2026-10-05T10:00:00Z")],
  a2: [inspection("a2", "poor", "2026-10-03T10:00:00Z")],
  a3: [],
  a4: [],
};

interface Init {
  params?: { path?: Record<string, string>; query?: Record<string, unknown> };
}
function mockApi(list: Asset[], extra: (path: string, init: Init) => unknown = () => undefined) {
  return vi.spyOn(api, "GET").mockImplementation(((path: string, init: Init) => {
    const special = extra(path, init);
    const id = init.params?.path?.feature_id ?? "";
    const data =
      special !== undefined
        ? special
        : path.endsWith("/features")
          ? list
          : path.endsWith("/inspections")
            ? (inspections[id] ?? []).slice(0, Number(init.params?.query?.limit ?? 50))
            : path.endsWith("/revisions")
              ? [{ revision: 1, note: "Created", createdAt: "2026-10-01", content: {} }]
              : path.endsWith("/{feature_id}")
                ? list.find((item) => item.id === id)
                : [];
    return Promise.resolve({ data, response: new Response() });
  }) as never);
}

/** The land's fog and mission ground: computed from its assets and their inspections. */
const DERIVED = [
  ["land-knowledge", "pilot", land.id, 20],
  ["land-ground", "pilot", land.id],
];

function show(client = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
  return render(
    <QueryClientProvider client={client}>
      <AssetsLens land={land} />
    </QueryClientProvider>,
  );
}

/** A query client holding the land's derived answers, to see which a save asks again. */
function derivedClient() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  for (const key of DERIVED) client.setQueryData(key, { answer: true });
  const stale = () => DERIVED.map((key) => client.getQueryState(key)?.isInvalidated);
  return { client, stale };
}

beforeEach(() => {
  role.canEdit = true;
  useUi.setState({ activePanel: "land" });
  useLens.setState({ lens: "assets" });
  scene.areas.pickGround.mockReset();
  scene.areas.cancelPick.mockReset();
});
afterEach(() => {
  cleanup();
  localStorage.clear();
  useLandContext.getState().clear();
  useLandMap.setState({ pins: {}, focus: null });
  vi.restoreAllMocks();
});

describe("summary", () => {
  it("leads with the count and what needs attention, from the newest inspections", () => {
    const latest = { a1: inspections.a1![0]!, a2: inspections.a2![0]!, a3: null, a4: null };
    const summary = summarizeAssets([well, fence, barn, shed], latest);
    expect(summary.count).toBe(3);
    expect(summary.attention.map((item) => item.asset.name)).toEqual(["East fence"]);
    expect(summary.last?.asset.name).toBe("North well");
    expect(summary.unconfirmed).toBe(1);
    // Ties keep the fixed category order.
    expect(summary.categories.map((item) => item.category)).toEqual(["building", "water", "other"]);
    expect(assetsHeadline(summary)).toBe("3 assets: 1 needs attention.");
    expect(assetsHeadline(summarizeAssets([well], { a1: null }))).toBe(
      "1 asset, none inspected yet.",
    );
    expect(assetsHeadline(summarizeAssets([well], { a1: latest.a1 }))).toBe(
      "1 asset, none needing attention.",
    );
    expect(assetsHeadline(summarizeAssets([well, fence], {}, true))).toBe("2+ assets.");
  });

  it("names new assets without repeating, shortens pin labels and closes drawn areas", () => {
    expect(nextName("Well", ["well", "Well 2"])).toBe("Well 3");
    expect(nextName("Gate", ["Well"])).toBe("Gate");
    expect(shortLabel("The long fence along the northern creek")).toBe("The long fence along…");
    expect(shapeOf("LineString", [[1, 1]])).toBeNull();
    expect(
      shapeOf("Polygon", [
        [0, 0],
        [1, 0],
        [1, 1],
      ]),
    ).toEqual({
      type: "Polygon",
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 0],
        ],
      ],
    });
  });
});

describe("the card", () => {
  it("shows the summary, draws every asset in its kind's colour and pins the ones needing attention", async () => {
    mockApi([well, fence, barn, shed]);
    show();
    await screen.findByRole("heading", { name: "3 assets: 1 needs attention." });
    const layer = useLandContext.getState().layers.inventory;
    expect(layer?.features.map((feature) => feature.id)).toEqual(["a1", "a2", "a3"]);
    expect(new Set(layer?.features.map((feature) => feature.color)).size).toBe(3);
    expect(screen.getByText(/East fence \(poor\)/)).toBeInTheDocument();
    expect(screen.getByText(/North well, good/)).toBeInTheDocument();
    await waitFor(() => expect(useLandMap.getState().pins.assets).toHaveLength(1));
    expect(useLandMap.getState().pins.assets?.[0]).toMatchObject({
      label: "East fence · poor",
      tone: "warn",
    });
    // Colour by condition recolours the map without losing the features.
    fireEvent.click(screen.getByRole("button", { name: "Condition" }));
    expect(useLandContext.getState().layers.inventory?.features[1]?.color).toBe("#f0964a");
    // Retired assets are listed only on request.
    const list = screen.getByRole("list", { name: "Assets" });
    expect(within(list).queryByText("Old shed")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Retired/ }));
    expect(within(list).getByText("Old shed")).toBeInTheDocument();
  });

  it("selects from the list or the map, highlights it and records an inspection", async () => {
    mockApi([well, fence, barn]);
    const post = vi.spyOn(api, "POST").mockResolvedValue({
      data: inspection("a1", "critical", "2026-10-10T10:00:00Z"),
      response: new Response(),
    });
    const derived = derivedClient();
    show(derived.client);
    await screen.findByRole("heading", { name: "3 assets: 1 needs attention." });
    fireEvent.click(screen.getByRole("button", { name: /North well/ }));
    await screen.findByRole("heading", { name: "North well" });
    expect(useLandContext.getState().selectedInventoryId).toBe("a1");
    expect(useLandContext.getState().layers.inventory?.selectedIds).toEqual(["a1"]);
    expect(scene.camera.flyToRectangle).toHaveBeenCalled();
    expect(await screen.findByText(/Good, /)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Record an inspection" }));
    const save = screen.getByRole("button", { name: "Save inspection" });
    expect(save).toBeDisabled();
    fireEvent.click(screen.getByRole("radio", { name: "Critical" }));
    fireEvent.change(screen.getByLabelText("What did you see?"), {
      target: { value: "Pump housing cracked" },
    });
    expect(derived.stale()).toEqual([false, false]);
    fireEvent.click(save);
    await screen.findByText("Inspection saved.");
    // A dated inspection is evidence: the fog and the ground are asked again.
    expect(derived.stale()).toEqual([true, true]);
    const calls = post.mock.calls as unknown as [string, { body: Record<string, unknown> }][];
    expect(calls[0]?.[0]).toContain("/inspections");
    expect(calls[0]?.[1].body).toMatchObject({
      condition: "critical",
      notes: "Pump housing cracked",
    });
    expect(String(calls[0]?.[1].body.requestKey)).toMatch(/^[a-f0-9-]{36}$/);

    // A click on the map selects through the shared store.
    act(() => useLandContext.getState().selectInventory("a2"));
    await screen.findByRole("heading", { name: "East fence" });
    fireEvent.click(screen.getByRole("button", { name: "All assets" }));
    expect(useLandContext.getState().selectedInventoryId).toBeNull();
  });

  it("adds a point asset: choose what it is, click the map, name it, save", async () => {
    mockApi([well, asset("a5", "Well")]);
    scene.areas.pickGround
      .mockResolvedValueOnce({ longitude: 0.004, latitude: 0.006, x: 0.5, y: 0.5 })
      .mockImplementation(() => new Promise(() => undefined));
    const post = vi.spyOn(api, "POST").mockImplementation(((
      path: string,
      init: { body: unknown },
    ) =>
      Promise.resolve({
        data: path.endsWith("/geometry/preview")
          ? {
              geometry: (init.body as { geometry: unknown }).geometry,
              intersectsLand: true,
              distanceM: 0,
              areaM2: null,
              lengthM: null,
              perimeterM: null,
            }
          : asset("new", "Well 2"),
        response: new Response(),
      })) as never);
    const derived = derivedClient();
    show(derived.client);
    await screen.findByRole("heading", { name: "2 assets, none needing attention." });
    fireEvent.click(screen.getByRole("button", { name: "Add an asset" }));
    fireEvent.click(screen.getByRole("button", { name: "Well" }));
    await screen.findByRole("heading", { name: "New asset" });
    expect(useLandContext.getState().pointPicker).toBeNull();
    expect(screen.getByLabelText("Name")).toHaveValue("Well 2");
    expect(screen.getByRole("radio", { name: "Water" })).toBeChecked();
    expect(screen.getByText(/Point · On your land/)).toBeInTheDocument();
    expect(useLandContext.getState().layers["assets-draft"]?.features).toHaveLength(1);
    const key = `living-world-land-draft:${encodeURIComponent(landScope())}:inventory:${land.id}`;
    await waitFor(() => expect(localStorage.getItem(key)).toContain("Well 2"));
    expect(derived.stale()).toEqual([false, false]);
    fireEvent.click(screen.getByRole("button", { name: "Add asset" }));
    await waitFor(() => expect(useLandContext.getState().selectedInventoryId).toBe("new"));
    expect(derived.stale()).toEqual([true, true]);
    const calls = post.mock.calls as unknown as [string, { body: Record<string, unknown> }][];
    expect(calls[1]?.[0]).toBe("/api/v1/land/{land_id}/features");
    expect(calls[1]?.[1].body).toMatchObject({
      name: "Well 2",
      category: "water",
      status: "candidate",
      geometry: { type: "Point", coordinates: [0.004, 0.006] },
      source: { method: "drawn" },
    });
    expect(localStorage.getItem(key)).toBeNull();
    expect(useLandContext.getState().layers["assets-draft"]).toBeUndefined();
  });

  it("draws a line point by point with undo, and Cancel lets go of the map", async () => {
    mockApi([well]);
    const resolvers: ((value: unknown) => void)[] = [];
    scene.areas.pickGround.mockImplementation(
      () => new Promise((resolve) => resolvers.push(resolve)),
    );
    show();
    await screen.findByRole("heading", { name: "1 asset, none needing attention." });
    fireEvent.click(screen.getByRole("button", { name: "Add an asset" }));
    fireEvent.click(screen.getByRole("button", { name: "Fence" }));
    await screen.findByRole("heading", { name: "Draw the fence" });
    expect(useLandContext.getState().pointPicker).toBe(`assets-draw:${land.id}`);
    for (const point of [
      [0.001, 0.001],
      [0.002, 0.001],
      [0.003, 0.001],
    ])
      await act(() => {
        resolvers.at(-1)?.({ longitude: point[0], latitude: point[1] });
        return Promise.resolve();
      });
    expect(screen.getByText(/3 points · /)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(screen.getByText(/2 points · /)).toBeInTheDocument();
    expect(useLandContext.getState().layers["assets-drawing"]?.features).toHaveLength(3);
    expect(screen.getByRole("button", { name: "Done" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(scene.areas.cancelPick).toHaveBeenCalled();
    expect(useLandContext.getState().pointPicker).toBeNull();
    await screen.findByRole("heading", { name: "1 asset, none needing attention." });
    expect(useLandContext.getState().layers["assets-drawing"]).toBeUndefined();
  });

  it("offers one step when empty, and nothing to change for viewers", async () => {
    mockApi([]);
    const view = show();
    await screen.findByRole("heading", { name: "No assets mapped yet." });
    expect(screen.getByRole("button", { name: "Add an asset" })).toBeInTheDocument();
    view.unmount();
    role.canEdit = false;
    mockApi([well]);
    show();
    await screen.findByRole("heading", { name: /1 asset/ });
    expect(screen.queryByRole("button", { name: "Add an asset" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /North well/ }));
    await screen.findByRole("heading", { name: "North well" });
    expect(screen.queryByRole("button", { name: "Record an inspection" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
  });

  it("removes its layer and pins when the lens closes", async () => {
    mockApi([well, fence]);
    const view = show();
    await screen.findByRole("heading", { name: /2 assets/ });
    await waitFor(() => expect(useLandMap.getState().pins.assets).toHaveLength(1));
    view.unmount();
    expect(useLandContext.getState().layers.inventory).toBeUndefined();
    expect(useLandMap.getState().pins.assets).toBeUndefined();
  });
});

describe("drafts", () => {
  const draft = {
    requestKey: "22222222-2222-4222-8222-222222222222",
    name: "Pole",
    category: "other" as const,
    geometry: { type: "Point" as const, coordinates: [0.001, 0.001] },
    source: { method: "drawn" as const, label: "Synthetic fixture" },
  };
  const record = asset("33333333-3333-4333-8333-333333333333", "Pole", {
    ...draft,
    source: {
      ...draft.source,
      url: null,
      recordId: null,
      observedAt: null,
      attribution: null,
      meaning: "study-area",
    },
  });
  const snapshot = {
    version: 1,
    landId: land.id,
    boundaryRevision: 1,
    draft,
    editing: null,
    revisionNote: "",
    working: null,
  };
  const key = () =>
    `living-world-land-draft:${encodeURIComponent(landScope())}:inventory:${land.id}`;

  it("recognises a creation already saved after a lost response, without saving again", async () => {
    localStorage.setItem(key(), JSON.stringify(snapshot));
    mockApi([], (path) =>
      path.includes("/requests/")
        ? {
            current: { ...record, revision: 2, name: "Later name", requestKey: "x" },
            original: { ...draft, source: record.source },
          }
        : path.endsWith("/{feature_id}")
          ? record
          : undefined,
    );
    const post = vi.spyOn(api, "POST");
    show();
    fireEvent.click(await screen.findByRole("button", { name: "Recover asset draft" }));
    await screen.findByText(/already saved/);
    expect(localStorage.getItem(key())).toBeNull();
    expect(useLandContext.getState().selectedInventoryId).toBe(record.id);
    expect(post).not.toHaveBeenCalled();
  });

  it("blocks saving a recovered change until a newer saved version is reconciled", async () => {
    const mine = { ...draft, requestKey: "44444444-4444-4444-8444-444444444444", name: "My name" };
    const current = { ...record, revision: 2, name: "Another name", category: "power" as const };
    localStorage.setItem(
      key(),
      JSON.stringify({ ...snapshot, draft: mine, editing: record, revisionNote: "Renamed" }),
    );
    mockApi([], (path) => (path.endsWith("/{feature_id}") ? current : undefined));
    const put = vi.spyOn(api, "PUT").mockResolvedValue({
      data: { ...current, ...mine, revision: 3, category: "power" },
      response: new Response(),
    } as never);
    show();
    fireEvent.click(await screen.findByRole("button", { name: "Recover asset draft" }));
    await screen.findByRole("region", { name: "Newer saved version" });
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
    fireEvent.click(screen.getByRole("radio", { name: "Keep my name" }));
    fireEvent.click(screen.getByRole("button", { name: "Use these choices" }));
    expect(screen.getByRole("radio", { name: "Power" })).toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(put).toHaveBeenCalledOnce());
    const calls = put.mock.calls as unknown as [string, { body: unknown }][];
    expect(calls[0]?.[1].body).toMatchObject({
      expectedRevision: 2,
      name: "My name",
      category: "power",
      note: "Renamed",
    });
    await waitFor(() => expect(localStorage.getItem(key())).toBeNull());
  });
});

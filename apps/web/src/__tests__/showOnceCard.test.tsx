/**
 * The Show once, everywhere card: it prepares the land's 10 m signatures when there are none
 * for this boundary, starts a session on the freshest ones, shows the tapped cell, the slider
 * and its tally with the honest caveat, and hands the lit ground to the job maker.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { LandArea, ResearchStatus, SignatureSimilarity, components } from "@twin/contracts";

import { api } from "@/api/client";
import { signaturePeriods } from "@/features/land/rasterRuns";
import { ShowOnceCard } from "@/features/showOnce/ShowOnceCard";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useShowOnce } from "@/state/showOnce";
import { useUi } from "@/state/ui";

const makeJob = vi.hoisted(() =>
  vi.fn((..._args: unknown[]): Promise<unknown> => Promise.resolve(null)),
);
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => null }));
vi.mock("@/features/showOnce/makeJob", () => ({
  makeJob: (...args: unknown[]): Promise<unknown> => makeJob(...args),
}));

type LandRaster = components["schemas"]["LandRasterRead"];

const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Oak Bench Ranch",
  description: "",
  revision: 2,
  areaM2: 5_460_000,
  perimeterM: 9_000,
  createdAt: "2026-10-16",
  updatedAt: "2026-10-16",
  source: { method: "drawn", label: "Fixture" },
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-120.05, 34.69],
        [-120.03, 34.69],
        [-120.03, 34.71],
        [-120.05, 34.69],
      ],
    ],
  },
};

const status: ResearchStatus = {
  modelConfigured: true,
  model: "fixture",
  workerRequired: true,
  worker: {
    state: "online",
    lastSeenAt: "2026-10-16T17:46:00Z",
    secondsSinceSeen: 3,
    busy: false,
    checkInEverySeconds: 15,
  },
  queue: { queued: 0, oldestQueuedAt: null, oldestQueuedSeconds: null, running: 0 },
};

function signatureRaster(revision = 2): LandRaster {
  return {
    id: "raster-1",
    runId: "run-0",
    landId: land.id,
    boundaryRevision: revision,
    request: { dataset: "sentinel-2-signature", resolutionM: 10, maxDimension: 512, periods: [] },
    metadata: {
      boundaryCells: 54_596,
      vegetation: {
        observations: [
          { acquiredAt: "2025-01-15T18:54:46Z" },
          { acquiredAt: "2025-04-05T18:55:11Z" },
          { acquiredAt: "2025-07-24T18:55:12Z" },
          { acquiredAt: "2025-10-17T18:54:50Z" },
        ],
      },
      signature: { commonCells: 54_436 },
    },
    stale: revision !== land.revision,
  } as unknown as LandRaster;
}

let rasters: LandRaster[] = [];

function mount() {
  vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/rasters")
        ? rasters
        : path.endsWith("/status")
          ? status
          : {
              investigation: { id: "inv-1" },
              runs: [{ id: "run-1", status: "running", question: "Prepare" }],
              messages: [],
              findings: [],
              evidence: [],
              artifacts: [],
              page: { offset: 0, limit: 100, totals: {} },
            },
      response: new Response(),
    })) as never);
  const post = vi.spyOn(api, "POST").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/investigations")
        ? { id: "inv-1", landId: land.id }
        : { id: "run-1", status: "queued" },
      response: new Response(),
    })) as never);
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <ShowOnceCard land={land} />
    </QueryClientProvider>,
  );
  return post;
}

beforeEach(() => {
  useLand.getState().select(land);
  useLandContext.getState().setSection("ecology");
  useUi.getState().setPanel("land");
  useShowOnce.getState().stop();
  rasters = [];
  makeJob.mockReset();
});

afterEach(() => {
  cleanup();
  useShowOnce.getState().stop();
  useLandContext.getState().clear();
  useUi.getState().setPanel(null);
  useLand.getState().clear();
  vi.restoreAllMocks();
});

it("prepares four seasons of 10 m signatures when the land has none", async () => {
  const post = mount();
  const button = await screen.findByRole("button", { name: /Prepare 10 m signatures/ });
  expect(screen.getByTestId("show-once-prepare").textContent).toContain(
    "This land has no 10 m signatures yet",
  );
  fireEvent.click(button);
  const runs = () =>
    (post.mock.calls as unknown as [string, { body: Record<string, unknown> }][]).filter(([path]) =>
      path.endsWith("/runs"),
    );
  await waitFor(() => expect(runs()).toHaveLength(1));
  expect(runs()[0]?.[1].body).toMatchObject({
    kind: "raster",
    analysis: {
      dataset: "sentinel-2-signature",
      resolutionM: 10,
      maxDimension: 512,
      periods: signaturePeriods(),
    },
  });
  expect(signaturePeriods()).toHaveLength(4);
  await screen.findByText("Reading four seasonal Sentinel-2 scenes for 10 m signatures…");
});

it("says the signatures are for an earlier boundary", async () => {
  rasters = [signatureRaster(1)];
  mount();
  await screen.findByText(/made for an earlier boundary/);
  expect(screen.queryByTestId("show-once-start")).toBeNull();
});

it("starts on the freshest signatures and shows the tapped cell, the slider and its tally", async () => {
  rasters = [signatureRaster()];
  mount();
  const start = await screen.findByTestId("show-once-start");
  expect(start.textContent).toContain("Click a place");
  expect(screen.getByTestId("show-once-scenes").textContent).toContain(
    "Scenes Jan 15, Apr 5, Jul 24, Oct 17 2025",
  );
  expect(screen.getByTestId("show-once-caveat").textContent).toContain(
    "Similar seasonal satellite signature at 10 m. Not a species ID.",
  );
  fireEvent.click(start);
  expect(useShowOnce.getState().session).toEqual({
    landId: land.id,
    revision: 2,
    rasterId: "raster-1",
  });
  expect(useShowOnce.getState().shown).toBe(true);
  await screen.findByText(/Click a place inside Oak Bench Ranch to compare it/);

  const read = {
    seed: {
      longitude: -120.04,
      latitude: 34.7,
      cellLongitude: -120.04121,
      cellLatitude: 34.68931,
      windows: [
        { window: 0, acquiredAt: "2025-01-15T18:54:46Z", ndvi: 0.69 },
        { window: 1, acquiredAt: "2025-04-05T18:55:11Z", ndvi: 0.64 },
      ],
    },
    scenes: [],
    comparedCells: 3,
  } as unknown as SignatureSimilarity;
  act(() =>
    useShowOnce.getState().setResult({
      read,
      grid: {
        west: -120.05,
        south: 34.69,
        east: -120.03,
        north: 34.7,
        columns: 2,
        rows: 2,
        cellM: 10,
        cellAreaM2: 4046.8564224,
        scores: Uint8Array.from([250, 210, 255, 120]),
      },
    }),
  );
  expect(screen.getByText("Like the cell at 34.68931, -120.04121")).toBeTruthy();
  expect(screen.getByTestId("show-once-seed").textContent).toContain("Jan 15 0.69");
  expect(screen.getByTestId("show-once-tally").textContent).toContain("2 cells · ≈2.0 ac");
  // The slider is the store's threshold: one step up, and a cell drops out of the tally.
  fireEvent.keyDown(screen.getByRole("slider", { name: "How alike" }), { key: "ArrowRight" });
  expect(useShowOnce.getState().threshold).toBeCloseTo(0.81, 5);
  act(() => useShowOnce.getState().setThreshold(0.9));
  expect(screen.getByTestId("show-once-tally").textContent).toContain("1 cell · ≈1.0 ac");
  expect(screen.getByTestId("show-once-threshold").textContent).toBe("90%");

  act(() => useShowOnce.getState().setGuarded(Uint8Array.from([1, 0, 0, 0])));
  expect(screen.getByTestId("show-once-protected").textContent).toContain(
    "Protected, not part of this job: 1 cell ·",
  );
  fireEvent.click(screen.getByRole("button", { name: /Free all/ }));
  expect(useShowOnce.getState().guarded).toBeNull();

  makeJob.mockResolvedValue({
    areas: 3,
    acres: 9.96,
    protectedAcres: 0,
    dropped: { patches: 0, acres: 0 },
  });
  fireEvent.click(screen.getByTestId("show-once-make-job"));
  await screen.findByTestId("show-once-done");
  expect(screen.getByTestId("show-once-done").textContent).toContain(
    "Drafted as a job: 3 areas · 10.0 ac on Oak Bench Ranch",
  );
  expect(makeJob).toHaveBeenCalledWith(land, null);
  expect(useShowOnce.getState().session).toBeNull();
  expect(screen.getByTestId("show-once-start").textContent).toContain("Click another place");
});

it("says when nothing at the setting is big enough to be a job", async () => {
  rasters = [signatureRaster()];
  mount();
  fireEvent.click(await screen.findByTestId("show-once-start"));
  act(() =>
    useShowOnce.getState().setResult({
      read: {
        seed: { cellLongitude: 0, cellLatitude: 0, windows: [] },
        scenes: [],
        comparedCells: 1,
      } as unknown as SignatureSimilarity,
      grid: {
        west: 0,
        south: 0,
        east: 1,
        north: 1,
        columns: 1,
        rows: 1,
        cellM: 10,
        cellAreaM2: 100,
        scores: Uint8Array.from([250]),
      },
    }),
  );
  makeJob.mockResolvedValue({
    areas: 0,
    acres: 0,
    protectedAcres: 0,
    dropped: { patches: 1, acres: 0.02 },
  });
  fireEvent.click(screen.getByTestId("show-once-make-job"));
  await screen.findByText(/Nothing at this setting makes an area of half an acre or more/);
  expect(useShowOnce.getState().session).not.toBeNull();
});

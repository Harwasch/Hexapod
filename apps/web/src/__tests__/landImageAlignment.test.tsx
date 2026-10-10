import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { components } from "@twin/contracts";
import { api } from "@/api/client";
import { ArchiveMapAlignment } from "@/features/land/ArchiveMapAlignment";
import { useLandContext } from "@/state/landContext";
import { useUi } from "@/state/ui";

type Request = components["schemas"]["ImageRegistrationCreate"];
type Registration = components["schemas"]["ImageRegistrationRead"];
const scene = vi.hoisted(() => ({
  areas: { pickGround: vi.fn(), cancelPick: vi.fn() },
  camera: { flyToRectangle: vi.fn() },
}));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));
const image: components["schemas"]["ArchiveImageRead"] = {
  evidenceId: "map",
  width: 600,
  height: 400,
  sha256: "a".repeat(64),
  sourceSha256: "b".repeat(64),
  sourceUrl: "https://prd-tnm.s3.amazonaws.com/fixture",
  sourceMediaType: "image/png",
  byteSize: 100,
  sourceByteSize: 100,
  createdAt: "2026-10-09T12:00:00Z",
};
const fit: components["schemas"]["ImageRegistrationResult"] = {
  algorithm: "affine-aeqd-v1",
  crs: "local projection",
  transform: [2, 0, 0, 0, -2, 0],
  imageWidth: 600,
  imageHeight: 400,
  footprint: {
    type: "Polygon",
    coordinates: [
      [
        [-77.05, 38.88],
        [-77.04, 38.88],
        [-77.04, 38.89],
        [-77.05, 38.88],
      ],
    ],
  },
  bounds: [-77.05, 38.88, -77.04, 38.89],
  rmsErrorM: 2,
  maximumErrorM: 3,
  leaveOneOutRmsM: 5,
  controlPointCoverage: 1,
  approximateMPerPixel: 2,
  points: [],
  warnings: ["Fit error does not establish surveyed accuracy."],
};
let rows: Registration[] = [];
let posts: { path: string; body: Request }[] = [];
function mount() {
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={cache}>
      <ArchiveMapAlignment
        id="map"
        image={image}
        url="blob:private-map"
        title="Historical sheet"
        attribution="USGS; public domain"
      />
    </QueryClientProvider>,
  );
}
function fill(label: string, value: string | number) {
  fireEvent.change(screen.getByLabelText(label, { exact: true }), {
    target: { value: String(value) },
  });
}
function add(x: number, y: number, lon: number, lat: number) {
  fill("Image X", x);
  fill("Image Y", y);
  fill("Longitude", lon);
  fill("Latitude", lat);
  fireEvent.click(screen.getByRole("button", { name: "Add point match" }));
}
function fourPoints() {
  fireEvent.click(screen.getByText("Enter map coordinates instead"));
  add(0, 0, -77.055, 38.891);
  add(600, 0, -77.045, 38.891);
  add(600, 400, -77.045, 38.886);
  add(0, 400, -77.055, 38.886);
}
beforeEach(() => {
  rows = [];
  posts = [];
  localStorage.clear();
  useUi.getState().setPanel("land");
  vi.spyOn(api, "GET").mockImplementation((() =>
    Promise.resolve({ data: rows, response: new Response() })) as typeof api.GET);
  vi.spyOn(api, "POST").mockImplementation(((path: string, options: { body: Request }) => {
    posts.push({ path, body: options.body });
    if (path.endsWith("/preview")) return Promise.resolve({ data: fit, response: new Response() });
    const row: Registration = {
      id: "alignment",
      evidenceId: "map",
      request: options.body,
      result: fit,
      sha256: "c".repeat(64),
      byteSize: 100,
      displayWidth: 600,
      displayHeight: 400,
      createdAt: "2026-10-09T13:00:00Z",
    };
    rows = [row];
    return Promise.resolve({ data: row, response: new Response() });
  }) as typeof api.POST);
});
afterEach(() => {
  cleanup();
  useLandContext.getState().clear();
  localStorage.clear();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});
it("matches points, checks fit, saves a private overlay, and invalidates the fit after changes", async () => {
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Align this map" }));
  expect(screen.getByRole("button", { name: "Check alignment" })).toBeDisabled();
  fourPoints();
  fireEvent.click(screen.getByRole("button", { name: "Check alignment" }));
  await screen.findByText("Leave-one-out RMS");
  fireEvent.click(screen.getByRole("button", { name: "Save alignment and show overlay" }));
  await screen.findByRole("article", { name: "Saved alignment: Historical sheet" });
  expect(posts[0]?.body.requestKey).toBe(posts[1]?.body.requestKey);
  expect(posts[1]?.body.imageSha256).toBe(image.sha256);
  expect(useLandContext.getState().rasters.alignment).toMatchObject({
    kind: "archive-alignment",
    opacity: 0.6,
    bounds: fit.bounds,
  });
  fireEvent.change(screen.getByRole("slider", { name: "Aligned map opacity" }), {
    target: { value: "0.3" },
  });
  expect(useLandContext.getState().rasters.alignment?.opacity).toBe(0.3);
  fireEvent.click(screen.getByRole("button", { name: "Hide aligned map" }));
  expect(useLandContext.getState().rasters.alignment).toBeUndefined();
  fireEvent.click(screen.getByRole("button", { name: "Refine a copy" }));
  expect(screen.getByRole("button", { name: "Save alignment and show overlay" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Check alignment" }));
  await waitFor(() => expect(posts).toHaveLength(3));
  expect(posts[2]?.body.requestKey).not.toBe(posts[1]?.body.requestKey);
});
it("recovers unfinished matches on remount and rejects invalid or duplicate coordinates", async () => {
  const view = mount();
  fireEvent.click(screen.getByRole("button", { name: "Align this map" }));
  fourPoints();
  add(0, 0, -77, 38);
  expect(screen.getByRole("alert")).toHaveTextContent("distinct");
  fill("Latitude", 95);
  expect(screen.getByRole("button", { name: "Add point match" })).toBeDisabled();
  view.unmount();
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Align this map · 4 draft matches" }));
  expect(screen.getByRole("list", { name: "Point matches" }).children).toHaveLength(4);
  expect(screen.getByRole("button", { name: "Check alignment" })).toBeEnabled();
  await waitFor(() => expect(api.GET).toHaveBeenCalled());
});
it("pairs a ground pick and cancels pending selection when the research section changes", async () => {
  let resolve!: (point: { longitude: number; latitude: number } | null) => void;
  scene.areas.pickGround.mockImplementation(
    () =>
      new Promise<{ longitude: number; latitude: number } | null>((done) => {
        resolve = done;
      }),
  );
  scene.areas.cancelPick.mockImplementation(() => resolve(null));
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Align this map" }));
  fill("Image X", 100);
  fill("Image Y", 150);
  fireEvent.click(screen.getByRole("button", { name: "Pick matching map point" }));
  expect(useLandContext.getState().pointPicker).toBe("map");
  await act(async () => {
    resolve({ longitude: -77.05, latitude: 38.89 });
    await Promise.resolve();
  });
  expect(screen.getByRole("list", { name: "Point matches" }).children).toHaveLength(1);
  expect(useLandContext.getState().pointPicker).toBeNull();
  fill("Image X", 200);
  fill("Image Y", 250);
  fireEvent.click(screen.getByRole("button", { name: "Pick matching map point" }));
  await act(async () => {
    useLandContext.getState().setSection("records");
    await Promise.resolve();
  });
  expect(scene.areas.cancelPick).toHaveBeenCalledOnce();
  expect(useLandContext.getState().pointPicker).toBeNull();
  expect(useLandContext.getState().layers["alignment-controls:map"]).toBeUndefined();
});
it("ignores a late preview after the user changes a point", async () => {
  let resolve!: (response: { data: typeof fit; response: Response }) => void;
  vi.spyOn(api, "POST").mockImplementation(
    (() =>
      new Promise<{ data: typeof fit; response: Response }>((done) => {
        resolve = done;
      })) as typeof api.POST,
  );
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Align this map" }));
  fourPoints();
  fireEvent.click(screen.getByRole("button", { name: "Check alignment" }));
  fireEvent.click(screen.getByRole("button", { name: "Remove Point 4" }));
  await act(async () => {
    resolve({ data: fit, response: new Response() });
    await Promise.resolve();
  });
  expect(screen.queryByText("Leave-one-out RMS")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Save alignment and show overlay" })).toBeDisabled();
});

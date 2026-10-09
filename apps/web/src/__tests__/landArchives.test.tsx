import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { components, LandEvidence } from "@twin/contracts";
import { api } from "@/api/client";
import { ArchiveGallery } from "@/features/land/ArchiveGallery";
import { useLandContext } from "@/state/landContext";
import { useLand } from "@/state/land";

const flyToRectangle = vi.hoisted(() => vi.fn());
vi.mock("@/cesium/SceneContext", () => ({
  useScene: () => ({ camera: { flyToRectangle } }),
}));
const photo: LandEvidence = {
  id: "photo",
  runId: "run",
  provider: "commons-place-images",
  title: "A historical view",
  attribution: "A. Photographer; Wikimedia Commons; CC BY-SA 4.0",
  excerpt: "Catalog metadata",
  license: "CC BY-SA 4.0",
  spatialRelevance: "within",
  relevanceNote: "Catalog coordinate only",
  retrievedAt: "2026-10-09T12:00:00Z",
  media: {
    kind: "photograph",
    title: "A historical view",
    description: "Creation date is uncertain.",
    sourceUrl: "https://commons.wikimedia.org/wiki/File:Test.jpg",
    previewUrl: "https://upload.wikimedia.org/wikipedia/commons/a/a1/Test.jpg",
    creator: "A. Photographer",
    license: "CC BY-SA 4.0",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
    sourceDate: "circa 1890–1895",
    dateMeaning: "Upload dates are not event dates.",
    location: { type: "Point", coordinates: [-77.05, 38.888] },
    locationMeaning: "catalog-coordinate",
    relevance: "The coordinate may locate the camera.",
    sourceVersion: "Commons file SHA-1 example",
  },
};
const sheet: LandEvidence = {
  ...photo,
  id: "map",
  title: "Historical sheet",
  media: {
    ...photo.media!,
    kind: "historical-map",
    title: "Historical sheet",
    sourceDate: "1890",
    locationMeaning: "catalog-footprint",
    location: {
      type: "Polygon",
      coordinates: [
        [
          [-77.1, 38.8],
          [-77, 38.8],
          [-77, 38.9],
          [-77.1, 38.9],
          [-77.1, 38.8],
        ],
      ],
    },
  },
};
function mount(evidence: LandEvidence[] = [photo, sheet], onAsk = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(
    <QueryClientProvider client={client}>
      <ArchiveGallery ids={["photo", "map"]} evidence={evidence} onAsk={onAsk} />
    </QueryClientProvider>,
  );
  return onAsk;
}
beforeEach(() => {
  vi.spyOn(api, "GET").mockResolvedValue({ data: null, response: new Response() });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  flyToRectangle.mockClear();
  useLandContext.getState().clear();
});
it("uses saved metadata with date uncertainty and creator/license attribution", () => {
  const get = vi.spyOn(api, "GET");
  mount();
  expect(screen.getByText("circa 1890–1895")).toBeVisible();
  expect(screen.getByRole("link", { name: "CC BY-SA 4.0" })).toHaveAttribute(
    "href",
    photo.media!.licenseUrl,
  );
  fireEvent.click(screen.getByText("Dates, attribution and interpretation"));
  expect(screen.getByText("Upload dates are not event dates.")).toBeVisible();
  expect(screen.getByText(/public archive and may change/)).toBeVisible();
  expect(get).not.toHaveBeenCalledWith(
    "/api/v1/research/evidence/{evidence_id}",
    expect.anything(),
  );
});
it("navigates sources and keeps attribution and source links when a preview fails", () => {
  mount();
  fireEvent.error(screen.getByRole("img"));
  expect(screen.getByText(/remote preview is unavailable/)).toBeVisible();
  expect(screen.getByRole("link", { name: "Open original source" })).toHaveAttribute(
    "href",
    photo.media!.sourceUrl,
  );
  fireEvent.click(screen.getByRole("button", { name: "Next source" }));
  expect(screen.getByRole("img", { name: "Historical sheet" })).toBeVisible();
  expect(screen.getByText("2 of 2")).toBeVisible();
  expect(screen.getByRole("button", { name: "Next source" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Previous source" }));
  expect(screen.getByText("1 of 2")).toBeVisible();
});
it("maps catalog footprints without changing the selected land", () => {
  mount();
  const original = useLand.getState();
  fireEvent.click(screen.getByRole("button", { name: "Next source" }));
  fireEvent.click(screen.getByRole("button", { name: "Show sheet footprint" }));
  expect(useLandContext.getState().layers["archive:map"]?.features[0]?.geometry).toEqual(
    sheet.media!.location,
  );
  expect(flyToRectangle).toHaveBeenCalledWith(-77.1, 38.8, -77, 38.9);
  expect(useLand.getState()).toBe(original);
  fireEvent.click(screen.getByRole("button", { name: "Hide catalog location" }));
  expect(useLandContext.getState().layers).toEqual({});
});
it("preserves the question draft and opens research with the saved evidence reference", () => {
  useLandContext.getState().setResearchQuestion("My existing question");
  const onAsk = mount();
  fireEvent.click(screen.getByRole("button", { name: "Ask about this source" }));
  expect(useLandContext.getState().researchQuestion).toMatch(
    /^My existing question\n\nInvestigate/,
  );
  expect(useLandContext.getState().researchQuestion).toContain("(evidence photo)");
  expect(onAsk).toHaveBeenCalledOnce();
});
it("loads a source through the private evidence API when it is outside the current results page", async () => {
  const get = vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/image") ? null : photo,
      response: new Response(),
    })) as typeof api.GET);
  mount([]);
  expect(await screen.findByText("circa 1890–1895")).toBeVisible();
  expect(get).toHaveBeenCalledWith("/api/v1/research/evidence/{evidence_id}", {
    params: { path: { evidence_id: "photo" } },
  });
});

const savedImage: components["schemas"]["ArchiveImageRead"] = {
  evidenceId: "photo",
  createdAt: "2026-10-09T12:00:00Z",
  width: 640,
  height: 480,
  sha256: "a".repeat(64),
  sourceSha256: "b".repeat(64),
  byteSize: 100,
  sourceByteSize: 80,
  sourceMediaType: "image/jpeg",
  sourceUrl: photo.media!.previewUrl,
};
it("saves a private image snapshot and restores its preview and provenance", async () => {
  const createUrl = vi.fn(() => "blob:saved-preview"),
    revoke = vi.fn();
  const previousCreate = Object.getOwnPropertyDescriptor(URL, "createObjectURL"),
    previousRevoke = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");
  URL.createObjectURL = createUrl;
  URL.revokeObjectURL = revoke;
  try {
    vi.spyOn(api, "GET").mockImplementation(((path: string) =>
      Promise.resolve({
        data: path.endsWith("/preview") ? new Blob(["preview"], { type: "image/png" }) : null,
        response: new Response(),
      })) as typeof api.GET);
    const post = vi
      .spyOn(api, "POST")
      .mockResolvedValue({ data: savedImage, response: new Response() });
    mount();
    const save = screen.getByRole("button", { name: "Save image to workspace" });
    await waitFor(() => expect(save).toBeEnabled());
    fireEvent.click(save);
    await waitFor(() =>
      expect(screen.getByRole("img")).toHaveAttribute("src", "blob:saved-preview"),
    );
    expect(post).toHaveBeenCalledWith(
      "/api/v1/research/evidence/{evidence_id}/image",
      expect.objectContaining({
        params: { path: { evidence_id: "photo" } },
      }),
    );
    expect(screen.getByText(/640 × 480 pixels/)).toBeVisible();
    fireEvent.click(screen.getByText("Saved image provenance"));
    expect(screen.getByText(savedImage.sourceSha256)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Next source" }));
    expect(revoke).toHaveBeenCalledWith("blob:saved-preview");
    fireEvent.click(screen.getByRole("button", { name: "Previous source" }));
    // A persisted snapshot is the source for this gallery entry; the original remote URL is never overwritten.
    await waitFor(() => expect(createUrl).toHaveBeenCalledTimes(2));
    expect(photo.media!.previewUrl).toMatch(/^https:/);
  } finally {
    cleanup();
    if (previousCreate) Object.defineProperty(URL, "createObjectURL", previousCreate);
    else Reflect.deleteProperty(URL, "createObjectURL");
    if (previousRevoke) Object.defineProperty(URL, "revokeObjectURL", previousRevoke);
    else Reflect.deleteProperty(URL, "revokeObjectURL");
  }
});

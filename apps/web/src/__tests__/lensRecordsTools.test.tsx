import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { components } from "@twin/contracts";

import { api } from "@/api/client";
import { ArchiveGallery } from "@/features/land/ArchiveGallery";
import { LandDocumentViewer } from "@/features/land/LandDocumentViewer";
import { useAgentBus } from "@/features/land/surface/agentBus";
import { useLens } from "@/features/land/surface/lenses";
import { RecordsLens } from "@/features/land/surface/lenses/RecordsLens";
import { RecordsAlignment } from "@/features/land/surface/lenses/records/RecordsAlignment";
import { RecordsDocumentOcr } from "@/features/land/surface/lenses/records/RecordsDocumentOcr";
import { useRecords } from "@/features/land/surface/lenses/records/recordsStore";
import { useLandContext } from "@/state/landContext";
import { useLandMap } from "@/state/landMap";
import { useUi } from "@/state/ui";

import {
  archiveInvestigation,
  deed,
  detail,
  fit,
  image,
  land,
  photo,
  photoMedia,
  registration,
  sheet,
} from "./lensRecordsFixtures";

type Request = components["schemas"]["ImageRegistrationCreate"];
type Registration = components["schemas"]["ImageRegistrationRead"];
type Reading = components["schemas"]["DocumentOcrRead"];

const scene = vi.hoisted(() => ({
  areas: { pickGround: vi.fn(), cancelPick: vi.fn() },
  camera: { flyToRectangle: vi.fn() },
}));
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));

function wrap(child: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(<QueryClientProvider client={client}>{child}</QueryClientProvider>);
}

const createUrl = vi.fn(() => "blob:preview");
const revokeUrl = vi.fn();
const originalCreate = Object.getOwnPropertyDescriptor(URL, "createObjectURL");
const originalRevoke = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");

beforeEach(() => {
  localStorage.clear();
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
  URL.createObjectURL = createUrl;
  URL.revokeObjectURL = revokeUrl;
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.clearAllMocks();
  useLandContext.getState().clear();
  useLandMap.setState({ pins: {}, focus: null });
  localStorage.clear();
  if (originalCreate) Object.defineProperty(URL, "createObjectURL", originalCreate);
  else Reflect.deleteProperty(URL, "createObjectURL");
  if (originalRevoke) Object.defineProperty(URL, "revokeObjectURL", originalRevoke);
  else Reflect.deleteProperty(URL, "revokeObjectURL");
});

describe("documents", () => {
  function serve(pending = false) {
    const row = pending ? { ...deed, status: "awaiting-upload" as const, pageCount: 0 } : deed;
    return vi.spyOn(api, "GET").mockImplementation(((
      path: string,
      options?: { params?: { path?: { page?: number } } },
    ) => {
      const page = options?.params?.path?.page;
      const data =
        path === "/api/v1/land/{land_id}/investigations"
          ? [archiveInvestigation]
          : path === "/api/v1/research/investigations/{investigation_id}"
            ? detail([])
            : path === "/api/v1/land/{land_id}/documents"
              ? [row]
              : path === "/api/v1/land/{land_id}/documents/links"
                ? []
                : path === "/api/v1/land/{land_id}/documents/{document_id}/pages/{page}"
                  ? {
                      documentId: "deed",
                      page,
                      text: `Recorded passage on page ${page}`,
                      sha256: deed.sha256,
                      truncated: false,
                    }
                  : path === "/api/v1/land/{land_id}/documents/{document_id}"
                    ? row
                    : null;
      return Promise.resolve({ data, response: new Response() });
    }) as typeof api.GET);
  }

  it("retries an interrupted upload against the same document without creating another", async () => {
    serve();
    const post = vi.spyOn(api, "POST").mockResolvedValue({ data: deed, response: new Response() });
    const put = vi
      .spyOn(api, "PUT")
      .mockRejectedValueOnce(new Error("Connection lost"))
      .mockResolvedValue({ data: deed, response: new Response() });
    wrap(<RecordsLens land={land} />);
    fireEvent.click(await screen.findByRole("button", { name: "Add a document" }));
    fireEvent.change(screen.getByLabelText("Original PDF or text file"), {
      target: { files: [new File(["deed"], "deed.txt", { type: "text/plain" })] },
    });
    expect(screen.getByLabelText("Title")).toHaveValue("deed");
    fireEvent.change(screen.getByLabelText("Where did this record come from?"), {
      target: { value: "County archive" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save and read record" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("nothing will be doubled");
    fireEvent.click(screen.getByRole("button", { name: "Save and read record" }));
    await screen.findByRole("region", { name: "Document page" });
    expect(post).toHaveBeenCalledOnce();
    expect(put).toHaveBeenCalledTimes(2);
    const calls = put.mock.calls as unknown as [
      string,
      { params: { path: unknown }; bodySerializer: () => File },
    ][];
    expect(calls[0]?.[1].params.path).toEqual(calls[1]?.[1].params.path);
    expect(calls[1]?.[1].bodySerializer().name).toBe("deed.txt");
  });

  it("relates the viewed page to another record and asks the agent about it", async () => {
    serve();
    useRecords.setState({
      landId: land.id,
      view: { kind: "document", id: "deed", page: 1, jump: 0 },
    });
    wrap(<RecordsLens land={land} />);
    expect(await screen.findByText("Names this land's parcel")).toBeVisible();
    fireEvent.change(await screen.findByLabelText("Document page number"), {
      target: { value: "2" },
    });
    await screen.findByText("Recorded passage on page 2");
    fireEvent.click(screen.getByRole("button", { name: "Relate this record to another" }));
    expect(screen.getByLabelText("Source page")).toHaveValue(2);
    fireEvent.click(screen.getByRole("button", { name: "Ask about this page" }));
    const question = useAgentBus.getState().request?.question ?? "";
    expect(question).toContain("page 2 (document deed");
    expect(question).toContain("County deed");
  });

  it("finds a document in the details list and opens it", async () => {
    serve();
    wrap(<RecordsLens land={land} />);
    const list = await screen.findByRole("list", { name: "All documents" });
    fireEvent.click(await within(list).findByRole("button", { name: /County deed.*Text ready/ }));
    expect(await screen.findByRole("region", { name: "Document page" })).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Records" }));
    expect(await screen.findByRole("heading", { name: /county parcel record/ })).toBeVisible();
  });

  it("resumes a pending upload and checks the file before sending it", async () => {
    serve(true);
    const put = vi.spyOn(api, "PUT").mockResolvedValue({ data: deed, response: new Response() });
    wrap(<LandDocumentViewer landId={land.id} documentId="deed" />);
    const input = await screen.findByLabelText("Resume original upload");
    fireEvent.change(input, { target: { files: [new File(["wrong"], "other.txt")] } });
    expect(await screen.findByRole("alert")).toHaveTextContent("Choose the original deed.txt");
    expect(put).not.toHaveBeenCalled();
    fireEvent.change(input, { target: { files: [new File(["deed"], "deed.txt")] } });
    await waitFor(() => expect(put).toHaveBeenCalledOnce());
    await screen.findByText("Recorded passage on page 1");
    expect(screen.queryByLabelText("Resume original upload")).not.toBeInTheDocument();
  });
});

describe("machine reading", () => {
  const reading: Reading = {
    id: "ocr",
    documentId: "deed",
    page: 2,
    language: "eng",
    text: "Mineral rights reserved.",
    sha256: "a".repeat(64),
    textSha256: "b".repeat(64),
    engine: "Tesseract",
    engineVersion: "Tesseract 5",
    renderMaxPixels: 2400,
    meanWordConfidence: 70,
    truncated: false,
    warnings: ["Verify machine readings against the original."],
    createdAt: "2026-10-09T12:00:00Z",
  };
  function mountOcr(readings: Reading[] = [], pinnedId?: string) {
    vi.spyOn(api, "GET").mockImplementation(((path: string) =>
      Promise.resolve({
        data: path.endsWith("/ocr-capabilities")
          ? { available: true, languages: ["eng"], reason: "One page at a time." }
          : readings,
        response: new Response(),
      })) as typeof api.GET);
    const selected = vi.fn();
    wrap(
      <RecordsDocumentOcr
        landId={land.id}
        documentId="deed"
        page={2}
        pinnedId={pinnedId}
        onSelected={selected}
      />,
    );
    return selected;
  }

  it("reads only the chosen page and offers the reading for a question", async () => {
    const post = vi
      .spyOn(api, "POST")
      .mockResolvedValue({ data: reading, response: new Response() });
    const selected = mountOcr();
    fireEvent.click(await screen.findByRole("button", { name: "Read page 2 with OCR" }));
    await screen.findByText("Mineral rights reserved.");
    expect(screen.getByText("Verify machine readings against the original.")).toBeVisible();
    await waitFor(() => expect(selected).toHaveBeenCalledWith("ocr"));
    const calls = post.mock.calls as unknown as [
      string,
      { body: unknown; params: { path: unknown } },
    ][];
    expect(calls[0]?.[1]).toMatchObject({
      params: { path: { land_id: land.id, document_id: "deed", page: 2 } },
      body: { language: "eng" },
    });
  });

  it("says a cited reading is missing instead of showing another one in its place", async () => {
    const post = vi.spyOn(api, "POST");
    mountOcr([reading], "missing");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "cited OCR extraction is not available",
    );
    expect(screen.queryByText(/Cited machine reading/)).not.toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
  });
});

describe("old maps and photos", () => {
  function mountGallery(evidence = [photo, sheet], onAsk = vi.fn()) {
    wrap(<ArchiveGallery ids={["photo", "map"]} evidence={evidence} onAsk={onAsk} />);
    return onAsk;
  }

  it("shows the catalog date, creator and licence, and keeps links when a preview fails", () => {
    vi.spyOn(api, "GET").mockResolvedValue({ data: null, response: new Response() });
    mountGallery();
    expect(screen.getByText(/circa 1905–1910/)).toBeVisible();
    expect(screen.getByRole("link", { name: "CC BY-SA 4.0" })).toHaveAttribute(
      "href",
      photoMedia.licenseUrl,
    );
    expect(screen.getByText(/public archive and may change/)).toBeInTheDocument();
    fireEvent.error(screen.getByRole("img"));
    expect(screen.getByText(/archive preview is not available/)).toBeVisible();
    expect(screen.getByRole("link", { name: /Open original source/ })).toHaveAttribute(
      "href",
      photoMedia.sourceUrl,
    );
    fireEvent.click(screen.getByRole("button", { name: "Next source" }));
    expect(screen.getByRole("img", { name: "Mount Vernon, WA" })).toBeVisible();
    expect(screen.getByText("2 of 2")).toBeVisible();
    expect(screen.getByRole("button", { name: "Next source" })).toBeDisabled();
  });

  it("hands the agent a question about the source, keeping catalog claims apart", () => {
    vi.spyOn(api, "GET").mockResolvedValue({ data: null, response: new Response() });
    const onAsk = mountGallery();
    fireEvent.click(screen.getByRole("button", { name: "Ask about this" }));
    expect(useAgentBus.getState().request?.question).toContain("(archive source photo)");
    fireEvent.click(screen.getByRole("button", { name: "Look closely at the image" }));
    expect(useAgentBus.getState().request?.question).toContain(
      "distinguish visible details from catalog claims",
    );
    expect(onAsk).toHaveBeenCalledTimes(2);
  });

  it("loads a source that is not on the current page through the evidence API", async () => {
    const get = vi.spyOn(api, "GET").mockImplementation(((path: string) =>
      Promise.resolve({
        data: path.endsWith("/image") ? null : photo,
        response: new Response(),
      })) as typeof api.GET);
    mountGallery([]);
    expect(await screen.findByText(/circa 1905–1910/)).toBeVisible();
    expect(get).toHaveBeenCalledWith("/api/v1/research/evidence/{evidence_id}", {
      params: { path: { evidence_id: "photo" } },
    });
  });

  it("saves a private copy and shows it with its fingerprints", async () => {
    vi.spyOn(api, "GET").mockImplementation(((path: string) =>
      Promise.resolve({
        data: path.endsWith("/preview") ? new Blob(["preview"], { type: "image/png" }) : null,
        response: new Response(),
      })) as typeof api.GET);
    const post = vi
      .spyOn(api, "POST")
      .mockResolvedValue({ data: { ...image, evidenceId: "photo" }, response: new Response() });
    mountGallery();
    const save = screen.getByRole("button", { name: "Save a copy to this workspace" });
    await waitFor(() => expect(save).toBeEnabled());
    fireEvent.click(save);
    await waitFor(() => expect(screen.getByRole("img")).toHaveAttribute("src", "blob:preview"));
    expect(post).toHaveBeenCalledWith(
      "/api/v1/research/evidence/{evidence_id}/image",
      expect.objectContaining({ params: { path: { evidence_id: "photo" } } }),
    );
    expect(screen.getByText(/600 × 400 pixels/)).toBeInTheDocument();
    expect(screen.getByText(new RegExp(image.sourceSha256))).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next source" }));
    expect(revokeUrl).toHaveBeenCalledWith("blob:preview");
    expect(photoMedia.previewUrl).toMatch(/^https:/);
  });
});

describe("lining up a historical map", () => {
  let rows: Registration[] = [];
  let posts: { path: string; body: Request }[] = [];

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
    fireEvent.click(screen.getByText("Type positions instead"));
    add(0, 0, -77.055, 38.891);
    add(600, 0, -77.045, 38.891);
    add(600, 400, -77.045, 38.886);
    add(0, 400, -77.055, 38.886);
  }
  async function mountEditor() {
    const view = wrap(<RecordsAlignment evidence={sheet} />);
    await screen.findByRole("button", { name: "Choose a spot on the old map" });
    return view;
  }

  beforeEach(() => {
    rows = [];
    posts = [];
    vi.spyOn(api, "GET").mockImplementation(((path: string) =>
      Promise.resolve({
        data: path.endsWith("/registrations")
          ? rows
          : path.endsWith("/preview")
            ? new Blob(["map"], { type: "image/png" })
            : path.endsWith("/image")
              ? image
              : null,
        response: new Response(),
      })) as typeof api.GET);
    vi.spyOn(api, "POST").mockImplementation(((path: string, options: { body: Request }) => {
      posts.push({ path, body: options.body });
      if (path.endsWith("/preview"))
        return Promise.resolve({ data: fit, response: new Response() });
      const row = registration("alignment", options.body);
      rows = [row];
      return Promise.resolve({ data: row, response: new Response() });
    }) as typeof api.POST);
  });

  it("matches points, checks the fit, saves an overlay and fades it", async () => {
    await mountEditor();
    expect(screen.getByRole("button", { name: "Check alignment" })).toBeDisabled();
    fourPoints();
    fireEvent.click(screen.getByRole("button", { name: "Check alignment" }));
    await screen.findByText("Each point left out");
    fireEvent.click(screen.getByRole("button", { name: "Save and show on the land" }));
    await screen.findByRole("article", { name: "Saved alignment: Mount Vernon, WA, 1894" });
    expect(posts[0]?.body.requestKey).toBe(posts[1]?.body.requestKey);
    expect(posts[1]?.body.imageSha256).toBe(image.sha256);
    expect(useLandContext.getState().rasters.alignment).toMatchObject({
      kind: "archive-alignment",
      opacity: 0.6,
      bounds: fit.bounds,
    });
    fireEvent.keyDown(screen.getByRole("slider", { name: "Aligned map opacity" }), {
      key: "ArrowLeft",
    });
    expect(useLandContext.getState().rasters.alignment?.opacity).toBeCloseTo(0.55);
    fireEvent.click(screen.getByRole("button", { name: "Hide aligned map" }));
    expect(useLandContext.getState().rasters.alignment).toBeUndefined();
    fireEvent.click(screen.getByRole("button", { name: "Refine a copy" }));
    expect(screen.getByRole("button", { name: "Save and show on the land" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Check alignment" }));
    await waitFor(() => expect(posts).toHaveLength(3));
    expect(posts[2]?.body.requestKey).not.toBe(posts[1]?.body.requestKey);
  });

  it("keeps unfinished matches across a remount and rejects repeated or impossible ones", async () => {
    const view = await mountEditor();
    fourPoints();
    add(0, 0, -77, 38);
    expect(screen.getByRole("alert")).toHaveTextContent("its own spot");
    fill("Latitude", 95);
    expect(screen.getByRole("button", { name: "Add point match" })).toBeDisabled();
    expect(
      localStorage.getItem(`living-world-land-draft:pilot:alignment:map:${image.sha256}`),
    ).toContain("Point 4");
    view.unmount();
    await mountEditor();
    expect(screen.getByRole("list", { name: "Point matches" }).children).toHaveLength(4);
    expect(screen.getByRole("button", { name: "Check alignment" })).toBeEnabled();
  });

  it("pairs a pick on the map and cancels it when the lens changes", async () => {
    let resolve!: (point: { longitude: number; latitude: number } | null) => void;
    scene.areas.pickGround.mockImplementation(
      () =>
        new Promise<{ longitude: number; latitude: number } | null>((done) => {
          resolve = done;
        }),
    );
    scene.areas.cancelPick.mockImplementation(() => resolve(null));
    await mountEditor();
    fireEvent.click(screen.getByText("Type positions instead"));
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
    expect(useLandContext.getState().layers["records-alignment"]?.features).toHaveLength(1);
    fill("Image X", 200);
    fill("Image Y", 250);
    fireEvent.click(screen.getByRole("button", { name: "Pick matching map point" }));
    await act(async () => {
      useLens.getState().setLens("water");
      await Promise.resolve();
    });
    expect(scene.areas.cancelPick).toHaveBeenCalledOnce();
    expect(useLandContext.getState().pointPicker).toBeNull();
    expect(useLandContext.getState().layers["records-alignment"]).toBeUndefined();
  });

  it("ignores a late fit after a point changes", async () => {
    let resolve!: (response: { data: typeof fit; response: Response }) => void;
    vi.spyOn(api, "POST").mockImplementation(
      (() =>
        new Promise<{ data: typeof fit; response: Response }>((done) => {
          resolve = done;
        })) as typeof api.POST,
    );
    await mountEditor();
    fourPoints();
    fireEvent.click(screen.getByRole("button", { name: "Check alignment" }));
    fireEvent.click(screen.getByRole("button", { name: "Remove Point 4" }));
    await act(async () => {
      resolve({ data: fit, response: new Response() });
      await Promise.resolve();
    });
    expect(screen.queryByText("Each point left out")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save and show on the land" })).toBeDisabled();
  });

  it("starts a refined copy from the map sheet's saved alignments", async () => {
    rows = [
      registration("older", {
        name: "First try",
        notes: "Road crossings",
        points: [
          { label: "Point 1", imageX: 1, imageY: 1, longitude: -77.05, latitude: 38.89 },
          { label: "Point 2", imageX: 500, imageY: 1, longitude: -77.04, latitude: 38.89 },
        ],
        requestKey: crypto.randomUUID(),
        imageSha256: image.sha256,
      }),
    ];
    useRecords.setState({ landId: land.id, view: { kind: "archive", id: "map" } });
    vi.mocked(api.GET).mockImplementation(((path: string) =>
      Promise.resolve({
        data:
          path === "/api/v1/land/{land_id}/investigations"
            ? [archiveInvestigation]
            : path === "/api/v1/research/investigations/{investigation_id}"
              ? detail([sheet, photo])
              : path.endsWith("/registrations")
                ? rows
                : path.endsWith("/preview")
                  ? new Blob(["map"], { type: "image/png" })
                  : path.endsWith("/image")
                    ? image
                    : path.endsWith("/documents") || path.endsWith("/links")
                      ? []
                      : null,
        response: new Response(),
      })) as typeof api.GET);
    wrap(<RecordsLens land={land} />);
    fireEvent.click(await screen.findByRole("button", { name: "Refine a copy" }));
    expect(useRecords.getState().view).toEqual({ kind: "align", id: "map" });
    const matches = await screen.findByRole("list", { name: "Point matches" });
    expect(matches.children).toHaveLength(2);
    expect(useLandContext.getState().rasters.older).toBeDefined();
  });
});

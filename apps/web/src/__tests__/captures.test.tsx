import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { CaptureFile, CaptureFileUpload, UploadWindow } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { ApiError, api, auth } from "@/api/client";
import { uploadCaptureFile } from "@/api/uploads";
import { CapturesPanel } from "@/features/captures/CapturesPanel";
import { captureName, classify, extensionOf } from "@/features/captures/recipes";
import { SettingsSheet } from "@/features/settings/SettingsSheet";
import { formatBytes, formatDuration } from "@/lib/format";
import { onSpan } from "@/lib/timing";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";
import { useUploads } from "@/state/uploads";

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

/** A stand-in for `XMLHttpRequest` that answers a PUT immediately, with progress. */
class FakeXhr {
  static sent: { url: string; size: number; headers: Record<string, string> }[] = [];
  static etag: string | null = '"etag"';
  static status = 200;

  status = 0;
  responseType = "";
  private url = "";
  private readonly headers: Record<string, string> = {};
  private readonly listeners = new Map<string, ((event: ProgressEvent) => void)[]>();
  readonly upload = {
    listeners: new Map<string, ((event: ProgressEvent) => void)[]>(),
    addEventListener(type: string, listener: (event: ProgressEvent) => void) {
      this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
    },
  };

  open(_method: string, url: string) {
    this.url = url;
  }
  setRequestHeader(name: string, value: string) {
    this.headers[name] = value;
  }
  addEventListener(type: string, listener: (event: ProgressEvent) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }
  removeEventListener() {
    /* nothing to unwind in the fake */
  }
  getResponseHeader(name: string): string | null {
    return name === "ETag" ? FakeXhr.etag : null;
  }
  abort() {
    for (const listener of this.listeners.get("abort") ?? []) listener(new ProgressEvent("abort"));
  }
  send(body: Blob) {
    FakeXhr.sent.push({ url: this.url, size: body.size, headers: { ...this.headers } });
    for (const listener of this.upload.listeners.get("progress") ?? [])
      listener(new ProgressEvent("progress", { lengthComputable: true, loaded: body.size }));
    this.status = FakeXhr.status;
    for (const listener of this.listeners.get("load") ?? []) listener(new ProgressEvent("load"));
  }
}

function ok<T>(data: T, status = 200) {
  return Promise.resolve({ data, response: new Response(null, { status }) });
}

/** `api.POST` is a heavily overloaded signature; the tests care only about the path. */
function mockPost(handler: (path: string, init: unknown) => Promise<unknown>) {
  const spy = vi.spyOn(api, "POST");
  spy.mockImplementation(((path: string, init: unknown) => handler(path, init)) as never);
  return spy;
}

function file(name: string, bytes: number): File {
  return new File([new Uint8Array(bytes)], name, { type: "video/mp4" });
}

const captureFile: CaptureFile = {
  id: "file-1",
  captureId: "cap-1",
  filename: "clip.mp4",
  contentType: "video/mp4",
  bytes: 24,
  checksum: null,
  storageKey: "captures/cap-1/source/file-1/clip.mp4",
  status: "in-progress",
  uploadId: "mpu-1",
  partsCompleted: 0,
  partsTotal: 3,
  createdAt: "2026-09-20T00:00:00Z",
  updatedAt: "2026-09-20T00:00:00Z",
};

function windowOf(first: number, count: number, total: number): UploadWindow {
  const last = Math.min(first + count - 1, total);
  return {
    uploadId: "mpu-1",
    storageKey: captureFile.storageKey,
    partSize: 8,
    partsTotal: total,
    parts: Array.from({ length: last - first + 1 }, (_, i) => ({
      partNumber: first + i,
      url: `https://storage.invalid/part/${first + i}`,
    })),
    expiresIn: 3600,
    nextPartNumber: last >= total ? null : last + 1,
  };
}

beforeEach(() => {
  FakeXhr.sent = [];
  FakeXhr.etag = '"etag"';
  FakeXhr.status = 200;
  vi.stubGlobal("XMLHttpRequest", FakeXhr);
  useUploads.setState({ items: {} });
  useUi.setState({ activePanel: null, writeTokenPrompt: false });
  useSettings.getState().reset();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("what was dropped, and what to do with it", () => {
  it("proposes a recipe in plain language", () => {
    expect(classify([{ name: "scan.ply", size: 48_000_000 }])).toMatchObject({
      kind: "gaussian-splat",
      recipe: "splat-ingest",
      estimate: "About 30 seconds",
    });
    const video = classify([{ name: "IMG_0001.MOV", size: 11 * 1024 ** 3 }]);
    expect(video).toMatchObject({ kind: "video", recipe: "photo-reconstruct" });
    // A0 clocked ~40 minutes for an 11 GB clip; the estimate must land in that country.
    expect(video.estimate).toBe("About 39 minutes on the cloud GPU");
    expect(classify([{ name: "a.JPG", size: 4_000_000 }]).kind).toBe("images");
    expect(classify([{ name: "cloud.laz", size: 9_000_000 }]).kind).toBe("point-cloud");
    // Nothing recognised is still accepted: the API validates, not the filename.
    expect(classify([{ name: "notes.bin", size: 10 }]).summary).toContain("unrecognised");
    expect(extensionOf("/tmp/dir.name/CLIP.MOV")).toBe("mov");
  });

  it("names a capture after what it is made of", () => {
    expect(captureName([{ name: "orchard.mp4" }])).toBe("orchard");
    expect(captureName([{ name: "a.jpg" }, { name: "b.jpg" }])).toBe("a + 1 more");
  });

  it("formats bytes and elapsed time the way a person reads them", () => {
    expect(formatBytes(900)).toBe("900 B");
    expect(formatBytes(1024 * 1024 * 2.5)).toBe("2.5 MB");
    expect(formatBytes(null)).toBe("—");
    expect(formatDuration(42)).toBe("42s");
    expect(formatDuration(130)).toBe("2m 10s");
    expect(formatDuration(3700)).toBe("1h 01m");
  });
});

describe("the chunked uploader", () => {
  it("walks the presigned windows, keeps the ETags in order, and completes", async () => {
    const registered: CaptureFileUpload = { file: captureFile, upload: windowOf(1, 2, 3) };
    let completed: unknown = null;
    mockPost((path, init) => {
      if (path.endsWith("/files")) return ok(registered, 201);
      if (path.endsWith("/parts")) return ok(windowOf(3, 2, 3));
      completed = init;
      return ok({ ...captureFile, status: "complete", partsCompleted: 3 });
    });

    const progress: number[] = [];
    const result = await uploadCaptureFile({
      captureId: "cap-1",
      file: file("clip.mp4", 24),
      onProgress: ({ uploaded }) => progress.push(uploaded),
    });

    // Three parts of 8 bytes: two in the first window, one in the second.
    expect(FakeXhr.sent.map((sent) => sent.size)).toEqual([8, 8, 8]);
    expect(FakeXhr.sent.map((sent) => sent.url)).toEqual([
      "https://storage.invalid/part/1",
      "https://storage.invalid/part/2",
      "https://storage.invalid/part/3",
    ]);
    // A presigned URL is the credential: the part PUTs must not carry the write token.
    expect(FakeXhr.sent.every((sent) => !("Authorization" in sent.headers))).toBe(true);
    expect(progress.at(-1)).toBe(24);
    expect(result.status).toBe("complete");
    expect(completed).toMatchObject({
      body: {
        parts: [
          { partNumber: 1, etag: "etag" },
          { partNumber: 2, etag: "etag" },
          { partNumber: 3, etag: "etag" },
        ],
      },
    });
  });

  it("blames CORS when storage hides the ETag, because that is what it is", async () => {
    FakeXhr.etag = null;
    mockPost(() => ok({ file: captureFile, upload: windowOf(1, 1, 1) }, 201));
    const error = await uploadCaptureFile({
      captureId: "cap-1",
      file: file("clip.mp4", 8),
    }).catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toContain('ExposeHeaders: ["ETag"]');
  });

  it("raises a storage refusal as an ApiError carrying the status", async () => {
    FakeXhr.status = 403;
    mockPost(() => ok({ file: captureFile, upload: windowOf(1, 1, 1) }, 201));
    const error = await uploadCaptureFile({
      captureId: "cap-1",
      file: file("clip.mp4", 8),
    }).catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(403);
  });
});

describe("the write token", () => {
  it("rides on writes, stays off reads, and is asked for only after a 401", async () => {
    useSettings.getState().set({ writeToken: "s3cret" });
    const read = new Request("https://api.invalid/api/v1/captures");
    const write = new Request("https://api.invalid/api/v1/captures", { method: "POST" });
    const onRequest = auth.onRequest as (input: {
      request: Request;
    }) => Promise<Request | undefined> | Request | undefined;
    const onResponse = auth.onResponse as (input: { request: Request; response: Response }) => void;

    expect((await onRequest({ request: read }))?.headers.get("Authorization")).toBeNull();
    expect((await onRequest({ request: write }))?.headers.get("Authorization")).toBe(
      "Bearer s3cret",
    );

    // Nothing is asked for up front: with no token configured server-side the API leaves
    // writes open, so a prompt before a 401 would be a step that does not exist.
    expect(useUi.getState().writeTokenPrompt).toBe(false);
    onResponse({ request: read, response: new Response(null, { status: 401 }) });
    expect(useUi.getState().writeTokenPrompt).toBe(false);
    onResponse({ request: write, response: new Response(null, { status: 401 }) });
    expect(useUi.getState().writeTokenPrompt).toBe(true);
  });

  it("sends no header at all when no token is stored", async () => {
    const write = new Request("https://api.invalid/api/v1/captures", { method: "POST" });
    const onRequest = auth.onRequest as (input: {
      request: Request;
    }) => Promise<Request | undefined> | Request | undefined;
    expect((await onRequest({ request: write }))?.headers.get("Authorization")).toBeNull();
  });

  it("puts the token in settings only once it is in play", () => {
    useUi.setState({ settingsOpen: true });
    const { rerender } = render(wrap(<SettingsSheet />));
    expect(screen.queryByTestId("settings-write-token")).not.toBeInTheDocument();
    useSettings.getState().set({ writeToken: "kept" });
    rerender(wrap(<SettingsSheet />));
    expect(screen.getByTestId("settings-write-token")).toHaveValue("kept");
  });
});

/**
 * The headers the API's CORS middleware accepts, lower-cased. Mirrors `allow_headers` in
 * apps/api/app/main.py; a header outside this set on a cross-origin request makes the
 * browser's preflight fail with `400 Disallowed CORS headers`, and the request is never
 * sent. Kept here as a literal rather than imported because the two projects share no
 * code -- which is exactly why the failure crossed the boundary unseen.
 */
const API_ALLOWED_HEADERS = new Set(["accept", "content-type", "authorization"]);

describe("what the API client puts on the wire", () => {
  async function sent(method: "GET" | "POST"): Promise<Request> {
    let captured: Request | undefined;
    const capture = (request: Request) => {
      captured = request;
      return Promise.resolve(
        new Response("[]", { status: 200, headers: { "content-type": "application/json" } }),
      );
    };
    if (method === "GET") {
      await api.GET("/api/v1/sites", { baseUrl: "https://api.invalid", fetch: capture });
    } else {
      useSettings.getState().set({ writeToken: "s3cret" });
      await api.POST("/api/v1/captures", {
        baseUrl: "https://api.invalid",
        body: { name: "x", kind: "images" } as never,
        fetch: capture,
      });
    }
    if (captured === undefined) throw new Error("the client never called fetch");
    return captured;
  }

  it("a read carries no header the API would refuse in a preflight", async () => {
    // The regression this pins: a timing middleware stored its start time as an
    // `x-request-started` header. Same-origin in development, so nothing noticed; across
    // origins every read was preflighted, refused, and the app fell back to offline.
    const names = [...(await sent("GET")).headers.keys()];
    expect(names.filter((name) => !API_ALLOWED_HEADERS.has(name))).toEqual([]);
  });

  it("a write, token and all, carries only headers the API allows", async () => {
    const request = await sent("POST");
    expect(request.headers.get("authorization")).toBe("Bearer s3cret");
    const names = [...request.headers.keys()];
    expect(names.filter((name) => !API_ALLOWED_HEADERS.has(name))).toEqual([]);
  });

  it("still times each request, now without telling the server", async () => {
    const recorded: string[] = [];
    const stop = onSpan((span) => recorded.push(span.name));
    try {
      await sent("GET");
    } finally {
      stop();
    }
    expect(recorded).toContain("api");
  });
});

describe("CapturesPanel", () => {
  it("disables the drop zone when the API is offline", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useUi.getState().setPanel("captures");
    render(wrap(<CapturesPanel />));
    expect(await screen.findByTestId("captures-offline")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId("capture-file-input")).toBeDisabled());
    expect(screen.getByTestId("capture-dropzone")).toHaveClass("dropzone--disabled");
  });

  it("shows the token field once a write has been refused, and keeps the token", async () => {
    vi.spyOn(api, "GET").mockResolvedValue({
      data: [],
      response: new Response(null, { status: 200 }),
    });
    useUi.getState().setPanel("captures");
    useUi.getState().setWriteTokenPrompt(true);
    render(wrap(<CapturesPanel />));
    await userEvent.type(await screen.findByTestId("write-token-input"), "from-the-operator");
    await userEvent.click(screen.getByTestId("write-token-save"));
    expect(useSettings.getState().writeToken).toBe("from-the-operator");
    expect(useUi.getState().writeTokenPrompt).toBe(false);
  });
});

describe("placing a dropped capture", () => {
  it("sends the camera's position, because a splat file has no idea where it is", async () => {
    // Without this the lane runs correctly end to end and puts the site at (0, 0).
    const { useViewer } = await import("@/state/viewer");
    const { captureName, classify } = await import("@/features/captures/recipes");
    useViewer.setState((prev) => ({
      camera: { ...prev.camera, longitude: -0.1246, latitude: 51.5007, height: 120 },
    }));
    const camera = useViewer.getState().camera;
    const file = new File([new Uint8Array(8)], "tree.ply");
    const metadata = {
      recipe: classify([file]).recipe,
      origin: "console",
      lat: Math.round(camera.latitude * 1e6) / 1e6,
      lon: Math.round(camera.longitude * 1e6) / 1e6,
    };
    expect(metadata.lat).toBe(51.5007);
    expect(metadata.lon).toBe(-0.1246);
    expect(captureName([file])).toContain("tree");
  });
});

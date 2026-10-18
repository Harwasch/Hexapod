import { describe, expect, it, vi, afterEach } from "vitest";
import {
  parseLibraryArchive,
  validateRecord,
  getSettings,
  saveSettings,
  getApiToken,
  setApiToken,
  type LibraryArchive,
} from "./storage";
import { createWorldApi, validateApiBaseUrl } from "./api";
import { parseTrajectory, replayTrajectory, bindingSupport } from "./controls";
import { getModel } from "./catalog";
import type { WorldProject, ControlEvent } from "./types";

function project(): WorldProject {
  return {
    id: "project-1",
    name: "Forest",
    prompt: "A forest",
    modelId: "astronex-world",
    providerId: "runpod",
    createdAt: 1000,
    updatedAt: 1000,
    settings: { performance: "balanced", seed: 0 },
    assetIds: [],
    characterIds: [],
  };
}
function archive(): LibraryArchive {
  return {
    format: "hexapod-worlds",
    version: 1,
    exportedAt: 1000,
    records: {
      projects: [project()],
      scenes: [],
      characters: [],
      replays: [],
      assets: [],
      benchmarks: [],
      reconstructions: [],
    },
    blobs: [],
  };
}
afterEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
  localStorage.clear();
  vi.useRealTimers();
});
describe("portable local library validation", () => {
  it("accepts a complete portable library with zero-valued seed", () => {
    expect(parseLibraryArchive(JSON.stringify(archive())).records.projects[0]?.settings.seed).toBe(
      0,
    );
  });
  it("rejects malformed or future versions before storage is changed", () => {
    expect(() => parseLibraryArchive("{")).toThrow("valid JSON");
    expect(() => parseLibraryArchive(JSON.stringify({ ...archive(), version: 2 }))).toThrow(
      "Unsupported",
    );
    const data = archive();
    (data.records as unknown as Record<string, unknown>).projects = [{}];
    expect(() => parseLibraryArchive(JSON.stringify(data))).toThrow("Invalid");
  });
  it("rejects dangling references, duplicate IDs and corrupt media", () => {
    const data = archive();
    data.records.projects[0]!.assetIds = ["missing"];
    expect(() => parseLibraryArchive(JSON.stringify(data))).toThrow("missing media");
    data.records.projects[0]!.assetIds = [];
    data.records.projects.push(project());
    expect(() => parseLibraryArchive(JSON.stringify(data))).toThrow("Duplicate");
    data.records.projects.pop();
    data.records.assets.push({
      id: "a",
      name: "a",
      createdAt: 1000,
      size: 3,
      mimeType: "image/png",
      kind: "image",
    });
    data.blobs.push({ id: "a", mimeType: "image/png", base64: "!!!" });
    expect(() => parseLibraryArchive(JSON.stringify(data))).toThrow("Corrupted");
  });
  it("rejects impossible metrics, negative seed and non-finite action vectors", () => {
    expect(() =>
      validateRecord("projects", { ...project(), settings: { performance: "balanced", seed: -1 } }),
    ).toThrow("Seed");
    expect(() =>
      validateRecord("benchmarks", {
        id: "b",
        name: "b",
        projectId: "project-1",
        createdAt: 0,
        modelId: "a",
        providerId: "runpod",
        durationMs: 100,
        frameCount: 1,
        measuredFPS: Infinity,
        events: [],
      }),
    ).toThrow("measuredFPS");
    expect(() =>
      validateRecord("scenes", {
        id: "s",
        name: "s",
        projectId: "project-1",
        createdAt: 0,
        modelId: "a",
        prompt: "",
        assetIds: [],
        resumeKind: "visual",
        events: [{ id: "e", timestampMs: 0, type: "native", values: { x: Infinity } }],
      }),
    ).toThrow("action vector");
  });
  it("does not persist credentials in exportable settings", () => {
    setApiToken("test-token");
    saveSettings({ defaultProvider: "runpod" });
    expect(getApiToken()).toBe("test-token");
    expect(JSON.stringify(getSettings())).not.toContain("test-token");
    expect(localStorage.getItem("hexapod.worlds.settings.v1")).not.toContain("test-token");
  });
});
describe("secure manager routing", () => {
  it("allows same-origin, TLS and loopback but rejects insecure remote hosts or URL credentials", () => {
    expect(validateApiBaseUrl("")).toBe("");
    expect(validateApiBaseUrl("http://localhost:8000/api/v1/worlds/")).toBe(
      "http://localhost:8000",
    );
    expect(validateApiBaseUrl("https://worker.example")).toBe("https://worker.example");
    for (const url of [
      "http://worker.example",
      "https://user:password@worker.example",
      "javascript:alert(1)",
      "https://worker.example?token=secret",
    ])
      expect(() => validateApiBaseUrl(url)).toThrow();
  });
  it("attaches session auth without forwarding browser cookies", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ providers: [] }), {
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    setApiToken("session-secret");
    await createWorldApi("https://manager.example").providers();
    expect(fetchMock.mock.calls[0]?.[0]).toBe("https://manager.example/api/v1/worlds/providers");
    const options = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(options.credentials).toBe("omit");
    expect(options.redirect).toBe("error");
    expect(new Headers(options.headers).get("Authorization")).toBe("Bearer session-secret");
  });
  it("rejects SPA fallback HTML and exposes configured-service failures clearly", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(new Response("<html/>", { headers: { "content-type": "text/html" } })),
    );
    await expect(createWorldApi().providers()).rejects.toThrow("web page");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "GPU unavailable" }), {
          status: 503,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    await expect(createWorldApi().providers()).rejects.toThrow("GPU unavailable");
  });
});
describe("capability-aware replay controls", () => {
  it("only exposes implemented next-clip actions and never claims realtime", () => {
    expect(
      bindingSupport(
        { id: "binding", key: "KeyW", label: "Move", type: "native", action: "jump" },
        getModel("astronex-world"),
      ).supported,
    ).toBe(false);
    expect(getModel("astronex-world").capabilities.runtime.realtime).toBe(false);
    expect(getModel("astronex-world").capabilities.runtime.resolutionOptions).toEqual(["832x480"]);
    expect(getModel("not-a-model").capabilities.runtime.realtime).toBe(false);
  });
  it("validates imported trajectories before dispatch", () => {
    expect(() =>
      parseTrajectory(
        JSON.stringify({
          format: "worlds-control-trajectory",
          version: 1,
          prompt: "test",
          events: [{ id: "a", timestampMs: -1, type: "native" }],
        }),
      ),
    ).toThrow("Invalid event");
  });
  it("preserves event order and cancels pending dispatch cleanly", async () => {
    vi.useFakeTimers();
    const events: ControlEvent[] = [
      { id: "b", timestampMs: 200, type: "pause" },
      { id: "a", timestampMs: 100, type: "resume" },
    ];
    const send = vi.fn().mockResolvedValue(undefined);
    const controller = new AbortController();
    const promise = replayTrajectory(events, send, { signal: controller.signal });
    const assertion = expect(promise).rejects.toMatchObject({ name: "AbortError" });
    await vi.advanceTimersByTimeAsync(100);
    expect(send).toHaveBeenCalledWith(events[1]);
    controller.abort();
    await assertion;
    expect(send).toHaveBeenCalledTimes(1);
  });
});

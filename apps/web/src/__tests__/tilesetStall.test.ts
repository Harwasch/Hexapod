/**
 * A site's tileset at a URL fails only when its JSON stalls, not when it is merely slow
 * (providers/tiles.ts, createSiteTileset with deadlines).
 *
 * Each attempt used to have 15 s in all, so a large tileset.json on a slow phone link failed
 * every retry the same way, each starting the download over. Now the root JSON is fetched here
 * with a stall timeout and handed to CesiumJS ready-made. CesiumJS's `fromUrl` is faked; what
 * is checked is what it is handed, and when the attempt gives up.
 */
import { Cesium3DTileset, type Resource } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SiteAsset } from "@twin/contracts";

import { createSiteTileset } from "@/cesium/providers/tiles";

vi.mock("@/lib/tileProxy", () => ({ tileUrl: (url: string) => Promise.resolve(url) }));

const URL_ = "https://bucket.test/yard/tileset.json";
const JSON_ = JSON.stringify({ asset: { version: "1.1" }, root: { padding: "x".repeat(4000) } });

const mesh = {
  id: "mesh",
  name: "Yard mesh",
  representation: "mesh",
  source: { type: "3d-tiles-url", url: URL_ },
  renderConfig: {},
} as unknown as SiteAsset;

/** The root JSON in four chunks, one every ten seconds; quiet after `stopAfter` of them. */
function slowLink(stopAfter = 4) {
  const encoded = new TextEncoder().encode(JSON_);
  const size = Math.ceil(encoded.length / 4);
  return vi.fn((_url: string, init?: RequestInit) => {
    let sent = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        return new Promise<void>((resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
          if (sent >= stopAfter) return;
          setTimeout(() => {
            controller.enqueue(encoded.slice(sent * size, (sent + 1) * size));
            sent += 1;
            if (sent >= 4) controller.close();
            resolve();
          }, 10_000);
        });
      },
    });
    return Promise.resolve(new Response(body));
  });
}

describe("a tileset's JSON on a slow link", () => {
  let handed: { url: string; json: unknown }[] = [];

  beforeEach(() => {
    vi.useFakeTimers();
    handed = [];
    vi.spyOn(Cesium3DTileset, "fromUrl").mockImplementation(async (given) => {
      const resource = given as Resource;
      handed.push({ url: resource.url, json: await resource.fetchJson() });
      return {
        destroy: vi.fn(),
        isDestroyed: () => false,
        cacheBytes: 0,
        maximumCacheOverflowBytes: 0,
        totalMemoryUsageInBytes: 0,
      } as unknown as Cesium3DTileset;
    });
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("loads when the whole download takes far longer than the deadline, while bytes come", async () => {
    const link = slowLink();
    vi.stubGlobal("fetch", link);
    const made = createSiteTileset(
      mesh,
      { maximumScreenSpaceError: 16 },
      { stallMs: 15_000, what: "Yard mesh" },
    );
    await vi.advanceTimersByTimeAsync(45_000);
    await expect(made).resolves.toBeDefined();
    // CesiumJS is handed the JSON already downloaded, at the tileset's own URL (its tiles are
    // relative to it), and nothing is downloaded twice.
    expect(handed).toEqual([{ url: URL_, json: JSON.parse(JSON_) as unknown }]);
    expect(link).toHaveBeenCalledOnce();
  });

  it("fails, and says why, once the download stalls", async () => {
    vi.stubGlobal("fetch", slowLink(2));
    const made = createSiteTileset(
      mesh,
      { maximumScreenSpaceError: 16 },
      { stallMs: 15_000, what: "Yard mesh" },
    );
    const failed = expect(made).rejects.toThrow("Yard mesh stopped arriving for 15 s");
    await vi.advanceTimersByTimeAsync(40_000);
    await failed;
    expect(handed).toEqual([]);
  });
});

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { loadInstances } from "@/lib/instances";
import {
  DESKTOP_PAYLOAD_BYTES,
  HANDHELD_PAYLOAD_BYTES,
  PayloadCache,
  scanPayloads,
} from "@/lib/payloadCache";
import { loadSkin } from "@/lib/skin";
import { loadSupersedes } from "@/lib/supersedes";

const YARD = resolve(process.cwd(), "../../data/tiles/synthetic-yard");

/** A load that resolves with `value` of `bytes`, counting its calls. */
function loader<T>(value: T, bytes: number) {
  const fn = vi.fn(() => Promise.resolve({ value, bytes }));
  return fn;
}

describe("scan payloads kept in memory (LRU by bytes)", () => {
  it("loads a URL once, then hands back the same document", async () => {
    const cache = new PayloadCache(() => 100);
    const load = loader({ doc: 1 }, 10);
    const a = await cache.get("u", load);
    const b = await cache.get("u", load);
    expect(a).toBe(b);
    expect(load).toHaveBeenCalledTimes(1);
    expect(cache.hits).toBe(1);
    expect(cache.misses).toBe(1);
    expect(cache.bytes).toBe(10);
  });

  it("shares a load still in flight", async () => {
    const cache = new PayloadCache(() => 100);
    const load = loader("x", 5);
    const [a, b] = await Promise.all([cache.get("u", load), cache.get("u", load)]);
    expect(a).toBe("x");
    expect(b).toBe("x");
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("lets the least recently used go when the bytes pass the cap", async () => {
    const cache = new PayloadCache(() => 100);
    await cache.get("a", loader("a", 40));
    await cache.get("b", loader("b", 40));
    // `a` used again: `b` is now the oldest.
    await cache.get("a", loader("a", 40));
    await cache.get("c", loader("c", 40));
    expect(cache.keys).toEqual(["a", "c"]);
    expect(cache.bytes).toBe(80);
    const again = loader("b", 40);
    await cache.get("b", again);
    expect(again).toHaveBeenCalledTimes(1);
  });

  it("hands back but does not keep a payload larger than the whole cap", async () => {
    const cache = new PayloadCache(() => 100);
    await cache.get("small", loader("s", 30));
    const big = await cache.get("big", loader("B", 150));
    expect(big).toBe("B");
    expect(cache.keys).toEqual(["small"]);
    expect(cache.bytes).toBe(30);
  });

  it("does not keep a load that failed: the next asks again", async () => {
    const cache = new PayloadCache(() => 100);
    const failing = vi.fn(() => Promise.reject(new Error("503")));
    await expect(cache.get("u", failing)).rejects.toThrow("503");
    expect(cache.keys).toEqual([]);
    const ok = loader("ok", 1);
    await expect(cache.get("u", ok)).resolves.toBe("ok");
    expect(ok).toHaveBeenCalledTimes(1);
  });

  it("keeps a few hundred MB on a desktop, less on a phone", () => {
    expect(DESKTOP_PAYLOAD_BYTES).toBeGreaterThanOrEqual(256 * 1024 * 1024);
    expect(DESKTOP_PAYLOAD_BYTES).toBeLessThanOrEqual(512 * 1024 * 1024);
    expect(HANDHELD_PAYLOAD_BYTES).toBeLessThan(DESKTOP_PAYLOAD_BYTES);
    // The Minnetonka tree's largest skin (75 MB) fits either.
    expect(HANDHELD_PAYLOAD_BYTES).toBeGreaterThan(75_046_496);
  });
});

describe("switching methods and back fetches nothing again", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  /** `fetch` answering from a table of URL suffixes, counting what it was asked for. */
  function stubFetch(files: Record<string, () => BodyInit>) {
    const asked: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string) => {
        const url = input;
        asked.push(url);
        const hit = Object.entries(files).find(([suffix]) => url.endsWith(suffix));
        return Promise.resolve(
          hit ? new Response(hit[1]()) : new Response("missing", { status: 404 }),
        );
      }),
    );
    return asked;
  }

  it("a skin (skin.json and skin.bin): Today, Limbs, Today, Limbs", async () => {
    const json = readFileSync(resolve(YARD, "skin/skin.json"), "utf8");
    const bin = (): BodyInit => Uint8Array.from(readFileSync(resolve(YARD, "skin/skin.bin")));
    const asked = stubFetch({
      "skin/skin.json": () => json,
      "skin/skin.bin": bin,
      "limbs/skin.json": () => json,
      "limbs/skin.bin": bin,
    });
    const tileset = "https://bucket.example/sites/tree/splat/tileset.json";
    const today = { uri: "skin/skin.json", count: 0 };
    const limbs = { uri: "variants/skins/limbs/skin.json", count: 0 };
    const t1 = await loadSkin(tileset, today);
    const l1 = await loadSkin(tileset, limbs);
    const t2 = await loadSkin(tileset, today);
    const l2 = await loadSkin(tileset, limbs);
    expect(t2).toBe(t1);
    expect(l2).toBe(l1);
    // Two files each, the first time only.
    expect(asked).toHaveLength(4);
    expect(scanPayloads.bytes).toBeGreaterThan(0);
  });

  it("an objects variant's instances.json and a fill's supersedes.json", async () => {
    const instances = readFileSync(resolve(YARD, "instances/instances.json"), "utf8");
    const asked = stubFetch({
      "instances.json": () => instances,
      "supersedes.json": () => JSON.stringify({ superseded: 2, tiles: { abc: [1, 2] } }),
    });
    const tileset = "https://bucket.example/runs/r/p0123456789abcdef/package/splat/tileset.json";
    const ref = { uri: "variants/objects/a/instances.json", count: 0 };
    const first = await loadInstances(tileset, ref);
    expect(await loadInstances(tileset, ref)).toBe(first);
    const sup = await loadSupersedes(tileset, "variants/fill/f/supersedes.json");
    expect(await loadSupersedes(tileset, "variants/fill/f/supersedes.json")).toBe(sup);
    expect(asked).toHaveLength(2);
  });
});

import { afterEach, describe, expect, it, vi } from "vitest";

import { resetTileProxy, tileUrl } from "@/lib/tileProxy";

const MANAGED =
  "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs/abc/package/splat/tileset.json";

function answer(headers: Record<string, string>, status = 204): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response(null, { status, headers }))),
  );
}

describe("tiles through the web app's own origin", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    resetTileProxy();
  });

  it("routes the managed bucket URL through /r2/ once the function answers", async () => {
    answer({ "X-Tile-Proxy": "1" });
    expect(await tileUrl(MANAGED)).toBe(
      `${location.origin}/r2/pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs/abc/package/splat/tileset.json`,
    );
  });

  it("leaves it alone where no function answers (a dev server, a plain static host)", async () => {
    answer({ "Content-Type": "text/html" }, 200);
    expect(await tileUrl(MANAGED)).toBe(MANAGED);
    resetTileProxy();
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new Error("offline"))),
    );
    expect(await tileUrl(MANAGED)).toBe(MANAGED);
  });

  it("never touches other hosts, and asks only once", async () => {
    answer({ "X-Tile-Proxy": "1" });
    expect(await tileUrl("https://tiles.example.com/a/tileset.json")).toBe(
      "https://tiles.example.com/a/tileset.json",
    );
    expect(await tileUrl("/fixture-tiles/x/tileset.json")).toBe("/fixture-tiles/x/tileset.json");
    await tileUrl(MANAGED);
    await tileUrl(MANAGED);
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1);
  });
});

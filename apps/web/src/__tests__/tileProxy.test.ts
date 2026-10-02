import { afterEach, describe, expect, it, vi } from "vitest";

import { resetTileProxy, tileUrl } from "@/lib/tileProxy";

const OURS = "pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev";
const MANAGED = `https://${OURS}/runs/abc/package/splat/tileset.json`;
/** Another public bucket's tileset: added through Add data, a demo bucket, an older one. */
const ELSEWHERE = "https://pub-0123456789abcdef0123456789abcdef.r2.dev/demo/splat/tileset.json";

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

  it("routes the pinned bucket's URLs through /r2/ once the function names it", async () => {
    answer({ "X-Tile-Proxy": OURS });
    expect(await tileUrl(MANAGED)).toBe(
      `${location.origin}/r2/${OURS}/runs/abc/package/splat/tileset.json`,
    );
  });

  it("leaves every other r2.dev bucket at its own URL, which the function would 404", async () => {
    answer({ "X-Tile-Proxy": OURS });
    expect(await tileUrl(ELSEWHERE)).toBe(ELSEWHERE);
    expect(await tileUrl(MANAGED)).toBe(
      `${location.origin}/r2/${OURS}/runs/abc/package/splat/tileset.json`,
    );
  });

  it("routes nothing on an answer that names no bucket (the old `1` among them)", async () => {
    for (const value of ["1", "", "tiles.example.com", `${OURS}.evil.com`]) {
      resetTileProxy();
      answer({ "X-Tile-Proxy": value });
      expect(await tileUrl(MANAGED), value).toBe(MANAGED);
      expect(await tileUrl(ELSEWHERE), value).toBe(ELSEWHERE);
    }
    resetTileProxy();
    answer({ "X-Tile-Proxy": OURS }, 404);
    expect(await tileUrl(MANAGED)).toBe(MANAGED);
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
    answer({ "X-Tile-Proxy": OURS });
    expect(await tileUrl("https://tiles.example.com/a/tileset.json")).toBe(
      "https://tiles.example.com/a/tileset.json",
    );
    expect(await tileUrl("/fixture-tiles/x/tileset.json")).toBe("/fixture-tiles/x/tileset.json");
    await tileUrl(MANAGED);
    await tileUrl(MANAGED);
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1);
  });
});

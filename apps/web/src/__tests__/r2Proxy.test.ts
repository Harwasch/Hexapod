/**
 * The Pages Function in `functions/r2/` (the tile proxy), run as the Worker it is: a
 * Request in, a Response out, `fetch` stubbed for the bucket. It lives here rather than
 * beside the function because every file under `functions/` becomes a route.
 *
 * What it guards is this origin: anything the function answers is served from the place
 * the console keeps the write token, so the tests are mostly about what it refuses.
 */
import { resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

interface ProxyModule {
  onRequest: (context: {
    request: Request;
    env?: { TILE_PROXY_HOST?: string };
  }) => Promise<Response>;
}

// Through a computed path so TypeScript does not try to resolve a plain-JS module with no
// types (this project does not allowJs); the interface above is the contract it is held to.
// From the working directory, as the fixture tests find data/tiles: vitest runs in apps/web.
const modulePath = resolve(process.cwd(), "../../functions/r2/[[path]].js");
const { onRequest } = (await import(/* @vite-ignore */ modulePath)) as ProxyModule;

const OURS = "pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev";
/** A published generation's tileset directory (apps/api/app/services/published.py). */
const GENERATION = "runs/a/p0123456789abcdef/package/splat";
const THEIRS = "pub-0123456789abcdef0123456789abcdef.r2.dev";
const ORIGIN = "https://twin-web.pages.dev";
const env = { TILE_PROXY_HOST: OURS };

function get(path: string, headers: Record<string, string> = {}, method = "GET"): Request {
  return new Request(`${ORIGIN}${path}`, { method, headers });
}

/** The bucket: answers every fetch with `body` as `upstreamType`, and records the calls. */
function bucket(init: ResponseInit = {}, body: BodyInit | null = "bytes"): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve(
        new Response(body, {
          status: 200,
          ...init,
          headers: { "Content-Type": "text/html", ...(init.headers as Record<string, string>) },
        }),
      ),
    ),
  );
}

function fetched(): { url: string; init: RequestInit }[] {
  return vi.mocked(fetch).mock.calls.map(([url, init]) => ({
    url: url instanceof Request ? url.url : url.toString(),
    init: init ?? {},
  }));
}

/** The edge cache lifetimes the function asked Cloudflare for, by upstream status. */
function edgeTtl(init: RequestInit): unknown {
  return (init as { cf?: { cacheTtlByStatus?: unknown } }).cf?.cacheTtlByStatus;
}

describe("the /r2 tile proxy", () => {
  beforeEach(() => bucket());
  afterEach(() => vi.unstubAllGlobals());

  it("names the one host it serves on the probe, once a host is pinned", async () => {
    const response = await onRequest({ request: get("/r2/", {}, "HEAD"), env });
    expect(response.status).toBe(204);
    expect(response.headers.get("X-Tile-Proxy")).toBe(OURS);
    // However the operator wrote it, the probe names the bare hostname the web app compares.
    const asUrl = { TILE_PROXY_HOST: ` https://${OURS.toUpperCase()}/ ` };
    const probe = await onRequest({ request: get("/r2/", {}, "HEAD"), env: asUrl });
    expect(probe.headers.get("X-Tile-Proxy")).toBe(OURS);
  });

  it("with no host pinned, answers the probe without the header and proxies nothing", async () => {
    for (const unset of [undefined, {}, { TILE_PROXY_HOST: "" }, { TILE_PROXY_HOST: "evil.com" }]) {
      const probe = await onRequest({ request: get("/r2/", {}, "HEAD"), env: unset });
      expect(probe.headers.get("X-Tile-Proxy")).toBeNull();
      expect(probe.ok).toBe(false);
      const tile = await onRequest({ request: get(`/r2/${OURS}/runs/a/0.glb`), env: unset });
      expect(tile.status).toBe(404);
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it("accepts the pinned host as a URL as well as a hostname", async () => {
    const asUrl = { TILE_PROXY_HOST: ` https://${OURS}/ ` };
    const response = await onRequest({ request: get(`/r2/${OURS}/runs/a/0.glb`), env: asUrl });
    expect(response.status).toBe(200);
    expect(fetched()[0]?.url).toBe(`https://${OURS}/runs/a/0.glb`);
  });

  it("refuses every other bucket, without fetching it", async () => {
    for (const path of [
      `/r2/${THEIRS}/x.json`,
      `/r2/${THEIRS}/page.html`,
      "/r2/example.com/x.json",
      `/r2/${OURS}`,
      `/r2/${OURS}/`,
    ]) {
      const response = await onRequest({ request: get(path), env });
      expect(response.status).toBe(404);
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it("labels a response by the key's extension, never by what the bucket says", async () => {
    const cases: [string, string][] = [
      ["sites/x/tileset.json", "application/json"],
      ["runs/a/package/splat/0/1.glb", "model/gltf-binary"],
      ["runs/a/package/mesh/0.b3dm", "application/octet-stream"],
      ["runs/a/package/points/0.pnts", "application/octet-stream"],
      ["runs/a/package/mesh/buffer.bin", "application/octet-stream"],
      ["runs/a/package/splat/0.emb", "application/octet-stream"],
      ["runs/a/package/splat/0.f32", "application/octet-stream"],
      ["runs/a/package/splat/0.u8", "application/octet-stream"],
      ["runs/a/train/splat.ply", "application/octet-stream"],
      ["runs/a/package/splat.spz", "application/octet-stream"],
      ["sites/x/thumbnail.webp", "image/webp"],
      ["sites/x/thumbnail.jpg", "image/jpeg"],
      ["sites/x/thumbnail.JPEG", "image/jpeg"],
      ["sites/x/thumbnail.png", "image/png"],
    ];
    for (const [key, type] of cases) {
      const response = await onRequest({ request: get(`/r2/${OURS}/${key}`), env });
      expect(response.status, key).toBe(200);
      expect(response.headers.get("Content-Type"), key).toBe(type);
    }
  });

  it("404s an extension it does not know, before fetching anything", async () => {
    for (const key of [
      "x.html",
      "x.htm",
      "x.svg",
      "x.js",
      "x.xml",
      "x.constructor",
      "runs/a/noextension",
      "runs/a/.json",
      // Encoded, the dot is still a dot to R2: this is the object `x.html`.
      "x%2Ehtml",
      "x.json%2F..%2Fy.html",
      "bad%E0%A4%A",
    ]) {
      const response = await onRequest({ request: get(`/r2/${OURS}/${key}`), env });
      expect(response.status, key).toBe(404);
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it("sends nosniff and a sandboxing CSP on every proxied response", async () => {
    const ok = await onRequest({ request: get(`/r2/${OURS}/runs/a/0.glb`), env });
    bucket({ status: 404 }, "<html><script>alert(1)</script></html>");
    const missing = await onRequest({ request: get(`/r2/${OURS}/runs/a/1.glb`), env });
    const refused = await onRequest({ request: get(`/r2/${THEIRS}/x.json`), env });
    for (const response of [ok, missing, refused]) {
      expect(response.headers.get("X-Content-Type-Options")).toBe("nosniff");
      expect(response.headers.get("Content-Security-Policy")).toBe("sandbox; default-src 'none'");
    }
    // An upstream error passes its status and nothing of its body.
    expect(missing.status).toBe(404);
    expect(await missing.text()).toBe("");
    expect(missing.headers.get("Cache-Control")).toBe("no-store");
  });

  it("passes Range and conditional headers up, and the partial answer back", async () => {
    bucket(
      { status: 206, headers: { "Content-Range": "bytes 0-3/100", "Accept-Ranges": "bytes" } },
      "abcd",
    );
    const response = await onRequest({
      request: get(`/r2/${OURS}/runs/a/0.glb`, {
        Range: "bytes=0-3",
        "If-None-Match": '"etag"',
        "If-Modified-Since": "Tue, 01 Sep 2026 00:00:00 GMT",
        Cookie: "never=forwarded",
      }),
      env,
    });
    expect(response.status).toBe(206);
    expect(response.headers.get("Content-Range")).toBe("bytes 0-3/100");
    expect(await response.text()).toBe("abcd");
    const sent = new Headers(fetched()[0]?.init.headers);
    expect(sent.get("range")).toBe("bytes=0-3");
    expect(sent.get("if-none-match")).toBe('"etag"');
    expect(sent.get("if-modified-since")).toBe("Tue, 01 Sep 2026 00:00:00 GMT");
    expect(sent.get("cookie")).toBeNull();
  });

  it("keeps the cache lifetimes: a year for a generation's tiles, minutes for the rest", async () => {
    const tile = await onRequest({ request: get(`/r2/${OURS}/${GENERATION}/0.glb`), env });
    expect(tile.headers.get("Cache-Control")).toBe("public, max-age=31536000, immutable");
    const tileset = await onRequest({
      request: get(`/r2/${OURS}/${GENERATION}/tileset.json`),
      env,
    });
    expect(tileset.headers.get("Cache-Control")).toBe(
      "public, max-age=300, stale-while-revalidate=604800",
    );
    // A run's tiles outside a generation -- published before generations existed, under
    // names a Refine rewrote -- are not promised to anyone for a year.
    for (const key of ["runs/a/package/splat/0.glb", "runs/a/p0123/0.glb", "runs/a/0.glb"]) {
      const response = await onRequest({ request: get(`/r2/${OURS}/${key}`), env });
      expect(response.headers.get("Cache-Control"), key).toBe(
        "public, max-age=300, stale-while-revalidate=604800",
      );
    }
    bucket({ status: 304 }, null);
    const unchanged = await onRequest({ request: get(`/r2/${OURS}/sites/x/tileset.json`), env });
    expect(unchanged.status).toBe(304);
    expect(unchanged.headers.get("Cache-Control")).toBe(
      "public, max-age=300, stale-while-revalidate=604800",
    );
  });

  it("decides the lifetime on the key R2 reads: an encoded or upper-case .json is JSON", async () => {
    for (const key of ["runs/a/tileset%2Ejson", "runs/a/tileset.JSON", "runs/a/tileset%2EJson"]) {
      const response = await onRequest({ request: get(`/r2/${OURS}/${key}`), env });
      expect(response.status, key).toBe(200);
      expect(response.headers.get("Content-Type"), key).toBe("application/json");
      expect(response.headers.get("Cache-Control"), key).toBe(
        "public, max-age=300, stale-while-revalidate=604800",
      );
    }
    for (const { init } of fetched()) {
      expect(edgeTtl(init)).toEqual({ "200-299": 300, "300-599": -1 });
    }
  });

  it("keeps an edge copy of the object only, never of a 404 or a 304", async () => {
    await onRequest({ request: get(`/r2/${OURS}/${GENERATION}/0.glb`), env });
    const { init } = fetched()[0] ?? { init: {} };
    // A year for what the bucket answered with the object, and not cached otherwise: a
    // plain cacheTtl would have kept a tile's 404 at the edge for as long as the tile.
    expect(edgeTtl(init)).toEqual({ "200-299": 31536000, "300-599": -1 });
    expect((init as { cf?: Record<string, unknown> }).cf).not.toHaveProperty("cacheTtl");
  });

  it("answers only GET and HEAD", async () => {
    const response = await onRequest({
      request: get(`/r2/${OURS}/runs/a/0.glb`, {}, "POST"),
      env,
    });
    expect(response.status).toBe(405);
  });
});

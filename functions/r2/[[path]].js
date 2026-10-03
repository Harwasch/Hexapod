/**
 * Cloudflare Pages Function: the public tile bucket on the web app's own origin.
 *
 * `GET /r2/<host>/<key>` answers with `https://<host>/<key>`, where `<host>` is the public
 * bucket's managed URL (`pub-<32 hex>.r2.dev`). Why not the managed URL itself: r2.dev
 * answers over HTTP/1.1 only -- six connections per origin, so a scan's hundreds of tiles
 * queue behind each other -- with no Cache-Control, so a browser revalidates or refetches
 * tiles it already has. Here the browser gets HTTP/2 or 3 from the nearest Cloudflare edge,
 * one connection, and a cache lifetime: a year, immutable, for the binary content of a
 * published generation (`runs/<job>/p<generation>/...`, which every publish and every
 * sidecar attach writes afresh and nothing writes again: apps/api/app/services/published.py,
 * attach.py), and five minutes with a week of stale-while-revalidate for anything else --
 * JSON, `sites/<slug>/...`, which can be republished, and a run's copies from before
 * generations (`runs/<job>/package/splat/...`), which were, and beside which the backfill
 * workflows used to write their sidecars in place.
 *
 * **One host, and never the upstream's Content-Type.** Whatever this answers is served from
 * the web app's origin, which is the origin the console keeps the write token on
 * (`twin.settings.v1` in localStorage, apps/web/src/state/settings.ts). So this must not be
 * a way to put somebody else's bytes on that origin. It once was: it fetched *any*
 * `pub-<32 hex>.r2.dev` host -- every R2 bucket anyone has made public matches that
 * pattern, and creating one is free -- and passed the upstream Content-Type through, so
 * `/r2/pub-<attacker>.r2.dev/x.html` was an attacker's HTML page, scripts and all, running
 * on our origin with the token one `localStorage.getItem` away. Three things close that,
 * and each would on its own:
 *
 * - The host is pinned. `TILE_PROXY_HOST` (a Pages environment variable: wrangler.toml's
 *   `[vars]`, which deploy.yml fills in from the public bucket's URL) names the one bucket
 *   this fetches from; any other host is a 404 that never leaves the edge.
 * - The Content-Type is ours, decided by the key's extension from a short list of what the
 *   bucket actually serves (tilesets, tiles, splat sidecars, thumbnails). An extension not
 *   on it -- `.html`, `.svg`, `.js`, none at all -- is a 404 before anything is fetched.
 * - Every proxied response says `X-Content-Type-Options: nosniff` and
 *   `Content-Security-Policy: sandbox; default-src 'none'`, so even a mislabelled object
 *   opened as a page is an opaque-origin document that can run nothing and load nothing.
 *
 * `HEAD /r2/` answers 204 with `X-Tile-Proxy: <the pinned host>` when a host is pinned, and
 * the web app routes a URL here only when its host is *that* host
 * (apps/web/src/lib/tileProxy.ts). It used to answer `X-Tile-Proxy: 1`, and the web app
 * then sent every `pub-*.r2.dev` URL here -- which, once the host was pinned, was a 404 for
 * any tileset in another public bucket: one added through Add data, a demo or seed bucket,
 * a public bucket the deployment used before. Naming the host makes the probe say exactly
 * what this serves, so everything else keeps loading from where it is. With no host pinned
 * -- the variable unset, empty, or not an r2.dev hostname -- the probe is a 404 without the
 * header and every proxied path is a 404 too, so the web app keeps loading the managed URL
 * directly. That is the safe way to be misconfigured: slower tiles, not an open proxy.
 *
 * Pages runs a function only for its own routes (the generated _routes.json), so nothing
 * else on the site is affected.
 */

const MANAGED_HOST = /^pub-[0-9a-f]{32}\.r2\.dev$/;
const FORWARDED = ["range", "if-none-match", "if-modified-since"];
const YEAR_S = 31536000;
const SHORT_S = 300;
/**
 * A key inside a published generation: the only keys written exactly once. The same
 * pattern as `PUBLISHED_KEY` in apps/api/app/services/published.py, which
 * tests/test_worker_outputs.py holds this to.
 */
const PUBLISHED_KEY = /^runs\/[^/]+\/p[0-9a-f]{16}\//;

/**
 * What the public bucket holds, by extension, and the only types this will label anything
 * with. Binary formats Cesium reads as an ArrayBuffer are `application/octet-stream`: it
 * does not look at the type, and no browser renders that as a page.
 *
 * `.json` tilesets and catalogs, and the JSON sidecars (instances, skin, materials, telemetry,
 * rig, the streamed package's `lod-meta.json` and chunk `meta.json`); `.glb`/`.b3dm`/`.pnts`
 * tile content (a split object's and an inferred layer's tiles too); `.bin` glTF buffers and
 * the binary sidecars (`collision.bin`, `skin.bin`, `viewcones.bin`); `.emb`, `.f32`, `.u8`
 * the splat packages' sidecars (embeddings, float and byte arrays); `.ply`/`.spz` whole
 * splats; `.webp` the streamed package's planes and, with the other image types, site and
 * capture thumbnails.
 *
 * A Map rather than an object literal so that `x.constructor` is not an extension.
 *
 * @type {ReadonlyMap<string, string>}
 */
const CONTENT_TYPES = new Map([
  ["json", "application/json"],
  ["glb", "model/gltf-binary"],
  ["b3dm", "application/octet-stream"],
  ["pnts", "application/octet-stream"],
  ["bin", "application/octet-stream"],
  ["emb", "application/octet-stream"],
  ["f32", "application/octet-stream"],
  ["u8", "application/octet-stream"],
  ["ply", "application/octet-stream"],
  ["spz", "application/octet-stream"],
  ["webp", "image/webp"],
  ["jpg", "image/jpeg"],
  ["jpeg", "image/jpeg"],
  ["png", "image/png"],
]);

/** Sent on every proxied response, whatever its status. */
const SECURITY_HEADERS = Object.freeze({
  "X-Content-Type-Options": "nosniff",
  "Content-Security-Policy": "sandbox; default-src 'none'",
});

/**
 * The pinned host, or null where there is none. Accepted with or without a scheme or a
 * trailing slash, since "the bucket's public URL" is the value an operator has to hand; and
 * only ever an r2.dev hostname, because that is the only host the web app sends here -- any
 * other value would turn this into a proxy for a site nobody meant to proxy.
 *
 * @param {string | undefined} value
 * @returns {string | null}
 */
function pinnedHost(value) {
  const host = (value ?? "")
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\//, "")
    .replace(/\/+$/, "");
  return MANAGED_HOST.test(host) ? host : null;
}

/**
 * The key as R2 will read it, or null where it does not decode. Every decision about an
 * object -- its type, its lifetime -- is made on this, because this is the object R2 will
 * serve: `x%2Ehtml` is `x.html`, and `tileset%2Ejson` is `tileset.json`.
 *
 * @param {string} key the key as the browser encoded it
 * @returns {string | null}
 */
function decodedKey(key) {
  try {
    return decodeURIComponent(key);
  } catch {
    return null;
  }
}

/**
 * The Content-Type for a decoded key, or null when its extension is not on the list.
 *
 * @param {string} name the decoded key
 * @returns {string | null}
 */
function contentTypeFor(name) {
  const file = name.slice(name.lastIndexOf("/") + 1);
  const dot = file.lastIndexOf(".");
  if (dot <= 0) return null;
  return CONTENT_TYPES.get(file.slice(dot + 1).toLowerCase()) ?? null;
}

/**
 * Whether a decoded key may be cached for a year: inside a published generation, and not
 * JSON. It was every non-JSON key under `runs/`, and a phone's Refine rewrites a run's tiles
 * under the same names -- so the preview's tiles stayed cached, at the edge and in browsers,
 * under the new `tileset.json`. A publish now writes a generation of its own, and only a
 * generation's own keys are never rewritten.
 *
 * A generation's sidecars included. They were held short while the backfill workflows wrote
 * `instances.emb`, `collision.bin`, `sog/...` and `inferred/<layer>/...` beside a published
 * tileset in place, inside its generation, under the same names on every run (a year of an
 * old `instances.emb` under a new `instances.json` pairs one segmentation's embeddings with
 * another's objects). Every one of them now attaches through the API, which copies the
 * generation into a new one with the new files beside it (apps/api/app/services/attach.py),
 * so nothing writes a generation's sidecar twice either. Where those workflows did write in
 * place for good -- the legacy prefixes, `runs/<job>/package/splat/`, published before
 * generations -- is outside any generation, and short whatever the file.
 *
 * Its JSON test is the extension's, in any case, exactly as `contentTypeFor` reads it: it
 * once looked at the key as the browser encoded it, case and all, so
 * `runs/<id>/tileset%2Ejson` or `tileset.JSON` -- labelled JSON by the content type, which
 * decodes and lower-cases -- was cached for a year as if it were a tile, and a backfill's
 * rewrite of it never reached anyone.
 *
 * @param {string} name the decoded key
 */
function immutable(name) {
  return PUBLISHED_KEY.test(name) && !/\.json$/i.test(name);
}

/** @param {string} name the decoded key */
function cacheControl(name) {
  return immutable(name)
    ? `public, max-age=${YEAR_S}, immutable`
    : `public, max-age=${SHORT_S}, stale-while-revalidate=604800`;
}

function notFound() {
  return new Response("Not found", {
    status: 404,
    headers: { "Content-Type": "text/plain; charset=utf-8", ...SECURITY_HEADERS },
  });
}

/** @param {{ request: Request, env?: { TILE_PROXY_HOST?: string } }} context */
export async function onRequest({ request, env }) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response(null, { status: 405, headers: { Allow: "GET, HEAD" } });
  }
  const pinned = pinnedHost(env?.TILE_PROXY_HOST);
  // The raw path, as the browser encoded it: the key goes upstream byte for byte.
  const rest = new URL(request.url).pathname.replace(/^\/r2\/?/, "");
  if (rest === "") {
    // The probe. Without a pinned host it must not say "here", or the web app would send
    // every tile to a function that is going to refuse them all; with one, it says which,
    // so the web app sends only that bucket's URLs and leaves every other bucket alone.
    return pinned
      ? new Response(null, {
          status: 204,
          headers: { "X-Tile-Proxy": pinned, "Cache-Control": "no-store" },
        })
      : new Response(null, { status: 404, headers: { "Cache-Control": "no-store" } });
  }
  const slash = rest.indexOf("/");
  const host = slash > 0 ? rest.slice(0, slash).toLowerCase() : "";
  const key = slash > 0 ? rest.slice(slash + 1) : "";
  if (pinned === null || host !== pinned || key === "") return notFound();
  const name = decodedKey(key);
  const contentType = name === null ? null : contentTypeFor(name);
  if (name === null || contentType === null) return notFound();

  const headers = new Headers();
  for (const name of FORWARDED) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const lifetime = immutable(name) ? YEAR_S : SHORT_S;
  const upstream = await fetch(`https://${pinned}/${key}`, {
    method: request.method,
    headers,
    // Where the platform allows it, the edge keeps a copy too -- of an answer that is the
    // object, and of nothing else. A plain `cacheTtl` applies to every cacheable status,
    // 404 included, so a tile asked for a moment before its copy landed (or a sidecar a
    // backfill adds later) would have been "not found" at the edge for a year. A negative
    // TTL is Cloudflare's "do not cache".
    cf: { cacheEverything: true, cacheTtlByStatus: { "200-299": lifetime, "300-599": -1 } },
  });
  if (!upstream.ok && upstream.status !== 304) {
    // The status, and nothing of R2's error page: there is no body here worth serving
    // from this origin, and an error must not be cached as if it were the tile.
    return new Response(null, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: { "Cache-Control": "no-store", ...SECURITY_HEADERS },
    });
  }
  // Upstream's own headers ride along -- Content-Length, Content-Range, Accept-Ranges,
  // ETag, Last-Modified are what make a Range read and a revalidation work -- with the
  // Content-Type replaced by ours, whatever R2 had stored with the object.
  const response = new Response(upstream.body, upstream);
  response.headers.set("Content-Type", contentType);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) response.headers.set(name, value);
  response.headers.set("Cache-Control", cacheControl(name));
  return response;
}

/**
 * Cloudflare Pages Function: the public tile bucket on the web app's own origin.
 *
 * `GET /r2/<host>/<key>` answers with `https://<host>/<key>`, where `<host>` is the bucket's
 * managed public URL (`pub-<32 hex>.r2.dev`, the only host it will fetch). Why not the
 * managed URL itself: r2.dev answers over HTTP/1.1 only -- six connections per origin, so a
 * scan's hundreds of tiles queue behind each other -- with no Cache-Control, so a browser
 * revalidates or refetches tiles it already has. Here the browser gets HTTP/2 or 3 from the
 * nearest Cloudflare edge, one connection, and a cache lifetime: a year, immutable, for a
 * run's published package (`runs/<id>/...`, never rewritten), and five minutes with a week
 * of stale-while-revalidate for anything else (`sites/<slug>/...` can be republished).
 *
 * `HEAD /r2/` answers 204 with `X-Tile-Proxy: 1`: the web app routes tiles here only once it
 * has seen that (apps/web/src/lib/tileProxy.ts), so a deployment without this function
 * keeps working on the managed URL.
 *
 * Pages runs a function only for its own routes (the generated _routes.json), so nothing
 * else on the site is affected.
 */

const HOST = /^pub-[0-9a-f]{32}\.r2\.dev$/;
const FORWARDED = ["range", "if-none-match", "if-modified-since"];
const YEAR_S = 31536000;
const SHORT_S = 300;

/** @param {string} key */
function cacheControl(key) {
  return key.startsWith("runs/")
    ? `public, max-age=${YEAR_S}, immutable`
    : `public, max-age=${SHORT_S}, stale-while-revalidate=604800`;
}

/** @param {{ request: Request }} context */
export async function onRequest({ request }) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response(null, { status: 405, headers: { Allow: "GET, HEAD" } });
  }
  // The raw path, as the browser encoded it: the key goes upstream byte for byte.
  const rest = new URL(request.url).pathname.replace(/^\/r2\/?/, "");
  if (rest === "") {
    return new Response(null, {
      status: 204,
      headers: { "X-Tile-Proxy": "1", "Cache-Control": "no-store" },
    });
  }
  const slash = rest.indexOf("/");
  const host = slash > 0 ? rest.slice(0, slash) : "";
  const key = slash > 0 ? rest.slice(slash + 1) : "";
  if (!HOST.test(host) || key === "") return new Response("Not found", { status: 404 });

  const headers = new Headers();
  for (const name of FORWARDED) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const lifetime = key.startsWith("runs/") ? YEAR_S : SHORT_S;
  const upstream = await fetch(`https://${host}/${key}`, {
    method: request.method,
    headers,
    // Where the platform allows it, the edge keeps a copy too.
    cf: { cacheEverything: true, cacheTtl: lifetime },
  });
  const response = new Response(upstream.body, upstream);
  if (upstream.ok || upstream.status === 304) {
    response.headers.set("Cache-Control", cacheControl(key));
  } else {
    response.headers.set("Cache-Control", "no-store");
  }
  return response;
}

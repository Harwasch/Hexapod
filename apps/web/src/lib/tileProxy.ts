/**
 * Tiles from the public bucket through the web app's own origin, when it offers them there
 * (the Pages Function in `functions/r2/`): HTTP/2 or 3 and a cache lifetime, where the
 * bucket's managed r2.dev URL gives HTTP/1.1 and none.
 *
 * The function serves exactly one bucket, and says which by answering `HEAD /r2/` with
 * `X-Tile-Proxy: <that bucket's host>`. Only a URL on that host is rewritten. It used to
 * answer `X-Tile-Proxy: 1` and every `pub-*.r2.dev` URL was sent through it, which broke
 * every tileset in any other public bucket -- one added through Add data, a demo or seed
 * bucket, an older public bucket of this deployment -- as soon as the function was pinned
 * to one host and 404'd the rest. Those now load from their own URL, as they would with no
 * proxy at all.
 *
 * Asked once. Until it answers with a host, every URL is left as it is -- a local server, a
 * preview without functions, a function with no host pinned, or an answer this version does
 * not understand (the old `1` among them).
 */

const MANAGED_HOST = /^pub-[0-9a-f]{32}\.r2\.dev$/;

let served: Promise<string | null> | null = null;

/** The one bucket host the function serves, or null where nothing is served. */
function proxiedHost(): Promise<string | null> {
  served ??= fetch("/r2/", { method: "HEAD", cache: "no-store" })
    .then((response) => {
      if (!response.ok) return null;
      const host = (response.headers.get("X-Tile-Proxy") ?? "").trim().toLowerCase();
      // Only ever an r2.dev hostname: anything else is not an answer to this question.
      return MANAGED_HOST.test(host) ? host : null;
    })
    .catch(() => null);
  return served;
}

/** The URL to load a tileset (or any public-bucket object) from. */
export async function tileUrl(url: string): Promise<string> {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return url;
  }
  if (parsed.protocol !== "https:" || !MANAGED_HOST.test(parsed.hostname)) return url;
  if ((await proxiedHost()) !== parsed.hostname) return url;
  return `${location.origin}/r2/${parsed.hostname}${parsed.pathname}${parsed.search}`;
}

/** For tests: forget the answer. */
export function resetTileProxy(): void {
  served = null;
}

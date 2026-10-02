/**
 * Tiles from the public bucket through the web app's own origin, when it offers them there
 * (the Pages Function in `functions/r2/`): HTTP/2 or 3 and a cache lifetime, where the
 * bucket's managed r2.dev URL gives HTTP/1.1 and none. The function says it is there by
 * answering `HEAD /r2/` with `X-Tile-Proxy: 1`; asked once, and until it answers so, every
 * URL is left as it is -- a local server, a preview without functions, or another host.
 */

const MANAGED_HOST = /^pub-[0-9a-f]{32}\.r2\.dev$/;

let available: Promise<boolean> | null = null;

function proxyAvailable(): Promise<boolean> {
  available ??= fetch("/r2/", { method: "HEAD", cache: "no-store" })
    .then((response) => response.ok && response.headers.get("X-Tile-Proxy") === "1")
    .catch(() => false);
  return available;
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
  if (!(await proxyAvailable())) return url;
  return `${location.origin}/r2/${parsed.hostname}${parsed.pathname}${parsed.search}`;
}

/** For tests: forget the answer. */
export function resetTileProxy(): void {
  available = null;
}

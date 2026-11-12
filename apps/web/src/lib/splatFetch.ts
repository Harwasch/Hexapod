/** Fetch a scan resource, resolving private live files without a three-origin redirect.
 * Same-origin API requests follow their 302 normally. With a separate API origin,
 * resolve first so the bucket sees the web origin allowed by its existing CORS rule.
 */
export async function fetchSplatResource(url: string, init?: RequestInit): Promise<Response> {
  const target = new URL(url, location.href);
  if (
    target.origin !== location.origin &&
    /^\/api\/v1\/(captures|jobs)\/[^/]+\/live\/tiles\/\d+\//.test(target.pathname)
  ) {
    const resolver = new URL(target);
    resolver.searchParams.set("resolve", "true");
    const response = await fetch(resolver.href, init);
    if (!response.ok) return response;
    const value: unknown = await response.json();
    if (!value || typeof value !== "object" || !("url" in value) || typeof value.url !== "string") {
      throw new Error("The live tile URL is missing.");
    }
    const signed = new URL(value.url);
    if (signed.protocol !== "https:" && signed.protocol !== "http:") {
      throw new Error("The live tile URL is invalid.");
    }
    return fetch(signed.href, init);
  }
  return fetch(url, init);
}

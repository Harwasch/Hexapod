import { afterEach, expect, it, vi } from "vitest";

import { fetchSplatResource } from "@/lib/splatFetch";

afterEach(() => vi.unstubAllGlobals());

it.each(["tileset.json", "splat_0.glb", "collision.bin"])(
  "resolves cross-origin live %s then fetches the bucket directly with the same abort signal",
  async (file) => {
    const api = `https://api.example/api/v1/captures/c/live/tiles/30/${file}`;
    const signed = `https://private.example/${file}?signed=1`;
    const body = new Response("tile bytes");
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ url: signed }))
      .mockResolvedValueOnce(body);
    vi.stubGlobal("fetch", fetch);
    const signal = new AbortController().signal;
    expect(await fetchSplatResource(api, { signal })).toBe(body);
    expect(fetch.mock.calls).toEqual([
      [`${api}?resolve=true`, { signal }],
      [signed, { signal }],
    ]);
  },
);

it("allows same-origin live redirects and ordinary published tile URLs to use fetch directly", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response("tile"));
  vi.stubGlobal("fetch", fetch);
  const live = "/api/v1/captures/c/live/tiles/30/tileset.json";
  const published = "https://tiles.example/scan/tileset.json";
  await fetchSplatResource(live);
  await fetchSplatResource(published);
  expect(fetch.mock.calls).toEqual([
    [live, undefined],
    [published, undefined],
  ]);
});

it("propagates missing files and rejects malformed signing responses", async () => {
  const missing = new Response(null, { status: 404 });
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(missing)
    .mockResolvedValueOnce(Response.json({ url: "data:text/plain,bad" }));
  vi.stubGlobal("fetch", fetch);
  const api = "https://api.example/api/v1/jobs/r/live/tiles/30/root.glb";
  expect(await fetchSplatResource(api)).toBe(missing);
  await expect(fetchSplatResource(api)).rejects.toThrow("invalid");
  expect(fetch).toHaveBeenCalledTimes(2);
});

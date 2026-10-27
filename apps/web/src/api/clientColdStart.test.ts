import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it("waits out a cold API before returning the catalog", async () => {
  vi.useFakeTimers();
  const transport = vi
    .fn()
    .mockResolvedValueOnce(new Response(null, { status: 503 }))
    .mockResolvedValueOnce(Response.json([]));
  vi.stubGlobal("fetch", transport);
  const { api, unwrap } = await import("./client");
  const result = unwrap(api.GET("/api/v1/sites", { baseUrl: "https://api.invalid" }));
  const outcome = result.catch((error: unknown) => error);
  await vi.advanceTimersByTimeAsync(1000);
  expect(transport).toHaveBeenCalledTimes(2);
  expect(await outcome).toEqual([]);
});

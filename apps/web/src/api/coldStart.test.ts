import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchApi, API_RETRY_BUDGET_MS, API_ATTEMPT_TIMEOUT_MS } from "./coldStart";
import { apiWaking } from "./waking";

const url = "https://api.invalid/api/v1/sites";
const transport = vi.fn<typeof fetch>();

beforeEach(async () => {
  vi.useFakeTimers();
  transport.mockReset();
  vi.stubGlobal("fetch", transport);
  // A healthy response starts each test with the API available, including after an outage.
  transport.mockResolvedValueOnce(new Response());
  await fetchApi(url);
  transport.mockReset();
});
afterEach(() => {
  expect(apiWaking.getSnapshot()).toBe(false);
  expect(vi.getTimerCount()).toBe(0);
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("cold-start transport", () => {
  it.each(["network", "timeout", 503] as const)(
    "makes one quiet recovery attempt after exhausting the %s budget, then rearms on success",
    async (failure) => {
      transport.mockImplementation(() => {
        if (failure === 503) return Promise.resolve(new Response(null, { status: 503 }));
        return Promise.reject(
          failure === "network"
            ? new TypeError("offline")
            : new DOMException("timeout", "TimeoutError"),
        );
      });
      const first = fetchApi(url).catch((error: unknown) => error);
      await vi.advanceTimersByTimeAsync(API_RETRY_BUDGET_MS);
      await first;
      transport.mockClear();
      const controller = new AbortController();
      const recovery = fetchApi(url, { signal: controller.signal }).catch(
        (error: unknown) => error,
      );
      try {
        await vi.advanceTimersByTimeAsync(8000);
        expect(transport).toHaveBeenCalledTimes(1);
        expect(apiWaking.getSnapshot()).toBe(false);
      } finally {
        controller.abort();
        await recovery;
      }

      transport.mockResolvedValueOnce(new Response());
      await fetchApi(url);
      transport.mockClear();
      transport
        .mockResolvedValueOnce(new Response(null, { status: 503 }))
        .mockResolvedValueOnce(new Response());
      const laterSleep = fetchApi(url);
      await vi.advanceTimersByTimeAsync(0);
      expect(apiWaking.getSnapshot()).toBe(true);
      await vi.advanceTimersByTimeAsync(1000);
      expect((await laterSleep).ok).toBe(true);
      expect(transport).toHaveBeenCalledTimes(2);
    },
  );

  it("clears the waking note when one of several staggered requests gives up", async () => {
    transport.mockRejectedValue(new TypeError("offline"));
    const controller = new AbortController();
    const first = fetchApi(url).catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(5000);
    const second = fetchApi(url, { signal: controller.signal }).catch((error: unknown) => error);
    try {
      await vi.advanceTimersByTimeAsync(API_RETRY_BUDGET_MS - 5000);
      await first;
      expect(apiWaking.getSnapshot()).toBe(false);
      const calls = transport.mock.calls.length;
      await vi.advanceTimersByTimeAsync(8000);
      expect(transport).toHaveBeenCalledTimes(calls);
    } finally {
      controller.abort();
      await second;
    }
  });

  it("bounds a hung recovery read by one attempt without a waking note", async () => {
    transport.mockRejectedValue(new TypeError("offline"));
    const first = fetchApi(url).catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(API_RETRY_BUDGET_MS);
    await first;
    transport.mockReset();
    transport.mockImplementation(() => new Promise(() => undefined));
    const recovery = fetchApi(url).catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(API_ATTEMPT_TIMEOUT_MS);
    expect(await recovery).toMatchObject({ name: "TimeoutError" });
    expect(transport).toHaveBeenCalledTimes(1);
    expect(apiWaking.getSnapshot()).toBe(false);
    expect(transport.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
  });

  it.each(["network", 502, 503, 504, "timeout"] as const)(
    "backs off 1/2/4/8/8 s on %s",
    async (failure) => {
      transport.mockImplementation(() => {
        if (failure === "network") return Promise.reject(new TypeError("offline"));
        if (failure === "timeout")
          return Promise.reject(new DOMException("timeout", "TimeoutError"));
        return Promise.resolve(new Response(null, { status: failure }));
      });
      const changed = vi.fn();
      const unsubscribe = apiWaking.subscribe(changed);
      const pending = fetchApi(url);
      expect(apiWaking.getSnapshot()).toBe(false);
      await vi.advanceTimersByTimeAsync(0);
      expect(apiWaking.getSnapshot()).toBe(true);
      for (const [index, delay] of [1000, 2000, 4000, 8000, 8000].entries()) {
        await vi.advanceTimersByTimeAsync(delay - 1);
        expect(transport).toHaveBeenCalledTimes(index + 1);
        if (index === 4) transport.mockResolvedValueOnce(Response.json([]));
        await vi.advanceTimersByTimeAsync(1);
        expect(transport).toHaveBeenCalledTimes(index + 2);
      }
      expect((await pending).ok).toBe(true);
      expect(apiWaking.getSnapshot()).toBe(false);
      expect(changed).toHaveBeenCalledTimes(2);
      unsubscribe();
    },
  );

  it("stops at 45 s and surfaces the last network error", async () => {
    let last: Error;
    transport.mockImplementation(() => {
      last = new TypeError(`failure ${String(transport.mock.calls.length)}`);
      return Promise.reject(last);
    });
    const result = fetchApi(url).catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(API_RETRY_BUDGET_MS - 1);
    expect(apiWaking.getSnapshot()).toBe(true);
    await vi.advanceTimersByTimeAsync(1);
    expect(await result).toBe(last!);
    expect(transport).toHaveBeenCalledTimes(8); // 0, 1, 3, 7, 15, 23, 31, 39 s
  });

  it("preserves the final HTTP response body for the client's error handling", async () => {
    transport.mockImplementation(() =>
      Promise.resolve(Response.json({ detail: "asleep" }, { status: 503 })),
    );
    const pending = fetchApi(url);
    await vi.advanceTimersByTimeAsync(API_RETRY_BUDGET_MS);
    const response = await pending;
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ detail: "asleep" });
  });

  it.each(["POST", "PUT", "PATCH", "DELETE"])("never retries %s", async (method) => {
    transport.mockRejectedValue(new TypeError("offline"));
    await expect(fetchApi(url, { method, body: "data" })).rejects.toThrow("offline");
    expect(transport).toHaveBeenCalledTimes(1);
    transport.mockResolvedValue(new Response(null, { status: 503 }));
    expect((await fetchApi(url, { method })).status).toBe(503);
    expect(transport).toHaveBeenCalledTimes(2);
  });

  it.each(["HEAD", "OPTIONS"])("retries %s and preserves request headers", async (method) => {
    transport
      .mockResolvedValueOnce(new Response(null, { status: 503 }))
      .mockResolvedValueOnce(new Response());
    const pending = fetchApi(
      new Request(url, { method, headers: { Authorization: "Bearer test" } }),
    );
    await vi.advanceTimersByTimeAsync(1000);
    expect((await pending).ok).toBe(true);
    expect(transport).toHaveBeenCalledTimes(2);
    const request = transport.mock.calls[1]?.[0] as Request;
    expect(request.method).toBe(method);
    expect(request.headers.get("Authorization")).toBe("Bearer test");
  });

  it.each([400, 401, 403, 404, 429, 500])("does not retry HTTP %s", async (status) => {
    transport.mockResolvedValue(new Response(null, { status }));
    expect((await fetchApi(url)).status).toBe(status);
    expect(transport).toHaveBeenCalledTimes(1);
  });

  it.each(["before", "backoff", "attempt"])("stops when aborted during %s", async (when) => {
    const controller = new AbortController();
    if (when === "before") controller.abort();
    transport
      .mockRejectedValueOnce(new TypeError("offline"))
      .mockImplementation(() => new Promise(() => undefined));
    const result = fetchApi(url, { signal: controller.signal }).catch((error: unknown) => error);
    if (when === "backoff") await vi.advanceTimersByTimeAsync(0);
    if (when === "attempt") await vi.advanceTimersByTimeAsync(1000);
    controller.abort();
    expect(await result).toBe(controller.signal.reason);
    expect(transport).toHaveBeenCalledTimes(when === "before" ? 0 : when === "backoff" ? 1 : 2);
  });

  it("honours a signal carried by a Request", async () => {
    const controller = new AbortController();
    transport.mockRejectedValue(new TypeError("offline"));
    const result = fetchApi(new Request(url, { signal: controller.signal })).catch(
      (error: unknown) => error,
    );
    await vi.advanceTimersByTimeAsync(1000);
    controller.abort();
    expect(await result).toBe(controller.signal.reason);
    expect(transport).toHaveBeenCalledTimes(2);
  });

  it("retries a hung attempt and bounds hung requests by the total budget", async () => {
    transport.mockImplementation(() => new Promise(() => undefined));
    const result = fetchApi(url).catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(API_ATTEMPT_TIMEOUT_MS);
    expect(apiWaking.getSnapshot()).toBe(true);
    expect(transport.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
    await vi.advanceTimersByTimeAsync(1000);
    expect(transport).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(API_RETRY_BUDGET_MS - API_ATTEMPT_TIMEOUT_MS - 1000);
    expect(await result).toMatchObject({ name: "TimeoutError" });
  });

  it("keeps waking true until every concurrent retry finishes", async () => {
    transport
      .mockRejectedValueOnce(new TypeError("offline"))
      .mockRejectedValueOnce(new TypeError("offline"))
      .mockResolvedValueOnce(new Response())
      .mockRejectedValueOnce(new TypeError("offline"))
      .mockResolvedValueOnce(new Response());
    const first = fetchApi(url);
    const second = fetchApi(url);
    await vi.advanceTimersByTimeAsync(1000);
    await first;
    expect(apiWaking.getSnapshot()).toBe(true);
    await vi.advanceTimersByTimeAsync(2000);
    await second;
  });
});

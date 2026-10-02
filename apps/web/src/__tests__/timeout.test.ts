import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  fetchJsonUnlessStalled,
  isTimeout,
  StallError,
  TimeoutError,
  withTimeout,
} from "@/lib/timeout";

describe("withTimeout", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("passes a result that beats the deadline straight through", async () => {
    await expect(withTimeout(Promise.resolve(7), 1_000)).resolves.toBe(7);
  });

  it("passes a failure through untouched, so its shape still says what went wrong", async () => {
    // Cesium's request errors are not Error instances; ion.ts reads their status code.
    const refused = { statusCode: 401 };
    await expect(withTimeout(Promise.reject(refused as unknown as Error), 1_000)).rejects.toBe(
      refused,
    );
  });

  it("rejects at the deadline, names what stalled and aborts what can be aborted", async () => {
    const controller = new AbortController();
    const stalled = withTimeout(new Promise<never>(() => undefined), 8_000, {
      what: "The site's details",
      controller,
    });
    const settled = expect(stalled).rejects.toThrow("The site's details did not answer within 8 s");
    await vi.advanceTimersByTimeAsync(8_000);
    await settled;
    await expect(stalled).rejects.toSatisfy(isTimeout);
    expect(controller.signal.aborted).toBe(true);
    expect(controller.signal.reason).toBeInstanceOf(TimeoutError);
  });

  it("reports the deadline, not the abort it causes in work that listens for one", async () => {
    const controller = new AbortController();
    const listening = new Promise<never>((_, reject) =>
      controller.signal.addEventListener("abort", () => reject(new Error("aborted"))),
    );
    const settled = expect(withTimeout(listening, 2_000, { controller })).rejects.toBeInstanceOf(
      TimeoutError,
    );
    await vi.advanceTimersByTimeAsync(2_000);
    await settled;
  });

  it("hands a result that arrives after the deadline to onLate, to be disposed of", async () => {
    let finish: (value: { destroyed: boolean }) => void = () => undefined;
    const work = new Promise<{ destroyed: boolean }>((resolve) => (finish = resolve));
    const onLate = vi.fn((tileset: { destroyed: boolean }) => (tileset.destroyed = true));
    const late = withTimeout(work, 1_000, { onLate });
    const settled = expect(late).rejects.toBeInstanceOf(TimeoutError);
    await vi.advanceTimersByTimeAsync(1_000);
    await settled;
    const tileset = { destroyed: false };
    finish(tileset);
    await vi.advanceTimersByTimeAsync(0);
    expect(onLate).toHaveBeenCalledOnce();
    expect(tileset.destroyed).toBe(true);
  });

  it("leaves nothing pending once the work has settled", async () => {
    await withTimeout(Promise.resolve("done"), 5_000);
    expect(vi.getTimerCount()).toBe(0);
  });
});

/**
 * A response whose body arrives in `chunks`, one every `everyMs` (fake timers), or goes quiet
 * after `stopAfter` of them; the fetch's signal aborts the body as a real one would.
 */
function trickle(text: string, chunks: number, everyMs: number, stopAfter = chunks) {
  const encoded = new TextEncoder().encode(text);
  const size = Math.ceil(encoded.length / chunks);
  return vi.fn((_url: string, init?: RequestInit) => {
    let sent = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        return new Promise<void>((resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
          if (sent >= stopAfter) return; // the link goes quiet
          setTimeout(() => {
            const chunk = encoded.slice(sent * size, (sent + 1) * size);
            sent += 1;
            if (chunk.length > 0) controller.enqueue(chunk);
            if (sent >= chunks) controller.close();
            resolve();
          }, everyMs);
        });
      },
    });
    return Promise.resolve(new Response(body, { status: 200 }));
  });
}

describe("fetchJsonUnlessStalled", () => {
  const tileset = JSON.stringify({
    asset: { version: "1.1" },
    root: { padding: "x".repeat(4000) },
  });

  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("finishes a download that takes far longer than the stall timeout, while bytes come", async () => {
    // Forty seconds for the whole file, a chunk every ten: a slow phone link, not a stall.
    vi.stubGlobal("fetch", trickle(tileset, 4, 10_000));
    const json = fetchJsonUnlessStalled("https://bucket.test/tileset.json", {
      stallMs: 15_000,
      what: "The yard",
    });
    await vi.advanceTimersByTimeAsync(45_000);
    await expect(json).resolves.toMatchObject({ asset: { version: "1.1" } });
  });

  it("fails once no byte has come for the stall timeout, and says so", async () => {
    vi.stubGlobal("fetch", trickle(tileset, 4, 10_000, 2));
    const json = fetchJsonUnlessStalled("https://bucket.test/tileset.json", {
      stallMs: 15_000,
      what: "The yard",
    });
    const failed = expect(json).rejects.toThrow("The yard stopped arriving for 15 s");
    await vi.advanceTimersByTimeAsync(20_000 + 15_000);
    await failed;
    await expect(json).rejects.toBeInstanceOf(StallError);
    await expect(json).rejects.toSatisfy(isTimeout);
  });

  it("says a refusal's status, in the shape ion.ts reads", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response("", { status: 404 }))),
    );
    await expect(
      fetchJsonUnlessStalled("https://bucket.test/tileset.json", { stallMs: 15_000 }),
    ).rejects.toThrow("Request failed with status 404");
  });
});

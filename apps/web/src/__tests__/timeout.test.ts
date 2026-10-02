import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { isTimeout, TimeoutError, withTimeout } from "@/lib/timeout";

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

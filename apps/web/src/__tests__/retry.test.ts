import { describe, expect, it, vi } from "vitest";

import { withRetry } from "@/lib/retry";

describe("withRetry", () => {
  it("tries again after a passing failure, waiting longer each time", async () => {
    vi.useFakeTimers();
    let calls = 0;
    const result = withRetry(
      () => {
        calls += 1;
        return calls < 3 ? Promise.reject(new Error("503")) : Promise.resolve("ok");
      },
      { firstDelayMs: 100 },
    );
    await vi.advanceTimersByTimeAsync(100);
    expect(calls).toBe(2);
    await vi.advanceTimersByTimeAsync(200);
    await expect(result).resolves.toBe("ok");
    expect(calls).toBe(3);
    vi.useRealTimers();
  });

  it("gives up at once on a failure waiting cannot fix, and after the last attempt", async () => {
    let calls = 0;
    const refused = withRetry(
      () => {
        calls += 1;
        return Promise.reject(new Error("401"));
      },
      { permanent: (error) => String(error).includes("401") },
    );
    await expect(refused).rejects.toThrow("401");
    expect(calls).toBe(1);
    calls = 0;
    const flaky = withRetry(
      () => {
        calls += 1;
        return Promise.reject(new Error("503"));
      },
      { attempts: 2, firstDelayMs: 1 },
    );
    await expect(flaky).rejects.toThrow("503");
    expect(calls).toBe(2);
  });
});

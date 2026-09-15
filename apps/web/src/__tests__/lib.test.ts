import { describe, expect, it, vi } from "vitest";

import { Emitter } from "@/lib/emitter";
import { formatDate, representationLabel, sourceLabel } from "@/lib/format";
import { throttle } from "@/lib/throttle";
import { recentSpans, recordSpan } from "@/lib/timing";

describe("lib", () => {
  it("emitter delivers typed events and unsubscribes", () => {
    const emitter = new Emitter<{ ping: number }>();
    const listener = vi.fn();
    const off = emitter.on("ping", listener);
    emitter.emit("ping", 1);
    off();
    emitter.emit("ping", 2);
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith(1);
  });

  it("throttle coalesces rapid calls", () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const throttled = throttle(fn, 50);
    throttled(1);
    throttled(2);
    throttled(3);
    expect(fn).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(60);
    expect(fn).toHaveBeenCalledTimes(2);
    expect(fn).toHaveBeenLastCalledWith(3);
    vi.useRealTimers();
  });

  it("formats labels", () => {
    expect(representationLabel("gaussian-splat")).toBe("Splat");
    expect(sourceLabel({ type: "cesium-ion", assetId: 5 })).toBe("Cesium ion asset 5");
    expect(
      sourceLabel({ type: "xyz", urlTemplate: "https://tiles.example.com/{z}/{x}/{y}.png" }),
    ).toBe("XYZ tiles · tiles.example.com");
    expect(formatDate(null)).toBe("—");
    expect(formatDate("2021-01-01T00:00:00Z")).toMatch(/2021/);
  });

  it("records timing spans", () => {
    recordSpan("api", 12, { path: "/x" });
    expect(recentSpans().at(-1)).toMatchObject({ name: "api", durationMs: 12 });
  });
});

import { describe, expect, it, vi } from "vitest";

import { Emitter } from "@/lib/emitter";
import { formatDate, representationLabel, sourceLabel } from "@/lib/format";
import { throttle, throttleProgress } from "@/lib/throttle";
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

  it("throttle flushes a waiting call at once, and only one", () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const throttled = throttle(fn, 250);
    throttled("loading 9");
    throttled("loading 4");
    throttled("done");
    expect(fn).toHaveBeenCalledTimes(1);
    // An end state is shown now, not at the end of the window.
    throttled.flush();
    expect(fn).toHaveBeenCalledTimes(2);
    expect(fn).toHaveBeenLastCalledWith("done");
    vi.advanceTimersByTime(300);
    expect(fn).toHaveBeenCalledTimes(2);
    // Nothing waiting: a flush calls nothing.
    throttled.flush();
    expect(fn).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it("loading progress reaches the UI four times a second, and the end of loading at once", () => {
    vi.useFakeTimers();
    const reports: [number, number][] = [];
    const progress = throttleProgress((pending, processing) => reports.push([pending, processing]));
    // Cesium reports once a rendered frame: 60 frames of loading.
    for (let frame = 0; frame < 60; frame++) {
      progress(60 - frame, 2);
      vi.advanceTimersByTime(1000 / 60);
    }
    expect(reports.length).toBeGreaterThanOrEqual(4);
    expect(reports.length).toBeLessThanOrEqual(5);
    // Done: shown now, not a quarter of a second later.
    progress(0, 0);
    expect(reports.at(-1)).toEqual([0, 0]);
    const shown = reports.length;
    vi.advanceTimersByTime(1000);
    expect(reports).toHaveLength(shown);
    // A tileset unloaded mid-load: nothing arrives after it.
    progress(5, 1);
    progress(4, 1);
    progress.cancel();
    vi.advanceTimersByTime(1000);
    expect(reports.at(-1)).toEqual([5, 1]);
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

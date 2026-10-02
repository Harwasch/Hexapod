/**
 * The overlay's frame meter (cesium/scanView/frameMeter.ts): a reading is the latest gesture's
 * motion frames -- their rate, the 95th percentile interval where a fast turn's hitches show,
 * and the median draw-call time -- kept after the camera stops, and a short nudge or a gap
 * between gestures never blends two gestures into one.
 */

import { describe, expect, it } from "vitest";

import {
  FrameMeter,
  GAP_MS,
  LIVE_MS,
  MIN_INTERVALS,
  WINDOW_MS,
} from "@/cesium/scanView/frameMeter";

/** `count` frames `everyMs` apart from `start`, each taking `cpuMs` to draw. */
function frames(
  meter: FrameMeter,
  start: number,
  count: number,
  everyMs: number,
  cpuMs = 2,
): number {
  for (let i = 0; i < count; i++) meter.record(start + i * everyMs, cpuMs);
  return start + (count - 1) * everyMs;
}

describe("the overlay's frame meter", () => {
  it("has no reading before a gesture", () => {
    const meter = new FrameMeter();
    expect(meter.reading(0)).toBeNull();
    // A nudge: fewer intervals than a reading needs.
    frames(meter, 0, MIN_INTERVALS, 16);
    expect(meter.reading(200)).toBeNull();
  });

  it("reads a steady gesture's rate, and says when it is over", () => {
    const meter = new FrameMeter();
    const last = frames(meter, 1000, 61, 1000 / 60, 3);
    const live = meter.reading(last + 10);
    expect(live?.fps).toBeCloseTo(60, 5);
    expect(live?.p95Ms).toBeCloseTo(1000 / 60, 5);
    expect(live?.cpuMs).toBe(3);
    expect(live?.live).toBe(true);
    expect(meter.reading(last + LIVE_MS + 1)?.live).toBe(false);
    expect(meter.reading(last + 60_000)?.fps).toBeCloseTo(60, 5);
  });

  it("shows a hitch in the p95 that the average hides", () => {
    const meter = new FrameMeter();
    let t = frames(meter, 0, 40, 16);
    // Two 100 ms hitches in a fast turn.
    meter.record((t += 100), 2);
    t = frames(meter, t + 16, 10, 16);
    meter.record((t += 100), 2);
    const reading = meter.reading(t);
    expect(reading?.fps).toBeGreaterThan(40);
    expect(reading?.p95Ms).toBeGreaterThanOrEqual(16);
    expect(reading?.p95Ms).toBeLessThan(100);
    // One hitch in twenty intervals is the p95.
    const short = new FrameMeter();
    let u = frames(short, 0, 19, 16);
    short.record((u += 100), 2);
    expect(short.reading(u)?.p95Ms).toBe(100);
  });

  it("reads the latest gesture alone, past a nudge since", () => {
    const meter = new FrameMeter();
    // A slow gesture, a rest, a fast one, a rest, then two frames (a nudge).
    let t = frames(meter, 0, 30, 50, 9);
    t = frames(meter, t + GAP_MS + 1, 30, 16, 1);
    const fast = meter.reading(t);
    frames(meter, t + GAP_MS + 1, 2, 16, 5);
    const reading = meter.reading(t + 2 * GAP_MS);
    expect(reading).toEqual({ ...fast, live: false });
    expect(reading?.fps).toBeCloseTo(62.5, 5);
    expect(reading?.cpuMs).toBe(1);
  });

  it("covers the last couple of seconds of a long gesture", () => {
    const meter = new FrameMeter();
    // Five seconds at 20 fps, then two at 60: the reading is the recent 60.
    let t = frames(meter, 0, 100, 50);
    t = frames(meter, t + 16, 120, 1000 / 60);
    const reading = meter.reading(t);
    expect(reading?.fps).toBeCloseTo(60, 0);
    expect(reading?.frames).toBeLessThanOrEqual(Math.ceil(WINDOW_MS / (1000 / 60)));
  });
});

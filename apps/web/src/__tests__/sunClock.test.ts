/**
 * "At capture"'s hold on the scene clock (cesium/sunClock.ts): the clock stands at the time
 * asked, still, and is given back exactly -- the time it would read had it never been held, at
 * its own rate, its own animate flag and step -- unless someone else set it meanwhile.
 */
import { Clock, ClockStep, JulianDate } from "cesium";
import { describe, expect, it } from "vitest";

import { holdClock } from "@/cesium/sunClock";

const iso = (t: JulianDate) => JulianDate.toIso8601(t, 3);

describe("the clock hold", () => {
  const start = JulianDate.fromIso8601("2026-10-20T12:00:00Z");
  const at = JulianDate.fromIso8601("2026-06-01T15:00:00Z");

  function running(multiplier = 1) {
    const clock = new Clock({ currentTime: start, shouldAnimate: true, multiplier });
    let wall = 1_000;
    return { clock, wall: () => wall, pass: (ms: number) => (wall += ms) };
  }

  it("stands the clock at the time asked, still", () => {
    const { clock, wall } = running();
    holdClock(clock, at, wall);
    expect(iso(clock.currentTime)).toBe("2026-06-01T15:00:00.000Z");
    expect(clock.shouldAnimate).toBe(false);
  });

  it("gives back the time the running clock would read had it never been held", () => {
    const { clock, wall, pass } = running();
    const release = holdClock(clock, at, wall);
    pass(42_500);
    release();
    expect(iso(clock.currentTime)).toBe("2026-10-20T12:00:42.500Z");
    expect(clock.shouldAnimate).toBe(true);
    expect(clock.clockStep).toBe(ClockStep.SYSTEM_CLOCK_MULTIPLIER);
  });

  it("at the clock's own rate", () => {
    const { clock, wall, pass } = running(60);
    const release = holdClock(clock, at, wall);
    pass(2_000);
    release();
    expect(iso(clock.currentTime)).toBe("2026-10-20T12:02:00.000Z");
  });

  it("gives a still clock back exactly where it stood", () => {
    const { clock, wall, pass } = running();
    clock.shouldAnimate = false;
    const release = holdClock(clock, at, wall);
    pass(10_000);
    release();
    expect(iso(clock.currentTime)).toBe("2026-10-20T12:00:00.000Z");
    expect(clock.shouldAnimate).toBe(false);
  });

  it("lets a time someone else set meanwhile stand", () => {
    const { clock, wall, pass } = running();
    const release = holdClock(clock, at, wall);
    clock.currentTime = JulianDate.fromIso8601("2027-01-01T00:00:00Z");
    pass(5_000);
    release();
    expect(iso(clock.currentTime)).toBe("2027-01-01T00:00:00.000Z");
    expect(clock.shouldAnimate).toBe(true);
  });

  it("puts a clock that tracked the system time back to tracking it", () => {
    const clock = new Clock({ clockStep: ClockStep.SYSTEM_CLOCK });
    const release = holdClock(clock, at);
    expect(clock.clockStep).toBe(ClockStep.SYSTEM_CLOCK_MULTIPLIER);
    release();
    expect(clock.clockStep).toBe(ClockStep.SYSTEM_CLOCK);
    expect(clock.shouldAnimate).toBe(true);
    expect(
      Math.abs(JulianDate.secondsDifference(clock.currentTime, JulianDate.now())),
    ).toBeLessThan(5);
  });

  it("is released once, however often it is asked", () => {
    const { clock, wall, pass } = running();
    const release = holdClock(clock, at, wall);
    pass(1_000);
    release();
    const after = iso(clock.currentTime);
    pass(1_000);
    release();
    expect(iso(clock.currentTime)).toBe(after);
  });
});

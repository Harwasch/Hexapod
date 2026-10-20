/**
 * "At capture" holds the scene clock at the shown date's capture time (ADR 0019 "Sunlight"):
 * the engine's sun follows `clock.currentTime` (and so do the sky, the atmosphere and the
 * relight), so a still clock at that moment is the sun the photographs were taken under.
 *
 * Releasing gives back the clock it found, exactly: the time it would read had it never been
 * held -- the held span added at the clock's own rate when it was running (multiplier x wall
 * seconds, `ClockStep.SYSTEM_CLOCK_MULTIPLIER`), nothing when it was still -- and its own
 * `shouldAnimate` and `clockStep`. If something else set the time meanwhile, that time stands.
 * A clock that tracks the system time (`ClockStep.SYSTEM_CLOCK`) goes back to tracking it.
 *
 * While held, nothing that runs on the scene clock moves: the Living Survey's wind stands where
 * it was (its own guard logs that the clock stopped), which the badge's moment is.
 */

import { ClockStep, JulianDate, type Clock } from "cesium";

export type HeldClock = Pick<Clock, "currentTime" | "shouldAnimate" | "clockStep" | "multiplier">;

/** Holds `clock` at `at`; returns the release, which is safe to call twice. */
export function holdClock(
  clock: HeldClock,
  at: JulianDate,
  wallMs: () => number = () => performance.now(),
): () => void {
  const saved = {
    time: JulianDate.clone(clock.currentTime),
    animate: clock.shouldAnimate,
    step: clock.clockStep,
    multiplier: clock.multiplier,
    wall: wallMs(),
  };
  const held = JulianDate.clone(at);
  clock.shouldAnimate = false;
  clock.currentTime = JulianDate.clone(held);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    const untouched = JulianDate.equals(clock.currentTime, held);
    if (untouched) {
      const ran = saved.animate && saved.step === ClockStep.SYSTEM_CLOCK_MULTIPLIER;
      const seconds = ran ? (saved.multiplier * (wallMs() - saved.wall)) / 1000 : 0;
      clock.currentTime = JulianDate.addSeconds(saved.time, seconds, new JulianDate());
    }
    clock.shouldAnimate = saved.animate;
    // Setting a time or the animate flag turns a system clock into a multiplier one; back.
    if (clock.clockStep !== saved.step) clock.clockStep = saved.step;
  };
}

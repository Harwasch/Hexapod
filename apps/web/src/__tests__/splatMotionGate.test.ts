import { describe, expect, it } from "vitest";

import { MAX_HOLD_MS, REST_MS, gateAt } from "@/cesium/splatMotionGate";

const never = Number.NEGATIVE_INFINITY;

describe("what splats may spend while the camera moves", () => {
  it("holds nothing at rest", () => {
    const state = gateAt(
      10_000,
      { moving: false, endedAt: 10_000 - REST_MS - 1, startedAt: 0 },
      { releaseAt: never },
    );
    expect(state).toEqual({ hold: false, release: false });
  });

  it("holds rebuilds while moving and for a moment after, so two gestures are one", () => {
    const moving = gateAt(
      1_000,
      { moving: true, endedAt: never, startedAt: 900 },
      { releaseAt: never },
    );
    expect(moving.hold).toBe(true);
    const justStopped = gateAt(
      1_000,
      { moving: false, endedAt: 1_000 - REST_MS / 2, startedAt: 500 },
      { releaseAt: never },
    );
    expect(justStopped.hold).toBe(true);
  });

  it("lets a rebuild through every MAX_HOLD_MS of continuous motion, for a short window", () => {
    const motion = { moving: true, endedAt: never, startedAt: 0 };
    expect(gateAt(MAX_HOLD_MS - 1, motion, { releaseAt: never }).hold).toBe(true);
    const due = gateAt(MAX_HOLD_MS, motion, { releaseAt: never });
    expect(due).toMatchObject({ hold: false, release: true });
    // The next frames stay open until the window closes, then the count starts again.
    const open = gateAt(MAX_HOLD_MS + 50, motion, { releaseAt: MAX_HOLD_MS });
    expect(open).toMatchObject({ hold: false, release: false });
    const closed = gateAt(MAX_HOLD_MS + 500, motion, { releaseAt: MAX_HOLD_MS });
    expect(closed.hold).toBe(true);
    expect(gateAt(2 * MAX_HOLD_MS, motion, { releaseAt: MAX_HOLD_MS })).toMatchObject({
      hold: false,
      release: true,
    });
  });
});

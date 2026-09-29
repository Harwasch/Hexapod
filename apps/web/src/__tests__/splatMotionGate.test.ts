import { describe, expect, it } from "vitest";

import { DECODE_INTERVAL_MS, MAX_HOLD_MS, REST_MS, gateAt } from "@/cesium/splatMotionGate";

const never = Number.NEGATIVE_INFINITY;

describe("what splats may spend while the camera moves", () => {
  it("caps nothing at rest", () => {
    const state = gateAt(
      10_000,
      { moving: false, endedAt: 10_000 - REST_MS - 1, startedAt: 0 },
      { releaseAt: never, decodeAt: never },
    );
    expect(state).toEqual({ hold: false, decodes: undefined, release: false });
  });

  it("holds rebuilds while moving and for a moment after, so two gestures are one", () => {
    const moving = gateAt(
      1_000,
      { moving: true, endedAt: never, startedAt: 900 },
      { releaseAt: never, decodeAt: never },
    );
    expect(moving.hold).toBe(true);
    const justStopped = gateAt(
      1_000,
      { moving: false, endedAt: 1_000 - REST_MS / 2, startedAt: 500 },
      { releaseAt: never, decodeAt: never },
    );
    expect(justStopped.hold).toBe(true);
  });

  it("lets one decode start per interval while moving", () => {
    const motion = { moving: true, endedAt: never, startedAt: 0 };
    expect(gateAt(1_000, motion, { releaseAt: never, decodeAt: never }).decodes).toBe(1);
    expect(gateAt(1_000, motion, { releaseAt: never, decodeAt: 1_000 - 10 }).decodes).toBe(0);
    expect(
      gateAt(1_000, motion, { releaseAt: never, decodeAt: 1_000 - DECODE_INTERVAL_MS }).decodes,
    ).toBe(1);
  });

  it("lets a rebuild through every MAX_HOLD_MS of continuous motion, for a short window", () => {
    const motion = { moving: true, endedAt: never, startedAt: 0 };
    expect(gateAt(MAX_HOLD_MS - 1, motion, { releaseAt: never, decodeAt: 0 }).hold).toBe(true);
    const due = gateAt(MAX_HOLD_MS, motion, { releaseAt: never, decodeAt: 0 });
    expect(due).toMatchObject({ hold: false, release: true });
    // The next frames stay open until the window closes, then the count starts again.
    const open = gateAt(MAX_HOLD_MS + 50, motion, { releaseAt: MAX_HOLD_MS, decodeAt: 0 });
    expect(open).toMatchObject({ hold: false, release: false });
    const closed = gateAt(MAX_HOLD_MS + 500, motion, { releaseAt: MAX_HOLD_MS, decodeAt: 0 });
    expect(closed.hold).toBe(true);
    expect(gateAt(2 * MAX_HOLD_MS, motion, { releaseAt: MAX_HOLD_MS, decodeAt: 0 })).toMatchObject({
      hold: false,
      release: true,
    });
  });
});

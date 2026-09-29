import { describe, expect, it } from "vitest";

import { WALK, step, type MoveInput, type MoveState, type MoveWorld } from "@/lib/firstPerson";
import type { Vec3 } from "@/lib/occupancy";

const idle: MoveInput = { forward: 0, right: 0, up: 0, sprint: false, jump: false };
const north = { heading: 0, pitch: 0 };

/** Flat ground at 0, with an optional box of height `h` for y in [ya, yb]. */
function world(box?: { ya: number; yb: number; h: number }, ledge?: number): MoveWorld {
  return {
    groundBelow: (_x, y, fromZ) => {
      let g = ledge !== undefined && y > ledge ? -2 : 0;
      if (box && y >= box.ya && y <= box.yb && box.h <= fromZ) g = Math.max(g, box.h);
      return g <= fromZ ? g : null;
    },
    sweep: (_from, to) => to,
  };
}

function run(
  state: MoveState,
  input: MoveInput,
  seconds: number,
  w: MoveWorld = world(),
  mode: "walk" | "fly" = "walk",
): MoveState[] {
  const frames: MoveState[] = [];
  for (let t = 0; t < seconds; t += 1 / 60) {
    state = step(state, input, north, mode, w, 1 / 60);
    frames.push(state);
  }
  return frames;
}

const standing = (y = 0): MoveState => ({
  position: [0, y, WALK.eyeHeight] as Vec3,
  velocity: [0, 0, 0],
  onGround: true,
});

describe("walking", () => {
  it("eases up to walking pace along the view, eye at eye height", () => {
    const frames = run(standing(), { ...idle, forward: 1 }, 2);
    const last = frames.at(-1);
    expect(last?.velocity[1]).toBeCloseTo(WALK.speed, 1);
    expect(last?.position[2]).toBeCloseTo(WALK.eyeHeight, 3);
    // Eased, not instant: the first frame is well under full speed.
    expect(frames[0]?.velocity[1] ?? 0).toBeLessThan(WALK.speed * 0.3);
  });

  it("goes slow with Ctrl", () => {
    const slow = run(standing(), { ...idle, forward: 1, slow: true }, 2).at(-1);
    expect(slow?.velocity[1]).toBeCloseTo(WALK.speed * WALK.slow, 1);
  });

  it("sprints with Shift, and stops smoothly when the key is let go", () => {
    const fast = run(standing(), { ...idle, forward: 1, sprint: true }, 2).at(-1);
    expect(fast?.velocity[1]).toBeCloseTo(WALK.speed * WALK.sprint, 1);
    const stopped = run(fast!, idle, 1).at(-1);
    expect(Math.abs(stopped?.velocity[1] ?? 1)).toBeLessThan(0.01);
  });

  it("jumps about 0.8 m and lands back at eye height", () => {
    const frames = run(standing(), { ...idle, jump: true }, 0.02).concat(
      run(run(standing(), { ...idle, jump: true }, 1 / 60).at(-1)!, idle, 1.5),
    );
    const peak = Math.max(...frames.map((f) => f.position[2]));
    expect(peak - WALK.eyeHeight).toBeGreaterThan(0.7);
    expect(peak - WALK.eyeHeight).toBeLessThan(0.9);
    const last = frames.at(-1);
    expect(last?.onGround).toBe(true);
    expect(last?.position[2]).toBeCloseTo(WALK.eyeHeight, 2);
  });

  it("walks up a step but not into a waist-high block", () => {
    const up = run(standing(), { ...idle, forward: 1 }, 3, world({ ya: 2, yb: 50, h: 0.3 })).at(-1);
    expect(up?.position[1]).toBeGreaterThan(3);
    expect(up?.position[2]).toBeCloseTo(0.3 + WALK.eyeHeight, 1);
    const blocked = run(standing(), { ...idle, forward: 1 }, 3, world({ ya: 2, yb: 50, h: 1 })).at(
      -1,
    );
    expect(blocked?.position[1]).toBeLessThan(2);
  });

  it("falls off a ledge and lands on the ground below", () => {
    const frames = run(standing(), { ...idle, forward: 1 }, 4, world(undefined, 3));
    expect(frames.some((f) => !f.onGround)).toBe(true);
    const last = frames.at(-1);
    expect(last?.onGround).toBe(true);
    expect(last?.position[2]).toBeCloseTo(-2 + WALK.eyeHeight, 1);
  });
});

describe("flying", () => {
  it("hovers without gravity, rises with up, and moves where it looks", () => {
    const hover = run(standing(), idle, 2, world(), "fly").at(-1);
    expect(hover?.position[2]).toBeCloseTo(WALK.eyeHeight, 5);
    const rise = run(standing(), { ...idle, up: 1 }, 1, world(), "fly").at(-1);
    expect(rise?.position[2] ?? 0).toBeGreaterThan(WALK.eyeHeight + 2);
    const climb = (() => {
      let s = standing();
      for (let i = 0; i < 60; i++) {
        s = step(
          s,
          { ...idle, forward: 1 },
          { heading: 0, pitch: Math.PI / 4 },
          "fly",
          world(),
          1 / 60,
        );
      }
      return s;
    })();
    expect(climb.position[2]).toBeGreaterThan(WALK.eyeHeight + 1);
  });
});

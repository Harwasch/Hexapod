import { describe, expect, it } from "vitest";

import { backToFront, sortBackToFront } from "@/lib/splatOrder";

describe("splat draw order", () => {
  it("draws the farthest first, whatever the direction", () => {
    const positions = new Float32Array([
      0,
      0,
      5, // 5 m
      -1,
      0,
      0, // 1 m
      0,
      30,
      0, // 30 m
      0,
      -0.01,
      0, // 1 cm
      100,
      0,
      0, // 100 m
    ]);
    expect([...backToFront(positions, 5, [0, 0, 0])]).toEqual([4, 2, 0, 1, 3]);
  });

  it("does not depend on where the camera looks, only where it is", () => {
    const positions = new Float32Array(3000).map(() => Math.random() * 20 - 10);
    const a = backToFront(positions, 1000, [1, 2, 3]);
    const distance = (i: number): number =>
      Math.hypot(
        (positions[i * 3] ?? 0) - 1,
        (positions[i * 3 + 1] ?? 0) - 2,
        (positions[i * 3 + 2] ?? 0) - 3,
      );
    // Non-increasing distance, within the key's resolution (0.03%).
    for (let i = 1; i < a.length; i++) {
      expect(distance(a[i] ?? 0)).toBeLessThanOrEqual(distance(a[i - 1] ?? 0) * 1.001 + 1e-9);
    }
    expect(new Set(a).size).toBe(1000);
  });

  it("leaves empty slots (NaN positions) out of the order", () => {
    // Slots 1 and 3 are empty: a freed range, a never-written slot.
    const positions = new Float32Array([0, 0, 1, NaN, NaN, NaN, 0, 0, 5, NaN, NaN, NaN, 0, 0, 3]);
    const order = backToFront(positions, 5, [0, 0, 0]);
    expect(Array.from(order)).toEqual([2, 4, 0]);
    expect(Array.from(backToFront(new Float32Array(6).fill(NaN), 2, [0, 0, 0]))).toEqual([]);
  });

  it("draws only shown slots, and says how near the nearest drawn splat is", () => {
    // Four written slots; the second is hidden (resident, not drawn).
    const positions = new Float32Array([0, 0, 1, 0, 0, 0.5, 0, 0, 4, 0, 0, 2]);
    const live = new Uint8Array([1, 0, 1, 1]);
    const { order, nearest } = sortBackToFront(positions, 4, [0, 0, 0], live);
    expect(Array.from(order)).toEqual([2, 3, 0]);
    expect(nearest).toBeCloseTo(1, 5);
    expect(sortBackToFront(positions, 4, [0, 0, 0], new Uint8Array(4)).nearest).toBe(
      Number.POSITIVE_INFINITY,
    );
  });
});

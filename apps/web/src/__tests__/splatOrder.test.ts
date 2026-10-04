import { describe, expect, it } from "vitest";

import { backToFront, sortBackToFront } from "@/lib/splatOrder";

/**
 * The order the sort used to give: a counting sort on a 16-bit key of ln(d²) over each sort's
 * own range (a `Math.log` per splat). The precision the bit-pattern key has to keep.
 */
function logKeyOrder(
  positions: Float32Array,
  count: number,
  eye: readonly [number, number, number],
): Uint32Array {
  const bins = 1 << 16;
  const logs = new Float32Array(count);
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < count; i++) {
    const dx = (positions[i * 3] ?? 0) - eye[0];
    const dy = (positions[i * 3 + 1] ?? 0) - eye[1];
    const dz = (positions[i * 3 + 2] ?? 0) - eye[2];
    logs[i] = Math.log(dx * dx + dy * dy + dz * dz + 1e-12);
    lo = Math.min(lo, logs[i] ?? 0);
    hi = Math.max(hi, logs[i] ?? 0);
  }
  const scale = hi > lo ? (bins - 1) / (hi - lo) : 0;
  const keys = Array.from(logs, (v) => bins - 1 - Math.min(bins - 1, Math.floor((v - lo) * scale)));
  return Uint32Array.from(
    Array.from({ length: count }, (_, i) => i).sort((a, b) => (keys[a] ?? 0) - (keys[b] ?? 0)),
  );
}

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

  it("orders as an exact sort does, at least as finely as the log key it replaced", () => {
    // A 150 m scan seen from inside it, splats from a few centimetres to 200 m away.
    let seed = 42;
    const random = (): number => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const count = 60_000;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = random() * 150 - 20;
      positions[i * 3 + 1] = random() * 150 - 75;
      positions[i * 3 + 2] = random() * 20;
    }
    for (const eye of [
      [0, 0, 1.7],
      [30, -10, 3],
      [-5, 60, 25],
    ] as [number, number, number][]) {
      const distance = (i: number): number =>
        Math.hypot(
          (positions[i * 3] ?? 0) - eye[0],
          (positions[i * 3 + 1] ?? 0) - eye[1],
          (positions[i * 3 + 2] ?? 0) - eye[2],
        );
      const exact = Array.from({ length: count }, (_, i) => distance(i)).sort((a, b) => b - a);
      /** Worst relative error, rank by rank, of an order's distances against the exact one. */
      const worst = (order: Uint32Array): number => {
        let error = 0;
        order.forEach((index, rank) => {
          const want = exact[rank] ?? 0;
          error = Math.max(error, Math.abs(distance(index) - want) / want);
        });
        return error;
      };
      const order = backToFront(positions, count, eye);
      expect(order).toHaveLength(count);
      expect(new Set(order).size).toBe(count);
      expect(worst(order)).toBeLessThanOrEqual(worst(logKeyOrder(positions, count, eye)));
      // A shimmer between overlapping splats starts around 0.1%: far from it.
      expect(worst(order)).toBeLessThan(1e-4);
    }
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

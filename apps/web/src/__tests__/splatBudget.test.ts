import { describe, expect, it } from "vitest";

import {
  AdaptiveSplatBudget,
  BUDGET_WINDOW_FRAMES,
  FAST_FRAME_MS,
  SLOW_FRAME_MS,
} from "@/lib/splatBudget";

function window(budget: AdaptiveSplatBudget, ms: number, drawn: number): boolean {
  let changed = false;
  for (let i = 0; i < BUDGET_WINDOW_FRAMES; i++) changed = budget.frame(ms, drawn) || changed;
  return changed;
}

describe("the splat budget follows motion frame time", () => {
  it("cuts a fifth after a slow window drawn near the budget, down to a floor", () => {
    const budget = new AdaptiveSplatBudget(3_000_000);
    expect(window(budget, SLOW_FRAME_MS * 1.5, 2_900_000)).toBe(true);
    expect(budget.budget).toBe(2_400_000);
    for (let i = 0; i < 20; i++) window(budget, SLOW_FRAME_MS * 1.5, budget.budget);
    expect(budget.budget).toBe(900_000);
  });

  it("leaves it alone when slow frames are not the splats' doing (few drawn)", () => {
    const budget = new AdaptiveSplatBudget(3_000_000);
    expect(window(budget, SLOW_FRAME_MS * 2, 500_000)).toBe(false);
    expect(budget.budget).toBe(3_000_000);
  });

  it("raises it back towards the ceiling when fast and budget-bound, never past it", () => {
    const budget = new AdaptiveSplatBudget(3_000_000);
    window(budget, SLOW_FRAME_MS * 2, 3_000_000);
    window(budget, SLOW_FRAME_MS * 2, 2_400_000);
    expect(budget.budget).toBe(1_920_000);
    window(budget, FAST_FRAME_MS / 2, 1_900_000);
    expect(budget.budget).toBe(2_208_000);
    for (let i = 0; i < 10; i++) window(budget, FAST_FRAME_MS / 2, budget.budget);
    expect(budget.budget).toBe(3_000_000);
  });

  it("decides only once a window is full, and ignores nonsense intervals", () => {
    const budget = new AdaptiveSplatBudget(1_000_000);
    for (let i = 0; i < BUDGET_WINDOW_FRAMES - 1; i++)
      expect(budget.frame(100, 1_000_000)).toBe(false);
    expect(budget.frame(Number.NaN, 1_000_000)).toBe(false);
    expect(budget.frame(5_000, 1_000_000)).toBe(false);
    expect(budget.frame(100, 1_000_000)).toBe(true);
    // A small device's floor is its whole budget below the minimum.
    const small = new AdaptiveSplatBudget(200_000);
    window(small, SLOW_FRAME_MS * 3, 200_000);
    expect(small.budget).toBe(200_000);
  });
});

/**
 * The per-cell rules' cap (lib/ruleCap.ts): a rule names at most four dates, those observed up
 * to the date it is As of. More would put more splats on PlayCanvas's one texture than it holds
 * (ADR 0017 § "Per-cell view rules under PlayCanvas").
 */
import { describe, expect, it } from "vitest";

import { MAX_RULE_DATES, ruleDateCount } from "@/lib/ruleCap";

import { datedSite } from "./epochFixtures";

describe("the dates a rule names", () => {
  it("are those observed up to As of, four at most", () => {
    const { epochs, assets } = datedSite(5);
    expect(MAX_RULE_DATES).toBe(4);
    // As of the latest: all five, one too many.
    expect(ruleDateCount(epochs, assets, null)).toBe(5);
    // As of the fourth: four.
    expect(ruleDateCount(epochs, assets, "e4")).toBe(4);
    expect(ruleDateCount(epochs, assets, "e1")).toBe(1);
  });
});

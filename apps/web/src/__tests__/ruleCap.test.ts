/**
 * The per-cell rules' cap (lib/viewModes.ts `MAX_RULE_DATES`): a rule names at most four dates, those observed up
 * to the date it is As of. More would put more splats on PlayCanvas's one texture than it holds
 * (ADR 0017 § "Per-cell view rules under PlayCanvas").
 */
import { describe, expect, it } from "vitest";

import { epochOptions, shownOption } from "@/lib/epochs";
import { MAX_RULE_DATES, ruleDateCount } from "@/lib/viewModes";

import { datedSite } from "./epochFixtures";

describe("the dates a rule names", () => {
  it("are those observed up to As of, four at most", () => {
    const { epochs, assets } = datedSite(5);
    const options = epochOptions(epochs, assets, "gaussian-splat");
    const asOf = (id: string | null) => shownOption(options, id)?.epoch.observedAt;
    expect(MAX_RULE_DATES).toBe(4);
    // As of the latest: all five, one too many.
    expect(ruleDateCount(options, asOf(null))).toBe(5);
    // As of the fourth: four.
    expect(ruleDateCount(options, asOf("e4"))).toBe(4);
    expect(ruleDateCount(options, asOf("e1"))).toBe(1);
  });
});

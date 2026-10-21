/**
 * How many dates a per-cell view rule names (ADR 0017 § "Per-cell view rules under
 * PlayCanvas"): those observed up to the date it is As of, four at most. Under PlayCanvas every
 * date the rule draws is resident at once in its one work buffer, within one budget split
 * between them; past four, a date's share of it is too small to draw it well, and the texture
 * PlayCanvas packs them into nears its limit. The control and the scene read the same count:
 * the control is offered disabled, saying so, and the scene refuses a rule or ends one, saying
 * why (`ViewRuleEnd` "dates").
 */

import type { Epoch, Representation, SiteAsset } from "@twin/contracts";

import { epochOptions, shownOption } from "./epochs";

/** The most dates a rule names. */
export const MAX_RULE_DATES = 4;

/** The dates up to As of (`epochId`, the date chosen; null for the latest) a rule would name. */
export function ruleDateCount(
  epochs: readonly Epoch[],
  assets: readonly SiteAsset[],
  epochId: string | null,
  representation: Representation = "gaussian-splat",
): number {
  const options = epochOptions(epochs, assets, representation);
  const until = Date.parse(shownOption(options, epochId)?.epoch.observedAt ?? "");
  return options.filter((o) => Date.parse(o.epoch.observedAt) <= until).length;
}

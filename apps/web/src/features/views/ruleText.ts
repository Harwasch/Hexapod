/**
 * The per-cell view rules' words (lib/compositor.ts, ADR 0017), as the Per cell control says
 * them: here, beside the control, so its chunk carries them and the scene's does not.
 */
import type { ViewRule, ViewRuleEnd } from "@/lib/compositor";

/** The rules as the control offers them. */
export const VIEW_RULES: readonly { value: ViewRule; label: string; hint: string }[] = [
  { value: "latest", label: "Latest", hint: "Each cell from the newest date that saw it" },
  { value: "ground-truth", label: "Ground truth", hint: "Latest, never anything inferred" },
  {
    value: "highlight-invented",
    label: "Highlight invented",
    hint: "Latest, inferred cells tinted",
  },
  { value: "data-age", label: "Data age", hint: "Latest, coloured by each cell's date" },
];

/** What the control says when a rule ended without being turned off. */
export const VIEW_RULE_ENDED: Record<ViewRuleEnd, string> = {
  compare: "Per-cell dates ended: Compare › Dates was turned on (it is one or the other)",
  changed: "Per-cell dates ended: the dates can no longer be drawn as one here",
  living:
    "Per-cell dates ended: a Living Survey rig moves this scan, and its motion needs the scan drawn on its own",
  "one-model": "Per-cell dates ended: one model was asked for by itself (a version, or Retry)",
  room: "Per-cell dates ended: not enough room to draw them all on this device",
  dates:
    "Per-cell dates ended: As of now has more than 4 dates up to it, and a rule draws 4 at most",
  unsupported: "Per-cell dates ended: this CesiumJS build cannot draw them",
};

/**
 * The control's notice when PlayCanvas refused the rule for want of room (state/splatRoom.ts):
 * the dates' root tiles alone pass the device's budget.
 */
export function roomText(dates: number): string {
  return `Not enough room to draw ${String(dates)} dates on this device: choose an earlier As of`;
}

/** The control's other words, here so the tests say the same. */
export const VIEW_RULE_TEXT = {
  label: "Per cell",
  off: "One date",
  living:
    "Off while a Living Survey rig moves this scan: its motion needs the scan drawn on its own",
  compare: "Choosing a rule ends Compare › Dates: the two are one or the other",
  tileBounds: "tile bounds, overstates what it saw",
  ownFrame: "placed in its own frame, not the site's",
  tooMany:
    "Draws 4 dates at most: choose an As of date with 4 or fewer up to it (the capture-date control)",
  holdsViews:
    "Per-cell dates are on: a date's own Confidence and Data age can be chosen again with One date",
} as const;

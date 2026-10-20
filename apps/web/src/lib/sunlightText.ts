/**
 * Sunlight's words (ADR 0019 "Sunlight", cesium/splatSunlight.ts), here so the control, the
 * badge and the tests say the same, and the globe's first load carries none of them.
 */
import type { Epoch, SiteAsset } from "@twin/contracts";

import { epochOf, formatEpochDateTime, observedAtFromUpload } from "@/lib/epochs";
import { useSites } from "@/state/sites";
import { useViewRule } from "@/state/viewRule";

export const SUNLIGHT_TEXT = {
  // Simulated: a model of light the capture did not see, so it says so (the repo's rule).
  label: "Simulated lighting",
  hint: "Simulated lighting: flat surfaces shaded as the sun would light them. The shadows the photographs had stay in them.",
  off: "Off",
  now: "Now",
  capture: "At capture",
  badge: "Simulated lighting",
  asCaptured: "Off while As captured is on",
  noCaptureTime: "This date's capture time was not recorded",
  nowInstead: "This date's capture time was not recorded: the sun of now is drawn",
  several:
    "At capture is off while more than one date is drawn: one scene clock holds one capture's sun, so the sun of now is drawn",
  night: "Sun below horizon",
} as const;

/**
 * When the shown model was captured, ISO 8601, or null when that is not known: a date whose
 * time is only when it was uploaded (`observedAtSource: "created"`) has no capture time to
 * relight at. A model without a date (seeded, or from before dates) has its own `observedAt`.
 */
export function captureTimeOf(
  asset: SiteAsset | null | undefined,
  epochs: readonly Epoch[],
): string | null {
  if (!asset) return null;
  const epoch = epochOf(asset, epochs);
  if (epoch) return observedAtFromUpload(epoch) ? null : epoch.observedAt;
  return asset.observedAt ?? null;
}

/**
 * Whether the site draws more than one date at once: two compared across the divider, or one
 * per cell under a view rule. The scene has one clock, so "At capture" cannot be each date's
 * own; it is off meanwhile, and the sun of now is drawn over every date.
 */
export function useSeveralDates(siteId: string | null): boolean {
  const compared = useSites((s) => siteId !== null && s.compareDates?.siteId === siteId);
  const ruled = useViewRule((s) => siteId !== null && s.active?.siteId === siteId);
  return compared || ruled;
}

/**
 * The badge's sentence: whose sun, over what. `capturedAt` null says the capture's time is not
 * known rather than inventing one; `several`, that more than one date is drawn, each over the
 * shadows of its own capture.
 */
export function relitSentence(
  sun: "now" | "at capture",
  capturedAt: string | null,
  several = false,
): string {
  const when = several
    ? "each date's own"
    : capturedAt
      ? formatEpochDateTime(capturedAt)
      : "time not recorded";
  return `Simulated lighting: the sun (${sun}) over shadows baked in at capture, ${when}`;
}

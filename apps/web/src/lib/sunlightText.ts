/**
 * Sunlight's words (ADR 0019 "Sunlight", cesium/splatSunlight.ts), here so the control, the
 * badge and the tests say the same, and the globe's first load carries none of them.
 */
import type { Epoch, SiteAsset } from "@twin/contracts";

import { epochOf, formatEpochDateTime, observedAtFromUpload } from "@/lib/epochs";

export const SUNLIGHT_TEXT = {
  label: "Sun",
  hint: "Shade flat surfaces by the real sun. The shadows the photographs had stay in them.",
  off: "Off",
  now: "Now",
  capture: "At capture",
  badge: "Sunlight",
  asCaptured: "Off while As captured is on",
  noCaptureTime: "This date's capture time was not recorded",
  nowInstead: "This date's capture time was not recorded: the sun of now is drawn",
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
 * The badge's sentence: whose sun, over what. `capturedAt` null says the capture's time is not
 * known rather than inventing one.
 */
export function relitSentence(sun: "now" | "at capture", capturedAt: string | null): string {
  const when = capturedAt ? formatEpochDateTime(capturedAt) : "time not recorded";
  return `Relit by the sun (${sun}) over shadows baked in at capture, ${when}`;
}

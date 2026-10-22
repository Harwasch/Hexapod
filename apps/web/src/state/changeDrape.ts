import { create } from "zustand";

/**
 * A change map asked for: of which site, from the older date to the newer (epoch ids), and the
 * two as Compare › Dates had them either side of the divider, so the panel shows them again.
 */
export interface ChangeDrapeRequest {
  siteId: string;
  from: string;
  to: string;
  left: string;
  right: string;
}

/**
 * The change drape (features/change, ADR 0018): the map Compare › Dates asked for, and why it
 * could not be drawn when it could not (`error`, written by the drape: a refusal in words).
 * Not persisted: off every visit. It outlives the Layers panel, as the date swipe does.
 */
interface ChangeDrapeState {
  request: ChangeDrapeRequest | null;
  error: string | null;
}

export const useChangeDrape = create<ChangeDrapeState>()(() => ({ request: null, error: null }));

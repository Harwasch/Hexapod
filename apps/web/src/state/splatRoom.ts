import { create } from "zustand";

/**
 * What PlayCanvas refused to draw for want of room on this device (scanView/scanRule.ts
 * `roomForRoots`): a per-cell rule's dates, or two dates compared, whose root tiles alone --
 * which a streamer keeps whatever its share of the budget -- do not fit the device's one
 * budget. The session that ended it writes it; the control reads it, and stays disabled with
 * the reason while the same dates would be asked for again (the same site and As of, or the
 * same pair), under PlayCanvas. Loaded with the rules' and the comparison's code, never on the
 * globe's first load; not persisted.
 */
export interface Refused {
  /** What was refused: `ruleRoomKey` or `compareRoomKey`. */
  key: string;
  /** The dates it would have drawn, the scan's own included. */
  dates: number;
}

interface SplatRoomState {
  rule: Refused | null;
  compare: Refused | null;
}

export const useSplatRoom = create<SplatRoomState>()(() => ({ rule: null, compare: null }));

/** A rule asked for at `siteId`, As of `epochId` (null: the latest). */
export function ruleRoomKey(siteId: string, epochId: string | null): string {
  return `${siteId}|${epochId ?? ""}`;
}

/** Two dates compared at `siteId`. */
export function compareRoomKey(siteId: string, left: string, right: string): string {
  return `${siteId}|${left}|${right}`;
}

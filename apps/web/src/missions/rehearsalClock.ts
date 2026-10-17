/**
 * Where a dress rehearsal's ghost is at a moment, and the speeds it plays at: the part of
 * `rehearsal.ts` the map needs on every frame (`cesium/RehearsalManager.ts`). Kept apart so
 * the planner itself loads only when "Rehearse" is pressed.
 */

import type { LonLat } from "@twin/geo";

import type { LaneRef, RehearsalTrack } from "./rehearsal";

/** The two speeds a rehearsal plays at: simulated seconds per real second. */
export const REHEARSAL_SPEEDS = [300, 600] as const;
export type RehearsalSpeed = (typeof REHEARSAL_SPEEDS)[number];

export function lerp(a: LonLat, b: LonLat, f: number): LonLat {
  return {
    longitude: a.longitude + (b.longitude - a.longitude) * f,
    latitude: a.latitude + (b.latitude - a.latitude) * f,
  };
}

/** Where a ghost is at `t`, and the lane it is on (null between lanes or when done). */
export function positionAt(
  track: RehearsalTrack,
  t: number,
): { at: LonLat; lane: LaneRef | null } | null {
  const segments = track.segments;
  const first = segments[0];
  const last = segments[segments.length - 1];
  if (!first || !last) return null;
  if (t <= first.t0) return { at: first.from, lane: null };
  if (t >= last.t1) return { at: last.to, lane: null };
  let lo = 0;
  let hi = segments.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((segments[mid]?.t1 ?? 0) < t) lo = mid + 1;
    else hi = mid;
  }
  const segment = segments[lo] ?? last;
  const span = segment.t1 - segment.t0;
  const f = span > 0 ? Math.min(1, Math.max(0, (t - segment.t0) / span)) : 1;
  return { at: lerp(segment.from, segment.to, f), lane: segment.lane };
}

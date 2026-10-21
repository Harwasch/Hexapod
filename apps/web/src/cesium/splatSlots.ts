/**
 * Which splats of an incremental splat primitive belong to a given set of tilesets: the slot
 * ranges of their tiles (`_tileSlots`, engine patch), packed for a shader uniform. Used by the
 * parts that act on a companion's splats alone -- an inferred layer drawn in the scan's sort
 * (`splatViewCones.ts`, `inferredLayers.ts`) -- or on everything but them (`splatCellView.ts`).
 * A module of its own so a part loaded on demand does not take the view cones with it.
 */

import type { VisibilityPrimitive } from "./splatVisibility";

/** Slot ranges a companion's part can tell apart: two to an `ivec4`. */
export const COMPANION_SLOT_VECS = 4;

/**
 * The slot ranges of `owners`' tiles in an incremental primitive (`_tileSlots`), merged where
 * they touch, as `[start, end)` pairs packed two to a vec4 (`COMPANION_SLOT_VECS` of them;
 * unused pairs empty). Tiles past what fits are left out, and `overflow` says how many.
 */
export function companionSlots(
  primitive: VisibilityPrimitive | undefined,
  owners: ReadonlySet<unknown>,
): { vecs: [number, number, number, number][]; overflow: number; count: number } {
  const ranges: [number, number][] = [];
  for (const [tile, slot] of primitive?._tileSlots ?? []) {
    if (!owners.has(tile.tileset)) continue;
    ranges.push([slot.start, slot.start + slot.count]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const range of ranges) {
    const last = merged[merged.length - 1];
    if (last && last[1] >= range[0]) last[1] = Math.max(last[1], range[1]);
    else merged.push([range[0], range[1]]);
  }
  const fits = COMPANION_SLOT_VECS * 2;
  const vecs: [number, number, number, number][] = [];
  for (let k = 0; k < COMPANION_SLOT_VECS; k += 1) {
    const a = merged[k * 2];
    const b = merged[k * 2 + 1];
    vecs.push([a?.[0] ?? 0, a?.[1] ?? 0, b?.[0] ?? 0, b?.[1] ?? 0]);
  }
  return {
    vecs,
    overflow: Math.max(0, merged.length - fits),
    count: Math.min(merged.length, fits),
  };
}

/** Whether `splatIndex` is in one of `vecs`' ranges (what the GLSL `inSlots` tests). */
export function inCompanionSlots(
  vecs: readonly (readonly [number, number, number, number])[],
  splatIndex: number,
): boolean {
  return vecs.some(
    ([s0, e0, s1, e1]) =>
      (splatIndex >= s0 && splatIndex < e0) || (splatIndex >= s1 && splatIndex < e1),
  );
}

/** GLSL: whether `splatIndex` is in the ranges of the `ivec4` array `slots`. */
export function companionSlotsGlsl(fn: string, slots: string): string {
  return `
bool ${fn}(uint splatIndex) {
    int i = int(splatIndex);
    for (int k = 0; k < ${String(COMPANION_SLOT_VECS)}; k++) {
        ivec4 r = ${slots}[k];
        if ((i >= r.x && i < r.y) || (i >= r.z && i < r.w)) {
            return true;
        }
    }
    return false;
}
`;
}

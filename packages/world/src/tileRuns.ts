/**
 * Per-tile, per-gaussian labels: the encoding every tile sidecar shares.
 *
 * A level-of-detail splat tileset's gaussians are identified per tile, by the tile's own
 * checksum (`checksumPositions` over its canonical positions, the keys of `rig.tileChecksums`),
 * and labelled in the tile's own gaussian order as run-length pairs `[label, count, ...]`.
 * `plants.json` (`plantBinding.ts`: which plant) and `instances.json` (the scene's instance
 * ids, docs/SCENE_OBJECTS.md §4) both carry their `tiles` this way, written by the same
 * `_rle` in `tools/captures/scene_plants.py`; this is the one reader.
 *
 * The checksum carries the tile's gaussian count, so the runs are checked to cover exactly it.
 */

/** `fnv1a32:<count>:<digest>`, as `checksumPositions` writes it. */
export const TILE_CHECKSUM = /^fnv1a32:(\d+):[0-9a-f]{8}$/;

/** The gaussian count a tile checksum carries, or `undefined` when it is not one. */
export function checksumCount(checksum: string): number | undefined {
  const match = TILE_CHECKSUM.exec(checksum);
  return match === null ? undefined : Number(match[1]);
}

/**
 * What is wrong with one tile's entry, or `undefined` when nothing is: the key must be a
 * checksum, the value `[label, count]` pairs with integer labels in `0..maxLabel` and positive
 * counts that add up to the checksum's gaussian count. `what` names a non-zero label in messages.
 */
export function tileRunsIssue(
  key: string,
  value: unknown,
  maxLabel: number,
  what = "a label",
): string | undefined {
  const expected = checksumCount(key);
  if (expected === undefined) return `${key}: not a checksumPositions digest`;
  if (!Array.isArray(value) || value.length % 2 !== 0) {
    return `${key}: runs must be [label, count] pairs`;
  }
  let total = 0;
  for (let i = 0; i < value.length; i += 2) {
    const label: unknown = value[i];
    const run: unknown = value[i + 1];
    if (!Number.isInteger(label) || (label as number) < 0 || (label as number) > maxLabel) {
      return `${key}: label ${String(label)} is not 0 or ${what}`;
    }
    if (!Number.isInteger(run) || (run as number) <= 0) {
      return `${key}: run length ${String(run)} must be a positive integer`;
    }
    total += run as number;
  }
  if (total !== expected) {
    return `${key}: runs cover ${total} gaussians, the tile holds ${String(expected)}`;
  }
  return undefined;
}

/** Gaussians the runs cover. */
export function runsLength(runs: ArrayLike<number>): number {
  let total = 0;
  for (let i = 1; i < runs.length; i += 2) total += runs[i] ?? 0;
  return total;
}

/**
 * Expands run-length pairs into one label per gaussian. `out` must hold `runsLength(runs)`
 * labels; a `Uint16Array` for plants, a `Uint32Array` for instance ids.
 */
export function decodeRuns<T extends Uint16Array | Uint32Array>(
  runs: ArrayLike<number>,
  out: T,
): T {
  let at = 0;
  for (let i = 0; i < runs.length; i += 2) {
    const run = runs[i + 1] ?? 0;
    out.fill(runs[i] ?? 0, at, at + run);
    at += run;
  }
  return out;
}

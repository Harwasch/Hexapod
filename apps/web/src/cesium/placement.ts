/**
 * Where a capture's model rests, and what it rests on.
 *
 * Until B4 the answer was: the lowest corner of the tileset's root bounding box, put on the
 * terrain sampled under the tileset's centre. Those are two different places. On anything
 * sloping the lowest corner is at the bottom of the slope and the sampled terrain is in the
 * middle of it, so the model came out floating by roughly the slope's own drop — and the only
 * cure was a person reading one number out of `tools/captures/ground_samples.py`, a second out
 * of a browser console, subtracting them, and typing the difference into `heightOffsetM`.
 *
 * The fix is to compare like with like. The pipeline already measures the capture's own ground
 * cell by cell (`splat_ground` → `ground_samples.json`); the catalog now carries those cells as
 * ellipsoid heights. Sample the ground at exactly those longitudes and latitudes, take the
 * median of the per-cell differences, and the slope cancels because both sides of every
 * subtraction are at the same point. That median *is* the subtraction that used to be done by
 * hand, done at load time against whatever terrain the viewer actually has.
 *
 * Pure on purpose: everything here is numbers in, numbers out, so the arithmetic that decides
 * where a model sits is testable without a globe.
 */

/** One cell of the capture's own measured ground, as the catalog stores it. */
export interface MeasuredGround {
  readonly lon: number;
  readonly lat: number;
  /** Ellipsoid height of the capture's own ground at this point, in metres. */
  readonly height: number;
}

/** What a measured clamp worked out, and how much of a measurement it was. */
export interface MeasuredClamp {
  /** Metres to raise the model by so its measured ground sits on the sampled ground. */
  readonly liftM: number;
  /** How many cells had ground under them. Cells whose sample failed are not counted. */
  readonly cells: number;
  /**
   * Median absolute deviation of the per-cell differences, in metres — how much the capture's
   * ground and the viewer's ground disagree in *shape* after the median has removed the
   * offset. Small means the two surfaces are parallel and the clamp is a translation. Large
   * means they are not the same surface, and no single lift makes them agree.
   */
  readonly spreadM: number;
}

/**
 * The ground at one point: the terrain, or the surface actually drawn there.
 *
 * The drawn world (the photorealistic tileset, another site's mesh) frequently sits metres off
 * the terrain mesh, and a model has to rest on what a person can see. But a drawn height far
 * from the terrain is a roof or a tree canopy rather than ground, so it is only believed within
 * `toleranceM`. This is the rule `clampToGround` already applied at the centre, kept exactly
 * and applied per point.
 */
export function groundAt(
  terrain: number | undefined,
  drawn: number | undefined,
  toleranceM: number,
): number | undefined {
  const hasTerrain = terrain !== undefined && Number.isFinite(terrain);
  if (drawn !== undefined && Number.isFinite(drawn)) {
    if (!hasTerrain) return drawn;
    if (Math.abs(drawn - terrain) < toleranceM) return drawn;
  }
  return hasTerrain ? terrain : undefined;
}

/**
 * The lift that rests the capture's measured ground on the ground under it.
 *
 * `ground[i]` is the ground sampled at `samples[i]`, or undefined where the sample failed —
 * terrain tiles do not always arrive. Returns null when nothing could be sampled, which is the
 * caller's signal to fall back to the bounding box rather than to place the model at a lift of
 * zero (a lift of zero is a claim; no answer is not).
 *
 * The median rather than the mean: a capture's "ground" cells include whatever the low
 * percentile of a cell happened to be, and one cell over a pond or under an overhang should not
 * drag the whole model.
 */
export function measuredClamp(
  samples: readonly MeasuredGround[],
  ground: readonly (number | undefined)[],
): MeasuredClamp | null {
  const differences: number[] = [];
  for (const [index, sample] of samples.entries()) {
    const under = ground[index];
    if (under === undefined || !Number.isFinite(under) || !Number.isFinite(sample.height)) continue;
    differences.push(under - sample.height);
  }
  if (differences.length === 0) return null;
  const liftM = median(differences);
  const spreadM = median(differences.map((d) => Math.abs(d - liftM)));
  return { liftM, cells: differences.length, spreadM };
}

/** Median of a non-empty list. The mean of the middle two when the count is even. */
export function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  const high = sorted[middle] ?? 0;
  if (sorted.length % 2 === 1) return high;
  return ((sorted[middle - 1] ?? 0) + high) / 2;
}

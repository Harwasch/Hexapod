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
 * where a model sits, and where a fly-to looks at it (`robustArrivalSphere`), is testable
 * without a globe.
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

/** A point or a vector in Earth-fixed metres, as plain numbers (what a `Cartesian3` holds). */
export type Vec3 = readonly [number, number, number];

/**
 * The model matrix that draws a tileset at `scale` times its registered size about `origin`,
 * then raises it by `lift`: column-major, as CesiumJS's `Matrix4`.
 *
 * It is `T(lift) · R · S(scale) · R⁻¹`, R being the root transform whose translation is
 * `origin` -- the placed coordinate, about which `PUT /assets/{id}/scale` also resizes the
 * site's boundary, the footprint and the ground samples (docs/DATA_MODEL.md "Runtime scale").
 * A uniform scale commutes with R's rotation, so whatever R's axes are that is
 * `[scale·I | (1 − scale)·origin + lift]`: the origin stays where it is, lift aside, and every
 * other point moves `scale` times as far from it. At scale 1 it is exactly the translation the
 * clamp has always set, so an asset nobody resized does not move by a bit.
 */
export function scaledModelMatrix(origin: Vec3, scale: number, lift: Vec3 = [0, 0, 0]): number[] {
  const keep = 1 - scale;
  const matrix = [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1];
  for (let axis = 0; axis < 3; axis++) {
    matrix[12 + axis] = keep * (origin[axis] ?? 0) + (lift[axis] ?? 0);
  }
  return matrix;
}

/**
 * How much a transform scales lengths: the length of its first column, which for the uniform
 * scale a model matrix may carry (`scaledModelMatrix`) is the scale, and 1 for a rigid one.
 * Lengths and radii in a tileset's own frame are this many times as long on the globe.
 */
export function uniformScale(matrix: ArrayLike<number>): number {
  const length = Math.hypot(matrix[0] ?? 0, matrix[1] ?? 0, matrix[2] ?? 0);
  return Number.isFinite(length) && length > 0 ? length : 1;
}

/**
 * Where a height ends up once the model is drawn at `scale` about an origin `originHeight`
 * high: a point `h − originHeight` above the origin is `scale` times as far above it. This is
 * the bounding-box clamp's lowest point under a runtime scale, worked out rather than read back
 * from a resized box, and the same linear rule the API moves the ground samples by.
 */
export function scaledHeight(originHeight: number, height: number, scale: number): number {
  return originHeight + (height - originHeight) * scale;
}

/**
 * The catalog's ground samples moved `factor` times as far from `origin`, across and in height,
 * as `PUT /assets/{id}/scale` moves them (apps/api placement.py `rescale_ground_samples`).
 *
 * Only a *preview*'s factor ever comes here: the scale being tried over the scale saved. The
 * samples the catalog sends already describe the saved scale -- every globe position the API
 * returns does -- so a factor of 1 hands them back untouched, and the saved scale itself is
 * never applied to them a second time.
 */
export function rescaleGround(
  samples: readonly MeasuredGround[],
  origin: MeasuredGround,
  factor: number,
): readonly MeasuredGround[] {
  if (factor === 1) return samples;
  return samples.map((sample) => ({
    lon: origin.lon + (sample.lon - origin.lon) * factor,
    lat: origin.lat + (sample.lat - origin.lat) * factor,
    height: origin.height + (sample.height - origin.height) * factor,
  }));
}

/** A footprint's rings, as GeoJSON nests them: a polygon's, or each of a multipolygon's. */
type Rings = number[][][];
interface FootprintLike {
  readonly type: "Polygon" | "MultiPolygon";
  readonly coordinates: Rings | Rings[];
}

/**
 * A footprint resized `factor` times about `origin` (degrees), linearly in longitude and
 * latitude as the API resizes the boundary and the footprint (shapely's `affinity.scale`). For
 * a preview only, like `rescaleGround`: what the catalog sends already fits the saved scale.
 */
export function rescaleFootprint<F extends FootprintLike>(
  footprint: F,
  origin: { readonly lon: number; readonly lat: number },
  factor: number,
): F {
  if (factor === 1) return footprint;
  const ring = (points: number[][]): number[][] =>
    points.map(([lon = 0, lat = 0, ...rest]) => [
      origin.lon + (lon - origin.lon) * factor,
      origin.lat + (lat - origin.lat) * factor,
      ...rest,
    ]);
  const coordinates =
    footprint.type === "Polygon"
      ? (footprint.coordinates as Rings).map(ring)
      : (footprint.coordinates as Rings[]).map((polygon) => polygon.map(ring));
  return { ...footprint, coordinates };
}

/** Median of a non-empty list. The mean of the middle two when the count is even. */
export function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  const high = sorted[middle] ?? 0;
  if (sorted.length % 2 === 1) return high;
  return ((sorted[middle - 1] ?? 0) + high) / 2;
}

/** A point on the globe: degrees, and an ellipsoid height in metres. */
export interface GeoPoint {
  readonly lon: number;
  readonly lat: number;
  readonly height: number;
}

/** A sphere on the globe, as a fly-to frames one: its centre, and its radius in metres. */
export interface ArrivalSphere extends GeoPoint {
  readonly radiusM: number;
}

/**
 * Metres per degree of latitude, and of longitude at the equator: the constant the pipeline
 * places ground cells with (`METRES_PER_DEGREE`, tools/pipeline/gaussians.py). Good to about
 * half a per cent, which over a capture's few hundred metres is far below what a view shows.
 */
const METRES_PER_DEGREE = 111_320;
/** Fewer ground cells than this say where the ground is, not how far it spreads. */
const MIN_SPREAD_CELLS = 3;
/** The share of ground cells the framed radius reaches: the rest may be background. */
const SPREAD_QUANTILE = 0.9;
/** A framed radius is never smaller than this (m): cells a metre apart are still a scan. */
const MIN_ARRIVAL_RADIUS_M = 1;

/**
 * What a fly-to frames at a placed scan: its placement origin, at the size of its own ground.
 *
 * Not the tileset's bounding sphere, which spans every gaussian the packer kept, floaters and
 * background shell included. Its centre sits wherever that junk pulls it, metres to the side
 * and tens of metres up, and its radius can be several times the subject's: a camp some 50 m
 * across had a 128 m sphere, and the camera arrived 400 m out, looking at a point in the air.
 * The pipeline has already measured both things robustly:
 *
 * * the **centre** is `origin`, the root transform's translation once the clamp has lifted
 *   the model. `orient` (tools/pipeline/gaussians.py) recentres every capture on the middle
 *   of its 2nd-98th percentile footprint and on its own measured ground, and the clamp rests
 *   that ground on the terrain, so the origin is the middle of the scan, on the ground.
 * * the **radius** is the spread of `samples`, the capture's densest ground cells: the 90th
 *   percentile of their distance from the origin, so a few dense patches of background do
 *   not count either. Fewer than three cells say nothing about a spread; then it is the
 *   smaller of the site's footprint radius and the bounding sphere's.
 *
 * Neither may go past the bounding sphere, which holds everything there is. An origin outside
 * it (a capture that was never recentred, whose frame starts wherever its app did) frames
 * nothing, and `bounds` comes back unchanged. Ground cells further out than the tiles reach
 * disagree with them, and the radius is held to the sphere's.
 */
export function robustArrivalSphere(
  origin: GeoPoint,
  bounds: ArrivalSphere,
  samples: readonly MeasuredGround[],
  footprintRadiusM?: number,
): ArrivalSphere {
  if (![origin.lon, origin.lat, origin.height].every(Number.isFinite)) return bounds;
  const [east, north] = offsetM(origin, bounds);
  if (Math.hypot(east, north, bounds.height - origin.height) > bounds.radiusM) return bounds;
  const distances: number[] = [];
  for (const sample of samples) {
    if (!Number.isFinite(sample.lon) || !Number.isFinite(sample.lat)) continue;
    distances.push(Math.hypot(...offsetM(origin, sample)));
  }
  let radiusM: number;
  if (distances.length >= MIN_SPREAD_CELLS) {
    radiusM = Math.max(quantile(distances, SPREAD_QUANTILE), MIN_ARRIVAL_RADIUS_M);
  } else if (footprintRadiusM !== undefined && footprintRadiusM > 0) {
    radiusM = footprintRadiusM;
  } else {
    radiusM = bounds.radiusM;
  }
  return {
    lon: origin.lon,
    lat: origin.lat,
    height: origin.height,
    radiusM: Math.min(radiusM, bounds.radiusM),
  };
}

/** East and north metres from `from` to `to`, on the tangent plane at `from`. */
function offsetM(from: GeoPoint, to: { lon: number; lat: number }): [number, number] {
  // Across the antimeridian the short way round, not the 359 degrees the other way.
  const dLon = ((((to.lon - from.lon + 180) % 360) + 360) % 360) - 180;
  const east = dLon * METRES_PER_DEGREE * Math.cos((from.lat * Math.PI) / 180);
  return [east, (to.lat - from.lat) * METRES_PER_DEGREE];
}

/** The `q` quantile (0 to 1) of a non-empty list, interpolated between its neighbours. */
function quantile(values: readonly number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const at = (sorted.length - 1) * q;
  const low = sorted[Math.floor(at)] ?? 0;
  const high = sorted[Math.ceil(at)] ?? low;
  return low + (high - low) * (at - Math.floor(at));
}

/**
 * A tileset drawn at its runtime scale (`renderConfig.scale`, docs/DATA_MODEL.md "Runtime
 * scale"), and how the code that works in a tileset's own frame stays right under one.
 *
 * The contract is the API's: every position the catalog puts on the globe -- the site's boundary
 * and centroid, the asset's footprint, its `groundSamples` -- already describes the scaled
 * model. What the viewer scales is only what it draws in the tileset's own frame: the tiles and
 * everything that follows the root's computed transform (instances, split objects, view cones,
 * collision, the other renderers' poses), about the root transform's origin, the placed
 * coordinate. So the model matrix is `T(lift) · R · S(scale) · R⁻¹` (`scaledModelMatrix`), and
 * the root's computed transform `M · R` is no longer rigid: a frame of its inverse is `1 / scale`
 * as long as a metre (`inverseScaledTransformation`), and a length measured in it is `scale`
 * metres a unit on the globe (`uniformScale`).
 *
 * The clamp's arithmetic lives here too, apart from the sampling (`SiteManager.clampToGround`):
 * where the scaled model's centre and lowest point are, and the lift that rests it on ground
 * already sampled, so a preview of another scale can be rested at once and sampled afresh after.
 */

import {
  Cartesian3,
  Cartographic,
  Math as CesiumMath,
  Matrix4,
  type Cesium3DTileset,
} from "cesium";

import type { Footprint, SiteAsset } from "@twin/contracts";

import { assetScale } from "@/lib/realSize";

import {
  measuredClamp,
  rescaleFootprint,
  rescaleGround,
  scaledHeight,
  scaledModelMatrix,
  uniformScale,
  type MeasuredGround,
} from "./placement";

/**
 * A root transform whose translation is shorter than this (metres) places nothing on the globe
 * (an identity root over Earth-fixed positions): its tileset is resized about its own centre.
 */
const PLACED_ORIGIN_MIN_M = 1_000_000;

/** How close to 1 a scale squared must be to be a rigid transform's (rounding, not a scale). */
const RIGID_EPSILON = 1e-12;

/** A tileset as it was registered, before any model matrix: what its placement is worked from. */
export interface PlacementFrame {
  /** The root transform's origin, Earth-fixed: the point the scale keeps where it is. */
  readonly origin: Cartesian3;
  /** The same point as a measured-ground cell is given: degrees and ellipsoid height. */
  readonly ground: MeasuredGround;
  /** The bounding sphere's centre and radius at scale 1, unlifted. */
  readonly center: Cartesian3;
  readonly radius: number;
  /**
   * Ellipsoid height of the model's lowest point at scale 1, unlifted: the root box's lowest
   * corner when it has a box, else the bottom of the sphere.
   */
  readonly bottom: number;
}

/**
 * What a clamp sampled, kept so that another scale can be rested at once: the measured cells
 * where they were sampled (at `scale`) and the ground found under each, and the ground under
 * the model's centre.
 */
export interface SampledGround {
  readonly scale: number;
  readonly samples: readonly MeasuredGround[];
  readonly ground: readonly (number | undefined)[];
  readonly under: number | undefined;
}

/**
 * A freshly created tileset's frame, read while its model matrix is still the identity.
 * `bottom` is its root box's lowest corner, when it has a box.
 */
export function placementFrame(
  tileset: Pick<Cesium3DTileset, "boundingSphere" | "root">,
  bottom: number | undefined,
): PlacementFrame {
  const sphere = tileset.boundingSphere;
  const center = Cartesian3.clone(sphere.center);
  const declared = (tileset.root as { transform?: Matrix4 } | undefined)?.transform;
  const translation = declared ? Matrix4.getTranslation(declared, new Cartesian3()) : undefined;
  const origin =
    translation && Cartesian3.magnitude(translation) > PLACED_ORIGIN_MIN_M ? translation : center;
  const carto = Cartographic.fromCartesian(origin) as Cartographic | undefined;
  const centre = Cartographic.fromCartesian(center) as Cartographic | undefined;
  return {
    origin,
    ground: {
      lon: CesiumMath.toDegrees(carto?.longitude ?? 0),
      lat: CesiumMath.toDegrees(carto?.latitude ?? 0),
      height: carto?.height ?? 0,
    },
    center,
    radius: sphere.radius,
    bottom: bottom ?? (centre?.height ?? 0) - sphere.radius,
  };
}

/**
 * Where the model's centre is drawn at `scale`: `scale` times as far from the origin. Exactly
 * the registered centre at scale 1, so an unscaled asset is clamped where it always was.
 */
export function scaledCenter(frame: PlacementFrame, scale: number): Cartesian3 {
  if (scale === 1) return Cartesian3.clone(frame.center);
  return Cartesian3.lerp(frame.origin, frame.center, scale, new Cartesian3());
}

/**
 * The model matrix for a tileset drawn at `scale` about its origin and raised `liftM` metres
 * along the vertical at its (scaled) centre -- the direction the clamp has always lifted along.
 */
export function placedMatrix(frame: PlacementFrame, scale: number, liftM: number): Matrix4 {
  let lift: [number, number, number] = [0, 0, 0];
  if (liftM !== 0) {
    const at = Cartographic.fromCartesian(scaledCenter(frame, scale)) as Cartographic | undefined;
    if (at) {
      const from = Cartesian3.fromRadians(at.longitude, at.latitude, at.height);
      const to = Cartesian3.fromRadians(at.longitude, at.latitude, at.height + liftM);
      const by = Cartesian3.subtract(to, from, new Cartesian3());
      lift = [by.x, by.y, by.z];
    }
  }
  const { x, y, z } = frame.origin;
  return Matrix4.fromArray(scaledModelMatrix([x, y, z], scale, lift));
}

/**
 * The lift (metres) that rests the model drawn at `scale` on ground a clamp already sampled,
 * `heightOffsetM` included, or null when nothing was sampled.
 *
 * Measured cells first, as the clamp does: the cells move with the scale (`rescaleGround`, by
 * `scale` over the scale they were sampled at) while the ground under each is taken as it was
 * -- exact at the sampled scale, and on level ground at any; on a slope the clamp proper,
 * sampled where the cells now are, corrects it. Without cells, the lowest point goes on the
 * ground under the centre, and it scales with the model (`scaledHeight`).
 */
export function liftAt(
  frame: PlacementFrame,
  sampled: SampledGround | undefined,
  scale: number,
  offsetM: number,
): { liftM: number; cells: number; spreadM: number } | null {
  if (!sampled) return null;
  if (sampled.samples.length > 0) {
    const moved = rescaleGround(sampled.samples, frame.ground, scale / sampled.scale);
    const clamp = measuredClamp(moved, sampled.ground);
    if (clamp) return { liftM: clamp.liftM + offsetM, cells: clamp.cells, spreadM: clamp.spreadM };
  }
  if (sampled.under === undefined) return null;
  const bottom = scaledHeight(frame.ground.height, frame.bottom, scale);
  return { liftM: sampled.under + offsetM - bottom, cells: 0, spreadM: 0 };
}

/**
 * The inverse of a root's computed transform: a rotation and a translation, times the uniform
 * runtime scale. `Matrix4.inverseTransformation` assumes the 3×3 is a rotation and transposes
 * it; for `s·A` that gives `s·Aᵀ` where the inverse is `Aᵀ / s`, so its 3×4 is divided by `s²`.
 * A rigid transform comes back exactly as `inverseTransformation` gives it.
 */
export function inverseScaledTransformation(matrix: Matrix4, result: Matrix4): Matrix4 {
  Matrix4.inverseTransformation(matrix, result);
  const squared = uniformScale(matrix) ** 2;
  if (Math.abs(squared - 1) < RIGID_EPSILON) return result;
  const values = Matrix4.toArray(result);
  for (const index of [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]) {
    values[index] = (values[index] ?? 0) / squared;
  }
  return Matrix4.fromArray(values, 0, result);
}

/**
 * Whether two records of one asset place it the same: a record refetched for another reason (a
 * bookmark saved) leaves the scan where it is, and one that was resized moves it.
 */
export function samePlacement(a: SiteAsset, b: SiteAsset): boolean {
  const key = (asset: SiteAsset) =>
    JSON.stringify([
      assetScale(asset),
      asset.renderConfig.groundSamples ?? [],
      asset.renderConfig.heightOffsetM ?? 0,
      asset.renderConfig.clampToGround ?? false,
      asset.footprint,
    ]);
  return key(a) === key(b);
}

/**
 * Where the scaled model's lowest point is: the registered one, `scale` times as far from the
 * origin in height (`scaledHeight`). The bounding-box clamp rests this on the ground.
 */
export function scaledBottom(frame: PlacementFrame, scale: number): number {
  return scale === 1 ? frame.bottom : scaledHeight(frame.ground.height, frame.bottom, scale);
}

/**
 * The catalog's measured cells where a preview puts them: moved by `factor`, the scale tried
 * over the scale saved, about the origin. A factor of 1 -- no preview -- leaves them exactly as
 * the catalog sent them, already at its scale.
 */
export function groundAtScale(
  frame: PlacementFrame,
  samples: readonly MeasuredGround[],
  factor: number,
): readonly MeasuredGround[] {
  return rescaleGround(samples, frame.ground, factor);
}

/** The catalog's footprint where a preview puts it: resized by `factor` about the origin. */
export function footprintAtScale(
  frame: PlacementFrame,
  footprint: Footprint,
  factor: number,
): Footprint {
  return rescaleFootprint(footprint, frame.ground, factor);
}

/**
 * Brings a tileset's root up to date with a model matrix just set, as the `boundingSphere`
 * getter does: the root's computed transform is otherwise refreshed only when CesiumJS next
 * traverses the tileset, and what reads the scan's frame -- a dedicated renderer drawing it
 * while CesiumJS keeps it hidden, the collider, a measurement on it -- must see a preview at once.
 */
export function syncRootTransform(tileset: Pick<Cesium3DTileset, "root" | "modelMatrix">): void {
  const root = tileset.root as { updateTransform?: (parent: Matrix4) => void } | undefined;
  root?.updateTransform?.(tileset.modelMatrix);
}

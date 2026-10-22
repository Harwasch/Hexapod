/**
 * Where one dated tileset is drawn in the frame of another, for a dedicated renderer drawing
 * both in one app (PlayCanvas): a per-cell view rule's dates (compositorSession.ts) and the
 * left-hand date of two compared (compareSession.ts), each streamed beside the scan the renderer
 * draws (scanView/ruleDates.ts). Loaded on demand with either; nothing of it is on the globe's
 * first load.
 */

import { Matrix4, type Cesium3DTileset } from "cesium";

import { invertAffine, type Mat4 } from "./splatFrames";

export const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export function urlOf(tileset: Cesium3DTileset): string | undefined {
  return (tileset as unknown as { resource?: { url?: string } }).resource?.url;
}

/** The root tile's transform as tileset.json declares it (the identity when it declares none). */
export function rootTransform(tileset: Cesium3DTileset): Matrix4 {
  return (tileset.root as { transform?: Matrix4 } | undefined)?.transform ?? Matrix4.IDENTITY;
}

/** The root tile's transform on the globe now (its placement's model matrix included). */
export function computedTransform(tileset: Cesium3DTileset): number[] {
  const m = (tileset.root as { computedTransform?: Mat4 } | undefined)?.computedTransform;
  return Array.from({ length: 16 }, (_, i) => m?.[i] ?? (i % 5 === 0 ? 1 : 0));
}

/** A tileset's URL as a dedicated renderer fetches it (ScanRendererHost `run`): absolute. */
export function absoluteUrl(tileset: Cesium3DTileset): string | undefined {
  const url = urlOf(tileset);
  return url === undefined ? undefined : new URL(url, location.href).toString();
}

/** `a · b` (column-major 4x4). */
export function times(a: readonly number[], b: readonly number[]): number[] {
  return Matrix4.toArray(
    Matrix4.multiply(Matrix4.fromArray([...a]), Matrix4.fromArray([...b]), new Matrix4()),
  );
}

/** `a⁻¹` (column-major 4x4, any affine: a runtime scale too). */
export function inverse(a: readonly number[]): number[] {
  return Matrix4.toArray(Matrix4.inverse(Matrix4.fromArray([...a]), new Matrix4()));
}

/** `a⁻¹ · b`, or null when it is the identity (to a micrometre): where `b`'s frame is in `a`'s. */
export function relative(a: readonly number[], b: readonly number[]): number[] | null {
  const m = times(inverse(a), b);
  return m.every((v, i) => Math.abs(v - (IDENTITY[i] ?? 0)) < 1e-6) ? null : m;
}

/**
 * Where `tileset`'s tiles go in the frame of `scan` (the tileset the renderer draws): its
 * placement `P = C_scan⁻¹ · C_date` (the two roots' computed transforms), null where they
 * already are; and the root its inferred layers and split objects are placed from, `S · P⁻¹`
 * (its own root `S`), so that a layer lands where the date's tiles do.
 */
export function placedIn(
  scan: Cesium3DTileset,
  tileset: Cesium3DTileset,
): { placement: number[] | null; root: number[] } {
  const placement = invertAffine(computedTransform(scan))
    ? relative(computedTransform(scan), computedTransform(tileset))
    : null;
  const own = Matrix4.toArray(rootTransform(tileset));
  return { placement, root: placement ? times(own, inverse(placement)) : own };
}

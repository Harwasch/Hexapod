/**
 * CesiumJS's own splats as a pick source (pickSources.ts): the tiles of the primitive's
 * committed snapshot (`snapshotTiles`, as the instance hooks read them), each un-baked to the
 * scan's frame (`unbakePositions`, so its checksum is the one `instances.json` lists), with
 * its splats' largest axis (`content.scales`) and opacity (the snapshot's colours). Built once
 * per tile content and bake; empty while the tileset is hidden (another renderer draws it), or
 * drawn only because it is seen from afar (`setSeenFromAfar`).
 *
 * `content.scales` are baked like the positions: the engine multiplies each splat's scales by
 * the bake's own scale -- a runtime scale (`renderConfig.scale`) on the model matrix, or any a
 * tile's transforms carry. Un-baked positions are in the scan's frame, so the radii are taken
 * back to it too (`localRadii`), or a scan drawn at half size would be picked with splats twice
 * as big as it draws them.
 */

import { checksumPositions } from "@twin/world";
import type { Cesium3DTileset, Matrix4 } from "cesium";

import type { PickTile } from "@/lib/splatPick";

import { uniformScale } from "../placement";
import { invertAffine, unbakePositions } from "../splatFrames";
import { splatTilesetOf } from "../splatInternals";
import { sameMatrix, snapshotTiles } from "../splatTiles";
import type { PickSource } from "./pickSources";

/**
 * Tilesets drawn only because they are seen from afar (SiteManager's far view, farView.ts): a
 * scan a few dozen pixels across is something on the map, not something to select objects
 * in. Its objects are picked once its site is engaged, as a dedicated renderer's are
 * (scanView/ScanRendererHost.ts).
 */
const seenFromAfar = new WeakSet<object>();

/** Marks `tileset` as drawn only from afar (`far`), or not. */
export function setSeenFromAfar(tileset: object, far: boolean): void {
  if (far) seenFromAfar.add(tileset);
  else seenFromAfar.delete(tileset);
}

/** A splat with no scale read: a couple of centimetres. */
const DEFAULT_RADIUS_M = 0.02;

/** Each splat's largest axis, from xyz scales (linear). */
export function largestAxes(scales: ArrayLike<number> | undefined, count: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    if (!scales || scales.length < (i + 1) * 3) {
      out[i] = DEFAULT_RADIUS_M;
      continue;
    }
    out[i] = Math.max(
      Math.abs(scales[i * 3] ?? 0),
      Math.abs(scales[i * 3 + 1] ?? 0),
      Math.abs(scales[i * 3 + 2] ?? 0),
    );
  }
  return out;
}

/**
 * Each splat's largest axis in the scan's frame, from scales baked by `bake`: divided by the
 * bake's uniform scale, which is 1 for a rigid one (the radii then are `largestAxes`' own).
 */
export function localRadii(
  scales: ArrayLike<number> | undefined,
  count: number,
  bake: ArrayLike<number>,
): Float32Array {
  const radii = largestAxes(scales, count);
  const scale = uniformScale(bake);
  if (Math.abs(scale - 1) < 1e-9) return radii;
  // A splat with no scale read keeps its default, which is in the scan's frame already.
  const read = Math.min(count, Math.floor((scales?.length ?? 0) / 3));
  for (let i = 0; i < read; i++) radii[i] = (radii[i] ?? 0) / scale;
  return radii;
}

export function cesiumPickSource(tileset: Cesium3DTileset): PickSource {
  const cache = new WeakMap<object, { bake: number[]; tile: PickTile }>();
  let last: readonly PickTile[] = [];
  return {
    renderer: "cesium",
    toWorld: () =>
      tileset.isDestroyed()
        ? undefined
        : (tileset.root as { computedTransform?: Matrix4 } | undefined)?.computedTransform,
    tiles: () => {
      if (tileset.isDestroyed() || !tileset.show || seenFromAfar.has(tileset)) return [];
      const like = splatTilesetOf(tileset);
      const primitive = like.gaussianSplatPrimitive;
      const positions = primitive?._positions;
      const numSplats = primitive?._numSplats ?? 0;
      if (!primitive || !positions || numSplats <= 0) return [];
      const listed = snapshotTiles(like, primitive, positions, numSplats);
      if (listed.kind === "wait") return last;
      const colors = primitive._colors;
      const out: PickTile[] = [];
      for (const tile of listed.tiles) {
        const cached = cache.get(tile.content);
        if (cached && sameMatrix(cached.bake, tile.bake)) {
          out.push(cached.tile);
          continue;
        }
        const inverse = invertAffine(tile.bake);
        if (!inverse) continue;
        const baked = positions.subarray(tile.start * 3, (tile.start + tile.count) * 3);
        const local = unbakePositions(baked, inverse);
        const scales = (tile.content as { scales?: Float32Array }).scales;
        const opacity = new Float32Array(tile.count);
        for (let i = 0; i < tile.count; i++) {
          opacity[i] = colors ? (colors[(tile.start + i) * 4 + 3] ?? 255) / 255 : 1;
        }
        const pick: PickTile = {
          checksum: checksumPositions(local),
          count: tile.count,
          positions: local,
          radii: localRadii(scales, tile.count, tile.bake),
          opacity,
        };
        cache.set(tile.content, { bake: Array.from(tile.bake), tile: pick });
        out.push(pick);
      }
      last = out;
      return out;
    },
  };
}

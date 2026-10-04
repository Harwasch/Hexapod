/**
 * One scan tile for the PlayCanvas splat renderer, off the main thread: fetched, its SPZ taken
 * out of the GLB, decoded with spz-loader (as CesiumJS decodes it) and laid out as the
 * columns PlayCanvas's GSplatData takes (lib/splatLayout.ts), centred on the tile: the
 * positions are sent relative to the middle of the tile's box and the entity is placed there,
 * so they stay small numbers wherever the tile is in the scan (a GPU half float is 6 cm apart
 * 100 m out).
 *
 * The columns arrive already in PlayCanvas's Morton order (`mortonOrder`, what
 * `GSplatData.reorderData` would do), with the order beside them: that sort and the copy of
 * every column used to run on the main thread as each tile landed. Spherical-harmonic bands
 * above `maxSh` are left out (a phone keeps one: scanView/quality.ts).
 *
 * It also digests the tile's own positions, before they are moved (`checksumPositions`, the
 * key `instances.json` lists each tile's object ids under), so the scan's objects can be hidden
 * and highlighted in this renderer too (scanInstances.ts); the order says where each of the
 * tile's own splats went.
 */

import { loadSpz } from "@spz-loader/core";
import { checksumPositions } from "@twin/world";

import {
  centreColumns,
  mortonOrder,
  playcanvasProperties,
  reorderColumns,
} from "@/lib/splatLayout";
import { spzFromGlb } from "@/view/glb";

self.onmessage = async (
  event: MessageEvent<{ id: number; url: string; maxSh?: number }>,
): Promise<void> => {
  const { id, url, maxSh } = event.data;
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`The scan's data answered ${String(response.status)}.`);
    const spz = spzFromGlb(await response.arrayBuffer());
    const cloud = await loadSpz(spz, { unpackOptions: { coordinateSystem: "UNSPECIFIED" } });
    const checksum = checksumPositions(cloud.positions.subarray(0, cloud.numPoints * 3));
    const columns = playcanvasProperties(cloud, maxSh ?? 3);
    const origin = centreColumns(columns);
    const empty = new Float32Array(0);
    const order = mortonOrder(columns.x ?? empty, columns.y ?? empty, columns.z ?? empty);
    const properties = reorderColumns(columns, order);
    self.postMessage(
      { id, count: cloud.numPoints, properties, origin, checksum, order },
      { transfer: [...Object.values(properties).map((values) => values.buffer), order.buffer] },
    );
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};

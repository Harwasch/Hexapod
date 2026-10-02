/**
 * One scan tile for the PlayCanvas splat renderer, off the main thread: fetched, its SPZ taken
 * out of the GLB, decoded with spz-loader (as CesiumJS decodes it) and laid out as the
 * columns PlayCanvas's GSplatData takes (lib/splatLayout.ts), centred on the tile: the
 * positions are sent relative to the middle of the tile's box and the entity is placed there,
 * so they stay small numbers wherever the tile is in the scan (a GPU half float is 6 cm apart
 * 100 m out).
 */

import { loadSpz } from "@spz-loader/core";

import { centreColumns, playcanvasProperties } from "@/lib/splatLayout";
import { spzFromGlb } from "@/view/glb";

self.onmessage = async (event: MessageEvent<{ id: number; url: string }>): Promise<void> => {
  const { id, url } = event.data;
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`The scan's data answered ${String(response.status)}.`);
    const spz = spzFromGlb(await response.arrayBuffer());
    const cloud = await loadSpz(spz, { unpackOptions: { coordinateSystem: "UNSPECIFIED" } });
    const properties = playcanvasProperties(cloud);
    const origin = centreColumns(properties);
    self.postMessage(
      { id, count: cloud.numPoints, properties, origin },
      { transfer: Object.values(properties).map((values) => values.buffer) },
    );
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};

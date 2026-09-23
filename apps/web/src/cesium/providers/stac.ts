/**
 * STAC adapter. A STAC item/collection is metadata: we resolve it to something
 * Cesium can actually stream (a 3D Tiles tileset, a tiled imagery template, or a
 * single georeferenced image). Cloud-optimized GeoTIFFs need a tile server and
 * are reported as such rather than hacked into the client.
 */

import { Rectangle, SingleTileImageryProvider, type ImageryProvider } from "cesium";

import type { Layer } from "@twin/contracts";

interface StacAsset {
  href: string;
  type?: string;
  roles?: string[];
  title?: string;
}

interface StacItem {
  type?: string;
  bbox?: number[];
  assets?: Record<string, StacAsset>;
  links?: { rel: string; href: string; type?: string }[];
}

export type StacResolution =
  | { kind: "3d-tiles"; url: string }
  | { kind: "imagery"; provider: ImageryProvider }
  | { kind: "xyz"; urlTemplate: string }
  | { kind: "unsupported"; reason: string };

export async function resolveStacLayer(
  layer: Layer,
  signal?: AbortSignal,
): Promise<StacResolution> {
  const source = layer.source;
  if (source.type !== "stac") throw new Error("layer is not a STAC source");
  const response = await fetch(source.url, { signal, headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`STAC request failed with HTTP ${response.status}`);
  const item = (await response.json()) as StacItem;
  const assets = item.assets ?? {};
  const candidates = source.assetKey
    ? [assets[source.assetKey]].filter(Boolean)
    : Object.values(assets);
  if (candidates.length === 0) {
    return { kind: "unsupported", reason: "The STAC document has no assets to render." };
  }
  for (const asset of candidates as StacAsset[]) {
    const type = (asset.type ?? "").toLowerCase();
    const href = asset.href;
    if (
      type.includes("3d-tiles") ||
      asset.roles?.includes("3d-tiles") ||
      href.endsWith("tileset.json")
    ) {
      return { kind: "3d-tiles", url: href };
    }
    if (href.includes("{z}") && href.includes("{x}") && href.includes("{y}")) {
      return { kind: "xyz", urlTemplate: href };
    }
    if (
      (type.startsWith("image/png") ||
        type.startsWith("image/jpeg") ||
        type.startsWith("image/webp")) &&
      item.bbox?.length === 4
    ) {
      const [west, south, east, north] = item.bbox as [number, number, number, number];
      const provider = await SingleTileImageryProvider.fromUrl(href, {
        rectangle: Rectangle.fromDegrees(west, south, east, north),
      });
      return { kind: "imagery", provider };
    }
  }
  const cog = (candidates as StacAsset[]).find((a) => (a.type ?? "").includes("geotiff"));
  if (cog) {
    return {
      kind: "unsupported",
      reason:
        "This STAC item only offers Cloud-Optimized GeoTIFF assets. Publish them through a tile server (e.g. TiTiler) and add that service as an imagery layer.",
    };
  }
  return {
    kind: "unsupported",
    reason:
      "No browser-renderable asset (3D Tiles, XYZ tiles, or a georeferenced image) was found.",
  };
}

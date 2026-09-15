import { CesiumTerrainProvider, EllipsoidTerrainProvider, Terrain } from "cesium";

import type { Layer } from "@twin/contracts";

export function isTerrainSource(
  source: Layer["source"],
): source is Extract<Layer["source"], { type: "cesium-ion-terrain" }> {
  return source.type === "cesium-ion-terrain";
}

export function createTerrain(layer: Layer): Terrain {
  const source = layer.source;
  if (!isTerrainSource(source)) throw new Error("layer is not a terrain source");
  return new Terrain(
    CesiumTerrainProvider.fromIonAssetId(source.assetId, {
      requestVertexNormals: source.requestVertexNormals ?? true,
      requestWaterMask: source.requestWaterMask ?? false,
    }),
  );
}

export function flatTerrain(): EllipsoidTerrainProvider {
  return new EllipsoidTerrainProvider();
}

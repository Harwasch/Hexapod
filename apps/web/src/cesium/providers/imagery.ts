import {
  ArcGisMapServerImageryProvider,
  IonImageryProvider,
  Rectangle,
  TileMapServiceImageryProvider,
  UrlTemplateImageryProvider,
  WebMapServiceImageryProvider,
  WebMapTileServiceImageryProvider,
  buildModuleUrl,
  type ImageryProvider,
} from "cesium";

import type { Layer } from "@twin/contracts";

import { creditFor } from "./credit";

type ImagerySource = Extract<
  Layer["source"],
  { type: "cesium-ion-imagery" | "xyz" | "wmts" | "wms" | "arcgis-mapserver" }
>;

export function isImagerySource(source: Layer["source"]): source is ImagerySource {
  return ["cesium-ion-imagery", "xyz", "wmts", "wms", "arcgis-mapserver"].includes(source.type);
}

/** Creates a Cesium imagery provider for a catalog layer. Never throws synchronously. */
export async function createImageryProvider(layer: Layer): Promise<ImageryProvider> {
  const source = layer.source;
  const credit = creditFor(layer.attribution, layer.name);
  const rectangle = layer.spatialExtent
    ? Rectangle.fromDegrees(
        layer.spatialExtent.west,
        layer.spatialExtent.south,
        layer.spatialExtent.east,
        layer.spatialExtent.north,
      )
    : undefined;

  switch (source.type) {
    case "cesium-ion-imagery":
      return IonImageryProvider.fromAssetId(source.assetId);
    case "xyz":
      return new UrlTemplateImageryProvider({
        url: source.urlTemplate,
        minimumLevel: source.minimumLevel ?? 0,
        maximumLevel: source.maximumLevel ?? undefined,
        subdomains: source.subdomains ?? undefined,
        tileWidth: source.tileWidth ?? 256,
        tileHeight: source.tileWidth ?? 256,
        credit,
        rectangle,
      });
    case "wmts":
      return new WebMapTileServiceImageryProvider({
        url: source.url,
        layer: source.layer,
        style: source.style ?? "default",
        format: source.format ?? "image/png",
        tileMatrixSetID: source.tileMatrixSetId,
        maximumLevel: source.maximumLevel ?? undefined,
        credit,
        rectangle,
      });
    case "wms":
      return new WebMapServiceImageryProvider({
        url: source.url,
        layers: source.layers,
        parameters: { transparent: true, format: "image/png", ...(source.parameters ?? {}) },
        credit,
        rectangle,
        enablePickFeatures: false,
      });
    case "arcgis-mapserver":
      return ArcGisMapServerImageryProvider.fromUrl(source.url, {
        layers: source.layers ?? undefined,
        credit,
        rectangle,
        enablePickFeatures: false,
      });
    default:
      throw new Error(`unsupported imagery source ${(source as { type: string }).type}`);
  }
}

/** Offline-safe basemap shipped with CesiumJS; used when no ion imagery is available. */
export function createNaturalEarthProvider(): Promise<TileMapServiceImageryProvider> {
  return TileMapServiceImageryProvider.fromUrl(buildModuleUrl("Assets/Textures/NaturalEarthII"));
}

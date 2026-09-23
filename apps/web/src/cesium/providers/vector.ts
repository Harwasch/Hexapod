import { Color, CzmlDataSource, GeoJsonDataSource, type DataSource } from "cesium";

import type { Layer } from "@twin/contracts";

import { creditFor } from "./credit";

export function isVectorSource(
  source: Layer["source"],
): source is Extract<Layer["source"], { type: "geojson" | "czml" }> {
  return source.type === "geojson" || source.type === "czml";
}

export async function createDataSource(layer: Layer): Promise<DataSource> {
  const source = layer.source;
  const credit = creditFor(layer.attribution, layer.name);
  if (source.type === "geojson") {
    const dataSource = await GeoJsonDataSource.load(source.url, {
      clampToGround: source.clampToGround ?? true,
      stroke: source.stroke
        ? Color.fromCssColorString(source.stroke)
        : Color.fromCssColorString("#0a84ff"),
      fill: source.fill
        ? Color.fromCssColorString(source.fill)
        : Color.fromCssColorString("#0a84ff").withAlpha(0.25),
      strokeWidth: 2,
      credit,
    });
    dataSource.name = layer.id;
    return dataSource;
  }
  if (source.type === "czml") {
    const dataSource = await CzmlDataSource.load(source.url, { credit });
    dataSource.name = layer.id;
    return dataSource;
  }
  throw new Error(`unsupported vector source ${(source as { type: string }).type}`);
}

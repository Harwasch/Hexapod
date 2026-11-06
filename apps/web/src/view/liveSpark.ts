/** The console and its offline check share Spark's precision, AA and snapshot LOD. */
import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import type { WebGLRenderer } from "three";
import { splatAntialiasing } from "./sparkStream";

export type ConsoleBudget = number | "full";

export function consoleSpark(
  renderer: WebGLRenderer,
  budget: ConsoleBudget,
  antialiased?: boolean | null,
): SparkRenderer {
  return new SparkRenderer({
    renderer,
    lodSplatCount: budget === "full" ? undefined : budget,
    enableLod: budget !== "full",
    accumExtSplats: true,
    ...splatAntialiasing(antialiased),
  });
}

export function consoleSplat(bytes: Uint8Array, budget: ConsoleBudget): SplatMesh {
  return new SplatMesh({
    fileBytes: bytes,
    fileType: SplatFileType.SPZ,
    lod: budget !== "full",
    lodAbove: budget === "full" ? undefined : budget,
    extSplats: true,
  });
}

export function sparkSettled(spark: SparkRenderer): boolean {
  return (
    !spark.sorting &&
    !spark.sortDirty &&
    spark.sortTimeoutId === -1 &&
    spark.display === spark.current
  );
}

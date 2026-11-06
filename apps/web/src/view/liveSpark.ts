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
    enableDriveLod: budget !== "full",
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

/** Spark 2.2's LOD exclusive task remains busy after its last RPC response. */
export function sparkWorkersIdle(spark: SparkRenderer): boolean {
  return [spark.sortWorker, spark.lodWorker].every(
    (worker) => !worker || (worker.queue === null && Object.keys(worker.messages).length === 0),
  );
}

/** Sample before rendering: onBeforeRender starts another LOD exclusive task even at rest. */
export function sparkLodSettled(spark: SparkRenderer): boolean {
  return (
    sparkWorkersIdle(spark) &&
    (!spark.enableLod ||
      (!spark.lodDirty &&
        spark.lodInitQueue.length === 0 &&
        spark.lodUpdates.length === 0 &&
        spark.currentLod === spark.lastLod))
  );
}

/**
 * Stop producing work synchronously, then drain the sort/readback and LOD tasks. Spark's
 * dispose() frees targets and rejects worker RPCs but does NOT cancel its update/sort timers.
 * Disposing immediately can therefore produce "No target" or "Worker terminate" after exit.
 * The owner must stop its render loop first, and retain meshes/WebGL until this resolves.
 */
export async function quiesceSpark(spark: SparkRenderer, timeoutMs = 120_000): Promise<void> {
  spark.autoUpdate = false;
  spark.enableDriveLod = false;
  if (spark.updateTimeoutId !== -1) clearTimeout(spark.updateTimeoutId);
  if (spark.sortTimeoutId !== -1) clearTimeout(spark.sortTimeoutId);
  spark.updateTimeoutId = -1;
  spark.sortTimeoutId = -1;
  spark.sortDirty = false; // An in-flight sort must not chain a second one on completion.
  const deadline = performance.now() + timeoutMs;
  while (spark.sorting || !sparkWorkersIdle(spark)) {
    // A crashed worker/driver must not leave an offline check waiting forever in finally.
    // Do not free resources still in use; the caller can close its browser/context on failure.
    if (performance.now() >= deadline) throw new Error("Spark readback/LOD did not drain");
    await new Promise<void>((resolve) => setTimeout(resolve, 16));
  }
}

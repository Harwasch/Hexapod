import type { SparkRenderer } from "@sparkjsdev/spark";
import { afterEach, describe, expect, it, vi } from "vitest";
import { quiesceSpark, sparkLodSettled, sparkSettled } from "@/view/liveSpark";

function state() {
  const accumulator = { numSplats: 12000 };
  return {
    autoUpdate: true,
    enableDriveLod: true,
    enableLod: true,
    updateTimeoutId: -1,
    sortTimeoutId: -1,
    sorting: false,
    sortDirty: false,
    current: accumulator,
    display: accumulator,
    sortWorker: null,
    lodWorker: { queue: null, messages: {} },
    lodDirty: false,
    lodInitQueue: [],
    lodUpdates: [],
    currentLod: undefined,
    lastLod: undefined,
  } as unknown as SparkRenderer;
}
afterEach(() => vi.useRealTimers());

describe("console Spark lifecycle", () => {
  it("cancels queued update/sort callbacks before targets are freed", async () => {
    vi.useFakeTimers();
    const spark = state();
    const readback = vi.fn(() => {
      throw new Error("No target");
    });
    spark.updateTimeoutId = window.setTimeout(readback, 1);
    spark.sortTimeoutId = window.setTimeout(readback, 20);
    spark.sortDirty = true;
    await quiesceSpark(spark);
    expect(spark.autoUpdate).toBe(false);
    expect(spark.enableDriveLod).toBe(false);
    expect(spark.sortDirty).toBe(false);
    await vi.advanceTimersByTimeAsync(100);
    expect(readback).not.toHaveBeenCalled();
    await quiesceSpark(spark); // stopping twice is harmless
  });
  it("retains resources through a readback pause, sort RPC and LOD exclusive tail", async () => {
    vi.useFakeTimers();
    const spark = state();
    spark.sorting = true; // driveSort sets this before its readPause await.
    spark.sortDirty = true;
    spark.lodWorker!.queue = [];
    let freed = false;
    const stop = quiesceSpark(spark).then(() => {
      freed = true;
    });
    await vi.advanceTimersByTimeAsync(32);
    expect(freed).toBe(false);
    spark.sorting = false;
    // RPC responses alone are insufficient: tryExclusive's callback still owns the worker.
    await vi.advanceTimersByTimeAsync(32);
    expect(freed).toBe(false);
    spark.lodWorker!.queue = null;
    await vi.advanceTimersByTimeAsync(32);
    await stop;
    expect(freed).toBe(true);
  });
  it("bounds a failed worker's drain instead of hanging the check's finally forever", async () => {
    vi.useFakeTimers();
    const spark = state();
    spark.sorting = true;
    const stopped = expect(quiesceSpark(spark, 32)).rejects.toThrow("did not drain");
    await vi.advanceTimersByTimeAsync(32);
    await stopped;
  });
  it("does not mistake a sorted parent for completed LOD work", () => {
    const spark = state();
    expect(sparkSettled(spark)).toBe(true);
    expect(sparkLodSettled(spark)).toBe(true);
    spark.lodWorker!.queue = [];
    expect(sparkLodSettled(spark)).toBe(false);
    spark.lodWorker!.queue = null;
    spark.lodDirty = true;
    expect(sparkLodSettled(spark)).toBe(false);
    spark.lodDirty = false;
    spark.lastLod = {} as NonNullable<SparkRenderer["lastLod"]>;
    expect(sparkLodSettled(spark)).toBe(false);
    spark.currentLod = spark.lastLod;
    expect(sparkLodSettled(spark)).toBe(true);
  });
});

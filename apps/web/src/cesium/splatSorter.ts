/**
 * The globe's splat sorter: every splat primitive's steady sorts go to one worker that orders
 * splats back to front by distance from the eye (splatSort.worker.ts), through the patched
 * engine's `GaussianSplatPrimitive.sortHook`.
 *
 * Why replace the engine's WASM sort: it orders by view depth, so every half degree of
 * turning asks for a new sort, and every sort copies all positions on the main thread to send
 * them. By distance, turning needs none (the engine patch then re-sorts only when the camera
 * has moved a centimetre); and the positions go to the worker once -- per written slot range
 * for an incremental primitive, per snapshot generation otherwise -- so a sort request is
 * the eye and a count.
 */

import * as CesiumBarrel from "cesium";

import { createLogger } from "@/lib/log";

const log = createLogger("splat-sort");

interface SortParameters {
  owner: object;
  positions: Float32Array;
  count: number;
  generation: number;
  eye: [number, number, number];
}

type SortHook = ((parameters: SortParameters) => Promise<Uint32Array> | undefined) & {
  sortByDistance?: boolean;
  /** Incremental primitives: a slot range's positions, once, when written. */
  write?: (owner: object, capacity: number, start: number, positions: Float32Array) => void;
};

interface PrimitiveModule {
  sortHook?: SortHook;
}

function primitiveModule(): PrimitiveModule | undefined {
  const candidate = (CesiumBarrel as unknown as Record<string, unknown>).GaussianSplatPrimitive;
  return typeof candidate === "function" ? (candidate as unknown as PrimitiveModule) : undefined;
}

/** Sorts in flight per primitive: one at a time, as the engine asks. */
const MAX_PENDING = 1;

export function installSplatSorter(): () => void {
  const module = primitiveModule();
  if (!module || typeof Worker === "undefined") return () => undefined;
  const worker = new Worker(new URL("./splatSort.worker.ts", import.meta.url), {
    type: "module",
  });
  const owners = new WeakMap<
    object,
    { id: number; generation: number; pending: number; slots: boolean }
  >();
  const ownerOf = (
    primitive: object,
  ): { id: number; generation: number; pending: number; slots: boolean } => {
    let owner = owners.get(primitive);
    if (!owner) {
      owner = { id: nextOwner++, generation: -1, pending: 0, slots: false };
      owners.set(primitive, owner);
    }
    return owner;
  };
  const waiting = new Map<number, (order: Uint32Array | null) => void>();
  let nextOwner = 1;
  let nextRequest = 1;
  worker.onmessage = (event: MessageEvent<{ id: number; order: Uint32Array | null }>) => {
    const resolve = waiting.get(event.data.id);
    waiting.delete(event.data.id);
    resolve?.(event.data.order);
  };
  worker.onerror = (event) => log.warn("sort worker failed", { message: event.message });

  const hook: SortHook = (parameters) => {
    const owner = ownerOf(parameters.owner);
    if (owner.pending >= MAX_PENDING) return undefined;
    // An incremental primitive's positions came as slot writes; others' come whole, once per
    // snapshot generation.
    if (!owner.slots && owner.generation !== parameters.generation) {
      // A new snapshot: its positions, once (copied, since the engine keeps its own).
      const positions = parameters.positions.slice(0, parameters.count * 3);
      worker.postMessage(
        {
          kind: "snapshot",
          owner: owner.id,
          generation: parameters.generation,
          positions,
          count: parameters.count,
        },
        [positions.buffer],
      );
      owner.generation = parameters.generation;
    }
    const id = nextRequest++;
    const current = owner;
    current.pending += 1;
    return new Promise<Uint32Array>((resolve) => {
      waiting.set(id, (order) => {
        current.pending -= 1;
        // A snapshot that changed under the sort answers with no order; an empty one is
        // what the engine takes as "stale, sort again" (a rejection would stop the primitive).
        resolve(order ?? new Uint32Array(0));
      });
      worker.postMessage({
        kind: "sort",
        owner: current.id,
        id,
        generation: parameters.generation,
        count: parameters.count,
        eye: parameters.eye,
      });
    });
  };
  hook.sortByDistance = true;
  hook.write = (primitive, capacity, start, positions) => {
    const owner = ownerOf(primitive);
    owner.slots = true;
    const copy = positions.slice();
    worker.postMessage({ kind: "write", owner: owner.id, capacity, start, positions: copy }, [
      copy.buffer,
    ]);
  };
  module.sortHook = hook;
  return () => {
    if (module.sortHook === hook) module.sortHook = undefined;
    worker.terminate();
  };
}

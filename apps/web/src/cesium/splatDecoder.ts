/**
 * The globe's SPZ decoder: splat tiles decode in a small pool of workers (spzDecode.worker.ts)
 * through the patched engine's `GltfSpzLoader.decodeHook`, instead of on the main thread.
 *
 * Each worker takes at most `MAX_PENDING` tiles at a time; when every one is full the hook
 * answers "not now" and the loader asks again next frame, so arrivals queue in CesiumJS's
 * own priority order rather than in ours, and a burst of them costs the main thread nothing
 * but the copy of each tile's compressed bytes.
 */

import * as CesiumBarrel from "cesium";

import { createLogger } from "@/lib/log";

const log = createLogger("spz-decode");

type DecodeHook = (spz: Uint8Array) => Promise<object> | undefined;

interface LoaderModule {
  decodeHook?: DecodeHook;
}

function loaderModule(): LoaderModule | undefined {
  const candidate = (CesiumBarrel as unknown as Record<string, unknown>).GltfSpzLoader;
  return typeof candidate === "function" ? (candidate as unknown as LoaderModule) : undefined;
}

const MAX_PENDING = 2;

/** Workers: a quarter of the cores, one to three. */
function poolSize(): number {
  const cores = typeof navigator === "undefined" ? 4 : (navigator.hardwareConcurrency ?? 4);
  return Math.min(3, Math.max(1, Math.floor(cores / 4)));
}

interface Slot {
  worker: Worker;
  pending: number;
}

export function installSplatDecoder(): () => void {
  const module = loaderModule();
  if (!module || typeof Worker === "undefined") return () => undefined;
  const waiting = new Map<
    number,
    { slot: Slot; resolve: (o: object) => void; reject: (e: Error) => void }
  >();
  const slots: Slot[] = [];
  let nextId = 1;
  for (let i = 0; i < poolSize(); i++) {
    const slot: Slot = {
      worker: new Worker(new URL("./spzDecode.worker.ts", import.meta.url), { type: "module" }),
      pending: 0,
    };
    slot.worker.onmessage = (
      event: MessageEvent<{ id: number; decoded?: object; error?: string }>,
    ) => {
      const entry = waiting.get(event.data.id);
      if (!entry) return;
      waiting.delete(event.data.id);
      entry.slot.pending -= 1;
      if (event.data.decoded) entry.resolve(event.data.decoded);
      else entry.reject(new Error(event.data.error ?? "SPZ decode failed"));
    };
    slot.worker.onerror = (event) => log.warn("decode worker failed", { message: event.message });
    slots.push(slot);
  }

  const hook: DecodeHook = (spz) => {
    let slot: Slot | undefined;
    for (const candidate of slots) {
      if (candidate.pending < MAX_PENDING && (!slot || candidate.pending < slot.pending)) {
        slot = candidate;
      }
    }
    if (!slot) return undefined;
    const chosen = slot;
    const id = nextId++;
    chosen.pending += 1;
    return new Promise<object>((resolve, reject) => {
      waiting.set(id, { slot: chosen, resolve, reject });
      chosen.worker.postMessage({ id, spz }, [spz.buffer]);
    });
  };
  module.decodeHook = hook;
  return () => {
    if (module.decodeHook === hook) module.decodeHook = undefined;
    for (const slot of slots) slot.worker.terminate();
    for (const entry of waiting.values()) entry.reject(new Error("SPZ decoder closed"));
    waiting.clear();
  };
}

/**
 * The globe's splat sorter: every splat primitive's steady sorts go to one worker that orders
 * splats back to front by distance from the eye (splatSort.worker.ts), through the patched
 * engine's `GaussianSplatPrimitive.sortHook`.
 *
 * Why replace the engine's WASM sort: it orders by view depth, so every half degree of
 * turning asks for a new sort, and every sort copies all positions on the main thread to send
 * them. By distance, turning needs none (the engine patch then re-sorts only when the camera
 * has moved `resortDistance`); and the positions go to the worker once -- per written slot
 * range for an incremental primitive, per snapshot generation otherwise -- so a sort request
 * is the eye and a count. For an incremental primitive the worker also keeps which slots are
 * drawn (show/hide/release), so a tile that left the view stays resident, out of the order.
 *
 * Splats a driver moves rigidly (`splatRigid.ts`) are ordered where they are drawn: the
 * driver names each splat's group and each group's motion (`setSortMotion`), every sort
 * carries the eye moved back by each group's motion (`groupEyes`), and a motion that has
 * moved a group's eye as far as the camera would have to move calls for a sort.
 *
 * The worker keeps a primitive's positions until it is told to `forget` them, which used to
 * be never: a large scan's slot arrays (12 bytes a splat, at 1.6 times the device budget)
 * stayed in the worker for the rest of the visit after its site unloaded -- about 58 MB a
 * scan. CesiumJS says nothing when a primitive goes (a tileset's destroy does not even
 * destroy its splat primitive), so the sorter watches for it itself (`whenGone`): the
 * tileset's or the primitive's own `destroy`, and, for whatever is dropped without either,
 * the garbage collector (`FinalizationRegistry`). Never on hide: slot positions are sent once.
 */

import { GaussianSplatPrimitive } from "cesium";

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
  /** Incremental primitives: a slot range drawn (its tile went live). */
  show?: (owner: object, capacity: number, start: number, count: number) => void;
  /** Incremental primitives: a slot range kept resident but not drawn (its tile left). */
  hide?: (owner: object, capacity: number, start: number, count: number) => void;
  /** Incremental primitives: a slot range freed; left out of orders until written again. */
  release?: (owner: object, capacity: number, start: number, count: number) => void;
  /** Orders leave empty and hidden slots out, so they may be shorter than the slot count. */
  compacts?: boolean;
  /** How far (m) the camera must move before a primitive is sorted again. */
  resortDistance?: (owner: object) => number;
};

interface PrimitiveModule {
  sortHook?: SortHook;
}

function primitiveModule(): PrimitiveModule | undefined {
  const candidate = GaussianSplatPrimitive;
  return typeof candidate === "function" ? (candidate as unknown as PrimitiveModule) : undefined;
}

/** Sorts in flight per primitive: one at a time, as the engine asks. */
const MAX_PENDING = 1;

/**
 * The camera moving this share of the nearest splat's distance calls for a new sort, within
 * these bounds (m). A step that small changes the order only of splats almost equidistant
 * from the eye, whose blending order does not show; sorting on every centimetre re-uploaded
 * the whole order about every third frame of a walk.
 */
export const RESORT_SHARE = 0.05;
export const RESORT_MIN_M = 0.05;
export const RESORT_MAX_M = 2;

export function resortDistance(nearest: number): number {
  if (!Number.isFinite(nearest)) return RESORT_MIN_M;
  return Math.min(RESORT_MAX_M, Math.max(RESORT_MIN_M, nearest * RESORT_SHARE));
}

/** What a driver says of a primitive's rigidly moving splats. */
interface SortMotion {
  /** Per splat index, its group (0: none). */
  groups: Uint16Array;
  /** Per group g, 12 numbers at 12g: the rows of `x' = M x + t` in the positions' frame. */
  motions: Float64Array;
}

const SORT_MOTIONS = new WeakMap<object, SortMotion>();

/**
 * Tells the sorter that `primitive`'s splats move rigidly by group (or, with `undefined`, that
 * none do). `groups` is kept, not copied: hand a new array when membership changes.
 */
export function setSortMotion(primitive: object, motion: SortMotion | undefined): void {
  if (motion) SORT_MOTIONS.set(primitive, motion);
  else SORT_MOTIONS.delete(primitive);
}

/** What the sorter was told of `primitive` (tests, diagnostics). */
export function sortMotionOf(primitive: object): Readonly<SortMotion> | undefined {
  return SORT_MOTIONS.get(primitive);
}

/**
 * Per group, the eye carried back by its motion, `M⁻¹(eye − t)`, at `3g..3g+2`: a splat's
 * distance from it is the moved splat's distance from the eye (exactly, for a rigid `M`).
 * A group whose motion is all zeros (unused) keeps the eye.
 */
export function groupEyes(
  motions: Float64Array,
  eye: readonly [number, number, number],
): Float64Array {
  const groups = Math.floor(motions.length / 12);
  const out = new Float64Array(groups * 3);
  for (let g = 0; g < groups; g += 1) {
    const m = (r: number, c: number): number => motions[g * 12 + r * 4 + c] ?? 0;
    const det =
      m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
      m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
      m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    if (!(Math.abs(det) > 1e-12)) {
      out.set(eye, g * 3);
      continue;
    }
    const v = [eye[0] - m(0, 3), eye[1] - m(1, 3), eye[2] - m(2, 3)];
    // Cramer's rule: M e = v.
    const solve = (col: number): number => {
      const a = (r: number, c: number): number => (c === col ? (v[r] ?? 0) : m(r, c));
      return (
        (a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
          a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
          a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0))) /
        det
      );
    };
    out[g * 3] = solve(0);
    out[g * 3 + 1] = solve(1);
    out[g * 3 + 2] = solve(2);
  }
  return out;
}

/**
 * Calls `gone` once, before `primitive` -- or the tileset it draws (`_tileset`) -- is
 * destroyed: their `destroy` is wrapped on the instance. CesiumJS has no event for it.
 */
export function whenGone(primitive: object, gone: () => void): void {
  let called = false;
  const once = (): void => {
    if (called) return;
    called = true;
    gone();
  };
  const wrap = (target: unknown): void => {
    const holder = target as { destroy?: unknown } | null | undefined;
    const destroy = holder?.destroy;
    if (!holder || typeof destroy !== "function") return;
    holder.destroy = function (this: unknown, ...args: unknown[]): unknown {
      once();
      return (destroy as (...a: unknown[]) => unknown).apply(this, args);
    };
  };
  wrap(primitive);
  wrap((primitive as { _tileset?: unknown })._tileset);
}

export function installSplatSorter(): () => void {
  const module = primitiveModule();
  if (!module || typeof Worker === "undefined") return () => undefined;
  const worker = new Worker(new URL("./splatSort.worker.ts", import.meta.url), {
    type: "module",
  });
  interface Owner {
    id: number;
    generation: number;
    pending: number;
    slots: boolean;
    nearest: number;
    /** The groups last sent to the worker. */
    groups: Uint16Array | undefined;
    /** The eye and the group eyes of the last sort. */
    eye: [number, number, number] | undefined;
    eyes: Float64Array | undefined;
    /** The worker was told to drop its positions: the primitive is gone. */
    forgotten: boolean;
  }
  let stopped = false;
  const forget = (id: number): void => {
    if (!stopped) worker.postMessage({ kind: "forget", owner: id });
  };
  // Whatever goes without a destroy (a tileset dropped, never destroyed) once it is collected.
  const collected =
    typeof FinalizationRegistry === "undefined" ? null : new FinalizationRegistry<number>(forget);
  const owners = new WeakMap<object, Owner>();
  const ownerOf = (primitive: object): Owner => {
    let owner = owners.get(primitive);
    if (!owner) {
      const created: Owner = {
        id: nextOwner++,
        generation: -1,
        pending: 0,
        slots: false,
        nearest: Number.POSITIVE_INFINITY,
        groups: undefined,
        eye: undefined,
        eyes: undefined,
        forgotten: false,
      };
      owner = created;
      owners.set(primitive, created);
      collected?.register(primitive, created.id, created);
      whenGone(primitive, () => {
        if (created.forgotten) return;
        created.forgotten = true;
        collected?.unregister(created);
        forget(created.id);
      });
    }
    return owner;
  };
  const waiting = new Map<number, (order: Uint32Array | null, nearest: number | null) => void>();
  let nextOwner = 1;
  let nextRequest = 1;
  worker.onmessage = (
    event: MessageEvent<{ id: number; order: Uint32Array | null; nearest: number | null }>,
  ) => {
    const resolve = waiting.get(event.data.id);
    waiting.delete(event.data.id);
    resolve?.(event.data.order, event.data.nearest);
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
    const motion = SORT_MOTIONS.get(parameters.owner);
    if (motion?.groups !== owner.groups) {
      const groups = motion ? motion.groups.slice() : null;
      worker.postMessage(
        { kind: "groups", owner: owner.id, groups },
        groups ? [groups.buffer] : [],
      );
      owner.groups = motion?.groups;
    }
    const eyes = motion ? groupEyes(motion.motions, parameters.eye) : undefined;
    owner.eye = [...parameters.eye];
    owner.eyes = eyes;
    const id = nextRequest++;
    const current = owner;
    current.pending += 1;
    return new Promise<Uint32Array>((resolve) => {
      waiting.set(id, (order, nearest) => {
        current.pending -= 1;
        if (nearest !== null) current.nearest = nearest;
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
        ...(eyes ? { eyes } : {}),
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
  const range =
    (kind: "show" | "hide" | "release") =>
    (primitive: object, capacity: number, start: number, count: number): void => {
      const owner = ownerOf(primitive);
      owner.slots = true;
      worker.postMessage({ kind, owner: owner.id, capacity, start, count });
    };
  hook.show = range("show");
  hook.hide = range("hide");
  hook.release = range("release");
  hook.compacts = true;
  hook.resortDistance = (primitive) => {
    const owner = ownerOf(primitive);
    const distance = resortDistance(owner.nearest);
    const motion = SORT_MOTIONS.get(primitive);
    // Membership changed, or a group's eye has moved as far as the camera would have to.
    if (motion?.groups !== owner.groups) return 0;
    if (motion && owner.eye) {
      const now = groupEyes(motion.motions, owner.eye);
      const before = owner.eyes;
      if (before?.length !== now.length) return 0;
      for (let k = 0; k < now.length; k += 3) {
        const d = Math.hypot(
          (now[k] ?? 0) - (before[k] ?? 0),
          (now[k + 1] ?? 0) - (before[k + 1] ?? 0),
          (now[k + 2] ?? 0) - (before[k + 2] ?? 0),
        );
        if (d >= distance) return 0;
      }
    }
    return distance;
  };
  module.sortHook = hook;
  return () => {
    if (module.sortHook === hook) module.sortHook = undefined;
    stopped = true;
    worker.terminate();
  };
}

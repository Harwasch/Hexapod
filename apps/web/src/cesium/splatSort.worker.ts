/**
 * Back-to-front order of a splat snapshot by distance from the eye, off the main thread.
 *
 * Distance, not view depth: an order by distance does not change when the camera turns, only
 * when it moves, so looking around needs no new sort (SuperSplat's viewer sorts the same way).
 * Positions are sent once per snapshot generation and kept here; each sort request carries
 * only the eye, so the main thread no longer copies every position for every sort.
 *
 * Slot mode (the engine's incremental primitive): positions arrive a range at a time
 * (`write`), and a range is drawn only while it is shown -- `show` when its tile goes live,
 * `hide` when it leaves the view but stays resident in its slot, `release` when the slot is
 * freed. Ranges not shown are left out of the order: neither uploaded nor drawn.
 */

import { sortBackToFront } from "@/lib/splatOrder";

interface Snapshot {
  generation: number;
  positions: Float32Array;
  count: number;
  /** Slot mode: which slots are drawn (1) or not (0). */
  live: Uint8Array | undefined;
}

interface RangeRequest {
  owner: number;
  capacity: number;
  start: number;
  count: number;
}
type SortRequest =
  | { kind: "snapshot"; owner: number; generation: number; positions: Float32Array; count: number }
  | { kind: "write"; owner: number; capacity: number; start: number; positions: Float32Array }
  | ({ kind: "show" } & RangeRequest)
  | ({ kind: "hide" } & RangeRequest)
  | ({ kind: "release" } & RangeRequest)
  | {
      kind: "sort";
      owner: number;
      id: number;
      generation: number;
      count: number;
      eye: [number, number, number];
    }
  | { kind: "forget"; owner: number };

const snapshots = new Map<number, Snapshot>();

/** The owner's slot arrays, created (every slot empty and hidden) at a new capacity. */
function slotsOf(owner: number, capacity: number): Snapshot & { live: Uint8Array } {
  const existing = snapshots.get(owner);
  if (existing?.live !== undefined && existing.positions.length === capacity * 3) {
    return existing as Snapshot & { live: Uint8Array };
  }
  const snapshot = {
    generation: -1,
    positions: new Float32Array(capacity * 3).fill(Number.NaN),
    count: 0,
    live: new Uint8Array(capacity),
  };
  snapshots.set(owner, snapshot);
  return snapshot;
}

self.onmessage = (event: MessageEvent<SortRequest>): void => {
  const request = event.data;
  switch (request.kind) {
    case "snapshot":
      snapshots.set(request.owner, {
        generation: request.generation,
        positions: request.positions,
        count: request.count,
        live: undefined,
      });
      return;
    case "write":
      slotsOf(request.owner, request.capacity).positions.set(request.positions, request.start * 3);
      return;
    case "show":
    case "hide":
    case "release": {
      const slots = slotsOf(request.owner, request.capacity);
      const end = request.start + request.count;
      slots.live.fill(request.kind === "show" ? 1 : 0, request.start, end);
      if (request.kind === "release") slots.positions.fill(Number.NaN, request.start * 3, end * 3);
      return;
    }
    case "forget":
      snapshots.delete(request.owner);
      return;
    case "sort":
      break;
  }
  const snapshot = snapshots.get(request.owner);
  // Slot mode sorts whatever is shown, up to the high-water mark the request names; a
  // snapshot must be the one the request was made for.
  const slots = snapshot?.live !== undefined;
  const usable =
    snapshot !== undefined &&
    (slots
      ? request.count * 3 <= snapshot.positions.length
      : snapshot.generation === request.generation);
  if (!usable) {
    self.postMessage({ id: request.id, order: null, nearest: null });
    return;
  }
  const count = slots ? request.count : snapshot.count;
  const { order, nearest } = sortBackToFront(snapshot.positions, count, request.eye, snapshot.live);
  self.postMessage({ id: request.id, order, nearest }, { transfer: [order.buffer] });
};

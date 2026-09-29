/**
 * Back-to-front order of a splat snapshot by distance from the eye, off the main thread.
 *
 * Distance, not view depth: an order by distance does not change when the camera turns, only
 * when it moves, so looking around needs no new sort (SuperSplat's viewer sorts the same way).
 * Positions are sent once per snapshot generation and kept here; each sort request carries
 * only the eye, so the main thread no longer copies every position for every sort.
 */

import { backToFront } from "@/lib/splatOrder";

interface Snapshot {
  generation: number;
  positions: Float32Array;
  count: number;
  /** Slot mode: positions arrive a range at a time (`write`), and stay until overwritten. */
  slots: boolean;
}

type SortRequest =
  | { kind: "snapshot"; owner: number; generation: number; positions: Float32Array; count: number }
  | { kind: "write"; owner: number; capacity: number; start: number; positions: Float32Array }
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
self.onmessage = (event: MessageEvent<SortRequest>): void => {
  const request = event.data;
  if (request.kind === "snapshot") {
    snapshots.set(request.owner, {
      generation: request.generation,
      positions: request.positions,
      count: request.count,
      slots: false,
    });
    return;
  }
  if (request.kind === "write") {
    let snapshot = snapshots.get(request.owner);
    if (snapshot?.slots !== true || snapshot.positions.length !== request.capacity * 3) {
      snapshot = {
        generation: -1,
        positions: new Float32Array(request.capacity * 3),
        count: 0,
        slots: true,
      };
      snapshots.set(request.owner, snapshot);
    }
    snapshot.positions.set(request.positions, request.start * 3);
    return;
  }
  if (request.kind === "forget") {
    snapshots.delete(request.owner);
    return;
  }
  const snapshot = snapshots.get(request.owner);
  // Slot mode sorts whatever is written, up to the high-water mark the request names; a
  // snapshot must be the one the request was made for.
  const usable =
    snapshot !== undefined &&
    (snapshot.slots
      ? request.count * 3 <= snapshot.positions.length
      : snapshot.generation === request.generation);
  if (!usable) {
    self.postMessage({ id: request.id, order: null });
    return;
  }
  const count = snapshot.slots ? request.count : snapshot.count;
  const order = backToFront(snapshot.positions, count, request.eye);
  self.postMessage({ id: request.id, order }, { transfer: [order.buffer] });
};

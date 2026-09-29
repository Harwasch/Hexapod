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
}

type SortRequest =
  | { kind: "snapshot"; owner: number; generation: number; positions: Float32Array; count: number }
  | { kind: "sort"; owner: number; id: number; generation: number; eye: [number, number, number] }
  | { kind: "forget"; owner: number };

const snapshots = new Map<number, Snapshot>();
self.onmessage = (event: MessageEvent<SortRequest>): void => {
  const request = event.data;
  if (request.kind === "snapshot") {
    snapshots.set(request.owner, {
      generation: request.generation,
      positions: request.positions,
      count: request.count,
    });
    return;
  }
  if (request.kind === "forget") {
    snapshots.delete(request.owner);
    return;
  }
  const snapshot = snapshots.get(request.owner);
  if (snapshot?.generation !== request.generation) {
    self.postMessage({ id: request.id, order: null });
    return;
  }
  const order = backToFront(snapshot.positions, snapshot.count, request.eye);
  self.postMessage({ id: request.id, order }, { transfer: [order.buffer] });
};

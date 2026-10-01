/**
 * Swapping tiles without a hole. The streamer (view/stream.ts) swaps a tile for its children
 * in one step, but a splat renderer draws what it is given only after work of its own --
 * Spark builds each new mesh's level-of-detail tree and sorts before the mesh shows, and
 * PlayCanvas sorts its merged buffer in a worker -- so taking the old tile off at once left
 * its patch of the scan empty (the black flash) until the new one drew.
 *
 * Here what goes on screen goes on at once, and what comes off stays until everything put on
 * with it or before it has been drawn (or `maxWaitMs` passes, so a renderer that never says
 * cannot keep a tile forever). For those few frames both are drawn, which reads as detail
 * sharpening rather than a flicker.
 */

export interface HandoverTarget<M> {
  add(mesh: M): void;
  remove(mesh: M): void;
  /** Whether the renderer has drawn `mesh` since it was added `sinceMs` ago. */
  isDrawn(mesh: M, sinceMs: number): boolean;
}

/**
 * Longest a replaced tile waits for its replacement to draw. Long: Spark builds each new
 * mesh's level-of-detail tree one at a time, so walking into the middle of a scan queues
 * dozens and the last are seconds away -- at 1.5 s their parents came off first and the
 * ground showed black where the world is clipped away under the scan. Waiting costs only a
 * coarse tile drawn over a fine one for those seconds.
 */
export const HANDOVER_MAX_MS = 10_000;

export class Handover<M> {
  /** On screen, not drawn yet: when each was added. */
  private readonly settling = new Map<M, number>();
  /** Off the streamer's cut, still on screen until what replaces it draws. */
  private readonly retiring = new Map<M, { waitFor: M[]; since: number }>();

  constructor(
    private readonly target: HandoverTarget<M>,
    private readonly maxWaitMs = HANDOVER_MAX_MS,
  ) {}

  show(mesh: M, now: number): void {
    // Swapped back before it came off: it never left the screen.
    if (this.retiring.delete(mesh)) return;
    this.target.add(mesh);
    this.settling.set(mesh, now);
  }

  hide(mesh: M, now: number): void {
    if (this.settling.delete(mesh)) {
      // Never drawn, so nothing on screen depends on it.
      this.target.remove(mesh);
      return;
    }
    this.retiring.set(mesh, { waitFor: [...this.settling.keys()], since: now });
  }

  /** Before `mesh` is disposed: off the screen now, whatever it was waiting for. */
  forget(mesh: M): void {
    if (this.retiring.delete(mesh)) this.target.remove(mesh);
    this.settling.delete(mesh);
  }

  /** After each frame: settles what has drawn, and retires what no longer needs to stay. */
  tick(now: number): void {
    for (const [mesh, added] of this.settling) {
      if (this.target.isDrawn(mesh, now - added)) this.settling.delete(mesh);
    }
    for (const [mesh, { waitFor, since }] of this.retiring) {
      const ready = waitFor.every((waiting) => !this.settling.has(waiting));
      if (ready || now - since > this.maxWaitMs) {
        this.retiring.delete(mesh);
        this.target.remove(mesh);
      }
    }
  }

  /** Tiles kept on screen past their cut (tests, the debug panel). */
  get retained(): number {
    return this.retiring.size;
  }
}

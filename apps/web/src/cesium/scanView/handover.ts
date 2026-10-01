/**
 * Swapping tiles without a hole, and without a pop. The streamer (view/stream.ts) swaps a
 * tile for its children in one step, but a splat renderer draws what it is given only after
 * work of its own -- Spark sorts before a new mesh shows, and PlayCanvas sorts its merged
 * buffer in a worker -- so taking the old tile off at once left its patch of the scan empty
 * (the black flash) until the new one drew.
 *
 * Here what goes on screen goes on at once, and what comes off stays until everything put on
 * with it or before it has been drawn (or `maxWaitMs` passes, so a renderer that never says
 * cannot keep a tile forever). Where the renderer can fade a mesh, the new one fades in over
 * `FADE_MS` from when it goes on (never held fully transparent: a renderer may skip what it
 * cannot see, and never say it drew it), and the old one comes off only once the new one is
 * drawn and in -- a game's level-of-detail cross-fade: detail sharpens instead of popping.
 */

export interface HandoverTarget<M> {
  add(mesh: M): void;
  remove(mesh: M): void;
  /** Whether the renderer has drawn `mesh` since it was added `sinceMs` ago. */
  isDrawn(mesh: M, sinceMs: number): boolean;
  /** Sets how opaque `mesh` is drawn, 0 to 1, when the renderer can. */
  fade?(mesh: M, alpha: number): void;
}

/**
 * Longest a replaced tile waits for its replacement to draw. Long: a renderer can take
 * seconds to show the last of dozens of tiles that arrive together, and taking parents off
 * first showed black where the world is clipped away under the scan. Waiting costs only a
 * coarse tile drawn under a fine one for those seconds.
 */
export const HANDOVER_MAX_MS = 10_000;
/** How long new detail takes to fade in. */
export const FADE_MS = 300;

export class Handover<M> {
  /** On screen, not drawn yet: when each was added. */
  private readonly settling = new Map<M, number>();
  /** Fading in: when each went on. */
  private readonly fading = new Map<M, number>();
  /** Off the streamer's cut, still on screen until what replaces it is in. */
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
    if (this.target.fade) {
      this.target.fade(mesh, 0);
      this.fading.set(mesh, now);
    }
  }

  hide(mesh: M, now: number): void {
    const settling = this.settling.delete(mesh);
    const fading = this.fading.delete(mesh);
    if (settling || fading) {
      // Never fully in, so nothing on screen depends on it.
      this.target.remove(mesh);
      return;
    }
    this.retiring.set(mesh, {
      waitFor: [...this.settling.keys(), ...this.fading.keys()],
      since: now,
    });
  }

  /** Before `mesh` is disposed: off the screen now, whatever it was waiting for. */
  forget(mesh: M): void {
    if (this.retiring.delete(mesh)) this.target.remove(mesh);
    this.settling.delete(mesh);
    this.fading.delete(mesh);
  }

  /** After each frame: settles what has drawn, fades it in, and retires what is replaced. */
  tick(now: number): void {
    for (const [mesh, added] of this.settling) {
      if (this.target.isDrawn(mesh, now - added)) this.settling.delete(mesh);
    }
    for (const [mesh, started] of this.fading) {
      const alpha = Math.min(1, (now - started) / FADE_MS);
      this.target.fade?.(mesh, alpha);
      if (alpha >= 1) this.fading.delete(mesh);
    }
    for (const [mesh, { waitFor, since }] of this.retiring) {
      const ready = waitFor.every(
        (waiting) => !this.settling.has(waiting) && !this.fading.has(waiting),
      );
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

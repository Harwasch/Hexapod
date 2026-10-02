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
 *
 * The overlay draws only when something changes (overlayFrames.ts), so each tick says what it
 * still needs (`HandoverStep`): another frame while a fade runs or once a tile came off, and
 * the time a replaced tile stops waiting. Whether a new tile has drawn is the renderer's to
 * say when it knows (a sort finished: it asks for a frame itself), so waiting for that costs
 * no frames.
 */

/** What a tick leaves for the frames after it. */
export interface HandoverStep {
  /** Something on screen changed in this tick (a fade moved, a tile came off): draw it. */
  changed: boolean;
  /** A fade is still running: draw every frame. */
  animating: boolean;
  /** When a replaced tile stops waiting for its replacement regardless, or null. */
  nextAt: number | null;
}

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
  tick(now: number): HandoverStep {
    let changed = false;
    for (const [mesh, added] of this.settling) {
      if (this.target.isDrawn(mesh, now - added)) this.settling.delete(mesh);
    }
    for (const [mesh, started] of this.fading) {
      const alpha = Math.min(1, (now - started) / FADE_MS);
      this.target.fade?.(mesh, alpha);
      changed = true;
      if (alpha >= 1) this.fading.delete(mesh);
    }
    let nextAt: number | null = null;
    for (const [mesh, { waitFor, since }] of this.retiring) {
      const ready = waitFor.every(
        (waiting) => !this.settling.has(waiting) && !this.fading.has(waiting),
      );
      if (ready || now - since > this.maxWaitMs) {
        this.retiring.delete(mesh);
        this.target.remove(mesh);
        changed = true;
      } else {
        // Just past the longest wait: `> maxWaitMs` above.
        const giveUp = since + this.maxWaitMs + 1;
        nextAt = nextAt === null ? giveUp : Math.min(nextAt, giveUp);
      }
    }
    return { changed, animating: this.fading.size > 0, nextAt };
  }

  /** Tiles kept on screen past their cut (tests, the debug panel). */
  get retained(): number {
    return this.retiring.size;
  }

  /** Whether anything is still settling, fading or waiting to come off. */
  get busy(): boolean {
    return this.settling.size > 0 || this.fading.size > 0 || this.retiring.size > 0;
  }
}

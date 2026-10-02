/**
 * The main-thread part of turning a fetched tile into something a renderer draws, a little at
 * a time while the camera moves.
 *
 * Fetching and decoding happen off the main thread (a worker per renderer), but what comes
 * back still has to be made into the renderer's own object on it: PlayCanvas packs every
 * splat's centre, covariance and colour into its textures as it builds a `GSplatResource`, and
 * Spark's tiles are digested for their object ids. That was done the moment each tile
 * arrived, however many arrived together and whatever the camera was doing -- three tiles
 * landing in one frame of a pan were three resources built in that frame, and the pan
 * hitched. A game streamer budgets this: so much main-thread time a frame for loading, less
 * while the player moves.
 *
 * `TileWork.run` queues a job and runs it when the current animation frame still has budget
 * for it: `MOVING_BUDGET_MS` a frame while the camera moves, `RESTING_BUDGET_MS` at rest (a
 * click still lands promptly). A job cannot be split, so one always runs in a frame that has
 * spent nothing yet, however long it takes; the next waits for the next frame.
 */

/** Main-thread milliseconds a frame for tile work, while the camera moves and at rest. */
export const MOVING_BUDGET_MS = 4;
export const RESTING_BUDGET_MS = 12;

export interface WorkClock {
  now(): number;
  nextFrame(callback: () => void): void;
}

const browserClock: WorkClock = {
  now: () => performance.now(),
  nextFrame: (callback) => {
    requestAnimationFrame(() => callback());
  },
};

export class TileWork {
  /** Whether the camera moves (the host says, each overlay frame). */
  moving = false;
  private readonly queue: (() => void)[] = [];
  /** Spent in the current frame, and whether a frame boundary is awaited. */
  private spent = 0;
  private waiting = false;
  private stopped = false;
  /** Jobs run, and how often the queue had to wait for a later frame (tests, diagnostics). */
  ran = 0;
  deferred = 0;

  constructor(private readonly clock: WorkClock = browserClock) {}

  /** Runs `job` on the main thread within the frame budget; resolves with what it returns. */
  run<T>(job: () => T): Promise<T> {
    if (this.stopped) return Promise.reject(new Error("The renderer stopped."));
    return new Promise<T>((resolve, reject) => {
      this.queue.push(() => {
        try {
          resolve(job());
        } catch (error) {
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      });
      this.pump();
    });
  }

  /** Jobs waiting for a frame with budget. */
  get queued(): number {
    return this.queue.length;
  }

  /** Drops what is queued (their promises never settle) and refuses more. */
  stop(): void {
    this.stopped = true;
    this.queue.length = 0;
  }

  private get budget(): number {
    return this.moving ? MOVING_BUDGET_MS : RESTING_BUDGET_MS;
  }

  private pump(): void {
    while (this.queue.length > 0 && !this.stopped) {
      if (this.spent > 0 && this.spent >= this.budget) {
        this.deferred += 1;
        this.awaitFrame();
        return;
      }
      const job = this.queue.shift();
      const started = this.clock.now();
      job?.();
      this.ran += 1;
      // At least a little, so a frame that ran anything counts as having spent.
      this.spent += Math.max(0.01, this.clock.now() - started);
    }
    // Spent this frame; the budget is fresh once the next one starts.
    if (this.spent > 0) this.awaitFrame();
  }

  private awaitFrame(): void {
    if (this.waiting) return;
    this.waiting = true;
    this.clock.nextFrame(() => {
      this.waiting = false;
      this.spent = 0;
      this.pump();
    });
  }
}

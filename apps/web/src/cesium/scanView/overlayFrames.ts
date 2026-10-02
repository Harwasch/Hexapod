/**
 * When the splat overlay (ScanRendererHost) draws: only when something it draws from changed.
 *
 * The globe renders on demand (request-render mode, CesiumSceneManager): a still view costs
 * nothing. The overlay used to draw on every display frame the globe did not -- the identical
 * frame, sixty times a second, for as long as the page was open -- because it had no way to
 * know that nothing had changed. Now it is told, the way the globe is:
 *
 * - **The globe rendered** (`postRender`) and what the overlay is drawn from moved with it --
 *   the camera, the canvas, the scan's frame (`OverlayInputs`). Wind on another site redraws
 *   the globe every tick without touching any of that, and draws nothing here.
 * - **Something woke it** (`wake`): a tile arrived or failed, a deferred swap, the renderer
 *   finished a sort or streamed new detail (PlayCanvas's `frame:request`, Spark's `onDirty`), a
 *   hide or highlight changed, the budget moved. The frame is drawn on the next display frame,
 *   with the globe's if the globe draws one.
 * - **A deadline came** (`wakeBy`): the full-resolution frame once the camera has rested, the
 *   re-plan the 150 ms throttle held back, a replaced tile's longest wait.
 *
 * A frame says what it still needs (`FrameOutcome`): another at once while something animates
 * (a fade) or a change it made is not on screen yet, and a time by which it wants the next.
 * While detail is still arriving and sharpening that is a frame every display frame, as
 * before; once everything is in and still, nothing at all.
 *
 * In step with the globe: a frame is drawn right after CesiumJS renders (its `postRender`, so
 * the overlay is drawn from exactly the camera the globe was), and a wake on a frame the globe
 * does not render is drawn from the animation frame -- but never a camera pose ahead of the
 * globe: a moved camera waits up to `MAX_WAIT_FOR_GLOBE` frames for the globe's own render.
 * Drawing only from the animation frame put the overlay a pose behind whenever its callback ran
 * before CesiumJS's -- the order of animation-frame callbacks, which a render-loop restart
 * (render-error recovery) flips -- and the scan slid on the map as the view moved.
 */

import { Cartesian3, Matrix4, type Camera } from "cesium";

import { countOverlayWake } from "./stats";

/** Frames in a row the overlay waits for the globe's own while the camera moves. */
export const MAX_WAIT_FOR_GLOBE = 2;

/** What one overlay frame asks of the next. */
export interface FrameOutcome {
  /** Another frame on the next display frame: something animates, or a change is not drawn. */
  again: boolean;
  /** A frame by this time (`performance.now()` milliseconds) at the latest, or null. */
  by: number | null;
}

/** The clock and the frame and timer callbacks, injectable for tests. */
export interface FrameClock {
  now(): number;
  requestFrame(callback: () => void): number;
  cancelFrame(handle: number): void;
  setTimer(callback: () => void, ms: number): unknown;
  clearTimer(handle: unknown): void;
}

export const browserFrameClock: FrameClock = {
  now: () => performance.now(),
  requestFrame: (callback) => requestAnimationFrame(callback),
  cancelFrame: (handle) => cancelAnimationFrame(handle),
  setTimer: (callback, ms) => setTimeout(callback, ms),
  clearTimer: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

/** What drives the overlay: whether its inputs moved, and drawing a frame. */
export interface FrameSource {
  /** Whether what a frame is drawn from differs from what the last frame was drawn from. */
  changed(): boolean;
  draw(): FrameOutcome;
}

/**
 * Schedules the overlay's frames (see the file comment). `onGlobeRender` registers a listener
 * for the globe's renders and returns its remover (`scene.postRender.addEventListener`).
 */
export class OverlayFrames {
  /** A frame is wanted and not drawn yet. */
  private pending = false;
  /** The globe rendered since the last animation frame. */
  private globeDrew = false;
  private waited = 0;
  private frame: number | null = null;
  private timer: unknown = null;
  private deadline = Number.POSITIVE_INFINITY;
  private deadlineReason = "deadline";
  private stopped = false;
  private readonly removeGlobeListener: () => void;

  constructor(
    onGlobeRender: (listener: () => void) => () => void,
    private readonly source: FrameSource,
    private readonly clock: FrameClock = browserFrameClock,
  ) {
    this.removeGlobeListener = onGlobeRender(() => this.globeRendered());
  }

  /** Whether a frame is wanted and not drawn yet (tests). */
  get waiting(): boolean {
    return this.pending;
  }

  /** Asks for a frame on the next display frame (or the globe's, if it renders first). */
  wake(reason: string): void {
    if (this.stopped) return;
    countOverlayWake(reason);
    this.pending = true;
    this.schedule();
  }

  /**
   * Asks for a frame by `time` at the latest. One timer serves every deadline: an earlier one
   * replaces a later one, and the frame it draws asks again for whatever is still due.
   */
  wakeBy(time: number, reason: string): void {
    if (this.stopped || time >= this.deadline) return;
    if (this.timer !== null) this.clock.clearTimer(this.timer);
    this.deadline = time;
    this.deadlineReason = reason;
    this.timer = this.clock.setTimer(
      () => {
        this.timer = null;
        this.deadline = Number.POSITIVE_INFINITY;
        this.wake(this.deadlineReason);
      },
      Math.max(0, time - this.clock.now()),
    );
  }

  stop(): void {
    this.stopped = true;
    this.pending = false;
    this.removeGlobeListener();
    if (this.frame !== null) this.clock.cancelFrame(this.frame);
    if (this.timer !== null) this.clock.clearTimer(this.timer);
    this.frame = null;
    this.timer = null;
  }

  private schedule(): void {
    if (this.frame === null && !this.stopped) {
      this.frame = this.clock.requestFrame(this.tick);
    }
  }

  private globeRendered(): void {
    if (this.stopped) return;
    this.globeDrew = true;
    this.waited = 0;
    if (this.pending || this.source.changed()) this.drawNow();
  }

  private readonly tick = (): void => {
    this.frame = null;
    const drew = this.globeDrew;
    this.globeDrew = false;
    if (this.stopped || !this.pending) return;
    // Drawn with the globe this display frame already; what woke it since waits for the next.
    if (drew) {
      this.schedule();
      return;
    }
    // Moved, and the globe has not drawn it yet: it will this frame.
    if (this.source.changed() && this.waited++ < MAX_WAIT_FOR_GLOBE) {
      this.schedule();
      return;
    }
    this.waited = 0;
    this.drawNow();
  };

  private drawNow(): void {
    // Cleared first: a wake while drawing (a renderer asking for another frame) stands.
    this.pending = false;
    const outcome = this.source.draw();
    if (outcome.again) this.wake("again");
    if (outcome.by !== null) this.wakeBy(outcome.by, "deadline");
  }
}

/** Slots of an `OverlayInputs` snapshot and how far each may move before it counts. */
const POSITION = 0;
const DIRECTION = 3;
const UP = 6;
const FRUSTUM = 9;
const SIZE = 12;
const TRANSFORM = 15;
const SLOTS = TRANSFORM + 16;
/** Metres: the camera's position, as the overlay's motion test always measured it. */
const POSITION_EPSILON = 1e-3;
/** Unit vectors: direction and up. */
const DIRECTION_EPSILON = 1e-5;

/**
 * Everything one overlay frame is drawn from, as numbers: the camera (position, direction and
 * up in Earth-fixed coordinates, field of view, near and far), the canvas (CSS size and the
 * pixel ratio it is drawn at) and the scan's frame (the tileset root's transform). Two frames
 * from equal snapshots draw the same pixels, so the second is not drawn.
 */
export class OverlayInputs {
  private readonly last = new Float64Array(SLOTS).fill(Number.NaN);
  private readonly next = new Float64Array(SLOTS);

  /** Whether the inputs differ from the last `commit`. */
  changed(
    camera: Camera,
    size: { width: number; height: number; pixelRatio: number },
    transform: Matrix4,
  ): boolean {
    this.capture(camera, size, transform, this.next);
    const a = this.last;
    const b = this.next;
    for (let i = 0; i < SLOTS; i++) {
      const tolerance = i < DIRECTION ? POSITION_EPSILON : i < FRUSTUM ? DIRECTION_EPSILON : 0;
      const x = a[i] ?? Number.NaN;
      const y = b[i] ?? Number.NaN;
      if (!(Math.abs(x - y) <= tolerance)) return true;
    }
    return false;
  }

  /** Records the inputs a frame was just drawn from. */
  commit(
    camera: Camera,
    size: { width: number; height: number; pixelRatio: number },
    transform: Matrix4,
  ): void {
    this.capture(camera, size, transform, this.last);
  }

  private capture(
    camera: Camera,
    size: { width: number; height: number; pixelRatio: number },
    transform: Matrix4,
    out: Float64Array,
  ): void {
    Cartesian3.pack(camera.positionWC, out as unknown as number[], POSITION);
    Cartesian3.pack(camera.directionWC, out as unknown as number[], DIRECTION);
    Cartesian3.pack(camera.upWC, out as unknown as number[], UP);
    const frustum = camera.frustum as { fovy?: number; near?: number; far?: number };
    out[FRUSTUM] = frustum.fovy ?? 0;
    out[FRUSTUM + 1] = frustum.near ?? 0;
    out[FRUSTUM + 2] = frustum.far ?? 0;
    out[SIZE] = size.width;
    out[SIZE + 1] = size.height;
    out[SIZE + 2] = size.pixelRatio;
    Matrix4.pack(transform, out as unknown as number[], TRANSFORM);
  }
}

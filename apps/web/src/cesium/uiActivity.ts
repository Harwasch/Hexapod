/**
 * Whether the person is working the interface -- a button, a menu, a field, a slider -- so
 * that streaming defers to it the way it defers to the camera (splatMotionGate.ts): for
 * `UI_PRIORITY_MS` after each such press, splat batches shrink to their moving size and no
 * new tile decode starts, so the frames in between are mostly the interface's.
 *
 * Only presses on interactive elements count: the globe's own canvas, and keys that walk or
 * fly the camera (their target is the page), are the camera's business.
 *
 * The interface also *holds* the 3D view (`UiActivity.holding`) while a popover is open over
 * the map (the site switcher, the command box, the map menu: anything marked
 * `data-hud-popover`), and for `UI_PRIORITY_MS` after a press on a control or the pointer
 * moved over one. Held, with the camera still, the redraws nobody on screen asked for -- a tile
 * landing, a sort result, the globe's own tile loads -- come at most every `HELD_FRAME_MS`
 * (`heldRedrawDelay`), the splat overlay re-plans at most every `HELD_REPLAN_MS`, and its tile
 * work is paced to the interface (scanView/tileWork.ts): the menu opening, its hover, its
 * scroll get the main thread and the GPU, not a scan sharpening behind it. A camera move is
 * never held: the globe and the overlay draw it at once, and everything streams as before.
 */

import type { Scene } from "cesium";

/** How long a press on the interface holds streaming back (ms). */
export const UI_PRIORITY_MS = 250;

/**
 * While the interface holds the view, at most one redraw this often (ms) that the camera did
 * not ask for: tiles still land, four times a second instead of every display frame.
 */
export const HELD_FRAME_MS = 250;

/** While the interface holds the view, the splat overlay re-plans its tile cut at most this
 *  often (ms), instead of every 150 ms. */
export const HELD_REPLAN_MS = 1000;

/** What counts as a popover over the map: while one is open, the view is held. */
export const POPOVER_SELECTOR = "[data-hud-popover]";

const INTERACTIVE =
  "button, a, input, select, textarea, summary, [role=button], [role=menu], [role=menuitem], " +
  "[role=listbox], [role=option], [role=tab], [role=slider], [role=switch], [role=checkbox], " +
  "[role=combobox], [role=dialog], [contenteditable=true]";

/** Whether an event's target is part of the interface (not the canvas, not the page). */
export function isInterfaceTarget(target: EventTarget | null, canvas: Element): boolean {
  if (!(target instanceof Element) || canvas.contains(target)) return false;
  return target.closest(INTERACTIVE) !== null;
}

/**
 * How long a redraw the camera did not ask for must still wait (ms) while the interface holds
 * the view: none once `HELD_FRAME_MS` have passed since the last one drawn, and none at all
 * when the view is not held. 0 means draw it now.
 */
export function heldRedrawDelay(
  holding: boolean,
  lastDrawAt: number,
  now: number,
  gapMs: number = HELD_FRAME_MS,
): number {
  if (!holding) return 0;
  return Math.max(0, lastDrawAt + gapMs - now);
}

export class UiActivity {
  private until = 0;
  /** Until when the pointer's latest move over a control holds the view. */
  private hoverUntil = 0;
  /** A popover was added or removed since `popoverOpen` last looked. */
  private popoverDirty = true;
  private popover = false;
  private readonly off: (() => void)[] = [];

  constructor(private readonly canvas: Element) {
    const note = (event: Event): void => {
      if (isInterfaceTarget(event.target, this.canvas))
        this.until = performance.now() + UI_PRIORITY_MS;
    };
    for (const type of ["pointerdown", "keydown", "input", "wheel"] as const) {
      document.addEventListener(type, note, { capture: true, passive: true });
      this.off.push(() => document.removeEventListener(type, note, { capture: true }));
    }
    // Pointing at controls (a hover) holds the view without counting as a press: the splat
    // decoders keep their pace (`active`), the redraws theirs (`holding`).
    const hover = (event: Event): void => {
      if (isInterfaceTarget(event.target, this.canvas))
        this.hoverUntil = performance.now() + UI_PRIORITY_MS;
    };
    document.addEventListener("pointermove", hover, { capture: true, passive: true });
    this.off.push(() => document.removeEventListener("pointermove", hover, { capture: true }));
    // Popovers come and go with React's commits; whether one is open is looked up only when
    // asked, and only after the page's elements changed.
    if (typeof MutationObserver !== "undefined" && document.body) {
      const observer = new MutationObserver(() => {
        this.popoverDirty = true;
      });
      observer.observe(document.body, { childList: true, subtree: true });
      this.off.push(() => observer.disconnect());
    }
  }

  /** The interface was used within the last `UI_PRIORITY_MS`. */
  get active(): boolean {
    return performance.now() < this.until;
  }

  /** A popover is open over the map (`POPOVER_SELECTOR`). */
  get popoverOpen(): boolean {
    if (this.popoverDirty) {
      this.popoverDirty = false;
      this.popover = document.querySelector(POPOVER_SELECTOR) !== null;
    }
    return this.popover;
  }

  /** The interface holds the 3D view: a popover is open, or a control was just used or
   *  pointed at. */
  get holding(): boolean {
    return this.active || performance.now() < this.hoverUntil || this.popoverOpen;
  }

  destroy(): void {
    for (const off of this.off) off();
  }
}

/** The private flag CesiumJS sets for `requestRender` and clears when it renders. */
interface RenderRequest {
  _renderRequested?: boolean;
}

/**
 * Holds the globe's requested renders while the interface holds the view (`holding`): a
 * `requestRender` -- a tile loaded, a sort result, a manager's change -- is drawn at most every
 * `HELD_FRAME_MS`, and the one held back is drawn once that time has passed. A camera that
 * moved still renders at once (CesiumJS's own camera check, which this leaves alone), and so
 * does everything once the interface lets go. Returns the uninstaller.
 */
export function holdGlobeRenders(
  scene: Pick<Scene, "preUpdate" | "postRender" | "requestRender">,
  holding: () => boolean,
  clock: {
    now(): number;
    setTimer(run: () => void, ms: number): unknown;
    clearTimer(handle: unknown): void;
  } = {
    now: () => performance.now(),
    setTimer: (run, ms) => setTimeout(run, ms),
    clearTimer: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
  },
): () => void {
  const request = scene as unknown as RenderRequest;
  let lastRenderAt = Number.NEGATIVE_INFINITY;
  let timer: unknown = null;
  const removePre = scene.preUpdate.addEventListener(() => {
    if (request._renderRequested !== true) return;
    const wait = heldRedrawDelay(holding(), lastRenderAt, clock.now());
    if (wait <= 0) return;
    // Not this frame: asked for again once the wait is over (a camera move renders anyway).
    request._renderRequested = false;
    if (timer === null) {
      timer = clock.setTimer(() => {
        timer = null;
        scene.requestRender();
      }, wait);
    }
  });
  const removePost = scene.postRender.addEventListener(() => {
    lastRenderAt = clock.now();
  });
  return () => {
    removePre();
    removePost();
    if (timer !== null) clock.clearTimer(timer);
    timer = null;
  };
}

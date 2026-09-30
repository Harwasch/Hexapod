/**
 * Whether the person is working the interface -- a button, a menu, a field, a slider -- so
 * that streaming defers to it the way it defers to the camera (splatMotionGate.ts): for
 * `UI_PRIORITY_MS` after each such press, splat batches shrink to their moving size and no
 * new tile decode starts, so the frames in between are mostly the interface's.
 *
 * Only presses on interactive elements count: the globe's own canvas, and keys that walk or
 * fly the camera (their target is the page), are the camera's business.
 */

/** How long a press on the interface holds streaming back (ms). */
export const UI_PRIORITY_MS = 250;

const INTERACTIVE =
  "button, a, input, select, textarea, summary, [role=button], [role=menu], [role=menuitem], " +
  "[role=listbox], [role=option], [role=tab], [role=slider], [role=switch], [role=checkbox], " +
  "[role=combobox], [role=dialog], [contenteditable=true]";

/** Whether an event's target is part of the interface (not the canvas, not the page). */
export function isInterfaceTarget(target: EventTarget | null, canvas: Element): boolean {
  if (!(target instanceof Element) || canvas.contains(target)) return false;
  return target.closest(INTERACTIVE) !== null;
}

export class UiActivity {
  private until = 0;
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
  }

  /** The interface was used within the last `UI_PRIORITY_MS`. */
  get active(): boolean {
    return performance.now() < this.until;
  }

  destroy(): void {
    for (const off of this.off) off();
  }
}

/** How long a finger must rest on the map before the map menu opens. */
export const LONG_PRESS_MS = 550;
/** How far it may wander meanwhile and still be a press, not a pan (CSS px). */
export const LONG_PRESS_SLOP_PX = 10;

/**
 * A finger held still on the map: the touch screen's right-click. Our own timer on the
 * canvas's pointer events, because iOS raises no `contextmenu` for a touch and Cesium's own
 * touch hold (a RIGHT_CLICK after 1.5 s) is too slow to read as a press. A move past the slop,
 * a second finger (a pinch) or the finger lifting cancels it. Once it fired, the click the
 * lifting finger would make is not a click on the map (`takeClick`).
 */
export class LongPress {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private start: { id: number; x: number; y: number } | null = null;
  private readonly touches = new Set<number>();
  private fired = false;

  /** `press` gets CSS px from the element's top left; true when it took the press. */
  constructor(
    private readonly element: HTMLElement,
    private readonly press: (x: number, y: number) => boolean,
    private readonly delayMs = LONG_PRESS_MS,
  ) {
    element.addEventListener("pointerdown", this.onDown);
    window.addEventListener("pointermove", this.onMove);
    window.addEventListener("pointerup", this.onUp);
    window.addEventListener("pointercancel", this.onUp);
  }

  /** A finger is on the map: Cesium's own touch hold, raised as a RIGHT_CLICK, is not ours. */
  get touching(): boolean {
    return this.touches.size > 0;
  }

  /** True, once, for the click that follows a long press that opened the menu. */
  takeClick(): boolean {
    const fired = this.fired;
    this.fired = false;
    return fired;
  }

  private readonly onDown = (event: PointerEvent): void => {
    if (event.pointerType === "mouse") return;
    // A new touch is a new gesture: the last press's click, if it was going to come, came.
    this.fired = false;
    this.touches.add(event.pointerId);
    if (this.touches.size > 1) {
      this.cancel();
      return;
    }
    const rect = this.element.getBoundingClientRect();
    this.start = { id: event.pointerId, x: event.clientX - rect.left, y: event.clientY - rect.top };
    this.timer = setTimeout(() => {
      this.timer = null;
      const start = this.start;
      if (start) this.fired = this.press(start.x, start.y);
    }, this.delayMs);
  };

  private readonly onMove = (event: PointerEvent): void => {
    const start = this.start;
    if (start?.id !== event.pointerId || this.timer === null) return;
    const rect = this.element.getBoundingClientRect();
    const dx = event.clientX - rect.left - start.x;
    const dy = event.clientY - rect.top - start.y;
    if (Math.hypot(dx, dy) > LONG_PRESS_SLOP_PX) this.cancel();
  };

  private readonly onUp = (event: PointerEvent): void => {
    if (!this.touches.delete(event.pointerId)) return;
    if (event.pointerId === this.start?.id) this.cancel();
  };

  private cancel(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.start = null;
  }

  destroy(): void {
    this.cancel();
    this.element.removeEventListener("pointerdown", this.onDown);
    window.removeEventListener("pointermove", this.onMove);
    window.removeEventListener("pointerup", this.onUp);
    window.removeEventListener("pointercancel", this.onUp);
  }
}

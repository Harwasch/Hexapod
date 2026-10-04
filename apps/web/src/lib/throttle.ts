/**
 * Trailing-edge throttle used to rate-limit Cesium → React state updates: the first call goes
 * through at once, later ones within `waitMs` are coalesced into one call with the latest
 * arguments at the end of the window. `flush()` makes that call now (an end state the UI
 * should show without waiting for the window), and `cancel()` drops it.
 */
export function throttle<Args extends unknown[]>(fn: (...args: Args) => void, waitMs: number) {
  let last = Number.NEGATIVE_INFINITY;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let pending: Args | undefined;
  const invoke = () => {
    last = performance.now();
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
    if (pending) {
      const args = pending;
      pending = undefined;
      fn(...args);
    }
  };
  const throttled = (...args: Args) => {
    pending = args;
    const elapsed = performance.now() - last;
    if (elapsed >= waitMs && timer === undefined) {
      invoke();
    } else {
      timer ??= setTimeout(invoke, Math.max(0, waitMs - elapsed));
    }
  };
  throttled.cancel = () => {
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
    pending = undefined;
  };
  /** The coalesced call, now, if one is waiting. */
  throttled.flush = () => {
    if (pending) invoke();
  };
  return throttled;
}

/** How often loading progress reaches the UI at most (ms): four times a second. */
export const PROGRESS_REPORT_MS = 250;

/**
 * A loading-progress report (`pending`, `processing`, as Cesium's `loadProgress` gives them)
 * throttled to `waitMs`, except the end of loading -- both zero -- which goes through at once,
 * so "loaded" never shows a quarter of a second late.
 */
export function throttleProgress(
  report: (pending: number, processing: number) => void,
  waitMs = PROGRESS_REPORT_MS,
) {
  const throttled = throttle(report, waitMs);
  const progress = (pending: number, processing: number): void => {
    throttled(pending, processing);
    if (pending === 0 && processing === 0) throttled.flush();
  };
  progress.cancel = throttled.cancel;
  return progress;
}

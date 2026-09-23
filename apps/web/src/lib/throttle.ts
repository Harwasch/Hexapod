/** Trailing-edge throttle used to rate-limit Cesium → React state updates. */
export function throttle<Args extends unknown[]>(fn: (...args: Args) => void, waitMs: number) {
  let last = Number.NEGATIVE_INFINITY;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let pending: Args | undefined;
  const invoke = () => {
    last = performance.now();
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
  return throttled;
}

/** Framework-free so the phone and scan pages share the same transport. */
const listeners = new Set<() => void>();
let pending = 0;
let unavailable = false;

export const apiWaking = {
  getSnapshot: (): boolean => pending > 0 && !unavailable,
  subscribe: (listener: () => void): (() => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};

/** Keep the offline catalog available while a single recovery request is pending. */
export const apiUnavailable = {
  getSnapshot: (): boolean => unavailable,
  subscribe: apiWaking.subscribe,
};

/** A spent retry budget marks the API down; a successful response rearms cold-start retries. */
export function setApiUnavailable(value: boolean): void {
  if (unavailable === value) return;
  unavailable = value;
  listeners.forEach((listener) => listener());
}

/** Each retrying request owns one release, including cancellation and failure paths. */
export function beginApiWake(): () => void {
  pending += 1;
  if (pending === 1) listeners.forEach((listener) => listener());
  let released = false;
  return () => {
    if (released) return;
    released = true;
    pending -= 1;
    if (pending === 0) listeners.forEach((listener) => listener());
  };
}

import { apiUnavailable, beginApiWake, setApiUnavailable } from "./waking";

// Tune against the production measurements in docs/DEPLOYMENT.md.
export const API_RETRY_BUDGET_MS = 45_000;
export const API_RETRY_DELAYS_MS = [1000, 2000, 4000, 8000] as const;
export const API_ATTEMPT_TIMEOUT_MS = 10_000;

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);
const COLD_STATUSES = new Set([502, 503, 504]);

function observeResponse(response: Response): Response {
  if (response.ok) setApiUnavailable(false);
  return response;
}

function isTransient(error: unknown): boolean {
  return (
    error instanceof TypeError ||
    ((error instanceof Error || error instanceof DOMException) && error.name === "TimeoutError")
  );
}

/** An abortable delay, with no timer or listener left behind after cancellation. */
function wait(ms: number, signal?: AbortSignal | null): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = (): void => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    };
    const abort = (): void => {
      cleanup();
      // AbortSignal permits arbitrary reasons; cancellation must preserve the caller’s value.
      // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
      reject(signal?.reason);
    };
    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
  });
}

async function attempt(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  caller: AbortSignal | null | undefined,
  ms: number,
): Promise<Response> {
  const timeout = new AbortController();
  const signal = caller ? AbortSignal.any([caller, timeout.signal]) : timeout.signal;
  const timer = setTimeout(
    () => timeout.abort(new DOMException("The API did not answer in time", "TimeoutError")),
    ms,
  );
  let onAbort = (): void => undefined;
  const aborted = new Promise<never>((_, reject) => {
    // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors -- Preserve AbortSignal.reason.
    onAbort = () => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
  });
  try {
    // Race as well as abort: even a transport that ignores its signal cannot exceed the cap.
    return observeResponse(await Promise.race([fetch(input, { ...init, signal }), aborted]));
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
  }
}

/**
 * The one API transport for all four pages. Safe reads wait out a sleeping Fly machine;
 * writes keep fetch's single-attempt semantics. Final HTTP failures remain Responses so
 * each client's existing problem/auth handling sees the status and body unchanged.
 * Once a budget is spent, recovery reads make one bounded attempt without a waking note
 * until a successful response proves the API is back.
 */
export async function fetchApi(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const request = input instanceof Request ? input : undefined;
  const method = (init?.method ?? request?.method ?? "GET").toUpperCase();
  const signal = init?.signal ?? request?.signal;
  signal?.throwIfAborted();
  if (!SAFE_METHODS.has(method)) return fetch(input, init).then(observeResponse);
  if (apiUnavailable.getSnapshot()) {
    return attempt(input, init, signal, API_ATTEMPT_TIMEOUT_MS);
  }

  const started = performance.now();
  const remaining = (): number => API_RETRY_BUDGET_MS - (performance.now() - started);
  let release: (() => void) | undefined;
  let retry = 0;
  try {
    for (;;) {
      signal?.throwIfAborted();
      let response: Response | undefined;
      let failure: unknown;
      try {
        response = await attempt(
          input,
          init,
          signal,
          Math.min(API_ATTEMPT_TIMEOUT_MS, remaining()),
        );
        if (!COLD_STATUSES.has(response.status)) return response;
      } catch (error) {
        signal?.throwIfAborted();
        if (!isTransient(error)) throw error;
        failure = error;
      }
      const left = remaining();
      if (left > 0 && !apiUnavailable.getSnapshot()) {
        release ??= beginApiWake();
        const delay = API_RETRY_DELAYS_MS[retry++] ?? API_RETRY_DELAYS_MS[3];
        // Keep the last HTTP body intact if there is no time for another attempt.
        await wait(Math.min(delay, left), signal);
        if (remaining() > 0 && !apiUnavailable.getSnapshot()) {
          void response?.body?.cancel().catch(() => undefined);
          continue;
        }
      }
      if (remaining() <= 0) setApiUnavailable(true);
      if (response) return response;
      throw failure;
    }
  } finally {
    release?.();
  }
}

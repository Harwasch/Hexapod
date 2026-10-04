/**
 * A deadline on a promise that has none of its own.
 *
 * `fetch` and CesiumJS's `Resource` wait as long as the socket does: a cold API behind a
 * proxy, or a tile host that accepted the connection and then went quiet, leaves the caller
 * waiting with no error ever arriving -- and `withRetry` (retry.ts) only retries what fails,
 * not what stalls. A site whose detail request hung used to sit at "loading" for the rest of
 * the session. Here the caller is told at the deadline instead, with an error that says so.
 *
 * The work itself cannot be cancelled from outside (a tileset half-way through `fromUrl` has
 * no abort), so it is left to finish, and whatever it produces late is handed to `onLate` --
 * typically to destroy a tileset nobody will ever add to the scene. Where the work does take
 * an `AbortSignal`, pass `controller` too and it is aborted at the deadline; this is the
 * backstop for where it does not.
 *
 * A failure that beats the deadline is passed through untouched: callers tell a refused ion
 * key from a missing asset by the error's own shape (cesium/ion.ts), which a wrapper would
 * hide.
 */

/** The deadline passed. A stall is the textbook transient failure, so it is worth a retry. */
export class TimeoutError extends Error {
  constructor(
    readonly what: string,
    readonly ms: number,
  ) {
    super(`${what} did not answer within ${Math.round(ms / 1000)} s`);
    this.name = "TimeoutError";
  }
}

export interface TimeoutOptions<T> {
  /** Named in the error: "The site's details", "The model". */
  what?: string;
  /** Aborted at the deadline, for work that can stop early. */
  controller?: AbortController;
  /** Receives a result that arrives after the deadline, to dispose of it. */
  onLate?: (value: T) => void;
}

export function withTimeout<T>(
  work: Promise<T>,
  ms: number,
  options: TimeoutOptions<T> = {},
): Promise<T> {
  let timedOut = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      timedOut = true;
      const error = new TimeoutError(options.what ?? "The request", ms);
      // Rejected before the abort: abort-aware work rejects synchronously inside `abort()`,
      // and whichever settles first is what the race reports.
      reject(error);
      options.controller?.abort(error);
    }, ms);
  });
  work.then(
    (value) => {
      clearTimeout(timer);
      if (timedOut) options.onLate?.(value);
    },
    () => clearTimeout(timer),
  );
  return Promise.race([work, deadline]);
}

export function isTimeout(error: unknown): error is TimeoutError {
  return error instanceof TimeoutError;
}

/** A download stopped making progress: no bytes for `ms`. A stall, so worth a retry too. */
export class StallError extends TimeoutError {
  constructor(what: string, ms: number) {
    super(what, ms);
    this.message = `${what} stopped arriving for ${String(Math.round(ms / 1000))} s`;
    this.name = "StallError";
  }
}

/**
 * Fetches a JSON document that may be large and the connection slow, failing only when it
 * stalls: no answer, or no new bytes, for `stallMs`. A deadline on the whole download is the
 * wrong question for a big file on a phone -- a 6 MB tileset.json over a slow link takes longer
 * than any total that also catches a stall quickly, so every retry started over and failed the
 * same way. Here the deadline moves with every chunk that arrives.
 *
 * A refusal is an error whose message carries the status ("Request failed with status 404"),
 * the shape callers already read (cesium/ion.ts).
 */
export async function fetchJsonUnlessStalled(
  url: string,
  options: { stallMs: number; what?: string; headers?: Record<string, string> },
): Promise<unknown> {
  const what = options.what ?? "The request";
  const controller = new AbortController();
  let stalled: StallError | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const arm = (): void => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      stalled = new StallError(what, options.stallMs);
      controller.abort(stalled);
    }, options.stallMs);
  };
  arm();
  try {
    const response = await fetch(url, {
      headers: { Accept: "application/json,*/*;q=0.01", ...options.headers },
      signal: controller.signal,
    });
    if (!response.ok) {
      throw Object.assign(new Error(`Request failed with status ${String(response.status)}`), {
        statusCode: response.status,
      });
    }
    if (!response.body) return (await response.json()) as unknown;
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    for (;;) {
      arm();
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    const bytes = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    return JSON.parse(new TextDecoder().decode(bytes)) as unknown;
  } catch (error) {
    throw stalled ?? error;
  } finally {
    clearTimeout(timer);
  }
}

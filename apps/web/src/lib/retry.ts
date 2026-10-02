/**
 * Retrying what fails for a moment. The API scales to zero between visits (fly.toml
 * `min_machines_running = 0`), so the first request after a quiet spell meets a cold start;
 * a CDN or Cesium ion can drop one request. A world layer or a site that failed once used to
 * stay off until the page was reloaded -- the "blue ball" start. Here a failure is tried
 * again a few times, waiting longer each time, unless it is one that waiting cannot fix.
 */

export interface RetryOptions {
  /** Attempts in all, the first included. */
  attempts?: number;
  /** Wait before the second attempt (ms); doubles each time after. */
  firstDelayMs?: number;
  /** Errors not worth another attempt (a refused key, a missing asset). */
  permanent?: (error: unknown) => boolean;
  signal?: AbortSignal | undefined;
  /** Told of each failure that will be tried again. */
  onRetry?: (error: unknown, attempt: number) => void;
}

export const RETRY_ATTEMPTS = 4;
export const RETRY_FIRST_DELAY_MS = 1000;

export async function withRetry<T>(run: () => Promise<T>, options: RetryOptions = {}): Promise<T> {
  const attempts = options.attempts ?? RETRY_ATTEMPTS;
  let delay = options.firstDelayMs ?? RETRY_FIRST_DELAY_MS;
  for (let attempt = 1; ; attempt++) {
    try {
      return await run();
    } catch (error) {
      if (attempt >= attempts || options.permanent?.(error) || options.signal?.aborted) throw error;
      options.onRetry?.(error, attempt);
      await new Promise<void>((done) => setTimeout(done, delay));
      if (options.signal?.aborted) throw error;
      delay *= 2;
    }
  }
}

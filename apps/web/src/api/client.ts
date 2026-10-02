import createClient, { type Middleware } from "openapi-fetch";

import type { paths, Problem } from "@twin/contracts";

import { env } from "@/app/env";
import { ApiError } from "./error";
import { recordSpan } from "@/lib/timing";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

/** Error thrown for non-2xx responses, carrying the API's problem payload when present. */
export { ApiError } from "./error";

/**
 * When each in-flight request started, keyed by openapi-fetch's per-request `id`.
 *
 * Deliberately NOT a header. This used to be `x-request-started`, set on the outgoing
 * request so `onResponse` could read it back -- and a header on a request goes over the
 * wire. That made every GET a non-simple CORS request, so the browser sent a preflight
 * asking to send `x-request-started`, and the API answered `400 Disallowed CORS headers`
 * because nothing on the server has any use for a client's clock reading. Every read
 * failed, the app fell back to its offline catalogue, and the globe lost the layers the
 * API serves.
 *
 * It was invisible in development, where Vite proxies `/api` to the API and the page and
 * the API are the same origin: no CORS, no preflight. The first cross-origin deployment
 * was the first place it could fail, and it failed there on every request.
 *
 * The fix is here rather than in the API's `allow_headers`, because the header was the
 * bug: allowing it would ship a client-internal timestamp to the server and keep a
 * preflight round trip in front of every read.
 */
const requestStarted = new Map<string, number>();

const timing: Middleware = {
  onRequest({ id }) {
    requestStarted.set(id, performance.now());
  },
  onResponse({ id, request, response }) {
    const started = requestStarted.get(id);
    requestStarted.delete(id);
    if (started !== undefined) {
      recordSpan("api", performance.now() - started, {
        method: request.method,
        path: new URL(request.url).pathname,
        status: response.status,
      });
    }
    return response;
  },
  // A request that never gets a response -- offline, refused, a CORS failure -- still
  // has an entry, and without this the map would grow by one for every such request.
  onError({ id }) {
    requestStarted.delete(id);
  },
};

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

/**
 * Attaches the write token to mutating API calls, and notices when one is demanded.
 *
 * Two things this deliberately does not do. It does not read a `VITE_` variable: those
 * are inlined into the bundle, so a shared write token in one is published to every
 * visitor. And it does not run on the part PUTs of an upload — those go straight to
 * object storage, where the presigned URL *is* the credential and an extra
 * `Authorization` header can invalidate the SigV4 signature. That is structural rather
 * than conditional: the uploader uses `XMLHttpRequest` directly (it also needs byte
 * progress, which `openapi-fetch` cannot give), so storage requests never pass through
 * this client at all.
 *
 * Exported so its two rules can be tested directly, rather than through a mocked fetch.
 */
export const auth: Middleware = {
  onRequest({ request }) {
    const token = useSettings.getState().writeToken.trim();
    if (token && !SAFE_METHODS.has(request.method)) {
      request.headers.set("Authorization", `Bearer ${token}`);
    }
    return request;
  },
  onResponse({ request, response }) {
    // The affordance appears on a 401 and never before it: with no token configured
    // server-side the API leaves writes open, and local development must need no prompt.
    if (response.status === 401 && !SAFE_METHODS.has(request.method)) {
      useUi.getState().setWriteTokenPrompt(true);
    }
    return response;
  },
};

export const api = createClient<paths>({ baseUrl: env.apiBaseUrl });
api.use(auth);
api.use(timing);

/** Unwraps openapi-fetch results into data or a thrown ApiError. */
export async function unwrap<T>(
  promise: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await promise;
  if (error !== undefined || !response.ok) {
    throw new ApiError(
      response.status,
      asProblem(error),
      `${response.status} ${response.statusText}`,
    );
  }
  return data as T;
}

function asProblem(value: unknown): Problem | undefined {
  if (typeof value === "object" && value !== null && "title" in value && "status" in value) {
    return value as Problem;
  }
  return undefined;
}

export function isOffline(error: unknown): boolean {
  return error instanceof TypeError || (error instanceof ApiError && error.status >= 500);
}

/** True when a write was refused for want of the token `ApiError` already carries the status of. */
export function isUnauthorized(error: unknown): boolean {
  return error instanceof ApiError && error.status === 401;
}

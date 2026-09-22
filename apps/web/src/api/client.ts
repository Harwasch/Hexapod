import createClient, { type Middleware } from "openapi-fetch";

import type { paths, Problem } from "@twin/contracts";

import { env } from "@/app/env";
import { recordSpan } from "@/lib/timing";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

/** Error thrown for non-2xx responses, carrying the API's problem payload when present. */
export class ApiError extends Error {
  readonly status: number;
  readonly problem: Problem | undefined;

  constructor(status: number, problem: Problem | undefined, fallback: string) {
    super(problem?.detail ?? problem?.title ?? fallback);
    this.name = "ApiError";
    this.status = status;
    this.problem = problem;
  }

  /** Field-level validation messages, when the API returned any. */
  get fieldErrors(): string[] {
    return (this.problem?.errors ?? []).map((e) => {
      const loc = Array.isArray(e.loc) ? e.loc.filter((p) => p !== "body").join(".") : "";
      return loc ? `${loc}: ${String(e.msg)}` : String(e.msg);
    });
  }
}

const timing: Middleware = {
  onRequest({ request }) {
    request.headers.set("x-request-started", String(performance.now()));
    return request;
  },
  onResponse({ request, response }) {
    const started = Number(request.headers.get("x-request-started"));
    if (Number.isFinite(started)) {
      recordSpan("api", performance.now() - started, {
        method: request.method,
        path: new URL(request.url).pathname,
        status: response.status,
      });
    }
    return response;
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

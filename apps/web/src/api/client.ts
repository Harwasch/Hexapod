import createClient, { type Middleware } from "openapi-fetch";

import type { paths, Problem } from "@twin/contracts";

import { env } from "@/app/env";
import { recordSpan } from "@/lib/timing";

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

export const api = createClient<paths>({ baseUrl: env.apiBaseUrl });
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

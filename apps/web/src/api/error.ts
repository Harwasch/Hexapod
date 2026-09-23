/**
 * `ApiError`, alone in its own module so that code which only needs to *raise* one does
 * not pull in the console.
 *
 * `client.ts` wires the write token, the settings store and the UI store into its
 * middleware. That is right for the console and wrong for the standalone upload page a
 * phone opens: it has no stores, no token, and no reason to carry them. Splitting the
 * error type out is what lets `putPart.ts` be shared by both.
 */
import type { Problem } from "@twin/contracts";

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

/** Raised when the caller aborts; callers distinguish it from a real failure. */
export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

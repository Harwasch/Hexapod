import { getApiToken } from "./storage";
import type {
  ActionBinding,
  ControlEvent,
  ModelCapabilities,
  ProviderId,
  ProviderInfo,
  ResumeKind,
} from "./types";
export interface WorkerHandle {
  id: string;
  provider: ProviderId;
  modelId?: string;
  status: string;
  createdAt: number | string;
  estimatedHourlyCost: number | null;
  error?: string;
  managed?: boolean;
  hardDeadline?: number;
  idleDeadline?: number;
  cleanupError?: string;
}
export interface SessionHandle {
  id: string;
  workerId: string;
  modelId: string;
  status: string;
  createdAt: number | string;
  capabilities?: Partial<ModelCapabilities>;
  leaseExpiresAt?: number;
  error?: string;
}
export interface GatewayReadiness {
  provider: string;
  modelId?: string;
  status: string;
  message?: string;
  error?: string;
  models: { id: string; status?: string; reason?: string; name?: string }[];
}
export interface WorldsReadiness {
  provisioningEnabled: boolean;
  configurationIssue?: string | null;
  providers: ProviderInfo[];
  gateways: GatewayReadiness[];
  modelProfiles?: {
    modelId: string;
    providers: string[];
    gatewayProviders?: string[];
    provisioningProviders?: string[];
  }[];
  lifecycle: {
    enabled: boolean;
    running: boolean;
    sessionLeaseSeconds: number;
    heartbeatIntervalSeconds: number;
    workerIdleSeconds: number;
    workerMaxLifetimeSeconds: number;
    workerStartupSeconds: number;
    maxManagedWorkers: number;
    maxWorkerHourlyCost: number | null;
  };
}
export interface SessionInput {
  workerId: string;
  modelId: string;
  prompt: string;
  seed?: number;
  quality?: string;
  resolution?: string;
  inputs?: { images?: string[]; video?: string; [key: string]: unknown };
}
export interface IntelligenceInput {
  prompt: string;
  modelId: string;
  capabilities?: ModelCapabilities;
  experience?: string;
  nativeActions?: string[];
}
export class WorldsApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code?: string,
  ) {
    super(message);
    this.name = "WorldsApiError";
  }
}
/** Custom manager URLs must be HTTPS except loopback for local GPU development. */
export function validateApiBaseUrl(base: string): string {
  if (!base.trim()) return "";
  let url: URL;
  try {
    url = new URL(base);
  } catch {
    throw new WorldsApiError("Enter a complete session manager URL, including https://.", 0);
  }
  if (url.username || url.password || url.search || url.hash)
    throw new WorldsApiError(
      "Session manager URLs must not include credentials, query strings, or fragments.",
      0,
    );
  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopback))
    throw new WorldsApiError(
      "Remote session managers require HTTPS. HTTP is supported only on localhost.",
      0,
    );
  return url
    .toString()
    .replace(/\/$/, "")
    .replace(/\/api\/v1\/worlds$/, "");
}
export function createWorldApi(baseUrl = "") {
  const base = `${validateApiBaseUrl(baseUrl)}/api/v1/worlds`;
  async function request<T>(path: string, options: RequestInit = {}, raw = false): Promise<T> {
    const token = getApiToken();
    const headers = new Headers(options.headers);
    if (options.body) headers.set("Content-Type", "application/json");
    if (token) headers.set("Authorization", `Bearer ${token}`);
    let response: Response;
    try {
      response = await fetch(`${base}${path}`, {
        ...options,
        headers,
        credentials: "omit",
        redirect: "error",
        signal: options.signal ?? AbortSignal.timeout(120_000),
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      throw new WorldsApiError(
        "Could not reach the Worlds session manager. Check the server URL and connection.",
        0,
      );
    }
    if (!response.ok) {
      let message =
        response.status === 401
          ? "The session manager requires an access token. Add it in Settings."
          : response.status === 503
            ? "This service is not configured. Connect a model worker or AI service in your session manager."
            : `Worlds request failed (${response.status}).`;
      let code: string | undefined;
      try {
        const body = (await response.json()) as {
          detail?: unknown;
          message?: unknown;
          code?: string;
        };
        if (typeof body.detail === "string") message = body.detail;
        else if (typeof body.message === "string") message = body.message;
        code = body.code;
      } catch {
        /* Reverse proxies may return an HTML error. */
      }
      throw new WorldsApiError(message, response.status, code);
    }
    if (response.status === 204) return null as T;
    if (raw) return (await response.blob()) as T;
    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.includes("application/json"))
      throw new WorldsApiError(
        "The server returned a web page instead of the Worlds API. Check the session manager URL.",
        502,
      );
    return (await response.json()) as T;
  }
  const post = <T>(path: string, body: unknown = {}) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) });
  return {
    request,
    providers: () => request<{ providers: ProviderInfo[] }>("/providers"),
    readiness: (signal?: AbortSignal) =>
      request<WorldsReadiness>("/readiness", { signal: signal ?? AbortSignal.timeout(15_000) }),
    catalog: () =>
      request<{
        gateways: { provider: string; status: string; models: unknown[] }[];
        providers: ProviderInfo[];
        primaryProvider: string;
      }>("/catalog"),
    workers: () => request<{ workers: WorkerHandle[] }>("/workers"),
    createWorker: (input: { provider: ProviderId; gpuTypeId?: string; modelId?: string }) =>
      post<WorkerHandle>("/workers", input),
    worker: (id: string) => request<WorkerHandle>(`/workers/${encodeURIComponent(id)}`),
    stopWorker: (id: string) => post<WorkerHandle>(`/workers/${encodeURIComponent(id)}/stop`),
    destroyWorker: (id: string) =>
      request<unknown>(`/workers/${encodeURIComponent(id)}`, { method: "DELETE" }),
    createSession: (input: SessionInput) => post<SessionHandle>("/sessions", input),
    session: (id: string) => request<SessionHandle>(`/sessions/${encodeURIComponent(id)}`),
    action: (id: string, event: Omit<ControlEvent, "id" | "timestampMs">) =>
      post<unknown>(`/sessions/${encodeURIComponent(id)}/actions`, event),
    frame: (id: string, signal?: AbortSignal) =>
      request<Blob | null>(
        `/sessions/${encodeURIComponent(id)}/frame`,
        { signal, cache: "no-store" },
        true,
      ),
    snapshot: async (
      id: string,
    ): Promise<{ resumeKind: ResumeKind; state?: Record<string, unknown> }> => {
      const value = await post<{ resumeKind: string; state?: Record<string, unknown> }>(
        `/sessions/${encodeURIComponent(id)}/snapshot`,
      );
      const resumeKind: ResumeKind =
        ["exact", "exact-resume"].includes(value.resumeKind) && value.state
          ? "exact"
          : ["approximate", "approximate-resume"].includes(value.resumeKind)
            ? "approximate"
            : "visual";
      return { resumeKind, state: value.state };
    },
    offer: (id: string, offer: RTCSessionDescriptionInit) =>
      post<RTCSessionDescriptionInit>(`/sessions/${encodeURIComponent(id)}/offer`, offer),
    endSession: (id: string) =>
      request<unknown>(`/sessions/${encodeURIComponent(id)}`, { method: "DELETE" }),
    enhancePrompt: (input: {
      prompt: string;
      modelId: string;
      capabilities?: ModelCapabilities;
      experience?: string;
    }) =>
      post<{ original: string; enhanced: string; source: "llm" | "local-guide"; notes?: string[] }>(
        "/prompts/optimize",
        input,
      ),
    generateControls: (input: IntelligenceInput) =>
      post<{ bindings: ActionBinding[]; source: "llm" | "local-guide"; notes: string[] }>(
        "/controls/generate",
        input,
      ),
    createGame: (input: {
      prompt: string;
      modelId: string;
      capabilities: ModelCapabilities;
      experience?: string;
      nativeActions?: string[];
    }) =>
      post<{
        name: string;
        premise: string;
        objective: string;
        prompt: string;
        events: { atSeconds: number; prompt: string }[];
        source: "llm" | "local-guide";
        notes: string[];
      }>("/games/create", input),
  };
}
export type WorldsApi = ReturnType<typeof createWorldApi>;

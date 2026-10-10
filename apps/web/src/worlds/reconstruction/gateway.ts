import { validateApiBaseUrl } from "../core/api";
import { getApiToken } from "../core/storage";

export interface ReconstructionArtifact {
  format: "ply" | "spz" | "glb" | "gltf" | "point-cloud";
  url: string;
}
export interface ReconstructionJob {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress?: number | null;
  stage?: string | null;
  artifacts: ReconstructionArtifact[];
  error?: string | null;
  diagnosticsUrl?: string | null;
  camerasUrl?: string | null;
}
export interface ReconstructionCapabilities {
  configured: boolean;
  maxUploadBytes: number;
  message: string;
}

async function json<T>(
  base: string,
  path: string,
  token: string | undefined,
  init?: RequestInit,
): Promise<T> {
  const headers = new Headers(init?.headers);
  if (token?.trim()) headers.set("Authorization", `Bearer ${token.trim()}`);
  const response = await fetch(`${validateApiBaseUrl(base)}/api/v1/worlds/reconstructions${path}`, {
    ...init,
    headers,
    credentials: "omit",
    redirect: "error",
    signal: AbortSignal.timeout(120_000),
  });
  if (!response.ok) {
    let message = `Reconstruction service returned ${response.status}.`;
    try {
      const value: unknown = await response.json();
      if (
        value &&
        typeof value === "object" &&
        "detail" in value &&
        typeof value.detail === "string"
      )
        message = value.detail;
    } catch {
      /* An offline HTML fallback is not a service response. */
    }
    throw new Error(message);
  }
  if (!response.headers.get("content-type")?.includes("application/json"))
    throw new Error("The server returned a web page instead of the reconstruction API.");
  return response.json() as Promise<T>;
}

export const reconstructionGateway = {
  capabilities: (base: string) =>
    json<ReconstructionCapabilities>(base, "/capabilities", getApiToken()),
  submit: (base: string, body: FormData) =>
    json<ReconstructionJob>(base, "", getApiToken(), { method: "POST", body }),
  status: (base: string, id: string) =>
    json<ReconstructionJob>(base, `/${encodeURIComponent(id)}`, getApiToken()),
  delete: (base: string, id: string) =>
    json<{ deleted: unknown }>(base, `/${encodeURIComponent(id)}`, getApiToken(), {
      method: "DELETE",
    }),
};

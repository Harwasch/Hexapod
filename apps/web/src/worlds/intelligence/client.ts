import { createWorldApi } from "../core/api";
import type { ModelCapabilities } from "../core/types";
export interface ProposedAction {
  type: "native" | "prompt" | "semantic";
  action?: string | null;
  prompt?: string | null;
}
export interface IntelligenceStatus {
  configured: boolean;
  visionConfigured: boolean;
  imageConfigured: boolean;
  message: string;
}
export interface SceneContext {
  serverUrl: string;
  modelId: string;
  capabilities: ModelCapabilities;
  nativeActions: string[];
  prompt: string;
  frame: Blob;
  objective?: string;
  elapsedSeconds?: number;
  recentEvents?: string[];
  signal?: AbortSignal;
}
export interface CommandProposal {
  observation: string;
  explanation: string;
  action: ProposedAction | null;
  source: "vision-llm";
  experimental: true;
}
export interface DirectorProposal {
  observation: string;
  objective: {
    progress: "unknown" | "not-started" | "in-progress" | "appears-complete";
    confidence: number;
    evidence: string;
  };
  event: ProposedAction | null;
  reason: string;
  source: "vision-llm";
}
export interface AppearancePackage {
  appearance: string;
  clothing: string;
  distinguishingFeatures: string[];
  consistencyNotes: string[];
  conditioningPrompt: string;
  source: "vision-llm";
  identityMethod: "reference-images";
}
export interface ConsistencyAssessment {
  score: number;
  confidence: number;
  evidence: string[];
  differences: string[];
  source: "vision-llm";
  metric: "qualitative-appearance-consistency";
  note: string;
}
export interface Highlight {
  startSeconds: number;
  endSeconds: number;
  title: string;
  evidence: string;
}
export interface HighlightsResult {
  highlights: Highlight[];
  source: "vision-llm";
  note: string;
}
export function getIntelligenceStatus(serverUrl: string, signal?: AbortSignal) {
  return createWorldApi(serverUrl).request<IntelligenceStatus>("/intelligence/status", { signal });
}
export function postIntelligence<T>(
  serverUrl: string,
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<T> {
  return createWorldApi(serverUrl).request<T>(path, {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  });
}
/** Resizing through a canvas strips local filenames and image metadata before upload. */
export async function imageData(blob: Blob): Promise<string> {
  if (!["image/jpeg", "image/png", "image/webp"].includes(blob.type))
    throw new Error("Use a JPEG, PNG, or WebP image.");
  if (blob.size > 16 * 1024 * 1024)
    throw new Error("Reference exceeds 16 MB. Resize it before analysis.");
  const bitmap = await createImageBitmap(blob);
  try {
    if (bitmap.width * bitmap.height > 16_000_000)
      throw new Error("Reference exceeds 16 megapixels. Resize it before analysis.");
    const scale = Math.min(1, 768 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Image conversion is unavailable in this browser.");
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  } finally {
    bitmap.close();
  }
}
async function scenePayload(input: SceneContext) {
  const frame = await imageData(input.frame);
  input.signal?.throwIfAborted();
  return {
    prompt: input.prompt.slice(0, 8000),
    modelId: input.modelId,
    capabilities: input.capabilities,
    nativeActions: input.nativeActions.slice(0, 64),
    frame,
    objective: (input.objective ?? "").slice(0, 1000),
    elapsedSeconds: Math.min(86400, Math.max(0, input.elapsedSeconds ?? 0)),
    recentEvents: (input.recentEvents ?? []).slice(-20).map((e) => e.slice(0, 1000)),
  };
}
export async function interpretCommand(
  input: SceneContext & { command: string },
): Promise<CommandProposal> {
  return postIntelligence(
    input.serverUrl,
    "/intelligence/command",
    { ...(await scenePayload(input)), command: input.command.slice(0, 1000) },
    input.signal,
  );
}
export async function observeWorld(
  input: SceneContext & { mode: "relaxed" | "cinematic" | "challenging" | "chaotic" },
): Promise<DirectorProposal> {
  return postIntelligence(
    input.serverUrl,
    "/intelligence/director",
    { ...(await scenePayload(input)), mode: input.mode },
    input.signal,
  );
}
export function actionSupported(
  action: ProposedAction,
  caps: ModelCapabilities,
  nativeActions: string[],
) {
  return action.type === "native"
    ? !!action.action && nativeActions.includes(action.action)
    : action.type === "semantic"
      ? !!action.prompt && caps.control.semanticActions
      : !!action.prompt && (caps.control.promptDuringRollout || caps.control.promptSwitching);
}
export function dataImageBlob(value: string): Blob {
  const match = /^data:image\/(jpeg|png|webp);base64,([A-Za-z0-9+/=]+)$/.exec(value);
  if (!match?.[1] || !match[2] || match[2].length > 4_000_000)
    throw new Error("The image service returned invalid image data.");
  const raw = atob(match[2]);
  return new Blob([Uint8Array.from(raw, (c) => c.charCodeAt(0))], { type: `image/${match[1]}` });
}

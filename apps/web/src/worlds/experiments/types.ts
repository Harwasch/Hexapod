import type {
  BenchmarkResult,
  Character,
  ControlEvent,
  ProviderId,
  WorldProject,
} from "../core/types";
export interface ExperimentTarget {
  modelId: string;
  providerId: ProviderId;
  label?: string;
  discoveryAction?: string;
  characterView?: "front" | "side" | "environment";
}
export interface ExperimentPlan {
  project: WorldProject;
  targets: ExperimentTarget[];
  seed: number;
  events: ControlEvent[];
  durationSeconds: number;
  maxWallTimeSeconds: number;
  maxEstimatedCostUSD: number;
  mode: "compare" | "action-discovery" | "return-to-view" | "character-consistency";
  character?: Character;
  visionConsent?: boolean;
  evaluationFrames?: number;
}
export interface FrameObservation {
  timestampMs: number;
  assetId?: string;
  pixelChange?: number;
  motionX?: number;
  motionY?: number;
  motionConfidence?: number;
  sharpness: number;
}
export interface ActionObservation {
  eventId: string;
  scheduledMs: number;
  dispatchedMs: number;
  latencyMs: number;
  accepted: boolean;
  error?: string;
}
export interface ExperimentResult extends BenchmarkResult {
  experimentVersion: 1;
  experimentId: string;
  status: "completed" | "cancelled" | "failed";
  mode: ExperimentPlan["mode"];
  seed: number;
  inputFingerprint: string;
  targetLabel: string;
  characterEvaluation?: CharacterEvaluation;
  effectivePrompt?: string;
  effectiveQuality?: string;
  requestedEvents?: ControlEvent[];
  returnAlignment?: { x: number; y: number; confidence: number };
  discoveryAction?: string;
  assetIds: string[];
  videoAssetId?: string;
  observations: FrameObservation[];
  actionObservations: ActionObservation[];
  startupMs?: number;
  deliveredFPS: number;
  firstFrameMs?: number;
  generatedFPS?: number;
  sourceFrameCount?: number;
  returnFrameSimilarity?: number;
  resolution?: string;
  priceSource?: string;
  errors: string[];
  cleanupErrors: string[];
  billedCostUSD: null;
}
export interface CharacterSampleAssessment {
  assetId: string;
  timestampMs: number;
  score: number;
  confidence: number;
  evidence: string[];
  differences: string[];
}
export interface CharacterEvaluation {
  characterId: string;
  characterName: string;
  view: "front" | "side" | "environment";
  conditioningMethod: string;
  samples: CharacterSampleAssessment[];
  errors: string[];
  meanScore?: number;
  meanConfidence?: number;
  metric: "qualitative-appearance-consistency";
  source: "vision-llm" | "not-evaluated";
}
export interface ExperimentProgress {
  target: string;
  index: number;
  total: number;
  phase: string;
  elapsedSeconds: number;
  frames: number;
  estimatedCostUSD: number;
}
export interface GrayFrame {
  width: number;
  height: number;
  pixels: Float32Array;
  sourceWidth?: number;
  sourceHeight?: number;
}
function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
function strings(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}
/** Extended import data is untrusted even when the base benchmark schema is valid. */
export function isExperimentResult(value: BenchmarkResult): value is ExperimentResult {
  if (
    !record(value) ||
    value.experimentVersion !== 1 ||
    !Array.isArray(value.observations) ||
    !Array.isArray(value.actionObservations)
  )
    return false;
  if (
    !["completed", "cancelled", "failed"].includes(String(value.status)) ||
    typeof value.targetLabel !== "string" ||
    typeof value.inputFingerprint !== "string" ||
    !finite(value.seed) ||
    !finite(value.deliveredFPS)
  )
    return false;
  for (const name of ["assetIds", "errors", "cleanupErrors"])
    if (!strings(value[name])) return false;
  for (const name of [
    "generatedFPS",
    "firstFrameMs",
    "startupMs",
    "sourceFrameCount",
    "returnFrameSimilarity",
  ])
    if (value[name] !== undefined && !finite(value[name])) return false;
  for (const name of ["effectivePrompt", "effectiveQuality", "videoAssetId", "priceSource"])
    if (value[name] !== undefined && typeof value[name] !== "string") return false;
  if (
    !value.observations.every(
      (item) =>
        record(item) &&
        finite(item.timestampMs) &&
        finite(item.sharpness) &&
        (item.assetId === undefined || typeof item.assetId === "string") &&
        ["pixelChange", "motionX", "motionY", "motionConfidence"].every(
          (key) => item[key] === undefined || finite(item[key]),
        ),
    )
  )
    return false;
  if (
    !value.actionObservations.every(
      (item) =>
        record(item) &&
        typeof item.eventId === "string" &&
        finite(item.dispatchedMs) &&
        finite(item.latencyMs) &&
        typeof item.accepted === "boolean" &&
        (item.error === undefined || typeof item.error === "string"),
    )
  )
    return false;
  if (value.characterEvaluation !== undefined) {
    const evaluation = value.characterEvaluation;
    if (
      !record(evaluation) ||
      typeof evaluation.characterId !== "string" ||
      !["front", "side", "environment"].includes(String(evaluation.view)) ||
      !["vision-llm", "not-evaluated"].includes(String(evaluation.source)) ||
      evaluation.metric !== "qualitative-appearance-consistency" ||
      typeof evaluation.characterName !== "string" ||
      typeof evaluation.conditioningMethod !== "string" ||
      !strings(evaluation.errors) ||
      !Array.isArray(evaluation.samples)
    )
      return false;
    for (const key of ["meanScore", "meanConfidence"])
      if (evaluation[key] !== undefined && !finite(evaluation[key])) return false;
    if (
      !evaluation.samples.every(
        (sample) =>
          record(sample) &&
          finite(sample.score) &&
          finite(sample.confidence) &&
          finite(sample.timestampMs) &&
          typeof sample.assetId === "string" &&
          strings(sample.evidence) &&
          strings(sample.differences),
      )
    )
      return false;
  }
  return true;
}

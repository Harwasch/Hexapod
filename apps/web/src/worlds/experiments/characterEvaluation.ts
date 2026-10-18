import type { ConsistencyAssessment } from "../intelligence/client";
import type { ExperimentResult, FrameObservation } from "./types";
export function validatedAssessment(value: unknown): ConsistencyAssessment {
  if (!value || typeof value !== "object")
    throw new Error("The vision evaluator returned no assessment.");
  const result = value as Record<string, unknown>;
  if (
    result.source !== "vision-llm" ||
    result.metric !== "qualitative-appearance-consistency" ||
    typeof result.score !== "number" ||
    !Number.isFinite(result.score) ||
    result.score < 0 ||
    result.score > 1 ||
    typeof result.confidence !== "number" ||
    !Number.isFinite(result.confidence) ||
    result.confidence < 0 ||
    result.confidence > 1
  )
    throw new Error("Invalid qualitative appearance score.");
  if (
    !Array.isArray(result.evidence) ||
    !result.evidence.length ||
    result.evidence.length > 10 ||
    !result.evidence.every((value) => typeof value === "string" && value.length <= 1000) ||
    !Array.isArray(result.differences) ||
    result.differences.length > 10 ||
    !result.differences.every((value) => typeof value === "string" && value.length <= 1000) ||
    typeof result.note !== "string"
  )
    throw new Error("The vision evaluator omitted valid visual evidence.");
  return result as unknown as ConsistencyAssessment;
}
export function evaluationSamples(
  observations: FrameObservation[],
  limit: number,
): FrameObservation[] {
  const available = observations.filter((frame) => frame.assetId);
  const count = Math.min(limit, available.length);
  if (count === 0) return [];
  if (count === 1) return available.slice(-1);
  return Array.from(
    { length: count },
    (_, index) => available[Math.round((index * (available.length - 1)) / (count - 1))],
  ).filter((frame): frame is FrameObservation => !!frame);
}
export function updateCharacterAggregate(result: ExperimentResult): void {
  const evaluation = result.characterEvaluation;
  if (!evaluation) return;
  if (!evaluation.samples.length) {
    evaluation.source = "not-evaluated";
    return;
  }
  evaluation.source = "vision-llm";
  evaluation.meanScore =
    evaluation.samples.reduce((sum, sample) => sum + sample.score, 0) / evaluation.samples.length;
  evaluation.meanConfidence =
    evaluation.samples.reduce((sum, sample) => sum + sample.confidence, 0) /
    evaluation.samples.length;
}

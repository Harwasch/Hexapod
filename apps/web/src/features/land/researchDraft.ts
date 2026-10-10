import type { components } from "@twin/contracts";
import type { LandContextState } from "@/state/landContext";

export type QuestionFocus = LandContextState["researchFocus"];
export type QuestionBudget = Required<components["schemas"]["ResearchBudget"]>;
export interface CapturedResearchRequest {
  investigationId: string;
  boundaryRevision: number;
  question: string;
  key: string;
  budget: QuestionBudget;
  focus: QuestionFocus;
}
export interface ResearchDraft {
  version: 1;
  landId: string;
  boundaryRevision: number;
  question: string;
  focus: QuestionFocus;
  budget: QuestionBudget;
  newTopic: boolean;
  investigationId: string | null;
  pending: CapturedResearchRequest | null;
}
export const researchDraftKey = (scope: string, landId: string) =>
  `living-world-land-draft:${encodeURIComponent(scope)}:question:${landId}`;
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i;
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("The saved question has invalid fields.");
  return value as Record<string, unknown>;
}
function identifier(value: unknown): string {
  if (typeof value !== "string" || !UUID.test(value))
    throw new Error("The saved question has an invalid record reference.");
  return value;
}
function number(value: unknown, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < min || value > max)
    throw new Error("The saved question has invalid limits.");
  return value;
}
function question(value: unknown): string {
  if (typeof value !== "string" || value.length > 10_000)
    throw new Error("The saved question is invalid or too long.");
  return value;
}
function focus(raw: unknown): QuestionFocus {
  if (raw === null) return null;
  const value = object(raw);
  if (typeof value.label !== "string" || value.label.length > 300)
    throw new Error("The saved map feature label is invalid.");
  return {
    artifactId: identifier(value.artifactId),
    featureIndex: number(value.featureIndex, 0, 1999),
    label: value.label,
  };
}
function budget(raw: unknown): QuestionBudget {
  const value = object(raw);
  return {
    maxSteps: number(value.maxSteps, 1, 100),
    maxSeconds: number(value.maxSeconds, 10, 1800),
    maxOutputTokens: number(value.maxOutputTokens, 500, 100_000),
    maxWebSearches: number(value.maxWebSearches, 0, 30),
  };
}
export function parseResearchDraft(text: string, landId: string): ResearchDraft {
  if (text.length > 100_000) throw new Error("The saved question exceeds the recovery limit.");
  const value = object(JSON.parse(text));
  if (value.version !== 1 || value.landId !== landId || typeof value.newTopic !== "boolean")
    throw new Error("This saved question belongs to other land or uses an unsupported format.");
  let pending: CapturedResearchRequest | null = null;
  if (value.pending !== null) {
    const operation = object(value.pending);
    const text = question(operation.question);
    if (!text.trim()) throw new Error("The captured research request has no question.");
    pending = {
      investigationId: identifier(operation.investigationId),
      boundaryRevision: number(operation.boundaryRevision, 1, Number.MAX_SAFE_INTEGER),
      question: text,
      key: identifier(operation.key),
      budget: budget(operation.budget),
      focus: focus(operation.focus),
    };
  }
  return {
    version: 1,
    landId,
    boundaryRevision: number(value.boundaryRevision, 1, Number.MAX_SAFE_INTEGER),
    question: question(value.question),
    focus: focus(value.focus),
    budget: budget(value.budget),
    newTopic: value.newTopic,
    investigationId: value.investigationId === null ? null : identifier(value.investigationId),
    pending,
  };
}

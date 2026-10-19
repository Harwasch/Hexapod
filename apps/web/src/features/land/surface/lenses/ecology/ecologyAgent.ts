import type { SurveyRead } from "./useEcologySurveys";

/** What the lens hands the agent when the person asks about its results; they still send it. */
export const ECOLOGY_QUESTION =
  "Review the saved ecological evidence and any field surveys. What local reference evidence is still needed to define restoration targets?";

export function agentQuestionForSurvey(survey: Pick<SurveyRead, "id" | "name">): string {
  return `Interpret field survey ${survey.id} (${survey.name}), including sampling and identification limitations, and suggest restoration questions.`;
}

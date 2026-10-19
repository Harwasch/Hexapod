import { SolarAssessmentCompact } from "./surface/lenses/solar/SolarAssessmentInfo";
import type { SolarAssessment } from "./surface/lenses/solar/solarDraft";

/**
 * A saved hourly solar assessment outside the Solar lens (research results, the scenario
 * form). The lens owns the full view; this keeps the old entry point working.
 */
export function SolarAssessmentView({
  id,
  onUse,
}: {
  id: string;
  onUse?: (assessment: SolarAssessment) => void;
}) {
  return <SolarAssessmentCompact id={id} onUse={onUse} />;
}

import { useState } from "react";
import { Download, Focus, PencilLine, Sparkles } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { agentQuestionForSurvey } from "./ecologyAgent";
import {
  DESIGN_LABELS,
  formatDay,
  IDENTIFICATION_LABELS,
  METHOD_LABELS,
  percent,
  topSpecies,
} from "./ecologyData";
import { downloadJson, type SurveyRead, type SurveySummary } from "./useEcologySurveys";
import { askAgent } from "../../agentBus";
import { areaLabel } from "../../landFlow";
import { useFlyTo } from "../../mapHooks";

/** Species by cover, as bars: the picture of a survey. */
export function SpeciesBars({
  summary,
  limit,
  label = "Species cover",
}: {
  summary: SurveySummary;
  limit?: number;
  label?: string;
}) {
  const species = topSpecies(summary, limit ?? summary.species.length);
  if (!species.length)
    return <p className="land-muted">No species were recorded. That is not proof of absence.</p>;
  return (
    <ul className="land-eco-bars" aria-label={label}>
      {species.map((item) => (
        <li key={`${item.taxon}:${item.stratum}`}>
          <span className="land-eco-bars__name">
            <i>{item.taxon}</i>
            <small>{item.stratum}</small>
          </span>
          <span className="land-eco-bars__value">{percent(item.meanPercent)}</span>
          <span
            className="land-eco-bars__track"
            role="img"
            aria-label={`${item.meanPercent.toFixed(2)} percent area-weighted cover`}
          >
            <span style={{ width: `${Math.min(100, item.meanPercent)}%` }} />
          </span>
        </li>
      ))}
    </ul>
  );
}

/** What a survey (or a reviewed draft) found: how much was sampled and the cover of each species. */
export function SurveySummaryView({ summary }: { summary: SurveySummary }) {
  const plots = summary.plotAreasM2.length;
  return (
    <section className="land-eco-summary" aria-label="Survey results">
      <p className="land-lede">
        Sampled {areaLabel(summary.sampledAreaM2)} in {plots} {plots === 1 ? "plot" : "plots"},{" "}
        {(summary.sampledFraction * 100).toFixed(2)}% of the land.
      </p>
      <SpeciesBars summary={summary} />
      <p className="land-muted">
        Cover is the area-weighted mean over the plots. Species and layers overlap, so totals may
        exceed 100%.
      </p>
      {summary.species.length > 0 && (
        <details className="land-eco-group">
          <summary>Figures for each species</summary>
          <ul className="land-eco-plain">
            {summary.species.map((item) => (
              <li key={`${item.taxon}:${item.stratum}`} className="land-muted">
                <i>{item.taxon}</i> ({item.stratum}): {item.meanPercent.toFixed(2)}% mean, plot
                range {item.minimumPercent.toFixed(2)}–{item.maximumPercent.toFixed(2)}%, measured
                in {item.measuredPlots} {item.measuredPlots === 1 ? "plot" : "plots"} (
                {(item.assessedSampleFraction * 100).toFixed(1)}% of the sampled area).
                Identification:{" "}
                {item.identifications.map((id) => IDENTIFICATION_LABELS[id] ?? id).join(", ")}.
              </li>
            ))}
          </ul>
        </details>
      )}
      <details className="land-eco-group">
        <summary>How to interpret this survey</summary>
        <ul className="land-eco-plain">
          {summary.limitations.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </details>
    </section>
  );
}

/** A saved survey in full, for Details: its record, every plot and observation, and its tools. */
export function SurveyRecord({
  land,
  survey,
  canEdit,
  onCorrect,
}: {
  land: LandArea;
  survey: SurveyRead;
  canEdit: boolean;
  onCorrect: (survey: SurveyRead) => void;
}) {
  const fly = useFlyTo();
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="land-eco-record">
      <p className="land-muted">
        {formatDay(survey.observedOn)} · {survey.observer} · {METHOD_LABELS[survey.method]} ·{" "}
        {DESIGN_LABELS[survey.design]} · layers: {survey.assessedStrata.join(", ")}
      </p>
      {survey.methodNotes && <p className="land-eco-quote">{survey.methodNotes}</p>}
      {survey.stale && (
        <p className="land-note">
          Recorded on an earlier boundary (revision {survey.boundaryRevision}; this land is on{" "}
          {land.revision}).
        </p>
      )}
      <SurveySummaryView summary={survey.summary} />
      <ul className="land-eco-plain">
        {survey.plots.map((plot, index) => (
          <li key={`${plot.label}:${index}`} className="land-eco-plot">
            <div className="land-row">
              <strong>{plot.label}</strong>
              <span className="land-muted">
                {areaLabel(survey.summary.plotAreasM2[index] ?? 0)}
                {plot.samplePoints != null ? ` · ${plot.samplePoints} sampled points` : ""}
                {plot.completeInventory ? " · inventory complete" : " · unlisted species unknown"}
              </span>
              <GlassButton
                size="sm"
                variant="ghost"
                leadingIcon={<Focus size={14} aria-hidden="true" />}
                onClick={() => fly(plot.boundary)}
              >
                Frame {plot.label}
              </GlassButton>
              <button
                type="button"
                className="land-link"
                aria-expanded={open === index}
                onClick={() => setOpen(open === index ? null : index)}
              >
                {plot.observations.length}{" "}
                {plot.observations.length === 1 ? "observation" : "observations"}
              </button>
            </div>
            {open === index && (
              <ul className="land-eco-plain">
                {plot.observations.map((observation) => (
                  <li key={`${observation.taxon}:${observation.stratum}`}>
                    <i>{observation.taxon}</i> · {observation.stratum} ·{" "}
                    {observation.percentCover != null
                      ? `${observation.percentCover}% cover`
                      : `${observation.hits ?? 0} points with taxon`}{" "}
                    · {IDENTIFICATION_LABELS[observation.identification]}
                    {observation.notes ? `. ${observation.notes}` : ""}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
      <p className="land-muted">
        Saved immutably on boundary revision {survey.boundaryRevision}
        {survey.supersedesId ? ", correcting an earlier survey" : ""}. SHA-256{" "}
        <code className="land-eco-code">{survey.sha256}</code>
      </p>
      <div className="land-row">
        <GlassButton
          size="sm"
          leadingIcon={<Download size={14} aria-hidden="true" />}
          onClick={() => downloadJson(survey, "field-survey.json")}
        >
          Export field survey
        </GlassButton>
        {canEdit && (
          <GlassButton
            size="sm"
            leadingIcon={<PencilLine size={14} aria-hidden="true" />}
            onClick={() => onCorrect(survey)}
          >
            Create corrected copy
          </GlassButton>
        )}
        <GlassButton
          size="sm"
          variant="ghost"
          leadingIcon={<Sparkles size={14} aria-hidden="true" />}
          onClick={() => askAgent(agentQuestionForSurvey(survey))}
        >
          Ask about this survey
        </GlassButton>
      </div>
    </div>
  );
}

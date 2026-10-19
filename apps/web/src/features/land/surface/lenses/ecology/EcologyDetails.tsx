import { useId, useState } from "react";
import { Play, Upload } from "lucide-react";

import type { LandArea, LandEvidence } from "@twin/contracts";
import { GlassButton, GlassSelect, GlassTextarea } from "@twin/ui";

import { EvidenceView } from "../../../EvidenceView";
import {
  PROVIDERS,
  SOURCE_LABELS,
  STATUS_LABELS,
  formatDay,
  type EcologyContext,
} from "./ecologyData";
import { EcologyTable } from "./EcologyTable";
import { SurveyRecord } from "./EcologySurveyView";
import type { EcologyResearch } from "./useEcologyResearch";
import { PAGE, type SurveyCatalog, type SurveyDraftState } from "./useEcologySurveys";

const EVIDENCE_PAGE = 10;

/**
 * Everything behind the summary: the source and name choices, the saved runs, every table the
 * sources returned, methods, licences and evidence, and the selected survey's full record.
 */
export function EcologyDetails({
  land,
  research,
  context,
  catalog,
  survey,
  surveyNames,
}: {
  land: LandArea;
  research: EcologyResearch;
  context: EcologyContext | null;
  catalog: SurveyCatalog;
  survey: SurveyDraftState;
  /** Names in the survey on screen (the draft, or the selected saved survey). */
  surveyNames: string[];
}) {
  const id = useId();
  const [evidence, setEvidence] = useState<LandEvidence | null>(null);
  const [evidencePage, setEvidencePage] = useState(0);
  const { choices, setChoices, chosenRequest, canEdit } = research;
  const sources = Object.values(context?.sources ?? {});
  const evidenceItems = context?.evidence ?? [];
  const evidencePages = Math.max(1, Math.ceil(evidenceItems.length / EVIDENCE_PAGE));
  const licences = licencesOf(evidenceItems);
  const methods = methodsOf(context);
  const surveys = catalog.surveys;

  return (
    <>
      {canEdit && (
        <details className="land-eco-group">
          <summary>Choose sources and names</summary>
          <div className="land-eco-group__body">
            <fieldset className="land-eco-fieldset">
              <legend className="land-muted">Sources</legend>
              {(
                [
                  ["regions", "EPA ecoregions · U.S., 2011 edition"],
                  ["sites", "USDA ecological sites at a sample point · U.S."],
                  ["occurrences", "GBIF species records · worldwide, uneven coverage"],
                ] as const
              ).map(([key, label]) => (
                <label key={key} className="land-eco-check">
                  <input
                    type="checkbox"
                    checked={choices[key]}
                    disabled={research.busy}
                    onChange={(event) => setChoices({ [key]: event.target.checked })}
                  />
                  {label}
                </label>
              ))}
            </fieldset>
            <label className="land-field" htmlFor={`${id}-names`}>
              Scientific names to review (optional)
              <GlassTextarea
                id={`${id}-names`}
                aria-label="Scientific names to review"
                rows={3}
                disabled={research.busy}
                value={choices.names}
                placeholder={"Quercus alba\nOne scientific name per line"}
                onChange={(event) => setChoices({ names: event.target.value })}
              />
            </label>
            <label className="land-field" htmlFor={`${id}-kingdom`}>
              Kingdom hint
              <GlassSelect
                id={`${id}-kingdom`}
                value={choices.kingdom}
                disabled={research.busy}
                onChange={(event) => setChoices({ kingdom: event.target.value })}
              >
                <option value="">Unknown or mixed</option>
                <option value="Plantae">Plants</option>
                <option value="Animalia">Animals</option>
                <option value="Fungi">Fungi</option>
              </GlassSelect>
            </label>
            {surveyNames.length > 0 && (
              <button
                type="button"
                className="land-link"
                disabled={research.busy}
                onClick={() => setChoices({ names: surveyNames.join("\n") })}
              >
                Use names from the displayed survey
              </button>
            )}
            <p className="land-muted">
              Up to 20 distinct names. A name match does not confirm what was seen in the field,
              whether it is native, or whether it suits restoration.
            </p>
            {!chosenRequest && (
              <p role="status" className="land-muted">
                Choose at least one source or name. Use at most 20 distinct names, each under 201
                characters.
              </p>
            )}
            <div className="land-row">
              <GlassButton
                size="sm"
                leadingIcon={<Play size={14} aria-hidden="true" />}
                disabled={!chosenRequest || research.busy || research.running}
                onClick={() => chosenRequest && research.start(chosenRequest)}
              >
                {research.busy ? "Starting…" : "Run with these choices"}
              </GlassButton>
            </div>
          </div>
        </details>
      )}

      {research.saved.length > 1 && (
        <label className="land-field" htmlFor={`${id}-run`}>
          Saved ecological research
          <GlassSelect
            id={`${id}-run`}
            value={research.selectedId ?? ""}
            onChange={(event) => research.choose(event.target.value)}
          >
            {research.saved.map((item) => (
              <option key={item.id} value={item.id}>
                {new Date(item.createdAt).toLocaleString()} ·{" "}
                {item.boundaryRevision === land.revision ? "current boundary" : "earlier boundary"}
              </option>
            ))}
          </GlassSelect>
        </label>
      )}

      {context?.regions && (
        <details className="land-eco-group">
          <summary>Ecoregions ({context.regions.table.rows.length})</summary>
          <div className="land-eco-group__body">
            <EcologyTable
              label="Ecoregions"
              columns={context.regions.table.columns}
              rows={context.regions.table.rows}
            />
          </div>
        </details>
      )}

      {context?.sites && (
        <details className="land-eco-group">
          <summary>
            Plant communities at the sample point ({context.sites.table.rows.length})
          </summary>
          <div className="land-eco-group__body">
            <p className="land-muted">
              One soil map unit at one point inside the land. Its components may not match the whole
              land.
            </p>
            <EcologyTable
              label="Ecological sites"
              columns={context.sites.table.columns}
              rows={context.sites.table.rows}
            />
          </div>
        </details>
      )}

      {context?.species && (
        <details className="land-eco-group">
          <summary>Species records ({context.species.records})</summary>
          <div className="land-eco-group__body">
            <p className="land-muted">
              A sample of openly licensed public records whose published position falls in the land.
              Positions are not shown, to protect sensitive species. Records do not prove a species
              is here now.
            </p>
            <EcologyTable
              label="Species records"
              columns={context.species.table.columns}
              rows={context.species.table.rows}
              italic="species"
            />
          </div>
        </details>
      )}

      {context?.names && (
        <details className="land-eco-group">
          <summary>Name review ({context.names.rows.length})</summary>
          <div className="land-eco-group__body">
            <EcologyTable
              label="Name review"
              columns={context.names.tables[0]?.columns ?? []}
              rows={context.names.rows}
              italic="supplied"
            />
          </div>
        </details>
      )}

      {(sources.length > 0 || research.eventsError) && (
        <details className="land-eco-group">
          <summary>Source coverage</summary>
          <div className="land-eco-group__body">
            <ul className="land-eco-plain">
              {sources.map((source) => (
                <li key={source.provider}>
                  <strong>{SOURCE_LABELS[source.provider] ?? source.provider}</strong>:{" "}
                  {STATUS_LABELS[source.status] ?? source.status}. {source.message}
                </li>
              ))}
            </ul>
            {research.eventsError && (
              <p role="status" className="land-muted">
                Source activity could not be loaded.{" "}
                <button type="button" className="land-link" onClick={research.retryEvents}>
                  Retry source activity
                </button>
              </p>
            )}
          </div>
        </details>
      )}

      {(methods.length > 0 || licences.length > 0) && (
        <details className="land-eco-group">
          <summary>Methods and licences</summary>
          <div className="land-eco-group__body">
            {methods.map((method) => (
              <p key={method.title} className="land-muted">
                <strong>{method.title}.</strong> {method.text}
              </p>
            ))}
            <ul className="land-eco-plain">
              {licences.map((item) => (
                <li key={item.provider} className="land-muted">
                  <strong>{SOURCE_LABELS[item.provider] ?? item.provider}</strong>:{" "}
                  {item.licences.join(", ")}. {item.attribution}
                </li>
              ))}
            </ul>
          </div>
        </details>
      )}

      {evidenceItems.length > 0 && (
        <details className="land-eco-group">
          <summary>Evidence ({evidenceItems.length})</summary>
          <div className="land-eco-group__body">
            {evidence && <EvidenceView evidence={evidence} onClose={() => setEvidence(null)} />}
            <ul className="land-eco-plain">
              {evidenceItems
                .slice(evidencePage * EVIDENCE_PAGE, evidencePage * EVIDENCE_PAGE + EVIDENCE_PAGE)
                .map((item, index) => (
                  <li key={item.id}>
                    <button type="button" className="land-link" onClick={() => setEvidence(item)}>
                      Source {evidencePage * EVIDENCE_PAGE + index + 1}
                    </button>{" "}
                    <span className="land-muted">
                      {SOURCE_LABELS[item.provider] ?? item.provider} · {item.title}
                    </span>
                  </li>
                ))}
            </ul>
            {evidencePages > 1 && (
              <div className="land-row">
                <button
                  type="button"
                  className="land-link"
                  disabled={evidencePage === 0}
                  onClick={() => setEvidencePage(evidencePage - 1)}
                >
                  Previous sources
                </button>
                <button
                  type="button"
                  className="land-link"
                  disabled={evidencePage >= evidencePages - 1}
                  onClick={() => setEvidencePage(evidencePage + 1)}
                >
                  More sources
                </button>
              </div>
            )}
          </div>
        </details>
      )}

      {catalog.selected && (
        <details className="land-eco-group" open>
          <summary>Survey record · {catalog.selected.name}</summary>
          <div className="land-eco-group__body">
            <SurveyRecord
              land={land}
              survey={catalog.selected}
              canEdit={canEdit}
              onCorrect={survey.correct}
            />
          </div>
        </details>
      )}
      {catalog.selectedError && (
        <p role="alert" className="land-alert">
          The selected survey could not be loaded.
        </p>
      )}

      {(canEdit || surveys.length > 0 || catalog.offset > 0) && (
        <details className="land-eco-group">
          <summary>Field survey files</summary>
          <div className="land-eco-group__body">
            {surveys.length > 0 && (
              <ul className="land-eco-plain">
                {surveys.map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      className="land-link"
                      aria-pressed={catalog.selectedId === item.id}
                      onClick={() =>
                        catalog.select(catalog.selectedId === item.id ? null : item.id)
                      }
                    >
                      {item.name}
                    </button>{" "}
                    <span className="land-muted">
                      {formatDay(item.observedOn)}
                      {item.stale ? " · older boundary" : ""}
                      {item.supersedesId ? " · correction" : ""}
                    </span>
                  </li>
                ))}
              </ul>
            )}
            {(catalog.offset > 0 || surveys.length >= PAGE) && (
              <div className="land-row">
                <button
                  type="button"
                  className="land-link"
                  disabled={!catalog.offset}
                  onClick={() => catalog.setOffset(Math.max(0, catalog.offset - PAGE))}
                >
                  Previous surveys
                </button>
                <button
                  type="button"
                  className="land-link"
                  disabled={surveys.length < PAGE}
                  onClick={() => catalog.setOffset(catalog.offset + PAGE)}
                >
                  More surveys
                </button>
              </div>
            )}
            {canEdit && !survey.draft && (
              <label className="land-eco-file">
                <Upload size={14} aria-hidden="true" />
                Import survey JSON
                <input
                  type="file"
                  accept=".json,application/json"
                  disabled={survey.busy}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) survey.importFile(file);
                    event.target.value = "";
                  }}
                />
              </label>
            )}
            {survey.error && !survey.draft && (
              <p role="alert" className="land-alert">
                {survey.error}
              </p>
            )}
          </div>
        </details>
      )}
    </>
  );
}

/** Each artifact's method, and each finding's caveat where the method does not already say it. */
function methodsOf(context: EcologyContext | null): { title: string; text: string }[] {
  if (!context) return [];
  const out = context.artifacts.map((artifact) => ({
    title: artifact.title,
    text: artifact.method,
  }));
  for (const finding of context.findings) {
    if (!finding.uncertainty) continue;
    if (out.some((item) => item.text.includes(finding.uncertainty))) continue;
    out.push({ title: finding.title, text: finding.uncertainty });
  }
  const seen = new Set<string>();
  return out.filter((item) => {
    const key = `${item.title}:${item.text}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/** Licences and attribution, one line per source (species records carry their own licences). */
function licencesOf(evidence: LandEvidence[]) {
  const byProvider = new Map<string, { licences: Set<string>; attribution: string }>();
  for (const item of evidence) {
    const entry = byProvider.get(item.provider) ?? {
      licences: new Set<string>(),
      attribution:
        item.provider === PROVIDERS.species
          ? "GBIF and the original data publishers; each record names its publisher."
          : item.attribution,
    };
    entry.licences.add(item.license);
    byProvider.set(item.provider, entry);
  }
  return [...byProvider.entries()].map(([provider, entry]) => ({
    provider,
    licences: [...entry.licences],
    attribution: entry.attribution,
  }));
}

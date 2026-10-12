import { useEffect, useRef, useState, type CSSProperties } from "react";
import { ArrowLeft, Calculator, Download, Plus, X } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";

import type { components, LandArea } from "@twin/contracts";
import { GlassButton, GlassInput, GlassSelect, GlassTextarea } from "@twin/ui";

import { ApiError, api, unwrap } from "@/api/client";

import { RestorationTargets } from "../../../RestorationTargets";
import { ScenarioResultView } from "../../../ScenarioResultView";
import {
  downloadScenarioDraft,
  scenarioDraftKey,
  serializeScenarioDraft,
  type ScenarioDraft,
} from "../../../scenarioDraft";
import type { RestorationInputs, ScenarioResult } from "../../../scenarioDefaults";
import { LensCard, LensDetails } from "../../LensCard";
import { useLens } from "../../lenses";
import { PlanField } from "./PlanField";
import { useDraftAutosave, type ScenarioSeed } from "./planRecovery";
import { planError, useScenarioSurveys } from "./plansApi";
import { scenarioNumbers } from "./plansFormat";
import "../../../restoration.css";

/** One acre is exactly 0.40468564224 hectares. */
const ACRES_PER_HECTARE = 1 / 0.40468564224;

function shown(value: number, scale = 1): number | "" {
  return Number.isFinite(value) ? Number((value * scale).toPrecision(10)) : "";
}

/**
 * A restoration scenario: what the land should become, its cover now and as a target, the
 * treatments with their areas, costs and years, and the monitoring visits. Calculated on the
 * server before it can be saved; species goals and the estimate's basis are folded away.
 */
export function PlanScenarioEditor({
  land,
  scope,
  seed,
  onSaved,
  onClose,
}: {
  land: LandArea;
  scope: string;
  seed: ScenarioSeed;
  onSaved: (id: string) => void;
  onClose: (discarded: boolean) => void;
}) {
  const cache = useQueryClient();
  const surveys = useScenarioSurveys(land);
  const [name, setName] = useState(seed.name);
  const [inputs, setInputs] = useState(seed.inputs);
  const [monitoringText, setMonitoringText] = useState(seed.monitoringText);
  const [boundaryRevision, setBoundaryRevision] = useState(seed.boundaryRevision);
  const [fieldSurveyIds, setFieldSurveyIds] = useState(seed.fieldSurveyIds);
  const [solarAssessmentId] = useState(seed.solarAssessmentId);
  const [evidenceIds] = useState(seed.evidenceIds);
  const [editing, setEditing] = useState(seed.editing);
  const [conflict, setConflict] = useState(seed.conflict);
  const [requestKey, setRequestKey] = useState(seed.requestKey);
  const [notice] = useState(seed.notice);
  const [preview, setPreview] = useState<{ signature: string; result: ScenarioResult } | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const operation = useRef(0);
  useEffect(
    () => () => {
      operation.current += 1;
    },
    [],
  );

  const payload = {
    name,
    boundaryRevision,
    inputs,
    evidenceIds,
    fieldSurveyIds,
    solarAssessmentId,
  } satisfies components["schemas"]["ScenarioCreate"];
  const signature = serializeScenarioDraft(payload);
  const snapshot: ScenarioDraft = {
    version: 1,
    landId: land.id,
    requestKey,
    payload,
    editing,
    monitoringText,
  };
  const serialized = serializeScenarioDraft(snapshot);
  const [initial] = useState(serialized);
  const { storageError, clear } = useDraftAutosave(
    scenarioDraftKey(scope, land.id),
    serialized,
    2_000_000,
  );
  const result = preview?.signature === signature ? preview.result : null;

  if (inputs.kind !== "restoration")
    return (
      <LensCard lens="plans" headline="This is a solar scenario.">
        <p className="land-lede">Solar scenarios are made and changed in the Solar lens.</p>
        <div className="land-row">
          <GlassButton variant="primary" onClick={() => useLens.getState().setLens("solar")}>
            Open in Solar
          </GlassButton>
          <GlassButton variant="ghost" onClick={() => onClose(false)}>
            Back
          </GlassButton>
        </div>
      </LensCard>
    );
  const restoration: RestorationInputs = inputs;
  const set = (change: Partial<RestorationInputs>) => setInputs({ ...restoration, ...change });
  const targetTotal = restoration.cover.reduce(
    (sum, row) => sum + (Number.isFinite(row.targetPercent) ? row.targetPercent : 0),
    0,
  );

  const calculate = async (save: boolean) => {
    if (save && conflict) return;
    const ticket = ++operation.current;
    setBusy(true);
    setError(null);
    try {
      if (!save) {
        const value = await unwrap(
          api.POST("/api/v1/land/{land_id}/scenarios/preview", {
            params: { path: { land_id: land.id } },
            body: { ...payload, requestKey },
          }),
        );
        if (ticket === operation.current) setPreview({ signature, result: value });
      } else {
        const saved = editing
          ? await unwrap(
              api.PUT("/api/v1/land/{land_id}/scenarios/{scenario_id}", {
                params: { path: { land_id: land.id, scenario_id: editing.id } },
                body: { ...payload, requestKey, expectedRevision: editing.revision },
              }),
            )
          : await unwrap(
              api.POST("/api/v1/land/{land_id}/scenarios", {
                params: { path: { land_id: land.id } },
                body: { ...payload, requestKey },
              }),
            );
        await Promise.all([
          cache.invalidateQueries({ queryKey: ["land-scenarios", scope, land.id] }),
          // A plan citing this scenario is now out of date; its stale reasons say so.
          cache.invalidateQueries({ queryKey: ["land-actions", scope, land.id] }),
        ]);
        if (ticket !== operation.current) return;
        clear();
        onSaved(saved.id);
      }
    } catch (cause) {
      if (ticket !== operation.current) return;
      setError(planError(cause));
      if (editing && cause instanceof ApiError && cause.status === 409) {
        try {
          const current = await unwrap(
            api.GET("/api/v1/land/{land_id}/scenarios/{scenario_id}", {
              params: { path: { land_id: land.id, scenario_id: editing.id } },
            }),
          );
          if (ticket === operation.current) setConflict(current);
        } catch {
          /* The original error and the browser copy remain. */
        }
      }
    } finally {
      if (ticket === operation.current) setBusy(false);
    }
  };

  const surveyChoices = surveys.data?.filter((item) => item.boundaryRevision === boundaryRevision);
  const cover = restoration.cover;
  const treatments = restoration.treatments;

  return (
    <LensCard
      lens="plans"
      headline={editing ? "Change this scenario" : "A restoration scenario"}
      action={
        <GlassButton
          size="sm"
          variant="ghost"
          aria-label="Back to the comparison"
          leadingIcon={<ArrowLeft size={14} aria-hidden="true" />}
          onClick={() => {
            const untouched = !seed.recovered && serialized === initial;
            if (untouched) clear();
            onClose(untouched);
          }}
        >
          Compare
        </GlassButton>
      }
    >
      <p className="land-lede">
        Say what the land should become, how its cover would change, and what the work costs.
        Starting values are guesses to replace, not measurements.
      </p>
      {notice && (
        <p className="land-muted" role="status">
          {notice}
        </p>
      )}
      {storageError && (
        <p className="land-alert" role="alert">
          {storageError}
        </p>
      )}
      {conflict && (
        <div className="land-note" role="region" aria-label="Review newer scenario revision">
          <p>
            This scenario is now revision {conflict.revision}. Your draft started from revision{" "}
            {editing?.revision}.
          </p>
          <details className="plans-more">
            <summary>The saved assumptions</summary>
            <pre className="plans-pre">{JSON.stringify(conflict.inputs, null, 2)}</pre>
          </details>
          <div className="land-row">
            <GlassButton
              size="sm"
              variant="primary"
              disabled={busy}
              onClick={() => {
                setEditing(null);
                setConflict(null);
                setRequestKey(crypto.randomUUID());
                setPreview(null);
                setError(null);
              }}
            >
              Keep my assumptions as a new scenario
            </GlassButton>
            <GlassButton
              size="sm"
              disabled={busy}
              onClick={() => {
                setEditing({ id: conflict.id, revision: conflict.revision, landId: land.id });
                setName(conflict.name);
                setInputs(conflict.inputs);
                if (conflict.inputs.kind === "restoration")
                  setMonitoringText(conflict.inputs.monitoringYears.join(", "));
                setBoundaryRevision(conflict.boundaryRevision);
                setFieldSurveyIds(conflict.fieldSurveyIds ?? []);
                setRequestKey(crypto.randomUUID());
                setConflict(null);
                setPreview(null);
              }}
            >
              Load latest saved assumptions
            </GlassButton>
          </div>
        </div>
      )}
      {boundaryRevision !== land.revision && (
        <p className="land-note">
          Made for an earlier outline of the land.{" "}
          <button
            type="button"
            className="land-link"
            onClick={() => setBoundaryRevision(land.revision)}
          >
            Recalculate against the current outline
          </button>
        </p>
      )}
      <form
        className="plans-edit"
        aria-label="Restoration scenario"
        onSubmit={(event) => {
          event.preventDefault();
          void calculate(false);
        }}
      >
        <fieldset disabled={busy}>
          <PlanField label="Scenario name">
            <GlassInput
              required
              maxLength={200}
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </PlanField>
          <PlanField label="What should it become?">
            <GlassInput
              aria-label="Reference ecosystem"
              required
              placeholder="e.g. Native oak savanna"
              value={restoration.referenceEcosystem}
              onChange={(event) => set({ referenceEcosystem: event.target.value })}
            />
          </PlanField>

          <h4 className="plans-edit__heading">
            Cover now and as a target
            <span
              className={Math.abs(targetTotal - 100) < 0.01 ? "land-muted" : "plans-warn"}
              aria-live="polite"
            >
              Targets total {Math.round(targetTotal * 100) / 100}%
              {Math.abs(targetTotal - 100) < 0.01 ? "" : " · should be 100%"}
            </span>
          </h4>
          <ul className="plans-cover">
            {cover.map((row, index) => (
              <li key={index}>
                <div className="plans-cover__head">
                  <GlassInput
                    aria-label={`Cover class ${index + 1} name`}
                    required
                    value={row.name}
                    onChange={(event) =>
                      set({
                        cover: cover.map((item, i) =>
                          i === index ? { ...item, name: event.target.value } : item,
                        ),
                      })
                    }
                  />
                  <GlassButton
                    iconOnly
                    size="sm"
                    variant="ghost"
                    aria-label={`Remove cover class ${index + 1}`}
                    onClick={() => set({ cover: cover.filter((_, i) => i !== index) })}
                  >
                    <X size={14} aria-hidden="true" />
                  </GlassButton>
                </div>
                <div className="plans-cover__bars" aria-hidden="true">
                  <i style={{ "--w": `${shown(row.baselinePercent) || 0}%` } as CSSProperties} />
                  <i
                    className="is-target"
                    style={{ "--w": `${shown(row.targetPercent) || 0}%` } as CSSProperties}
                  />
                </div>
                <div className="plans-edit-grid">
                  <PlanField label="Now (%)">
                    <GlassInput
                      aria-label={`Cover class ${index + 1} current cover (%)`}
                      type="number"
                      min={0}
                      max={100}
                      step="any"
                      value={shown(row.baselinePercent)}
                      onChange={(event) =>
                        set({
                          cover: cover.map((item, i) =>
                            i === index
                              ? { ...item, baselinePercent: event.target.valueAsNumber }
                              : item,
                          ),
                        })
                      }
                    />
                  </PlanField>
                  <PlanField label="Target (%)">
                    <GlassInput
                      aria-label={`Cover class ${index + 1} target cover (%)`}
                      type="number"
                      min={0}
                      max={100}
                      step="any"
                      value={shown(row.targetPercent)}
                      onChange={(event) =>
                        set({
                          cover: cover.map((item, i) =>
                            i === index
                              ? { ...item, targetPercent: event.target.valueAsNumber }
                              : item,
                          ),
                        })
                      }
                    />
                  </PlanField>
                </div>
                <PlanField label="How you know">
                  <GlassInput
                    aria-label={`Cover class ${index + 1} evidence`}
                    required
                    value={row.evidenceBasis}
                    onChange={(event) =>
                      set({
                        cover: cover.map((item, i) =>
                          i === index ? { ...item, evidenceBasis: event.target.value } : item,
                        ),
                      })
                    }
                  />
                </PlanField>
              </li>
            ))}
          </ul>
          <div className="land-row">
            <GlassButton
              size="sm"
              leadingIcon={<Plus size={14} aria-hidden="true" />}
              onClick={() =>
                set({
                  cover: [
                    ...cover,
                    { name: "", baselinePercent: 0, targetPercent: 0, evidenceBasis: "" },
                  ],
                })
              }
            >
              Add cover class
            </GlassButton>
          </div>

          <h4 className="plans-edit__heading">Work and costs</h4>
          <ul className="plans-cover">
            {treatments.map((treatment, index) => (
              <li key={index}>
                <div className="plans-cover__head">
                  <GlassInput
                    aria-label={`Treatment ${index + 1} name`}
                    placeholder="e.g. Remove invasive brush"
                    required
                    value={treatment.name}
                    onChange={(event) =>
                      set({
                        treatments: treatments.map((item, i) =>
                          i === index ? { ...item, name: event.target.value } : item,
                        ),
                      })
                    }
                  />
                  <GlassButton
                    iconOnly
                    size="sm"
                    variant="ghost"
                    aria-label={`Remove treatment ${index + 1}`}
                    onClick={() => set({ treatments: treatments.filter((_, i) => i !== index) })}
                  >
                    <X size={14} aria-hidden="true" />
                  </GlassButton>
                </div>
                <PlanField label="What it does">
                  <GlassInput
                    aria-label={`Treatment ${index + 1} objective`}
                    required
                    value={treatment.objective}
                    onChange={(event) =>
                      set({
                        treatments: treatments.map((item, i) =>
                          i === index ? { ...item, objective: event.target.value } : item,
                        ),
                      })
                    }
                  />
                </PlanField>
                <div className="plans-edit-grid">
                  <PlanField label="Acres">
                    <GlassInput
                      aria-label={`Treatment ${index + 1} area (acres)`}
                      type="number"
                      min={0}
                      step="any"
                      value={shown(treatment.areaHa, ACRES_PER_HECTARE)}
                      onChange={(event) =>
                        set({
                          treatments: treatments.map((item, i) =>
                            i === index
                              ? { ...item, areaHa: event.target.valueAsNumber / ACRES_PER_HECTARE }
                              : item,
                          ),
                        })
                      }
                    />
                  </PlanField>
                  <PlanField label="Cost per acre">
                    <GlassInput
                      aria-label={`Treatment ${index + 1} cost per acre`}
                      type="number"
                      min={0}
                      step="any"
                      value={shown(treatment.costPerHa, 1 / ACRES_PER_HECTARE)}
                      onChange={(event) =>
                        set({
                          treatments: treatments.map((item, i) =>
                            i === index
                              ? {
                                  ...item,
                                  costPerHa: event.target.valueAsNumber * ACRES_PER_HECTARE,
                                }
                              : item,
                          ),
                        })
                      }
                    />
                  </PlanField>
                  <PlanField label="Year">
                    <GlassInput
                      aria-label={`Treatment ${index + 1} year`}
                      type="number"
                      min={0}
                      step={1}
                      value={shown(treatment.year)}
                      onChange={(event) =>
                        set({
                          treatments: treatments.map((item, i) =>
                            i === index ? { ...item, year: event.target.valueAsNumber } : item,
                          ),
                        })
                      }
                    />
                  </PlanField>
                </div>
              </li>
            ))}
          </ul>
          <div className="land-row">
            <GlassButton
              size="sm"
              leadingIcon={<Plus size={14} aria-hidden="true" />}
              onClick={() =>
                set({
                  treatments: [
                    ...treatments,
                    {
                      name: "",
                      objective: "",
                      areaHa: land.areaM2 / 10_000,
                      costPerHa: 0,
                      year: 0,
                    },
                  ],
                })
              }
            >
              Add treatment
            </GlassButton>
          </div>
          <div className="plans-edit-grid plans-edit-grid--two">
            <PlanField label="Monitoring years (comma separated)">
              <GlassInput
                value={monitoringText}
                onChange={(event) => {
                  setMonitoringText(event.target.value);
                  set({
                    monitoringYears: event.target.value
                      .split(",")
                      .map((value) => (value.trim() ? Number(value.trim()) : Number.NaN)),
                  });
                }}
              />
            </PlanField>
            <PlanField label="Cost per monitoring visit">
              <GlassInput
                type="number"
                step="any"
                min={0}
                value={shown(restoration.monitoringCostPerVisit)}
                onChange={(event) => set({ monitoringCostPerVisit: event.target.valueAsNumber })}
              />
            </PlanField>
          </div>

          <LensDetails summary="How the estimate was made">
            <PlanField label="Assessment date">
              <GlassInput
                type="date"
                required
                value={restoration.surveyDate}
                onChange={(event) => set({ surveyDate: event.target.value })}
              />
            </PlanField>
            <PlanField label="Evidence type">
              <GlassSelect
                value={restoration.confidence}
                onChange={(event) =>
                  set({ confidence: event.target.value as RestorationInputs["confidence"] })
                }
              >
                <option value="user-estimate">My estimate</option>
                <option value="remote-estimate">Remote sensing estimate</option>
                <option value="field-survey">Field survey</option>
              </GlassSelect>
            </PlanField>
            <PlanField label="Survey or estimation method">
              <GlassTextarea
                required
                rows={2}
                value={restoration.surveyMethod}
                onChange={(event) => set({ surveyMethod: event.target.value })}
              />
            </PlanField>
            <fieldset className="plans-checks">
              <legend>Field surveys behind it</legend>
              {surveyChoices?.map((survey) => (
                <label className="plans-check" key={survey.id}>
                  <input
                    type="checkbox"
                    checked={fieldSurveyIds.includes(survey.id)}
                    onChange={(event) =>
                      setFieldSurveyIds(
                        event.target.checked
                          ? [...fieldSurveyIds, survey.id]
                          : fieldSurveyIds.filter((id) => id !== survey.id),
                      )
                    }
                  />
                  {survey.name} · {survey.observedOn}
                </label>
              ))}
              {!surveyChoices?.length && (
                <p className="land-muted">Record field surveys in Ecology to cite them here.</p>
              )}
              {fieldSurveyIds
                .filter((id) => !surveyChoices?.some((survey) => survey.id === id))
                .map((id) => (
                  <p key={id} className="land-muted">
                    Kept a survey from an earlier outline.{" "}
                    <button
                      type="button"
                      className="land-link"
                      onClick={() =>
                        setFieldSurveyIds((ids) => ids.filter((value) => value !== id))
                      }
                    >
                      Remove reference
                    </button>
                  </p>
                ))}
              <p className="land-muted">
                Species in a survey can overlap. Turn them into separate cover classes yourself.
              </p>
            </fieldset>
            <div className="plans-edit-grid">
              <PlanField label="Contingency (%)">
                <GlassInput
                  type="number"
                  step="any"
                  min={0}
                  max={100}
                  value={shown(restoration.contingencyFraction, 100)}
                  onChange={(event) =>
                    set({ contingencyFraction: event.target.valueAsNumber / 100 })
                  }
                />
              </PlanField>
              <PlanField label="Discount rate (%)">
                <GlassInput
                  type="number"
                  step="any"
                  min={0}
                  max={50}
                  value={shown(restoration.discountRate, 100)}
                  onChange={(event) => set({ discountRate: event.target.valueAsNumber / 100 })}
                />
              </PlanField>
              <PlanField label="Currency code">
                <GlassInput
                  required
                  pattern="[A-Z]{3}"
                  maxLength={3}
                  value={restoration.currency}
                  onChange={(event) => set({ currency: event.target.value.toUpperCase() })}
                />
              </PlanField>
            </div>
          </LensDetails>
          <LensDetails summary="Species goals and monitoring">
            <RestorationTargets
              landId={land.id}
              plan={restoration.ecology}
              onChange={(ecology) => set({ ecology })}
              surveys={surveyChoices ?? []}
              treatmentNames={treatments.map((treatment) => treatment.name)}
            />
          </LensDetails>
          <PlanField label="Assumptions and missing evidence">
            <GlassTextarea
              required
              rows={2}
              value={restoration.assumptions}
              onChange={(event) => set({ assumptions: event.target.value })}
            />
          </PlanField>
        </fieldset>
        {error && (
          <p className="land-alert" role="alert">
            {error}
          </p>
        )}
        {result && (
          <div className="plans-preview" aria-live="polite">
            <ul className="land-facts">
              {scenarioNumbers({ inputs: restoration, result }).map((item) => (
                <li key={item.label} className="land-fact">
                  <span className="land-fact__label">{item.label}</span>
                  <span className="land-fact__value">{item.value}</span>
                </li>
              ))}
            </ul>
            <details className="plans-more">
              <summary>The full calculation</summary>
              <ScenarioResultView result={result} />
            </details>
          </div>
        )}
        <div className="land-row plans-edit__actions">
          <GlassButton
            type="submit"
            variant={result ? "default" : "primary"}
            loading={busy}
            disabled={busy}
            leadingIcon={<Calculator size={14} aria-hidden="true" />}
          >
            {busy ? "Calculating…" : "Calculate scenario"}
          </GlassButton>
          <GlassButton
            variant={result ? "primary" : "default"}
            disabled={busy || !result || Boolean(conflict)}
            onClick={() => void calculate(true)}
          >
            {editing ? "Save scenario revision" : "Save scenario"}
          </GlassButton>
          <GlassButton
            variant="ghost"
            disabled={busy}
            onClick={() => {
              clear();
              onClose(true);
            }}
          >
            Discard
          </GlassButton>
          <GlassButton
            size="sm"
            variant="ghost"
            className="plans-edit__download"
            leadingIcon={<Download size={13} aria-hidden="true" />}
            onClick={() => downloadScenarioDraft(snapshot)}
          >
            Download scenario draft
          </GlassButton>
        </div>
      </form>
    </LensCard>
  );
}

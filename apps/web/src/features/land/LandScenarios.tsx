import { useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, LandArea } from "@twin/contracts";
import { api, unwrap, ApiError } from "@/api/client";
import { useLandContext } from "@/state/landContext";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { SolarStudy } from "./SolarStudy";
import { SolarAssessmentView } from "./SolarAssessmentView";
import { physicalFields, solarFinance, type SolarAssessment } from "./solarStudy";
import { ScenarioResultView } from "./ScenarioResultView";
import { fieldLabel, valueLabel } from "./scenarioDefaults";
import {
  restorationDefaults,
  solarDefaults,
  solarGroups,
  type Scenario,
  type ScenarioInputs,
  type ScenarioResult,
} from "./scenarioDefaults";

export function LandScenarios({ land }: { land: LandArea }) {
  const scope = useLandScope(),
    ready = useLandAccessReady(),
    canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const key = ["land-scenarios", scope, land.id];
  const catalog = useQuery({
    queryKey: key,
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/scenarios", { params: { path: { land_id: land.id } } }),
      ),
    retry: false,
  });
  const [inputs, setInputs] = useState<ScenarioInputs | null>(null);
  const [name, setName] = useState("");
  const [monitoringText, setMonitoringText] = useState("1, 3, 5");
  const [editing, setEditing] = useState<Scenario | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [history, setHistory] = useState<Scenario[]>([]);
  const [evidenceIds, setEvidenceIds] = useState<string[]>([]);
  const [solarAssessmentId, setSolarAssessmentId] = useState<string | null>(null);
  const [fieldSurveyIds, setFieldSurveyIds] = useState<string[]>([]);
  const surveys = useQuery({
    queryKey: ["land-surveys", scope, land.id, "scenario-references"],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/surveys", {
          params: { path: { land_id: land.id }, query: { limit: 100 } },
        }),
      ),
    retry: false,
  });
  const [boundaryRevision, setBoundaryRevision] = useState(land.revision);
  const [preview, setPreview] = useState<{ signature: string; result: ScenarioResult } | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const operation = useRef(0);
  const requestKey = useRef(crypto.randomUUID());
  const payload = inputs
    ? ({
        name,
        boundaryRevision,
        inputs,
        evidenceIds,
        fieldSurveyIds,
        solarAssessmentId,
      } satisfies components["schemas"]["ScenarioCreate"])
    : null;
  const signature = JSON.stringify(payload);
  const visiblePreview = preview?.signature === signature ? preview.result : null;
  const choices = catalog.data?.filter((scenario) => selected.includes(scenario.id)) ?? [];
  const reset = () => {
    setHistory([]);
    operation.current++;
    requestKey.current = crypto.randomUUID();
    setInputs(null);
    setEditing(null);
    setPreview(null);
    setError(null);
    setBusy(false);
  };
  const begin = (kind: "solar" | "restoration") => {
    reset();
    setInputs(kind === "solar" ? solarDefaults() : restorationDefaults());
    setName(kind === "solar" ? "Solar option" : "Restoration option");
    setBoundaryRevision(land.revision);
    setEvidenceIds([]);
    setFieldSurveyIds([]);
    setSolarAssessmentId(null);
    setMonitoringText("1, 3, 5");
  };
  const calculate = async (save: boolean) => {
    if (!payload) return;
    const ticket = ++operation.current;
    setBusy(true);
    setError(null);
    try {
      if (!save) {
        const result = await unwrap(
          api.POST("/api/v1/land/{land_id}/scenarios/preview", {
            params: { path: { land_id: land.id } },
            body: { ...payload, requestKey: requestKey.current },
          }),
        );
        if (ticket === operation.current) setPreview({ signature, result });
      } else {
        const saved = editing
          ? await unwrap(
              api.PUT("/api/v1/land/{land_id}/scenarios/{scenario_id}", {
                params: { path: { land_id: land.id, scenario_id: editing.id } },
                body: { ...payload, expectedRevision: editing.revision },
              }),
            )
          : await unwrap(
              api.POST("/api/v1/land/{land_id}/scenarios", {
                params: { path: { land_id: land.id } },
                body: { ...payload, requestKey: requestKey.current },
              }),
            );
        if (ticket === operation.current) {
          setSelected([saved.id]);
          setInputs(null);
          setEditing(null);
          setPreview(null);
        }
        await cache.invalidateQueries({ queryKey: key });
      }
    } catch (cause) {
      if (ticket === operation.current)
        setError(
          cause instanceof ApiError && cause.fieldErrors.length
            ? cause.fieldErrors.join(". ")
            : describeError(cause),
        );
    } finally {
      if (ticket === operation.current) setBusy(false);
    }
  };
  const edit = (scenario: Scenario) => {
    reset();
    setEditing(scenario);
    setName(scenario.name);
    setInputs(scenario.inputs);
    if (scenario.inputs.kind === "restoration")
      setMonitoringText(scenario.inputs.monitoringYears.join(", "));
    setBoundaryRevision(scenario.boundaryRevision);
    setEvidenceIds(scenario.evidenceIds ?? []);
    setFieldSurveyIds(scenario.fieldSurveyIds ?? []);
    setSolarAssessmentId(scenario.solarAssessmentId ?? null);
  };
  const useSolar = (assessment: SolarAssessment) => {
    reset();
    setInputs(solarFinance(assessment));
    setName(`Solar economics · ${assessment.request.year}`);
    setBoundaryRevision(assessment.boundaryRevision);
    setSolarAssessmentId(assessment.id);
    setEvidenceIds([]);
    setFieldSurveyIds([]);
  };
  return (
    <section className="land-scenarios" aria-label="Land scenarios">
      <div className="land-place-heading">
        <div>
          <span className="land-eyebrow">Explore what could be</span>
          <h3>Compare possibilities.</h3>
        </div>
      </div>
      <p>
        Turn evidence and your assumptions into a calculation you can inspect, revise and compare.
      </p>
      {!inputs && <SolarStudy land={land} onUse={useSolar} />}
      {!inputs && canEdit && (
        <div className="land-actions">
          <button type="button" onClick={() => begin("solar")}>
            Solar and economics
          </button>
          <button type="button" onClick={() => begin("restoration")}>
            Restoration and cover
          </button>
        </div>
      )}
      {error && (
        <p className="land-error" role="alert">
          {error}
        </p>
      )}
      {catalog.isError && (
        <p role="alert">
          Scenarios could not be loaded.{" "}
          <button type="button" onClick={() => void catalog.refetch()}>
            Retry
          </button>
        </p>
      )}
      {inputs && (
        <form
          className="land-scenario-form"
          onSubmit={(event) => {
            event.preventDefault();
            void calculate(false);
          }}
        >
          <label className="land-name">
            Scenario name
            <input
              value={name}
              maxLength={200}
              required
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <p className="land-notice">
            Starting values are editable planning assumptions, not measurements. Enter the resource,
            survey and cost data you want to test.
          </p>
          <p>Uses land boundary revision {boundaryRevision}.</p>
          {solarAssessmentId && (
            <div className="land-notice">
              <p>
                Uses saved hourly AC generation. Physical assumptions are locked to that assessment;
                financial assumptions remain editable.
              </p>
              <SolarAssessmentView id={solarAssessmentId} />
              <button type="button" onClick={() => setSolarAssessmentId(null)}>
                Switch to manual generation assumptions
              </button>
            </div>
          )}
          {inputs.kind === "restoration" && (
            <fieldset>
              <legend>Field survey references</legend>
              <p className="land-footnote">
                Link the observations behind your assumptions. Species can overlap; interpret them
                before defining exclusive cover classes.
              </p>
              {surveys.data
                ?.filter((s) => s.boundaryRevision === boundaryRevision)
                .map((s) => (
                  <label className="land-check" key={s.id}>
                    <input
                      type="checkbox"
                      checked={fieldSurveyIds.includes(s.id)}
                      onChange={(e) =>
                        setFieldSurveyIds(
                          e.target.checked
                            ? [...fieldSurveyIds, s.id]
                            : fieldSurveyIds.filter((id) => id !== s.id),
                        )
                      }
                    />
                    {s.name} · {s.observedOn}
                  </label>
                ))}
              {!surveys.data?.length && (
                <p>Record field observations in Ecology to add survey references.</p>
              )}
              {fieldSurveyIds
                .filter(
                  (id) =>
                    !surveys.data?.some(
                      (s) => s.id === id && s.boundaryRevision === boundaryRevision,
                    ),
                )
                .map((id) => (
                  <p key={id}>
                    Retained survey reference {id}.{" "}
                    <button
                      type="button"
                      onClick={() => setFieldSurveyIds((ids) => ids.filter((v) => v !== id))}
                    >
                      Remove reference
                    </button>
                  </p>
                ))}
            </fieldset>
          )}
          {boundaryRevision !== land.revision && (
            <button type="button" onClick={() => setBoundaryRevision(land.revision)}>
              Recalculate against current boundary revision {land.revision}
            </button>
          )}
          {inputs.kind === "solar" ? (
            <>
              {solarGroups.map((group, index) => (
                <details key={group.title} open={index === 0}>
                  <summary>{group.title}</summary>
                  <div className="land-scenario-grid">
                    {group.fields.map((field) => {
                      const value = inputs[field.key];
                      return (
                        <label key={field.key} className="land-name">
                          {field.label}
                          <input
                            type="number"
                            disabled={!!solarAssessmentId && physicalFields.has(field.key)}
                            step="any"
                            required={field.key !== "replacementYear"}
                            value={
                              typeof value === "number"
                                ? Number((value * (field.percent ? 100 : 1)).toPrecision(10))
                                : ""
                            }
                            onChange={(event) =>
                              setInputs({
                                ...inputs,
                                [field.key]:
                                  event.target.value === "" && field.key === "replacementYear"
                                    ? null
                                    : Number(event.target.value) / (field.percent ? 100 : 1),
                              })
                            }
                          />
                        </label>
                      );
                    })}
                  </div>
                </details>
              ))}
              <label className="land-name">
                Solar resource source and method
                <textarea
                  required
                  rows={2}
                  readOnly={!!solarAssessmentId}
                  value={inputs.irradiationBasis}
                  onChange={(event) =>
                    setInputs({ ...inputs, irradiationBasis: event.target.value })
                  }
                  placeholder="Source, period, and how tilt/orientation were accounted for"
                />
              </label>
            </>
          ) : (
            <>
              <label className="land-name">
                Reference ecosystem
                <input
                  required
                  value={inputs.referenceEcosystem}
                  onChange={(event) =>
                    setInputs({ ...inputs, referenceEcosystem: event.target.value })
                  }
                />
              </label>
              <label className="land-name">
                Assessment date
                <input
                  type="date"
                  required
                  value={inputs.surveyDate}
                  onChange={(event) => setInputs({ ...inputs, surveyDate: event.target.value })}
                />
              </label>
              <label className="land-name">
                Evidence type
                <select
                  value={inputs.confidence}
                  onChange={(event) =>
                    setInputs({
                      ...inputs,
                      confidence: event.target.value as typeof inputs.confidence,
                    })
                  }
                >
                  <option value="user-estimate">User estimate</option>
                  <option value="remote-estimate">Remote sensing estimate</option>
                  <option value="field-survey">Field survey</option>
                </select>
              </label>
              <label className="land-name">
                Survey or estimation method
                <textarea
                  required
                  value={inputs.surveyMethod}
                  onChange={(event) => setInputs({ ...inputs, surveyMethod: event.target.value })}
                />
              </label>
              <h4>Cover classes · targets must total 100%</h4>
              {inputs.cover.map((cover, index) => (
                <fieldset key={index}>
                  <legend>Cover class {index + 1}</legend>
                  <label className="land-name">
                    Species or cover class
                    <input
                      required
                      value={cover.name}
                      onChange={(event) =>
                        setInputs({
                          ...inputs,
                          cover: inputs.cover.map((row, i) =>
                            i === index ? { ...row, name: event.target.value } : row,
                          ),
                        })
                      }
                    />
                  </label>
                  <div className="land-scenario-grid">
                    {(["baselinePercent", "targetPercent"] as const).map((field) => (
                      <label className="land-name" key={field}>
                        {field === "baselinePercent" ? "Current cover (%)" : "Target cover (%)"}
                        <input
                          type="number"
                          min={0}
                          max={100}
                          step="any"
                          value={cover[field]}
                          onChange={(event) =>
                            setInputs({
                              ...inputs,
                              cover: inputs.cover.map((row, i) =>
                                i === index ? { ...row, [field]: Number(event.target.value) } : row,
                              ),
                            })
                          }
                        />
                      </label>
                    ))}
                  </div>
                  <label className="land-name">
                    Evidence for this estimate
                    <input
                      required
                      value={cover.evidenceBasis}
                      onChange={(event) =>
                        setInputs({
                          ...inputs,
                          cover: inputs.cover.map((row, i) =>
                            i === index ? { ...row, evidenceBasis: event.target.value } : row,
                          ),
                        })
                      }
                    />
                  </label>
                  <button
                    type="button"
                    onClick={() =>
                      setInputs({ ...inputs, cover: inputs.cover.filter((_, i) => i !== index) })
                    }
                  >
                    Remove class
                  </button>
                </fieldset>
              ))}
              <button
                type="button"
                onClick={() =>
                  setInputs({
                    ...inputs,
                    cover: [
                      ...inputs.cover,
                      { name: "", baselinePercent: 0, targetPercent: 0, evidenceBasis: "" },
                    ],
                  })
                }
              >
                Add cover class
              </button>
              <h4>Treatments and costs</h4>
              {inputs.treatments.map((treatment, index) => (
                <fieldset key={index}>
                  <legend>Treatment {index + 1}</legend>
                  {(["name", "objective"] as const).map((field) => (
                    <label className="land-name" key={field}>
                      {fieldLabel(field)}
                      <input
                        required
                        value={treatment[field]}
                        onChange={(event) =>
                          setInputs({
                            ...inputs,
                            treatments: inputs.treatments.map((row, i) =>
                              i === index ? { ...row, [field]: event.target.value } : row,
                            ),
                          })
                        }
                      />
                    </label>
                  ))}
                  <div className="land-scenario-grid">
                    {(["areaHa", "costPerHa", "year"] as const).map((field) => (
                      <label className="land-name" key={field}>
                        {fieldLabel(field)}
                        <input
                          type="number"
                          step="any"
                          min={0}
                          value={treatment[field]}
                          onChange={(event) =>
                            setInputs({
                              ...inputs,
                              treatments: inputs.treatments.map((row, i) =>
                                i === index ? { ...row, [field]: Number(event.target.value) } : row,
                              ),
                            })
                          }
                        />
                      </label>
                    ))}
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      setInputs({
                        ...inputs,
                        treatments: inputs.treatments.filter((_, i) => i !== index),
                      })
                    }
                  >
                    Remove treatment
                  </button>
                </fieldset>
              ))}
              <button
                type="button"
                onClick={() =>
                  setInputs({
                    ...inputs,
                    treatments: [
                      ...inputs.treatments,
                      {
                        name: "",
                        objective: "",
                        areaHa: land.areaM2 / 10000,
                        costPerHa: 0,
                        year: 0,
                      },
                    ],
                  })
                }
              >
                Add treatment
              </button>
              <label className="land-name">
                Monitoring years (comma separated)
                <input
                  value={monitoringText}
                  onChange={(event) => {
                    setMonitoringText(event.target.value);
                    setInputs({
                      ...inputs,
                      monitoringYears: event.target.value
                        .split(",")
                        .map((value) => (value.trim() ? Number(value.trim()) : Number.NaN)),
                    });
                  }}
                />
              </label>
              <label className="land-name">
                Cost per monitoring visit
                <input
                  type="number"
                  step="any"
                  min={0}
                  value={inputs.monitoringCostPerVisit}
                  onChange={(event) =>
                    setInputs({ ...inputs, monitoringCostPerVisit: Number(event.target.value) })
                  }
                />
              </label>
              <label className="land-name">
                Contingency (%)
                <input
                  type="number"
                  step="any"
                  min={0}
                  max={100}
                  value={inputs.contingencyFraction * 100}
                  onChange={(event) =>
                    setInputs({ ...inputs, contingencyFraction: Number(event.target.value) / 100 })
                  }
                />
              </label>
              <label className="land-name">
                Discount rate (%)
                <input
                  type="number"
                  step="any"
                  min={0}
                  max={50}
                  value={inputs.discountRate * 100}
                  onChange={(event) =>
                    setInputs({ ...inputs, discountRate: Number(event.target.value) / 100 })
                  }
                />
              </label>
            </>
          )}
          <label className="land-name">
            Currency code
            <input
              required
              pattern="[A-Z]{3}"
              maxLength={3}
              value={inputs.currency}
              onChange={(event) =>
                setInputs({ ...inputs, currency: event.target.value.toUpperCase() })
              }
            />
          </label>
          <label className="land-name">
            Assumptions and missing evidence
            <textarea
              required
              rows={3}
              value={inputs.assumptions}
              onChange={(event) => setInputs({ ...inputs, assumptions: event.target.value })}
            />
          </label>
          <div className="land-actions">
            <button disabled={busy} type="submit">
              {busy ? "Calculating…" : "Calculate scenario"}
            </button>
            <button
              type="button"
              disabled={busy || !visiblePreview}
              onClick={() => void calculate(true)}
            >
              {editing ? "Save scenario revision" : "Save scenario"}
            </button>
            <button type="button" onClick={reset}>
              Cancel
            </button>
          </div>
          {visiblePreview && <ScenarioResultView result={visiblePreview} />}
        </form>
      )}
      <div className="land-scenario-list">
        {catalog.data?.map((scenario) => (
          <label key={scenario.id}>
            <input
              type="checkbox"
              checked={selected.includes(scenario.id)}
              disabled={!selected.includes(scenario.id) && choices.length >= 3}
              onChange={() =>
                setSelected(
                  selected.includes(scenario.id)
                    ? selected.filter((id) => id !== scenario.id)
                    : [...selected, scenario.id],
                )
              }
            />
            <span>
              {scenario.name} · revision {scenario.revision}
              {scenario.stale ? " · earlier boundary" : ""}
            </span>
          </label>
        ))}
      </div>
      {choices.length > 1 && (
        <div className="land-table-scroll">
          <table>
            <caption>Scenario comparison · entered currencies retained</caption>
            <thead>
              <tr>
                <th>Measure</th>
                {choices.map((scenario) => (
                  <th key={scenario.id}>{scenario.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ...new Set(choices.flatMap((scenario) => Object.keys(scenario.result.summary))),
              ].map((field) => (
                <tr key={field}>
                  <th>{fieldLabel(field)}</th>
                  {choices.map((scenario) => (
                    <td key={scenario.id}>{valueLabel(scenario.result.summary[field])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {choices.map((scenario) => (
        <article key={scenario.id}>
          <h4>{scenario.name}</h4>
          <p>{scenario.inputs.assumptions}</p>
          <ScenarioResultView result={scenario.result} />
          {scenario.solarAssessmentId && (
            <details>
              <summary>Linked hourly solar assessment</summary>
              <SolarAssessmentView id={scenario.solarAssessmentId} />
            </details>
          )}
          {!!scenario.fieldSurveyIds?.length && (
            <div className="land-actions">
              {scenario.fieldSurveyIds.map((id) => (
                <button
                  type="button"
                  key={id}
                  onClick={() => {
                    const context = useLandContext.getState();
                    context.selectSurvey(id);
                    context.setSection("ecology");
                  }}
                >
                  Open survey {surveys.data?.find((s) => s.id === id)?.name ?? id}
                </button>
              ))}
            </div>
          )}
          <div className="land-actions">
            {canEdit && (
              <button type="button" onClick={() => edit(scenario)}>
                Edit assumptions
              </button>
            )}
            <button
              type="button"
              onClick={() => {
                void unwrap(
                  api.GET("/api/v1/land/{land_id}/scenarios/{scenario_id}/revisions", {
                    params: { path: { land_id: land.id, scenario_id: scenario.id } },
                  }),
                )
                  .then(setHistory)
                  .catch((cause: unknown) => setError(describeError(cause)));
              }}
            >
              Revision history
            </button>
            <button
              type="button"
              onClick={() => {
                const url = URL.createObjectURL(
                  new Blob([JSON.stringify(scenario, null, 2)], { type: "application/json" }),
                );
                const anchor = document.createElement("a");
                anchor.href = url;
                anchor.download = "land-scenario.json";
                anchor.click();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
              }}
            >
              Export assumptions and results
            </button>
          </div>
        </article>
      ))}
      {history.length > 0 && (
        <details open>
          <summary>Scenario revisions</summary>
          {history.map((scenario) => (
            <div key={`${scenario.id}:${scenario.revision}`}>
              <strong>
                {scenario.name} · revision {scenario.revision}
              </strong>
              <p>
                Boundary revision {scenario.boundaryRevision} ·{" "}
                {new Date(scenario.updatedAt).toLocaleDateString()}
              </p>
              <ScenarioResultView result={scenario.result} />
              {scenario.solarAssessmentId && (
                <details>
                  <summary>Linked hourly solar assessment</summary>
                  <SolarAssessmentView id={scenario.solarAssessmentId} />
                </details>
              )}
              {!!scenario.fieldSurveyIds?.length && (
                <div className="land-actions">
                  {scenario.fieldSurveyIds.map((id) => (
                    <button
                      type="button"
                      key={id}
                      onClick={() => {
                        const context = useLandContext.getState();
                        context.selectSurvey(id);
                        context.setSection("ecology");
                      }}
                    >
                      Open survey {surveys.data?.find((s) => s.id === id)?.name ?? id}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </details>
      )}
    </section>
  );
}

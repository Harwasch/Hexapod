import type { components } from "@twin/contracts";
import { LandEvidencePicker } from "./LandEvidencePicker";

type Plan = components["schemas"]["RestorationEcology"];
type Target = components["schemas"]["SpeciesTarget"];
type Response = components["schemas"]["CoverResponse"];
type Survey = components["schemas"]["SurveyRead"];

function NumberField({
  label,
  value,
  onChange,
  max = 100,
  integer = false,
}: {
  label: string;
  value: number | null | undefined;
  onChange: (value: number | null) => void;
  max?: number;
  integer?: boolean;
}) {
  return (
    <label className="land-name">
      {label}
      <input
        type="number"
        min={0}
        max={max}
        step={integer ? 1 : "any"}
        value={value != null && Number.isFinite(value) ? value : ""}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
      />
    </label>
  );
}

export function RestorationTargets({
  landId,
  plan,
  onChange,
  surveys,
  treatmentNames,
}: {
  landId: string;
  plan: Plan | null | undefined;
  onChange: (plan: Plan | null) => void;
  surveys: Survey[];
  treatmentNames: string[];
}) {
  const update = (index: number, patch: Partial<Target>) => {
    if (plan)
      onChange({
        ...plan,
        speciesTargets: plan.speciesTargets.map((target, i) =>
          i === index ? { ...target, ...patch } : target,
        ),
      });
  };
  const newTarget = (): Target => ({
    taxon: "",
    stratum: "herb",
    baselineSurveyId: null,
    baselinePercent: null,
    baselineBasis: "",
    targetLow: Number.NaN,
    targetHigh: Number.NaN,
    targetYear: 5,
    targetBasis: "hypothetical",
    rationale: "",
    evidenceIds: [],
    treatmentNames: [],
    monitoringMethod: "",
    monitoringSeason: "",
    responseIfOffTrack: "",
    response: null,
  });
  if (!plan)
    return (
      <div className="land-notice">
        <p>
          Plan species-level outcomes alongside your cover and cost comparison. Preserve sampled
          baselines, define independent targets and specify how you will check them.
        </p>
        <button
          type="button"
          onClick={() =>
            onChange({
              referenceBasis: "",
              siteConstraints: "",
              referenceEvidenceIds: [],
              speciesTargets: [newTarget()],
            })
          }
        >
          Add species targets and monitoring
        </button>
      </div>
    );
  return (
    <section className="land-restoration-targets" aria-label="Species targets and monitoring">
      <h4>Species targets and monitoring</h4>
      <p>
        Species and vegetation layers may overlap. Their target percentages do not need to total
        100%. These goals are separate from the exclusive cover classes above.
      </p>
      <label className="land-name">
        Why this reference community fits the site
        <textarea
          required
          value={plan.referenceBasis}
          onChange={(e) => onChange({ ...plan, referenceBasis: e.target.value })}
        />
      </label>
      <LandEvidencePicker
        landId={landId}
        label="Reference community evidence"
        value={plan.referenceEvidenceIds ?? []}
        onChange={(ids) => onChange({ ...plan, referenceEvidenceIds: ids })}
      />
      <label className="land-name">
        Site constraints and unresolved questions
        <textarea
          required
          value={plan.siteConstraints}
          onChange={(e) => onChange({ ...plan, siteConstraints: e.target.value })}
          placeholder="Soils, hydrology, disturbance, land-use constraints and missing local evidence"
        />
      </label>
      {plan.speciesTargets.map((target, index) => {
        const survey = surveys.find((item) => item.id === target.baselineSurveyId);
        const response = target.response;
        const updateResponse = (patch: Partial<Response>) => {
          if (response) update(index, { response: { ...response, ...patch } });
        };
        return (
          <fieldset key={index}>
            <legend>Species target {index + 1}</legend>
            <label className="land-name">
              Taxon or field identifier
              <input
                required
                maxLength={200}
                value={target.taxon}
                onChange={(e) => update(index, { taxon: e.target.value })}
              />
            </label>
            <label className="land-name">
              Vegetation layer
              <select
                aria-label="Vegetation layer"
                value={target.stratum}
                onChange={(e) => update(index, { stratum: e.target.value as Target["stratum"] })}
              >
                {(["canopy", "shrub", "herb", "ground", "aquatic"] as const).map((stratum) => (
                  <option key={stratum}>{stratum}</option>
                ))}
              </select>
            </label>
            <label className="land-name">
              Baseline source
              <select
                aria-label="Baseline source"
                value={target.baselineSurveyId ?? ""}
                onChange={(e) =>
                  update(index, { baselineSurveyId: e.target.value || null, baselinePercent: null })
                }
              >
                <option value="">Unknown or entered assumption</option>
                {target.baselineSurveyId && !survey && (
                  <option value={target.baselineSurveyId}>Saved baseline survey</option>
                )}
                {surveys.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} · {item.observedOn}
                  </option>
                ))}
              </select>
            </label>
            {survey ? (
              <>
                <label className="land-name">
                  Use a recorded taxon and layer
                  <select
                    aria-label="Use a recorded taxon and layer"
                    value=""
                    onChange={(e) => {
                      const species = survey.summary.species[Number(e.target.value)];
                      if (e.target.value && species)
                        update(index, { taxon: species.taxon, stratum: species.stratum });
                    }}
                  >
                    <option value="">Choose an observation</option>
                    {survey.summary.species.map((species, i) => (
                      <option key={i} value={String(i)}>
                        {species.taxon} · {species.stratum} · {species.meanPercent.toFixed(2)}%
                      </option>
                    ))}
                  </select>
                </label>
                <p className="land-footnote">
                  The calculation reads this survey's saved sampled-plot mean. It does not
                  extrapolate to the whole land or treat unrecorded species as absent.
                </p>
              </>
            ) : (
              !target.baselineSurveyId && (
                <NumberField
                  label="Assumed baseline cover (%) — blank means unknown"
                  value={target.baselinePercent}
                  onChange={(value) => update(index, { baselinePercent: value })}
                />
              )
            )}
            <label className="land-name">
              Baseline method, scope and limitations
              <textarea
                required
                value={target.baselineBasis}
                onChange={(e) => update(index, { baselineBasis: e.target.value })}
              />
            </label>
            <div className="land-scenario-grid">
              <NumberField
                label="Target cover lower bound (%)"
                value={target.targetLow}
                onChange={(value) => update(index, { targetLow: value ?? Number.NaN })}
              />
              <NumberField
                label="Target cover upper bound (%)"
                value={target.targetHigh}
                onChange={(value) => update(index, { targetHigh: value ?? Number.NaN })}
              />
              <NumberField
                label="Target assessment year"
                value={target.targetYear}
                max={50}
                integer
                onChange={(value) => update(index, { targetYear: value ?? Number.NaN })}
              />
            </div>
            <label className="land-name">
              Target basis
              <select
                aria-label="Target basis"
                value={target.targetBasis ?? "hypothetical"}
                onChange={(e) =>
                  update(index, { targetBasis: e.target.value as Target["targetBasis"] })
                }
              >
                <option value="hypothetical">Hypothetical planning target</option>
                <option value="reference-evidence">Supported by linked reference evidence</option>
              </select>
            </label>
            <label className="land-name">
              Reason for the target and local applicability
              <textarea
                required
                value={target.rationale}
                onChange={(e) => update(index, { rationale: e.target.value })}
              />
            </label>
            <LandEvidencePicker
              landId={landId}
              label={`Species target ${index + 1} evidence`}
              limit={10}
              value={target.evidenceIds ?? []}
              onChange={(ids) => update(index, { evidenceIds: ids })}
            />
            <details>
              <summary>Link planned treatments · {target.treatmentNames?.length ?? 0}</summary>
              <p>
                Links explain intent. Treatment names and costs do not determine biological
                response.
              </p>
              {treatmentNames.filter(Boolean).map((name, i) => (
                <label key={i} className="land-check">
                  <input
                    type="checkbox"
                    checked={target.treatmentNames?.includes(name) ?? false}
                    onChange={(e) =>
                      update(index, {
                        treatmentNames: e.target.checked
                          ? [...(target.treatmentNames ?? []), name]
                          : target.treatmentNames?.filter((value) => value !== name),
                      })
                    }
                  />
                  {name}
                </label>
              ))}
              {target.treatmentNames
                ?.filter((name) => !treatmentNames.includes(name))
                .map((name) => (
                  <p key={name}>
                    Missing treatment: {name}{" "}
                    <button
                      type="button"
                      onClick={() =>
                        update(index, {
                          treatmentNames: target.treatmentNames?.filter((value) => value !== name),
                        })
                      }
                    >
                      Remove link
                    </button>
                  </p>
                ))}
            </details>
            <label className="land-name">
              Monitoring method and comparable sampling design
              <textarea
                required
                value={target.monitoringMethod}
                onChange={(e) => update(index, { monitoringMethod: e.target.value })}
              />
            </label>
            <label className="land-name">
              Monitoring season and timing
              <input
                required
                value={target.monitoringSeason}
                onChange={(e) => update(index, { monitoringSeason: e.target.value })}
              />
            </label>
            <label className="land-name">
              Response if observations are off track
              <textarea
                required
                value={target.responseIfOffTrack}
                onChange={(e) => update(index, { responseIfOffTrack: e.target.value })}
              />
            </label>
            <p className="land-footnote">
              Include year {Number.isFinite(target.targetYear) ? target.targetYear : "…"} in the
              scenario's monitoring schedule. Monitoring costs are counted in that schedule once.
            </p>
            <details>
              <summary>Optional conditional response curve</summary>
              <p>
                A what-if envelope from explicit rate and asymptote assumptions. No fitted
                ecological forecast or probability is implied.
              </p>
              {!response ? (
                <button
                  type="button"
                  onClick={() =>
                    update(index, {
                      response: {
                        startYear: 0,
                        asymptoteLow: Number.NaN,
                        asymptoteHigh: Number.NaN,
                        annualRateLow: Number.NaN,
                        annualRateHigh: Number.NaN,
                        basis: "",
                        evidenceIds: [],
                      },
                    })
                  }
                >
                  Enter response assumptions
                </button>
              ) : (
                <>
                  <div className="land-scenario-grid">
                    <NumberField
                      label="Response start year"
                      value={response.startYear}
                      max={49}
                      integer
                      onChange={(value) => updateResponse({ startYear: value ?? Number.NaN })}
                    />
                    <NumberField
                      label="Asymptotic cover lower bound (%)"
                      value={response.asymptoteLow}
                      onChange={(value) => updateResponse({ asymptoteLow: value ?? Number.NaN })}
                    />
                    <NumberField
                      label="Asymptotic cover upper bound (%)"
                      value={response.asymptoteHigh}
                      onChange={(value) => updateResponse({ asymptoteHigh: value ?? Number.NaN })}
                    />
                    <NumberField
                      label="Response rate lower bound (per year)"
                      value={response.annualRateLow}
                      max={5}
                      onChange={(value) => updateResponse({ annualRateLow: value ?? Number.NaN })}
                    />
                    <NumberField
                      label="Response rate upper bound (per year)"
                      value={response.annualRateHigh}
                      max={5}
                      onChange={(value) => updateResponse({ annualRateHigh: value ?? Number.NaN })}
                    />
                  </div>
                  <p className="land-footnote">
                    Rate controls approach to the assumed eventual cover. For example, 0.693 per
                    year halves the remaining gap each year; it is not a 69.3 percentage-point
                    annual gain.
                  </p>
                  <label className="land-name">
                    Parameter source, applicability and assumptions
                    <textarea
                      required
                      value={response.basis}
                      onChange={(e) => updateResponse({ basis: e.target.value })}
                    />
                  </label>
                  <LandEvidencePicker
                    landId={landId}
                    label={`Species target ${index + 1} response evidence`}
                    limit={10}
                    value={response.evidenceIds ?? []}
                    onChange={(ids) => updateResponse({ evidenceIds: ids })}
                  />
                  <button type="button" onClick={() => update(index, { response: null })}>
                    Remove response curve
                  </button>
                </>
              )}
            </details>
            <button
              type="button"
              onClick={() =>
                onChange({
                  ...plan,
                  speciesTargets: plan.speciesTargets.filter((_, i) => i !== index),
                })
              }
            >
              Remove species target
            </button>
          </fieldset>
        );
      })}
      <div className="land-actions">
        <button
          type="button"
          disabled={plan.speciesTargets.length >= 50}
          onClick={() =>
            onChange({ ...plan, speciesTargets: [...plan.speciesTargets, newTarget()] })
          }
        >
          Add another species target
        </button>
        <button type="button" onClick={() => onChange(null)}>
          Remove species plan
        </button>
      </div>
    </section>
  );
}

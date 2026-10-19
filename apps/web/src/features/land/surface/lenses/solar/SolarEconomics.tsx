import { useId, useState } from "react";
import { Download } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton, GlassInput, GlassTextarea } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { useLandContext } from "@/state/landContext";

import { landError } from "../../landFlow";
import {
  MAIN_FIELDS,
  MORE_FIELDS,
  downloadJson,
  physicalFields,
  scenarioNumbers,
  type MoneyField,
  type Scenario,
  type ScenarioResult,
  type SolarInputs,
} from "./solarEconomics";
import { dateLabel, energyLabel, moneyLabel, paybackWords } from "./solarWords";
import type { SolarEconomicsState } from "./useSolarEconomics";

function MoneyInput({
  field,
  inputs,
  onChange,
  locked,
}: {
  field: MoneyField;
  inputs: SolarInputs;
  onChange: (inputs: SolarInputs) => void;
  locked?: boolean;
}) {
  const id = useId();
  const value = inputs[field.key];
  const scale = field.percent ? 100 : 1;
  return (
    <label className="land-field" htmlFor={id}>
      {field.label}
      <GlassInput
        id={id}
        type="number"
        step="any"
        inputMode="decimal"
        disabled={locked}
        value={
          typeof value === "number" && Number.isFinite(value)
            ? Number((value * scale).toPrecision(10))
            : ""
        }
        onChange={(event) => {
          const text = event.target.value.trim();
          onChange({
            ...inputs,
            [field.key]: text === "" ? (field.optional ? null : Number.NaN) : Number(text) / scale,
          });
        }}
      />
    </label>
  );
}

/**
 * Costs and payback for an array. The few values that decide it come first; the rest wait
 * under "More assumptions". The result follows the values; Save keeps it.
 */
export function SolarEconomicsEditor({
  econ,
  land,
}: {
  econ: SolarEconomicsState;
  land: LandArea;
}) {
  const id = useId();
  const payload = econ.payload;
  const inputs = payload?.inputs;
  if (!payload || inputs?.kind !== "solar") return null;
  const linked = Boolean(payload.solarAssessmentId);
  const setInputs = econ.setInputs;
  return (
    <form
      className="land-solar-econ"
      aria-label="Costs and payback"
      onSubmit={(event) => {
        event.preventDefault();
        void econ.save();
      }}
    >
      <p className="land-lede">
        {linked
          ? "Uses the yearly energy from the hourly model. Enter what the system costs and what its power is worth."
          : "Enter the panel area, the sunlight and the money. Running the hourly model first gives a better energy figure."}
      </p>
      <fieldset className="land-solar-econ__fields" disabled={!econ.canEdit || econ.saving}>
        <legend className="land-visually-hidden">Costs and payback</legend>
        <div className="land-solar-grid">
          {MAIN_FIELDS.map((field) => (
            <MoneyInput key={field.key} field={field} inputs={inputs} onChange={setInputs} />
          ))}
          <label className="land-field" htmlFor={`${id}-currency`}>
            Currency
            <GlassInput
              id={`${id}-currency`}
              required
              maxLength={3}
              pattern="[A-Z]{3}"
              value={inputs.currency}
              onChange={(event) =>
                setInputs({ ...inputs, currency: event.target.value.toUpperCase() })
              }
            />
          </label>
        </div>
        <details className="land-solar-fold">
          <summary>More assumptions</summary>
          <label className="land-field" htmlFor={`${id}-name`}>
            Name
            <GlassInput
              id={`${id}-name`}
              required
              maxLength={200}
              value={payload.name}
              onChange={(event) => econ.update({ name: event.target.value })}
            />
          </label>
          {MORE_FIELDS.map((group) => (
            <div key={group.title} className="land-solar-group">
              <h4 className="land-solar-heading">{group.title}</h4>
              {linked && group.fields.some((field) => physicalFields.has(field.key)) && (
                <p className="land-solar-hint">
                  The array values come from the hourly model.{" "}
                  <button type="button" className="land-link" onClick={econ.unlink}>
                    Enter them yourself instead
                  </button>
                </p>
              )}
              <div className="land-solar-grid">
                {group.fields.map((field) => (
                  <MoneyInput
                    key={field.key}
                    field={field}
                    inputs={inputs}
                    onChange={setInputs}
                    locked={linked && physicalFields.has(field.key)}
                  />
                ))}
              </div>
            </div>
          ))}
          <label className="land-field" htmlFor={`${id}-basis`}>
            Where the sunlight figure came from
            <GlassTextarea
              id={`${id}-basis`}
              required
              rows={2}
              readOnly={linked}
              value={inputs.irradiationBasis}
              placeholder="Source, period, and how tilt and facing were accounted for"
              onChange={(event) => setInputs({ ...inputs, irradiationBasis: event.target.value })}
            />
          </label>
          <label className="land-field" htmlFor={`${id}-assumptions`}>
            Assumptions and missing evidence
            <GlassTextarea
              id={`${id}-assumptions`}
              required
              rows={3}
              value={inputs.assumptions}
              onChange={(event) => setInputs({ ...inputs, assumptions: event.target.value })}
            />
          </label>
          <p className="land-muted">
            Uses land outline {payload.boundaryRevision}
            {payload.boundaryRevision !== land.revision ? " (an earlier outline)" : ""}.{" "}
            {!linked && payload.boundaryRevision !== land.revision && (
              <button type="button" className="land-link" onClick={econ.pinCurrentBoundary}>
                Use the current outline
              </button>
            )}
          </p>
        </details>
      </fieldset>
      <p className="land-muted" aria-live="polite">
        {econ.problem ?? (econ.calculating ? "Working out payback…" : "")}
      </p>
      {econ.calcError && (
        <p className="land-alert" role="alert">
          The payback could not be worked out. {econ.calcError}
        </p>
      )}
      {econ.result && <SolarScenarioResult result={econ.result} />}
      {econ.conflict && (
        <div className="land-note" role="region" aria-label="Newer saved costs">
          <p>
            Someone saved these costs again (now revision {econ.conflict.revision}) while you were
            editing revision {econ.draft?.editing?.revision}.
          </p>
          <div className="land-row">
            <GlassButton size="sm" onClick={econ.keepAsNew}>
              Keep mine as new costs
            </GlassButton>
            <GlassButton size="sm" variant="ghost" onClick={econ.loadLatest}>
              Load the newer values
            </GlassButton>
          </div>
        </div>
      )}
      {econ.error && (
        <p className="land-alert" role="alert">
          {econ.error}
        </p>
      )}
      {econ.storageError && (
        <p className="land-alert" role="alert">
          {econ.storageError}
        </p>
      )}
      <div className="land-row">
        <GlassButton
          type="submit"
          size="sm"
          variant="primary"
          disabled={!econ.result || econ.saving || Boolean(econ.conflict) || !econ.canEdit}
        >
          {econ.saving ? "Saving…" : econ.draft?.editing ? "Save changes" : "Save costs"}
        </GlassButton>
        <GlassButton size="sm" variant="ghost" onClick={econ.cancel}>
          Cancel
        </GlassButton>
        <button
          type="button"
          className="land-link land-solar-push"
          onClick={() => downloadJson(econ.draft, "scenario-draft.json")}
        >
          Download draft
        </button>
      </div>
    </form>
  );
}

/** Cumulative cash over the years, one line across a zero rule; hover reads a year out. */
function CashChart({ result }: { result: ScenarioResult }) {
  const [hover, setHover] = useState<number | null>(null);
  const { currency, payback } = scenarioNumbers(result);
  const points = result.rows
    .map((row) => ({ year: Number(row.year), cash: row.cumulativeCashFlow }))
    .filter((row): row is { year: number; cash: number } => typeof row.cash === "number");
  if (points.length < 2) return null;
  const width = 288;
  const top = 8;
  const bottom = 72;
  const low = Math.min(0, ...points.map((point) => point.cash));
  const high = Math.max(1, ...points.map((point) => point.cash));
  const y = (value: number) => bottom - ((value - low) / (high - low || 1)) * (bottom - top);
  const x = (index: number) => 4 + (index / (points.length - 1)) * (width - 8);
  const first = points[0];
  const last = points.at(-1);
  const turn =
    typeof payback === "number" ? points.findIndex((point) => point.year === payback) : -1;
  const shown = hover === null ? null : points[hover];
  return (
    <figure className="land-solar-chart" onMouseLeave={() => setHover(null)}>
      <svg
        viewBox={`0 0 ${width} 84`}
        role="img"
        aria-label={`Cash over time: ${moneyLabel(first?.cash ?? 0, currency)} at the start, ${moneyLabel(last?.cash ?? 0, currency)} after ${last?.year ?? 0} years.`}
      >
        <line className="land-solar-chart__base" x1="0" x2={width} y1={y(0)} y2={y(0)} />
        <polyline
          className="land-solar-chart__line"
          points={points.map((point, index) => `${x(index)},${y(point.cash)}`).join(" ")}
        />
        {turn >= 0 && (
          <circle
            className="land-solar-chart__dot"
            cx={x(turn)}
            cy={y(points[turn]?.cash ?? 0)}
            r={4}
          />
        )}
        {shown && hover !== null && (
          <circle className="land-solar-chart__dot" cx={x(hover)} cy={y(shown.cash)} r={4} />
        )}
        {points.map((point, index) => (
          <rect
            key={point.year}
            className="land-solar-chart__hit"
            x={x(index) - width / points.length / 2}
            y={0}
            width={width / points.length}
            height={84}
            onMouseEnter={() => setHover(index)}
          >
            <title>
              Year {point.year}: {moneyLabel(point.cash, currency)}
            </title>
          </rect>
        ))}
      </svg>
      <figcaption className="land-muted" aria-hidden="true">
        {shown
          ? `Year ${shown.year} · ${moneyLabel(shown.cash, currency)} in total`
          : "Cash in total, year by year"}
      </figcaption>
    </figure>
  );
}

/** A solar scenario's result: when it pays back, what it is worth, and how sure that is. */
export function SolarScenarioResult({ result }: { result: ScenarioResult }) {
  const numbers = scenarioNumbers(result);
  const years = Math.max(0, result.rows.length - 1);
  const money = (value: number | null) =>
    value === null ? "—" : moneyLabel(value, numbers.currency);
  const columns = Object.keys(result.rows[0] ?? {});
  return (
    <section className="land-solar-result" aria-label="Payback">
      <p className="land-solar-payback">{paybackWords(numbers.payback, years)}</p>
      <dl className="land-solar-list">
        <div>
          <dt>Net present value</dt>
          <dd>{money(numbers.netPresentValue)}</dd>
        </div>
        <div>
          <dt>Your up-front cost</dt>
          <dd>{money(numbers.equity)}</dd>
        </div>
        <div>
          <dt>First-year energy</dt>
          <dd>{numbers.firstYearKwh === null ? "—" : energyLabel(numbers.firstYearKwh, 3)}</dd>
        </div>
        <div>
          <dt>Cash after {years} years</dt>
          <dd>{money(numbers.lifetime)}</dd>
        </div>
        {numbers.loanPayment !== null && numbers.loanPayment > 0 && (
          <div>
            <dt>Yearly loan payment</dt>
            <dd>{money(numbers.loanPayment)}</dd>
          </div>
        )}
      </dl>
      <CashChart result={result} />
      {(result.sensitivity?.length ?? 0) > 0 && (
        <div className="land-solar-table">
          <table>
            <caption>If things change: net present value</caption>
            <tbody>
              {result.sensitivity?.map((row) => (
                <tr key={String(row.case)}>
                  <th scope="row">{String(row.case)}</th>
                  <td>
                    {typeof row.netPresentValue === "number"
                      ? moneyLabel(row.netPresentValue, numbers.currency)
                      : (row.netPresentValue ?? "—")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {result.rows.length > 0 && (
        <details className="land-solar-fold">
          <summary>Year by year</summary>
          {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- Wide tables need keyboard scrolling. */}
          <div className="land-solar-table" role="region" aria-label="Year by year" tabIndex={0}>
            <table>
              <thead>
                <tr>
                  {columns.map((column) => (
                    <th key={column} scope="col">
                      {COLUMN_LABELS[column] ?? column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row, index) => (
                  <tr key={String(row.year ?? index)}>
                    {columns.map((column) => {
                      const value = row[column];
                      return (
                        <td key={column}>
                          {typeof value === "number"
                            ? value.toLocaleString("en-US", { maximumFractionDigits: 0 })
                            : (value ?? "—")}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      <details className="land-solar-fold">
        <summary>What this cannot tell you</summary>
        <ul className="land-solar-bullets">
          {result.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
        <p className="land-muted">Calculation {result.algorithm}</p>
      </details>
    </section>
  );
}

const COLUMN_LABELS: Record<string, string> = {
  year: "Year",
  generationKwh: "Energy (kWh)",
  energyValue: "Energy value",
  maintenanceCost: "Upkeep",
  debtService: "Loan payment",
  replacementCost: "Replacement",
  netCashFlow: "Net cash",
  cumulativeCashFlow: "Cash in total",
  discountedCashFlow: "Discounted cash",
};

function SavedScenario({
  scenario,
  econ,
  land,
}: {
  scenario: Scenario;
  econ: SolarEconomicsState;
  land: LandArea;
}) {
  const [history, setHistory] = useState<Scenario[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const numbers = scenarioNumbers(scenario.result);
  const years = Math.max(0, scenario.result.rows.length - 1);
  return (
    <li>
      <details className="land-solar-fold">
        <summary>
          {scenario.name} · {paybackWords(numbers.payback, years).toLowerCase()}
          {scenario.stale ? " · earlier outline" : ""}
        </summary>
        <p className="land-muted">
          Revision {scenario.revision} · {dateLabel(scenario.updatedAt)}
        </p>
        <p className="land-solar-note">{scenario.inputs.assumptions}</p>
        <SolarScenarioResult result={scenario.result} />
        <div className="land-row">
          {scenario.solarAssessmentId && (
            <GlassButton
              size="sm"
              variant="ghost"
              onClick={() =>
                useLandContext.getState().selectSolar(scenario.solarAssessmentId ?? null)
              }
            >
              Show its array
            </GlassButton>
          )}
          {econ.canEdit && (
            <GlassButton
              size="sm"
              variant="ghost"
              disabled={Boolean(econ.draft)}
              onClick={() => econ.edit(scenario)}
            >
              Edit assumptions
            </GlassButton>
          )}
          <GlassButton
            size="sm"
            variant="ghost"
            onClick={() => {
              setError(null);
              void unwrap(
                api.GET("/api/v1/land/{land_id}/scenarios/{scenario_id}/revisions", {
                  params: { path: { land_id: land.id, scenario_id: scenario.id } },
                }),
              )
                .then(setHistory)
                .catch((cause: unknown) => setError(landError(cause)));
            }}
          >
            Revision history
          </GlassButton>
          <GlassButton
            size="sm"
            variant="ghost"
            leadingIcon={<Download size={15} aria-hidden="true" />}
            onClick={() => downloadJson(JSON.stringify(scenario, null, 2), "land-scenario.json")}
          >
            Export assumptions and results
          </GlassButton>
        </div>
        {error && (
          <p className="land-alert" role="alert">
            {error}
          </p>
        )}
        {history && (
          <ol className="land-solar-history" aria-label={`Revisions of ${scenario.name}`}>
            {history.map((revision) => {
              const past = scenarioNumbers(revision.result);
              return (
                <li key={`${revision.id}:${revision.revision}`}>
                  <strong>
                    Revision {revision.revision} · {dateLabel(revision.updatedAt)}
                  </strong>
                  <span className="land-muted">
                    {" "}
                    {paybackWords(past.payback, Math.max(0, revision.result.rows.length - 1))} · net
                    present value{" "}
                    {past.netPresentValue === null
                      ? "—"
                      : moneyLabel(past.netPresentValue, past.currency)}{" "}
                    · land outline {revision.boundaryRevision}
                  </span>
                </li>
              );
            })}
          </ol>
        )}
      </details>
    </li>
  );
}

/** Every saved solar costing on this land, each opening to its numbers and its tools. */
export function SolarScenarios({
  scenarios,
  econ,
  land,
}: {
  scenarios: Scenario[];
  econ: SolarEconomicsState;
  land: LandArea;
}) {
  if (!scenarios.length) return null;
  return (
    <section className="land-solar-section" aria-label="Saved costs and payback">
      <h4 className="land-solar-heading">Costs and payback</h4>
      <ul className="land-solar-saved">
        {scenarios.map((scenario) => (
          <SavedScenario key={scenario.id} scenario={scenario} econ={econ} land={land} />
        ))}
      </ul>
    </section>
  );
}

import { useState } from "react";
import type { ScenarioResult } from "./scenarioDefaults";
import { fieldLabel, valueLabel } from "./scenarioDefaults";

export function ScenarioResultView({ result }: { result: ScenarioResult }) {
  const [rowsOpen, setRowsOpen] = useState(false);
  const values = result.rows
    .map((row) => row.cumulativeCashFlow)
    .filter((value): value is number => typeof value === "number");
  const minimum = Math.min(0, ...values),
    maximum = Math.max(1, ...values);
  const y = (value: number) => 125 - ((value - minimum) / (maximum - minimum)) * 105;
  return (
    <section className="land-scenario-result" aria-label="Scenario result">
      <dl className="land-scenario-summary">
        {Object.entries(result.summary).map(([key, value]) => (
          <div key={key}>
            <dt>{fieldLabel(key === "paybackYear" ? "equityRecoveryYear" : key)}</dt>
            <dd>{valueLabel(value)}</dd>
          </div>
        ))}
      </dl>
      {values.length > 1 && (
        <figure>
          <svg
            viewBox="0 0 400 155"
            role="img"
            aria-label="Cumulative equity cash flow across the modeled years"
          >
            <line x1="20" y1={y(0)} x2="380" y2={y(0)} stroke="currentColor" opacity="0.4" />
            <polyline
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              points={values
                .map((value, index) => `${20 + (index / (values.length - 1)) * 360},${y(value)}`)
                .join(" ")}
            />
            <text x="20" y="150" fill="currentColor" fontSize="10">
              Year 0
            </text>
            <text x="345" y="150" fill="currentColor" fontSize="10">
              Year {values.length - 1}
            </text>
          </svg>
          <figcaption>
            Cumulative equity cash flow · {String(result.summary.currency ?? "")}
          </figcaption>
        </figure>
      )}
      {result.algorithm.startsWith("restoration") && (
        <figure className="land-cover-chart" aria-label="Current and target cover">
          <figcaption>Current cover → proposed target</figcaption>
          {result.rows.map((row) => (
            <div key={String(row.class)}>
              <strong>{String(row.class)}</strong>
              <div>
                <span>Current {valueLabel(row.baselinePercent)}%</span>
                <i style={{ width: `${Number(row.baselinePercent)}%` }} />
              </div>
              <div>
                <span>Target {valueLabel(row.targetPercent)}%</span>
                <i
                  className="land-cover-target"
                  style={{ width: `${Number(row.targetPercent)}%` }}
                />
              </div>
            </div>
          ))}
        </figure>
      )}
      {(result.sensitivity?.length ?? 0) > 0 && (
        <div className="land-table-scroll">
          <table>
            <caption>
              {result.algorithm.startsWith("solar")
                ? "Sensitivity: net present value"
                : "Cost schedule including contingency"}
            </caption>
            <tbody>
              {result.sensitivity?.map((row) => (
                <tr key={String(row.case)}>
                  <th scope="row">{row.case}</th>
                  <td>{valueLabel(row.netPresentValue ?? row.cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <button type="button" onClick={() => setRowsOpen(!rowsOpen)} aria-expanded={rowsOpen}>
        {rowsOpen ? "Hide" : "Inspect"} calculation rows
      </button>
      {rowsOpen && result.rows.length > 0 && (
        <div className="land-table-scroll">
          <table>
            <thead>
              <tr>
                {Object.keys(result.rows[0] ?? {}).map((key) => (
                  <th key={key}>{fieldLabel(key)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row, index) => (
                <tr key={String(row.year ?? row.class ?? index)}>
                  {Object.values(row).map((value, column) => (
                    <td key={`${column}:${String(value)}`}>{valueLabel(value)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <details>
        <summary>Assumptions and limits of this calculation</summary>
        <ul>
          {result.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
        <small>Calculation version: {result.algorithm}</small>
      </details>
    </section>
  );
}

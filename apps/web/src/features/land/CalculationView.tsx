/* eslint-disable jsx-a11y/no-noninteractive-tabindex -- Overflow tables need keyboard scrolling. */
import { useState } from "react";
import type { LandEvidence, ResearchArtifact } from "@twin/contracts";

import { ChartView } from "./ResearchChart";

type Calculation = Extract<ResearchArtifact["output"], { kind: "calculation" }>;
const PAGE_SIZE = 25;
const number = (value: number | null | undefined) => (value == null ? "No result" : String(value));

export function CalculationView({
  artifact,
  output,
  evidence = [],
  onEvidence,
}: {
  artifact: ResearchArtifact;
  output: Calculation;
  evidence?: LandEvidence[];
  onEvidence?: (id: string) => void;
}) {
  const [offset, setOffset] = useState(0);
  const [metric, setMetric] = useState("");
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const request = output.request;
  const start = Math.min(offset, Math.max(0, output.rows.length - 1));
  const end = Math.min(start + PAGE_SIZE, output.rows.length);
  const formula = request.formulas.find((item) => item.name === metric);
  const issues = (output.issues ?? []).filter((issue) => issue.row >= start && issue.row < end);
  const download = () => {
    try {
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(artifact, null, 2)], { type: "application/json" }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = "land-calculation.json";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setDownloadError(null);
    } catch {
      setDownloadError("The calculation could not be downloaded. Please try again.");
    }
  };
  return (
    <div className="land-calculation">
      <p>{request.purpose}</p>
      <p className="land-footnote">
        Calculated from supplied inputs. Assumptions are not measurements.
      </p>
      <p>{request.limitations}</p>
      <div className="land-actions">
        <button type="button" onClick={download}>
          Download calculation
        </button>
      </div>
      {downloadError && <p role="alert">{downloadError}</p>}
      <p aria-live="polite">
        Rows {start + 1}–{end} of {output.rows.length}
      </p>
      <div className="land-table-wrap" tabIndex={0} role="region" aria-label="Calculated results">
        <table>
          <thead>
            <tr>
              <th scope="col">Case</th>
              {request.formulas.map((item) => (
                <th key={item.name} scope="col">
                  {item.label} ({item.unit})
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {output.rows.slice(start, end).map((row, index) => (
              <tr key={start + index}>
                <th scope="row">{request.rowLabels[start + index]}</th>
                {request.formulas.map((item) => (
                  <td key={item.name}>{number(row[item.name])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {output.rows.length > PAGE_SIZE && (
        <div className="land-actions" aria-label="Calculation pages">
          <button
            type="button"
            disabled={start === 0}
            onClick={() => setOffset(Math.max(0, start - PAGE_SIZE))}
          >
            Previous results
          </button>
          <button
            type="button"
            disabled={end === output.rows.length}
            onClick={() => setOffset(end)}
          >
            Next results
          </button>
        </div>
      )}
      {!!output.issues?.length && (
        <p role="status">
          {output.issues.length} result{output.issues.length === 1 ? "" : "s"} could not be
          calculated. Missing or undefined values remain gaps.
        </p>
      )}
      {issues.length > 0 && (
        <details>
          <summary>Explain gaps on this page ({issues.length})</summary>
          <ul>
            {issues.map((issue) => (
              <li key={`${issue.row}/${issue.column}`}>
                {request.rowLabels[issue.row]} ·{" "}
                {request.formulas.find((item) => item.name === issue.column)?.label ?? issue.column}
                : {issue.message}
              </li>
            ))}
          </ul>
        </details>
      )}
      {output.rows.length > 1 && (
        <label>
          Plot a result
          <select value={metric} onChange={(event) => setMetric(event.target.value)}>
            <option value="">Choose a result</option>
            {request.formulas.map((item) => (
              <option key={item.name} value={item.name}>
                {item.label} ({item.unit})
              </option>
            ))}
          </select>
        </label>
      )}
      {formula && (
        <ChartView
          output={{
            kind: "chart",
            chartType: "bar",
            xLabel: "Case",
            yLabel: formula.label,
            unit: formula.unit,
            labels: request.rowLabels.slice(start, end),
            series: [
              {
                label: formula.label,
                values: output.rows.slice(start, end).map((row) => row[formula.name] ?? null),
              },
            ],
          }}
        />
      )}
      <details>
        <summary>Inputs, assumptions and formulas</summary>
        <p className="land-footnote">
          Inputs and chart values follow the selected results page. Shared inputs apply to every
          row. Units are labels; conversions must be explicit in formulas.
        </p>
        {request.inputs.map((item) => (
          <section key={item.name}>
            <h5>
              {item.label} ({item.unit})
            </h5>
            <p>
              <strong>
                {item.origin === "evidence"
                  ? "Cited input"
                  : item.origin === "question"
                    ? "From the question"
                    : "Assumption"}
              </strong>{" "}
              · {item.basis}
            </p>
            {(item.evidenceIds ?? []).map((id, index) =>
              onEvidence ? (
                <button key={id} type="button" onClick={() => onEvidence(id)}>
                  Inspect{" "}
                  {evidence.find((source) => source.id === id)?.title ??
                    `input source ${index + 1}`}
                </button>
              ) : (
                <p key={id}>Source: {evidence.find((source) => source.id === id)?.title ?? id}</p>
              ),
            )}
            {item.values.length === 1 ? (
              <p>Shared value: {number(item.values[0])}</p>
            ) : (
              <div
                className="land-table-wrap"
                tabIndex={0}
                role="region"
                aria-label={`${item.label} inputs`}
              >
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Case</th>
                      <th scope="col">Value ({item.unit})</th>
                    </tr>
                  </thead>
                  <tbody>
                    {item.values.slice(start, end).map((value, index) => (
                      <tr key={start + index}>
                        <th scope="row">{request.rowLabels[start + index]}</th>
                        <td>{number(value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <p className="land-footnote">
              Formula name: <code>{item.name}</code>
            </p>
          </section>
        ))}
        <ol>
          {request.formulas.map((item) => (
            <li key={item.name}>
              <strong>
                {item.label} ({item.unit})
              </strong>
              <p>
                <code>
                  {item.name} = {item.expression}
                </code>
              </p>
            </li>
          ))}
        </ol>
        <p className="land-footnote">
          Engine: {output.engineVersion}. The JSON download includes the recipe hash, full-precision
          values and all rows.
        </p>
      </details>
    </div>
  );
}

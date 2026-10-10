import type { components } from "@twin/contracts";

export function SurveySummaryView({
  summary,
}: {
  summary: components["schemas"]["SurveySummary"];
}) {
  return (
    <section className="land-survey-summary" aria-label="Survey results">
      <h4>Observed species coverage</h4>
      <p>
        {summary.sampledAreaM2.toLocaleString(undefined, { maximumFractionDigits: 1 })} m² in{" "}
        {summary.plotAreasM2.length} {summary.plotAreasM2.length === 1 ? "plot" : "plots"} ·{" "}
        {(summary.sampledFraction * 100).toFixed(2)}% of the land boundary.
      </p>
      <p className="land-footnote">
        Percentages describe sampled plots. Species and vegetation layers can overlap; totals may
        exceed 100%.
      </p>
      <ul className="land-species-list">
        {summary.species.map((s) => (
          <li key={`${s.taxon}:${s.stratum}`}>
            <div>
              <strong>{s.taxon}</strong> · {s.stratum}
            </div>
            <div
              className="land-species-bar"
              role="img"
              aria-label={`${s.meanPercent.toFixed(2)} percent area-weighted cover`}
            >
              <span style={{ width: `${s.meanPercent}%` }} />
            </div>
            <p>
              <strong>{s.meanPercent.toFixed(2)}%</strong> area-weighted cover · plot range{" "}
              {s.minimumPercent.toFixed(2)}–{s.maximumPercent.toFixed(2)}%
            </p>
            <p className="land-footnote">
              Assessed in {s.measuredPlots} {s.measuredPlots === 1 ? "plot" : "plots"} (
              {(s.assessedSampleFraction * 100).toFixed(1)}% of sampled area). Identification:{" "}
              {s.identifications.join(", ")}.
            </p>
          </li>
        ))}
      </ul>
      {!summary.species.length && (
        <p>No species observations recorded. This is not proof of absence.</p>
      )}
      <details>
        <summary>How to interpret this survey</summary>
        <ul>
          {summary.limitations.map((v) => (
            <li key={v}>{v}</li>
          ))}
        </ul>
      </details>
    </section>
  );
}

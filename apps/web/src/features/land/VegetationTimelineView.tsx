import type { components } from "@twin/contracts";

type Timeline = components["schemas"]["VegetationTimeline"];
const number = (value: number | null | undefined) => (value == null ? "No data" : value.toFixed(3));
export function VegetationTimelineView({
  series,
  onChoose,
  band,
}: {
  series: Timeline;
  onChoose: (band: number) => void;
  band: number;
}) {
  const dated = series.observations.map((o) => ({
    ...o,
    t: Date.parse(o.acquiredAt ?? o.period.startDate),
  }));
  const first = Math.min(...dated.map((o) => o.t)),
    last = Math.max(...dated.map((o) => o.t));
  const x = (t: number) => 35 + ((t - first) / Math.max(1, last - first)) * 285;
  const y = (n: number) => 15 + ((1 - n) / 2) * 120;
  return (
    <section className="land-vegetation-timeline" aria-label="Vegetation observations">
      <h5>Compare the same land cells</h5>
      <p className="land-footnote">
        {series.commonCells.toLocaleString()} clear cells shared by every observation ·{" "}
        {((series.commonCoverageFraction ?? 0) * 100).toFixed(1)}% of the boundary grid. The trend
        below uses only these cells.
      </p>
      {series.commonCells > 0 ? (
        <svg
          viewBox="0 0 350 170"
          role="img"
          aria-label="NDVI on common clear land cells, from minus one to one. Exact values and acquisition dates follow."
        >
          {[-1, 0, 1].map((n) => (
            <g key={n}>
              <line x1={35} x2={320} y1={y(n)} y2={y(n)} stroke="currentColor" opacity={0.2} />
              <text x={7} y={y(n) + 4} fill="currentColor" fontSize={11}>
                {n}
              </text>
            </g>
          ))}
          {dated.map((o, i) => {
            const previous = dated[i - 1];
            return (
              <g key={o.band}>
                {previous?.commonMeanNdvi != null && o.commonMeanNdvi != null && (
                  <line
                    x1={x(previous.t)}
                    y1={y(previous.commonMeanNdvi)}
                    x2={x(o.t)}
                    y2={y(o.commonMeanNdvi)}
                    stroke="#91d6a2"
                    strokeWidth={2}
                  />
                )}
                {o.commonMeanNdvi != null && (
                  <circle cx={x(o.t)} cy={y(o.commonMeanNdvi)} r={4} fill="#91d6a2" />
                )}
              </g>
            );
          })}
          <text x={35} y={160} fill="currentColor" fontSize={10}>
            {dated[0]?.period.startDate.slice(0, 7)}
          </text>
          <text x={320} y={160} textAnchor="end" fill="currentColor" fontSize={10}>
            {dated.at(-1)?.period.startDate.slice(0, 7)}
          </text>
        </svg>
      ) : (
        <p className="land-notice">
          No comparable trend: these observations share no clear land cells. Try different months or
          inspect each observation separately.
        </p>
      )}
      <ol className="land-vegetation-observations">
        {series.observations.map((o) => (
          <li key={o.band}>
            <div className="land-actions">
              <button type="button" aria-pressed={band === o.band} onClick={() => onChoose(o.band)}>
                {o.period.startDate.slice(0, 7)} observation
              </button>
            </div>
            <p>
              {o.acquiredAt
                ? `Acquired ${new Date(o.acquiredAt).toISOString().slice(0, 10)}`
                : "No acquisition selected"}{" "}
              · {((o.coverageFraction ?? 0) * 100).toFixed(1)}% clear usable coverage
            </p>
            <dl>
              <div>
                <dt>NDVI on common cells</dt>
                <dd>{number(o.commonMeanNdvi)}</dd>
              </div>
              <div>
                <dt>NDVI on all usable cells this date</dt>
                <dd>{number(o.meanNdvi)}</dd>
              </div>
            </dl>
            <details>
              <summary>Acquisition and quality</summary>
              <p>{o.explanation}</p>
              <p>
                Window: {o.period.startDate} through {o.period.endDate}.{" "}
                {o.qualityCandidatesExamined} quality masks examined from {o.candidateCount}{" "}
                returned catalog scenes{o.catalogTruncated ? " (catalog truncated)" : ""}.
              </p>
              {o.sceneId && <p>Scene: {o.sceneId}</p>}
              <ul>
                {Object.entries(o.qualityCounts)
                  .filter(([, count]) => count > 0)
                  .map(([label, count]) => (
                    <li key={label}>
                      {label}: {count.toLocaleString()} cells
                    </li>
                  ))}
              </ul>
            </details>
          </li>
        ))}
      </ol>
      {series.changeBand != null && (
        <div className="land-vegetation-change">
          <p>
            Last minus first: <strong>{number(series.meanChange)} NDVI</strong> on{" "}
            {series.changeCells.toLocaleString()} cells valid at both endpoints.
          </p>
          <div className="land-actions">
            <button
              type="button"
              disabled={!series.changeCells}
              aria-pressed={band === series.changeBand}
              onClick={() => {
                if (series.changeBand != null) onChoose(series.changeBand);
              }}
            >
              View vegetation change
            </button>
          </div>
        </div>
      )}
      <p className="land-footnote">
        These are sampled vegetation signals. Season, weather and measurement differences can affect
        them; they do not establish species, causes of change or restoration success.
      </p>
    </section>
  );
}

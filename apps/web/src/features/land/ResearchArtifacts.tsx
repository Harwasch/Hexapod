/* eslint-disable jsx-a11y/no-noninteractive-tabindex -- Overflow tables need keyboard scrolling. */
import { useLandContext } from "@/state/landContext";
import { useScene } from "@/cesium/SceneContext";
import { boundsOf } from "@twin/geo";
import type { ResearchArtifact } from "@twin/contracts";

import { LandRasterView } from "./LandRasterView";

type Chart = Extract<ResearchArtifact["output"], { kind: "chart" }>;

function ChartView({ output }: { output: Chart }) {
  const values = output.series.flatMap((series) =>
    series.values.filter((value): value is number => value !== null && Number.isFinite(value)),
  );
  const min = Math.min(0, ...values),
    max = Math.max(0, ...values),
    range = max - min || 1;
  const x = (i: number) => 40 + i * (520 / Math.max(1, output.labels.length - 1));
  const y = (value: number) => 170 - ((value - min) / range) * 145;
  return (
    <>
      <svg
        viewBox="0 0 600 215"
        role="img"
        aria-label={`${output.yLabel}, ${output.unit}, by ${output.xLabel}. Values are in the table below.`}
        className="land-chart"
      >
        <line x1="35" x2="565" y1={y(0)} y2={y(0)} stroke="currentColor" opacity="0.25" />
        <text x="5" y="18" fontSize="10" fill="currentColor">
          {max.toLocaleString(undefined, { maximumFractionDigits: 1 })}
        </text>
        {output.series.map((series, seriesIndex) => (
          <g
            key={series.label}
            className={`land-chart-series land-chart-series-${seriesIndex % 3}`}
          >
            {output.chartType === "line" &&
              series.values.map((value, i) => {
                const previous = series.values[i - 1];
                return value !== null && previous !== null && previous !== undefined ? (
                  <line
                    key={i}
                    x1={x(i - 1)}
                    x2={x(i)}
                    y1={y(previous)}
                    y2={y(value)}
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                ) : null;
              })}
            {series.values.map((value, i) =>
              value === null ? null : output.chartType === "bar" ? (
                <rect
                  key={i}
                  x={x(i) - 10 + seriesIndex * (20 / output.series.length)}
                  y={Math.min(y(0), y(value))}
                  width={Math.max(1, 20 / output.series.length - 1)}
                  height={Math.max(1, Math.abs(y(value) - y(0)))}
                  fill="currentColor"
                >
                  <title>
                    {series.label}: {output.labels[i]} · {value} {output.unit}
                  </title>
                </rect>
              ) : (
                <circle key={i} cx={x(i)} cy={y(value)} r="3" fill="currentColor">
                  <title>
                    {series.label}: {output.labels[i]} · {value} {output.unit}
                  </title>
                </circle>
              ),
            )}
          </g>
        ))}
        {output.labels.map((label, i) =>
          i % Math.max(1, Math.ceil(output.labels.length / 12)) === 0 ? (
            <text key={i} x={x(i)} y="192" fontSize="9" textAnchor="middle" fill="currentColor">
              {label.slice(0, 12)}
            </text>
          ) : null,
        )}
      </svg>
      <p className="land-footnote">
        {output.yLabel} · {output.unit}
      </p>
      <details>
        <summary>View chart values</summary>
        <div className="land-table-wrap" tabIndex={0} role="region" aria-label="Chart values">
          <table>
            <thead>
              <tr>
                <th scope="col">{output.xLabel}</th>
                {output.series.map((series) => (
                  <th key={series.label} scope="col">
                    {series.label} ({output.unit})
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {output.labels.map((label, i) => (
                <tr key={i}>
                  <th scope="row">{label}</th>
                  {output.series.map((series) => (
                    <td key={series.label}>{series.values[i] ?? "No data"}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

export function ResearchArtifactView({ artifact }: { artifact: ResearchArtifact }) {
  const output = artifact.output;
  const scene = useScene();
  const shown = useLandContext((state) => Boolean(state.layers[artifact.id]));
  const toggleMap = () => {
    if (output.kind !== "map") return;
    if (shown) {
      useLandContext.getState().removeLayer(artifact.id);
      return;
    }
    useLandContext.getState().setLayer({
      id: artifact.id,
      title: artifact.title,
      features: output.features.map((feature, index) => ({ id: String(index), ...feature })),
    });
    const points = output.features.flatMap(({ geometry }) => {
      if (geometry.type === "Point") return [geometry.coordinates];
      if (geometry.type === "LineString") return geometry.coordinates;
      const b = boundsOf(geometry);
      return [
        [b.west, b.south],
        [b.east, b.north],
      ];
    });
    if (points.length) {
      const xs = points.map((point) => point[0] ?? 0),
        ys = points.map((point) => point[1] ?? 0);
      const west = Math.min(...xs),
        east = Math.max(...xs),
        south = Math.min(...ys),
        north = Math.max(...ys);
      const dx = Math.max(0.0005, (east - west) * 0.1),
        dy = Math.max(0.0005, (north - south) * 0.1);
      scene?.camera.flyToRectangle(
        Math.max(-180, west - dx),
        Math.max(-90, south - dy),
        Math.min(180, east + dx),
        Math.min(90, north + dy),
      );
    }
  };
  return (
    <article className="land-artifact">
      <h4>{artifact.title}</h4>
      {output.kind === "raster" && <LandRasterView id={output.rasterId} />}
      {output.kind === "chart" && <ChartView output={output} />}
      {output.kind === "table" && (
        <div className="land-table-wrap" tabIndex={0} role="region" aria-label={artifact.title}>
          <table>
            <thead>
              <tr>
                {output.columns.map((column) => (
                  <th key={column.key} scope="col">
                    {column.label}
                    {column.unit ? ` (${column.unit})` : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {output.rows.map((row, i) => (
                <tr key={i}>
                  {output.columns.map((column) => (
                    <td key={column.key}>{String(row[column.key] ?? "No data")}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {output.kind === "document" && <div className="land-document">{output.markdown}</div>}
      {output.kind === "timeline" && (
        <ol className="land-timeline">
          {output.entries.map((entry, i) => (
            <li key={i}>
              <time>{entry.date}</time>
              <strong>{entry.title}</strong>
              <p>{entry.description}</p>
            </li>
          ))}
        </ol>
      )}
      {output.kind === "map" && (
        <p>
          {output.features.length} mapped features. {output.legend}
        </p>
      )}
      {output.kind === "map" && (
        <div className="land-actions">
          <button type="button" aria-pressed={shown} disabled={!scene} onClick={toggleMap}>
            {shown ? "Hide map layer" : "Show map layer"}
          </button>
        </div>
      )}
      <details>
        <summary>Method and limitations</summary>
        <p>{artifact.method}</p>
      </details>
    </article>
  );
}

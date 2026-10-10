import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { boundsOf } from "@twin/geo";
import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { useLandContext } from "@/state/landContext";
import { useLand } from "@/state/land";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";
import { describeError } from "@/lib/log";
import type { SolarAssessment } from "./solarStudy";
import { valueLabel as number } from "./scenarioDefaults";
import "./solar.css";
const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export function SolarAssessmentView({
  id,
  onUse,
}: {
  id: string;
  onUse?: (a: SolarAssessment) => void;
}) {
  const scope = useLandScope(),
    ready = useLandAccessReady(),
    scene = useScene();
  const revision = useLand((s) => s.active?.revision);
  const shown = useLandContext((s) => !!s.layers[`solar:${id}`]);
  const [error, setError] = useState<string | null>(null),
    [busy, setBusy] = useState(false);
  const query = useQuery({
    queryKey: ["land-solar-assessment", scope, id, revision],
    enabled: ready,
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/solar-assessments/{assessment_id}", {
          params: { path: { assessment_id: id } },
        }),
      ),
  });
  const a = query.data;
  async function download() {
    setBusy(true);
    setError(null);
    try {
      const blob = await unwrap(
        api.GET("/api/v1/land/solar-assessments/{assessment_id}/download", {
          params: { path: { assessment_id: id } },
          parseAs: "blob",
        }),
      );
      const url = URL.createObjectURL(blob),
        anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `solar-${id}.zip`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  }
  if (!a)
    return (
      <p role={query.isError ? "alert" : "status"}>
        {query.isError ? "Solar assessment could not be loaded." : "Loading solar assessment…"}{" "}
        {query.isError && (
          <button type="button" onClick={() => void query.refetch()}>
            Retry
          </button>
        )}
      </p>
    );
  const m = a.metadata,
    maximum = Math.max(
      1,
      ...m.monthly.map((v) => v.unshadedGenerationKwh ?? 0),
      ...m.monthly.map((v) => v.generationKwh ?? 0),
    );
  return (
    <section className="land-solar-result" aria-label="Hourly solar assessment">
      <h4>{a.request.year} · modeled solar generation</h4>
      {a.stale && (
        <p className="land-notice">Uses earlier boundary revision {a.boundaryRevision}.</p>
      )}
      <dl className="land-scenario-summary">
        <div>
          <dt>{m.completeYear ? "Annual AC generation" : "Generation in valid hours"}</dt>
          <dd>{number(m.modeledGenerationKwh)} kWh</dd>
        </div>
        <div>
          <dt>DC capacity / AC inverter</dt>
          <dd>
            {number(m.capacityKwDc)} / {number(m.inverterKwAc)} kW
          </dd>
        </div>
        <div>
          <dt>Weather coverage</dt>
          <dd>
            {m.validHours.toLocaleString()} / {m.expectedHours.toLocaleString()} hours
          </dd>
        </div>
        <div>
          <dt>Open-horizon comparison</dt>
          <dd>{number(m.unshadedGenerationKwh)} kWh</dd>
        </div>
      </dl>
      {!m.completeYear && (
        <p className="land-notice">
          Incomplete weather coverage. This sum cannot be used as annual generation.
        </p>
      )}
      <p className="land-footnote">
        One historical weather year and entered array assumptions. Open-horizon comparison removes
        the entered horizon and additional shade, with the same equipment and weather.
      </p>
      <figure>
        <svg
          className="land-chart"
          viewBox="0 0 480 160"
          role="img"
          aria-label="Monthly modeled solar generation and open-horizon comparison; exact values in the table below"
        >
          {m.monthly.map((v, i) => (
            <g key={v.month}>
              {v.unshadedGenerationKwh != null && (
                <rect
                  x={16 + i * 38}
                  y={130 - (v.unshadedGenerationKwh / maximum) * 110}
                  width="12"
                  height={(v.unshadedGenerationKwh / maximum) * 110}
                  fill="currentColor"
                  opacity=".25"
                />
              )}
              {v.generationKwh != null && (
                <rect
                  x={28 + i * 38}
                  y={130 - (v.generationKwh / maximum) * 110}
                  width="12"
                  height={(v.generationKwh / maximum) * 110}
                  fill="currentColor"
                >
                  <title>
                    {months[i]}: {number(v.generationKwh)} kWh
                  </title>
                </rect>
              )}
              <text x={27 + i * 38} y="150" textAnchor="middle" fill="currentColor" fontSize="10">
                {months[i]}
              </text>
            </g>
          ))}
        </svg>
        <figcaption>
          Monthly AC energy (kWh) · solid: modeled shading · faint: open horizon
        </figcaption>
      </figure>
      <details>
        <summary>Monthly values and coverage</summary>
        <div className="land-table-scroll">
          <table>
            <thead>
              <tr>
                <th>Month</th>
                <th>AC kWh</th>
                <th>Open horizon kWh</th>
                <th>Valid hours</th>
              </tr>
            </thead>
            <tbody>
              {m.monthly.map((v, i) => (
                <tr key={v.month}>
                  <th>{months[i]}</th>
                  <td>{v.generationKwh == null ? "No data" : number(v.generationKwh)}</td>
                  <td>
                    {v.unshadedGenerationKwh == null ? "No data" : number(v.unshadedGenerationKwh)}
                  </td>
                  <td>
                    {v.validHours}/{v.expectedHours}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <div className="land-actions">
        <button
          type="button"
          disabled={!scene}
          onClick={() => {
            if (shown) useLandContext.getState().removeLayer(`solar:${id}`);
            else {
              useLandContext
                .getState()
                .setLayer({
                  id: `solar:${id}`,
                  title: "Solar array zone",
                  features: [
                    { id, label: `Solar ${a.request.year}`, geometry: a.request.arrayZone },
                  ],
                });
              const b = boundsOf(a.request.arrayZone);
              scene?.camera.flyToRectangle(b.west, b.south, b.east, b.north);
            }
          }}
        >
          {shown ? "Hide array zone" : "Show array zone"}
        </button>
        <button type="button" disabled={busy} onClick={() => void download()}>
          Download hourly data and sources
        </button>
        {onUse && (
          <button type="button" disabled={!m.completeYear || a.stale} onClick={() => onUse(a)}>
            Use in financial scenario
          </button>
        )}
      </div>
      <details>
        <summary>Assumptions, source and limitations</summary>
        <p>{a.request.zoneBasis}</p>
        <p>{a.request.horizonBasis}</p>
        <p>{a.request.assumptions}</p>
        <dl>
          <dt>Tilt / azimuth</dt>
          <dd>
            {a.request.tiltDegrees}° / {a.request.azimuthDegrees}°
          </dd>
          <dt>Module face area</dt>
          <dd>{number(a.request.moduleAreaM2)} m²</dd>
          <dt>Visible isotropic sky fraction</dt>
          <dd>{number(m.horizonSkyFraction * 100)}%</dd>
        </dl>
        <ul>
          {m.limitations.map((v) => (
            <li key={v}>{v}</li>
          ))}
        </ul>
        <p>
          <a href={m.sourceUrl} target="_blank" rel="noreferrer">
            NASA POWER weather source
          </a>{" "}
          ·{" "}
          <a href={m.documentationUrl} target="_blank" rel="noreferrer">
            Source documentation
          </a>
        </p>
        <p>
          {m.attribution} · {m.license}
        </p>
        <p>
          {m.algorithm} · {m.modelVersion}
        </p>
        <p className="land-solar-hash">Archive SHA-256: {a.sha256}</p>
      </details>
      {error && (
        <p className="land-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

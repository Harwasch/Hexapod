import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { components } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import { EvidenceView } from "./LandResearch";

type Result = components["schemas"]["RestorationEcologyResult"];
const percent = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 2 });
const relations = {
  "not-modeled": "No response curve calculated",
  "inside-target": "The entire assumed envelope falls within the target range",
  "overlaps-target": "The assumed envelope partly overlaps the target range",
  "outside-target": "The assumed envelope falls outside the target range",
};
export function RestorationTargetResults({ result }: { result: Result }) {
  const scope = useLandScope(),
    ready = useLandAccessReady();
  const [inspected, setInspected] = useState<string | null>(null);
  const evidence = useQuery({
    queryKey: ["land-source", scope, inspected],
    enabled: ready && !!inspected,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}", {
          params: { path: { evidence_id: inspected ?? "" } },
        }),
      ),
    retry: false,
  });
  const sources = (ids: string[]) =>
    !!ids.length && (
      <div className="land-actions">
        {ids.map((id, i) => (
          <button type="button" key={id} onClick={() => setInspected(id)}>
            Inspect source {i + 1}
          </button>
        ))}
      </div>
    );
  return (
    <section className="land-restoration-results" aria-label="Species restoration targets">
      <h4>Species goals and monitoring</h4>
      <p>{result.referenceBasis}</p>
      {sources(result.referenceEvidenceIds)}
      <p>
        <strong>Site constraints:</strong> {result.siteConstraints}
      </p>
      {result.species.map((species) => {
        const target = species.target,
          rows = species.projection ?? [];
        const x = (year: number) => 35 + (year / target.targetYear) * 335;
        const y = (value: number) => 140 - value * 1.2;
        return (
          <article key={`${target.taxon}:${target.stratum}`} className="land-restoration-species">
            <h4>
              {target.taxon} · {target.stratum}
            </h4>
            <p>
              <strong>Baseline:</strong>{" "}
              {species.baselinePercent == null ? "Unknown" : `${percent(species.baselinePercent)}%`}{" "}
              ·{" "}
              {species.baselineScope === "surveyed-plots"
                ? "sampled plots"
                : species.baselineScope === "entered-assumption"
                  ? "entered assumption"
                  : "not measured"}
            </p>
            {species.observedOn && (
              <p className="land-footnote">
                Observed {species.observedOn} · assessed plots cover{" "}
                {percent((species.assessedLandFraction ?? 0) * 100)}% of the land · identification:{" "}
                {species.identificationStatus?.length
                  ? species.identificationStatus.join(", ")
                  : "unreported"}
                .
              </p>
            )}
            <p>
              <strong>Goal for year {target.targetYear}:</strong> {percent(target.targetLow)}–
              {percent(target.targetHigh)}% ·{" "}
              {target.targetBasis === "reference-evidence"
                ? "linked reference evidence"
                : "hypothetical target"}
            </p>
            <p>{target.rationale}</p>
            {sources(target.evidenceIds ?? [])}
            {!!target.treatmentNames?.length && (
              <p>Planned treatments: {target.treatmentNames.join(", ")}.</p>
            )}
            {rows.length > 1 && (
              <figure>
                <svg
                  viewBox="0 0 400 170"
                  role="img"
                  aria-label={`${target.taxon} conditional cover envelope and target range; numerical values are available below`}
                >
                  <rect
                    x={35}
                    y={y(target.targetHigh)}
                    width={335}
                    height={Math.max(1, (target.targetHigh - target.targetLow) * 1.2)}
                    fill="currentColor"
                    opacity="0.09"
                  />
                  <line
                    x1={35}
                    x2={370}
                    y1={y(target.targetLow)}
                    y2={y(target.targetLow)}
                    stroke="currentColor"
                    strokeDasharray="4 4"
                    opacity="0.6"
                  />
                  <line
                    x1={35}
                    x2={370}
                    y1={y(target.targetHigh)}
                    y2={y(target.targetHigh)}
                    stroke="currentColor"
                    strokeDasharray="4 4"
                    opacity="0.6"
                  />
                  <polygon
                    points={[
                      ...rows.map((row) => `${x(row.year)},${y(row.high)}`),
                      ...[...rows].reverse().map((row) => `${x(row.year)},${y(row.low)}`),
                    ].join(" ")}
                    fill="currentColor"
                    opacity="0.2"
                  />
                  {(["low", "high"] as const).map((key) => (
                    <polyline
                      key={key}
                      points={rows.map((row) => `${x(row.year)},${y(row[key])}`).join(" ")}
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={2}
                    />
                  ))}
                  <text x={2} y={22} fontSize={10} fill="currentColor">
                    100%
                  </text>
                  <text x={12} y={140} fontSize={10} fill="currentColor">
                    0%
                  </text>
                  <text x={35} y={162} fontSize={10} fill="currentColor">
                    Year 0
                  </text>
                  <text x={333} y={162} fontSize={10} fill="currentColor">
                    Year {target.targetYear}
                  </text>
                </svg>
                <figcaption>
                  Solid envelope: conditional assumptions. Dashed bounds: target range. This is not
                  a measured trend or probability interval.
                </figcaption>
              </figure>
            )}
            <p>{relations[species.targetRelation]}.</p>
            {target.response && (
              <details>
                <summary>Response method and entered assumptions</summary>
                <p>
                  After year {target.response.startYear}, cover = baseline + (asymptotic cover −
                  baseline) × (1 − exp(−rate × elapsed years)). Before then, the assumed baseline is
                  held constant. The envelope uses all corners of the entered parameter ranges.
                </p>
                <p>
                  Asymptote {percent(target.response.asymptoteLow)}–
                  {percent(target.response.asymptoteHigh)}%; annual rate{" "}
                  {target.response.annualRateLow}–{target.response.annualRateHigh}.
                </p>
                <p>{target.response.basis}</p>
                {sources(target.response.evidenceIds ?? [])}
                <div className="land-table-scroll">
                  <table>
                    <caption>Conditional cover envelope (%)</caption>
                    <thead>
                      <tr>
                        <th>Year</th>
                        <th>Lower bound</th>
                        <th>Upper bound</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row) => (
                        <tr key={row.year}>
                          <th scope="row">{row.year}</th>
                          <td>{percent(row.low)}</td>
                          <td>{percent(row.high)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
            <details>
              <summary>Monitoring and response plan</summary>
              <p>{target.monitoringMethod}</p>
              <p>Timing: {target.monitoringSeason}</p>
              <p>If off track: {target.responseIfOffTrack}</p>
              <p>Baseline basis: {target.baselineBasis}</p>
            </details>
            {target.baselineSurveyId && (
              <button
                type="button"
                onClick={() => {
                  useLandContext.getState().selectSurvey(target.baselineSurveyId ?? null);
                  useLandContext.getState().setSection("ecology");
                }}
              >
                Open baseline field survey
              </button>
            )}
            <details>
              <summary>Scope and limitations</summary>
              <ul>
                {species.limitations.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </details>
          </article>
        );
      })}
      {inspected && evidence.data && (
        <EvidenceView evidence={evidence.data} onClose={() => setInspected(null)} />
      )}
      {evidence.isError && <p role="alert">The saved source could not be read.</p>}
      <details>
        <summary>Species-plan method and limitations</summary>
        <ul>
          {result.limitations.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
        <small>{result.algorithm}</small>
      </details>
    </section>
  );
}

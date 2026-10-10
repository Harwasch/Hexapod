/* eslint-disable jsx-a11y/no-noninteractive-tabindex -- Overflow tables need keyboard scrolling. */
import { useLandContext } from "@/state/landContext";
import type { LandEvidence, ResearchArtifact } from "@twin/contracts";

import { ResearchMapExplorer } from "./ResearchMapExplorer";
import { ArchiveGallery } from "./ArchiveGallery";
import { SolarAssessmentView } from "./SolarAssessmentView";
import { LandRasterView } from "./LandRasterView";
import { ChartView } from "./ResearchChart";
import { CalculationView } from "./CalculationView";

export function ResearchArtifactView({
  artifact,
  evidence,
  onAsk,
  onEvidence,
}: {
  artifact: ResearchArtifact;
  evidence?: LandEvidence[];
  onAsk?: () => void;
  onEvidence?: (id: string) => void;
}) {
  const output = artifact.output;
  return (
    <article className="land-artifact">
      <h4>{artifact.title}</h4>
      {output.kind === "gallery" && (
        <ArchiveGallery ids={output.evidenceIds} evidence={evidence} onAsk={onAsk} />
      )}
      {output.kind === "solar" && (
        <>
          <SolarAssessmentView id={output.assessmentId} />
          <button
            type="button"
            onClick={() => {
              useLandContext.getState().selectSolar(output.assessmentId);
              useLandContext.getState().setSection("scenarios");
            }}
          >
            Explore solar economics
          </button>
        </>
      )}
      {output.kind === "raster" && <LandRasterView id={output.rasterId} />}
      {output.kind === "calculation" && (
        <CalculationView
          artifact={artifact}
          output={output}
          evidence={evidence}
          onEvidence={onEvidence}
        />
      )}
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
        <ResearchMapExplorer artifactId={artifact.id} title={artifact.title} output={output} />
      )}
      <details>
        <summary>Method and limitations</summary>
        <p>{artifact.method}</p>
      </details>
    </article>
  );
}

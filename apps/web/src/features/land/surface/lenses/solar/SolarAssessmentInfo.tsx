import { useState, type ReactNode } from "react";
import { Download } from "lucide-react";

import { GlassButton } from "@twin/ui";

import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";

import { Fact } from "../../mapLinks";
import { useLens } from "../../lenses";
import { landError } from "../../landFlow";
import { downloadAssessment, useSolarAssessment } from "./solarData";
import { zoneTarget, type SolarAssessment } from "./solarDraft";
import { SolarMonthly, SolarMonthlyTable } from "./SolarMonthly";
import { SolarSettingsList } from "./SolarSettings";
import {
  assessmentHeadline,
  dateLabel,
  energyLabel,
  powerLabel,
  shadingWords,
  zoneArea,
} from "./solarWords";
import "./solarLens.css";

/** Size, energy and shading of a saved array; each points at the array on the map. */
export function SolarAssessmentFacts({
  assessment,
  extra,
}: {
  assessment: SolarAssessment;
  /** A fact of the caller's (payback) appended to the list. */
  extra?: ReactNode;
}) {
  const { metadata, request } = assessment;
  const target = zoneTarget(assessment);
  const complete = metadata.completeYear && metadata.annualGenerationKwh != null;
  return (
    <ul className="land-facts">
      <Fact
        label="Size"
        value={`${powerLabel(metadata.capacityKwDc)} of panels on ${zoneArea(metadata.mappedZoneAreaM2)}`}
        target={target}
      />
      {complete ? (
        <Fact
          label="Yearly energy"
          value={`About ${energyLabel(metadata.annualGenerationKwh ?? 0)} with ${request.year} weather`}
          target={target}
        />
      ) : (
        <Fact
          label={`Energy in ${request.year}`}
          tone="warn"
          value={`${energyLabel(metadata.modeledGenerationKwh)} in ${metadata.validHours.toLocaleString("en-US")} of ${metadata.expectedHours.toLocaleString("en-US")} hours`}
          note="The weather record has gaps, so this is not a full year."
          target={target}
        />
      )}
      <Fact label="Shading" value={shadingWords(assessment)} />
      {extra}
    </ul>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

const number = (value: number, digits = 0) =>
  Number.isFinite(value) ? value.toLocaleString("en-US", { maximumFractionDigits: digits }) : "—";

/**
 * Everything recorded with a saved assessment: the numbers, the settings, the notes, what the
 * model cannot tell you, and where the weather came from, with the hourly data to download.
 */
export function SolarAssessmentAbout({ assessment }: { assessment: SolarAssessment }) {
  const { metadata, request } = assessment;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <div className="land-solar-about">
      <details className="land-solar-fold">
        <summary>Monthly values</summary>
        <SolarMonthlyTable assessment={assessment} />
      </details>
      <details className="land-solar-fold">
        <summary>About this model run</summary>
        <dl className="land-solar-list">
          <Row label="Panels (DC)">{number(metadata.capacityKwDc, 1)} kW</Row>
          <Row label="Inverters (AC)">{number(metadata.inverterKwAc, 1)} kW</Row>
          <Row label="Highest output">{number(metadata.peakAcKw, 1)} kW AC</Row>
          <Row label="Energy in hours with data">{number(metadata.modeledGenerationKwh)} kWh</Row>
          <Row label="Energy with an open sky">{number(metadata.unshadedGenerationKwh)} kWh</Row>
          <Row label="Sunlight on the panels">
            {number(metadata.planeIrradiationKwhM2)} kWh/m² (
            {number(metadata.unshadedPlaneIrradiationKwhM2)} with an open sky)
          </Row>
          <Row label="Sky the horizon leaves open">
            {number(metadata.horizonSkyFraction * 100, 1)}%
          </Row>
          <Row label="Weather hours">
            {number(metadata.validHours)} of {number(metadata.expectedHours)}
            {metadata.completeYear ? "" : " (incomplete year)"}
          </Row>
          <Row label="Array area">{number(metadata.mappedZoneAreaM2)} m²</Row>
          <Row label="Weather point">
            {metadata.latitude.toFixed(4)}, {metadata.longitude.toFixed(4)} ·{" "}
            {number(metadata.sourceElevationM)} m
          </Row>
          <Row label="Made">
            {dateLabel(assessment.createdAt)} · land outline {assessment.boundaryRevision}
            {assessment.stale ? " (earlier outline)" : ""}
          </Row>
        </dl>
        <SolarSettingsList request={request} />
        <p className="land-solar-note">{request.zoneBasis}</p>
        <p className="land-solar-note">{request.horizonBasis}</p>
        <p className="land-solar-note">{request.assumptions}</p>
      </details>
      {metadata.limitations.length > 0 && (
        <details className="land-solar-fold">
          <summary>What this cannot tell you</summary>
          <ul className="land-solar-bullets">
            {metadata.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </details>
      )}
      <details className="land-solar-fold">
        <summary>Sources</summary>
        <p>
          <a href={metadata.sourceUrl} target="_blank" rel="noreferrer">
            NASA POWER hourly weather
          </a>{" "}
          ·{" "}
          <a href={metadata.documentationUrl} target="_blank" rel="noreferrer">
            How the data is made
          </a>
        </p>
        <p className="land-muted">
          {metadata.attribution} · {metadata.license}
        </p>
        <p className="land-muted">
          {metadata.algorithm} · {metadata.modelVersion} · retrieved{" "}
          {dateLabel(metadata.retrievedAt)}
        </p>
        <p className="land-muted land-solar-hash">Archive SHA-256 {assessment.sha256}</p>
        <p className="land-muted land-solar-hash">Weather SHA-256 {metadata.sourceSha256}</p>
      </details>
      <div className="land-row">
        <GlassButton
          size="sm"
          variant="ghost"
          disabled={busy}
          leadingIcon={<Download size={15} aria-hidden="true" />}
          onClick={() => {
            setBusy(true);
            setError(null);
            void downloadAssessment(assessment.id)
              .catch((cause: unknown) => setError(`The download failed. ${landError(cause)}`))
              .finally(() => setBusy(false));
          }}
        >
          Download hourly data and sources
        </GlassButton>
      </div>
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

/**
 * A saved assessment on its own, for places outside the Solar lens (research results, the
 * scenario form): the sentence, the facts, the months and the details, plus a way to the lens.
 */
export function SolarAssessmentCompact({
  id,
  onUse,
}: {
  id: string;
  onUse?: (assessment: SolarAssessment) => void;
}) {
  const land = useLand((s) => s.active);
  const query = useSolarAssessment(id, land?.revision);
  const assessment = query.data;
  if (!assessment)
    return (
      <p
        className={query.isError ? "land-alert" : "land-muted"}
        role={query.isError ? "alert" : "status"}
      >
        {query.isError ? "This solar assessment could not load. " : "Loading the solar assessment…"}
        {query.isError && (
          <button type="button" className="land-link" onClick={() => void query.refetch()}>
            Try again
          </button>
        )}
      </p>
    );
  return (
    <section className="land-solar land-solar-compact" aria-label="Hourly solar assessment">
      <p className="land-lede">
        {assessmentHeadline(assessment, land?.id === assessment.landId ? land : null)}
      </p>
      {assessment.stale && <p className="land-note">Made for an earlier outline of the land.</p>}
      <SolarAssessmentFacts assessment={assessment} />
      <SolarMonthly assessment={assessment} />
      <div className="land-row">
        <GlassButton
          size="sm"
          onClick={() => {
            useLandContext.getState().selectSolar(assessment.id);
            useLens.getState().setLens("solar");
          }}
        >
          Open in Solar
        </GlassButton>
        {onUse && (
          <GlassButton
            size="sm"
            variant="ghost"
            disabled={!assessment.metadata.completeYear || assessment.stale}
            onClick={() => onUse(assessment)}
          >
            Use in financial scenario
          </GlassButton>
        )}
      </div>
      <SolarAssessmentAbout assessment={assessment} />
    </section>
  );
}

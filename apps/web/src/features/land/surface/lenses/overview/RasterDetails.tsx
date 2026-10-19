import { useState, type ReactNode } from "react";
import { Download } from "lucide-react";

import type { UnitSystem } from "@twin/geo";
import { GlassButton } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";

import {
  className,
  GREENNESS_SCALE,
  heightLabel,
  isBareEarth,
  monthLabel,
  percentLabel,
  presentClasses,
  rasterKind,
  rasterSource,
  shareLabel,
  SLOPE_SCALE,
  slopePercent,
  type Observation,
  type Raster,
  type RasterBand,
  type Timeline,
} from "./overviewData";
import "./overview.css";

const number = (value: number, digits = 1) =>
  value.toLocaleString(undefined, { maximumFractionDigits: digits });
const ndvi = (value: number | null | undefined) => (value == null ? "No data" : value.toFixed(2));
const signed = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(2)}`;

/**
 * Everything one saved raster holds, for the folded Details: how much of the land it covers,
 * full band statistics with their distribution, land-cover classes or vegetation observations
 * with their quality, warnings, sources and licences, the method, and the GeoTIFF download.
 */
export function RasterDetails({
  raster,
  units,
  band,
  onChooseBand,
}: {
  raster: Raster;
  units: UnitSystem;
  /** The band on the map, so a vegetation month can show as chosen. */
  band?: number | null;
  /** Shows a band on the map (a vegetation month, the change). */
  onChooseBand?: (band: number) => void;
}) {
  const kind = rasterKind(raster);
  const metadata = raster.metadata;
  const coverage = metadata.boundaryCells
    ? `${shareLabel(metadata.validCells / metadata.boundaryCells)} of the land has data (${metadata.validCells.toLocaleString()} of ${metadata.boundaryCells.toLocaleString()} grid cells)`
    : "No grid cells fall inside this land at this resolution.";
  return (
    <div className="overview-raster">
      <p className="land-muted">
        {rasterSource(raster)} · {number(metadata.resolutionM)} m grid · made{" "}
        {new Date(raster.createdAt).toLocaleDateString()} for outline {raster.boundaryRevision}
        {raster.stale ? " (an earlier one)" : ""}
      </p>
      <p className="overview-raster__coverage">{coverage}</p>
      {kind === "terrain" && <TerrainStats raster={raster} units={units} />}
      {kind === "cover" && <CoverClasses raster={raster} />}
      {kind === "signature" && <SignatureSummary raster={raster} />}
      {(kind === "vegetation" || kind === "signature") && metadata.vegetation && (
        <VegetationObservations
          raster={raster}
          series={metadata.vegetation}
          band={band ?? null}
          onChooseBand={onChooseBand}
          seasons={kind === "signature"}
        />
      )}
      {metadata.warnings.length > 0 && (
        <ul className="overview-warnings" aria-label="Warnings">
          {metadata.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      <Sources raster={raster} />
      <p className="land-muted">
        Method: {metadata.method} ({metadata.algorithm}, {metadata.width} × {metadata.height} cells,{" "}
        {metadata.crs})
      </p>
      <DownloadRaster raster={raster} />
    </div>
  );
}

// ---- Terrain ---------------------------------------------------------------

function TerrainStats({ raster, units }: { raster: Raster; units: UnitSystem }) {
  const boundaryCells = raster.metadata.boundaryCells;
  const bare = isBareEarth(raster);
  return (
    <>
      {raster.metadata.bands.map((band) => {
        const slope = band.index === 2;
        const format = slope
          ? (value: number) => percentLabel(slopePercent(value))
          : (value: number) => heightLabel(value, units);
        return (
          <BandStats
            key={band.index}
            title={slope ? "Slope" : band.index === 1 ? "Elevation" : band.name}
            band={band}
            format={format}
            boundaryCells={boundaryCells}
            note={
              slope
                ? `${SLOPE_SCALE} The model gives degrees; percent is rise over run. Median ${band.percentiles["50"] == null ? "unknown" : `${number(band.percentiles["50"])}°`}.`
                : bare
                  ? "Ground height with trees and buildings left out (bare earth), in NAVD88 metres. Not a ground survey."
                  : "Surface height, including buildings and tree tops. Not a ground survey."
            }
          />
        );
      })}
    </>
  );
}

const STATS: readonly [string, (band: RasterBand) => number | null | undefined][] = [
  ["Lowest", (band) => band.minimum],
  ["5%", (band) => band.percentiles["5"]],
  ["25%", (band) => band.percentiles["25"]],
  ["Median", (band) => band.percentiles["50"]],
  ["75%", (band) => band.percentiles["75"]],
  ["95%", (band) => band.percentiles["95"]],
  ["Highest", (band) => band.maximum],
  ["Average", (band) => band.mean],
];

/** One band's full statistics and its distribution. */
export function BandStats({
  title,
  band,
  format,
  boundaryCells,
  note,
}: {
  title: string;
  band: RasterBand;
  format: (value: number) => string;
  boundaryCells: number;
  note?: ReactNode;
}) {
  return (
    <section className="overview-band" aria-label={`${title} statistics`}>
      <h5>{title}</h5>
      {band.validCells ? (
        <>
          <dl className="overview-stats">
            {STATS.map(([label, read]) => {
              const value = read(band);
              return value == null ? null : (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd>{format(value)}</dd>
                </div>
              );
            })}
            {band.standardDeviation != null && (
              <div>
                <dt>Spread (sd)</dt>
                <dd>
                  {number(band.standardDeviation, 2)} {band.unit}
                </dd>
              </div>
            )}
            <div>
              <dt>Cells with data</dt>
              <dd>
                {band.validCells.toLocaleString()}
                {boundaryCells ? ` (${shareLabel(band.validCells / boundaryCells)})` : ""}
              </dd>
            </div>
          </dl>
          <Histogram band={band} format={format} title={title} />
        </>
      ) : (
        <p className="land-muted">No cells with data inside this land.</p>
      )}
      {note && <p className="land-muted">{note}</p>}
    </section>
  );
}

function Histogram({
  band,
  format,
  title,
}: {
  band: RasterBand;
  format: (value: number) => string;
  title: string;
}) {
  const counts = band.histogramCounts;
  const edges = band.histogramEdges;
  const first = edges[0];
  const last = edges.at(-1);
  if (!counts.length || first == null || last == null) return null;
  const most = Math.max(1, ...counts);
  const width = 300 / counts.length;
  return (
    <figure className="overview-histogram">
      <svg
        viewBox="0 0 300 56"
        preserveAspectRatio="none"
        role="img"
        aria-label={`${title}: how the land's cells spread from ${format(first)} to ${format(last)}`}
      >
        {counts.map((count, index) => {
          const height = (count / most) * 52;
          const low = edges[index];
          const high = edges[index + 1];
          return (
            <rect
              key={index}
              x={index * width + 0.5}
              y={56 - height}
              width={Math.max(1, width - 1)}
              height={height}
            >
              <title>
                {low == null || high == null ? "" : `${format(low)} to ${format(high)}: `}
                {count.toLocaleString()} cells
              </title>
            </rect>
          );
        })}
      </svg>
      <figcaption className="overview-ends">
        <span>{format(first)}</span>
        <span>{format(last)}</span>
      </figcaption>
    </figure>
  );
}

// ---- Land cover ------------------------------------------------------------

function CoverClasses({ raster }: { raster: Raster }) {
  const band = raster.metadata.bands[0];
  const classes = presentClasses(band ?? null);
  return (
    <section className="overview-band" aria-label="Land-cover classes">
      <h5>Classes</h5>
      {classes.length ? (
        <ul className="overview-class-table">
          {classes.map((item) => (
            <li key={item.code}>
              <span
                className="overview-chip"
                style={{ background: item.color }}
                aria-hidden="true"
              />
              <span className="overview-class-table__name">
                {className(item)}
                {className(item) !== item.label && <small> ({item.label})</small>}
              </span>
              <strong>{shareLabel(item.fraction)}</strong>
              <span className="overview-class-table__bar" aria-hidden="true">
                <span style={{ width: `${(item.fraction ?? 0) * 100}%`, background: item.color }} />
              </span>
              <small className="overview-class-table__area">
                {number(item.sampledAreaM2 / 10000, 2)} ha sampled · {item.cells.toLocaleString()}{" "}
                cells
              </small>
            </li>
          ))}
        </ul>
      ) : (
        <p className="land-muted">No land-cover classes were sampled inside this land.</p>
      )}
      <p className="land-muted">
        Broad classes mapped in 2021. They do not identify species or show today&apos;s conditions.
        Shares count cells with a known class only.
      </p>
    </section>
  );
}

// ---- Seasonal signatures -----------------------------------------------------

/** What a signature holds: the cells with a full signature, and which bands go where. */
function SignatureSummary({ raster }: { raster: Raster }) {
  const signature = raster.metadata.signature;
  const bands = raster.metadata.bands.length;
  return (
    <section className="overview-band" aria-label="Seasonal signature">
      <h5>Seasonal signature</h5>
      {signature && (
        <p>
          {signature.commonCells.toLocaleString()} cells clear in every season carry a full{" "}
          {signature.features.length}-band signature ({shareLabel(signature.commonCoverageFraction)}{" "}
          of the land).
        </p>
      )}
      <p className="land-muted">
        The map shows each season&apos;s NDVI; the download holds all {bands} measurement bands
        (green, red, near-infrared, shortwave-infrared and NDVI per season). Show once in Ecology
        compares places by them. A similar signature means similar reflectance on these dates, not
        the same species.
      </p>
    </section>
  );
}

// ---- Vegetation ------------------------------------------------------------

function VegetationObservations({
  raster,
  series,
  band,
  onChooseBand,
  seasons = false,
}: {
  raster: Raster;
  series: Timeline;
  band: number | null;
  onChooseBand?: (band: number) => void;
  /** A signature's seasons rather than a comparison of months. */
  seasons?: boolean;
}) {
  const changeBand = series.changeBand;
  return (
    <section className="overview-band" aria-label="Vegetation observations">
      <h5>{seasons ? "Greenness by season" : "Greenness through time"}</h5>
      <p className="land-muted">
        {series.commonCells.toLocaleString()} cells were seen clearly every time (
        {shareLabel(series.commonCoverageFraction)} of the land). The trend uses only those.
      </p>
      <TrendChart series={series} />
      <ol className="overview-observations">
        {series.observations.map((item) => (
          <ObservationItem
            key={item.band}
            item={item}
            raster={raster}
            chosen={band === item.band}
            onChoose={onChooseBand}
          />
        ))}
      </ol>
      {changeBand != null && (
        <div className="overview-change">
          <p>
            Last minus first:{" "}
            <strong>{series.meanChange == null ? "No data" : signed(series.meanChange)}</strong> on{" "}
            {series.changeCells.toLocaleString()} cells seen clearly both times.
          </p>
          {onChooseBand && (
            <GlassButton
              size="sm"
              disabled={!series.changeCells}
              active={band === changeBand}
              onClick={() => onChooseBand(changeBand)}
            >
              View vegetation change
            </GlassButton>
          )}
        </div>
      )}
      <p className="land-muted">
        {GREENNESS_SCALE} Season, weather and sensors affect it; it does not show species, causes or
        restoration success.
      </p>
    </section>
  );
}

function ObservationItem({
  item,
  raster,
  chosen,
  onChoose,
}: {
  item: Observation;
  raster: Raster;
  chosen: boolean;
  onChoose?: (band: number) => void;
}) {
  const month = item.period.startDate.slice(0, 7);
  const band = raster.metadata.bands.find((value) => value.index === item.band);
  const quality = Object.entries(item.qualityCounts).filter(([, count]) => count > 0);
  return (
    <li className="overview-observation">
      <div className="overview-observation__head">
        <strong>{monthLabel(item.period.startDate)}</strong>
        {onChoose && (
          <GlassButton
            size="sm"
            variant="ghost"
            active={chosen}
            disabled={!item.validCells}
            onClick={() => onChoose(item.band)}
          >
            {month} observation
          </GlassButton>
        )}
      </div>
      <dl className="overview-stats">
        <div>
          <dt>Seen on</dt>
          <dd>{item.acquiredAt ? item.acquiredAt.slice(0, 10) : "No clear scene"}</dd>
        </div>
        <div>
          <dt>Clear view</dt>
          <dd>{shareLabel(item.coverageFraction)}</dd>
        </div>
        <div>
          <dt>Greenness, shared cells</dt>
          <dd>{ndvi(item.commonMeanNdvi)}</dd>
        </div>
        <div>
          <dt>Greenness, all clear cells</dt>
          <dd>{ndvi(item.meanNdvi)}</dd>
        </div>
      </dl>
      <details className="overview-quality">
        <summary>Acquisition and quality</summary>
        <p>{item.explanation}</p>
        <p>
          Window {item.period.startDate} to {item.period.endDate}. {item.qualityCandidatesExamined}{" "}
          quality masks checked from {item.candidateCount} catalog scenes
          {item.catalogTruncated ? " (catalog truncated)" : ""}.
        </p>
        {item.sceneId && <p>Scene {item.sceneId}</p>}
        {quality.length > 0 && (
          <ul className="overview-quality__counts" aria-label="Cells by quality">
            {quality.map(([label, count]) => (
              <li key={label}>
                {label}: {count.toLocaleString()} cells
              </li>
            ))}
          </ul>
        )}
        {band && (
          <BandStats
            title={`Greenness, ${monthLabel(item.period.startDate)}`}
            band={band}
            format={(value) => value.toFixed(2)}
            boundaryCells={raster.metadata.boundaryCells}
          />
        )}
      </details>
    </li>
  );
}

function TrendChart({ series }: { series: Timeline }) {
  const points = series.observations
    .filter((item) => item.commonMeanNdvi != null)
    .map((item) => ({
      t: Date.parse(item.acquiredAt ?? item.period.startDate),
      value: item.commonMeanNdvi ?? 0,
      label: monthLabel(item.period.startDate),
    }));
  if (!series.commonCells || points.length < 2)
    return (
      <p className="land-note">
        No comparable trend: these months share no clearly seen cells. Try other months.
      </p>
    );
  const first = Math.min(...points.map((point) => point.t));
  const last = Math.max(...points.map((point) => point.t));
  const low = Math.min(0, ...points.map((point) => point.value));
  const x = (t: number) => 12 + ((t - first) / Math.max(1, last - first)) * 276;
  const y = (value: number) => 8 + ((1 - value) / (1 - low || 1)) * 64;
  const path = points.map((point, i) => `${i ? "L" : "M"}${x(point.t)},${y(point.value)}`).join("");
  return (
    <figure className="overview-trend">
      <svg
        viewBox="0 0 300 80"
        role="img"
        aria-label={`Greenness on cells seen clearly every time: ${points
          .map((point) => `${point.label} ${point.value.toFixed(2)}`)
          .join(", ")}`}
      >
        <line x1={12} x2={288} y1={y(0)} y2={y(0)} className="overview-trend__axis" />
        <path d={path} className="overview-trend__line" />
        {points.map((point) => (
          <circle key={point.t} cx={x(point.t)} cy={y(point.value)} r={3.5}>
            <title>
              {point.label}: {point.value.toFixed(2)}
            </title>
          </circle>
        ))}
      </svg>
      <figcaption className="overview-ends">
        <span>{points[0]?.label}</span>
        <span>{points.at(-1)?.label}</span>
      </figcaption>
    </figure>
  );
}

// ---- Sources and download --------------------------------------------------

function Sources({ raster }: { raster: Raster }) {
  const [all, setAll] = useState(false);
  const sources = raster.metadata.sources;
  if (!sources.length) return null;
  const shown = all ? sources : sources.slice(0, 3);
  return (
    <section className="overview-band" aria-label="Sources and licences">
      <h5>Sources and licences</h5>
      <ul className="overview-sources">
        {shown.map((source, index) => (
          <li key={`${source.id}:${source.band ?? index}`}>
            <a href={source.catalogUrl} target="_blank" rel="noopener noreferrer">
              {source.id}
            </a>
            {source.band ? ` · ${source.band}` : ""} · {source.observationPeriod}
            <br />
            <a href={source.licenseUrl} target="_blank" rel="noopener noreferrer">
              {source.license}
            </a>{" "}
            · {source.attribution}
          </li>
        ))}
      </ul>
      {sources.length > shown.length && (
        <button type="button" className="land-link" onClick={() => setAll(true)}>
          Show all {sources.length} sources
        </button>
      )}
    </section>
  );
}

function DownloadRaster({ raster }: { raster: Raster }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const download = async () => {
    setBusy(true);
    setError(null);
    try {
      const blob = await unwrap(
        api.GET("/api/v1/land/rasters/{raster_id}/download", {
          params: { path: { raster_id: raster.id } },
          parseAs: "blob",
        }),
      );
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `land-${rasterKind(raster)}-${raster.id}.tif`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) {
      setError(`The download failed. ${describeError(cause)}`);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="land-row">
      <GlassButton
        size="sm"
        variant="ghost"
        loading={busy}
        leadingIcon={<Download size={14} aria-hidden="true" />}
        onClick={() => void download()}
      >
        Download GeoTIFF
      </GlassButton>
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

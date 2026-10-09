import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { components } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLandContext } from "@/state/landContext";
import { useLandScope, useLandAccessReady } from "@/state/landIdentity";
import { useLand } from "@/state/land";
import { useSelection } from "@/state/selection";

type Sample = components["schemas"]["RasterSample"];
const number = (value: number | null | undefined) =>
  value == null ? "No data" : value.toLocaleString(undefined, { maximumFractionDigits: 1 });

export function LandRasterView({ id }: { id: string }) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const revision = useLand((state) => state.active?.revision);
  const scene = useScene();
  const shown = useLandContext((state) => state.rasters[id]);
  const tileError = useLandContext((state) => state.rasterErrors[id]);
  const selection = useSelection((state) => state.selection);
  const [preferredBand, setChosenBand] = useState(shown?.band ?? 1);
  const [preferredOpacity, setOpacity] = useState(shown?.opacity ?? 0.8);
  const chosenBand = shown?.band ?? preferredBand;
  const opacity = shown?.opacity ?? preferredOpacity;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sample, setSample] = useState<Sample | null>(null);
  const query = useQuery({
    queryKey: ["land-rasters", scope, id, revision],
    queryFn: () =>
      unwrap(api.GET("/api/v1/land/rasters/{raster_id}", { params: { path: { raster_id: id } } })),
    enabled: ready,
    retry: false,
  });
  const raster = query.data;
  const band = raster?.metadata.bands.find((item) => item.index === chosenBand);
  const show = (index = chosenBand, alpha = opacity) => {
    if (!raster) return;
    useLandContext.getState().setRaster({
      id,
      band: index,
      bounds: raster.metadata.bounds,
      opacity: alpha,
      attribution: [...new Set(raster.metadata.sources.map((source) => source.attribution))].join(
        " · ",
      ),
    });
  };
  const download = async () => {
    setBusy(true);
    setError(null);
    try {
      const blob = await unwrap(
        api.GET("/api/v1/land/rasters/{raster_id}/download", {
          params: { path: { raster_id: id } },
          parseAs: "blob",
        }),
      );
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `land-terrain-${id}.tif`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };
  const samplePoint = async () => {
    if (!selection) return;
    setBusy(true);
    setError(null);
    try {
      setSample(
        await unwrap(
          api.GET("/api/v1/land/rasters/{raster_id}/sample", {
            params: {
              path: { raster_id: id },
              query: { longitude: selection.longitude, latitude: selection.latitude },
            },
          }),
        ),
      );
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };
  if (query.isPending) return <p role="status">Loading terrain analysis…</p>;
  if (!raster || !band)
    return (
      <div className="land-error" role="alert">
        Terrain analysis could not be loaded.{" "}
        <button type="button" onClick={() => void query.refetch()}>
          Retry terrain
        </button>
      </div>
    );
  const metadata = raster.metadata;
  return (
    <section className="land-raster" aria-label="Terrain analysis">
      <p className="land-footnote">
        Copernicus GLO-30 · {number(metadata.resolutionM)} m analysis grid · boundary{" "}
        {raster.boundaryRevision}
      </p>
      {raster.stale && (
        <p className="land-notice">
          This map uses an older boundary. Run terrain analysis again for your current land.
        </p>
      )}
      <label className="land-name">
        Map measurement
        <select
          value={chosenBand}
          onChange={(event) => {
            const value = Number(event.target.value);
            setChosenBand(value);
            if (shown) show(value);
          }}
        >
          {metadata.bands.map((item) => (
            <option key={item.index} value={item.index}>
              {item.name} ({item.unit})
            </option>
          ))}
        </select>
      </label>
      <dl className="land-raster-stats">
        <div>
          <dt>Minimum</dt>
          <dd>
            {number(band.minimum)} {band.minimum !== null && band.unit}
          </dd>
        </div>
        <div>
          <dt>Mean</dt>
          <dd>
            {number(band.mean)} {band.mean !== null && band.unit}
          </dd>
        </div>
        <div>
          <dt>Maximum</dt>
          <dd>
            {number(band.maximum)} {band.maximum !== null && band.unit}
          </dd>
        </div>
      </dl>
      {band.validCells > 0 && (
        <>
          <div className={`land-raster-ramp ${band.palette}`} aria-hidden="true" />
          <p className="land-raster-scale">
            <span>
              {number(band.minimum)} {band.unit}
            </span>
            <span>
              {number(band.maximum)} {band.unit}
            </span>
          </p>
        </>
      )}
      <p className="land-footnote">
        {band.validCells.toLocaleString()} valid cells ·{" "}
        {metadata.boundaryCells
          ? number((band.validCells / metadata.boundaryCells) * 100) +
            "% coverage of boundary grid cells"
          : "No grid cells inside this boundary"}
      </p>
      <div className="land-actions">
        <button
          type="button"
          aria-pressed={Boolean(shown)}
          disabled={!scene || !band.validCells}
          onClick={() => {
            if (shown) {
              setChosenBand(shown.band);
              setOpacity(shown.opacity);
              useLandContext.getState().removeRaster(id);
            } else show();
          }}
        >
          {shown ? "Hide terrain map" : "Show terrain map"}
        </button>
        <button
          type="button"
          disabled={!scene}
          onClick={() => {
            const [w = 0, s = 0, e = 0, n = 0] = metadata.bounds;
            scene?.camera.flyToRectangle(w, s, e, n);
          }}
        >
          Frame terrain
        </button>
        <button type="button" disabled={busy} onClick={() => void download()}>
          Download GeoTIFF
        </button>
      </div>
      {shown && (
        <label className="land-raster-opacity">
          Map opacity {Math.round(opacity * 100)}%
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(event) => {
              const value = Number(event.target.value);
              setOpacity(value);
              show(chosenBand, value);
            }}
          />
        </label>
      )}
      {shown && <p className="land-footnote">Up to two analysis maps can be visible together.</p>}
      {(error ?? tileError) && (
        <p className="land-error" role="alert">
          {error ?? tileError}
        </p>
      )}
      <details>
        <summary>Sample a point</summary>
        <p>Select a point on the map, then read its nearest analysis cell.</p>
        <div className="land-actions">
          <button type="button" disabled={!selection || busy} onClick={() => void samplePoint()}>
            Sample selected point
          </button>
        </div>
        {sample && (
          <div aria-live="polite">
            <p>
              {sample.longitude.toFixed(5)}, {sample.latitude.toFixed(5)}
            </p>
            {metadata.bands.map((item, index) => (
              <p key={item.index}>
                {item.name}: {number(sample.values[index])}{" "}
                {sample.values[index] != null && item.unit}
              </p>
            ))}
            <p className="land-footnote">{sample.interpretation}</p>
          </div>
        )}
      </details>
      <details>
        <summary>Distribution and coverage</summary>
        <p>Percentiles describe valid sampled cells, not every point on the land.</p>
        <dl>
          {Object.entries(band.percentiles).map(([key, value]) => (
            <div key={key}>
              <dt>{key} percentile</dt>
              <dd>
                {number(value)} {band.unit}
              </dd>
            </div>
          ))}
        </dl>
        <svg
          className="land-chart"
          viewBox="0 0 300 90"
          role="img"
          aria-label={`${band.name} distribution. Percentiles are listed above.`}
        >
          {band.histogramCounts.map((count, i) => {
            const height = (count / Math.max(1, ...band.histogramCounts)) * 75;
            return (
              <rect
                key={i}
                x={(i * 300) / band.histogramCounts.length}
                y={85 - height}
                width={Math.max(1, 300 / band.histogramCounts.length - 1)}
                height={height}
                fill="currentColor"
              >
                <title>
                  {number(band.histogramEdges[i])}–{number(band.histogramEdges[i + 1])} {band.unit}:{" "}
                  {count} cells
                </title>
              </rect>
            );
          })}
        </svg>
      </details>
      <details>
        <summary>Sources and limitations</summary>
        {metadata.warnings.map((warning) => (
          <p key={warning}>{warning}</p>
        ))}
        {metadata.sources.map((source) => (
          <div key={source.id}>
            <a href={source.catalogUrl} target="_blank" rel="noopener noreferrer">
              {source.id}
            </a>
            <p>{source.observationPeriod}</p>
            <a href={source.licenseUrl} target="_blank" rel="noopener noreferrer">
              {source.license}
            </a>
            <p className="land-footnote">{source.attribution}</p>
          </div>
        ))}
        <p className="land-footnote">
          Computed {new Date(raster.createdAt).toLocaleString()} · {metadata.algorithm}. The
          download preserves both measurement bands and their coordinate system.
        </p>
      </details>
    </section>
  );
}

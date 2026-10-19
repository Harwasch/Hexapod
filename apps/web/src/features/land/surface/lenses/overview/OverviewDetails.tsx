import { useId, type ReactNode } from "react";

import type { UnitSystem } from "@twin/geo";
import { GlassSelect } from "@twin/ui";

import { askAgent } from "../../agentBus";
import { LensDetails } from "../../LensCard";
import { OverviewLegend } from "./OverviewLegend";
import { OverviewPeriods } from "./OverviewPeriods";
import {
  mapBands,
  monthLabel,
  rasterSource,
  terrainOf,
  type OverviewData,
  type Period,
} from "./overviewData";
import { RasterDetails } from "./RasterDetails";
import {
  setOverviewOpacity,
  useOverviewMap,
  type OverviewView,
  type ShownLayer,
} from "./useOverview";

interface Choice {
  value: string;
  label: string;
  view: OverviewView;
}

function choices(data: OverviewData): Choice[] {
  const list: Choice[] = [];
  if (terrainOf(data)) {
    list.push({ value: "elevation", label: "Elevation", view: { layer: "elevation" } });
    list.push({ value: "slope", label: "Slope", view: { layer: "slope" } });
  }
  if (data.cover)
    list.push({ value: "cover", label: "Land cover, 2021", view: { layer: "cover" } });
  const series = data.vegetation?.metadata.vegetation;
  for (const item of series?.observations ?? [])
    list.push({
      value: `vegetation:${item.band}`,
      label: `Greenness, ${monthLabel(item.period.startDate)}`,
      view: { layer: "vegetation", band: item.band },
    });
  if (series?.changeBand != null)
    list.push({
      value: `vegetation:${series.changeBand}`,
      label: "Greenness change",
      view: { layer: "vegetation", band: series.changeBand },
    });
  // A signature's seasons, by their dated NDVI: the reflectance bands never go on the map.
  const signature = data.signature;
  const ndvi = new Set(signature ? mapBands(signature).map((band) => band.index) : []);
  for (const item of signature?.metadata.vegetation?.observations ?? [])
    if (ndvi.has(item.band))
      list.push({
        value: `signature:${item.band}`,
        label: `Seasonal greenness, ${monthLabel(item.period.startDate)}`,
        view: { layer: "signature", band: item.band },
      });
  return list;
}

function valueOf(shown: ShownLayer | null): string {
  if (!shown) return "";
  return shown.view.layer === "vegetation" || shown.view.layer === "signature"
    ? `${shown.view.layer}:${shown.band}`
    : shown.view.layer;
}

/**
 * The Overview's full tools, folded away: what the map shows and how strongly, then each
 * analysis in full (statistics, distribution, classes, observations, sources, downloads), the
 * months a greenness comparison uses, and the analyses offered on demand (bare-earth terrain,
 * seasonal signatures).
 */
export function OverviewDetails({
  data,
  shown,
  rasterLayerId,
  units,
  canEdit,
  busy,
  onView,
  onCompare,
  compareBlocked,
  spotNote,
  bareEarth,
  signatures,
  idle,
}: {
  data: OverviewData;
  shown: ShownLayer | null;
  rasterLayerId: string | null;
  units: UnitSystem;
  canEdit: boolean;
  busy: boolean;
  onView: (view: OverviewView | null) => void;
  onCompare: (periods: Period[]) => void;
  /** Why a greenness comparison cannot start now (one is queued or running). */
  compareBlocked: string | null;
  spotNote: string | null;
  /** The bare-earth analysis, offered until this outline has it. */
  bareEarth?: ReactNode;
  /** The seasonal signatures, offered until this outline has them. */
  signatures?: ReactNode;
  /** The research worker's absence, when the card has no start of its own to say it by. */
  idle?: ReactNode;
}) {
  const compareReasonId = useId();
  const opacity = useOverviewMap((s) => s.opacity);
  const options = choices(data);
  const series = data.vegetation?.metadata.vegetation;
  const previousMonths = series?.observations.map((item) => item.period.startDate.slice(0, 7));
  const vegetationBand = shown?.view.layer === "vegetation" ? shown.band : null;
  const signatureBand = shown?.view.layer === "signature" ? shown.band : null;
  const ground = terrainOf(data);
  // The other terrain model, when both were run: the card reads `ground`.
  const otherTerrain = ground === data.bareEarth ? data.terrain : data.bareEarth;
  return (
    <LensDetails>
      {options.length > 0 && (
        <div className="overview-map-controls">
          <label className="land-field">
            Map measurement
            <GlassSelect
              value={valueOf(shown)}
              onChange={(event) => {
                const picked = options.find((item) => item.value === event.target.value);
                onView(picked ? picked.view : null);
              }}
            >
              <option value="">Nothing</option>
              {options.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </GlassSelect>
          </label>
          {rasterLayerId && (
            <label className="land-field">
              Map opacity {Math.round(opacity * 100)}%
              <input
                type="range"
                className="overview-range"
                min="0.1"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(event) => setOverviewOpacity(rasterLayerId, Number(event.target.value))}
              />
            </label>
          )}
          {/* A season's NDVI has no fact on the card to carry its legend. */}
          {shown?.view.layer === "signature" && <OverviewLegend shown={shown} units={units} />}
        </div>
      )}
      {spotNote && <p className="land-muted">Spot readings: {spotNote}</p>}
      {(ground ?? bareEarth) && (
        <section className="overview-section" aria-label="Terrain details">
          <h4>Terrain</h4>
          {ground && <RasterDetails raster={ground} units={units} />}
          {otherTerrain && (
            <details className="overview-quality">
              <summary>Also saved: {rasterSource(otherTerrain)}</summary>
              <RasterDetails raster={otherTerrain} units={units} />
            </details>
          )}
          {bareEarth}
        </section>
      )}
      {data.cover && (
        <section className="overview-section" aria-label="Land cover details">
          <h4>Land cover</h4>
          <RasterDetails raster={data.cover} units={units} />
        </section>
      )}
      {(Boolean(data.vegetation) || canEdit) && (
        <section className="overview-section" aria-label="Vegetation details">
          <h4>Vegetation</h4>
          {data.vegetation && (
            <RasterDetails
              raster={data.vegetation}
              units={units}
              band={vegetationBand}
              onChooseBand={(band) => onView({ layer: "vegetation", band })}
            />
          )}
          {canEdit && (
            <>
              <h5>{data.vegetation ? "Compare other months" : "Compare months you choose"}</h5>
              <OverviewPeriods
                key={previousMonths?.join(",") ?? "default"}
                disabled={busy || Boolean(compareBlocked)}
                busy={busy}
                initial={previousMonths}
                onStart={onCompare}
                describedBy={compareBlocked ? compareReasonId : undefined}
              />
              {compareBlocked && (
                <p id={compareReasonId} className="land-muted" role="status">
                  {compareBlocked}
                </p>
              )}
            </>
          )}
        </section>
      )}
      {(data.signature ?? signatures) && (
        <section className="overview-section" aria-label="Seasonal signature details">
          <h4>Seasonal signatures</h4>
          {data.signature && (
            <RasterDetails
              raster={data.signature}
              units={units}
              band={signatureBand}
              onChooseBand={(band) => onView({ layer: "signature", band })}
            />
          )}
          {signatures}
        </section>
      )}
      {idle}
      <p>
        <button
          type="button"
          className="land-link"
          onClick={() =>
            askAgent(
              "What do this land's terrain, land cover and greenness mean for how it could be used?",
            )
          }
        >
          Ask the agent about this land
        </button>
      </p>
    </LensDetails>
  );
}

import type { UnitSystem } from "@twin/geo";
import { GlassSelect } from "@twin/ui";

import { askAgent } from "../../agentBus";
import { LensDetails } from "../../LensCard";
import { OverviewPeriods } from "./OverviewPeriods";
import { monthLabel, type OverviewData, type Period } from "./overviewData";
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
  if (data.terrain) {
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
  return list;
}

function valueOf(shown: ShownLayer | null): string {
  if (!shown) return "";
  return shown.view.layer === "vegetation" ? `vegetation:${shown.band}` : shown.view.layer;
}

/**
 * The Overview's full tools, folded away: what the map shows and how strongly, then each
 * analysis in full (statistics, distribution, classes, observations, sources, downloads) and
 * the months a greenness comparison uses.
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
  spotNote,
}: {
  data: OverviewData;
  shown: ShownLayer | null;
  rasterLayerId: string | null;
  units: UnitSystem;
  canEdit: boolean;
  busy: boolean;
  onView: (view: OverviewView | null) => void;
  onCompare: (periods: Period[]) => void;
  spotNote: string | null;
}) {
  const opacity = useOverviewMap((s) => s.opacity);
  const options = choices(data);
  const series = data.vegetation?.metadata.vegetation;
  const previousMonths = series?.observations.map((item) => item.period.startDate.slice(0, 7));
  const vegetationBand = shown?.view.layer === "vegetation" ? shown.band : null;
  return (
    <LensDetails>
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
      </div>
      {spotNote && <p className="land-muted">Spot readings: {spotNote}</p>}
      {data.terrain && (
        <section className="overview-section" aria-label="Terrain details">
          <h4>Terrain</h4>
          <RasterDetails raster={data.terrain} units={units} />
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
                disabled={busy}
                busy={busy}
                initial={previousMonths}
                onStart={onCompare}
              />
            </>
          )}
        </section>
      )}
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

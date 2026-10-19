import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { GlassButton, GlassSelect } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { useLandContext } from "@/state/landContext";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";
import { useSettings } from "@/state/settings";

import {
  bandOf,
  defaultVegetationBand,
  mapBands,
  rasterKind,
} from "./surface/lenses/overview/overviewData";
import { RasterDetails } from "./surface/lenses/overview/RasterDetails";

const LABEL = {
  terrain: "Terrain analysis",
  cover: "Land-cover analysis",
  vegetation: "Vegetation analysis",
  signature: "Seasonal signature analysis",
} as const;

/**
 * One saved raster where a research result names it (the agent's artifacts): a switch to put
 * it on the map, the band it shows, and the Overview lens's full details for it. The Overview
 * lens itself reads rasters directly; this stays for research results that point at one.
 */
export function LandRasterView({ id }: { id: string }) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const units = useSettings((s) => s.units);
  const shown = useLandContext((s) => s.rasters[id]);
  const tileError = useLandContext((s) => s.rasterErrors[id]);
  const [preferred, setPreferred] = useState<number | null>(null);
  const query = useQuery({
    queryKey: ["land-rasters", scope, id],
    queryFn: () =>
      unwrap(api.GET("/api/v1/land/rasters/{raster_id}", { params: { path: { raster_id: id } } })),
    enabled: ready,
    retry: false,
  });
  const raster = query.data;
  if (query.isPending)
    return (
      <p className="land-muted" role="status">
        Loading saved analysis…
      </p>
    );
  if (!raster)
    return (
      <div className="land-row" role="alert">
        The saved analysis could not be loaded.
        <GlassButton size="sm" onClick={() => void query.refetch()}>
          Try again
        </GlassButton>
      </div>
    );
  const kind = rasterKind(raster);
  // A signature's map shows its dated NDVI bands only; every band stays in the download.
  const choices = mapBands(raster);
  const band =
    shown?.band ??
    preferred ??
    (kind === "vegetation" || kind === "signature"
      ? defaultVegetationBand(raster)
      : (choices[0]?.index ?? 1));
  const definition = bandOf(raster, band);
  const show = (index: number) =>
    useLandContext.getState().setRaster({
      id,
      band: index,
      categorical: bandOf(raster, index)?.palette === "categorical",
      bounds: raster.metadata.bounds,
      opacity: shown?.opacity ?? 0.8,
      attribution: [...new Set(raster.metadata.sources.map((source) => source.attribution))].join(
        " · ",
      ),
    });
  const choose = (index: number) => {
    setPreferred(index);
    if (shown) show(index);
  };
  return (
    <section className="overview-raster-view" aria-label={LABEL[kind]}>
      <div className="land-row">
        <GlassButton
          size="sm"
          active={Boolean(shown)}
          disabled={!definition?.validCells}
          onClick={() => {
            if (shown) useLandContext.getState().removeRaster(id);
            else show(band);
          }}
        >
          {shown ? "Hide from map" : "Show on map"}
        </GlassButton>
        {choices.length > 1 && (
          <label className="land-field">
            Map measurement
            <GlassSelect value={band} onChange={(event) => choose(Number(event.target.value))}>
              {choices.map((item) => (
                <option key={item.index} value={item.index}>
                  {item.name} ({item.unit})
                </option>
              ))}
            </GlassSelect>
          </label>
        )}
      </div>
      {tileError && (
        <p className="land-alert" role="alert">
          {tileError}
        </p>
      )}
      <RasterDetails raster={raster} units={units} band={band} onChooseBand={choose} />
    </section>
  );
}

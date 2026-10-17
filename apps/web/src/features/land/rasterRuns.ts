/**
 * What each saved-raster analysis asks for and is called, in one table, so every button that
 * starts one (LandResearch now; the Show once and Rehearsal cards later) sends the same request.
 */
import type { components, RasterDataset } from "@twin/contracts";

type Period = components["schemas"]["VegetationPeriod"];
type RasterRequest = components["schemas"]["RasterRequest"];

export const RASTER_RUNS: Record<
  RasterDataset,
  {
    title: string;
    question: string;
    resolutionM: number;
    /** The run's time budget: at least the processor's limit (wall_seconds in
     * apps/api/app/analysis/raster_runner.py: 200 s signature, 165 s NDVI, 90 s otherwise)
     * plus its 15 s to save, or the worker stops the processor early. */
    maxSeconds: number;
  }
> = {
  "cop-dem-glo-30": {
    title: "Surface elevation and slope",
    question: "Analyze surface elevation and slope inside this land.",
    resolutionM: 30,
    maxSeconds: 180,
  },
  "esa-worldcover-2021": {
    title: "Land cover in 2021",
    question: "Analyze the broad land-cover classes mapped in 2021 inside this land.",
    resolutionM: 10,
    maxSeconds: 180,
  },
  "sentinel-2-ndvi": {
    title: "Vegetation through time",
    question: "Compare dated vegetation signals on common clear cells inside this land.",
    resolutionM: 20,
    maxSeconds: 240,
  },
  "sentinel-2-signature": {
    title: "Seasonal 10 m signatures",
    question: "Prepare seasonal 10 m signatures for this land.",
    resolutionM: 10,
    maxSeconds: 300,
  },
  "usgs-3dep-13": {
    title: "Bare-earth elevation and slope (USGS 3DEP)",
    question: "Analyze bare-earth elevation and slope inside this land.",
    resolutionM: 10,
    maxSeconds: 180,
  },
};

export function rasterRequest(dataset: RasterDataset, periods?: Period[]): RasterRequest {
  return {
    dataset,
    resolutionM: RASTER_RUNS[dataset].resolutionM,
    maxDimension: 512,
    ...(periods ? { periods } : {}),
  };
}

/** The signature's four seasons: 30-day windows centred on mid-January, April, July and
 * October of the latest complete year, so every window has already happened. */
export function signaturePeriods(today = new Date().toISOString().slice(0, 10)): Period[] {
  const year = Number(today.slice(0, 4)) - 1;
  return ["01", "04", "07", "10"].map((month) => ({
    startDate: `${year}-${month}-01`,
    endDate: `${year}-${month}-30`,
  }));
}

/** The rough United States extent the server's US-only sources answer for (covers_us in
 * apps/api/app/research/providers/us_land.py). A hint for the button; the server decides. */
export function coversUs(longitude: number, latitude: number): boolean {
  return (
    (longitude >= -180 && longitude <= -60 && latitude >= 17 && latitude <= 73) ||
    (longitude >= 140 && longitude <= 146 && latitude >= 12 && latitude <= 22)
  );
}

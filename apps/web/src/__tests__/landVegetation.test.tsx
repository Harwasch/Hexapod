import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components } from "@twin/contracts";
import { vegetationPeriods } from "@/features/land/vegetationPeriods";
import { OverviewPeriods } from "@/features/land/surface/lenses/overview/OverviewPeriods";
import { RasterDetails } from "@/features/land/surface/lenses/overview/RasterDetails";
import { defaultMonths } from "@/features/land/surface/lenses/overview/overviewData";

afterEach(cleanup);

it("creates chronological calendar windows, handles leap years and caps the current month at today", () => {
  expect(vegetationPeriods(["2026-10", "2024-02"], "2026-10-09")).toEqual([
    { startDate: "2024-02-01", endDate: "2024-02-29" },
    { startDate: "2026-10-01", endDate: "2026-10-09" },
  ]);
  for (const months of [["2024-02", "2024-02"], ["2027-01"], ["2015-06"], ["2024-13"], [""], []])
    expect(vegetationPeriods(months, "2026-10-09")).toBeNull();
});

it("starts from last month and the same month a year before", () => {
  expect(defaultMonths(new Date("2026-10-11T00:00:00Z"))).toEqual(["2025-09", "2026-09"]);
  expect(defaultMonths(new Date("2026-01-05T00:00:00Z"))).toEqual(["2024-12", "2025-12"]);
});

it("allows month editing and rejects duplicates before starting", () => {
  const onStart = vi.fn();
  render(<OverviewPeriods disabled={false} busy={false} onStart={onStart} />);
  fireEvent.change(screen.getByLabelText("Observation month 1"), { target: { value: "2024-07" } });
  fireEvent.change(screen.getByLabelText("Observation month 2"), { target: { value: "2025-07" } });
  fireEvent.click(screen.getByRole("button", { name: "Compare these months" }));
  expect(onStart).toHaveBeenCalledWith([
    { startDate: "2024-07-01", endDate: "2024-07-31" },
    { startDate: "2025-07-01", endDate: "2025-07-31" },
  ]);
  fireEvent.change(screen.getByLabelText("Observation month 2"), { target: { value: "2024-07" } });
  expect(screen.getByRole("button", { name: "Compare these months" })).toBeDisabled();
  expect(screen.getByRole("status")).toHaveTextContent(/different months/);
  fireEvent.click(screen.getByRole("button", { name: "Remove observation month 2" }));
  expect(screen.getByRole("button", { name: "Compare these months" })).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Add a month" }));
  expect(screen.getByLabelText("Observation month 2")).toHaveValue("");
});

const series: components["schemas"]["VegetationTimeline"] = {
  observations: [2024, 2025].map((year, index) => ({
    period: { startDate: `${year}-07-01`, endDate: `${year}-07-31` },
    band: index + 1,
    sceneId: `scene-${year}`,
    acquiredAt: `${year}-07-16T12:00:00Z`,
    candidateCount: 25,
    catalogTruncated: true,
    qualityCandidatesExamined: 3,
    validCells: index ? 80 : 100,
    coverageFraction: index ? 0.8 : 1,
    meanNdvi: index ? 0.65 : 0.45,
    commonMeanNdvi: index ? 0.6 : 0.5,
    qualityCounts: { Vegetation: 80, "Cloud shadow": 20 },
    explanation: "One selected acquisition.",
  })),
  commonCells: 80,
  commonCoverageFraction: 0.8,
  changeBand: 3,
  changeCells: 80,
  meanChange: 0.1,
};

function ndviBand(index: number): components["schemas"]["RasterBand"] {
  return {
    index,
    name: `NDVI band ${index}`,
    unit: "NDVI",
    validCells: 80,
    minimum: 0.1,
    maximum: 0.9,
    mean: 0.5,
    standardDeviation: 0.1,
    percentiles: { "50": 0.5 },
    histogramEdges: [0.1, 0.5, 0.9],
    histogramCounts: [40, 40],
    palette: index === 3 ? "rdylgn" : "viridis",
  };
}

const raster: components["schemas"]["LandRasterRead"] = {
  id: "ndvi",
  landId: "land",
  runId: "run",
  boundaryRevision: 1,
  request: { dataset: "sentinel-2-ndvi" },
  sha256: "a".repeat(64),
  byteSize: 100,
  stale: false,
  createdAt: "2026-10-09T12:00:00Z",
  metadata: {
    algorithm: "ndvi-v1",
    method: "Synthetic fixture",
    bounds: [-77.055, 38.886, -77.045, 38.891],
    crs: "EPSG:4326",
    resolutionM: 20,
    width: 4,
    height: 4,
    boundaryCells: 100,
    validCells: 80,
    coverageFraction: 0.8,
    sampledAreaM2: 7200,
    downloadedBytes: 100,
    warnings: [],
    sources: [],
    bands: [ndviBand(1), ndviBand(2), ndviBand(3)],
    vegetation: series,
  },
};

it("shows shared-cell values apart from each month's own and links months to map bands", () => {
  const choose = vi.fn();
  render(<RasterDetails raster={raster} units="metric" band={1} onChooseBand={choose} />);
  expect(screen.getByRole("img", { name: /seen clearly every time/ })).toBeVisible();
  // The first month's value on shared cells differs from its value on all of its own clear cells.
  expect(screen.getAllByText("Greenness, shared cells")[0]!.parentElement).toHaveTextContent(
    "0.50",
  );
  expect(screen.getAllByText("Greenness, all clear cells")[0]!.parentElement).toHaveTextContent(
    "0.45",
  );
  expect(screen.getByText(/80 cells were seen clearly every time/)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "2025-07 observation" }));
  expect(choose).toHaveBeenCalledWith(2);
  fireEvent.click(screen.getByRole("button", { name: "View vegetation change" }));
  expect(choose).toHaveBeenCalledWith(3);
  expect(screen.getByText("+0.10")).toBeVisible();
  fireEvent.click(screen.getAllByText("Acquisition and quality")[0]!);
  expect(screen.getAllByText(/catalog truncated/)[0]).toBeVisible();
  expect(screen.getAllByText("Cloud shadow: 20 cells")[0]).toBeVisible();
});

it("shows a missing comparison without drawing a zero-valued trend", () => {
  const missing = {
    ...series,
    commonCells: 0,
    commonCoverageFraction: 0,
    changeCells: 0,
    meanChange: null,
    observations: series.observations.map((o) => ({ ...o, commonMeanNdvi: null })),
  };
  render(
    <RasterDetails
      raster={{ ...raster, metadata: { ...raster.metadata, vegetation: missing } }}
      units="metric"
      band={1}
      onChooseBand={vi.fn()}
    />,
  );
  expect(screen.queryByRole("img", { name: /seen clearly every time/ })).not.toBeInTheDocument();
  expect(screen.getByText(/No comparable trend/)).toBeVisible();
  expect(screen.getByRole("button", { name: "View vegetation change" })).toBeDisabled();
});

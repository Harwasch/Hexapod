import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components } from "@twin/contracts";
import { VegetationStart } from "@/features/land/VegetationStart";
import { VegetationTimelineView } from "@/features/land/VegetationTimelineView";
import { vegetationPeriods } from "@/features/land/vegetationPeriods";

afterEach(cleanup);
it("creates chronological calendar windows, handles leap years and caps the current month at today", () => {
  expect(vegetationPeriods(["2026-10", "2024-02"], "2026-10-09")).toEqual([
    { startDate: "2024-02-01", endDate: "2024-02-29" },
    { startDate: "2026-10-01", endDate: "2026-10-09" },
  ]);
  for (const months of [["2024-02", "2024-02"], ["2027-01"], ["2015-06"], ["2024-13"], [""], []])
    expect(vegetationPeriods(months, "2026-10-09")).toBeNull();
});
it("allows month editing and rejects duplicates before creating a saved investigation", () => {
  const onStart = vi.fn();
  render(<VegetationStart disabled={false} busy={false} onStart={onStart} />);
  fireEvent.click(screen.getByText("Choose observation months"));
  fireEvent.change(screen.getByLabelText("Observation month 1"), { target: { value: "2024-07" } });
  fireEvent.change(screen.getByLabelText("Observation month 2"), { target: { value: "2025-07" } });
  fireEvent.click(screen.getByRole("button", { name: "Compare vegetation" }));
  expect(onStart).toHaveBeenCalledWith([
    { startDate: "2024-07-01", endDate: "2024-07-31" },
    { startDate: "2025-07-01", endDate: "2025-07-31" },
  ]);
  fireEvent.change(screen.getByLabelText("Observation month 2"), { target: { value: "2024-07" } });
  expect(screen.getByRole("button", { name: "Compare vegetation" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Remove observation month 2" }));
  expect(screen.getByRole("button", { name: "Compare vegetation" })).toBeEnabled();
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
it("shows common-cell values separately from changing observation footprints and links dates to map bands", () => {
  const choose = vi.fn();
  render(<VegetationTimelineView series={series} onChoose={choose} band={1} />);
  expect(screen.getByRole("img")).toHaveAccessibleName(/common clear land cells/);
  expect(screen.getByText("0.500")).toBeVisible();
  expect(screen.getByText("0.450")).toBeVisible();
  expect(screen.getByText(/80 clear cells shared/)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "2025-07 observation" }));
  expect(choose).toHaveBeenCalledWith(2);
  fireEvent.click(screen.getByRole("button", { name: "View vegetation change" }));
  expect(choose).toHaveBeenCalledWith(3);
  fireEvent.click(screen.getAllByText("Acquisition and quality")[0]!);
  expect(screen.getAllByText(/catalog truncated/)[0]).toBeVisible();
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
  render(<VegetationTimelineView series={missing} onChoose={vi.fn()} band={1} />);
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  expect(screen.getByText(/No comparable trend/)).toBeVisible();
  expect(screen.getByRole("button", { name: "View vegetation change" })).toBeDisabled();
});

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { ResearchArtifact } from "@twin/contracts";
import { CalculationView } from "@/features/land/CalculationView";

const output: Extract<ResearchArtifact["output"], { kind: "calculation" }> = {
  kind: "calculation",
  engineVersion: "arithmetic-v1",
  requestSha256: "a".repeat(64),
  request: {
    purpose: "Compare hypothetical planting costs.",
    limitations: "Synthetic assumptions, not a field estimate.",
    rowLabels: Array.from({ length: 30 }, (_, index) => `Case ${index + 1}`),
    inputs: [
      {
        name: "area",
        label: "Area",
        unit: "ha",
        origin: "assumption",
        basis: "Hypothetical test cases.",
        values: Array.from({ length: 30 }, (_, index) => (index === 1 ? null : index)),
      },
      {
        name: "rate",
        label: "Rate",
        unit: "USD/ha",
        origin: "evidence",
        basis: "Transcribed from a fixture source.",
        values: [1200],
        evidenceIds: ["source"],
      },
    ],
    formulas: [{ name: "cost", label: "Planting cost", unit: "USD", expression: "area * rate" }],
  },
  rows: Array.from({ length: 30 }, (_, index) => ({ cost: index === 1 ? null : index * 1200 })),
  issues: [
    { row: 1, column: "cost", kind: "missing", message: "Missing input or prior result: area" },
  ],
};
const artifact: ResearchArtifact = {
  id: "calculation",
  runId: "run",
  title: "Comparison",
  method: "Recomputed",
  evidenceIds: ["source"],
  output,
};
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("keeps zeroes and gaps distinct and pages input, result and chart values together", () => {
  render(<CalculationView artifact={artifact} output={output} />);
  const table = screen.getByRole("region", { name: "Calculated results" });
  expect(within(table).getAllByRole("row")).toHaveLength(26);
  expect(within(table).getByRole("cell", { name: "0" })).toBeVisible();
  expect(within(table).getByText("No result")).toBeVisible();
  expect(within(table).getByRole("columnheader", { name: "Planting cost (USD)" })).toBeVisible();
  fireEvent.click(screen.getByText("Explain gaps on this page (1)"));
  expect(screen.getByText(/Missing input or prior result: area/)).toBeVisible();
  fireEvent.click(screen.getByText("Inputs, assumptions and formulas"));
  expect(screen.getByText("Assumption", { exact: true })).toBeVisible();
  expect(screen.getByText("cost = area * rate")).toBeVisible();
  fireEvent.change(screen.getByRole("combobox", { name: "Plot a result" }), {
    target: { value: "cost" },
  });
  expect(screen.getByRole("img")).toHaveAccessibleName(/Planting cost, USD/);
  fireEvent.click(screen.getByRole("button", { name: "Next results" }));
  expect(screen.getByText("Rows 26–30 of 30")).toBeVisible();
  expect(within(table).getAllByRole("row")).toHaveLength(6);
  expect(
    within(screen.getByRole("region", { name: "Area inputs" })).getByText("Case 26"),
  ).toBeVisible();
  expect(within(table).queryByText("Case 1")).not.toBeInTheDocument();
  expect(screen.queryByText("Explain gaps on this page (1)")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Next results" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Previous results" }));
  expect(screen.getByRole("button", { name: "Previous results" })).toBeDisabled();
});

it("opens the exact input source and exports all rows with original precision and recipe", async () => {
  const onEvidence = vi.fn();
  let exported: Blob | undefined;
  vi.stubGlobal("URL", {
    createObjectURL: (blob: Blob) => {
      exported = blob;
      return "blob:calculation";
    },
    revokeObjectURL: vi.fn(),
  });
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  render(<CalculationView artifact={artifact} output={output} onEvidence={onEvidence} />);
  fireEvent.click(screen.getByText("Inputs, assumptions and formulas"));
  fireEvent.click(screen.getByRole("button", { name: "Inspect input source 1" }));
  expect(onEvidence).toHaveBeenCalledWith("source");
  fireEvent.click(screen.getByRole("button", { name: "Download calculation" }));
  expect(click).toHaveBeenCalledOnce();
  const data = await new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : "");
    reader.readAsText(exported!);
  });
  expect(JSON.parse(data)).toEqual(artifact);
});

it("reports a download failure without losing the calculation", () => {
  vi.stubGlobal("URL", {
    createObjectURL: () => {
      throw new Error("Unavailable");
    },
  });
  render(<CalculationView artifact={artifact} output={output} />);
  fireEvent.click(screen.getByRole("button", { name: "Download calculation" }));
  expect(screen.getByRole("alert")).toHaveTextContent("could not be downloaded");
  expect(screen.getByRole("region", { name: "Calculated results" })).toBeVisible();
});

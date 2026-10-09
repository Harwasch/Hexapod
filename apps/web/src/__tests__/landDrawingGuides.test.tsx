import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { LandCandidatePicker } from "@/features/land/LandCandidatePicker";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { api } from "@/api/client";

afterEach(() => {
  cleanup();
  useLand.getState().clear();
  useLandContext.getState().clear();
  vi.restoreAllMocks();
});

it("keeps only reviewed candidates as drawing guides without writing a boundary", () => {
  const post = vi.spyOn(api, "POST");
  useLand.getState().begin("candidates");
  const geometry = {
    type: "LineString" as const,
    coordinates: [
      [-77.052, 38.888],
      [-77.048, 38.888],
    ],
  };
  useLandContext.getState().setCandidates([
    {
      id: "line-a",
      label: "Public fixture line",
      geometry,
      distanceM: 0,
      properties: {},
      source: { method: "mapped-feature", meaning: "physical-feature", label: "Synthetic fixture" },
    },
    {
      id: "line-b",
      label: "Other fixture",
      geometry,
      distanceM: 1,
      properties: {},
      source: { method: "mapped-feature", meaning: "physical-feature", label: "Synthetic fixture" },
    },
  ]);
  render(<LandCandidatePicker />);
  fireEvent.click(screen.getByRole("button", { name: /Public fixture line/ }));
  fireEvent.click(screen.getByRole("button", { name: "Trace using selected features" }));
  expect(useLand.getState().mode).toBe("corridor");
  expect(useLand.getState().points).toEqual([]);
  expect(useLand.getState().draft).toBeNull();
  expect(useLand.getState().traceSources[0]?.label).toBe("Synthetic fixture");
  expect(useLandContext.getState().layers["drawing-guides"]?.features).toEqual([
    { id: "line-a", label: "Public fixture line", geometry },
  ]);
  expect(post).not.toHaveBeenCalled();
  useLandContext.getState().removeLayer("drawing-guides");
  expect(useLand.getState().traceSources[0]?.label).toBe("Synthetic fixture");
});

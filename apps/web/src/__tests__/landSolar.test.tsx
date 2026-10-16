import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { SolarStudy } from "@/features/land/SolarStudy";
import {
  parseSolarDraft,
  solarFinance,
  solarRequest,
  type SolarAssessment,
} from "@/features/land/solarStudy";
import { useLandContext } from "@/state/landContext";
const land: LandArea = {
  id: "11111111-1111-4111-8111-111111111111",
  name: "Public fixture",
  description: "",
  revision: 1,
  areaM2: 10000,
  perimeterM: 400,
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-77.055, 38.886],
        [-77.045, 38.886],
        [-77.045, 38.891],
        [-77.055, 38.886],
      ],
    ],
  },
  source: { method: "drawn", label: "Synthetic" },
  createdAt: "2026-10-09",
  updatedAt: "2026-10-09",
};
function mount() {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}
    >
      <SolarStudy land={land} onUse={() => undefined} />
    </QueryClientProvider>,
  );
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
  useLandContext.getState().clear();
});
it("preserves blank numeric inputs in recoverable drafts", () => {
  const draft = solarRequest(land.boundary);
  draft.moduleAreaM2 = Number.NaN;
  expect(parseSolarDraft(JSON.stringify(draft)).moduleAreaM2).toBeNaN();
  expect(() =>
    parseSolarDraft(
      JSON.stringify({ ...draft, arrayZone: { type: "Point", coordinates: [0, 0] } }),
    ),
  ).toThrow();
  expect(() =>
    parseSolarDraft(JSON.stringify({ ...draft, horizon: [{ azimuthDegrees: "bad" }] })),
  ).toThrow();
});
it("links physical assumptions without fabricating an equivalent irradiation or accepting a partial year", () => {
  // Only fields consumed by the mapping; server completeness and ownership are separately validated.
  const a = {
    request: solarRequest(land.boundary),
    metadata: {
      completeYear: true,
      annualGenerationKwh: 12345,
      unshadedPlaneIrradiationKwhM2: 1456,
    },
  } as SolarAssessment;
  const finance = solarFinance(a);
  expect(finance.annualPlaneIrradiationKwhM2).toBe(1456);
  expect(finance.shadeLoss).toBe(a.request.additionalShadeLoss);
  expect(finance.systemLoss).toBe(a.request.systemLoss);
  expect(() =>
    solarFinance({
      ...a,
      metadata: { ...a.metadata, completeYear: false, annualGenerationKwh: null },
    }),
  ).toThrow(/complete/);
});
it("invalidates review on edits and restores unfinished assumptions after reload", async () => {
  vi.spyOn(api, "GET").mockResolvedValue({ data: [], response: new Response() });
  const post = vi.spyOn(api, "POST").mockResolvedValue({
    data: {
      mappedZoneAreaM2: 10000,
      capacityKwDc: 20,
      inverterKwAc: 16.67,
      longitude: -77.05,
      latitude: 38.889,
    },
    response: new Response(),
  });
  const view = mount();
  fireEvent.click(screen.getByRole("button", { name: "New hourly solar assessment" }));
  fireEvent.click(screen.getByRole("button", { name: "Use land boundary" }));
  fireEvent.click(screen.getByRole("button", { name: "Review array" }));
  await screen.findByRole("button", { name: "Run hourly analysis" });
  fireEvent.change(screen.getByLabelText("Total module face area (m²)"), {
    target: { value: "125" },
  });
  expect(screen.queryByRole("button", { name: "Run hourly analysis" })).toBeNull();
  fireEvent.change(screen.getByLabelText("Tilt above horizontal (degrees)"), {
    target: { value: "" },
  });
  expect(post).toHaveBeenCalledTimes(1);
  view.unmount();
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Recover solar draft" }));
  expect(screen.getByLabelText("Total module face area (m²)")).toHaveValue(125);
  expect(screen.getByLabelText("Tilt above horizontal (degrees)")).toHaveValue(null);
  expect(screen.queryByRole("button", { name: "Run hourly analysis" })).toBeNull();
});

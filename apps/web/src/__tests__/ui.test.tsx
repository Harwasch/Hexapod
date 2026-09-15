import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { InspectorPanel } from "@/features/inspector/InspectorPanel";
import { LayersPanel } from "@/features/layers/LayersPanel";
import { OnboardingCard } from "@/features/onboarding/OnboardingCard";
import { CommandBar } from "@/features/mission/CommandBar";
import { ToolRail } from "@/features/shell/ToolRail";
import { useSelection } from "@/state/selection";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";
import { useViewer } from "@/state/viewer";

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  useUi.setState({ activePanel: null, inspectorOpen: false });
  useSelection.setState({ selection: null });
  useSettings.getState().reset();
});

describe("ToolRail + panels", () => {
  it("opens the layers panel and shows the built-in catalog when the API is offline", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("Failed to fetch"));
    render(
      wrap(
        <>
          <ToolRail />
          <LayersPanel />
        </>,
      ),
    );
    await userEvent.click(screen.getByRole("button", { name: "Layers" }));
    expect(await screen.findByTestId("layers-panel")).toBeInTheDocument();
    expect(await screen.findByText("Built-in catalog (API offline)")).toBeInTheDocument();
    expect(await screen.findByTestId("layer-card-openstreetmap")).toBeInTheDocument();
    await userEvent.type(screen.getByTestId("layers-filter"), "terrain");
    await waitFor(() =>
      expect(screen.queryByTestId("layer-card-openstreetmap")).not.toBeInTheDocument(),
    );
    expect(screen.getByTestId("layer-card-cesium-world-terrain")).toBeInTheDocument();
    vi.restoreAllMocks();
  });
});

describe("InspectorPanel", () => {
  it("opens when a selection exists and shows position and attribution", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    render(wrap(<InspectorPanel />));
    expect(screen.queryByTestId("inspector")).not.toBeInTheDocument();
    useSelection.getState().setSelection({
      kind: "ground",
      title: "Location",
      longitude: -122.138,
      latitude: 47.644,
      height: 120,
      terrainHeight: 118.5,
      attribution: [{ text: "Cesium World Terrain", organization: null, url: null }],
      at: Date.now(),
    });
    useUi.getState().setInspectorOpen(true);
    expect(await screen.findByTestId("inspector")).toBeInTheDocument();
    expect(screen.getByTestId("inspector-position")).toHaveTextContent("47.64400° N, 122.13800° W");
    expect(screen.getByText("Cesium World Terrain")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Close panel" }));
    await waitFor(() => expect(screen.queryByTestId("inspector")).not.toBeInTheDocument());
    vi.restoreAllMocks();
  });
});

describe("CommandBar + Onboarding", () => {
  it("renders altitude in the selected unit system", () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useViewer.getState().setCamera({
      ...useViewer.getState().camera,
      altitude: 1500,
      scaleBand: "city",
      metersPerPixel: 2,
    });
    const { rerender } = render(wrap(<CommandBar />));
    expect(screen.getByTestId("status-altitude")).toHaveTextContent("1.5 km");
    useSettings.getState().set({ units: "imperial" });
    rerender(wrap(<CommandBar />));
    expect(screen.getByTestId("status-altitude")).toHaveTextContent("ft");
  });

  it("onboarding appears once the viewer is ready and stays dismissed", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useViewer.getState().setStatus("ready");
    render(wrap(<OnboardingCard />));
    expect(await screen.findByTestId("onboarding")).toBeInTheDocument();
    await userEvent.click(screen.getByTestId("onboarding-explore"));
    await waitFor(() => expect(screen.queryByTestId("onboarding")).not.toBeInTheDocument());
    expect(useSettings.getState().onboardingDismissed).toBe(true);
    vi.restoreAllMocks();
  });
});

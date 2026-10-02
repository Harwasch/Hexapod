import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { AddPanel } from "@/features/add-data/AddPanel";
import { SettingsSheet } from "@/features/settings/SettingsSheet";
import { useRendererOverride, useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
  useSettings.getState().reset();
});
afterEach(() => {
  vi.restoreAllMocks();
  useUi.setState({ settingsOpen: false, activePanel: null, addTab: "upload" });
});

describe("Settings › Advanced", () => {
  it("holds the developer readouts and the splat renderer, folded until asked for", async () => {
    const user = userEvent.setup();
    useUi.setState({ settingsOpen: true });
    render(wrap(<SettingsSheet />));
    const advanced = screen.getByTestId("settings-advanced");
    expect(advanced).not.toHaveAttribute("open");
    await user.click(within(advanced).getByText("Advanced"));
    expect(advanced).toHaveAttribute("open");

    const readouts = within(advanced).getByRole("switch", { name: "Show developer readouts" });
    expect(readouts).toHaveAttribute("aria-checked", "false");
    await user.click(readouts);
    expect(useSettings.getState().devReadouts).toBe(true);

    const renderer = within(advanced).getByRole("radiogroup", { name: "Splat renderer" });
    await user.click(within(renderer).getByRole("radio", { name: "Draw splats with Spark" }));
    expect(useSettings.getState().splatRenderer).toBe("spark");
    await user.click(
      within(renderer).getByRole("radio", {
        name: "Draw splats with PlayCanvas on WebGPU (beta)",
      }),
    );
    expect(useSettings.getState().splatRenderer).toBe("playcanvas-webgpu");
  });

  it("shows the renderer the page address chose, and a choice here ends it", async () => {
    const user = userEvent.setup();
    useRendererOverride.setState({ renderer: "playcanvas-webgpu" });
    useUi.setState({ settingsOpen: true });
    render(wrap(<SettingsSheet />));
    const advanced = screen.getByTestId("settings-advanced");
    await user.click(within(advanced).getByText("Advanced"));
    const renderer = within(advanced).getByRole("radiogroup", { name: "Splat renderer" });
    const webgpu = within(renderer).getByRole("radio", {
      name: "Draw splats with PlayCanvas on WebGPU (beta)",
    });
    expect(webgpu).toHaveAttribute("aria-checked", "true");
    expect(within(advanced).getByTestId("splat-renderer-now")).toHaveTextContent(
      "Chosen by the page address for this visit",
    );
    await user.click(within(renderer).getByRole("radio", { name: "Draw splats with PlayCanvas" }));
    expect(useRendererOverride.getState().renderer).toBeNull();
    expect(useSettings.getState().splatRenderer).toBe("playcanvas");
    expect(within(advanced).queryByTestId("splat-renderer-now")).not.toBeInTheDocument();
  });
});

describe("Add", () => {
  it("leads with the upload and keeps the hosted-source forms on the second tab", async () => {
    const user = userEvent.setup();
    useUi.getState().openAdd("upload");
    render(wrap(<AddPanel />));
    const panel = screen.getByTestId("add-panel");
    const tabs = within(panel).getByRole("radiogroup", { name: "What to add" });
    expect(within(tabs).getByRole("radio", { name: /Upload a capture/ })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    // The uploader: the drop zone and the phone handoff, then the list.
    expect(within(panel).getByTestId("captures-panel")).toBeInTheDocument();
    expect(within(panel).getByTestId("capture-from-phone")).toBeInTheDocument();
    expect(within(panel).queryByTestId("site-form")).not.toBeInTheDocument();

    await user.click(within(tabs).getByRole("radio", { name: /Link a source/ }));
    expect(useUi.getState().addTab).toBe("link");
    expect(within(panel).getByTestId("site-form")).toBeInTheDocument();
    await user.click(within(panel).getByTestId("add-tab-stac"));
    expect(within(panel).getByTestId("add-tab-stac")).toHaveAttribute("aria-selected", "true");
  });
});

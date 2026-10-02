import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { AddDataSheet } from "@/features/add-data/AddDataSheet";
import { SettingsSheet } from "@/features/settings/SettingsSheet";
import { useSettings } from "@/state/settings";
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
  useUi.setState({ settingsOpen: false, addDataOpen: false });
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
  });
});

describe("Add data", () => {
  it("leads with the upload and folds the hosted-source forms away", async () => {
    const user = userEvent.setup();
    useUi.setState({ addDataOpen: true });
    render(wrap(<AddDataSheet />));
    const sheet = screen.getByTestId("add-data");
    expect(within(sheet).getByTestId("add-data-upload")).toHaveTextContent(
      "Upload a video, photos or a splat",
    );
    const link = within(sheet).getByTestId("add-link");
    expect(link).not.toHaveAttribute("open");
    await user.click(within(sheet).getByTestId("add-link-toggle"));
    expect(link).toHaveAttribute("open");
    await user.click(within(sheet).getByTestId("add-tab-stac"));
    expect(within(sheet).getByTestId("add-tab-stac")).toHaveAttribute("aria-selected", "true");
  });
});

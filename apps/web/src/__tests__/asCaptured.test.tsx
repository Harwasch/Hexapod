/**
 * As captured (state/asCaptured.ts, ADR 0019): one switch under which nothing simulated or
 * inferred is drawn, read in one place per effect, never writing a viewer's own choice; asked
 * for by a page address; said in the HUD strip and switched in the Splat view control.
 * The scene's effects are tested beside the code they change (inferred.test.ts,
 * scanLayers.test.ts, supersedes.test.ts, compositorSession.test.ts, telemetry.test.ts,
 * rehearsalManager.test.ts); this file holds the store, the poses and the controls.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { queryKeys } from "@/api/queries";
import { CompareMethodsPanel } from "@/features/sites/CompareMethods";
import { InferredStyleControl } from "@/features/sites/InferredStyle";
import { ViewsSlot } from "@/features/views/ViewsSlot";
import { AS_CAPTURED_TEXT } from "@/lib/asCapturedText";
import { REST_POSE, type SplitObjectRef } from "@/lib/sceneObjects";
import { variantsOf } from "@/lib/variants";
import {
  effectiveInferredStyle,
  HIDDEN_AS_CAPTURED,
  onInferredStyle,
  useAsCaptured,
} from "@/state/asCaptured";
import { objectPose, useSceneObjects } from "@/state/sceneObjects";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useSkinPoke } from "@/state/skinPoke";
import { useVariants } from "@/state/variants";

import { SITE_ID, twoEpochSite } from "./epochFixtures";

afterEach(() => {
  cleanup();
  useAsCaptured.getState().set(false);
  useSettings.getState().reset();
  useSkinPoke.setState({ active: false, grabbed: null });
  useSites.setState({ activeSiteId: null, representation: {}, epoch: {} });
});

describe("the store", () => {
  it("draws Inferred as Hide while on, and never writes the viewer's choice", () => {
    useSettings.getState().set({ inferredStyle: "highlight" });
    const heard: string[] = [];
    const off = onInferredStyle((style) => heard.push(style));
    expect(effectiveInferredStyle()).toBe("highlight");
    useAsCaptured.getState().set(true);
    expect(effectiveInferredStyle()).toBe("hide");
    expect(useSettings.getState().inferredStyle).toBe("highlight");
    // A change of the viewer's own style meanwhile is kept, and draws nothing new.
    useSettings.getState().set({ inferredStyle: "show" });
    expect(effectiveInferredStyle()).toBe("hide");
    useAsCaptured.getState().set(false);
    expect(effectiveInferredStyle()).toBe("show");
    expect(heard).toEqual(["hide", "show"]);
    // Off with Hide chosen: nothing changes, so nothing is said.
    useSettings.getState().set({ inferredStyle: "hide" });
    useAsCaptured.getState().set(true);
    useAsCaptured.getState().set(false);
    expect(heard).toEqual(["hide", "show", "hide"]);
    off();
  });

  it("turns the poke tool off, and straight off again if it is asked for meanwhile", () => {
    useSkinPoke.getState().setActive(true);
    useAsCaptured.getState().set(true);
    expect(useSkinPoke.getState().active).toBe(false);
    useSkinPoke.getState().toggle();
    expect(useSkinPoke.getState().active).toBe(false);
    useAsCaptured.getState().set(false);
    useSkinPoke.getState().toggle();
    expect(useSkinPoke.getState().active).toBe(true);
  });
});

describe("split objects", () => {
  const ref: SplitObjectRef = {
    uri: "objects/3/tileset.json",
    instance: 3,
    origin: [2, 1, 0],
    pose: { translation: [0.5, 0, 0], rotation: [0, 0, 0, 1] },
    splats: 10,
    fill: "fills/3/tileset.json",
  };
  const moved = { translation: [0, 2, 0] as const, rotation: [0, 0, 0, 1] as const };

  it("are drawn at rest, where they were measured, while As captured is on", () => {
    useSceneObjects.getState().setPose("scan", 3, moved);
    expect(objectPose("scan", ref)).toEqual(moved);
    expect(objectPose("other", ref)).toBe(ref.pose);
    expect(objectPose(undefined, ref)).toBe(ref.pose);
    useAsCaptured.getState().set(true);
    expect(objectPose("scan", ref)).toBe(REST_POSE);
    expect(objectPose(undefined, ref)).toBe(REST_POSE);
    // The pose set is kept for after.
    useAsCaptured.getState().set(false);
    expect(objectPose("scan", ref)).toEqual(moved);
    useSceneObjects.getState().setPose("scan", 3, null);
  });
});

const EVIDENCE = {
  kind: "inferred",
  filler: "fixture",
  views: 2,
  gaussians: 9,
  meanConfidence: 0.5,
};

describe("the controls of what it hides", () => {
  it("disables Inferred's Show · Highlight · Hide, keeping the choice for after", () => {
    useSettings.getState().set({ inferredStyle: "show" });
    useAsCaptured.getState().set(true);
    render(<InferredStyleControl evidence={[EVIDENCE as never]} />);
    const group = screen.getByRole("radiogroup", { name: "Inferred fill" });
    for (const radio of within(group).getAllByRole("radio")) expect(radio).toBeDisabled();
    expect(within(group).getByRole("radio", { name: "Show inferred fill" })).toBeChecked();
    expect(screen.getByTestId("inferred-style")).toHaveTextContent(HIDDEN_AS_CAPTURED);
    act(() => useAsCaptured.getState().set(false));
    for (const radio of within(group).getAllByRole("radio")) expect(radio).toBeEnabled();
    expect(screen.getByTestId("inferred-style")).not.toHaveTextContent(HIDDEN_AS_CAPTURED);
  });

  it("holds Compare methods' Fill row, and leaves the other systems' rows", () => {
    useVariants.setState({
      offered: {
        scan: {
          ...variantsOf({
            variants: {
              fill: [
                {
                  name: "mound",
                  label: "Mound",
                  about: "A mound.",
                  inferredLayers: [{ uri: "m/tileset.json", evidence: EVIDENCE }],
                },
              ],
              skins: [{ name: "stiff", label: "Stiff", about: "Stiffness.", skin: "a.json" }],
            },
          }),
          today: { objects: false, fill: false, skins: true },
        },
      },
      picks: {},
      status: {},
    });
    useAsCaptured.getState().set(true);
    render(<CompareMethodsPanel assetId="scan" />);
    const fill = screen.getByRole("group", { name: "Fill" });
    for (const radio of within(fill).getAllByRole("radio")) expect(radio).toBeDisabled();
    expect(fill).toHaveTextContent(HIDDEN_AS_CAPTURED);
    const motion = screen.getByRole("group", { name: "Motion" });
    for (const radio of within(motion).getAllByRole("radio")) expect(radio).toBeEnabled();
    expect(useSettings.getState().inferredStyle).toBe("hide");
  });
});

function showViews() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(queryKeys.site(SITE_ID), twoEpochSite());
  useSettings.getState().set({ splatRenderer: "playcanvas" });
  useSites.setState({
    activeSiteId: SITE_ID,
    representation: { [SITE_ID]: "gaussian-splat" },
    epoch: {},
  });
  return render(
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>
        <ViewsSlot />
      </GlassTooltipProvider>
    </QueryClientProvider>,
  );
}

describe("the switch and the chip", () => {
  it("switches in the Splat view control under any renderer, and the chip says it with the way back", async () => {
    const user = userEvent.setup();
    showViews();
    const toggle = await screen.findByRole("switch", { name: AS_CAPTURED_TEXT.label });
    expect(toggle).toBeEnabled();
    expect(screen.queryByTestId("as-captured-chip")).toBeNull();
    await user.click(toggle);
    expect(useAsCaptured.getState().on).toBe(true);
    expect(toggle).toHaveAttribute("aria-checked", "true");
    const chip = await screen.findByTestId("as-captured-chip");
    expect(chip).toHaveTextContent("As captured — only what was measured is drawn; plans stay as annotations");
    expect(chip).toHaveAttribute("role", "status");
    await user.click(within(chip).getByRole("button", { name: AS_CAPTURED_TEXT.off }));
    expect(useAsCaptured.getState().on).toBe(false);
    expect(screen.queryByTestId("as-captured-chip")).toBeNull();
    expect(toggle).toHaveAttribute("aria-checked", "false");
  });

  it("shows the chip on any site while on, a splat scan or not", async () => {
    useAsCaptured.getState().set(true);
    // No active site, so no Splat view control: the chip alone.
    render(<ViewsSlot />);
    expect(await screen.findByTestId("as-captured-chip")).toBeVisible();
  });
});

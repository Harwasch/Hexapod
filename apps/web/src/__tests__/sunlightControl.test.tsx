/**
 * Sunlight in the HUD (ADR 0019): the Sun control in the splat view control, the badge that
 * says the scan is relit and over what, and "At capture" holding the scene clock at the shown
 * date's capture time while it is drawn -- given back whatever ends it.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Clock, JulianDate } from "cesium";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import { queryKeys } from "@/api/queries";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { SceneContext, SceneRegistry } from "@/cesium/SceneContext";
import { RelitSlot, ViewsSlot } from "@/features/views/ViewsSlot";
import { captureTimeOf, relitSentence, SUNLIGHT_TEXT } from "@/lib/sunlightText";
import { useAsCaptured } from "@/state/asCaptured";
import { useLighting } from "@/state/lighting";
import { useSettings, type SplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useSunReport } from "@/state/sunReport";
import { useViewRule } from "@/state/viewRule";

import { A1, A2, E1, E2, epoch, SITE_ID, splat, twoEpochSite } from "./epochFixtures";

const iso = (t: JulianDate) => JulianDate.toIso8601(t, 3);

describe("the capture time", () => {
  it("is the shown date's, unless it was only the upload's", () => {
    expect(captureTimeOf(A2, [E1, E2])).toBe("2026-06-01T15:00:00Z");
    const uploaded = epoch("e3", { observedAtSource: "created", currentAssetId: "a3" });
    expect(captureTimeOf(splat("a3", { epochId: "e3" }), [uploaded])).toBeNull();
    expect(captureTimeOf(splat("a4", { observedAt: "2025-05-05T10:00:00Z" }), [])).toBe(
      "2025-05-05T10:00:00Z",
    );
    expect(captureTimeOf(null, [])).toBeNull();
  });

  it("is said in UTC in the badge, or said to be unknown", () => {
    expect(relitSentence("now", "2026-06-01T15:00:00Z")).toMatch(
      /^Relit by the sun \(now\) over shadows baked in at capture, .*2026.* UTC$/,
    );
    expect(relitSentence("at capture", null)).toBe(
      "Relit by the sun (at capture) over shadows baked in at capture, time not recorded",
    );
  });
});

/** A scene manager with what the badge reads: the widget's clock and a render request. */
function fakeScene() {
  const clock = new Clock({
    currentTime: JulianDate.fromIso8601("2026-10-20T12:00:00Z"),
    shouldAnimate: true,
  });
  let renders = 0;
  const manager = {
    viewer: { clock },
    scene: { requestRender: () => (renders += 1) },
  } as unknown as CesiumSceneManager;
  const registry = new SceneRegistry();
  registry.set(manager);
  return { clock, registry, renders: () => renders };
}

function show(
  renderer: SplatRenderer,
  options: { site?: ReturnType<typeof twoEpochSite>; registry?: SceneRegistry } = {},
) {
  useSettings.getState().set({ splatRenderer: renderer });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(queryKeys.site(SITE_ID), options.site ?? twoEpochSite());
  useSites.setState({
    activeSiteId: SITE_ID,
    representation: { [SITE_ID]: "gaussian-splat" },
    epoch: {},
  });
  const tree = (children: ReactNode) => (
    <SceneContext.Provider value={options.registry ?? new SceneRegistry()}>
      <QueryClientProvider client={client}>
        <GlassTooltipProvider>{children}</GlassTooltipProvider>
      </QueryClientProvider>
    </SceneContext.Provider>
  );
  return render(
    tree(
      <>
        <ViewsSlot />
        <RelitSlot />
      </>,
    ),
  );
}

afterEach(() => {
  cleanup();
  useLighting.setState({ sun: "off" });
  useAsCaptured.setState({ on: false });
  useSunReport.setState({ reports: {} });
  useSettings.getState().reset();
  useSites.setState({ activeSiteId: null, representation: {}, epoch: {}, compareDates: null });
  useViewRule.setState({ active: null });
});

/** Two dates drawn at once: compared across the divider, or one per cell under a view rule. */
const comparing = () =>
  act(() => useSites.setState({ compareDates: { siteId: SITE_ID, left: E1.id, right: E2.id } }));

const sunGroup = () => screen.getByRole("radiogroup", { name: SUNLIGHT_TEXT.label });
const sunRadio = (name: string) => screen.getByRole("radio", { name });

describe("the Sun control", () => {
  it("offers Off, Now and At capture under CesiumJS, and sets the sun", async () => {
    show("cesium");
    // The first test loads the control's chunk: its transform is slow on a loaded machine.
    await screen.findByTestId("views-sun", {}, { timeout: 10_000 });
    expect(sunGroup()).toBeTruthy();
    for (const name of ["Off", "Now", "At capture"]) expect(sunRadio(name)).toBeEnabled();
    expect(sunRadio("Off")).toHaveAttribute("aria-checked", "true");
    await userEvent.click(sunRadio("Now"));
    expect(useLighting.getState().sun).toBe("now");
    await userEvent.click(sunRadio("At capture"));
    expect(useLighting.getState().sun).toBe("capture");
  });

  it("is offered disabled under another renderer, beside the shared notice and switch", async () => {
    show("playcanvas");
    await screen.findByTestId("views-sun");
    for (const name of ["Off", "Now", "At capture"]) expect(sunRadio(name)).toBeDisabled();
    expect(screen.getByTestId("views-needs-cesium")).toHaveTextContent(
      "Needs the CesiumJS renderer",
    );
  });

  it("shows Off, disabled, while As captured is on, says why, and keeps the choice for after", async () => {
    useLighting.setState({ sun: "now" });
    useAsCaptured.setState({ on: true });
    show("cesium");
    await screen.findByTestId("views-sun");
    expect(sunRadio("Off")).toHaveAttribute("aria-checked", "true");
    expect(sunRadio("Now")).toBeDisabled();
    expect(screen.getByTestId("views-sun-as-captured")).toHaveTextContent(
      "Off while As captured is on",
    );
    act(() => useAsCaptured.getState().set(false));
    expect(sunRadio("Now")).toHaveAttribute("aria-checked", "true");
    expect(useLighting.getState().sun).toBe("now");
  });

  it("offers no At capture while two dates are drawn, and says why", async () => {
    show("cesium");
    await screen.findByTestId("views-sun");
    act(() => useLighting.getState().set("capture"));
    expect(screen.queryByTestId("views-sun-several")).toBeNull();
    comparing();
    expect(sunRadio("At capture")).toBeDisabled();
    expect(sunRadio("Now")).toBeEnabled();
    expect(screen.getByTestId("views-sun-several")).toHaveTextContent(SUNLIGHT_TEXT.several);
    // The choice is kept, and offered again once one date is drawn.
    expect(useLighting.getState().sun).toBe("capture");
    act(() => useSites.setState({ compareDates: null }));
    expect(sunRadio("At capture")).toBeEnabled();
    expect(screen.queryByTestId("views-sun-several")).toBeNull();
    // A per-cell view rule draws several dates as one: the same.
    act(() => useViewRule.setState({ active: { siteId: SITE_ID, rule: "latest" } }));
    expect(sunRadio("At capture")).toBeDisabled();
    expect(screen.getByTestId("views-sun-several")).toBeTruthy();
  });

  it("offers no At capture for a date whose capture time was not recorded", async () => {
    const uploaded = epoch("e2", {
      ordinal: 2,
      observedAt: "2026-06-01T15:00:00Z",
      observedAtSource: "created",
      currentAssetId: "a2",
    });
    show("cesium", { site: twoEpochSite({ epochs: [E1, uploaded] }) });
    await screen.findByTestId("views-sun");
    expect(sunRadio("Now")).toBeEnabled();
    expect(sunRadio("At capture")).toBeDisabled();
    expect(screen.getByTestId("views-sun").getAttribute("title")).toContain(
      SUNLIGHT_TEXT.noCaptureTime,
    );
  });
});

describe("the badge", () => {
  const day = { at: "2026-10-20T12:00:00.000Z", sine: 0.4 };

  it("says nothing until the scene reports the shown model relit, then whose sun and over what", async () => {
    show("cesium");
    act(() => useLighting.getState().set("now"));
    await waitFor(() => expect(screen.queryByTestId("relit-badge")).toBeNull());
    // A report for another model is not this one's.
    act(() => useSunReport.getState().report(A1.id, day));
    expect(screen.queryByTestId("relit-badge")).toBeNull();
    act(() => useSunReport.getState().report(A2.id, day));
    const badge = await screen.findByTestId("relit-badge");
    expect(badge).toHaveTextContent(relitSentence("now", "2026-06-01T15:00:00Z"));
    expect(screen.queryByTestId("relit-night")).toBeNull();
    act(() => useSunReport.getState().report(A2.id, { ...day, sine: -0.2 }));
    expect(screen.getByTestId("relit-night")).toHaveTextContent("Sun below horizon");
  });

  it("holds the clock only while the shown date is reported relit", async () => {
    const scene = fakeScene();
    useSunReport.getState().report(A2.id, day);
    show("cesium", { registry: scene.registry });
    act(() => useLighting.getState().set("capture"));
    await waitFor(() => expect(scene.clock.shouldAnimate).toBe(false));
    // The scene stops drawing it relit (the model unloaded, say): the sky is not left moved.
    act(() => useSunReport.getState().report(A2.id, null));
    await waitFor(() => expect(scene.clock.shouldAnimate).toBe(true));
    expect(JulianDate.toDate(scene.clock.currentTime).getUTCMonth()).toBe(9);
    expect(screen.queryByTestId("relit-badge")).toBeNull();
    act(() => useSunReport.getState().report(A2.id, day));
    await waitFor(() => expect(scene.clock.shouldAnimate).toBe(false));
    expect(iso(scene.clock.currentTime)).toBe("2026-06-01T15:00:00.000Z");
  });

  it("holds the clock at the shown date's capture while At capture is on, and gives it back", async () => {
    const scene = fakeScene();
    useSunReport.getState().report(A2.id, day);
    show("cesium", { registry: scene.registry });
    act(() => useLighting.getState().set("capture"));
    await screen.findByTestId("relit-badge");
    expect(iso(scene.clock.currentTime)).toBe("2026-06-01T15:00:00.000Z");
    expect(scene.clock.shouldAnimate).toBe(false);
    expect(screen.getByTestId("relit-badge")).toHaveTextContent("Relit by the sun (at capture)");
    expect(scene.renders()).toBeGreaterThan(0);

    // Another date: its own capture time, once that date is relit.
    act(() => useSites.setState({ epoch: { [SITE_ID]: E1.id } }));
    await waitFor(() => expect(scene.clock.shouldAnimate).toBe(true));
    act(() => useSunReport.getState().report(A1.id, day));
    await waitFor(() => expect(iso(scene.clock.currentTime)).toBe("2026-03-01T15:00:00.000Z"));

    // As captured: the relight is not drawn, and the clock is given back, running.
    act(() => useAsCaptured.getState().set(true));
    await waitFor(() => expect(scene.clock.shouldAnimate).toBe(true));
    expect(JulianDate.toDate(scene.clock.currentTime).getUTCFullYear()).toBe(2026);
    expect(JulianDate.toDate(scene.clock.currentTime).getUTCMonth()).toBe(9);
    expect(screen.queryByTestId("relit-badge")).toBeNull();

    // Back, then the sun off: given back again.
    act(() => useAsCaptured.getState().set(false));
    await waitFor(() => expect(scene.clock.shouldAnimate).toBe(false));
    act(() => useLighting.getState().set("off"));
    await waitFor(() => expect(scene.clock.shouldAnimate).toBe(true));
    expect(JulianDate.toDate(scene.clock.currentTime).getUTCMonth()).toBe(9);
  });

  it("holds no clock while two dates are drawn, and says the sun of now over each date's capture", async () => {
    const scene = fakeScene();
    // Compared: the halves' dates are drawn and report; the single date (e2) is hidden.
    useSunReport.getState().report(A1.id, day);
    show("cesium", { registry: scene.registry });
    comparing();
    act(() => useLighting.getState().set("capture"));
    const badge = await screen.findByTestId("relit-badge");
    expect(scene.clock.shouldAnimate).toBe(true);
    expect(iso(scene.clock.currentTime)).toBe("2026-10-20T12:00:00.000Z");
    expect(badge).toHaveTextContent(relitSentence("now", null, true));
    expect(badge).toHaveTextContent(SUNLIGHT_TEXT.several);
    // One date again, and reported relit: held at its capture.
    act(() => useSites.setState({ compareDates: null }));
    act(() => useSunReport.getState().report(A2.id, day));
    await waitFor(() => expect(iso(scene.clock.currentTime)).toBe("2026-06-01T15:00:00.000Z"));
    expect(screen.getByTestId("relit-badge")).not.toHaveTextContent(SUNLIGHT_TEXT.several);
    // Comparing again gives the clock back.
    comparing();
    await waitFor(() => expect(scene.clock.shouldAnimate).toBe(true));
  });

  it("gives the clock back under another renderer, and draws the sun of now where no capture time is known", async () => {
    const scene = fakeScene();
    useSunReport.getState().report(A2.id, day);
    const uploaded = epoch("e2", {
      ordinal: 2,
      observedAt: "2026-06-01T15:00:00Z",
      observedAtSource: "created",
      currentAssetId: "a2",
    });
    show("cesium", { registry: scene.registry, site: twoEpochSite({ epochs: [E1, uploaded] }) });
    act(() => useLighting.getState().set("capture"));
    const badge = await screen.findByTestId("relit-badge");
    expect(scene.clock.shouldAnimate).toBe(true);
    expect(badge).toHaveTextContent(
      "Relit by the sun (now) over shadows baked in at capture, time not recorded",
    );
    expect(badge).toHaveTextContent(SUNLIGHT_TEXT.nowInstead);

    // A date with one: held; another renderer: given back, and nothing said.
    act(() => useSunReport.getState().report(A1.id, day));
    act(() => useSites.setState({ epoch: { [SITE_ID]: E1.id } }));
    await waitFor(() => expect(scene.clock.shouldAnimate).toBe(false));
    act(() => useSettings.getState().set({ splatRenderer: "spark" }));
    await waitFor(() => expect(scene.clock.shouldAnimate).toBe(true));
    expect(screen.queryByTestId("relit-badge")).toBeNull();
  });
});

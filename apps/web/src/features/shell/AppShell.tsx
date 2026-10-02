import { lazy, Suspense, useEffect } from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { env } from "@/app/env";
import { HOTKEYS } from "@/app/hotkeys";
import { useScene } from "@/cesium/SceneContext";
import { useHotkey } from "@/lib/hotkeys";
import { bindDockRules, useLayout } from "@/state/layout";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

import { AddPanel } from "../add-data/AddPanel";
import { CommandBox } from "../command-palette/CommandBox";
import { ShortcutSheet } from "../command-palette/ShortcutSheet";
import { CompareSplit } from "../compare/Compare";
import { DevPanel } from "../dev/DevPanel";
import { ExploreHud } from "../explore/ExploreHud";
import { InspectorPanel } from "../inspector/InspectorPanel";
import { LayerAboutSheet } from "../layers/LayerAboutSheet";
import { LayersPanel } from "../layers/LayersPanel";
import { SimulatedBadge } from "../living/SimulatedBadge";
import { MeasurePanel } from "../measure/MeasurePanel";
import { FeedsPanel } from "../mission/FeedsPanel";
import { FleetPanel } from "../mission/FleetPanel";
import { MissionOverlays } from "../mission/MissionOverlays";
import { PlansPanel } from "../mission/PlansPanel";
import { ProjectCard } from "../mission/ProjectCard";
import { openSiteSwitcher } from "../mission/siteSwitcher";
import { SelectionCard } from "../mission/SelectionCard";
import { StatusLine } from "../mission/StatusLine";
import { ViewTabs } from "../mission/ViewTabs";
import { Toasts } from "../notices/Toasts";
import { OnboardingCard } from "../onboarding/OnboardingCard";
import { RepresentationSwitcher } from "../sites/RepresentationSwitcher";
import { SiteLoadStatus } from "../sites/SiteLoadStatus";
import { TimelineControl } from "../timeline/TimelineControl";
import { DevReadouts } from "./DevReadouts";
import { MapCorner } from "./MapCorner";
import { PhoneTabBar } from "./PhoneTabBar";
import { ToolRail } from "./ToolRail";

const SettingsSheet = lazy(() =>
  import("../settings/SettingsSheet").then((m) => ({ default: m.SettingsSheet })),
);

/** The app's own keys, bound from the registry (`app/hotkeys.ts`) that the `?` sheet prints. */
function GlobalHotkeys() {
  const scene = useScene();
  const ui = useUi();
  const settings = useSettings();
  const mission = useMission();
  useHotkey(HOTKEYS.layers.combo, () => ui.openLayers("browse"));
  useHotkey(HOTKEYS.compare.combo, () => ui.openLayers("compare"));
  useHotkey(HOTKEYS.measure.combo, () => ui.togglePanel("measure"));
  useHotkey(HOTKEYS.captures.combo, () =>
    ui.activePanel === "add" && ui.addTab === "upload" ? ui.setPanel(null) : ui.openAdd("upload"),
  );
  // Sites and saved views live in the site switcher; their keys open it where they are.
  useHotkey(HOTKEYS.sites.combo, () =>
    mission.projectsOpen && ui.switcherFocus === "sites"
      ? mission.setProjectsOpen(false)
      : openSiteSwitcher("sites"),
  );
  useHotkey(HOTKEYS.bookmarks.combo, () =>
    mission.projectsOpen && ui.switcherFocus === "views"
      ? mission.setProjectsOpen(false)
      : openSiteSwitcher("views"),
  );
  useHotkey(HOTKEYS.resetNorth.combo, () => scene?.camera.resetNorth());
  useHotkey(HOTKEYS.topDown.combo, () => scene?.camera.topDown());
  useHotkey(HOTKEYS.home.combo, () => scene?.camera.flyHome());
  useHotkey(HOTKEYS.explore.combo, () => {
    if (!ui.exploreMode) scene?.explore.setSpeed(settings.exploreSpeed);
    ui.setExploreMode(!ui.exploreMode);
  });
  useHotkey(HOTKEYS.mapView.combo, () => mission.setView("map"));
  useHotkey(HOTKEYS.planView.combo, () => mission.setView("plan"));
  useHotkey(HOTKEYS.fleetView.combo, () => mission.setView("fleet"));
  useHotkey(HOTKEYS.agent.combo, () => ui.setActivityOpen(!ui.activityOpen));
  useHotkey(HOTKEYS.settings.combo, () => ui.setSettingsOpen(true));
  useHotkey(HOTKEYS.shortcuts.combo, () => ui.setShortcutsOpen(!ui.shortcutsOpen));
  useHotkey(
    HOTKEYS.devTools.combo,
    () => env.devToolsEnabled && settings.set({ devToolsOpen: !settings.devToolsOpen }),
  );
  useHotkey(HOTKEYS.escape.combo, () => {
    if (ui.shortcutsOpen) ui.setShortcutsOpen(false);
    else if (ui.moreOpen) ui.setMoreOpen(false);
    else if (ui.activityOpen) ui.setActivityOpen(false);
    else if (ui.writeTokenPrompt) ui.setWriteTokenPrompt(false);
    else if (ui.measureMode) ui.setMeasureMode(null);
    else if (mission.projectsOpen) mission.setProjectsOpen(false);
    else if (mission.feedsOpen) mission.setFeedsOpen(false);
    else if (mission.composer && mission.composer.status !== "drafting") mission.closeComposer();
    else if (mission.selection) {
      mission.select(null);
      scene?.mission.setSelectedZone(null);
    } else if (mission.view !== "map") mission.setView("map");
    else if (ui.inspectorOpen) {
      ui.setInspectorOpen(false);
      scene?.selection.clear();
    } else if (ui.activePanel) ui.setPanel(null);
  });
  return null;
}

function SettingsSheetLazy() {
  const open = useUi((s) => s.settingsOpen);
  if (!open) return null;
  return (
    <Suspense fallback={null}>
      <SettingsSheet />
    </Suspense>
  );
}

/**
 * Composes the HUD over the world as a fixed set of screen regions.
 *
 * Every surface lives in exactly one region, and regions are cells of one CSS grid
 * (`.hud` in `app.css`), so two surfaces can never be drawn over each other: they can
 * only share a dock, where they stack. The regions:
 *
 * - **top**: the site switcher, the command box (search, actions, the agent), view tabs.
 * - **rail**: four labelled tools on the left edge — Layers, Measure, Add, Settings.
 * - **left dock**: the tool panel you opened.
 * - **drawer**: Plan or Fleet, full height on the right edge. The map beside it stays live:
 *   it takes clicks, and a machine picked from the Fleet table is flown to with its card open.
 * - **right dock**: messages (toasts), what you selected, the inspector.
 * - **center**: first-run welcome; otherwise the map.
 * - **strip**: controls for what is in view (representation and its load, dates, explore).
 * - **bar**: the status line on the left (fleet, agent, a degraded connection), developer
 *   readouts when they are on, and the compass, Earth and data credits in one pill on the
 *   right.
 * - **tabs**: on a phone only, Map / Plan / Fleet / More along the bottom.
 *
 * Things that float on purpose, over the regions, and close with Escape or a click away: the
 * command box's results, the site switcher, the agent's activity log, and on a phone the More
 * sheet and the folded data credits (`data-hud-popover`).
 *
 * On narrow screens the docks and the drawer collapse into one bottom sheet; `data-sheet`
 * says which was touched last, and that one is shown (`state/layout.ts`).
 */
export function AppShell() {
  const sheet = useLayout((s) => s.focus);
  const view = useMission((s) => s.view);
  useEffect(() => bindDockRules(), []);
  return (
    <>
      <GlobalHotkeys />
      <div className="hud" data-sheet={sheet ?? "none"} data-view={view} data-testid="hud">
        <ErrorBoundary inline label="Fleet overlay">
          <MissionOverlays />
        </ErrorBoundary>
        <CompareSplit />
        <header className="hud-region hud-top">
          <ErrorBoundary inline label="Site">
            <ProjectCard />
          </ErrorBoundary>
          <div className="hud-top__search">
            <ErrorBoundary inline label="Command box">
              <CommandBox />
            </ErrorBoundary>
          </div>
          <ViewTabs />
        </header>
        <nav className="hud-region hud-rail" aria-label="Tools">
          <ToolRail />
        </nav>
        <section className="hud-region hud-dock hud-dock--left" aria-label="Panel">
          <ErrorBoundary inline label="Panel">
            <LayersPanel />
            <MeasurePanel />
            <AddPanel />
          </ErrorBoundary>
        </section>
        <section className="hud-region hud-drawer" aria-label="Plan and fleet">
          <ErrorBoundary inline label="Plans">
            <PlansPanel />
            <FleetPanel />
          </ErrorBoundary>
        </section>
        <div className="hud-region hud-center">
          <OnboardingCard />
        </div>
        <aside className="hud-region hud-dock hud-dock--right" aria-label="Details">
          <div className="hud-stack hud-messages">
            <ErrorBoundary inline label="Simulated motion">
              <SimulatedBadge />
            </ErrorBoundary>
            <Toasts />
          </div>
          <div className="hud-stack hud-detail">
            <ErrorBoundary inline label="Selection">
              <SelectionCard />
              <FeedsPanel />
            </ErrorBoundary>
            <ErrorBoundary inline label="Inspector">
              <InspectorPanel />
              <DevPanel />
            </ErrorBoundary>
          </div>
        </aside>
        <div className="hud-region hud-strip">
          <ExploreHud />
          <TimelineControl />
          <RepresentationSwitcher />
          <SiteLoadStatus />
        </div>
        <footer className="hud-region hud-bar">
          <ErrorBoundary inline label="Status">
            <StatusLine />
          </ErrorBoundary>
          <DevReadouts />
          <MapCorner />
        </footer>
        <div className="hud-region hud-tabs">
          <PhoneTabBar />
        </div>
      </div>
      <SettingsSheetLazy />
      <LayerAboutSheet />
      <ShortcutSheet />
    </>
  );
}

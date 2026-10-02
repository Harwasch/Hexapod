import { lazy, Suspense, useEffect, type ReactNode } from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { env } from "@/app/env";
import { HOTKEYS } from "@/app/hotkeys";
import { useScene } from "@/cesium/SceneContext";
import { useHotkey } from "@/lib/hotkeys";
import { bindDockRules, useLayout } from "@/state/layout";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

import { AddDataSheet } from "../add-data/AddDataSheet";
import { BookmarksPanel } from "../bookmarks/BookmarksPanel";
import { CapturesPanel } from "../captures/CapturesPanel";
import { CommandBox } from "../command-palette/CommandBox";
import { ShortcutSheet } from "../command-palette/ShortcutSheet";
import { ComparePanel } from "../compare/ComparePanel";
import { DevPanel } from "../dev/DevPanel";
import { ExploreHud } from "../explore/ExploreHud";
import { InspectorPanel } from "../inspector/InspectorPanel";
import { LayerAboutSheet } from "../layers/LayerAboutSheet";
import { LayersPanel } from "../layers/LayersPanel";
import { SimulatedBadge } from "../living/SimulatedBadge";
import { MeasurePanel } from "../measure/MeasurePanel";
import { FeedsPanel } from "../mission/FeedsPanel";
import { FleetPanel } from "../mission/FleetPanel";
import { LayerPills } from "../mission/LayerPills";
import { MissionOverlays } from "../mission/MissionOverlays";
import { PlansPanel } from "../mission/PlansPanel";
import { ProjectCard } from "../mission/ProjectCard";
import { SelectionCard } from "../mission/SelectionCard";
import { StatusLine } from "../mission/StatusLine";
import { ViewTabs } from "../mission/ViewTabs";
import { NavControls } from "../nav/NavControls";
import { Toasts } from "../notices/Toasts";
import { OnboardingCard } from "../onboarding/OnboardingCard";
import { RepresentationSwitcher } from "../sites/RepresentationSwitcher";
import { SiteLoadStatus } from "../sites/SiteLoadStatus";
import { SitesPanel } from "../sites/SitesPanel";
import { TimelineControl } from "../timeline/TimelineControl";
import { CreditSlot } from "./CreditSlot";
import { DevReadouts } from "./DevReadouts";
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
  useHotkey(HOTKEYS.layers.combo, () => ui.togglePanel("layers"));
  useHotkey(HOTKEYS.sites.combo, () => ui.togglePanel("sites"));
  useHotkey(HOTKEYS.measure.combo, () => ui.togglePanel("measure"));
  useHotkey(HOTKEYS.compare.combo, () => ui.togglePanel("compare"));
  useHotkey(HOTKEYS.bookmarks.combo, () => ui.togglePanel("bookmarks"));
  useHotkey(HOTKEYS.captures.combo, () => ui.togglePanel("captures"));
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

function MapOnly({ children }: { children: ReactNode }) {
  const view = useMission((s) => s.view);
  return view === "map" ? <>{children}</> : null;
}

/**
 * Composes the HUD over the world as a fixed set of screen regions.
 *
 * Every surface lives in exactly one region, and regions are cells of one CSS grid
 * (`.hud` in `app.css`), so two surfaces can never be drawn over each other: they can
 * only share a dock, where they stack. The regions:
 *
 * - **top**: project, the command box (search, actions, the agent), view tabs.
 * - **rail**: tools on the left edge.
 * - **left dock**: the one panel you opened — a tool panel or the Plan / Fleet window.
 * - **right dock**: quick layers, messages (toasts), what you selected.
 * - **center**: first-run welcome; otherwise the map.
 * - **strip**: controls for what is in view (representation and its load, dates, explore).
 * - **bar**: the status line on the left (fleet, agent, a degraded connection), developer
 *   readouts when they are on, and the compass, Earth and data credits in one pill on the
 *   right.
 *
 * Two things float on purpose, over the regions, and close with Escape or a click away: the
 * command box's results and the agent's activity log (`data-hud-popover`).
 *
 * On a phone the docks collapse into one bottom sheet; `data-sheet` says which dock was
 * touched last, and that one is shown (`state/layout.ts`).
 */
export function AppShell() {
  const sheet = useLayout((s) => s.focus);
  useEffect(() => bindDockRules(), []);
  return (
    <>
      <GlobalHotkeys />
      <div className="hud" data-sheet={sheet ?? "none"} data-testid="hud">
        <ErrorBoundary inline label="Fleet overlay">
          <MissionOverlays />
        </ErrorBoundary>
        <header className="hud-region hud-top">
          <ErrorBoundary inline label="Project">
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
            <SitesPanel />
            <CapturesPanel />
            <MeasurePanel />
            <ComparePanel />
            <BookmarksPanel />
          </ErrorBoundary>
          <ErrorBoundary inline label="Plans">
            <PlansPanel />
            <FleetPanel />
          </ErrorBoundary>
        </section>
        <div className="hud-region hud-center">
          <OnboardingCard />
        </div>
        <aside className="hud-region hud-dock hud-dock--right" aria-label="Details">
          <div className="hud-stack hud-pills">
            <LayerPills />
          </div>
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
          <MapOnly>
            <TimelineControl />
            <RepresentationSwitcher />
            <SiteLoadStatus />
          </MapOnly>
        </div>
        <footer className="hud-region hud-bar">
          <ErrorBoundary inline label="Status">
            <StatusLine />
          </ErrorBoundary>
          <DevReadouts />
          <div className="glass glass--strong hud-corner" data-testid="map-corner">
            <NavControls />
            <CreditSlot />
          </div>
        </footer>
      </div>
      <AddDataSheet />
      <SettingsSheetLazy />
      <LayerAboutSheet />
      <ShortcutSheet />
    </>
  );
}

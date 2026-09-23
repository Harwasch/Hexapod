import { lazy, Suspense, useEffect, type ReactNode } from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { env } from "@/app/env";
import { useScene } from "@/cesium/SceneContext";
import { useHotkey } from "@/lib/hotkeys";
import { bindDockRules, useLayout } from "@/state/layout";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

import { AddDataSheet } from "../add-data/AddDataSheet";
import { BookmarksPanel } from "../bookmarks/BookmarksPanel";
import { CapturesPanel } from "../captures/CapturesPanel";
import { CommandPalette } from "../command-palette/CommandPalette";
import { ComparePanel } from "../compare/ComparePanel";
import { DevPanel } from "../dev/DevPanel";
import { ExploreHud } from "../explore/ExploreHud";
import { InspectorPanel } from "../inspector/InspectorPanel";
import { LayerAboutSheet } from "../layers/LayerAboutSheet";
import { LayersPanel } from "../layers/LayersPanel";
import { SimulatedBadge } from "../living/SimulatedBadge";
import { MeasurePanel } from "../measure/MeasurePanel";
import { AgentStream } from "../mission/AgentStream";
import { CommandBar } from "../mission/CommandBar";
import { FeedsPanel } from "../mission/FeedsPanel";
import { FleetPanel } from "../mission/FleetPanel";
import { LayerPills } from "../mission/LayerPills";
import { MissionOverlays } from "../mission/MissionOverlays";
import { PlansPanel } from "../mission/PlansPanel";
import { ProjectCard } from "../mission/ProjectCard";
import { SelectionCard } from "../mission/SelectionCard";
import { ViewTabs } from "../mission/ViewTabs";
import { NavControls } from "../nav/NavControls";
import { SetupNotices } from "../notices/SetupNotices";
import { Toasts } from "../notices/Toasts";
import { OnboardingCard } from "../onboarding/OnboardingCard";
import { SearchPill } from "../search/SearchPill";
import { RepresentationSwitcher } from "../sites/RepresentationSwitcher";
import { SitesPanel } from "../sites/SitesPanel";
import { TimelineControl } from "../timeline/TimelineControl";
import { CreditSlot } from "./CreditSlot";
import { ToolRail } from "./ToolRail";

const SettingsSheet = lazy(() =>
  import("../settings/SettingsSheet").then((m) => ({ default: m.SettingsSheet })),
);

function GlobalHotkeys() {
  const scene = useScene();
  const ui = useUi();
  const settings = useSettings();
  const mission = useMission();
  useHotkey("l", () => ui.togglePanel("layers"));
  useHotkey("s", () => ui.togglePanel("sites"));
  useHotkey("m", () => ui.togglePanel("measure"));
  useHotkey("c", () => ui.togglePanel("compare"));
  useHotkey("b", () => ui.togglePanel("bookmarks"));
  useHotkey("u", () => ui.togglePanel("captures"));
  useHotkey("n", () => scene?.camera.resetNorth());
  useHotkey("t", () => scene?.camera.topDown());
  useHotkey("h", () => scene?.camera.flyHome());
  useHotkey("g", () => ui.setExploreMode(!ui.exploreMode));
  useHotkey("1", () => mission.setView("map"));
  useHotkey("2", () => mission.setView("plan"));
  useHotkey("3", () => mission.setView("fleet"));
  useHotkey("a", () => mission.setStreamOpen(!mission.streamOpen));
  useHotkey(",", () => ui.setSettingsOpen(true));
  useHotkey(
    "d",
    () => env.devToolsEnabled && settings.set({ devToolsOpen: !settings.devToolsOpen }),
  );
  useHotkey("escape", () => {
    if (ui.paletteOpen) ui.setPaletteOpen(false);
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
 * - **top**: project, search, view tabs.
 * - **rail** / **nav**: tools on the left edge, camera on the right edge.
 * - **left dock**: the one panel you opened — a tool panel or the Plan / Fleet window.
 * - **right dock**: status (notices, toasts), what you selected, and the agent.
 * - **center**: first-run welcome; otherwise the map.
 * - **strip**: controls for what is in view (representation, dates, explore).
 * - **bar**: the command bar and the data credits.
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
            <ErrorBoundary inline label="Search">
              <SearchPill />
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
            <SetupNotices />
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
          <div className="hud-stack hud-agent">
            <ErrorBoundary inline label="Agent">
              <AgentStream />
            </ErrorBoundary>
          </div>
        </aside>
        <nav className="hud-region hud-nav" aria-label="Camera">
          <NavControls />
        </nav>
        <div className="hud-region hud-strip">
          <ExploreHud />
          <MapOnly>
            <TimelineControl />
            <RepresentationSwitcher />
          </MapOnly>
        </div>
        <footer className="hud-region hud-bar">
          <ErrorBoundary inline label="Command bar">
            <CommandBar />
          </ErrorBoundary>
          <CreditSlot />
        </footer>
      </div>
      <AddDataSheet />
      <SettingsSheetLazy />
      <LayerAboutSheet />
      <CommandPalette />
    </>
  );
}

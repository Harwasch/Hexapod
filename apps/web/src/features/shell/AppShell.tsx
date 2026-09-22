import { lazy, Suspense, type ReactNode } from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { env } from "@/app/env";
import { useScene } from "@/cesium/SceneContext";
import { useHotkey } from "@/lib/hotkeys";
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

/** Composes the mission-control HUD over the world. The map is always the hero. */
export function AppShell() {
  return (
    <>
      <GlobalHotkeys />
      <div className="hud">
        <ErrorBoundary inline label="Fleet overlay">
          <MissionOverlays />
        </ErrorBoundary>
        <div className="hud-top-left">
          <ErrorBoundary inline label="Project">
            <ProjectCard />
          </ErrorBoundary>
          {/* Under the project badge on purpose: the same corner already distinguishes
              simulated demo data from real, and this is the same distinction. */}
          <ErrorBoundary inline label="Simulated motion">
            <SimulatedBadge />
          </ErrorBoundary>
        </div>
        <div className="hud-top-center">
          <ErrorBoundary inline label="Search">
            <SearchPill />
          </ErrorBoundary>
        </div>
        <div className="hud-top-right">
          <ViewTabs />
          <LayerPills />
          <SetupNotices />
        </div>
        <div className="hud-left">
          <ToolRail />
          <ErrorBoundary inline label="Panel">
            <LayersPanel />
            <SitesPanel />
            <CapturesPanel />
            <MeasurePanel />
            <ComparePanel />
            <BookmarksPanel />
          </ErrorBoundary>
        </div>
        <div className="hud-nav">
          <NavControls />
        </div>
        <div className="hud-right">
          <ErrorBoundary inline label="Inspector">
            <InspectorPanel />
            <DevPanel />
          </ErrorBoundary>
        </div>
        <ErrorBoundary inline label="Plans">
          <PlansPanel />
          <FleetPanel />
        </ErrorBoundary>
        <div className="hud-bottom-left">
          <ErrorBoundary inline label="Selection">
            <FeedsPanel />
            <SelectionCard />
          </ErrorBoundary>
        </div>
        <div className="hud-bottom-center">
          <ExploreHud />
          <MapOnly>
            <TimelineControl />
            <RepresentationSwitcher />
          </MapOnly>
        </div>
        <div className="hud-bottom-right">
          <ErrorBoundary inline label="Agent">
            <AgentStream />
          </ErrorBoundary>
        </div>
        <div className="hud-bottom-bar">
          <ErrorBoundary inline label="Command bar">
            <CommandBar />
          </ErrorBoundary>
        </div>
        <OnboardingCard />
        <Toasts />
      </div>
      <AddDataSheet />
      <SettingsSheetLazy />
      <LayerAboutSheet />
      <CommandPalette />
    </>
  );
}

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { env } from "@/app/env";
import { useScene } from "@/cesium/SceneContext";
import { useHotkey } from "@/lib/hotkeys";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

import { AddDataSheet } from "../add-data/AddDataSheet";
import { BookmarksPanel } from "../bookmarks/BookmarksPanel";
import { Brand } from "../brand/Brand";
import { CommandPalette } from "../command-palette/CommandPalette";
import { ComparePanel } from "../compare/ComparePanel";
import { DevPanel } from "../dev/DevPanel";
import { ExploreHud } from "../explore/ExploreHud";
import { InspectorPanel } from "../inspector/InspectorPanel";
import { LayerAboutSheet } from "../layers/LayerAboutSheet";
import { LayersPanel } from "../layers/LayersPanel";
import { MeasurePanel } from "../measure/MeasurePanel";
import { NavControls } from "../nav/NavControls";
import { SetupNotices } from "../notices/SetupNotices";
import { Toasts } from "../notices/Toasts";
import { OnboardingCard } from "../onboarding/OnboardingCard";
import { SearchPill } from "../search/SearchPill";
import { RepresentationSwitcher } from "../sites/RepresentationSwitcher";
import { SitesPanel } from "../sites/SitesPanel";
import { StatusBar } from "../status/StatusBar";
import { TimelineControl } from "../timeline/TimelineControl";
import { ToolRail } from "./ToolRail";

function GlobalHotkeys() {
  const scene = useScene();
  const ui = useUi();
  const settings = useSettings();
  useHotkey("l", () => ui.togglePanel("layers"));
  useHotkey("s", () => ui.togglePanel("sites"));
  useHotkey("m", () => ui.togglePanel("measure"));
  useHotkey("c", () => ui.togglePanel("compare"));
  useHotkey("b", () => ui.togglePanel("bookmarks"));
  useHotkey("n", () => scene?.camera.resetNorth());
  useHotkey("t", () => scene?.camera.topDown());
  useHotkey("h", () => scene?.camera.flyHome());
  useHotkey("g", () => ui.setExploreMode(!ui.exploreMode));
  useHotkey(",", () => ui.setSettingsOpen(true));
  useHotkey(
    "d",
    () => env.devToolsEnabled && settings.set({ devToolsOpen: !settings.devToolsOpen }),
  );
  useHotkey("=", () => scene?.camera.zoomBy(0.5));
  useHotkey("-", () => scene?.camera.zoomBy(-1));
  useHotkey("escape", () => {
    if (ui.paletteOpen) ui.setPaletteOpen(false);
    else if (ui.measureMode) ui.setMeasureMode(null);
    else if (ui.inspectorOpen) {
      ui.setInspectorOpen(false);
      scene?.selection.clear();
    } else if (ui.activePanel) ui.setPanel(null);
  });
  return null;
}

/** Composes the floating HUD over the world. The map is always the hero. */
export function AppShell() {
  return (
    <>
      <GlobalHotkeys />
      <div className="hud">
        <div className="hud-top-left">
          <Brand />
        </div>
        <div className="hud-top-center">
          <ErrorBoundary inline label="Search">
            <SearchPill />
          </ErrorBoundary>
        </div>
        <div
          className="hud-top-right"
          style={{ maxWidth: "24rem", flexDirection: "column", alignItems: "flex-end" }}
        >
          <SetupNotices />
        </div>
        <div className="hud-left">
          <ToolRail />
          <ErrorBoundary inline label="Panel">
            <LayersPanel />
            <SitesPanel />
            <MeasurePanel />
            <ComparePanel />
            <BookmarksPanel />
          </ErrorBoundary>
        </div>
        <div className="hud-right">
          <ErrorBoundary inline label="Inspector">
            <InspectorPanel />
            <DevPanel />
          </ErrorBoundary>
        </div>
        <div className="hud-bottom-center">
          <ExploreHud />
          <TimelineControl />
          <RepresentationSwitcher />
          <StatusBar />
        </div>
        <div className="hud-bottom-right">
          <NavControls />
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

import { lazy, Suspense } from "react";

const SettingsSheet = lazy(() =>
  import("../settings/SettingsSheet").then((m) => ({ default: m.SettingsSheet })),
);

function SettingsSheetLazy() {
  const open = useUi((s) => s.settingsOpen);
  if (!open) return null;
  return (
    <Suspense fallback={null}>
      <SettingsSheet />
    </Suspense>
  );
}

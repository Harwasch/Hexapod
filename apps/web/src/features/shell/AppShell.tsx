import {
  lazy,
  Suspense,
  useEffect,
  type ComponentType,
  type LazyExoticComponent,
  type ReactNode,
} from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { env } from "@/app/env";
import { HOTKEYS } from "@/app/hotkeys";
import { useScene } from "@/cesium/SceneContext";
import { useHotkey } from "@/lib/hotkeys";
import { bindDockRules, useLayout } from "@/state/layout";
import { useMission } from "@/state/mission";
import { useChangeDrape } from "@/state/changeDrape";
import { useSettings } from "@/state/settings";
import { useSkinPoke } from "@/state/skinPoke";
import { useLantern } from "@/state/lantern";
import { useToasts } from "@/state/toasts";
import { useUi } from "@/state/ui";

import { AddPanel } from "../add-data/AddPanel";
import { CommandBox } from "../command-palette/CommandBox";
import { ShortcutSheet } from "../command-palette/ShortcutSheet";
import { CompareSplit } from "../compare/Compare";
import { DescentArrival } from "../descent/DescentArrival";
import { DevPanel } from "../dev/DevPanel";
import { ExploreHud } from "../explore/ExploreHud";
import { LanternBridge } from "../lantern/LanternBridge";
import { LanternRim } from "../lantern/LanternRim";
import { UNAVAILABLE } from "../lantern/lanternWords";
import { TimeLanternControl } from "../lantern/TimeLanternControl";
import { LayerAboutSheet } from "../layers/LayerAboutSheet";
import { LayersPanel } from "../layers/LayersPanel";
import { LivingLabPanel } from "../living/LivingLabPanel";
import { PokeBadge } from "../living/PokeBadge";
import { SimulatedBadge } from "../living/SimulatedBadge";
import { MapContextMenu } from "../map/MapContextMenu";
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
import { RelitSlot, ViewsSlot } from "../views/ViewsSlot";
import { DevReadouts } from "./DevReadouts";
import { LandSurfaceSlot } from "./LandSurfaceSlot";
import { MapCorner } from "./MapCorner";
import { PhoneTabBar } from "./PhoneTabBar";
import { stepBack } from "./stepBack";
import { ToolRail } from "./ToolRail";
import { useUndoHotkeys } from "./undoHotkeys";

const SettingsSheet = lazy(() =>
  import("../settings/SettingsSheet").then((m) => ({ default: m.SettingsSheet })),
);

/** `React.lazy` for a module's named component (lazy wants a default export). */
function lazyNamed<K extends string, M extends Record<K, ComponentType>>(
  load: () => Promise<M>,
  name: K,
): LazyExoticComponent<M[K]> {
  return lazy(() => load().then((module) => ({ default: module[name] })));
}

// The inspector is always mounted (it opens and closes itself on the selection) but draws
// nothing until something is selected, so it loads on demand like SettingsSheet: its panel,
// the provenance and placement text and their helpers stay out of the globe's first load.
const InspectorPanel = lazyNamed(() => import("../inspector/InspectorPanel"), "InspectorPanel");

// The Land feature (VITE_ENABLE_LAND_EXPLORATION): its surface, strip, overlays and agent, the bridges that draw and sync it
// on the globe, the fog of knowing, and the workspace sign-in behind them (oidc-client-ts, which
// only the identity bridge and the surface reach). Each is loaded on demand like SettingsSheet,
// and only rendered with the flag on, so a build with the flag off never requests any of it
// (its chunks are still built) and the globe's first load carries none of it
// (`scripts/check-bundle.mjs` holds that line). With the flag on they load as the shell mounts
// and behave as before; a chunk that fails to load fails alone (`OnDemand` for the bridges,
// `LandSurfaceSlot`'s own boundary for the surface, the inline boundaries for the card and legend).
const LandIdentityBridge = lazyNamed(
  () => import("../land/WorkspaceIdentity"),
  "LandIdentityBridge",
);
const LandBridge = lazyNamed(() => import("@/cesium/LandBridge"), "LandBridge");
const LandContextBridge = lazyNamed(
  () => import("@/cesium/LandContextBridge"),
  "LandContextBridge",
);
const LandRasterBridge = lazyNamed(() => import("@/cesium/LandRasterBridge"), "LandRasterBridge");
const LandFocusBridge = lazyNamed(() => import("@/cesium/LandFocusBridge"), "LandFocusBridge");
const LandOverlays = lazyNamed(() => import("../land/surface/LandOverlays"), "LandOverlays");
const LandAgent = lazyNamed(() => import("../land/surface/LandAgent"), "LandAgent");
const LandStrip = lazyNamed(() => import("../land/surface/LandStrip"), "LandStrip");
const KnowledgeFogBridge = lazyNamed(
  () => import("@/cesium/KnowledgeFogBridge"),
  "KnowledgeFogBridge",
);
const ShowOnceBridge = lazyNamed(() => import("@/cesium/ShowOnceBridge"), "ShowOnceBridge");
const LandMissionBridge = lazyNamed(() => import("../land/LandMissionBridge"), "LandMissionBridge");
const KnowledgeCellCard = lazyNamed(
  () => import("../knowledge/KnowledgeCellCard"),
  "KnowledgeCellCard",
);
// The change map between two dates and its legend (Compare › Dates), loaded when one is asked for.
const ChangeDrape = lazyNamed(() => import("../change/ChangeDrape"), "ChangeDrape");
const KnowledgeLegend = lazyNamed(() => import("../knowledge/KnowledgeLegend"), "KnowledgeLegend");

/** The app's own keys, bound from the registry (`app/hotkeys.ts`) that the `?` sheet prints. */
function GlobalHotkeys() {
  const scene = useScene();
  const ui = useUi();
  const settings = useSettings();
  const mission = useMission();
  useHotkey(HOTKEYS.layers.combo, () => ui.openLayers("browse"));
  useHotkey(HOTKEYS.land.combo, () => {
    if (env.landExplorationEnabled) ui.togglePanel("land");
  });
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
  // The brush that paints a scan's objects to select them (cesium/sceneSelect).
  useHotkey(HOTKEYS.brush.combo, (event) => {
    if (scene?.sceneSelect.togglePainting()) event.preventDefault();
  });
  // Poke and drag objects that move by skins (cesium/skinPoke.ts).
  useHotkey(HOTKEYS.poke.combo, () => useSkinPoke.getState().toggle());
  // The Time Lantern (cesium/TimeLantern.ts): Look back, or close it.
  useHotkey(HOTKEYS.lantern.combo, () => {
    const lantern = useLantern.getState();
    if (lantern.mode === "off" && !lantern.availability.ok) {
      const reason = UNAVAILABLE[lantern.availability.reason];
      useToasts
        .getState()
        .push({ tone: "info", title: "The Time Lantern is not available here", body: reason });
      return;
    }
    lantern.toggle();
  });
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
  // One press, one step back (`stepBack.ts`): the brush or the selected object before panels.
  useHotkey(HOTKEYS.escape.combo, (event) => {
    if (stepBack(scene)) event.preventDefault();
  });
  // Ctrl+Z / Ctrl+Shift+Z: what was hidden, painted, switched or removed (`state/history.ts`).
  useUndoHotkeys();
  return null;
}

function SettingsSheetLazy() {
  const open = useUi((s) => s.settingsOpen);
  if (!open) return null;
  return (
    <OnDemand label="Settings">
      <SettingsSheet />
    </OnDemand>
  );
}

/**
 * Where a component loaded on demand mounts: nothing while its chunk loads, and nothing (with a
 * log line) if the chunk fails to load, from a flaky network or a tab left open across a deploy,
 * so the failure stays its own rather than replacing the whole HUD.
 */
function OnDemand({ label, children }: { label: string; children: ReactNode }) {
  return (
    <ErrorBoundary quiet label={label}>
      <Suspense fallback={null}>{children}</Suspense>
    </ErrorBoundary>
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
 * - **right dock**: messages (toasts), what you selected (one card: a machine, a zone or an
 *   object of a scan, `SelectionCard`), the inspector.
 * - **center**: first-run welcome; otherwise the map.
 * - **strip**: controls for what is in view (representation and its load, dates, explore).
 * - **bar**: the status line on the left (fleet, agent, a degraded connection), developer
 *   readouts when they are on, and the compass, Earth and data credits in one pill on the
 *   right. On a phone the credits take a thin line of their own above the status line.
 * - **tabs**: on a phone only, Map / Plan / Fleet / More along the bottom.
 *
 * Things that float on purpose, over the regions, and close with Escape or a click away: the
 * command box's results, the site switcher, the agent's activity log, the scan's objects panel
 * (from the strip), the map's menu (a right-click or long press on the map), and on a phone the
 * More sheet (`data-hud-popover`).
 *
 * On narrow screens the docks and the drawer collapse into one bottom sheet; `data-sheet`
 * says which was touched last, and that one is shown (`state/layout.ts`).
 */
export function AppShell() {
  const sheet = useLayout((s) => s.focus);
  const view = useMission((s) => s.view);
  const changeAsked = useChangeDrape((s) => s.request !== null);
  useEffect(() => bindDockRules(), []);
  return (
    <>
      <GlobalHotkeys />
      {env.landExplorationEnabled && (
        <OnDemand label="Land bridges">
          <LandIdentityBridge />
          <LandBridge />
          <LandContextBridge />
          <LandRasterBridge />
          <LandFocusBridge />
          <KnowledgeFogBridge />
          <ShowOnceBridge />
          <LandMissionBridge />
        </OnDemand>
      )}
      <LanternBridge />
      <div className="hud" data-sheet={sheet ?? "none"} data-view={view} data-testid="hud">
        <ErrorBoundary inline label="Fleet overlay">
          <MissionOverlays />
        </ErrorBoundary>
        {env.landExplorationEnabled && (
          <OnDemand label="Land overlay">
            <LandOverlays />
          </OnDemand>
        )}
        <CompareSplit />
        <LanternRim />
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
            {/* Its own boundary: a Land chunk that fails to load leaves the other panels. */}
            {env.landExplorationEnabled && <LandSurfaceSlot />}
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
          <DescentArrival />
        </div>
        <aside className="hud-region hud-dock hud-dock--right" aria-label="Details">
          <div className="hud-stack hud-messages">
            <ErrorBoundary inline label="Simulated motion">
              <SimulatedBadge />
              <PokeBadge />
              <LivingLabPanel />
            </ErrorBoundary>
            <RelitSlot />
            <Toasts />
          </div>
          <div className="hud-stack hud-detail">
            {changeAsked && (
              <OnDemand label="Change map">
                <ChangeDrape />
              </OnDemand>
            )}
            {env.landExplorationEnabled && (
              <OnDemand label="Land agent">
                <LandAgent />
              </OnDemand>
            )}
            <ErrorBoundary inline label="Selection">
              <SelectionCard />
              <FeedsPanel />
            </ErrorBoundary>
            <ErrorBoundary inline label="Inspector">
              <Suspense fallback={null}>
                <InspectorPanel />
              </Suspense>
              <DevPanel />
            </ErrorBoundary>
            {env.landExplorationEnabled && (
              <ErrorBoundary inline label="Fog of knowing">
                <Suspense fallback={null}>
                  <KnowledgeCellCard />
                </Suspense>
              </ErrorBoundary>
            )}
          </div>
        </aside>
        <div className="hud-region hud-strip">
          {env.landExplorationEnabled && (
            <OnDemand label="Land controls">
              <LandStrip />
            </OnDemand>
          )}
          <ExploreHud />
          {/* Dates, then the lens over dated imagery, then how old what you know is: time
              first, the way the strip reads left to right. */}
          <TimelineControl />
          {/* How the splat scan is coloured: Photo, Confidence, Data age (CesiumJS draws the
              analytic ones). A slot here; the control loads when a splat site is active. */}
          <ViewsSlot />
          <ErrorBoundary inline label="Time Lantern">
            <TimeLanternControl />
          </ErrorBoundary>
          {env.landExplorationEnabled && (
            <ErrorBoundary inline label="Fog of knowing">
              <Suspense fallback={null}>
                <KnowledgeLegend />
              </Suspense>
            </ErrorBoundary>
          )}
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
      <ErrorBoundary inline label="Map menu">
        <MapContextMenu />
      </ErrorBoundary>
    </>
  );
}

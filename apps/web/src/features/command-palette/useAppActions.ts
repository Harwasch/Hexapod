import {
  Bookmark,
  ClipboardList,
  CloudUpload,
  Columns2,
  Compass,
  Database,
  Earth,
  Footprints,
  Gauge,
  Images,
  Keyboard,
  Layers,
  Link2,
  Map as MapIcon,
  MapPin,
  MapPinPlus,
  Plus,
  Ruler,
  Settings2,
  Sparkles,
  Square,
  Terminal,
  Tractor,
  type LucideIcon,
} from "lucide-react";

import { env } from "@/app/env";
import { HOTKEYS, hotkeyKeys, type HotkeyId } from "@/app/hotkeys";
import { useScene } from "@/cesium/SceneContext";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

import { openSiteSwitcher } from "../mission/siteSwitcher";

export interface AppAction {
  id: string;
  label: string;
  /** Other words it answers to in the command box. */
  keywords: string;
  icon: LucideIcon;
  /** The key that does the same from the map, shown beside it. */
  shortcut?: readonly string[];
  run: () => void;
}

function keys(id: HotkeyId): readonly string[] {
  return hotkeyKeys(HOTKEYS[id]);
}

/**
 * Everything the console can do in one step, for the command box's Actions group, each with
 * the key that does it from the map (`app/hotkeys.ts`). The keys stay bound in `AppShell`;
 * an action opens what its key toggles, because a command is asked for on purpose.
 */
export function useAppActions(): AppAction[] {
  const scene = useScene();
  const exploreMode = useUi((s) => s.exploreMode);
  const devToolsOpen = useSettings((s) => s.devToolsOpen);
  const devReadouts = useSettings((s) => s.devReadouts);
  const ui = useUi.getState;
  const mission = useMission.getState;

  const actions: AppAction[] = [
    {
      id: "map-view",
      label: "Map view",
      keywords: "world globe",
      icon: MapIcon,
      shortcut: keys("mapView"),
      run: () => mission().setView("map"),
    },
    {
      id: "plan-view",
      label: "Plan view",
      keywords: "plans missions",
      icon: ClipboardList,
      shortcut: keys("planView"),
      run: () => mission().setView("plan"),
    },
    {
      id: "fleet-view",
      label: "Fleet view",
      keywords: "machines robots",
      icon: Tractor,
      shortcut: keys("fleetView"),
      run: () => mission().setView("fleet"),
    },
    {
      id: "new-plan",
      label: "New plan",
      keywords: "draft mission agent goal",
      icon: Plus,
      run: () => {
        mission().openComposer();
        // The goal is written where the agent listens: back in this box, empty.
        window.dispatchEvent(new CustomEvent("twin:bar", { detail: "" }));
      },
    },
    {
      id: "upload",
      label: "Upload a capture",
      keywords: "add captures video photos splat phone",
      icon: CloudUpload,
      shortcut: keys("captures"),
      run: () => ui().openAdd("upload"),
    },
    {
      id: "add-data",
      label: "Link a hosted source",
      keywords: "add data link tiles geojson imagery stac ion layer",
      icon: Link2,
      run: () => ui().openAdd("link"),
    },
    {
      id: "add-site",
      label: "Add a site",
      keywords: "new site reality model register",
      icon: MapPinPlus,
      run: () => ui().openAdd("link"),
    },
    {
      id: "top-down",
      label: "Top-down view",
      keywords: "camera overhead",
      icon: Square,
      shortcut: keys("topDown"),
      run: () => scene?.camera.topDown(),
    },
    {
      id: "reset-north",
      label: "Reset north",
      keywords: "camera compass",
      icon: Compass,
      shortcut: keys("resetNorth"),
      run: () => scene?.camera.resetNorth(),
    },
    {
      id: "compare",
      label: "Compare layers",
      keywords: "split swipe",
      icon: Columns2,
      shortcut: keys("compare"),
      run: () => ui().openLayers("compare"),
    },
    {
      id: "explore",
      label: exploreMode ? "Leave walk / explore mode" : "Walk / explore mode",
      keywords: "ground first person",
      icon: Footprints,
      shortcut: keys("explore"),
      run: () => {
        if (!exploreMode) scene?.explore.setSpeed(useSettings.getState().exploreSpeed);
        ui().setExploreMode(!exploreMode);
      },
    },
    {
      id: "measure",
      label: "Measure",
      keywords: "distance area height elevation",
      icon: Ruler,
      shortcut: keys("measure"),
      run: () => ui().setPanel("measure"),
    },
    {
      id: "saved-views",
      label: "Saved views",
      keywords: "bookmarks camera save view",
      icon: Bookmark,
      shortcut: keys("bookmarks"),
      run: () => openSiteSwitcher("views"),
    },
    {
      id: "settings",
      label: "Settings",
      keywords: "preferences theme units quality",
      icon: Settings2,
      shortcut: keys("settings"),
      run: () => ui().setSettingsOpen(true),
    },
    {
      id: "help",
      label: "Help: keyboard shortcuts",
      keywords: "keys hotkeys",
      icon: Keyboard,
      shortcut: keys("shortcuts"),
      run: () => ui().setShortcutsOpen(true),
    },
    {
      id: "layers",
      label: "Layers",
      keywords: "basemap data favourites",
      icon: Layers,
      shortcut: keys("layers"),
      run: () => ui().openLayers("browse"),
    },
    {
      id: "sites",
      label: "Switch site",
      keywords: "sites catalog reality models projects",
      icon: MapPin,
      shortcut: keys("sites"),
      run: () => openSiteSwitcher("sites"),
    },
    {
      id: "scan-gallery",
      label: "Scan gallery",
      keywords: "scans viewer phone captures 3d",
      icon: Images,
      run: () => window.location.assign("/view.html"),
    },
    {
      id: "data-console",
      label: "Data console",
      keywords: "admin runs outputs storage jobs",
      icon: Database,
      run: () => window.location.assign("/admin.html"),
    },
    {
      id: "earth",
      label: "Earth view",
      keywords: "home globe zoom out",
      icon: Earth,
      shortcut: keys("home"),
      run: () => scene?.camera.flyHome(),
    },
    {
      id: "agent-activity",
      label: "Agent activity",
      keywords: "log stream thread",
      icon: Sparkles,
      shortcut: keys("agent"),
      run: () => ui().setActivityOpen(true),
    },
    {
      id: "dev-readouts",
      label: devReadouts ? "Hide developer readouts" : "Show developer readouts",
      keywords: "altitude scale renderer telemetry",
      icon: Gauge,
      run: () => useSettings.getState().set({ devReadouts: !devReadouts }),
    },
  ];
  if (env.devToolsEnabled)
    actions.push({
      id: "dev-tools",
      label: devToolsOpen ? "Close developer tools" : "Developer tools",
      keywords: "debug performance",
      icon: Terminal,
      shortcut: keys("devTools"),
      run: () => useSettings.getState().set({ devToolsOpen: !devToolsOpen }),
    });
  return actions;
}

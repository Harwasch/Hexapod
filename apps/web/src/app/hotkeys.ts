import { MOD_LABEL } from "@/lib/hotkeys";

/**
 * Every keyboard shortcut the console answers to, in one list.
 *
 * `GlobalHotkeys` (features/shell/AppShell) binds the app's own keys from it, the command box
 * shows each action's key from it, and the shortcut sheet (`?`) prints all of it. Keys the
 * scene handles itself — camera navigation (`cesium/KeyboardNavigator`) and explore mode
 * (`cesium/ExploreController`) — are listed with `scene: true`: shown, never bound here, so
 * the sheet is the whole truth without a second place to keep in step. So are the keys a
 * selected scan object answers to (`cesium/sceneSelect/SceneSelectController`): the brush is
 * bound here, the cycling keys are the controller's own.
 */
export type HotkeyGroup = "General" | "Views" | "Tools" | "Objects" | "Camera" | "Explore mode";

export interface Hotkey {
  /** As `useHotkey` takes it: "mod+k", "shift+?", "l". Empty for scene keys. */
  combo: string;
  label: string;
  group: HotkeyGroup;
  /** Shown instead of the formatted combo, for keys the scene handles. */
  keys?: readonly string[];
  /** Handled by the scene, not bound by the app. */
  scene?: boolean;
  /** Only in builds with developer tools (`env.devToolsEnabled`). */
  dev?: boolean;
}

export const HOTKEY_GROUPS: readonly HotkeyGroup[] = [
  "General",
  "Views",
  "Tools",
  "Objects",
  "Camera",
  "Explore mode",
];

export const HOTKEYS = {
  command: { combo: "mod+k", label: "Search, run an action or ask the agent", group: "General" },
  search: { combo: "/", label: "Search", group: "General" },
  shortcuts: { combo: "shift+?", label: "Keyboard shortcuts", group: "General" },
  agent: { combo: "a", label: "Agent activity", group: "General" },
  settings: { combo: ",", label: "Settings", group: "General" },
  escape: { combo: "escape", label: "Close, or step back", group: "General" },
  devTools: { combo: "d", label: "Developer tools", group: "General", dev: true },
  mapView: { combo: "1", label: "Map view", group: "Views" },
  planView: { combo: "2", label: "Plan view", group: "Views" },
  fleetView: { combo: "3", label: "Fleet view", group: "Views" },
  layers: { combo: "l", label: "Layers", group: "Tools" },
  measure: { combo: "m", label: "Measure", group: "Tools" },
  captures: { combo: "u", label: "Add: upload a capture", group: "Tools" },
  compare: { combo: "c", label: "Compare layers", group: "Tools" },
  sites: { combo: "s", label: "Switch site", group: "Tools" },
  bookmarks: { combo: "v", label: "Saved views", group: "Tools" },
  brush: { combo: "b", label: "Paint to select objects (brush)", group: "Objects" },
  cycleObject: {
    combo: "",
    keys: ["[", "]"],
    label: "Previous / next candidate (Tab from the card or the map)",
    group: "Objects",
    scene: true,
  },
  paintModifiers: {
    combo: "",
    keys: ["Shift", "Alt"],
    label: "While painting: add to / take away from the painted area",
    group: "Objects",
    scene: true,
  },
  altWheel: {
    combo: "",
    keys: ["Alt", "Wheel"],
    label: "Size the brush, or cycle candidates",
    group: "Objects",
    scene: true,
  },
  resetNorth: { combo: "n", label: "Reset north", group: "Camera" },
  topDown: { combo: "t", label: "Top-down view", group: "Camera" },
  home: { combo: "h", label: "Earth view", group: "Camera" },
  explore: { combo: "g", label: "Walk / explore mode", group: "Camera" },
  pan: { combo: "", keys: ["←", "↑", "↓", "→"], label: "Pan", group: "Camera", scene: true },
  orbit: {
    combo: "",
    keys: ["Shift", "←", "↑", "↓", "→"],
    label: "Turn and tilt around the view centre",
    group: "Camera",
    scene: true,
  },
  zoom: { combo: "", keys: ["+", "−"], label: "Zoom in / out", group: "Camera", scene: true },
  walk: {
    combo: "",
    keys: ["W", "A", "S", "D"],
    label: "Move",
    group: "Explore mode",
    scene: true,
  },
  fly: {
    combo: "",
    keys: ["F"],
    label: "Switch between walking and flying",
    group: "Explore mode",
    scene: true,
  },
  jump: {
    combo: "",
    keys: ["Space"],
    label: "Jump; rise while flying",
    group: "Explore mode",
    scene: true,
  },
  sprint: { combo: "", keys: ["Shift"], label: "Sprint", group: "Explore mode", scene: true },
} as const satisfies Record<string, Hotkey>;

export type HotkeyId = keyof typeof HOTKEYS;

const KEY_NAMES: Record<string, string> = {
  mod: MOD_LABEL,
  shift: "Shift",
  alt: "Alt",
  escape: "Esc",
};

/** The keys to show for a shortcut, one token per key cap: "mod+k" → ["⌘", "K"]. */
export function hotkeyKeys(hotkey: Hotkey): string[] {
  if (hotkey.keys) return [...hotkey.keys];
  const parts = hotkey.combo.split("+");
  const key = parts.at(-1) ?? "";
  // "?" already means Shift on the keyboards that need it; "Shift ?" would read as two keys.
  const modifiers = parts.slice(0, -1).filter((part) => !(part === "shift" && key === "?"));
  return [
    ...modifiers.map((part) => KEY_NAMES[part] ?? part),
    KEY_NAMES[key] ?? (key.length === 1 ? key.toUpperCase() : key),
  ];
}

/** Every shortcut to list, grouped in display order; developer keys only when they are on. */
export function hotkeySheet(devTools: boolean): { group: HotkeyGroup; hotkeys: Hotkey[] }[] {
  const all = Object.values(HOTKEYS) as Hotkey[];
  return HOTKEY_GROUPS.map((group) => ({
    group,
    hotkeys: all.filter((hotkey) => hotkey.group === group && (devTools || !hotkey.dev)),
  })).filter((section) => section.hotkeys.length > 0);
}

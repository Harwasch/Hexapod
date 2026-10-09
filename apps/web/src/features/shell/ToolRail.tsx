import { LandPlot, Layers, Plus, Ruler, Settings2, type LucideIcon } from "lucide-react";

import { GlassButton, GlassPanel, GlassTooltip } from "@twin/ui";

import { HOTKEYS, hotkeyKeys, type HotkeyId } from "@/app/hotkeys";
import { env } from "@/app/env";
import { useUi, type ToolPanel } from "@/state/ui";

interface Tool {
  id: ToolPanel | "settings";
  label: string;
  icon: LucideIcon;
  hotkey: HotkeyId;
}

/**
 * The four tools, in rail order. Everything that used to have its own icon is inside one of
 * them or in the site switcher, and every one of those is still a command-box action:
 * Compare is a mode of Layers; Captures and "Add data" are the two tabs of Add; Sites and
 * Saved views are in the site switcher; the developer console is in Settings › Advanced.
 */
const TOOLS: readonly Tool[] = [
  ...(env.landExplorationEnabled
    ? [{ id: "land" as const, label: "Land", icon: LandPlot, hotkey: "land" as const }]
    : []),
  { id: "layers", label: "Layers", icon: Layers, hotkey: "layers" },
  { id: "measure", label: "Measure", icon: Ruler, hotkey: "measure" },
  { id: "add", label: "Add", icon: Plus, hotkey: "captures" },
  { id: "settings", label: "Settings", icon: Settings2, hotkey: "settings" },
];

/**
 * The tool buttons, labelled: an icon with its word under it, so nothing on the rail has to
 * be hovered to be understood. The same buttons fill the phone's More sheet (`layout="sheet"`),
 * where `onPick` closes the sheet behind the choice.
 */
export function ToolButtons({
  layout = "rail",
  onPick,
}: {
  layout?: "rail" | "sheet";
  onPick?: () => void;
}) {
  const activePanel = useUi((s) => s.activePanel);
  const settingsOpen = useUi((s) => s.settingsOpen);
  const togglePanel = useUi((s) => s.togglePanel);
  const setSettingsOpen = useUi((s) => s.setSettingsOpen);
  return (
    <>
      {TOOLS.map(({ id, label, icon: Icon, hotkey }) => {
        const active = id === "settings" ? settingsOpen : activePanel === id;
        const button = (
          <GlassButton
            variant="ghost"
            className="tool-rail__tool"
            active={active}
            onClick={() => {
              if (id === "settings") setSettingsOpen(true);
              else togglePanel(id);
              onPick?.();
            }}
            data-testid={layout === "rail" ? `tool-${id}` : `more-${id}`}
          >
            <Icon size={18} aria-hidden="true" />
            <span className="tool-rail__label">{label}</span>
          </GlassButton>
        );
        // The tooltip adds only the key: the word is already on the button.
        return layout === "rail" ? (
          <GlassTooltip
            key={id}
            content={label}
            shortcut={hotkeyKeys(HOTKEYS[hotkey]).join(" ")}
            side="right"
          >
            {button}
          </GlassTooltip>
        ) : (
          <span key={id} className="tool-rail__cell">
            {button}
          </span>
        );
      })}
    </>
  );
}

export function ToolRail() {
  return (
    <GlassPanel
      strong
      pill
      className="tool-rail"
      role="toolbar"
      aria-label="Tools"
      aria-orientation="vertical"
    >
      <ToolButtons />
    </GlassPanel>
  );
}

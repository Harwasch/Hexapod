import { Bookmark, Columns2, Layers, MapPin, Plus, Ruler, Settings2, Terminal } from "lucide-react";

import { GlassButton, GlassPanel, GlassTooltip } from "@twin/ui";

import { env } from "@/app/env";
import { useSettings } from "@/state/settings";
import { useUi, type ToolPanel } from "@/state/ui";

const tools: { id: ToolPanel; label: string; icon: typeof Layers; shortcut: string }[] = [
  { id: "layers", label: "Layers", icon: Layers, shortcut: "L" },
  { id: "sites", label: "Sites", icon: MapPin, shortcut: "S" },
  { id: "measure", label: "Measure", icon: Ruler, shortcut: "M" },
  { id: "compare", label: "Compare", icon: Columns2, shortcut: "C" },
  { id: "bookmarks", label: "Bookmarks", icon: Bookmark, shortcut: "B" },
];

export function ToolRail() {
  const activePanel = useUi((s) => s.activePanel);
  const togglePanel = useUi((s) => s.togglePanel);
  const setAddDataOpen = useUi((s) => s.setAddDataOpen);
  const setSettingsOpen = useUi((s) => s.setSettingsOpen);
  const devToolsOpen = useSettings((s) => s.devToolsOpen);
  const setSettings = useSettings((s) => s.set);
  return (
    <GlassPanel
      strong
      pill
      className="tool-rail"
      role="toolbar"
      aria-label="Tools"
      aria-orientation="vertical"
    >
      {tools.map(({ id, label, icon: Icon, shortcut }) => (
        <GlassTooltip key={id} content={label} shortcut={shortcut} side="right">
          <GlassButton
            iconOnly
            variant="ghost"
            aria-label={label}
            active={activePanel === id}
            onClick={() => togglePanel(id)}
            data-testid={`tool-${id}`}
          >
            <Icon size={18} aria-hidden="true" />
          </GlassButton>
        </GlassTooltip>
      ))}
      <div className="tool-rail__divider" role="separator" />
      <GlassTooltip content="Add data" side="right">
        <GlassButton
          iconOnly
          variant="ghost"
          aria-label="Add data"
          onClick={() => setAddDataOpen(true)}
          data-testid="tool-add-data"
        >
          <Plus size={18} aria-hidden="true" />
        </GlassButton>
      </GlassTooltip>
      <GlassTooltip content="Settings" shortcut="," side="right">
        <GlassButton
          iconOnly
          variant="ghost"
          aria-label="Settings"
          onClick={() => setSettingsOpen(true)}
          data-testid="tool-settings"
        >
          <Settings2 size={18} aria-hidden="true" />
        </GlassButton>
      </GlassTooltip>
      {env.devToolsEnabled && (
        <GlassTooltip content="Developer tools" shortcut="D" side="right">
          <GlassButton
            iconOnly
            variant="ghost"
            aria-label="Developer tools"
            active={devToolsOpen}
            onClick={() => setSettings({ devToolsOpen: !devToolsOpen })}
            data-testid="tool-dev"
          >
            <Terminal size={18} aria-hidden="true" />
          </GlassButton>
        </GlassTooltip>
      )}
    </GlassPanel>
  );
}

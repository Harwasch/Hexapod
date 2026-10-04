import { ClipboardList, Map as MapIcon, MoreHorizontal, Tractor, X } from "lucide-react";
import { useEffect, useId, useRef } from "react";

import { GlassPanel } from "@twin/ui";

import { useLayout } from "@/state/layout";
import { useMission, type MissionView } from "@/state/mission";
import { useUi } from "@/state/ui";

import { ToolButtons } from "./ToolRail";

const VIEWS: { id: MissionView; label: string; icon: typeof MapIcon }[] = [
  { id: "map", label: "Map", icon: MapIcon },
  { id: "plan", label: "Plan", icon: ClipboardList },
  { id: "fleet", label: "Fleet", icon: Tractor },
];

/**
 * The phone's bottom bar: Map, Plan and Fleet where a thumb reaches them, and More for the
 * four tools the rail carries on a wider screen (Layers, Measure, Add, Settings). Shown only
 * at phone widths (`app.css`); the view tabs at the top right do this job everywhere else.
 *
 * Tapping the view that is already showing brings its sheet back to the front: on a phone the
 * map has one sheet, and a machine picked from the Fleet table puts its card in front of it.
 */
export function PhoneTabBar() {
  const view = useMission((s) => s.view);
  const setView = useMission((s) => s.setView);
  const moreOpen = useUi((s) => s.moreOpen);
  const setMoreOpen = useUi((s) => s.setMoreOpen);
  const root = useRef<HTMLElement>(null);
  const sheetId = useId();

  // The More sheet is a popover: a press anywhere else closes it.
  useEffect(() => {
    if (!moreOpen) return;
    const onDown = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setMoreOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [moreOpen, setMoreOpen]);

  return (
    <nav className="phone-tabs" aria-label="Views" ref={root} data-testid="phone-tabs">
      {moreOpen && (
        <GlassPanel
          strong
          className="phone-more"
          id={sheetId}
          role="group"
          aria-label="More tools"
          data-hud-popover=""
          data-testid="phone-more"
        >
          <ToolButtons layout="sheet" onPick={() => setMoreOpen(false)} />
        </GlassPanel>
      )}
      <GlassPanel strong className="phone-tabs__bar">
        {VIEWS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            className={`phone-tabs__tab ${view === id ? "is-on" : ""}`}
            aria-pressed={view === id}
            onClick={() => {
              setMoreOpen(false);
              setView(id);
              if (id !== "map") useLayout.getState().setFocus("left");
            }}
            data-testid={`phone-tab-${id}`}
          >
            <Icon size={18} aria-hidden="true" />
            <span>{label}</span>
          </button>
        ))}
        <button
          type="button"
          className={`phone-tabs__tab ${moreOpen ? "is-on" : ""}`}
          aria-expanded={moreOpen}
          aria-controls={moreOpen ? sheetId : undefined}
          onClick={() => setMoreOpen(!moreOpen)}
          data-testid="phone-tab-more"
        >
          {moreOpen ? (
            <X size={18} aria-hidden="true" />
          ) : (
            <MoreHorizontal size={18} aria-hidden="true" />
          )}
          <span>More</span>
        </button>
      </GlassPanel>
    </nav>
  );
}

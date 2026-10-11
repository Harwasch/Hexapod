import { useRef, type KeyboardEvent } from "react";

import { GlassPanel } from "@twin/ui";

import { LENSES, useLens } from "./lenses";

/** The lenses along the bottom of the map; one is on at a time. */
export function LensDock() {
  const lens = useLens((s) => s.lens);
  const setLens = useLens((s) => s.setLens);
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const key = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    const next = (index + step + LENSES.length) % LENSES.length;
    const chosen = LENSES[next];
    if (!chosen) return;
    setLens(chosen.id);
    buttons.current[next]?.focus();
  };
  return (
    <GlassPanel
      strong
      pill
      className="land-dock"
      role="toolbar"
      aria-label="Ways to look at this land"
      data-testid="land-dock"
    >
      {LENSES.map((meta, index) => (
        <button
          key={meta.id}
          ref={(element) => {
            buttons.current[index] = element;
          }}
          type="button"
          aria-pressed={lens === meta.id}
          tabIndex={lens === meta.id ? 0 : -1}
          onKeyDown={(event) => key(event, index)}
          onClick={() => setLens(meta.id)}
          data-testid={`land-lens-${meta.id}`}
        >
          <span className="land-dock__dot" style={{ background: meta.color }} aria-hidden="true" />
          {meta.label}
        </button>
      ))}
    </GlassPanel>
  );
}

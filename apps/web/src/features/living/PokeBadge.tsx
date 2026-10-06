import { Hand, X } from "lucide-react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { useSkinPoke } from "@/state/skinPoke";

/**
 * Says the poke tool is on (`cesium/skinPoke.ts`, `K`): what a press does, and while something
 * is held or ringing, what it is and how fast it rings. The motion is simulated -- a model of
 * how the object bends -- and the badge says so, as the wind's badge does. A close button (and
 * `K` or Escape) turns the tool off.
 */
export function PokeBadge() {
  const active = useSkinPoke((s) => s.active);
  const grabbed = useSkinPoke((s) => s.grabbed);
  const setActive = useSkinPoke((s) => s.setActive);
  if (!active) return null;
  const detail = grabbed
    ? `${grabbed.label}: ${
        grabbed.movable ? "moves whole" : "rooted"
      }${grabbed.hz !== null ? `, rings at ${grabbed.hz.toFixed(grabbed.hz < 1 ? 2 : 1)} Hz` : ""}. Simulated.`
    : "Press on a plant or object that moves and drag; let go to see it ring. The camera still moves from anywhere else.";
  return (
    <GlassPanel
      strong
      compact
      className="living-badge"
      role="status"
      aria-label="Poke tool"
      data-testid="poke-badge"
    >
      <Hand className="notice__icon" size={16} aria-hidden="true" />
      <div className="notice__text">
        <strong>Poke</strong>
        <span>{detail}</span>
      </div>
      <GlassButton
        variant="ghost"
        size="sm"
        aria-label="Stop poking"
        title="Stop poking (K)"
        onClick={() => setActive(false)}
      >
        <X size={14} aria-hidden="true" />
      </GlassButton>
    </GlassPanel>
  );
}

import { ScanEye } from "lucide-react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { AS_CAPTURED_TEXT } from "@/lib/asCapturedText";
import { useAsCaptured } from "@/state/asCaptured";

/**
 * Said in the HUD strip for as long as As captured is on (state/asCaptured.ts), whatever the
 * site, with the way back: a viewer who arrived on a `?asCaptured=1` link may never have seen
 * the switch (Splat view, ViewsControl). Loaded on demand (ViewsSlot.tsx `AsCapturedSlot`).
 */
export function AsCapturedChip() {
  const set = useAsCaptured((s) => s.set);
  return (
    <GlassPanel
      strong
      pill
      className="views as-captured"
      role="status"
      aria-label={AS_CAPTURED_TEXT.label}
      data-testid="as-captured-chip"
    >
      <ScanEye size={14} className="glass-muted" aria-hidden="true" />
      <span>{AS_CAPTURED_TEXT.chip}</span>
      <GlassButton size="sm" variant="ghost" onClick={() => set(false)}>
        {AS_CAPTURED_TEXT.off}
      </GlassButton>
    </GlassPanel>
  );
}

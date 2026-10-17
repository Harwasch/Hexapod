import { lazy, Suspense } from "react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { useUi } from "@/state/ui";

const LandPanel = lazy(() =>
  import("../land/LandPanel").then((module) => ({ default: module.LandPanel })),
);

/**
 * The Land panel in the left dock, behind a boundary of its own. Its chunk loads as the shell
 * mounts, and when that load fails (a network blip, a tab left open across a deploy) Layers,
 * Measure and Add beside it, in the main bundle, keep working. A rejected chunk is not asked
 * for again, so opening Land then says to reload rather than offering a retry that cannot work.
 */
export function LandPanelSlot() {
  const open = useUi((s) => s.activePanel === "land");
  return (
    <ErrorBoundary
      label="Land panel"
      fallback={
        open ? (
          <GlassPanel padding="sm" compact role="alert" data-testid="land-panel-unavailable">
            <p className="glass-title">Land could not load</p>
            <p className="glass-muted" style={{ fontSize: "var(--text-xs)" }}>
              Its code did not download. Reload the page to load it again.
            </p>
            <GlassButton size="sm" onClick={() => window.location.reload()}>
              Reload
            </GlassButton>
          </GlassPanel>
        ) : null
      }
    >
      <Suspense fallback={null}>
        <LandPanel />
      </Suspense>
    </ErrorBoundary>
  );
}

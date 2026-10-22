import { lazy, Suspense } from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { useAsCaptured } from "@/state/asCaptured";
import { useSites } from "@/state/sites";

const ViewsControl = lazy(() =>
  import("./ViewsControl").then((m) => ({ default: m.ViewsControl })),
);
const AsCapturedChip = lazy(() =>
  import("./ViewsControl").then((m) => ({ default: m.AsCapturedChip })),
);

/**
 * The splat view control's place in the HUD strip: rendered while a splat scan is the active
 * site's model, the control itself (`ViewsControl`, its legend) loaded on demand then, so the
 * globe's first load carries only this. Beside it, As captured's chip (state/asCaptured.ts),
 * on any site while it is on, from the same chunk.
 */
export function ViewsSlot() {
  const activeSiteId = useSites((s) => s.activeSiteId);
  const splat = useSites(
    (s) => activeSiteId !== null && s.representation[activeSiteId] === "gaussian-splat",
  );
  const asCaptured = useAsCaptured((s) => s.on);
  const views = activeSiteId !== null && splat;
  if (!views && !asCaptured) return null;
  return (
    <ErrorBoundary inline label="Splat view">
      {/* Each its own: the chip loading must not blank the control that turned it on. */}
      {views && (
        <Suspense fallback={null}>
          <ViewsControl siteId={activeSiteId} />
        </Suspense>
      )}
      {asCaptured && (
        <Suspense fallback={null}>
          <AsCapturedChip />
        </Suspense>
      )}
    </ErrorBoundary>
  );
}

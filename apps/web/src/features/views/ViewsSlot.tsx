import { lazy, Suspense } from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { useSites } from "@/state/sites";

const ViewsControl = lazy(() =>
  import("./ViewsControl").then((m) => ({ default: m.ViewsControl })),
);

/**
 * The splat view control's place in the HUD strip: rendered while a splat scan is the active
 * site's model, the control itself (`ViewsControl`, its legend) loaded on demand then, so the
 * globe's first load carries only this.
 */
export function ViewsSlot() {
  const activeSiteId = useSites((s) => s.activeSiteId);
  const splat = useSites(
    (s) => activeSiteId !== null && s.representation[activeSiteId] === "gaussian-splat",
  );
  if (!activeSiteId || !splat) return null;
  return (
    <ErrorBoundary inline label="Splat view">
      <Suspense fallback={null}>
        <ViewsControl siteId={activeSiteId} />
      </Suspense>
    </ErrorBoundary>
  );
}

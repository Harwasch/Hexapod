import { lazy, Suspense } from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { useAsCaptured } from "@/state/asCaptured";
import { useLighting } from "@/state/lighting";
import { useSites } from "@/state/sites";

const ViewsControl = lazy(() =>
  import("./ViewsControl").then((m) => ({ default: m.ViewsControl })),
);
const AsCapturedChip = lazy(() =>
  import("./ViewsControl").then((m) => ({ default: m.AsCapturedChip })),
);
const RelitBadge = lazy(() => import("./RelitBadge").then((m) => ({ default: m.RelitBadge })));

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

/**
 * Sunlight's badge in the HUD's message stack (ADR 0019): the badge (and the clock it holds
 * for "At capture") loaded only once the sun is turned on; it decides itself whether the
 * relight can be drawn.
 */
export function RelitSlot() {
  const on = useLighting((s) => s.sun !== "off");
  return on ? (
    <ErrorBoundary inline label="Sunlight">
      <Suspense fallback={null}>
        <RelitBadge />
      </Suspense>
    </ErrorBoundary>
  ) : null;
}

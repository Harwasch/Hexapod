import { lazy, Suspense } from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { useChangeDrape } from "@/state/changeDrape";

const ChangeDrape = lazy(() => import("./ChangeDrape").then((m) => ({ default: m.ChangeDrape })));

/**
 * The change drape's place in the details dock: rendered while a change map is asked for
 * (Compare › Dates), the drape and its legend (`ChangeDrape`) loaded on demand then, so the
 * globe's first load carries only this.
 */
export function ChangeDrapeSlot() {
  const asked = useChangeDrape((s) => s.request !== null);
  return asked ? (
    <ErrorBoundary inline label="Change map">
      <Suspense fallback={null}>
        <ChangeDrape />
      </Suspense>
    </ErrorBoundary>
  ) : null;
}

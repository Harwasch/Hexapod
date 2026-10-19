import { useMemo } from "react";

import { useLensLayer, useLensPins } from "../../useLensMap";
import { avoidLayer, stepPins, stepsLayer, workAreaLayer } from "./plansMap";
import type { Action } from "./planModel";

/** What a plan looks like on the map: its work area, each step's area, the areas to avoid. */
export function usePlanOnMap(action: Action | null, onStep?: (id: string) => void) {
  const work = useMemo(
    () => (action ? workAreaLayer(action.effectiveBoundary, action.title) : null),
    [action],
  );
  const steps = useMemo(() => (action ? stepsLayer(action.steps) : null), [action]);
  const avoid = useMemo(() => (action ? avoidLayer(action.exclusions) : null), [action]);
  useLensLayer(work);
  useLensLayer(steps);
  useLensLayer(avoid);
  const pins = useMemo(
    () => (action ? stepPins(action.steps, action.effectiveBoundary, onStep) : null),
    // The pin callback only flashes a row; a new closure each render must not re-pin.
    [action],
  );
  useLensPins("plans", pins);
}

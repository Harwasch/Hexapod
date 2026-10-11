import { useEffect } from "react";

import { useLandContext, type LandContextLayer } from "@/state/landContext";
import { useLandMap, type MapPin } from "@/state/landMap";

/**
 * Shows a lens's layer on the map while the lens is open, and takes it off when it closes.
 * Pass a memoised layer (or null to show nothing); its id should start with the lens id.
 */
export function useLensLayer(layer: LandContextLayer | null): void {
  useEffect(() => {
    if (!layer) return;
    useLandContext.getState().setLayer(layer);
    return () => useLandContext.getState().removeLayer(layer.id);
  }, [layer]);
}

/** Shows a lens's pins while it is open. Pass a memoised list; `owner` is usually the lens id. */
export function useLensPins(owner: string, pins: MapPin[] | null): void {
  useEffect(() => {
    if (!pins?.length) {
      useLandMap.getState().clearPins(owner);
      return;
    }
    useLandMap.getState().setPins(owner, pins);
    return () => useLandMap.getState().clearPins(owner);
  }, [owner, pins]);
}

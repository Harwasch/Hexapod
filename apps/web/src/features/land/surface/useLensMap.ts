import { useEffect } from "react";

import { useLandContext, type LandContextLayer } from "@/state/landContext";
import { useLandMap, type MapPin } from "@/state/landMap";

/**
 * Shows a lens's layer on the map while the lens is open, and takes it off when it closes.
 * Pass a memoised layer (or null to show nothing); its id should start with the lens id.
 */
export function useLensLayer(layer: LandContextLayer | null): void {
  // A changed layer replaces the shown one in place (a selection recolours one feature);
  // only a closed lens, a cleared layer or a new id takes it off the map.
  const id = layer?.id ?? null;
  useEffect(() => {
    if (layer) useLandContext.getState().setLayer(layer);
  }, [layer]);
  useEffect(() => {
    if (!id) return;
    return () => useLandContext.getState().removeLayer(id);
  }, [id]);
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

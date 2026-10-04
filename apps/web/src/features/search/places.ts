import { useEffect, useState } from "react";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { useScene } from "@/cesium/SceneContext";
import type { GeocodeResult } from "@/cesium/types";
import { describeError } from "@/lib/log";

/** Fewer letters than this find half the world; the geocoder is not asked. */
export const PLACE_MIN_CHARS = 2;
/** Typing pauses this long before the geocoder is asked, so a word costs one request. */
export const PLACE_DEBOUNCE_MS = 280;

export interface PlaceSearch {
  places: GeocodeResult[];
  busy: boolean;
  /** Why the geocoder could not answer, in words; null when it did (even with nothing). */
  error: string | null;
  /**
   * The words `places` (and `error`) answer, trimmed. While the next words' search is pending
   * -- the debounce, then the request -- the last answer is kept, and this says it is stale.
   */
  query: string;
}

/**
 * Places for a query from the scene's geocoder, debounced, the stale request aborted when
 * the query changes. Places only: catalog sites and everything else the command box finds
 * are matched locally (`commandResults.ts`).
 */
export function usePlaceSearch(query: string, limit = 5): PlaceSearch {
  const scene = useScene();
  const [state, setState] = useState<PlaceSearch>({
    places: [],
    busy: false,
    error: null,
    query: "",
  });
  useEffect(() => {
    const trimmed = query.trim();
    const controller = new AbortController();
    if (!scene || trimmed.length < PLACE_MIN_CHARS) {
      setState({ places: [], busy: false, error: null, query: trimmed });
      return () => controller.abort();
    }
    setState((s) => ({ ...s, busy: true }));
    const timer = setTimeout(() => {
      scene.geocoder
        .search(trimmed, controller.signal)
        .then((results) => {
          if (!controller.signal.aborted)
            setState({ places: results.slice(0, limit), busy: false, error: null, query: trimmed });
        })
        .catch((error: unknown) => {
          if (!controller.signal.aborted)
            setState({ places: [], busy: false, error: describeError(error), query: trimmed });
        });
    }, PLACE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query, scene, limit]);
  return state;
}

/** Flies to a geocoded place: its bounding box when it has one, else a 2.5 km oblique view. */
export function flyToPlace(scene: CesiumSceneManager | null, place: GeocodeResult): void {
  if (!scene) return;
  const d = place.destination;
  if (d.kind === "rectangle") scene.camera.flyToRectangle(d.west, d.south, d.east, d.north);
  else scene.camera.flyTo(d.longitude, d.latitude, 2500, { pitch: -45 });
}

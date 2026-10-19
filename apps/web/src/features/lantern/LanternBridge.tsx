import { useEffect, useRef } from "react";

import { useScene } from "@/cesium/SceneContext";
import { useLand } from "@/state/land";
import { useLantern, type LanternBbox } from "@/state/lantern";
import { useViewer } from "@/state/viewer";

import { useDatedSources } from "./useDatedSources";

/** "Around the view" means this far each way from its centre (m), when nothing else says where. */
const AROUND_VIEW_M = 1_500;

function contains([west, south, east, north]: LanternBbox, longitude: number, latitude: number) {
  return longitude >= west && longitude <= east && latitude >= south && latitude <= north;
}

/**
 * Keeps the scene's Time Lantern (cesium/TimeLantern.ts) and the HUD's store in step, both ways:
 * the sources, the area and what the HUD asks for go down; what the scene decided (Alt held,
 * the wheel, Esc, being forced off) comes back. Mounted once. No Cesium object in React.
 */
export function LanternBridge() {
  const scene = useScene();
  const { sources, listed } = useDatedSources();
  const mode = useLantern((s) => s.mode);
  const index = useLantern((s) => s.index);
  const override = useLantern((s) => s.bboxOverride);
  const land = useLand((s) => s.active);
  const landId = land?.id ?? null;
  const shownLand = useRef(landId);

  useEffect(() => {
    if (!scene) return;
    const lantern = useLantern.getState();
    lantern.setAvailability(scene.lantern.available);
    return scene.events.on("lantern", (frame) => {
      const store = useLantern.getState();
      store.setAvailability(frame.availability);
      if (frame.mode !== store.mode) store.setMode(frame.mode);
      if (frame.index !== store.index) store.select(frame.index);
    });
  }, [scene]);

  // A land selected after a descent chose an area is what the lantern looks back on now: the
  // land just saved from "Outline this land", or another chosen elsewhere. An area a descent
  // chooses while a land is selected still holds (it is the newer choice).
  useEffect(() => {
    if (shownLand.current === landId) return;
    shownLand.current = landId;
    if (landId && useLantern.getState().bboxOverride) useLantern.getState().setBbox(null);
  }, [landId]);

  // Where the camera is, coarsely: a move of a few hundred metres asks whether the area still
  // holds the view.
  const cell = useViewer(
    (s) => `${Math.round(s.camera.longitude * 200)},${Math.round(s.camera.latitude * 200)}`,
  );

  // Opened with no land selected: look back around the view's centre. An area a descent chose
  // holds while the view stays on it; panned away, the view's centre takes over. The area is
  // re-centred only once the view's centre leaves it: the lens roams half as far again around
  // it (TimeLantern's reach), and every new area is a new catalog and new layers.
  useEffect(() => {
    if (!scene || mode === "off" || land) return;
    const centre = scene.camera.viewCenter();
    if (!centre) return;
    const store = useLantern.getState();
    if (override) {
      if (contains(override, centre.longitude, centre.latitude)) return;
      store.setBbox(null);
    }
    const dLat = AROUND_VIEW_M / 111_320;
    const dLon =
      AROUND_VIEW_M / (111_320 * Math.max(Math.cos((centre.latitude * Math.PI) / 180), 0.01));
    const current = store.autoBbox;
    if (current && contains(current, centre.longitude, centre.latitude)) return;
    store.setAutoBbox([
      centre.longitude - dLon,
      centre.latitude - dLat,
      centre.longitude + dLon,
      centre.latitude + dLat,
    ]);
  }, [scene, mode, override, land, cell]);

  // The sources with the area they were listed for: while the next area's list loads, the
  // last one (and its layers) stays on the lens.
  useEffect(() => {
    if (!scene) return;
    const [west = 0, south = 0, east = 0, north = 0] = listed ?? [];
    scene.lantern.setSources(sources, listed ? { west, south, east, north } : null);
  }, [scene, sources, listed]);

  useEffect(() => {
    scene?.lantern.setMode(mode);
  }, [scene, mode]);

  // What the HUD chose. A new list is not a choice: the scene keeps the source it showed (or
  // the nearest of its kind in time) and the store follows it through the "lantern" frame.
  useEffect(() => {
    scene?.lantern.select(index);
  }, [scene, index]);

  return null;
}

import { useEffect } from "react";

import { useLayers as useLayerCatalog, useSite, useSites as useSiteCatalog } from "@/api/queries";
import { api, unwrap } from "@/api/client";
import { builtinDemoSite } from "@/api/fallback";
import { anywhereProject } from "@/missions/anywhere";
import { reshapedZone } from "@/missions/areas";
import { DemoMissionProvider } from "@/missions/demo";
import { siteProject } from "@/missions/siteProject";
import { useLayers } from "@/state/layers";
import { useMission } from "@/state/mission";
import { useMeasurements } from "@/state/measurements";
import { useSelection } from "@/state/selection";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useToasts } from "@/state/toasts";
import { useUi } from "@/state/ui";
import { useViewer } from "@/state/viewer";

import type { Site } from "@twin/contracts";

import { useScene } from "./SceneContext";

/**
 * The only place where Cesium events flow into React state and React
 * settings flow back into the scene. Keeping it in one component makes the
 * boundary explicit and easy to audit.
 */
export function SceneBridge() {
  const scene = useScene();
  const layerCatalog = useLayerCatalog();
  const siteCatalog = useSiteCatalog();

  // Cesium → React
  useEffect(() => {
    if (!scene) return;
    const viewer = useViewer.getState();
    const layers = useLayers.getState();
    const sites = useSites.getState();
    const offs = [
      scene.events.on("status", ({ status, message }) =>
        useViewer.getState().setStatus(status, message ?? null),
      ),
      scene.events.on("token", (token) => useViewer.getState().setTokenState(token)),
      scene.events.on("camera", (pose) => viewer.setCamera(pose)),
      scene.events.on("motion", (moving) =>
        document.documentElement.toggleAttribute("data-moving", moving),
      ),
      scene.events.on("performance", (perf) => viewer.setPerformance(perf)),
      scene.events.on("layer", ({ id, patch }) => layers.update(id, patch)),
      scene.events.on("asset", ({ id, patch }) => sites.updateAsset(id, patch)),
      scene.events.on("site-near", (id) => sites.setNearSite(id)),
      scene.events.on("site-active", (id) => sites.setActiveSite(id)),
      scene.events.on("representation", ({ siteId, representation }) =>
        sites.setRepresentation(siteId, representation),
      ),
      scene.events.on("selection", (selection) => {
        useSelection.getState().setSelection(selection);
        if (selection) useUi.getState().setInspectorOpen(true);
      }),
      scene.events.on("hover", (info) => useSelection.getState().setHoverInfo(info)),
      scene.events.on("measurement", (m) => useMeasurements.getState().upsert(m)),
      scene.events.on("measurement-mode", () => useUi.getState().setMeasureMode(null)),
      scene.events.on("toast", (toast) => useToasts.getState().push(toast)),
      scene.events.on("world", (label) => viewer.setWorldLabel(label)),
      scene.events.on("tilesets", () =>
        viewer.setActiveTilesets([
          ...scene.layers.activeTilesetLabels(),
          ...scene.sites.activeTilesetLabels(),
        ]),
      ),
      scene.events.on("explore", (on) => useUi.getState().setExploreMode(on)),
      scene.events.on("mission-select", ({ kind, id }) => {
        useMission.getState().select({ kind, id });
        scene.mission.setSelectedZone(kind === "zone" ? id : null);
      }),
      // A corner dragged on the map reshapes the area in the store; the map follows it.
      scene.events.on("area-edit", ({ zoneId, footprint }) => {
        const mission = useMission.getState();
        const zone = mission.project?.zones.find((z) => z.id === zoneId);
        if (!mission.project || !zone) return;
        mission.addArea(mission.project.id, reshapedZone(zone, footprint));
      }),
    ];
    // Events raised while the viewer was constructing happened before we subscribed.
    viewer.setStatus(scene.isDestroyed ? "error" : "ready", null);
    viewer.setTokenState(scene.tokenState);
    return () => offs.forEach((off) => off());
  }, [scene]);

  // Catalog → scene
  useEffect(() => {
    if (!scene || layerCatalog.isLoading) return;
    scene.layers.register(layerCatalog.data);
    const runtime = useLayers.getState().runtime;
    for (const layer of layerCatalog.data) {
      if (
        layer.defaultVisible &&
        !runtime[layer.id]?.visible &&
        runtime[layer.id]?.loadState !== "error"
      ) {
        void scene.layers.setVisible(layer.id, true);
      }
    }
    void scene.layers.ensureFallbackBasemap();
  }, [scene, layerCatalog.data, layerCatalog.isLoading]);

  useEffect(() => {
    if (!scene || siteCatalog.isLoading) return;
    const resolver = async (id: string): Promise<Site | null> => {
      if (id.startsWith("builtin-")) return builtinDemoSite();
      try {
        return await unwrap<Site>(
          api.GET("/api/v1/sites/{site_id}", { params: { path: { site_id: id } } }),
        );
      } catch {
        return null;
      }
    };
    scene.sites.setCatalog(siteCatalog.data, resolver);
  }, [scene, siteCatalog.data, siteCatalog.isLoading]);

  // Active site → mission project: the demo provider for the demo site, otherwise a project
  // built from the catalog site itself (boundary as the one zone, no machines yet) so
  // planning works for every site; a backend/ROS bridge replaces this per site later.
  const activeSiteId = useSites((s) => s.activeSiteId);
  const activeSite = useSite(activeSiteId);
  useEffect(() => {
    if (!scene) return;
    const provider = new DemoMissionProvider();
    const summary = siteCatalog.data?.find((s) => s.id === activeSiteId) ?? null;
    const site = activeSite.data?.id === activeSiteId ? activeSite.data : null;
    const project =
      provider.projectForSite(activeSiteId, summary?.slug ?? null) ??
      (site ? siteProject(site) : null);
    const current = useMission.getState().project;
    // The active site is the *engaged* one and goes null when the camera leaves it; the
    // mission project follows the last site visited so plans stay put while flying around.
    // With no site ever visited, plans are made "anywhere" from drawn areas.
    if (!activeSiteId && !current) {
      useMission.getState().setProject(anywhereProject());
      return;
    }
    if (!activeSiteId || (!project && activeSiteId)) return;
    if (
      (project?.id ?? null) === (current?.id ?? null) &&
      (project?.siteId ?? null) === (current?.siteId ?? null)
    )
      return;
    useMission.getState().setProject(project);
  }, [scene, activeSiteId, siteCatalog.data, activeSite.data]);

  // The world draws whatever the store composes (provider zones plus drawn areas, plans).
  const missionProject = useMission((s) => s.project);
  useEffect(() => {
    if (!scene) return;
    scene.mission.setProject(missionProject);
    const layers = useMission.getState().layers;
    scene.mission.setLayer("zones", layers.zones);
    scene.mission.setLayer("tracks", layers.tracks);
  }, [scene, missionProject]);

  // Settings → scene
  const quality = useSettings((s) => s.quality);
  const manualSse = useSettings((s) => s.manualScreenSpaceError);
  const adaptive = useSettings((s) => s.adaptiveQuality);
  const units = useSettings((s) => s.units);
  const world = useSettings((s) => s.world);
  const exploreSpeed = useSettings((s) => s.exploreSpeed);

  useEffect(() => {
    scene?.performance.configure({ preset: quality, manualScreenSpaceError: manualSse, adaptive });
  }, [scene, quality, manualSse, adaptive]);

  useEffect(() => {
    scene?.measurement.setUnits(units);
  }, [scene, units]);

  useEffect(() => {
    scene?.explore.setSpeed(exploreSpeed);
  }, [scene, exploreSpeed]);

  useEffect(() => {
    if (!scene || layerCatalog.isLoading) return;
    const google = layerCatalog.data.find((l) => l.source.type === "google-photorealistic");
    if (!google) return;
    const wantPhotorealistic = world === "photorealistic";
    scene.useGoogleGeocoder(wantPhotorealistic);
    void scene.layers.setVisible(google.id, wantPhotorealistic);
  }, [scene, world, layerCatalog.data, layerCatalog.isLoading]);

  // Interaction modes → scene
  const measureMode = useUi((s) => s.measureMode);
  const exploreMode = useUi((s) => s.exploreMode);
  useEffect(() => {
    if (!scene) return;
    if (measureMode) {
      scene.setInteractionMode("measure");
      scene.measurement.start(measureMode);
    } else if (exploreMode) {
      scene.setInteractionMode("explore");
      scene.explore.enter(useSettings.getState().exploreSpeed);
    } else {
      scene.setInteractionMode("select");
    }
  }, [scene, measureMode, exploreMode]);

  return null;
}

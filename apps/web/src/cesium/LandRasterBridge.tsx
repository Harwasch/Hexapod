import { useEffect } from "react";
import {
  Credit,
  GeographicTilingScheme,
  ImageryLayer,
  Rectangle,
  Resource,
  UrlTemplateImageryProvider,
} from "cesium";
import { env } from "@/app/env";
import { useLandContext, type LandRasterLayer } from "@/state/landContext";
import { landUsesOidc, useLandIdentity } from "@/state/landIdentity";
import { useSettings } from "@/state/settings";
import { useScene } from "./SceneContext";

/** Authenticated, bounded overlays. The existing picker continues to own map interaction. */
export function LandRasterBridge() {
  const host = useScene();
  useEffect(() => {
    if (!host) return;
    let disposed = false;
    let fallbackRequested = false;
    const rendered = new Map<
      string,
      { definition: LandRasterLayer; layer: ImageryLayer; auth: string }
    >();
    const sync = () => {
      if (disposed || host.isDestroyed) return;
      const { rasters } = useLandContext.getState();
      const identity = useLandIdentity.getState();
      const token = landUsesOidc ? identity.accessToken : useSettings.getState().writeToken.trim();
      const headers: Record<string, string> = {};
      if (token) headers.Authorization = `Bearer ${token}`;
      if (landUsesOidc && identity.workspaceId) headers["X-Workspace-ID"] = identity.workspaceId;
      const auth = JSON.stringify(headers);
      for (const [id, current] of rendered) {
        const next = rasters[id];
        if (current.definition.band === next?.band && current.auth === auth) {
          current.layer.alpha = next.opacity;
          current.definition = next;
          continue;
        }
        host.viewer.imageryLayers.remove(current.layer);
        rendered.delete(id);
      }
      // Cesium stretches its bottom imagery layer globally. Wait for the local fallback
      // basemap before adding a clipped scientific overlay in that position.
      if (!host.viewer.imageryLayers.length) {
        if (!fallbackRequested && Object.keys(rasters).length) {
          fallbackRequested = true;
          void host.layers.ensureFallbackBasemap().then(() => {
            if (disposed || host.isDestroyed) return;
            if (host.viewer.imageryLayers.length) sync();
            else
              for (const id of Object.keys(useLandContext.getState().rasters))
                useLandContext
                  .getState()
                  .setRasterError(
                    id,
                    "The background map could not load. Reload the map to retry.",
                  );
          });
        }
        return;
      }
      for (const definition of Object.values(rasters)) {
        if (rendered.has(definition.id)) continue;
        const [west = 0, south = 0, east = 0, north = 0] = definition.bounds;
        const provider = new UrlTemplateImageryProvider({
          url: new Resource({
            url: `${env.apiBaseUrl}/api/v1/land/rasters/${definition.id}/tiles/${definition.band}/{z}/{x}/{y}.png`,
            headers,
          }),
          tilingScheme: new GeographicTilingScheme(),
          rectangle: Rectangle.fromDegrees(west, south, east, north),
          maximumLevel: 18,
          enablePickFeatures: false,
          credit: new Credit(definition.attribution),
        });
        provider.errorEvent.addEventListener(() => {
          if (!disposed && useLandContext.getState().rasters[definition.id])
            useLandContext
              .getState()
              .setRasterError(
                definition.id,
                "Some map tiles could not load. Hide and show the layer to retry; check your workspace connection.",
              );
        });
        const layer = new ImageryLayer(provider, { alpha: definition.opacity });
        host.viewer.imageryLayers.add(layer);
        rendered.set(definition.id, { definition, layer, auth });
      }
      host.scene.requestRender();
    };
    const offContext = useLandContext.subscribe(sync);
    const offIdentity = useLandIdentity.subscribe(sync);
    const offSettings = useSettings.subscribe(sync);
    const offLayer = host.viewer.imageryLayers.layerAdded.addEventListener(() =>
      queueMicrotask(sync),
    );
    sync();
    return () => {
      disposed = true;
      offContext();
      offIdentity();
      offSettings();
      offLayer();
      if (!host.isDestroyed) {
        for (const { layer } of rendered.values()) host.viewer.imageryLayers.remove(layer);
        host.scene.requestRender();
      }
    };
  }, [host]);
  return null;
}

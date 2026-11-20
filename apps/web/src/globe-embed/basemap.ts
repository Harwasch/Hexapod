import {
  CesiumTerrainProvider,
  ImageryLayer,
  Ion,
  IonImageryProvider,
  OpenStreetMapImageryProvider,
  type CesiumWidget,
  type ImageryProvider,
} from "cesium";
import { createNaturalEarthProvider } from "@/cesium/providers/imagery";

/** Keep bundled imagery below online layers, including on authentication/network failure. */
export function installBasemap(
  viewer: CesiumWidget,
  token: string | undefined,
  status: (message: string) => void,
): () => void {
  let disposed = false;
  const removers: (() => void)[] = [];
  async function add(provider: Promise<ImageryProvider>, online: boolean): Promise<void> {
    try {
      const ready = await provider;
      if (disposed) return;
      const layer = new ImageryLayer(ready);
      viewer.imageryLayers.add(layer, online ? undefined : 0);
      removers.push(
        ready.errorEvent.addEventListener(() => {
          if (online) layer.show = false;
          status(online ? "Map offline" : "No imagery");
          viewer.scene.requestRender();
        }),
      );
      viewer.scene.requestRender();
    } catch {
      if (!disposed) status(online ? "Map offline" : "No imagery");
    }
  }
  void add(createNaturalEarthProvider(), false);
  const accessToken = token?.trim();
  if (accessToken) {
    Ion.defaultAccessToken = accessToken;
    // Same public world imagery (2) and terrain (1) as the web catalog defaults.
    void add(IonImageryProvider.fromAssetId(2, { accessToken }), true);
    void CesiumTerrainProvider.fromIonAssetId(1, { requestVertexNormals: true }).then(
      (terrain) => {
        if (disposed) return;
        viewer.terrainProvider = terrain;
        removers.push(terrain.errorEvent.addEventListener(() => status("Terrain unavailable")));
        viewer.scene.requestRender();
      },
      () => {
        if (!disposed) status("Terrain unavailable");
      },
    );
  } else {
    status("Terrain unavailable");
    void add(
      Promise.resolve(
        new OpenStreetMapImageryProvider({
          url: "https://tile.openstreetmap.org/",
          maximumLevel: 19,
        }),
      ),
      true,
    );
  }
  return () => {
    disposed = true;
    removers.forEach((remove) => remove());
  };
}

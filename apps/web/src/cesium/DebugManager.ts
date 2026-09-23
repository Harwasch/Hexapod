import type { ImageryLayer, Scene, Viewer } from "cesium";

import type { SiteManager } from "./SiteManager";

export interface DebugFlags {
  boundingVolumes: boolean;
  wireframe: boolean;
  tileCoordinates: boolean;
  fpsOverlay: boolean;
  clipping: boolean;
}

/** Developer switches: tile debugging on tilesets and the globe, Cesium's own FPS overlay. */
export class DebugManager {
  private readonly scene: Scene;
  private readonly flags: DebugFlags = {
    boundingVolumes: false,
    wireframe: false,
    tileCoordinates: false,
    fpsOverlay: false,
    clipping: true,
  };

  constructor(
    private readonly viewer: Viewer,
    private readonly sites: SiteManager,
    private readonly onClippingToggle: (enabled: boolean) => void,
  ) {
    this.scene = viewer.scene;
  }

  get current(): DebugFlags {
    return { ...this.flags };
  }

  set(patch: Partial<DebugFlags>): DebugFlags {
    Object.assign(this.flags, patch);
    this.sites.setDebug({
      boundingVolumes: this.flags.boundingVolumes,
      wireframe: this.flags.wireframe,
    });
    this.scene.globe.showGroundAtmosphere = !this.flags.wireframe;
    this.scene.debugShowFramesPerSecond = this.flags.fpsOverlay;
    if (patch.clipping !== undefined) this.onClippingToggle(patch.clipping);
    if (patch.tileCoordinates !== undefined) void this.toggleTileCoordinates(patch.tileCoordinates);
    this.scene.requestRender();
    return this.current;
  }

  private tileCoordinateLayer: ImageryLayer | null = null;

  private async toggleTileCoordinates(on: boolean): Promise<void> {
    if (on && !this.tileCoordinateLayer) {
      const { TileCoordinatesImageryProvider } = await import("cesium");
      this.tileCoordinateLayer = this.viewer.imageryLayers.addImageryProvider(
        new TileCoordinatesImageryProvider(),
      );
    } else if (!on && this.tileCoordinateLayer) {
      this.viewer.imageryLayers.remove(this.tileCoordinateLayer, true);
      this.tileCoordinateLayer = null;
    }
  }

  destroy(): void {
    void this.toggleTileCoordinates(false);
  }
}

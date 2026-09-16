import {
  Cartesian2,
  Cartesian3,
  Cartographic,
  Cesium3DTileFeature,
  Cesium3DTileset,
  Color,
  type DataSource,
  Entity,
  HeightReference,
  Math as CesiumMath,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  sampleTerrainMostDetailed,
  type Scene,
  type Viewer,
} from "cesium";

import type { Site } from "@twin/contracts";
import { flattenRing, outerRings } from "@twin/geo";

import type { Emitter } from "@/lib/emitter";
import { createLogger } from "@/lib/log";
import type { Selection, SelectionProperty } from "@/state/selection";

import type { CameraController } from "./CameraController";
import type { LayerManager } from "./LayerManager";
import { ZONE_ENTITY_PREFIX } from "./MissionManager";
import type { SiteManager } from "./SiteManager";
import type { SceneEvents } from "./types";

const log = createLogger("selection");
const ACCENT = Color.fromCssColorString("#0a84ff");

interface PickContext {
  position: Cartesian3;
  carto: Cartographic;
}

/** Click-to-inspect: resolves what is under the cursor into a Selection and highlights it. */
/** How long the pointer must rest before a hover pick runs. */
const HOVER_REST_MS = 120;

export class SelectionManager {
  private readonly scene: Scene;
  private readonly handler: ScreenSpaceEventHandler;
  private highlighted: { feature: Cesium3DTileFeature; color: Color } | null = null;
  private highlightedEntity: { entity: Entity; restore: () => void } | null = null;
  private marker: Entity | null = null;
  private siteOutline: Entity | null = null;
  private enabled = true;
  private hoverTimer: ReturnType<typeof setTimeout> | null = null;
  /** A pointer button is held: the user is dragging, and every pick would stall the drag. */
  private pointerHeld = false;
  private readonly onPointerDown = (): void => {
    this.pointerHeld = true;
    if (this.hoverTimer !== null) clearTimeout(this.hoverTimer);
    this.hoverTimer = null;
  };
  private readonly onPointerUp = (): void => {
    this.pointerHeld = false;
  };
  private hoverPosition = new Cartesian2();

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
    private readonly camera: CameraController,
    private readonly layers: LayerManager,
    private readonly sites: SiteManager,
  ) {
    this.scene = viewer.scene;
    // Cesium's default double-click "track entity" behaviour would fight our navigation.
    viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(
      ScreenSpaceEventType.LEFT_DOUBLE_CLICK,
    );
    viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(ScreenSpaceEventType.LEFT_CLICK);
    this.handler = new ScreenSpaceEventHandler(viewer.canvas);
    this.handler.setInputAction((event: ScreenSpaceEventHandler.PositionedEvent) => {
      if (this.enabled) this.select(event.position);
    }, ScreenSpaceEventType.LEFT_CLICK);
    this.handler.setInputAction((event: ScreenSpaceEventHandler.PositionedEvent) => {
      if (this.enabled) this.selectAndFly(event.position);
    }, ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
    this.handler.setInputAction((event: ScreenSpaceEventHandler.MotionEvent) => {
      if (this.enabled) this.hover(event.endPosition);
    }, ScreenSpaceEventType.MOUSE_MOVE);
    window.addEventListener("pointerdown", this.onPointerDown, { capture: true });
    window.addEventListener("pointerup", this.onPointerUp, { capture: true });
    window.addEventListener("pointercancel", this.onPointerUp, { capture: true });
  }

  /** Disabled while measuring or exploring so those tools own the pointer. */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) this.viewer.canvas.style.cursor = "";
  }

  clear(): void {
    this.unhighlight();
    this.events.emit("selection", null);
  }

  private pickPosition(window: Cartesian2, picked: unknown): PickContext | null {
    let position: Cartesian3 | undefined;
    const ray = this.viewer.camera.getPickRay(window);
    // The depth-buffer pick is only precise at close range; far from the surface the
    // analytic ray/globe intersection is exact, so prefer it unless 3D content was hit.
    const far = this.camera.pose().altitude > 20_000;
    if (picked === undefined && far && ray) position = this.scene.globe.pick(ray, this.scene);
    if (!position && this.scene.pickPositionSupported) position = this.scene.pickPosition(window);
    if (!position && ray) position = this.scene.globe.pick(ray, this.scene);
    if (!position) return null;
    return { position, carto: Cartographic.fromCartesian(position) };
  }

  private select(window: Cartesian2): void {
    const picked: unknown = this.scene.pick(window);
    const context = this.pickPosition(window, picked);
    if (!context) {
      this.clear();
      return;
    }
    this.unhighlight();
    const longitude = CesiumMath.toDegrees(context.carto.longitude);
    const latitude = CesiumMath.toDegrees(context.carto.latitude);
    const base: Omit<Selection, "kind" | "title"> = {
      longitude,
      latitude,
      height: context.carto.height,
      terrainHeight: null,
      at: Date.now(),
    };
    if (
      isEntityPick(picked) &&
      typeof picked.id.id === "string" &&
      picked.id.id.startsWith(ZONE_ENTITY_PREFIX)
    ) {
      this.events.emit("mission-select", {
        kind: "zone",
        id: picked.id.id.slice(ZONE_ENTITY_PREFIX.length),
      });
      return;
    }
    let selection: Selection;
    if (picked instanceof Cesium3DTileFeature) {
      selection = this.selectTileFeature(picked, base);
    } else if (isEntityPick(picked)) {
      selection = this.selectEntity(picked.id, base);
    } else if (isPrimitivePick(picked) && picked.primitive instanceof Cesium3DTileset) {
      selection = this.selectTileset(picked.primitive, base);
    } else {
      selection = this.selectGround(base);
    }
    this.placeMarker(context.position, selection.kind === "ground");
    this.events.emit("selection", selection);
    void this.enrichWithTerrain(selection);
  }

  private selectAndFly(window: Cartesian2): void {
    const context = this.pickPosition(window, this.scene.pick(window));
    if (!context) return;
    this.select(window);
    const distance = Cartesian3.distance(this.viewer.camera.positionWC, context.position);
    const pose = this.camera.pose();
    const lon = CesiumMath.toDegrees(context.carto.longitude);
    const lat = CesiumMath.toDegrees(context.carto.latitude);
    // Halve the distance towards the clicked point while keeping the current tilt.
    const nextHeight =
      context.carto.height +
      Math.max(distance * 0.45, 2) * Math.sin(CesiumMath.toRadians(-pose.pitch));
    this.camera.flyTo(lon, lat, nextHeight, {
      pitch: pose.pitch,
      heading: pose.heading,
      durationS: 1.2,
    });
  }

  private selectTileFeature(
    feature: Cesium3DTileFeature,
    base: Omit<Selection, "kind" | "title">,
  ): Selection {
    const properties: SelectionProperty[] = feature
      .getPropertyIds()
      .slice(0, 40)
      .map((key) => ({ key, value: formatValue(feature.getProperty(key)) }));
    const layerId = this.layers.layerIdForPrimitive(feature.tileset);
    const layer = layerId ? this.layers.get(layerId) : undefined;
    const site = this.sites.activeSite;
    try {
      this.highlighted = { feature, color: Color.clone(feature.color) };
      feature.color = Color.lerp(feature.color, ACCENT, 0.6, new Color());
    } catch (error) {
      log.warn("could not highlight feature", { error: String(error) });
    }
    const name = properties.find((p) => /^(name|title|id)$/i.test(p.key))?.value;
    return {
      ...base,
      kind: "tile-feature",
      title: name ?? layer?.name ?? site?.name ?? "3D feature",
      layerId: layerId ?? undefined,
      siteId: layer ? undefined : site?.id,
      sourceLabel:
        layer?.name ??
        (site ? `${site.name} (${this.sites.activeRepresentation ?? "site"})` : "3D Tiles"),
      observedAt: layer?.observedAt ?? null,
      attribution: layer?.attribution ?? site?.attribution,
      properties,
    };
  }

  private selectEntity(entity: Entity, base: Omit<Selection, "kind" | "title">): Selection {
    const dataSource = this.findDataSourceOf(entity);
    const layerId = dataSource ? this.layers.layerIdForDataSource(dataSource) : null;
    const layer = layerId ? this.layers.get(layerId) : undefined;
    const properties: SelectionProperty[] = [];
    const bag = entity.properties;
    if (bag) {
      const values = bag.getValue(this.viewer.clock.currentTime) as
        Record<string, unknown> | undefined;
      const names = (bag.propertyNames as unknown[]).filter(
        (n): n is string => typeof n === "string",
      );
      for (const key of names.slice(0, 40)) {
        properties.push({ key, value: formatValue(values?.[key]) });
      }
    }
    this.highlightEntity(entity);
    return {
      ...base,
      kind: "layer-feature",
      title:
        entity.name ??
        properties.find((p) => /^name$/i.test(p.key))?.value ??
        layer?.name ??
        "Feature",
      layerId: layerId ?? undefined,
      sourceLabel: layer?.name ?? "Vector layer",
      observedAt: layer?.observedAt ?? null,
      attribution: layer?.attribution,
      properties,
    };
  }

  private selectTileset(
    tileset: Cesium3DTileset,
    base: Omit<Selection, "kind" | "title">,
  ): Selection {
    const site = this.sites.activeSite;
    const layerId = this.layers.layerIdForPrimitive(tileset);
    const layer = layerId ? this.layers.get(layerId) : undefined;
    if (site && !layer) {
      this.outlineSite(site);
      const rep = this.sites.activeRepresentation;
      const asset = site.assets.find((a) => a.representation === rep);
      return {
        ...base,
        kind: "site",
        title: site.name,
        siteId: site.id,
        sourceLabel: asset ? `${asset.name} · ${describeSource(asset.source)}` : "Reality model",
        observedAt: asset?.observedAt ?? null,
        attribution: asset?.attribution.length ? asset.attribution : site.attribution,
        properties: asset?.resolution
          ? [
              ...(asset.resolution.groundSampleDistanceM
                ? [
                    {
                      key: "Ground sample distance",
                      value: `${asset.resolution.groundSampleDistanceM} m`,
                    },
                  ]
                : []),
              ...(asset.resolution.description
                ? [{ key: "Resolution", value: asset.resolution.description }]
                : []),
            ]
          : undefined,
      };
    }
    return {
      ...base,
      kind: "layer-feature",
      title: layer?.name ?? "3D Tiles",
      layerId: layerId ?? undefined,
      sourceLabel: layer?.name ?? "3D Tiles",
      observedAt: layer?.observedAt ?? null,
      attribution: layer?.attribution,
    };
  }

  private selectGround(base: Omit<Selection, "kind" | "title">): Selection {
    const site = this.sites.activeSite;
    return {
      ...base,
      kind: "ground",
      title: "Location",
      siteId: undefined,
      sourceLabel: this.scene.globe.show ? "Terrain" : "Global 3D tiles",
      attribution: site ? undefined : undefined,
    };
  }

  private async enrichWithTerrain(selection: Selection): Promise<void> {
    const provider = this.scene.terrainProvider;
    try {
      const [sample] = await sampleTerrainMostDetailed(provider, [
        Cartographic.fromDegrees(selection.longitude, selection.latitude),
      ]);
      if (sample && Number.isFinite(sample.height)) {
        this.events.emit("selection", { ...selection, terrainHeight: sample.height });
      }
    } catch (error) {
      log.debug("terrain sample unavailable", { error: String(error) });
    }
  }

  /**
   * Hover only picks once the pointer has rested: every pick is a render pass, and on a
   * Gaussian splat that pass is as expensive as a frame, so picking on every mouse move
   * starves navigation. Nothing is picked while the camera is moving.
   */
  private hover(window: Cartesian2): void {
    if (this.pointerHeld) return;
    if (this.hoverTimer !== null) clearTimeout(this.hoverTimer);
    this.hoverPosition = Cartesian2.clone(window, this.hoverPosition);
    this.hoverTimer = setTimeout(() => {
      this.hoverTimer = null;
      // A pick is a render pass plus a GPU read-back; never during a gesture, whether the
      // camera is currently moving or merely paused between two mouse events of a drag.
      if (!this.enabled || this.camera.isMoving || this.pointerHeld) return;
      const picked: unknown = this.scene.pick(this.hoverPosition);
      const interactive = picked instanceof Cesium3DTileFeature || isEntityPick(picked);
      this.viewer.canvas.style.cursor = interactive ? "pointer" : "";
    }, HOVER_REST_MS);
  }

  private placeMarker(position: Cartesian3, ground: boolean): void {
    this.removeMarker();
    this.marker = this.viewer.entities.add({
      id: "selection-marker",
      position,
      point: {
        pixelSize: ground ? 9 : 7,
        color: ACCENT.withAlpha(0.95),
        outlineColor: Color.WHITE.withAlpha(0.9),
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        heightReference: HeightReference.NONE,
      },
    });
  }

  private outlineSite(site: Site): void {
    this.removeOutline();
    const ring = outerRings(site.boundary)[0];
    if (!ring) return;
    this.siteOutline = this.viewer.entities.add({
      id: "selection-site-outline",
      polyline: {
        positions: Cartesian3.fromDegreesArray(flattenRing(ring)),
        width: 2,
        material: ACCENT.withAlpha(0.8),
        clampToGround: true,
      },
    });
  }

  private highlightEntity(entity: Entity): void {
    const restores: (() => void)[] = [];
    if (entity.polygon) {
      const material = entity.polygon.material;
      entity.polygon.material = ACCENT.withAlpha(0.35) as never;
      restores.push(() => {
        if (entity.polygon) entity.polygon.material = material;
      });
    }
    if (entity.polyline) {
      const material = entity.polyline.material;
      entity.polyline.material = ACCENT as never;
      restores.push(() => {
        if (entity.polyline) entity.polyline.material = material;
      });
    }
    if (entity.point) {
      const color = entity.point.color;
      entity.point.color = ACCENT as never;
      restores.push(() => {
        if (entity.point) entity.point.color = color;
      });
    }
    this.highlightedEntity = { entity, restore: () => restores.forEach((r) => r()) };
  }

  private findDataSourceOf(entity: Entity): DataSource | null {
    const sources = this.viewer.dataSources;
    for (let i = 0; i < sources.length; i++) {
      const source = sources.get(i);
      if (source.entities.contains(entity)) return source;
    }
    return null;
  }

  private unhighlight(): void {
    if (this.highlighted) {
      try {
        this.highlighted.feature.color = this.highlighted.color;
      } catch (error) {
        log.debug("feature already evicted", { error: String(error) });
      }
      this.highlighted = null;
    }
    if (this.highlightedEntity) {
      this.highlightedEntity.restore();
      this.highlightedEntity = null;
    }
    this.removeMarker();
    this.removeOutline();
    this.scene.requestRender();
  }

  private removeMarker(): void {
    if (this.marker) {
      this.viewer.entities.remove(this.marker);
      this.marker = null;
    }
  }

  private removeOutline(): void {
    if (this.siteOutline) {
      this.viewer.entities.remove(this.siteOutline);
      this.siteOutline = null;
    }
  }

  destroy(): void {
    window.removeEventListener("pointerdown", this.onPointerDown, { capture: true });
    window.removeEventListener("pointerup", this.onPointerUp, { capture: true });
    window.removeEventListener("pointercancel", this.onPointerUp, { capture: true });
    if (this.hoverTimer !== null) clearTimeout(this.hoverTimer);
    this.unhighlight();
    this.handler.destroy();
  }
}

function isEntityPick(value: unknown): value is { id: Entity } {
  return typeof value === "object" && value !== null && "id" in value && value.id instanceof Entity;
}

function isPrimitivePick(value: unknown): value is { primitive: unknown } {
  return typeof value === "object" && value !== null && "primitive" in value;
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "string") return value;
  if (typeof value === "boolean" || typeof value === "bigint") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return "[object]";
  }
}

function describeSource(source: Site["assets"][number]["source"]): string {
  return source.type === "cesium-ion" ? `Cesium ion asset ${source.assetId}` : "3D Tiles URL";
}

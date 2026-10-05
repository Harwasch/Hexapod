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
  KeyboardEventModifier,
  Math as CesiumMath,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  sampleTerrainMostDetailed,
  type Scene,
  type CesiumWidget,
} from "cesium";

import type { Site } from "@twin/contracts";
import { flattenRing, outerRings } from "@twin/geo";

import type { Emitter } from "@/lib/emitter";
import { createLogger } from "@/lib/log";
import type { Selection, SelectionProperty } from "@/state/selection";

import { AREA_CANDIDATE_PREFIX, AREA_HANDLE_PREFIX } from "./AreaEditor";
import type { CameraController } from "./CameraController";
import type { LayerManager } from "./LayerManager";
import { LongPress } from "./longPress";
import { ZONE_ENTITY_PREFIX } from "./MissionManager";
import type { SiteManager } from "./SiteManager";
import type { SceneEvents } from "./types";
import type { SplatCollider } from "./SplatCollider";

const log = createLogger("selection");
const ACCENT = Color.fromCssColorString("#0a84ff");

interface PickContext {
  position: Cartesian3;
  carto: Cartographic;
  /** The splat tileset the position is on, when a scan's solids answered (SplatCollider). */
  splat?: object;
}

/**
 * What a pick is for. A plain click selects things on the map (a zone, a mapped feature, a 3D
 * tile's feature) and leaves the ground and a site's surface alone; the map menu's "What's
 * here" inspects whatever is at the point, the ground included.
 */
type SelectMode = "click" | "inspect";

/** How long the pointer must rest before a hover pick runs. */
const HOVER_REST_MS = 120;

/**
 * Click-to-select, the map menu and double-click: resolves what is under the cursor into a
 * Selection and highlights it, raises the map menu (right-click, Ctrl+click, a long press) at a
 * point, and flies towards a point.
 */
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
  private collider: SplatCollider | null = null;
  private pickTicket = 0;
  private hoverEnabled = true;
  /** Asked first on every click: true when it took the click (scene selection hit a scan). */
  private claim: ((position: Cartesian2) => boolean) | null = null;
  /** Asked before the map menu opens: false while a tool owns the pointer. */
  private menuGate: () => boolean = () => true;
  private readonly longPress: LongPress;
  /** The selection last reported, until cleared: what a terrain sample may still refine. */
  private current: Selection | null = null;

  constructor(
    private readonly viewer: CesiumWidget,
    private readonly events: Emitter<SceneEvents>,
    private readonly camera: CameraController,
    private readonly layers: LayerManager,
    private readonly sites: SiteManager,
  ) {
    this.scene = viewer.scene;
    // The clicks are ours alone. `Viewer` installed select-on-click and track-on-double-click,
    // which fought this navigation and were removed here; the `CesiumWidget` the scene now runs
    // on installs no input actions at all, so there is nothing left to take away.
    this.handler = new ScreenSpaceEventHandler(viewer.canvas);
    this.handler.setInputAction((event: ScreenSpaceEventHandler.PositionedEvent) => {
      // The finger lifting after a long press opened the menu is not a click on the map.
      if (this.longPress.takeClick()) return;
      if (!this.enabled) return;
      if (this.claimed(event.position)) return;
      void this.select(event.position, "click");
    }, ScreenSpaceEventType.LEFT_CLICK);
    this.handler.setInputAction((event: ScreenSpaceEventHandler.PositionedEvent) => {
      if (this.enabled) void this.flyTowards(event.position);
    }, ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
    // The map menu: a right-click, or Ctrl+click for a one-button Mac. A drag with either
    // orbits instead (CameraController); Cesium raises a click only for a press that moved
    // less than its click tolerance (5 px), so the two never meet. On touch it is the long
    // press below: Cesium's own touch hold also raises a RIGHT_CLICK, after 1.5 s, while the
    // finger is still down, and that one is left alone.
    const menu = (event: ScreenSpaceEventHandler.PositionedEvent): void => {
      if (!this.longPress.touching) this.openMenu(event.position.x, event.position.y);
    };
    this.handler.setInputAction(menu, ScreenSpaceEventType.RIGHT_CLICK);
    this.handler.setInputAction(menu, ScreenSpaceEventType.LEFT_CLICK, KeyboardEventModifier.CTRL);
    this.longPress = new LongPress(viewer.canvas, (x, y) => this.openMenu(x, y));
    this.handler.setInputAction((event: ScreenSpaceEventHandler.MotionEvent) => {
      if (this.enabled) this.hover(event.endPosition);
    }, ScreenSpaceEventType.MOUSE_MOVE);
    window.addEventListener("pointerdown", this.onPointerDown, { capture: true });
    window.addEventListener("pointerup", this.onPointerUp, { capture: true });
    window.addEventListener("pointercancel", this.onPointerUp, { capture: true });
  }

  /** Disabled while measuring or exploring so those tools own the pointer. */
  /**
   * Hover picking on or off. Off over a splat scan: nothing in it is hoverable, and a pick is
   * a render pass plus a read-back that waits for the GPU to finish the splat frame (half a
   * second of a busy one), so resting the pointer froze the page.
   */
  setHoverEnabled(enabled: boolean): void {
    this.hoverEnabled = enabled;
    if (!enabled && this.hoverTimer !== null) {
      clearTimeout(this.hoverTimer);
      this.hoverTimer = null;
    }
  }

  /**
   * Who is asked first on a click (the scan's objects, SceneSelectController): when it takes
   * the click, no card opens for it and an open one closes -- an object of the scan under the
   * cursor is what was clicked, not the site's boundary or the ground beneath it.
   */
  setClickClaim(claim: ((position: Cartesian2) => boolean) | null): void {
    this.claim = claim;
  }

  /**
   * Whether the map menu may open now, beyond selection being on: the scene says no while an
   * area's corners are being edited (a right-click there removes one) or the brush is out.
   */
  setMenuGate(gate: () => boolean): void {
    this.menuGate = gate;
  }

  private claimed(position: Cartesian2): boolean {
    let taken = false;
    try {
      taken = this.claim?.(position) ?? false;
    } catch (error) {
      log.warn("click claim failed", { error: String(error) });
    }
    if (taken) {
      // A newer click supersedes a pick still being answered, and an open card closes.
      this.pickTicket += 1;
      this.clear();
    }
    return taken;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) this.viewer.canvas.style.cursor = "";
  }

  clear(): void {
    this.unhighlight();
    this.current = null;
    this.events.emit("selection", null);
  }

  /** Splat solids, for positions on a scan found on the CPU (SplatCollider). */
  setCollider(collider: SplatCollider | null): void {
    this.collider = collider;
  }

  /**
   * Where a click landed. On a splat scan its solids answer on the CPU (splats write no depth
   * anyway); the depth buffer is read only for other 3D content that was hit, since that
   * read-back waits for the GPU to finish its frame.
   */
  private pickPosition(window: Cartesian2, picked: unknown): PickContext | null {
    let position: Cartesian3 | undefined;
    const ray = this.viewer.camera.getPickRay(window);
    const splat = ray ? this.collider?.raycast(ray) : undefined;
    if (splat) position = splat.point;
    // The depth-buffer pick is only precise at close range; far from the surface the
    // analytic ray/globe intersection is exact, so prefer it unless 3D content was hit.
    const far = this.camera.pose().altitude > 20_000;
    if (!position && picked === undefined && ray) position = this.scene.globe.pick(ray, this.scene);
    if (!position && far && ray) position = this.scene.globe.pick(ray, this.scene);
    if (!position && this.scene.pickPositionSupported) position = this.scene.pickPosition(window);
    if (!position && ray) position = this.scene.globe.pick(ray, this.scene);
    // No terrain drawn there yet: the bare ellipsoid still says where the point is.
    position ??= this.viewer.camera.pickEllipsoid(window, this.scene.globe.ellipsoid);
    if (!position) return null;
    return { position, carto: Cartographic.fromCartesian(position), splat: splat?.tileset };
  }

  /**
   * The object under a click, without holding the page: `pickAsync` reads the pick back
   * through a pixel buffer and a fence, so the main thread is free while the GPU finishes
   * (measured 0.6 ms held, against 590 ms for `scene.pick` behind a splat frame). A click
   * made while an earlier one is still being answered supersedes it.
   */
  private async pickObject(window: Cartesian2): Promise<{ picked: unknown; current: boolean }> {
    const ticket = ++this.pickTicket;
    const position = Cartesian2.clone(window);
    let picked: unknown;
    try {
      picked = await this.scene.pickAsync(position);
    } catch {
      picked = undefined;
    }
    return { picked, current: ticket === this.pickTicket && !this.scene.isDestroyed() };
  }

  /**
   * "What's here" (the map menu): whatever is at the point opens in the inspector, the ground
   * included. On a splat scan of the active site it is the site's card, so what the site is and
   * how it was placed can be read from its surface (splats are invisible to the GPU pick, which
   * sees the ground beneath them).
   */
  inspectAt(window: { x: number; y: number }): Promise<void> {
    return this.select(new Cartesian2(window.x, window.y), "inspect");
  }

  private async select(window: Cartesian2, mode: SelectMode): Promise<void> {
    const { picked, current } = await this.pickObject(window);
    if (!current) return;
    // Area handles and candidate outlines belong to the area editor, not to selection.
    if (
      isEntityPick(picked) &&
      typeof picked.id.id === "string" &&
      (picked.id.id.startsWith(AREA_HANDLE_PREFIX) ||
        picked.id.id.startsWith(AREA_CANDIDATE_PREFIX))
    )
      return;
    // A plain click is for things on the map. On the ground, or on a site's surface (its mesh,
    // or the ground under a scan's splats), it does nothing at all: no marker, no card, and
    // what is open stays open. "What's here" in the map menu is how to ask about a place.
    if (mode === "click" && !(picked instanceof Cesium3DTileFeature) && !isEntityPick(picked))
      return;
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
    // On the active site's scan the point is the scan's, whatever the GPU pick found under it
    // (a zone drawn on the ground, the terrain): splats are invisible to that pick.
    const scan = mode === "inspect" ? this.activeScan() : null;
    const onScan = scan !== null && context.splat === scan;
    if (
      !onScan &&
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
    if (scan && onScan) {
      selection = this.selectTileset(scan, base);
    } else if (picked instanceof Cesium3DTileFeature) {
      selection = this.selectTileFeature(picked, base);
    } else if (isEntityPick(picked)) {
      selection = this.selectEntity(picked.id, base);
    } else if (isPrimitivePick(picked) && picked.primitive instanceof Cesium3DTileset) {
      selection = this.selectTileset(picked.primitive, base);
    } else {
      selection = this.selectGround(base);
    }
    this.placeMarker(context.position, selection.kind === "ground");
    this.current = selection;
    this.events.emit("selection", selection);
    void this.enrichWithTerrain(selection);
  }

  /** The active site's splat scan, while the site shows it. */
  private activeScan(): Cesium3DTileset | null {
    const site = this.sites.activeSite;
    if (!site || this.sites.activeRepresentation !== "gaussian-splat") return null;
    return this.sites.tilesetFor(site.id, "gaussian-splat");
  }

  /**
   * The map menu at (`x`, `y`), CSS px from the canvas's top left, with the ground there: true
   * when it is on its way (a long press then swallows the click that follows). Not while a
   * tool owns the pointer, nor where there is no ground (the sky).
   */
  private openMenu(x: number, y: number): boolean {
    if (!this.enabled || !this.menuGate()) return false;
    void this.raiseMenu(new Cartesian2(x, y));
    return true;
  }

  private async raiseMenu(window: Cartesian2): Promise<void> {
    const { picked, current } = await this.pickObject(window);
    if (!current) return;
    const context = this.pickPosition(window, picked);
    // Asked again: measuring or an area's corners may have taken the pointer meanwhile.
    if (!context || !this.enabled || !this.menuGate()) return;
    this.events.emit("map-menu", {
      x: window.x,
      y: window.y,
      longitude: CesiumMath.toDegrees(context.carto.longitude),
      latitude: CesiumMath.toDegrees(context.carto.latitude),
      height: context.carto.height,
    });
  }

  /**
   * Double-click, and the map menu's "Fly here": halfway towards the point, keeping the tilt.
   * It selects nothing, and leaves a scan's objects alone (a click's business).
   */
  async flyTowards(window: { x: number; y: number }): Promise<void> {
    const position = new Cartesian2(window.x, window.y);
    const { picked, current } = await this.pickObject(position);
    if (!current) return;
    const context = this.pickPosition(position, picked);
    if (!context) return;
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
      // Only while it is still what is selected: a sample arriving after the card was closed,
      // or after something else was picked, must not bring the old place back.
      if (sample && Number.isFinite(sample.height) && this.current === selection) {
        this.current = { ...selection, terrainHeight: sample.height };
        this.events.emit("selection", this.current);
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
    if (this.pointerHeld || !this.hoverEnabled) return;
    if (this.hoverTimer !== null) clearTimeout(this.hoverTimer);
    this.hoverPosition = Cartesian2.clone(window, this.hoverPosition);
    this.hoverTimer = setTimeout(() => {
      this.hoverTimer = null;
      // A pick is a render pass plus a GPU read-back; never during a gesture, whether the
      // camera is currently moving or merely paused between two mouse events of a drag.
      if (!this.enabled || !this.hoverEnabled || this.camera.isMoving || this.pointerHeld) return;
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
    this.longPress.destroy();
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

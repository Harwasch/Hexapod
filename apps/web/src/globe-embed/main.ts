import {
  BoundingSphere,
  HeadingPitchRange,
  Cartesian3,
  Color,
  Entity,
  PolygonHierarchy,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  type Cartesian2,
} from "cesium";
import { createGlobeWidget } from "@/cesium/createGlobeWidget";
import { installBasemap } from "./basemap";
import { createEmbedScanTileset } from "@/cesium/providers/tiles";
import { addAnswerPins } from "@/features/answers/answerPinLayer";
import { dispatchMessage } from "./bridge";
import type { BridgeMessage, BridgeSnapshot } from "./generated";
import { ScanLayer } from "./ScanLayer";
import { installErrorForwarding } from "./errors";
import fixtures from "../../../ios/Modules/Globe/Tests/GlobeTests/Fixtures/messages.json";
import "./style.css";

declare global {
  interface Window {
    globeBridge?: { receive(value: unknown): void };
    webkit?: { messageHandlers?: { globe?: { postMessage(value: BridgeMessage): void } } };
  }
}
function element(id: string): HTMLElement {
  const found = document.querySelector<HTMLElement>(id);
  if (!found) throw Error(`Missing globe element ${id}`);
  return found;
}
const status = element("#status");
const mapStatus = document.createElement("p");
mapStatus.id = "map-status";
mapStatus.setAttribute("role", "status");
document.body.append(mapStatus);
const fixtureMode = new URLSearchParams(location.search).has("fixture");
function emit(message: BridgeMessage): void {
  window.webkit?.messageHandlers?.globe?.postMessage(message);
}
const removeErrors = installErrorForwarding(window, emit);
window.addEventListener("pagehide", removeErrors, { once: true });
try {
  const viewer = createGlobeWidget(element("#globe"), element("#credits"));
  const removeRenderError = viewer.scene.renderError.addEventListener(() => {
    status.textContent = "Globe rendering failed";
    emit({ version: 1, type: "failure", message: "Globe rendering failed" });
  });
  viewer.resolutionScale = 1 / Math.max(1, window.devicePixelRatio);
  viewer.scene.globe.tileCacheSize = 64;
  viewer.scene.globe.maximumScreenSpaceError = 4;
  const removeBasemap = installBasemap(
    viewer,
    import.meta.env.VITE_CESIUM_ION_ACCESS_TOKEN,
    (message) => {
      mapStatus.textContent = message;
      if (message) emit({ version: 1, type: "failure", message });
    },
  );
  let state: BridgeSnapshot = { pins: [], scans: [], boundary: [] };
  let removePins = (): void => {
    /* no pins before the first snapshot */
  };
  const outlines: Entity[] = [];
  let selectedId: string | null = null;
  let flightId: string | null = null;
  const scanLayer = new ScanLayer(
    createEmbedScanTileset,
    (tiles) => {
      viewer.scene.primitives.add(tiles);
      status.textContent = fixtureMode ? "Simulated fixture" : "";
      document.body.dataset.scanState = "ready";
      if (selectedId && flightId === selectedId) flyToScan(selectedId);
    },
    (tiles) => {
      viewer.scene.primitives.remove(tiles);
    },
    () => viewer.scene.requestRender(),
    () => {
      document.body.dataset.scanState = "failed";
      status.textContent = "Scan could not load";
      emit({ version: 1, type: "failure", message: "Scan could not load" });
    },
  );
  function selectScan(id: string): void {
    const scan = state.scans.find((s) => s.id === id);
    if (!scan) throw Error("Unknown scan");
    if (selectedId !== id) {
      status.textContent = scan.tilesetURL ? "Loading scan" : "Preview pending";
      document.body.dataset.scanState = scan.tilesetURL ? "loading" : "pending";
      selectedId = id;
      void scanLayer.select(scan.tilesetURL);
    }
    flyToScan(id);
    emit({ version: 1, type: "scanSelected", scanId: id });
  }
  function flyToScan(id: string): void {
    const scan = state.scans.find((s) => s.id === id);
    if (!scan) throw Error("Unknown scan");
    flightId = id;
    const tiles = selectedId === id ? scanLayer.visible : null;
    if (tiles) {
      viewer.camera.flyToBoundingSphere(tiles.boundingSphere, {
        offset: new HeadingPitchRange(0, -0.7, Math.max(20, tiles.boundingSphere.radius * 3)),
        duration: 0.35,
      });
      return;
    }
    viewer.camera.flyToBoundingSphere(
      new BoundingSphere(Cartesian3.fromDegrees(scan.geo.lon, scan.geo.lat, scan.geo.alt), 30),
      {
        offset: new HeadingPitchRange(0, -0.7, 120),
        duration: 0.35,
      },
    );
  }
  function polygons(values: number[][][][], id: string, color: Color): void {
    for (const [index, rings] of values.entries()) {
      if (!rings[0]?.length) continue;
      const points = (ring: number[][]): Cartesian3[] =>
        ring.map((p) => {
          const [lon, lat] = p;
          if (lon === undefined || lat === undefined) throw Error("Invalid polygon point");
          return Cartesian3.fromDegrees(lon, lat);
        });
      outlines.push(
        viewer.entities.add({
          id: `${id}:${index}`,
          polygon: {
            hierarchy: new PolygonHierarchy(
              points(rings[0]),
              rings.slice(1).map((r) => new PolygonHierarchy(points(r))),
            ),
            material: color.withAlpha(0.2),
            outline: true,
            outlineColor: color,
          },
        }),
      );
    }
  }
  window.globeBridge = {
    receive(value) {
      try {
        dispatchMessage(value, {
          snapshot(next) {
            scanLayer.dispose();
            selectedId = null;
            flightId = null;
            removePins();
            for (const entity of outlines) viewer.entities.remove(entity);
            outlines.length = 0;
            state = next;
            removePins = addAnswerPins(
              viewer.entities,
              next.pins.map((pin) => ({ ...pin, uncertaintyM: pin.uncertaintyM ?? null })),
            );
            polygons(next.boundary, "boundary", Color.GREEN);
            for (const scan of next.scans) polygons(scan.polygons, `scan:${scan.id}`, Color.CYAN);
            if (next.scans.length === 0) {
              status.textContent = "No scans yet";
              const points = next.boundary
                .flatMap((polygon) => polygon[0] ?? [])
                .flatMap((point) => {
                  const [lon, lat] = point;
                  return lon === undefined || lat === undefined
                    ? []
                    : [Cartesian3.fromDegrees(lon, lat)];
                });
              if (points.length)
                viewer.camera.flyToBoundingSphere(BoundingSphere.fromPoints(points), {
                  offset: new HeadingPitchRange(0, -0.7, 0),
                  duration: 0.35,
                });
            }
            viewer.scene.requestRender();
          },
          selectScan,
          flyToScan,
        });
      } catch {
        emit({ version: 1, type: "failure", message: "Invalid globe command" });
      }
    },
  };
  const handler = new ScreenSpaceEventHandler(viewer.canvas);
  handler.setInputAction((event: { position: Cartesian2 }) => {
    const pick: unknown = viewer.scene.pick(event.position);
    const entity = pick && typeof pick === "object" && "id" in pick ? pick.id : null;
    if (!(entity instanceof Entity)) return;
    const pin = state.pins.find((p) => entity.id === `property-answer:${p.siteId}:${p.id}`);
    if (pin) {
      emit({ version: 1, type: "pinTapped", pinId: pin.id });
      if (fixtureMode) status.textContent = pin.label;
      return;
    }
    const scan = state.scans.find((s) => entity.id.startsWith(`scan:${s.id}:`));
    if (scan) selectScan(scan.id);
  }, ScreenSpaceEventType.LEFT_CLICK);
  window.addEventListener(
    "pagehide",
    () => {
      removeBasemap();
      removeRenderError();
      mapStatus.remove();
      scanLayer.dispose();
      handler.destroy();
      viewer.destroy();
      delete window.globeBridge;
    },
    { once: true },
  );
  if (fixtureMode) {
    status.textContent = "Simulated fixture";
    for (const value of fixtures.slice(0, 3)) window.globeBridge.receive(value);
  }
  emit({ version: 1, type: "ready" });
} catch {
  status.textContent = "Globe could not start";
  emit({ version: 1, type: "failure", message: "Globe could not start" });
}

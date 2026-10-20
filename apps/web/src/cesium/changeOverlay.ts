/**
 * The change map's outlines over the scan (ADR 0018 § Web): an SVG laid over the globe above
 * every splat renderer's canvas, so the splats that cover the ground drape cannot hide it.
 *
 * The ground drape (cesium/changeDrape.ts) stays: it lies on the ground where nothing is
 * scanned. This draws each class's areas again on top, outlined with a light fill -- raised and
 * lowered in their colours, low confidence hatched in theirs, not comparable dashed in grey --
 * projected through CesiumJS's camera at the ground's height (the globe's, else the ellipsoid's;
 * the areas carry no heights of their own). A polygon with a point behind the camera is left
 * out rather than drawn torn. It is redrawn after a frame only when the view changed, at most
 * every `OVERLAY_MIN_INTERVAL_MS` (the last change always drawn). Never in a pointer's way, and
 * it does not touch the scan's own colours.
 */
import { Cartesian3, Cartographic, SceneTransforms, type Scene } from "cesium";
import type { Position } from "geojson";

import type { ChangeMap } from "@twin/contracts";

import { CHANGE_COLORS, changeBuckets, type ChangeClass } from "@/lib/changeDrape";

/** The least time between two redraws while the view moves. */
export const OVERLAY_MIN_INTERVAL_MS = 50;

const SVG = "http://www.w3.org/2000/svg";

/** A window position, or undefined where it cannot be drawn (behind the camera). */
export type Project = (position: Cartesian3) => { x: number; y: number } | undefined;

/** Fill opacity under the outline: light, so the scan reads through it. */
const FILL: Record<ChangeClass, number> = { raised: 0.22, lowered: 0.22, insufficient: 0.06 };

let made = 0;

function element<K extends keyof SVGElementTagNameMap>(
  name: K,
  attributes: Record<string, string>,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

/** Through the camera, or undefined for a point at or behind its plane. */
function projectThrough(scene: Scene): Project {
  const scratch = new Cartesian3();
  return (position) => {
    const camera = scene.camera;
    const toPoint = Cartesian3.subtract(position, camera.positionWC, scratch);
    if (Cartesian3.dot(toPoint, camera.directionWC) <= 0) return undefined;
    return SceneTransforms.worldToWindowCoordinates(scene, position) ?? undefined;
  };
}

interface Shape {
  cls: ChangeClass;
  lowConfidence: boolean;
  path: SVGPathElement;
  /** Polygons, each its rings, each ring of world positions. */
  polygons: Cartesian3[][][];
}

export class ChangeOverlay {
  private readonly svg: SVGSVGElement;
  private readonly project: Project;
  private readonly id = String((made += 1));
  private shapes: Shape[] = [];
  private removeListener: (() => void) | null;
  private drawnView = "";
  private lastDraw = Number.NEGATIVE_INFINITY;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly scene: Scene,
    options: { project?: Project } = {},
  ) {
    this.project = options.project ?? projectThrough(scene);
    this.svg = element("svg", { "aria-hidden": "true", "data-change-overlay": "" });
    Object.assign(this.svg.style, {
      position: "absolute",
      inset: "0",
      width: "100%",
      height: "100%",
      pointerEvents: "none",
      overflow: "hidden",
    });
    // Last among the globe's siblings: a splat renderer's canvas goes right after the globe's
    // (ScanRendererHost.overlayCanvas), so it stays under this one, made before or after.
    scene.canvas.parentElement?.appendChild(this.svg);
    this.removeListener = scene.postRender.addEventListener(() => this.frame());
  }

  /** Draws `map`'s areas, replacing whatever was drawn before. */
  show(map: Pick<ChangeMap, "areas">): void {
    this.clear();
    if (!this.removeListener) return;
    const defs = element("defs", {});
    this.svg.appendChild(defs);
    for (const bucket of changeBuckets(map)) {
      const color = CHANGE_COLORS[bucket.cls];
      const attributes: Record<string, string> = {
        "data-class": bucket.cls,
        "data-low-confidence": String(bucket.lowConfidence),
        stroke: color,
        "stroke-width": "1.5",
        "stroke-linejoin": "round",
        "fill-rule": "evenodd",
        fill: color,
        "fill-opacity": String(FILL[bucket.cls]),
      };
      if (bucket.cls === "insufficient") attributes["stroke-dasharray"] = "5 4";
      if (bucket.lowConfidence) {
        const id = `change-hatch-${bucket.cls}-${this.id}`;
        const pattern = element("pattern", {
          id,
          width: "8",
          height: "8",
          patternUnits: "userSpaceOnUse",
          patternTransform: "rotate(45)",
        });
        pattern.appendChild(
          element("line", {
            x1: "0",
            y1: "0",
            x2: "0",
            y2: "8",
            stroke: color,
            "stroke-width": "3",
          }),
        );
        defs.appendChild(pattern);
        attributes.fill = `url(#${id})`;
        attributes["fill-opacity"] = "0.6";
      }
      const path = element("path", attributes);
      this.svg.appendChild(path);
      this.shapes.push({
        cls: bucket.cls,
        lowConfidence: bucket.lowConfidence,
        path,
        polygons: bucket.polygons.map((rings) => rings.map((ring) => this.place(ring))),
      });
    }
    this.draw();
  }

  /** Takes every outline away. */
  clear(): void {
    this.shapes = [];
    this.svg.replaceChildren();
    this.cancel();
  }

  destroy(): void {
    this.clear();
    this.removeListener?.();
    this.removeListener = null;
    this.svg.remove();
  }

  /** A ring's positions on the ground. */
  private place(ring: Position[]): Cartesian3[] {
    const globe = (this.scene as { globe?: Scene["globe"] }).globe;
    return ring.map(([longitude = 0, latitude = 0]) => {
      const height = globe?.getHeight(Cartographic.fromDegrees(longitude, latitude)) ?? 0;
      return Cartesian3.fromDegrees(longitude, latitude, height);
    });
  }

  /** What the view is: the camera's view matrix and the canvas size. */
  private view(): string {
    const canvas = this.scene.canvas;
    const matrix = Array.from(
      { length: 16 },
      (_, i) => (this.scene.camera.viewMatrix as unknown as ArrayLike<number>)[i] ?? 0,
    );
    return `${matrix.map((v) => v.toFixed(4)).join(",")}|${String(canvas.clientWidth)}x${String(canvas.clientHeight)}`;
  }

  /** After a frame: drawn again if the view changed, now or once the interval has passed. */
  private frame(): void {
    if (!this.shapes.length || this.timer || this.view() === this.drawnView) return;
    const wait = this.lastDraw + OVERLAY_MIN_INTERVAL_MS - performance.now();
    if (wait <= 0) this.draw();
    else
      this.timer = setTimeout(() => {
        this.timer = null;
        this.draw();
      }, wait);
  }

  private cancel(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  private draw(): void {
    this.cancel();
    if (this.scene.isDestroyed()) return;
    this.drawnView = this.view();
    this.lastDraw = performance.now();
    for (const shape of this.shapes) {
      const parts: string[] = [];
      for (const rings of shape.polygons) {
        const projected = rings.map((ring) => ring.map((p) => this.project(p)));
        if (projected.some((ring) => ring.some((p) => !p))) continue;
        for (const ring of projected)
          parts.push(
            `M${ring.map((p) => `${(p?.x ?? 0).toFixed(1)},${(p?.y ?? 0).toFixed(1)}`).join("L")}Z`,
          );
      }
      shape.path.setAttribute("d", parts.join(""));
    }
  }
}

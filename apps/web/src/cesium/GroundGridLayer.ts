/**
 * One textured drape over the ground for a per-cell grid (a `LonLatGrid`): a single
 * `GroundPrimitive` rectangle whose Fabric material samples a canvas texture, one texel per
 * cell (row 0 at the north; the canvas is uploaded with flipY, so its top row lands on the
 * rectangle's north edge). Classified onto terrain and 3D tiles alike, so it shows on the
 * photorealistic tiles; never picked (`allowPicking: false`), so a tap goes through it to
 * whatever is underneath; inserted first among the ground primitives, so zone and inventory
 * overlays draw above it.
 *
 * Two materials:
 * - `fog` (the Fog of Knowing): density in R, inside the land in G. Soft, uncoloured fog
 *   broken up by fbm noise, not a heatmap. `lift(next)` dissolves it towards a new density
 *   along the noise over 1.5 s, rendering only for that long (the scene renders on request).
 * - `ramp` (batch 3, Show Once): a score in R, valid cells in G, protected cells in B, drawn
 *   above a `threshold` uniform in a two-colour ramp; moving a slider is a uniform update.
 *
 * Each consumer owns its instance (its bridge makes it); `useGridDrape` lets one drape show
 * at a time, so a second one taking the ground hides the first.
 */

import {
  Cartesian2,
  Cartesian3,
  ClassificationType,
  Color,
  type CesiumWidget,
  type Entity,
  GeometryInstance,
  GroundPrimitive,
  Material,
  MaterialAppearance,
  Rectangle,
  RectangleGeometry,
  type Scene,
  TextureMagnificationFilter,
  TextureMinificationFilter,
} from "cesium";

import type { LonLat } from "@twin/geo";

import { useGridDrape } from "@/state/gridDrape";

export type GridMaterialKind = "fog" | "ramp";

/** Pixels for the texture: one per cell, RGBA, row 0 at the north. */
export interface GridImage {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface GridBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

/** fbm noise, shared by the fog's dissolve and its wisps. Names are unique to this material. */
const NOISE = `
float knowledgeFogHash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float knowledgeFogNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(knowledgeFogHash(i), knowledgeFogHash(i + vec2(1.0, 0.0)), u.x),
    mix(knowledgeFogHash(i + vec2(0.0, 1.0)), knowledgeFogHash(i + vec2(1.0, 1.0)), u.x),
    u.y
  );
}
float knowledgeFogFbm(vec2 p) {
  float value = 0.0;
  float amplitude = 0.5;
  for (int i = 0; i < 5; i++) {
    value += amplitude * knowledgeFogNoise(p);
    p = p * 2.03 + vec2(17.0, 9.0);
    amplitude *= 0.5;
  }
  return value;
}
`;

/**
 * The fog. Density and the inside mask blend from `densityFrom` to `densityTo` where the noise
 * has burnt through at `lift` (0..1), so a lifting patch frays away rather than fading as a
 * block. Alpha follows density through a soft step, and two scales of noise keep even full
 * fog from reading as a flat sheet: broad billows (a few hundred metres) that thin it to
 * about half, and finer wisps. Only inside the land.
 */
export const FOG_SHADER = `${NOISE}
czm_material czm_getMaterial(czm_materialInput materialInput) {
  czm_material material = czm_getDefaultMaterial(materialInput);
  vec2 st = materialInput.st;
  vec4 from = texture(densityFrom, st);
  vec4 to = texture(densityTo, st);
  float n = knowledgeFogFbm(st * cells * 0.25 + drift);
  float burn = smoothstep(n - 0.08, n + 0.08, lift);
  float density = mix(from.r, to.r, burn);
  // The land's edge, softened over a cell so the 20 m staircase does not show.
  vec2 texel = 1.0 / cells;
  vec2 dx = vec2(texel.x, 0.0);
  vec2 dy = vec2(0.0, texel.y);
  float insideFrom = (from.g + texture(densityFrom, st + dx).g + texture(densityFrom, st - dx).g
    + texture(densityFrom, st + dy).g + texture(densityFrom, st - dy).g) / 5.0;
  float insideTo = (to.g + texture(densityTo, st + dx).g + texture(densityTo, st - dx).g
    + texture(densityTo, st + dy).g + texture(densityTo, st - dy).g) / 5.0;
  float inside = smoothstep(0.15, 0.85, mix(insideFrom, insideTo, burn));
  float billow = knowledgeFogFbm(st * cells * 0.08 + drift * 0.3);
  float wisp = knowledgeFogFbm(st * cells * 0.6 + drift * 0.5);
  float body = mix(0.42, 1.0, smoothstep(0.25, 0.75, billow)) * (0.82 + 0.36 * wisp);
  material.diffuse = tint.rgb * (0.9 + 0.14 * billow);
  material.emission = tint.rgb * 0.12;
  material.alpha = smoothstep(0.02, 0.9, density) * clamp(body, 0.0, 1.0) * inside * maxAlpha;
  return material;
}
`;

/** A score ramp above a threshold: crisp cells, protected ones hatched. */
export const RAMP_SHADER = `
czm_material czm_getMaterial(czm_materialInput materialInput) {
  czm_material material = czm_getDefaultMaterial(materialInput);
  vec4 cell = texture(values, materialInput.st);
  float lit = step(threshold, cell.r) * cell.g * (1.0 - cell.b);
  float t = clamp((cell.r - threshold) / max(1.0 - threshold, 0.001), 0.0, 1.0);
  float hatch = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) / 10.0)) * cell.b * cell.g;
  material.diffuse = mix(mix(low.rgb, high.rgb, t), protectedColor.rgb, hatch);
  material.alpha = max(lit * maxAlpha, hatch * protectedColor.a);
  return material;
}
`;

const FOG_TINT = new Color(0.88, 0.91, 0.92, 1);
const OUTLINE = Color.fromCssColorString("#f4f1e8");
const LIFT_MS = 1500;

/** A material's uniforms, which Cesium types as `any`. */
function uniformsOf(material: Material): Record<string, unknown> {
  return material.uniforms as Record<string, unknown>;
}

/** A canvas holding the image; a new one per update, since the material re-uploads on swap. */
function canvasOf(image: GridImage): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d");
  if (context) {
    const pixels = context.createImageData(image.width, image.height);
    pixels.data.set(image.data);
    context.putImageData(pixels, 0, 0);
  }
  return canvas;
}

/**
 * The material, its texture uniforms first declared as Cesium's default image and set to the
 * canvas after: Cesium caches a named material's fabric and deep-clones it for the next one,
 * and a canvas cannot be cloned.
 */
function materialFor(
  kind: GridMaterialKind,
  canvas: HTMLCanvasElement,
  cells: Cartesian2,
  drift: Cartesian2,
): Material {
  const placeholder = Material.DefaultImageId;
  if (kind === "fog") {
    const material = new Material({
      fabric: {
        type: "KnowledgeFog",
        uniforms: {
          densityFrom: placeholder,
          densityTo: placeholder,
          lift: 1,
          drift,
          cells,
          tint: FOG_TINT,
          maxAlpha: 0.86,
        },
        source: FOG_SHADER,
      },
      translucent: true,
    });
    const uniforms = uniformsOf(material);
    uniforms.densityFrom = canvas;
    uniforms.densityTo = canvas;
    return material;
  }
  const material = new Material({
    fabric: {
      type: "GridRamp",
      uniforms: {
        values: placeholder,
        threshold: 0.8,
        low: Color.fromCssColorString("#f0b254"),
        high: Color.fromCssColorString("#f7e3a1"),
        protectedColor: Color.fromCssColorString("#9db8f0").withAlpha(0.45),
        maxAlpha: 0.7,
      },
      source: RAMP_SHADER,
    },
    translucent: true,
    minificationFilter: TextureMinificationFilter.NEAREST,
    magnificationFilter: TextureMagnificationFilter.NEAREST,
  });
  uniformsOf(material).values = canvas;
  return material;
}

export class GroundGridLayer {
  private primitive: GroundPrimitive | null = null;
  private material: Material | null = null;
  private bounds: GridBounds | null = null;
  private outline: Entity | null = null;
  private frame: number | null = null;
  private current: HTMLCanvasElement | null = null;
  private readonly releaseDrape: () => void;

  constructor(
    private readonly widget: Pick<CesiumWidget, "entities"> & { scene: Scene },
    readonly kind: GridMaterialKind,
    /** Who holds the drape (`useGridDrape`), e.g. "fog" or "signature". */
    readonly owner: string,
    /** Called when another drape took the ground and this one hid itself. */
    private readonly onPreempted: () => void = () => undefined,
  ) {
    this.releaseDrape = useGridDrape.subscribe((state) => {
      if (this.primitive && state.owner !== this.owner) {
        this.hide();
        this.onPreempted();
      }
    });
  }

  get shown(): boolean {
    return this.primitive !== null;
  }

  private get scene(): Scene {
    return this.widget.scene;
  }

  /**
   * Shows `image` over `bounds` at once. A different rectangle rebuilds the primitive; the
   * same one only swaps the texture. `seed` varies the fog's pattern (a land id hash).
   */
  show(bounds: GridBounds, image: GridImage, seed = 0): void {
    useGridDrape.getState().claim(this.owner);
    this.stopLift();
    const canvas = canvasOf(image);
    const same =
      this.bounds &&
      this.primitive &&
      this.bounds.west === bounds.west &&
      this.bounds.south === bounds.south &&
      this.bounds.east === bounds.east &&
      this.bounds.north === bounds.north &&
      this.current?.width === image.width &&
      this.current.height === image.height;
    this.current = canvas;
    if (same && this.material) {
      const uniforms = uniformsOf(this.material);
      if (this.kind === "fog") {
        uniforms.densityFrom = canvas;
        uniforms.densityTo = canvas;
      } else {
        uniforms.values = canvas;
      }
      this.scene.requestRender();
      return;
    }
    this.removePrimitive();
    const drift = new Cartesian2((seed % 997) * 0.37, (seed % 991) * 0.53);
    const material = materialFor(
      this.kind,
      canvas,
      new Cartesian2(image.width, image.height),
      drift,
    );
    const primitive = new GroundPrimitive({
      geometryInstances: new GeometryInstance({
        geometry: new RectangleGeometry({
          rectangle: Rectangle.fromDegrees(bounds.west, bounds.south, bounds.east, bounds.north),
          vertexFormat: MaterialAppearance.MaterialSupport.TEXTURED.vertexFormat,
        }),
      }),
      appearance: new MaterialAppearance({ material, flat: true, translucent: true }),
      classificationType: ClassificationType.BOTH,
      allowPicking: false,
      asynchronous: true,
    });
    // First among the ground primitives: entity overlays (zones, inventory) draw over it.
    this.scene.groundPrimitives.add(primitive, 0);
    this.primitive = primitive;
    this.material = material;
    this.bounds = { ...bounds };
    this.scene.requestRender();
  }

  /**
   * The fog dissolves towards `image` along its noise over 1.5 s (fog only, same grid). The
   * scene renders for that long and then goes idle again. With reduced motion, or for a ramp,
   * it swaps at once.
   */
  lift(image: GridImage, reducedMotion = false): void {
    const material = this.material;
    if (!material || this.kind !== "fog" || reducedMotion || !this.bounds) {
      if (this.bounds) this.show(this.bounds, image);
      return;
    }
    this.stopLift();
    const canvas = canvasOf(image);
    // From and to are both the shown image here, so resetting the dissolve shows no change;
    // the new texture uploads on the next frame, before the dissolve has moved.
    const uniforms = uniformsOf(material);
    uniforms.lift = 0;
    uniforms.densityTo = canvas;
    this.current = canvas;
    const started = performance.now();
    const step = (now: number) => {
      if (this.material !== material) return;
      const t = Math.min(1, (now - started) / LIFT_MS);
      uniforms.lift = t;
      this.scene.requestRender();
      if (t < 1) {
        this.frame = requestAnimationFrame(step);
        return;
      }
      this.frame = null;
      // Settled: from and to are the same image, whatever `lift` says.
      uniforms.densityFrom = canvas;
      this.scene.requestRender();
    };
    this.frame = requestAnimationFrame(step);
  }

  /** A uniform of the material (the ramp's `threshold`), with a render. */
  setUniform(name: string, value: unknown): void {
    if (!this.material) return;
    uniformsOf(this.material)[name] = value;
    this.scene.requestRender();
  }

  /** Outlines one cell (or clears the outline with null). */
  setOutline(ring: LonLat[] | null): void {
    if (this.outline) this.widget.entities.remove(this.outline);
    this.outline = null;
    if (ring && ring.length >= 2) {
      this.outline = this.widget.entities.add({
        id: `grid-outline:${this.owner}`,
        polyline: {
          positions: Cartesian3.fromDegreesArray(ring.flatMap((p) => [p.longitude, p.latitude])),
          width: 2,
          material: OUTLINE,
          clampToGround: true,
        },
      });
    }
    this.scene.requestRender();
  }

  hide(): void {
    this.stopLift();
    this.removePrimitive();
    this.setOutline(null);
    useGridDrape.getState().release(this.owner);
  }

  destroy(): void {
    this.hide();
    this.releaseDrape();
  }

  private stopLift(): void {
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null;
  }

  private removePrimitive(): void {
    if (this.primitive && !this.scene.isDestroyed()) {
      this.scene.groundPrimitives.remove(this.primitive);
      this.scene.requestRender();
    }
    this.primitive = null;
    this.material = null;
    this.bounds = null;
    this.current = null;
  }
}

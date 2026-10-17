import { Cartesian4, CustomShader, UniformType, type Cesium3DTileset } from "cesium";

/**
 * The lens in drawing-buffer pixels, GL origin bottom left (timeLanternMath.toDrawingBuffer):
 * the mesh is cut away fully within `inner` and fades to whole by `outer`. `limit` is a circle
 * outside which nothing is cut, so the lens never opens where the globe is not drawn.
 */
export interface WorldLens {
  x: number;
  y: number;
  inner: number;
  outer: number;
  limit?: { x: number; y: number; r: number };
}

/**
 * Cuts the photorealistic mesh away inside the lens, with a feather: within the inner radius
 * every fragment is discarded, and between inner and outer a fragment is kept with probability
 * rising to 1, chosen by interleaved gradient noise (Jimenez 2014) so the feather reads as a
 * fine stipple rather than bands. Discard, never alpha: a translucent material would move the
 * whole Google world into the translucent pass (order-independent transparency, every tile,
 * every frame). The rim drawn over the edge (LanternRim) hides the stipple.
 */
export const WORLD_FRAGMENT_SHADER = /* glsl */ `
void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
  if (u_lantern.w <= 0.0) return;
  vec2 p = gl_FragCoord.xy;
  float d = distance(p, u_lantern.xy);
  if (d >= u_lantern.w) return;
  if (u_lanternLimit.z > 0.0 && distance(p, u_lanternLimit.xy) > u_lanternLimit.z) return;
  float keep = smoothstep(u_lantern.z, u_lantern.w, d);
  float n = fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
  if (n >= keep) discard;
}
`;

/**
 * The one owner of the photorealistic world tileset's `customShader`.
 *
 * A Cesium3DTileset has exactly one custom shader, so anything that shades the Google world goes
 * through here and adds its uniforms here, or two features silently overwrite each other. Today
 * that is the Time Lantern's lens (u_lantern, u_lanternLimit). The shader is created and
 * attached on the first lens, not before: assigning it recompiles every tile's shader once, a
 * cost nobody who never opens the lantern should pay. Once attached it stays attached; a closed
 * lens is w = 0, a uniform change, which is free. LayerManager binds the tileset as it loads and
 * unbinds it on dispose.
 */
export class WorldShader {
  private tileset: Cesium3DTileset | null = null;
  private shader: CustomShader | null = null;
  private readonly lens = new Cartesian4(0, 0, 0, 0);
  private readonly limit = new Cartesian4(0, 0, 0, 0);

  constructor(private readonly scene: { requestRender(): void }) {}

  /** The world tileset now on the globe, or null once it is gone. */
  setTileset(tileset: Cesium3DTileset | null): void {
    if (this.tileset === tileset) return;
    this.tileset = tileset;
    if (tileset && this.shader) tileset.customShader = this.shader;
    this.scene.requestRender();
  }

  /** Whether the shader is on the tileset (tests, the debug panel). */
  get attached(): boolean {
    return this.shader !== null && this.tileset?.customShader === this.shader;
  }

  /** Opens, moves or closes (null) the lens. */
  setLantern(lens: WorldLens | null): void {
    if (!lens) {
      if (this.shader && this.lens.w !== 0) {
        this.lens.w = 0;
        this.shader.setUniform("u_lantern", this.lens);
        this.scene.requestRender();
      }
      return;
    }
    this.shader ??= new CustomShader({
      uniforms: {
        u_lantern: { type: UniformType.VEC4, value: new Cartesian4(0, 0, 0, 0) },
        u_lanternLimit: { type: UniformType.VEC4, value: new Cartesian4(0, 0, 0, 0) },
      },
      fragmentShaderText: WORLD_FRAGMENT_SHADER,
    });
    if (this.tileset && this.tileset.customShader !== this.shader)
      this.tileset.customShader = this.shader;
    this.lens.x = lens.x;
    this.lens.y = lens.y;
    this.lens.z = Math.min(lens.inner, lens.outer);
    this.lens.w = Math.max(lens.outer, 1e-3);
    this.shader.setUniform("u_lantern", this.lens);
    this.limit.x = lens.limit?.x ?? 0;
    this.limit.y = lens.limit?.y ?? 0;
    this.limit.z = lens.limit?.r ?? 0;
    this.shader.setUniform("u_lanternLimit", this.limit);
    this.scene.requestRender();
  }
}

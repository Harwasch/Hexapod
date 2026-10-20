/**
 * Sunlight on a splat scan: each clearly flat splat shaded by the real sun at the scene clock's
 * time, in the CesiumJS splat primitive's vertex shader, as one part of its colour chain
 * (`splatColor.ts`). ADR 0019 "Sunlight".
 *
 * A splat carries no normal, so the part makes one from what the engine already holds: it
 * fetches the splat's covariance texel again (the odd column of `u_splatAttributeTexture`, at
 * the address the engine's own `u_splatRowMask`/`u_splatRowShift` give, as the vertex shader
 * does: `Shaders/PrimitiveGaussianSplatVS.glsl`), and takes the smallest axis of a clearly flat
 * one (`lib/splatNormals.ts`, which this GLSL transcribes). That normal is turned toward the
 * eye, into eye coordinates by `mat3(czm_modelView)`, and lit by `czm_sunDirectionEC` -- the
 * engine's own sun, which follows `clock.currentTime` (UniformState `setSunAndMoonDirections`),
 * so the relight's sun is the sky's sun. Automatic uniforms are declared by the engine only for
 * the `czm_` names a shader's text uses; this one's text uses it, so it is declared
 * (e2e/sunlightScene.spec.ts compiles it under SwiftShader).
 *
 * `c.rgb *= mix(1, 0.35 + 0.65 · lambert · sunUp, strength)`. A splat with no normal counts as
 * facing the sun: unchanged by day, at the ambient floor at night with everything else.
 * `sunUp` comes from the CPU (`sunSine` below, the engine's own sun arithmetic at the scan's
 * centre), once a frame, so the shader and the badge that says "Sun below horizon" agree.
 *
 * Splats a motion moves (the wind on skins, a split object turned) are relit by their moved
 * covariance: the motion chain's Jacobian, when it declares one, is applied as the engine
 * applies it before projecting.
 *
 * Order 3: after the objects' highlight (0), before Confidence / Data age (5), which tint by
 * the relit luminance, the inferred layers' Highlight (10) and the compositor's (11). Cost:
 * nothing while off (the part is not in the chain, `sunlightSlot.ts`); while on, one more
 * texel fetch, a closed-form eigen solution and a few dozen flops a splat.
 *
 * What it cannot do: take out the capture's own shading. The shadows the photographs had are
 * in the colours; the relight darkens what faces away from today's sun over them.
 */

import {
  Cartesian3,
  Cartesian4,
  Ellipsoid,
  Matrix3,
  Simon1994PlanetaryPositions,
  Transforms,
  type JulianDate,
} from "cesium";

import { FLAT_RATIO, ROUND_SPREAD, SUN_AMBIENT } from "@/lib/splatNormals";

import {
  addColorPart,
  hasColorPart,
  removeColorPart,
  type ColorPrimitive,
  type SplatColorPart,
} from "./splatColor";
import type { SplatShaderBuilder } from "./splatInternals";

const float = (x: number): string => {
  const text = String(x);
  return text.includes(".") || text.includes("e") ? text : `${text}.0`;
};

/**
 * GLSL of the part. Its own uniform: (strength, sunUp, -, -). The engine's: the attribute
 * texture and its row mask and shift (declared before any hook's lines), `czm_modelView`,
 * `czm_sunDirectionEC`.
 */
export const SUNLIGHT_GLSL = `
uniform vec4 u_sunlight;
const float SPLAT_SUN_FLAT = ${float(FLAT_RATIO)};
const float SPLAT_SUN_AMBIENT = ${float(SUN_AMBIENT)};

// The unit smallest axis of a clearly flat covariance; vec3(0) for anything rounder
// (lib/splatNormals.ts splatNormal: Smith's closed form, then the largest row cross product).
vec3 splatSunNormal(mat3 a) {
    float scale = max(max(a[0][0], a[1][1]), a[2][2]);
    if (!(scale > 0.0)) {
        return vec3(0.0);
    }
    mat3 b = a / scale;
    float q = (b[0][0] + b[1][1] + b[2][2]) / 3.0;
    float p1 = b[0][1] * b[0][1] + b[0][2] * b[0][2] + b[1][2] * b[1][2];
    vec3 d = vec3(b[0][0], b[1][1], b[2][2]) - q;
    float p = sqrt((dot(d, d) + 2.0 * p1) / 6.0);
    if (p < ${float(ROUND_SPREAD)}) {
        return vec3(0.0);
    }
    float r = clamp(determinant((b - q * mat3(1.0)) / p) * 0.5, -1.0, 1.0);
    float phi = acos(r) / 3.0;
    float l1 = q + 2.0 * p * cos(phi);
    float l3 = q + 2.0 * p * cos(phi + 2.0943951023931953);
    float l2 = 3.0 * q - l1 - l3;
    if (!(l3 < SPLAT_SUN_FLAT * l2)) {
        return vec3(0.0);
    }
    mat3 m = b - l3 * mat3(1.0);
    vec3 c01 = cross(m[0], m[1]);
    vec3 c02 = cross(m[0], m[2]);
    vec3 c12 = cross(m[1], m[2]);
    float d01 = dot(c01, c01);
    float d02 = dot(c02, c02);
    float d12 = dot(c12, c12);
    vec3 n = d01 >= d02 && d01 >= d12 ? c01 : (d02 >= d12 ? c02 : c12);
    float length2 = max(d01, max(d02, d12));
    return length2 > 0.0 ? n * inversesqrt(length2) : vec3(0.0);
}

vec4 splatSunlightColor(uint splatIndex, vec3 position, vec4 color) {
    if (u_sunlight.x <= 0.0) {
        return color;
    }
    float lambert = 1.0;
    uint rowMask = uint(u_splatRowMask);
    uint rowShift = uint(u_splatRowShift);
    uvec4 texel = uvec4(texelFetch(
        u_splatAttributeTexture,
        ivec2(int(((splatIndex & rowMask) << 1) | 1u), int(splatIndex >> rowShift)),
        0
    ));
    vec2 u1 = unpackHalf2x16(texel.x);
    vec2 u2 = unpackHalf2x16(texel.y);
    vec2 u3 = unpackHalf2x16(texel.z);
    mat3 covariance = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);
#if defined(HAS_SPLAT_VERTEX_MOTION) && defined(HAS_SPLAT_VERTEX_JACOBIAN)
    mat3 jacobian = splatVertexJacobian(splatIndex, position);
    covariance = jacobian * covariance * transpose(jacobian);
#endif
    vec3 n = splatSunNormal(covariance);
    if (dot(n, n) > 0.0) {
        vec3 positionEC = (czm_modelView * vec4(position, 1.0)).xyz;
        vec3 normalEC = normalize(mat3(czm_modelView) * n);
        // Toward the eye: a splat is seen from the side it is drawn on.
        if (dot(normalEC, positionEC) > 0.0) {
            normalEC = -normalEC;
        }
        lambert = max(dot(normalEC, czm_sunDirectionEC), 0.0);
    }
    float light = SPLAT_SUN_AMBIENT + (1.0 - SPLAT_SUN_AMBIENT) * lambert * u_sunlight.y;
    color.rgb *= mix(1.0, light, u_sunlight.x);
    return color;
}
`;

/** How the part makes its uniform value. The real one is a `Cartesian4`. */
export type Vec4Factory = (x: number, y: number, z: number, w: number) => unknown;

const cartesian4: Vec4Factory = (x, y, z, w) => new Cartesian4(x, y, z, w);

/** The sunlight part of one scan's primitive. */
export class SplatSunlight implements SplatColorPart {
  readonly colorFunction = "splatSunlightColor";
  /** After the objects' highlight (0), before Confidence / Data age (5). */
  readonly colorOrder = 3;
  readonly #vec4: Vec4Factory;
  #strength = 1;
  #sunUp = 1;
  #value: unknown;
  #primitive: ColorPrimitive | undefined;

  constructor(vec4: Vec4Factory = cartesian4) {
    this.#vec4 = vec4;
    this.#value = vec4(this.#strength, this.#sunUp, 0, 0);
  }

  /** How much of the direct term the sun gives now (`sunUpFactor`); uniforms only. */
  get sunUp(): number {
    return this.#sunUp;
  }

  set sunUp(value: number) {
    if (value === this.#sunUp) return;
    this.#sunUp = value;
    this.#value = this.#vec4(this.#strength, value, 0, 0);
  }

  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    _context: unknown,
  ): void {
    shaderBuilder.addVertexLines(SUNLIGHT_GLSL);
    uniformMap.u_sunlight = () => this.#value;
  }

  install(primitive: ColorPrimitive): boolean {
    if (this.#primitive === primitive && hasColorPart(primitive, this)) return true;
    if (this.#primitive && this.#primitive !== primitive) this.uninstall();
    if (!addColorPart(primitive, this)) return false;
    this.#primitive = primitive;
    return true;
  }

  get installed(): boolean {
    return hasColorPart(this.#primitive, this);
  }

  uninstall(): void {
    if (this.#primitive) removeColorPart(this.#primitive, this);
    this.#primitive = undefined;
  }
}

const icrfToFixed = new Matrix3();
const sun = new Cartesian3();
const up = new Cartesian3();

/**
 * The sine of the sun's elevation at `position` (Earth-fixed metres) at `time`, by the engine's
 * own arithmetic (UniformState `setSunAndMoonDirections`), so it is the sun the shader is lit
 * by. NaN where there is no surface normal (the Earth's centre: a scan not placed yet).
 */
export function sunSine(time: JulianDate, position: Cartesian3): number {
  const normal = Ellipsoid.WGS84.geodeticSurfaceNormal(position, up);
  if (!normal || !Number.isFinite(normal.x)) return Number.NaN;
  Transforms.computeIcrfToCentralBodyFixedMatrix(time, icrfToFixed);
  Simon1994PlanetaryPositions.computeSunPositionInEarthInertialFrame(time, sun);
  Matrix3.multiplyByVector(icrfToFixed, sun, sun);
  Cartesian3.normalize(sun, sun);
  return Cartesian3.dot(sun, normal);
}

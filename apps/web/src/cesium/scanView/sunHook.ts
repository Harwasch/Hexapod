/**
 * Sunlight under PlayCanvas (ADR 0019 "Sunlight"): the relight CesiumJS draws in its splat
 * primitive (cesium/splatSunlight.ts), drawn here by PlayCanvas's per-frame splat hook -- the
 * user material `app.scene.gsplat.material` and its `gsplatModifyVS` chunk, which PlayCanvas
 * copies onto the unified renderer's draw material whenever the material's version moves. In
 * that vertex shader `initCorner` hands `modifySplatRotationScale` the work buffer's rotation
 * (x, y, z, w) and scales, which this keeps, before `modifySplatColor(centre, colour)`; the
 * unified renderer's frame is its manager's identity node, which is the scan's frame (pose.ts),
 * and the eye comes from `view_position`, which the forward renderer sets for every camera.
 *
 * Per splat, every frame, as CesiumJS: the smallest axis as the normal when the splat is
 * clearly flat (`lib/splatNormals.ts` `scaleNormal`, the covariance's own test on the squared
 * scales), turned toward the eye, lambert against the sun's direction, and
 * `rgb *= mix(1, 0.35 + 0.65 · lambert · sunUp, strength)`. The sun's direction (in the scan's
 * frame) and `sunUp` are worked out on the CPU each frame by cesium/sunlight.ts, by the
 * engine's own sun arithmetic, and handed over as uniforms (state/sunReport.ts): a uniform set per
 * frame copies nothing into the work buffer. The relight's two constants (the flatness ratio and
 * the ambient floor) come with the sun too, so this chunk holds no copy of lib/splatNormals.ts
 * (whose `scaleSunlightFactor` is this GLSL in TypeScript) and the globe's first load gains no
 * chunk for it.
 *
 * Two things CesiumJS's chain has that a hook after the copy pass does not, carried by one
 * 32-bit word per splat, `hexapodSun`, an extra stream of the work buffer the copy pass writes
 * (playcanvasBackend.ts adds it the first time the sun is on, and every tile's modifier then
 * writes it):
 *
 * - Whether a copy has written it at all (bit 31): a new stream starts at zero, and a splat
 *   whose tile has not been copied since is drawn as it is, for the frame or so that takes.
 * - Whether the splat is lit at all. An inferred layer's tiles are drawn by the same renderer
 *   but are not the capture (CesiumJS draws them as a tileset of their own, which the relight
 *   never joins): their modifier sets bit 30, and the hook leaves them as they are.
 * - The order. CesiumJS lights before Confidence / Data age (order 3 before 5), which tint by the
 *   relit luminance; here the look is drawn in the copy pass and the hook comes after it. Every
 *   branch of the look is affine in the splat's colour scale -- `look(s·c) = A + s·(look(c) - A)`
 *   with `A = look(black)` -- so the copy pass also draws the look of a black splat and keeps it
 *   in the word (ten bits a channel), and the hook relights `A + (drawn - A) · factor`: the look
 *   of the relit splat, CesiumJS's order, within half a step of 1/1023. A word with no look is
 *   `A = 0`: the photograph times the factor.
 *
 * GLSL only, as the motion and the cell views: the WebGPU trial does not draw it.
 */

import type { Vec3 } from "@/lib/splatNormals";

/** The work buffer's extra stream (R32U): PlayCanvas names its accessors `load/writeHexapodSun`. */
export const SUN_STREAM = "hexapodSun";
/** Set in the word of a splat the relight leaves alone (an inferred layer's). */
export const SUN_SKIP_BIT = 2 ** 30;
/**
 * Set in every word a copy writes. A word with it clear is one no copy has written yet -- the
 * stream new with the first sun, or the work buffer made again -- and its splat is drawn as it
 * is until its tile's copy lands: neither lit with no look nor, a layer's, lit for a frame.
 */
export const SUN_WRITTEN_BIT = 2 ** 31;
const STEPS = 1023;

const float = (x: number): string => {
  const text = String(x);
  return text.includes(".") || text.includes("e") ? text : `${text}.0`;
};

/**
 * What the copy pass writes (playcanvasBackend.ts modifiers): `hexapodSunWord(black)` packs the
 * look's black (ten bits a channel); `HEXAPOD_SUN_SKIP`, an inferred layer's word. Both carry
 * the written bit.
 */
export const SUN_WORD_GLSL = `
const uint HEXAPOD_SUN_WRITTEN = ${String(SUN_WRITTEN_BIT)}u;
const uint HEXAPOD_SUN_SKIP = HEXAPOD_SUN_WRITTEN | ${String(SUN_SKIP_BIT)}u;
uint hexapodSunWord(vec3 black) {
    uvec3 q = uvec3(clamp(black, 0.0, 1.0) * ${float(STEPS)} + 0.5);
    return HEXAPOD_SUN_WRITTEN | q.r | (q.g << 10u) | (q.b << 20u);
}
`;

/**
 * The final draw's hook (`gsplatModifyVS` on `app.scene.gsplat.material`). Uniforms:
 * `uHexapodSun` (the sun's unit direction in the scan's frame, `sunUp`) and `uHexapodSunlight`
 * (strength, `SUN_AMBIENT`, `FLAT_RATIO`, -). Reads the copy pass's word (`loadHexapodSun`), so it is installed only
 * once the work buffer has the stream.
 */
export const SUN_HOOK_VS = `
uniform vec4 uHexapodSun;
uniform vec4 uHexapodSunlight;
uniform vec3 view_position;
vec4 hexapodSunRotation = vec4(0.0, 0.0, 0.0, 1.0);
vec3 hexapodSunScale = vec3(1.0);
// The unit smallest axis of a clearly flat splat (lib/splatNormals.ts scaleNormal: smallest
// variance under FLAT_RATIO, uHexapodSunlight.z, of the middle one); vec3(0) for anything rounder.
vec3 hexapodSunNormal() {
    vec3 v = hexapodSunScale * hexapodSunScale;
    vec3 axis;
    float lo;
    float mid;
    if (v.x <= v.y && v.x <= v.z) {
        axis = vec3(1.0, 0.0, 0.0); lo = v.x; mid = min(v.y, v.z);
    } else if (v.y <= v.z) {
        axis = vec3(0.0, 1.0, 0.0); lo = v.y; mid = min(v.x, v.z);
    } else {
        axis = vec3(0.0, 0.0, 1.0); lo = v.z; mid = min(v.x, v.y);
    }
    if (!(lo < uHexapodSunlight.z * mid) || !(dot(hexapodSunRotation, hexapodSunRotation) > 0.0)) {
        return vec3(0.0);
    }
    vec4 q = normalize(hexapodSunRotation);
    vec3 n = axis + 2.0 * cross(q.xyz, cross(q.xyz, axis) + q.w * axis);
    return normalize(n);
}
// Written by a copy, and not a layer's (sunHook.ts SUN_WRITTEN_BIT, SUN_SKIP_BIT).
bool hexapodSunLit(uint word) {
    return (word & ${String(SUN_WRITTEN_BIT)}u) != 0u && (word & ${String(SUN_SKIP_BIT)}u) == 0u;
}
vec3 hexapodSunBlack(uint word) {
    return vec3(float(word & 1023u), float((word >> 10u) & 1023u), float((word >> 20u) & 1023u)) / ${float(STEPS)};
}
void modifySplatCenter(inout vec3 center) {
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
    hexapodSunRotation = rotation;
    hexapodSunScale = scale;
}
void modifySplatColor(vec3 center, inout vec4 color) {
    uint word = loadHexapodSun().r;
    if (uHexapodSunlight.x <= 0.0 || !hexapodSunLit(word)) {
        return;
    }
    float lambert = 1.0;
    vec3 n = hexapodSunNormal();
    if (dot(n, n) > 0.0) {
        // Toward the eye: a splat is seen from the side it is drawn on.
        if (dot(n, view_position - center) < 0.0) {
            n = -n;
        }
        lambert = max(dot(n, uHexapodSun.xyz), 0.0);
    }
    float light = uHexapodSunlight.y + (1.0 - uHexapodSunlight.y) * lambert * uHexapodSun.w;
    vec3 black = hexapodSunBlack(word);
    color.rgb = black + (color.rgb - black) * mix(1.0, light, uHexapodSunlight.x);
}
`;

/**
 * The word the GLSL packs (unsigned): `black` (0-1 a channel) at ten bits each, the skip bit,
 * and the written bit, which every word a copy writes has.
 */
export function packSunWord(black: Vec3, skip: boolean): number {
  const q = (v: number) => Math.floor(Math.min(1, Math.max(0, v)) * STEPS + 0.5);
  const channels = q(black[0]) | (q(black[1]) << 10) | (q(black[2]) << 20);
  return SUN_WRITTEN_BIT + (skip ? SUN_SKIP_BIT : 0) + channels;
}

/** What the hook reads back from a word. */
export function unpackSunWord(word: number): {
  written: boolean;
  skip: boolean;
  black: [number, number, number];
} {
  return {
    written: word >>> 31 === 1,
    skip: ((word >>> 30) & 1) === 1,
    black: [(word & 1023) / STEPS, ((word >>> 10) & 1023) / STEPS, ((word >>> 20) & 1023) / STEPS],
  };
}

/** Whether the hook relights a splat with `word` (`hexapodSunLit`). */
export function sunWordLit(word: number): boolean {
  const { written, skip } = unpackSunWord(word);
  return written && !skip;
}

/** The hook's colour: `black + (drawn - black) · factor`, alpha untouched. */
export function relightAfterLook(
  drawn: readonly [number, number, number, number],
  black: readonly [number, number, number],
  factor: number,
): [number, number, number, number] {
  return [
    black[0] + (drawn[0] - black[0]) * factor,
    black[1] + (drawn[1] - black[1]) * factor,
    black[2] + (drawn[2] - black[2]) * factor,
    drawn[3],
  ];
}

/**
 * The compare cut's own decision (scanView/scanCompare.ts `COMPARE_HOOK_PS`): which fragments the
 * frame's fragment hook discards, read from the shader's text and run here for every case --
 * the side (none, left, right), pixels either side of the divider and at it, the cut on and
 * off. The pixels themselves are e2e/compareScene.spec.ts's; this is the fast check, which a
 * cut flipped or moved by a pixel fails without a GPU.
 */
import { describe, expect, it } from "vitest";

import { COMPARE_HOOK_PS, SIDE_LEFT, SIDE_NONE, SIDE_RIGHT } from "@/cesium/scanView/scanCompare";

/**
 * The hook's body as a function of (side, gl_FragCoord.x, divider x, on): true where it
 * discards. Only the few GLSL forms the hook is written in are read: the early return on the
 * cut being off, `bool left = …;`, and the `if (…) { discard; }` condition, with unsigned
 * literals (`1u`) and `==`. A shape it does not know fails the test rather than passing it.
 */
function discards(): (side: number, fragX: number, divider: number, on: number) => boolean {
  const off = /if \(uHexapodDivider\.y < 0\.5\) \{\s*return;\s*\}/.exec(COMPARE_HOOK_PS);
  const left = /bool left = (gl_FragCoord\.x\s*(?:<|<=|>|>=)\s*uHexapodDivider\.x);/.exec(
    COMPARE_HOOK_PS,
  );
  const cut = /if \(((?:[^{}])*)\)\s*\{\s*discard;\s*\}/.exec(COMPARE_HOOK_PS);
  if (!off || !left?.[1] || !cut?.[1]) throw new Error("the hook's shape is not the one read here");
  const js = (glsl: string) =>
    glsl
      .replace(/gl_FragCoord\.x/g, "fragX")
      .replace(/uHexapodDivider\.x/g, "divider")
      .replace(/(\d+)u\b/g, "$1")
      .replace(/==/g, "===");
  // eslint-disable-next-line @typescript-eslint/no-implied-eval -- the shader's own text, run.
  const run = new Function(
    "side",
    "fragX",
    "divider",
    "on",
    `if (on < 0.5) return false; const left = ${js(left[1])}; return ${js(cut[1])};`,
  ) as (side: number, fragX: number, divider: number, on: number) => boolean;
  return run;
}

describe("the compare cut, as the fragment hook decides it", () => {
  const discard = discards();
  const divider = 480;
  /** Pixel columns either side of the divider and on it, at their centres as gl_FragCoord has them. */
  const columns = [470, 477, 478, 479, 480, 481, 482, 490].map((c) => c + 0.5);

  it("keeps the left date left of the divider only, the right date right of it only", () => {
    for (const x of columns) {
      const leftOfDivider = x < divider;
      expect(discard(SIDE_LEFT, x, divider, 1)).toBe(!leftOfDivider);
      expect(discard(SIDE_RIGHT, x, divider, 1)).toBe(leftOfDivider);
    }
  });

  it("cuts exactly at the divider: column 479 is the left's, column 480 the right's", () => {
    expect(discard(SIDE_LEFT, 479.5, divider, 1)).toBe(false);
    expect(discard(SIDE_RIGHT, 479.5, divider, 1)).toBe(true);
    expect(discard(SIDE_LEFT, 480.5, divider, 1)).toBe(true);
    expect(discard(SIDE_RIGHT, 480.5, divider, 1)).toBe(false);
    // Each column is one date's: never both kept, never both discarded.
    for (const x of columns)
      expect(discard(SIDE_LEFT, x, divider, 1)).not.toBe(discard(SIDE_RIGHT, x, divider, 1));
  });

  it("discards nothing of a splat of no date, nor anything with the cut off", () => {
    for (const x of columns) {
      expect(discard(SIDE_NONE, x, divider, 1)).toBe(false);
      for (const side of [SIDE_NONE, SIDE_LEFT, SIDE_RIGHT])
        expect(discard(side, x, divider, 0)).toBe(false);
    }
  });
});

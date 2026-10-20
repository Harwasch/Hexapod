/**
 * The colours of the analytic splat views, shared by the shader part (`cesium/splatCellView.ts`)
 * and the legend (`features/views/CellLegend.tsx`), so the key always matches what is drawn.
 * RGB, 0 to 1, as the splat colours are; the shader shades each by the splat's own luminance.
 */

export type Rgb = readonly [number, number, number];

/**
 * Confidence: [unused: no evidence is the splat desaturated], weak, then four steps from the
 * weak threshold to 255 -- two "fair" (amber, yellow), two "strong" (yellow-green, green).
 */
export const CONFIDENCE_RAMP: readonly Rgb[] = [
  [0.5, 0.5, 0.5],
  [0.93, 0.13, 0.1],
  [0.97, 0.55, 0.12],
  [0.95, 0.82, 0.2],
  [0.55, 0.84, 0.28],
  [0.12, 0.72, 0.33],
];

/** Data age, by bucket (`lib/dataAge.ts` `AGE_BUCKETS`): fresh cool, old warm and dull. */
export const AGE_RAMP: readonly Rgb[] = [
  [0.16, 0.78, 0.92],
  [0.25, 0.55, 0.96],
  [0.5, 0.42, 0.92],
  [0.76, 0.38, 0.78],
  [0.78, 0.55, 0.36],
  [0.55, 0.5, 0.44],
];

/** CSS for a ramp colour, as the legend's swatch. */
export function rampCss([r, g, b]: Rgb): string {
  return `rgb(${String(Math.round(r * 255))} ${String(Math.round(g * 255))} ${String(Math.round(b * 255))})`;
}

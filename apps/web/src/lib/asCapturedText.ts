/**
 * As captured's words (state/asCaptured.ts, ADR 0019), here so the chip, the switch and the
 * tests say the same, and the globe's first load carries none of them.
 */
export const AS_CAPTURED_TEXT = {
  label: "As captured",
  // Exactly what it does: plans, zones' outlines and names stay drawn, as annotations.
  chip: "As captured — only what was measured is drawn; plans stay as annotations",
  hint: "Only what was measured: no inferred fill, no modelled motion, nothing a simulated machine did",
  off: "Turn off",
  rule: "Drawn as Ground truth while As captured is on",
} as const;

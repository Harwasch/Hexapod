/**
 * As captured's words (state/asCaptured.ts, ADR 0019), here so the chip, the switch and the
 * tests say the same, and the globe's first load carries none of them.
 */
export const AS_CAPTURED_TEXT = {
  label: "As captured",
  chip: "As captured — nothing simulated or inferred is drawn",
  hint: "Only what the capture measured: no inferred fill, no modelled motion, no simulated machine",
  off: "Turn off",
  rule: "Drawn as Ground truth while As captured is on",
} as const;

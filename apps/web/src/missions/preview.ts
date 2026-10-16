/**
 * The spacing the map draws a plan's coverage passes at: a preview, not a machine's swath.
 *
 * Pure (no Cesium) so the demo project can stand its machines on the same passes the plan
 * preview draws (`missions/demo.ts`) without importing the scene. A machine's own assumed
 * width and speed are its class's (apps/api/app/services/machine_classes.json); these two
 * numbers only decide how many lines the map shows.
 */

/** Spacing of the drawn passes, metres: "preview spacing", not a machine's swath. */
export const PREVIEW_SWATH_M = 6;
/** The preview never draws more passes than this per zone; the spacing widens instead. */
export const PREVIEW_MAX_PASSES = 28;

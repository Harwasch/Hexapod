/** Types for picker.js, the request form's site picker. */

export interface SitePicker {
  /** The total area as the form shows it, e.g. "52.3 acres · 21.2 ha". */
  summary: () => string;
  /** Parcels selected plus areas drawn. */
  count: () => number;
}

/** Starts the picker on the form's #pk-map canvas and its controls. */
export function startSitePicker(): SitePicker;

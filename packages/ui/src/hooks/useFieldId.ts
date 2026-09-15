import { useId } from "react";

/** Stable ids for label/control pairing inside forms. */
export function useFieldId(prefix = "field"): string {
  return `${prefix}-${useId()}`;
}

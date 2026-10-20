import { clsx } from "clsx";

import { GlassSegmentedControl } from "@twin/ui";

import {
  describeEvidence,
  INFERRED_LEGEND,
  INFERRED_STYLES,
  INFERRED_STYLE_LABELS,
  type InferredEvidence,
  type InferredStyle,
} from "@/lib/inferred";
import { HIDDEN_AS_CAPTURED, useAsCaptured } from "@/state/asCaptured";
import { useSettings } from "@/state/settings";

const options = (disabled: boolean) =>
  INFERRED_STYLES.map((style) => ({
    value: style,
    label: INFERRED_STYLE_LABELS[style],
    ariaLabel: `${INFERRED_STYLE_LABELS[style]} inferred fill`,
    disabled,
  }));

/**
 * Show · Highlight · Hide for a scan's inferred layers (lib/inferred.ts), beside the
 * representation switcher: as painted, purple and hatched (the measured splats untouched), or
 * not drawn. A viewer's setting (`inferredStyle`), kept on this device. What painted each layer
 * is its title. Disabled while As captured is on (state/asCaptured.ts), the setting kept for
 * after.
 */
export function InferredStyleControl({ evidence }: { evidence: readonly InferredEvidence[] }) {
  const style = useSettings((s) => s.inferredStyle);
  const set = useSettings((s) => s.set);
  const held = useAsCaptured((s) => s.on);
  return (
    <span
      className="rep-switch__inferred"
      data-testid="inferred-style"
      title={evidence.map(describeEvidence).join("\n")}
    >
      <span aria-hidden="true">Inferred</span>
      <GlassSegmentedControl
        aria-label="Inferred fill"
        className="rep-switch__inferred-control"
        value={style}
        onValueChange={(next: InferredStyle) => set({ inferredStyle: next })}
        options={options(held)}
      />
      {held && <span className="views__notice">{HIDDEN_AS_CAPTURED}</span>}
    </span>
  );
}

/** The one-line legend while inferred layers are drawn: what they are, and that they are not measured. */
export function InferredLegend({ style }: { style: InferredStyle }) {
  return (
    <p className="inferred-legend" data-testid="inferred-legend">
      <span
        className={clsx(
          "inferred-legend__swatch",
          style === "highlight" && "inferred-legend__swatch--highlight",
        )}
        aria-hidden="true"
      />
      {INFERRED_LEGEND}
    </p>
  );
}

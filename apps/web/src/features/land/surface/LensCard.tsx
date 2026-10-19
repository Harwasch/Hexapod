import type { ReactNode } from "react";

import { GlassPanel } from "@twin/ui";

import { lensMeta, type LensId } from "./lenses";

/**
 * The one card a lens keeps in the left dock: what the lens is, one plain sentence about the
 * land, then whatever the lens shows (facts that point at the map, a single next step), and
 * the full tools and sources folded away at the bottom.
 */
export function LensCard({
  lens,
  headline,
  action,
  children,
  testId,
}: {
  lens: LensId;
  headline?: ReactNode;
  /** A control at the top right: refresh, a menu, a status pill. */
  action?: ReactNode;
  children?: ReactNode;
  testId?: string;
}) {
  const meta = lensMeta(lens);
  return (
    <GlassPanel
      strong
      className="land-card land-card--scroll land-lens-card"
      role="region"
      aria-label={meta.label}
      data-testid={testId ?? `land-lens-card-${lens}`}
    >
      <div className="land-eyebrow-row">
        <span className="land-lens-label" style={{ color: meta.color }}>
          {meta.label}
        </span>
        {action}
      </div>
      {headline && <h3 className="land-headline">{headline}</h3>}
      {children}
    </GlassPanel>
  );
}

/** A lens with nothing to show yet: what it will show, and the one step that fetches it. */
export function LensEmpty({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="land-lens-empty">
      <p className="land-lede">{children}</p>
      {action && <div className="land-row">{action}</div>}
    </div>
  );
}

/** The full tools and raw data under a summary, folded away until asked for. */
export function LensDetails({
  summary = "Details and sources",
  children,
  defaultOpen,
}: {
  summary?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="land-details" open={defaultOpen}>
      <summary>{summary}</summary>
      <div className="land-details__body">{children}</div>
    </details>
  );
}

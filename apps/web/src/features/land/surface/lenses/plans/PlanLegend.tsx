/** A tiny key for the map colours a plan uses. */
export function PlanLegend({ steps, avoid }: { steps: boolean; avoid: boolean }) {
  if (!steps && !avoid) return null;
  return (
    <p className="plans-legend">
      <span>
        <i className="plans-legend__work" aria-hidden="true" /> Work area
      </span>
      {steps && (
        <span>
          <i className="plans-legend__step" aria-hidden="true" /> Step areas
        </span>
      )}
      {avoid && (
        <span>
          <i className="plans-legend__avoid" aria-hidden="true" /> Areas to avoid
        </span>
      )}
    </p>
  );
}

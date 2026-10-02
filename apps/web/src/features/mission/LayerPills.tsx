import { useQuickLayers } from "./quickLayers";

/** Quick layer toggles on the right edge (design: LAYER TOGGLES). The command box lists the same four. */
export function LayerPills() {
  const pills = useQuickLayers();
  return (
    <div
      className="glass glass--sm mc-pills"
      role="group"
      aria-label="Quick layers"
      data-testid="layer-pills"
    >
      {pills.map((pill) => (
        <button
          key={pill.id}
          type="button"
          className={`mc-pill ${pill.on ? "is-on" : ""}`}
          aria-pressed={pill.on}
          disabled={pill.disabled}
          onClick={pill.toggle}
          data-testid={`pill-${pill.id}`}
        >
          {pill.label}
        </button>
      ))}
    </div>
  );
}

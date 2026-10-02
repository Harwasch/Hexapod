import { useQuickLayers } from "../mission/quickLayers";

/**
 * The four layers an operator flips most, at the top of the Layers panel: imagery, vegetation,
 * zones and tracks. They used to be pills over the map's top-right corner; one tap still
 * flips each, and the command box lists the same four.
 */
export function LayerFavourites() {
  const pills = useQuickLayers();
  return (
    <section className="layer-favourites" aria-labelledby="layer-favourites-heading">
      <p className="glass-eyebrow" id="layer-favourites-heading">
        Favourites
      </p>
      <div
        className="mc-pills layer-favourites__pills"
        role="group"
        aria-labelledby="layer-favourites-heading"
        data-testid="layer-favourites"
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
    </section>
  );
}

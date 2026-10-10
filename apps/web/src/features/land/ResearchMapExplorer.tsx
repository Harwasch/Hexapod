import { useEffect, useState } from "react";
import type { ResearchArtifact } from "@twin/contracts";
import { useScene } from "@/cesium/SceneContext";
import { useLandContext } from "@/state/landContext";
import { mapValue, researchMapBounds } from "./researchMap";

type MapOutput = Extract<ResearchArtifact["output"], { kind: "map" }>;
export function ResearchMapExplorer({
  artifactId,
  title,
  output,
}: {
  artifactId: string;
  title: string;
  output: MapOutput;
}) {
  const scene = useScene();
  const layer = useLandContext((state) => state.layers[artifactId]);
  const selection = useLandContext((state) => state.selectedMapFeature);
  const [search, setSearch] = useState(""),
    [page, setPage] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const selected = selection?.layerId === artifactId ? selection : null;
  useEffect(() => {
    if (selected?.origin !== "map") return;
    const index = Number(selected.featureId);
    if (!Number.isInteger(index) || index < 0 || index >= output.features.length) return;
    let cancelled = false;
    queueMicrotask(() => {
      if (!cancelled) {
        setSearch("");
        setPage(Math.floor(index / 50));
        setExpanded(true);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [selected, output.features.length]);
  const show = () => {
    if (!layer)
      useLandContext.getState().setLayer({
        id: artifactId,
        researchArtifactId: artifactId,
        title,
        unit: output.unit,
        legend: output.legend,
        features: output.features.map((feature, index) => ({ id: String(index), ...feature })),
      });
  };
  const frame = (features: MapOutput["features"]) => {
    const bounds = researchMapBounds(features);
    if (bounds) scene?.camera.flyToRectangle(bounds.west, bounds.south, bounds.east, bounds.north);
  };
  const matches = output.features
    .map((feature, index) => ({ feature, index }))
    .filter(({ feature }) =>
      feature.label.toLocaleLowerCase().includes(search.toLocaleLowerCase()),
    );
  const pageCount = Math.max(1, Math.ceil(matches.length / 50)),
    shownPage = Math.min(page, pageCount - 1);
  return (
    <div className="land-map-explorer">
      <p>
        {output.features.length.toLocaleString()} mapped features. {output.legend}
      </p>
      <div className="land-actions">
        <button
          type="button"
          aria-pressed={Boolean(layer)}
          disabled={!scene || !output.features.length}
          onClick={() => {
            if (layer) useLandContext.getState().removeLayer(artifactId);
            else {
              show();
              frame(output.features);
            }
          }}
        >
          {layer ? "Hide map layer" : "Show map layer"}
        </button>
        <button
          type="button"
          disabled={!scene || !output.features.length}
          onClick={() => {
            show();
            frame(output.features);
          }}
        >
          Fit all features
        </button>
      </div>
      <details open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
        <summary>Explore mapped features</summary>
        <p>Select a feature here or on the map. Selected features appear in gold.</p>
        <label className="land-name">
          Filter {title}
          <input
            type="search"
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(0);
            }}
          />
        </label>
        <ul className="land-map-feature-list" aria-label={`${title} features`}>
          {matches.slice(shownPage * 50, (shownPage + 1) * 50).map(({ feature, index }) => (
            <li key={index} data-selected={selected?.featureId === String(index)}>
              <button
                type="button"
                aria-pressed={selected?.featureId === String(index)}
                disabled={!scene}
                onClick={() => {
                  show();
                  useLandContext.getState().selectMapFeature(artifactId, String(index));
                }}
              >
                <strong>{feature.label}</strong>{" "}
                <span>
                  {feature.geometry.type} · {mapValue(feature.value, output.unit)}
                </span>
              </button>
              <button
                type="button"
                disabled={!scene}
                aria-label={`Focus on ${feature.label}`}
                onClick={() => {
                  show();
                  useLandContext.getState().selectMapFeature(artifactId, String(index));
                  frame([feature]);
                }}
              >
                Focus
              </button>
            </li>
          ))}
        </ul>
        {!matches.length && <p>No matching features.</p>}
        <div className="land-actions">
          <button type="button" disabled={shownPage === 0} onClick={() => setPage(shownPage - 1)}>
            Previous features
          </button>
          <span>
            {shownPage + 1} of {pageCount} · {matches.length.toLocaleString()} matches
          </span>
          <button
            type="button"
            disabled={shownPage + 1 >= pageCount}
            onClick={() => setPage(shownPage + 1)}
          >
            Next features
          </button>
        </div>
      </details>
    </div>
  );
}

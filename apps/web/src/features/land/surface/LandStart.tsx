import { ChevronRight, LandPlot, MapPin, Pencil, Route, Upload, X } from "lucide-react";
import type { RefObject } from "react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { useLand } from "@/state/land";
import { useUi } from "@/state/ui";

import { useLandAreas } from "../api";
import { areaLabel, beginLand, openLand } from "./landFlow";

/**
 * No land yet: one instruction and the two common ways to outline it. Everything else (a
 * mapped area, a corridor, a file) waits behind "More ways", and saved land is listed below.
 */
export function LandStart({ fileInput }: { fileInput: RefObject<HTMLInputElement | null> }) {
  const scene = useScene();
  const catalog = useLandAreas(true);
  const active = useLand((s) => s.active);
  return (
    <GlassPanel strong className="land-card" data-testid="land-start">
      <div className="land-eyebrow-row">
        <span className="land-lens-label">Your land</span>
        <GlassButton
          iconOnly
          size="sm"
          variant="ghost"
          aria-label="Close land"
          onClick={() => useUi.getState().setPanel(null)}
        >
          <X size={16} aria-hidden="true" />
        </GlassButton>
      </div>
      <h2 className="land-headline">Outline the place you want to understand.</h2>
      <div className="land-row">
        <GlassButton
          variant="primary"
          disabled={!scene}
          leadingIcon={<Pencil size={16} aria-hidden="true" />}
          onClick={() => beginLand(scene, "draw")}
        >
          Draw on the map
        </GlassButton>
        <GlassButton
          disabled={!scene}
          leadingIcon={<LandPlot size={16} aria-hidden="true" />}
          onClick={() => beginLand(scene, "candidates")}
        >
          Pick parcels
        </GlassButton>
      </div>
      <details className="land-details">
        <summary>More ways</summary>
        <div className="land-details__body land-start__more">
          <button type="button" disabled={!scene} onClick={() => beginLand(scene, "pick")}>
            <MapPin size={17} aria-hidden="true" />
            <span>
              <strong>Pick a mapped area</strong>
              <small>A field, woodland, park or lake</small>
            </span>
            <ChevronRight size={15} aria-hidden="true" />
          </button>
          <button type="button" disabled={!scene} onClick={() => beginLand(scene, "corridor")}>
            <Route size={17} aria-hidden="true" />
            <span>
              <strong>Trace a corridor</strong>
              <small>Follow a road, river or line at a set width</small>
            </span>
            <ChevronRight size={15} aria-hidden="true" />
          </button>
          <button type="button" onClick={() => fileInput.current?.click()}>
            <Upload size={17} aria-hidden="true" />
            <span>
              <strong>Import a file</strong>
              <small>GeoJSON, KML, shapefile or GeoPackage</small>
            </span>
            <ChevronRight size={15} aria-hidden="true" />
          </button>
        </div>
      </details>
      {catalog.isError && (
        <p className="land-note">
          Saved land could not be loaded.{" "}
          <button type="button" className="land-link" onClick={() => void catalog.refetch()}>
            Try again
          </button>
        </p>
      )}
      {(catalog.data?.length ?? 0) > 0 && (
        <section className="land-saved" aria-label="Saved land">
          <h3 className="land-muted">Saved land</h3>
          <ul>
            {catalog.data?.map((land) => (
              <li key={land.id}>
                <button
                  type="button"
                  className={active?.id === land.id ? "is-selected" : ""}
                  onClick={() => openLand(scene, land)}
                >
                  <span>
                    <strong>{land.name}</strong>
                    <small>{areaLabel(land.areaM2)}</small>
                  </span>
                  <ChevronRight size={15} aria-hidden="true" />
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </GlassPanel>
  );
}

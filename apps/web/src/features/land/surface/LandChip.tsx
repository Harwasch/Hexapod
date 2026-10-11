import { useState } from "react";
import {
  Bookmark,
  ChevronLeft,
  Download,
  Focus,
  History,
  LandPlot,
  MoreHorizontal,
  Pencil,
  Plus,
  Repeat,
} from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton, GlassPanel } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { useLand } from "@/state/land";

import { useBoundaryHistory, useLandAreas } from "../api";
import { LandSavedViews } from "../LandSavedViews";
import { areaLabel, beginLand, exportBoundary, frameBoundary, openLand } from "./landFlow";
import { sourceLabel, sourceMeaning } from "./sourceLabels";

type Sheet = "history" | "views" | "switch" | null;

/**
 * The saved land, as one line: its name, size and where the outline came from. Everything
 * done to the land itself (edit, history, export, saved views, switching land) is in its menu.
 */
export function LandChip({ land }: { land: LandArea }) {
  const scene = useScene();
  const [menu, setMenu] = useState(false);
  const [sheet, setSheet] = useState<Sheet>(null);
  const open = (next: Sheet) => {
    setSheet(next);
    setMenu(false);
  };
  return (
    <>
      <GlassPanel strong className="land-chip" data-testid="land-chip">
        <span className="land-chip__swatch" aria-hidden="true">
          <LandPlot size={18} />
        </span>
        <span className="land-chip__text">
          <h2 className="land-chip__name">{land.name}</h2>
          <span className="land-chip__meta" title={sourceMeaning(land.source.meaning)}>
            {areaLabel(land.areaM2)} · {sourceLabel(land.source)}
          </span>
        </span>
        <GlassButton
          iconOnly
          size="sm"
          variant="ghost"
          aria-label="Show the land on the map"
          onClick={() => frameBoundary(scene, land.boundary)}
        >
          <Focus size={16} aria-hidden="true" />
        </GlassButton>
        <GlassButton
          iconOnly
          size="sm"
          variant="ghost"
          aria-label="Land options"
          aria-expanded={menu}
          active={menu}
          onClick={() => setMenu(!menu)}
        >
          <MoreHorizontal size={16} aria-hidden="true" />
        </GlassButton>
      </GlassPanel>
      {menu && (
        <GlassPanel strong className="land-card land-menu" role="menu" aria-label="Land options">
          <MenuItem
            icon={<Pencil size={16} />}
            label="Edit boundary"
            onSelect={() => {
              setMenu(false);
              useLand.getState().propose({
                name: land.name,
                description: land.description,
                boundary: land.boundary,
                source: land.source,
              });
              beginLand(scene, "edit");
            }}
          />
          <MenuItem
            icon={<History size={16} />}
            label="Boundary history"
            onSelect={() => open("history")}
          />
          <MenuItem
            icon={<Bookmark size={16} />}
            label="Saved map views"
            onSelect={() => open("views")}
          />
          <MenuItem
            icon={<Download size={16} />}
            label="Export boundary (GeoJSON)"
            onSelect={() => {
              setMenu(false);
              exportBoundary(land, land.revision);
            }}
          />
          <MenuItem
            icon={<Repeat size={16} />}
            label="Switch land"
            onSelect={() => open("switch")}
          />
          <MenuItem
            icon={<Plus size={16} />}
            label="Outline new land"
            onSelect={() => {
              setMenu(false);
              useLand.getState().clear();
            }}
          />
        </GlassPanel>
      )}
      {sheet && (
        <GlassPanel
          strong
          className="land-card land-card--scroll"
          data-testid={`land-sheet-${sheet}`}
        >
          <div className="land-eyebrow-row">
            <GlassButton
              size="sm"
              variant="ghost"
              leadingIcon={<ChevronLeft size={15} aria-hidden="true" />}
              onClick={() => setSheet(null)}
            >
              Back
            </GlassButton>
          </div>
          {sheet === "history" && <BoundaryHistory land={land} />}
          {sheet === "views" && <LandSavedViews land={land} />}
          {sheet === "switch" && <SwitchLand current={land} done={() => setSheet(null)} />}
        </GlassPanel>
      )}
    </>
  );
}

function MenuItem({
  icon,
  label,
  onSelect,
}: {
  icon: React.ReactNode;
  label: string;
  onSelect: () => void;
}) {
  return (
    <button type="button" role="menuitem" className="land-menu__item" onClick={onSelect}>
      <span aria-hidden="true">{icon}</span>
      {label}
    </button>
  );
}

function BoundaryHistory({ land }: { land: LandArea }) {
  const history = useBoundaryHistory(land.id);
  return (
    <section aria-label="Boundary history" className="land-history">
      <h3 className="land-headline">Boundary history</h3>
      {history.isPending && <p className="land-muted">Loading revisions…</p>}
      {history.isError && <p role="alert">History could not be loaded.</p>}
      <ul className="land-facts">
        {history.data?.map((revision) => (
          <li key={revision.revision} className="land-fact">
            <span className="land-fact__label">
              Revision {revision.revision} · {new Date(revision.createdAt).toLocaleDateString()}
            </span>
            <span className="land-fact__value">{revision.note}</span>
            <span className="land-fact__note">
              <button
                type="button"
                className="land-link"
                onClick={() =>
                  useLand.getState().propose({
                    name: land.name,
                    description: land.description,
                    boundary: revision.boundary,
                    source: revision.source,
                  })
                }
              >
                Use as new draft
              </button>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function SwitchLand({ current, done }: { current: LandArea; done: () => void }) {
  const scene = useScene();
  const catalog = useLandAreas(true);
  return (
    <section aria-label="Saved land" className="land-saved">
      <h3 className="land-headline">Saved land</h3>
      {catalog.isPending && <p className="land-muted">Loading saved land…</p>}
      <ul>
        {catalog.data?.map((land) => (
          <li key={land.id}>
            <button
              type="button"
              className={land.id === current.id ? "is-selected" : ""}
              onClick={() => {
                openLand(scene, land);
                done();
              }}
            >
              <span>
                <strong>{land.name}</strong>
                <small>{areaLabel(land.areaM2)}</small>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

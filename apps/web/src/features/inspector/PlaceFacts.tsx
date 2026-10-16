import { ExternalLink, LandPlot, RotateCw } from "lucide-react";

import { footprintContains } from "@twin/geo";
import { GlassButton } from "@twin/ui";

import { env } from "@/app/env";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLand } from "@/state/land";
import { useLandAccessReady, useLandCanEdit } from "@/state/landIdentity";
import type { Selection } from "@/state/selection";

import { useLandAreas } from "../land/api";
import { openLand, outlineLandAt } from "../land/landEntry";
import { usePlaceFact, type PlaceFact, type PlaceFactKind } from "../land/researchApi";

/** The rows, in order: what the row is called, and who answers it while it loads. */
const FACTS: readonly (readonly [PlaceFactKind, string, string])[] = [
  ["elevation", "Elevation", "USGS"],
  ["soil", "Soil", "USDA"],
  ["land-cover", "Land cover", "ESA"],
  ["flood-zone", "Flood zone", "FEMA"],
  ["ecoregion", "Ecoregion", "EPA"],
];

/** How each answer was made, so a model or a map is never read as a measurement here. */
const BASIS: Record<PlaceFact["basis"], string> = {
  "modelled-surface": "Elevation model",
  "mapped-survey": "Mapped survey",
  "classified-imagery": "Satellite classification",
  "regulatory-map": "Regulatory map",
  "regional-framework": "Regional framework",
};

interface Point {
  longitude: number;
  latitude: number;
}

/**
 * "At this point": what public data says about the inspected spot (elevation, soil, land
 * cover, flood zone, ecoregion), each row cited, labelled with how it was made, and with its
 * limits a click away. Looked up live by the API (GET /research/place/{fact}) and not saved:
 * the heading says so. Rows fill in as each agency answers; a failed one says why and offers
 * Retry, and is never shown as "nothing here".
 *
 * Below the rows, the way into the Land tool from here: research the saved land the point is
 * in, or outline the land around it (features/land/landEntry.ts).
 *
 * Only with land exploration on and its workspace ready, since the lookups sit behind it.
 */
export function PlaceFacts({ selection }: { selection: Selection }) {
  const ready = useLandAccessReady();
  if (!env.landExplorationEnabled || !ready) return null;
  return <PlaceSection point={{ longitude: selection.longitude, latitude: selection.latitude }} />;
}

function PlaceSection({ point }: { point: Point }) {
  return (
    <section aria-label="At this point" data-testid="inspector-place">
      <p className="glass-eyebrow">At this point · looked up live, not saved</p>
      <dl className="dl" style={{ marginTop: "0.3rem", fontSize: "var(--text-xs)" }}>
        {FACTS.map(([kind, label, short]) => (
          <FactRow key={kind} kind={kind} label={label} short={short} point={point} />
        ))}
      </dl>
      <LandEntry point={point} />
    </section>
  );
}

function FactRow({
  kind,
  label,
  short,
  point,
}: {
  kind: PlaceFactKind;
  label: string;
  short: string;
  point: Point;
}) {
  const fact = usePlaceFact(kind, point, true);
  const retry = (
    <GlassButton
      size="sm"
      variant="ghost"
      leadingIcon={<RotateCw size={11} aria-hidden="true" />}
      aria-label={`Retry ${label.toLowerCase()}`}
      onClick={() => void fact.refetch()}
      loading={fact.isFetching}
    >
      Retry
    </GlassButton>
  );
  let body;
  if (fact.isPending) {
    body = <span className="glass-subtle">checking {short}…</span>;
  } else if (fact.isError) {
    body = (
      <>
        <span>Could not look this up. {describeError(fact.error)}</span> {retry}
      </>
    );
  } else if (fact.data.status === "unavailable") {
    body = (
      <>
        <span>{fact.data.label}.</span> {retry}
      </>
    );
  } else if (fact.data.status === "uncovered") {
    body = <span className="glass-subtle">{fact.data.label}</span>;
  } else {
    const { data } = fact;
    body = (
      <>
        <strong>{data.label}</strong>
        {data.detail && <span> · {data.detail}</span>}
        <br />
        <span className="glass-subtle">
          {BASIS[data.basis]}
          {data.vintage && ` · ${data.vintage}`}
          {data.source && (
            <>
              {" · "}
              <a href={data.source.url} target="_blank" rel="noreferrer">
                {data.source.name} <ExternalLink size={10} aria-hidden="true" />
              </a>
            </>
          )}
        </span>
        {data.status === "empty" ? (
          // "Nothing mapped here" is the answer most easily over-read, so its limit is not
          // folded away: no mapped flood zone is not low flood risk.
          <>
            <br />
            <span className="glass-subtle">{data.caveat}</span>
          </>
        ) : (
          <details className="disclosure">
            <summary className="disclosure__summary">Limits</summary>
            <span className="glass-subtle">{data.caveat}</span>
            {data.source && (
              <span className="glass-subtle">
                {" "}
                {data.source.attribution} · License: {data.source.license}
              </span>
            )}
          </details>
        )}
      </>
    );
  }
  return (
    <div style={{ display: "contents" }}>
      <dt>{label}</dt>
      <dd data-testid={`place-${kind}`} aria-busy={fact.isFetching || undefined}>
        {body}
      </dd>
    </div>
  );
}

/** Research the saved land this point is in, or start outlining the land around it. */
function LandEntry({ point }: { point: Point }) {
  const scene = useScene();
  const canEdit = useLandCanEdit();
  const hasDraft = useLand((s) => s.draft !== null);
  const lands = useLandAreas(true).data;
  const inside = lands?.find((land) => footprintContains(land.boundary, point));
  if (hasDraft) {
    return (
      <div className="glass-stack" style={{ marginTop: "0.5rem" }}>
        <GlassButton
          size="sm"
          leadingIcon={<LandPlot size={13} aria-hidden="true" />}
          onClick={() => outlineLandAt(point, scene)}
        >
          Return to your unsaved outline
        </GlassButton>
        <p className="glass-subtle" style={{ fontSize: "var(--text-xs)" }}>
          Save or discard it before starting another.
        </p>
      </div>
    );
  }
  if (inside) {
    return (
      <div className="glass-stack" style={{ marginTop: "0.5rem" }}>
        <p style={{ fontSize: "var(--text-xs)" }}>
          Inside <strong>{inside.name}</strong>
        </p>
        <GlassButton
          size="sm"
          leadingIcon={<LandPlot size={13} aria-hidden="true" />}
          onClick={() => openLand(inside)}
        >
          Research {inside.name}
        </GlassButton>
      </div>
    );
  }
  if (!canEdit) {
    return (
      <p className="glass-subtle" style={{ marginTop: "0.5rem", fontSize: "var(--text-xs)" }}>
        Your workspace role can view land but not add it.
      </p>
    );
  }
  return (
    <div className="glass-stack" style={{ marginTop: "0.5rem" }}>
      <GlassButton
        size="sm"
        leadingIcon={<LandPlot size={13} aria-hidden="true" />}
        onClick={() => outlineLandAt(point, scene)}
      >
        Outline this land
      </GlassButton>
      <p className="glass-subtle" style={{ fontSize: "var(--text-xs)" }}>
        Finds mapped parcels, fields and lines at this point. Saving an outline starts its open-data
        research.
      </p>
    </div>
  );
}

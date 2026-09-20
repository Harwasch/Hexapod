import { Crosshair, ExternalLink } from "lucide-react";

import { formatLatLon, formatLength } from "@twin/geo";
import { GlassBadge, GlassButton, GlassTooltip } from "@twin/ui";

import { useLayers as useLayerCatalog, useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { formatDate } from "@/lib/format";
import { geometryProvenance } from "@/lib/provenance";
import { useLiving } from "@/state/living";
import { useSelection } from "@/state/selection";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";

import { FloatingPanel } from "../shell/FloatingPanel";

const KIND_LABEL = {
  ground: "Location",
  site: "Site",
  "layer-feature": "Feature",
  "tile-feature": "3D feature",
} as const;

/**
 * What the Living Survey is doing to the selected site, in one sentence.
 *
 * Only ever describes the motion. Whether the geometry under it was measured is a separate
 * question with a separate answer (`geometryProvenance`), and conflating the two is the mistake
 * this whole section exists to prevent.
 */
function motionState(ready: boolean, animating: boolean): string {
  if (!ready) return "The motion rig is still attaching, so nothing is moving yet.";
  if (!animating) return "Wind is off, so it stands exactly as it was loaded.";
  return "Modelled wind is moving it now. The loaded geometry is never written to — at calm it returns to exactly the positions above.";
}

/** Right-side context inspector; hidden until something is selected. */
export function InspectorPanel() {
  const scene = useScene();
  const selection = useSelection((s) => s.selection);
  const open = useUi((s) => s.inspectorOpen) && selection !== null;
  const setInspectorOpen = useUi((s) => s.setInspectorOpen);
  const setAboutLayerId = useUi((s) => s.setAboutLayerId);
  const units = useSettings((s) => s.units);
  const living = useLiving((s) => s.status);
  const layers = useLayerCatalog();
  const activeSiteId = useSites((s) => s.activeSiteId);
  const site = useSite(selection?.siteId ?? activeSiteId).data;
  const layer = selection?.layerId
    ? layers.data?.find((l) => l.id === selection.layerId)
    : undefined;
  // A site is "living" when the scene has a motion rig attached to one of its assets. The
  // provenance below is quoted for *that* asset, so the resolution shown belongs to the
  // representation actually being deformed.
  //
  // It falls back to the active site — the same fallback the panel already makes for `site` —
  // for a reason particular to this feature: Gaussian splats are invisible to picking, so a
  // click on a swaying tree lands on the terrain behind it and produces a `ground` selection
  // with no `siteId` at all. Without the fallback the one panel that can state the
  // Observed/Simulated split would be unreachable for exactly the sites that need it. The
  // section names the site it is talking about so it can never be read as describing the
  // clicked point.
  const livingSiteId = selection?.siteId ?? activeSiteId;
  const livingSite = livingSiteId
    ? living.sites.find((entry) => entry.siteId === livingSiteId)
    : undefined;
  const geometry = livingSite ? geometryProvenance(site, livingSite.assetId, units) : null;
  const attribution = selection?.attribution?.length
    ? selection.attribution
    : (layer?.attribution ?? (selection?.kind === "site" ? site?.attribution : undefined));

  const close = () => {
    setInspectorOpen(false);
    scene?.selection.clear();
  };

  return (
    <FloatingPanel
      open={open}
      from="right"
      title={selection?.title ?? "Inspector"}
      onClose={close}
      testId="inspector"
      actions={
        selection && (
          <GlassTooltip content="Fly to selection">
            <GlassButton
              iconOnly
              size="sm"
              variant="ghost"
              aria-label="Fly to selection"
              onClick={() =>
                scene?.camera.flyTo(
                  selection.longitude,
                  selection.latitude,
                  (selection.height ?? 0) + 80,
                  { pitch: -40, durationS: 1.4 },
                )
              }
            >
              <Crosshair size={14} aria-hidden="true" />
            </GlassButton>
          </GlassTooltip>
        )
      }
    >
      {selection && (
        <div className="glass-stack">
          <div className="glass-row" style={{ flexWrap: "wrap" }}>
            <GlassBadge tone="accent">{KIND_LABEL[selection.kind]}</GlassBadge>
            {selection.sourceLabel && <GlassBadge>{selection.sourceLabel}</GlassBadge>}
            {livingSite && living.animating && (
              <GlassBadge tone="warning">Simulated motion</GlassBadge>
            )}
          </div>
          <dl className="dl">
            <dt>Position</dt>
            <dd data-testid="inspector-position">
              {formatLatLon({ longitude: selection.longitude, latitude: selection.latitude })}
            </dd>
            <dt>Height</dt>
            <dd>{selection.height !== null ? formatLength(selection.height, units) : "—"}</dd>
            <dt>Terrain</dt>
            <dd>
              {selection.terrainHeight !== null
                ? formatLength(selection.terrainHeight, units)
                : "sampling…"}
            </dd>
            {selection.observedAt !== undefined && (
              <>
                <dt>Observed</dt>
                <dd>{selection.observedAt ? formatDate(selection.observedAt) : "Unknown"}</dd>
              </>
            )}
            {layer && (
              <>
                <dt>Layer</dt>
                <dd>
                  <button
                    type="button"
                    className="glass-button glass-button--ghost glass-button--sm"
                    onClick={() => setAboutLayerId(layer.id)}
                  >
                    About {layer.name}
                  </button>
                </dd>
              </>
            )}
            {selection.kind === "site" && site && (
              <>
                <dt>License</dt>
                <dd>{site.license?.name ?? "Not specified"}</dd>
              </>
            )}
          </dl>
          {livingSite && geometry && (
            <section
              aria-label={`Measured and simulated${site ? `: ${site.name}` : ""}`}
              data-testid="inspector-living"
            >
              <p className="glass-eyebrow">Measured and simulated{site ? ` · ${site.name}` : ""}</p>
              <dl className="dl" style={{ marginTop: "0.3rem", fontSize: "var(--text-xs)" }}>
                <dt>Geometry</dt>
                <dd data-testid="inspector-geometry">
                  {geometry.summary}
                  {geometry.note && (
                    <>
                      <br />
                      <span className="glass-subtle">{geometry.note}</span>
                    </>
                  )}
                </dd>
                <dt>Motion</dt>
                <dd data-testid="inspector-motion">
                  Simulated · {livingSite.rigSourceNote}
                  <br />
                  <span className="glass-subtle">
                    {motionState(livingSite.phase === "ready", living.animating)}
                  </span>
                </dd>
              </dl>
            </section>
          )}
          {attribution && attribution.length > 0 && (
            <section aria-label="Attribution">
              <p className="glass-eyebrow">Attribution</p>
              <ul
                className="glass-list"
                style={{ gap: "0.2rem", marginTop: "0.3rem", fontSize: "var(--text-xs)" }}
              >
                {attribution.map((a, i) => (
                  <li key={i}>
                    {a.url ? (
                      <a href={a.url} target="_blank" rel="noreferrer">
                        {a.text} <ExternalLink size={10} aria-hidden="true" />
                      </a>
                    ) : (
                      a.text
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}
          {selection.properties && selection.properties.length > 0 && (
            <section aria-label="Properties">
              <p className="glass-eyebrow">Properties</p>
              <dl className="dl" style={{ marginTop: "0.3rem", fontSize: "var(--text-xs)" }}>
                {selection.properties.map((p) => (
                  <div key={p.key} style={{ display: "contents" }}>
                    <dt>{p.key}</dt>
                    <dd>{p.value}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}
        </div>
      )}
    </FloatingPanel>
  );
}

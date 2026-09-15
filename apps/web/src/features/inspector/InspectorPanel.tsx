import { Crosshair, ExternalLink } from "lucide-react";

import { formatLatLon, formatLength } from "@twin/geo";
import { GlassBadge, GlassButton, GlassTooltip } from "@twin/ui";

import { useLayers as useLayerCatalog, useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { formatDate } from "@/lib/format";
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

/** Right-side context inspector; hidden until something is selected. */
export function InspectorPanel() {
  const scene = useScene();
  const selection = useSelection((s) => s.selection);
  const open = useUi((s) => s.inspectorOpen) && selection !== null;
  const setInspectorOpen = useUi((s) => s.setInspectorOpen);
  const setAboutLayerId = useUi((s) => s.setAboutLayerId);
  const units = useSettings((s) => s.units);
  const layers = useLayerCatalog();
  const activeSiteId = useSites((s) => s.activeSiteId);
  const site = useSite(selection?.siteId ?? activeSiteId).data;
  const layer = selection?.layerId
    ? layers.data?.find((l) => l.id === selection.layerId)
    : undefined;
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

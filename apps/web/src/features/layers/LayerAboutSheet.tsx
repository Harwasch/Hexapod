import { ExternalLink } from "lucide-react";

import { GlassBadge, GlassSheet } from "@twin/ui";

import { useLayers as useLayerCatalog } from "@/api/queries";
import { categoryLabel, formatDate, sourceLabel } from "@/lib/format";
import { useUi } from "@/state/ui";

/** "About this layer": provenance, license, attribution, coverage and legend. */
export function LayerAboutSheet() {
  const aboutLayerId = useUi((s) => s.aboutLayerId);
  const setAboutLayerId = useUi((s) => s.setAboutLayerId);
  const catalog = useLayerCatalog();
  const layer = catalog.data?.find((l) => l.id === aboutLayerId) ?? null;
  return (
    <GlassSheet
      open={Boolean(layer)}
      onOpenChange={(open) => !open && setAboutLayerId(null)}
      title={layer?.name ?? "Layer"}
      description={layer?.description ?? undefined}
      testId="layer-about"
    >
      {layer && (
        <div className="glass-stack">
          <div className="glass-row" style={{ flexWrap: "wrap" }}>
            <GlassBadge tone="accent">{categoryLabel(layer.category)}</GlassBadge>
            {layer.builtin && <GlassBadge>Curated</GlassBadge>}
            {layer.license && (
              <GlassBadge tone={layer.license.requiresAttribution ? "warning" : "success"}>
                {layer.license.spdxId ?? layer.license.name}
              </GlassBadge>
            )}
          </div>
          <dl className="dl">
            <dt>Source</dt>
            <dd>{sourceLabel(layer.source)}</dd>
            {layer.provenance?.sourceOrganization && (
              <>
                <dt>Organization</dt>
                <dd>{layer.provenance.sourceOrganization}</dd>
              </>
            )}
            {layer.provenance?.sourceUrl && (
              <>
                <dt>Source URL</dt>
                <dd>
                  <a href={layer.provenance.sourceUrl} target="_blank" rel="noreferrer">
                    {layer.provenance.sourceUrl} <ExternalLink size={12} aria-hidden="true" />
                  </a>
                </dd>
              </>
            )}
            {layer.provenance?.publishedAt && (
              <>
                <dt>Published</dt>
                <dd>{formatDate(layer.provenance.publishedAt, "long")}</dd>
              </>
            )}
            <dt>Observed</dt>
            <dd>
              {layer.temporalExtent?.start || layer.temporalExtent?.end
                ? `${formatDate(layer.temporalExtent.start)} – ${formatDate(layer.temporalExtent.end)}`
                : formatDate(layer.observedAt)}
            </dd>
            <dt>Resolution</dt>
            <dd>{layer.resolution ?? "—"}</dd>
            <dt>Coverage</dt>
            <dd>
              {layer.coverage ?? "—"}
              {layer.spatialExtent && (
                <span className="glass-subtle">
                  {" "}
                  · {layer.spatialExtent.west.toFixed(1)}, {layer.spatialExtent.south.toFixed(1)} →{" "}
                  {layer.spatialExtent.east.toFixed(1)}, {layer.spatialExtent.north.toFixed(1)}
                </span>
              )}
            </dd>
            <dt>License</dt>
            <dd>
              {layer.license ? (
                <>
                  {layer.license.url ? (
                    <a href={layer.license.url} target="_blank" rel="noreferrer">
                      {layer.license.name}
                    </a>
                  ) : (
                    layer.license.name
                  )}
                  {layer.license.notes && <div className="glass-subtle">{layer.license.notes}</div>}
                </>
              ) : (
                "Not specified"
              )}
            </dd>
            <dt>Attribution</dt>
            <dd>
              {layer.attribution.length === 0 && "—"}
              {layer.attribution.map((a, i) => (
                <div key={i}>
                  {a.url ? (
                    <a href={a.url} target="_blank" rel="noreferrer">
                      {a.text}
                    </a>
                  ) : (
                    a.text
                  )}
                  {a.organization && <span className="glass-subtle"> — {a.organization}</span>}
                </div>
              ))}
            </dd>
          </dl>
          {layer.legend?.entries && layer.legend.entries.length > 0 && (
            <section aria-label="Legend">
              <p className="glass-eyebrow">{layer.legend.title ?? "Legend"}</p>
              <ul
                className="glass-list"
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "0.3rem 0.75rem",
                  marginTop: "0.4rem",
                }}
              >
                {layer.legend.entries.map((entry) => (
                  <li
                    key={entry.label}
                    className="glass-row"
                    style={{ fontSize: "var(--text-xs)" }}
                  >
                    <span
                      aria-hidden="true"
                      style={{
                        width: 12,
                        height: 12,
                        borderRadius: 3,
                        background: entry.color,
                        flex: "0 0 auto",
                      }}
                    />
                    {entry.label}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}
    </GlassSheet>
  );
}

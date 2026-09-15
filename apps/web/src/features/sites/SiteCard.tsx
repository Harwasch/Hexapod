import { Calendar, Mountain } from "lucide-react";

import type { SiteSummary } from "@twin/contracts";
import { formatArea } from "@twin/geo";
import { GlassBadge } from "@twin/ui";

import { formatDate, representationLabel } from "@/lib/format";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";

export function SiteCard({ site, onSelect }: { site: SiteSummary; onSelect: () => void }) {
  const units = useSettings((s) => s.units);
  const active = useSites((s) => s.activeSiteId === site.id);
  return (
    <li>
      <button
        type="button"
        className={`card card--interactive ${active ? "card--selected" : ""}`}
        style={{ width: "100%", textAlign: "left", font: "inherit", color: "inherit" }}
        onClick={onSelect}
        aria-pressed={active}
        data-testid={`site-card-${site.slug}`}
      >
        {site.thumbnailUrl ? (
          <img className="card__thumb" src={site.thumbnailUrl} alt="" />
        ) : (
          <div className="card__thumb card__thumb--placeholder" aria-hidden="true">
            <Mountain size={28} />
          </div>
        )}
        <div className="card__row">
          <h3 className="card__title">{site.name}</h3>
          {active && <GlassBadge tone="accent">Loaded</GlassBadge>}
        </div>
        <div className="card__row" style={{ gap: "0.3rem", flexWrap: "wrap" }}>
          {site.representations.map((rep) => (
            <GlassBadge key={rep}>{representationLabel(rep)}</GlassBadge>
          ))}
        </div>
        <div className="card__meta">
          <span>{formatArea(site.areaM2, units)}</span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <Calendar size={12} aria-hidden="true" />
            {site.latestObservedAt ? formatDate(site.latestObservedAt) : "Capture date unknown"}
          </span>
          {site.quality?.resolutionDescription && <span>{site.quality.resolutionDescription}</span>}
        </div>
      </button>
    </li>
  );
}

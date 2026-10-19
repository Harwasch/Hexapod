import { Image, Map as MapIcon } from "lucide-react";

import type { LandFinding } from "@twin/contracts";

import type { ArchiveItem } from "./recordsModel";

function Rows({
  items,
  selectedId,
  onOpen,
  label,
}: {
  items: ArchiveItem[];
  selectedId?: string | null;
  onOpen: (id: string) => void;
  label: string;
}) {
  if (!items.length) return null;
  return (
    <ul className="records-list" aria-label={label}>
      {items.map((item) => (
        <li key={item.id}>
          <button
            type="button"
            className="records-row"
            aria-pressed={selectedId === item.id}
            onClick={() => onOpen(item.id)}
          >
            {item.kind === "historical-map" ? (
              <MapIcon size={15} aria-hidden="true" className="records-row__icon" />
            ) : (
              <Image size={15} aria-hidden="true" className="records-row__icon" />
            )}
            <span className="records-row__main">
              <strong>
                {item.year ?? "Undated"} · {item.title}
              </strong>
              <small>
                {item.kind === "photograph" ? (item.onLand ? "On the land · " : "Nearby · ") : ""}
                {item.media.license}
              </small>
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

/** Every old map and photo found, with its licence, and how the search was done. */
export function RecordsArchiveList({
  items,
  findings,
  selectedId,
  onOpen,
}: {
  items: ArchiveItem[];
  findings: LandFinding[];
  selectedId?: string | null;
  onOpen: (id: string) => void;
}) {
  const maps = items.filter((item) => item.kind === "historical-map");
  const photos = items.filter((item) => item.kind === "photograph");
  return (
    <section className="records-section" aria-label="Old maps and photos">
      <h5 className="records-subhead">
        Old maps ({maps.length}) and photos ({photos.length})
      </h5>
      <Rows items={maps} selectedId={selectedId} onOpen={onOpen} label="Old maps" />
      <Rows items={photos} selectedId={selectedId} onOpen={onOpen} label="Old photos" />
      <details className="records-more">
        <summary>How these were found</summary>
        {findings.map((finding) => (
          <div key={finding.id}>
            <p>{finding.summary}</p>
            {finding.uncertainty && <p className="land-muted">{finding.uncertainty}</p>}
          </div>
        ))}
        <p className="land-muted">
          Photos come from Wikimedia Commons within 10 km, openly licensed only. Map sheets come
          from the USGS historical topographic collection, up to 12 overlapping sheets. A catalog
          point may mark the camera or the subject, and a sheet outline is its catalog extent, not a
          surveyed position. Neither search is complete.
        </p>
      </details>
    </section>
  );
}

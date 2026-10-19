import { useState } from "react";

import type { LandEvidence } from "@twin/contracts";

import { useEvidence } from "./surface/lenses/records/recordsApi";
import { openRecords } from "./surface/lenses/records/recordsStore";
import { RecordsArchiveItem } from "./surface/lenses/records/RecordsArchiveItem";
import "./surface/lenses/records/records.css";

function GalleryItem({
  id,
  cached,
  onAsk,
  index,
  total,
  onMove,
}: {
  id: string;
  cached?: LandEvidence;
  onAsk?: () => void;
  index: number;
  total: number;
  onMove: (index: number) => void;
}) {
  const { evidence, query } = useEvidence(id, cached);
  if (!evidence)
    return query.isError ? (
      <p className="land-alert" role="alert">
        This source could not be loaded.{" "}
        <button type="button" className="land-link" onClick={() => void query.refetch()}>
          Try again
        </button>
      </p>
    ) : (
      <p className="land-muted" role="status">
        Loading the source…
      </p>
    );
  return (
    <RecordsArchiveItem
      evidence={evidence}
      onMap={false}
      onAsk={onAsk}
      navigation={{
        index,
        total,
        onPrevious: () => onMove(index - 1),
        onNext: () => onMove(index + 1),
      }}
      onAlign={
        evidence.media?.kind === "historical-map"
          ? () => openRecords({ kind: "align", id })
          : undefined
      }
    />
  );
}

/**
 * A research gallery artifact (old photos and map sheets), one at a time. The Records lens
 * shows the same items on the map; lining up a map sheet opens it there.
 */
export function ArchiveGallery({
  ids,
  evidence = [],
  onAsk,
}: {
  ids: string[];
  evidence?: LandEvidence[];
  onAsk?: () => void;
}) {
  const [index, setIndex] = useState(0);
  const id = ids[index] ?? ids[0];
  if (!id) return null;
  return (
    <section className="records-section" aria-label="Archive gallery">
      <GalleryItem
        key={id}
        id={id}
        cached={evidence.find((item) => item.id === id)}
        onAsk={onAsk}
        index={Math.min(index, ids.length - 1)}
        total={ids.length}
        onMove={(next) => setIndex(Math.max(0, Math.min(ids.length - 1, next)))}
      />
    </section>
  );
}

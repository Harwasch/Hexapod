import { useState } from "react";
import { FilePlus2 } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { useLandAccessReady, useLandCanEdit } from "@/state/landIdentity";

import { parcelIds } from "./surface/lenses/records/recordsModel";
import { RecordsAddDocument } from "./surface/lenses/records/RecordsAddDocument";
import { RecordsDocuments } from "./surface/lenses/records/RecordsDocuments";
import { RecordsDocumentViewer } from "./surface/lenses/records/RecordsDocumentViewer";
import "./surface/lenses/records/records.css";

/**
 * The land's documents outside the Records lens (the older workspace panel): add, search,
 * browse and read them. The Records lens is the main home of these tools.
 */
export function LandDocuments({ land }: { land: LandArea }) {
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit() && ready;
  const [adding, setAdding] = useState(false);
  const [selected, setSelected] = useState<{ id: string; page: number; ocrId?: string } | null>(
    null,
  );
  const ids = parcelIds(land.source);
  return (
    <section className="records-section" aria-label="Land records">
      {canEdit &&
        (adding ? (
          <RecordsAddDocument
            landId={land.id}
            onSaved={(document) => {
              setAdding(false);
              setSelected({ id: document.id, page: 1 });
            }}
            onCancel={() => setAdding(false)}
          />
        ) : (
          <div className="land-row">
            <GlassButton
              size="sm"
              leadingIcon={<FilePlus2 size={15} />}
              onClick={() => setAdding(true)}
            >
              Add a document
            </GlassButton>
          </div>
        ))}
      <RecordsDocuments
        landId={land.id}
        parcelIds={ids}
        selectedId={selected?.id}
        onOpen={(id, page, ocrId) => setSelected({ id, page, ocrId })}
      />
      {selected && (
        <RecordsDocumentViewer
          key={`${selected.id}:${selected.page}:${selected.ocrId ?? ""}`}
          landId={land.id}
          documentId={selected.id}
          initialPage={selected.page}
          pinnedOcrId={selected.ocrId}
          parcelIds={ids}
        />
      )}
    </section>
  );
}

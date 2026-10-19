import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Link2, MessageCircleQuestion } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { askAgent } from "../../agentBus";
import { useLandDocument } from "./recordsApi";
import { parcelIds } from "./recordsModel";
import { useRecords, type RecordsView } from "./recordsStore";
import { RecordsConnections } from "./RecordsConnections";
import { RecordsDocumentViewer } from "./RecordsDocumentViewer";
import { RecordsLinkForm } from "./RecordsLinkForm";

type DocumentView = Extract<RecordsView, { kind: "document" }>;

/** A document opened in the Records card: its pages, a question about a page, connections. */
export function RecordsDocumentView({ land, view }: { land: LandArea; view: DocumentView }) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit() && ready;
  const cache = useQueryClient();
  const record = useLandDocument(land.id, view.id);
  const [relating, setRelating] = useState(false);
  const open = (id: string, page: number) =>
    useRecords.getState().open({ kind: "document", id, page, jump: view.jump + 1 });
  const ask = () => {
    const title = record.data?.title ?? "this document";
    const reading = view.ocrId ? `, OCR reading ${view.ocrId} (an unverified machine reading)` : "";
    askAgent(
      `Read "${title}", page ${view.page} (document ${view.id}${reading}). What does it say about this land? Separate historical statements from rights or conditions that apply today, cite the pages, and flag missing evidence.`,
    );
  };
  return (
    <>
      <RecordsDocumentViewer
        key={`${view.id}:${view.jump}`}
        landId={land.id}
        documentId={view.id}
        initialPage={view.page}
        pinnedOcrId={view.ocrId}
        pinnedOcrPage={view.page}
        parcelIds={parcelIds(land.source)}
        onPageChange={(page) => {
          setRelating(false);
          useRecords.getState().open({ ...view, page, ocrId: undefined });
        }}
        onOcrSelected={(ocrId) => useRecords.getState().open({ ...view, ocrId })}
        onUploaded={() => cache.invalidateQueries({ queryKey: ["land-documents", scope, land.id] })}
      />
      {canEdit && record.data && record.data.pageCount > 0 && (
        <div className="land-row">
          <GlassButton size="sm" leadingIcon={<MessageCircleQuestion size={14} />} onClick={ask}>
            Ask about this page
          </GlassButton>
          <GlassButton
            size="sm"
            variant="ghost"
            leadingIcon={<Link2 size={14} />}
            aria-expanded={relating}
            onClick={() => setRelating(!relating)}
          >
            Relate this record to another
          </GlassButton>
        </div>
      )}
      {relating && (
        <RecordsLinkForm
          key={view.page}
          landId={land.id}
          documentId={view.id}
          page={view.page}
          onDone={() => setRelating(false)}
        />
      )}
      <RecordsConnections landId={land.id} documentId={view.id} onOpen={open} />
    </>
  );
}

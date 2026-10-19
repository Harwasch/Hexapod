import { useEffect, useMemo } from "react";
import { ArrowLeft, MessageCircleQuestion } from "lucide-react";

import type { components, LandArea, LandEvidence } from "@twin/contracts";

import type { LandPoint } from "@/state/land";

import { askAgent } from "../agentBus";
import { LensCard } from "../LensCard";
import { useLandAnalysis, useStartAnalysis } from "../useLandAnalysis";
import { useEvidence, useLandDocuments, useLatestAlignments } from "./records/recordsApi";
import {
  ARCHIVE_TITLE,
  archiveItems,
  recordsHeadline,
  type ArchiveItem,
} from "./records/recordsModel";
import { useRecords, type RecordsView } from "./records/recordsStore";
import { RecordsAddDocument } from "./records/RecordsAddDocument";
import { RecordsAlignment } from "./records/RecordsAlignment";
import { RecordsArchiveItem } from "./records/RecordsArchiveItem";
import { RecordsDocumentView } from "./records/RecordsDocumentView";
import { RecordsSummary } from "./records/RecordsSummary";
import { useRecordsMap } from "./records/useRecordsMap";
import "./records/records.css";

type Registration = components["schemas"]["ImageRegistrationRead"];

const SUMMARY: RecordsView = { kind: "summary" };

function Back({ label }: { label: string }) {
  return (
    <button type="button" className="records-back" onClick={() => useRecords.getState().back()}>
      <ArrowLeft size={14} aria-hidden="true" />
      {label}
    </button>
  );
}

/** Lays the newest saved alignment over the land, unless the person hid the old map. */
function useLatestOverlay(land: LandArea, items: ArchiveItem[]) {
  const maps = useMemo(() => items.filter((item) => item.kind === "historical-map"), [items]);
  const ids = useMemo(() => maps.map((item) => item.id), [maps]);
  const latest = useLatestAlignments(ids);
  const found = latest.reduce<{ row: Registration; item: ArchiveItem } | null>(
    (best, query, index) => {
      const row = query.data?.[0];
      const item = maps[index];
      return row && item && (!best || row.createdAt > best.row.createdAt) ? { row, item } : best;
    },
    null,
  );
  const rowId = found?.row.id;
  useEffect(() => {
    const state = useRecords.getState();
    if (!found || state.landId !== land.id || state.overlay || state.overlayHidden) return;
    const { row, item } = found;
    state.showOverlay({
      id: row.id,
      label: `${item.year ? `${item.year} · ` : ""}${item.title}`,
      bounds: row.result.bounds,
      attribution: item.evidence.attribution,
    });
    // Runs when a newer alignment appears, not on every render of the same one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rowId, land.id]);
}

function ArchiveView({
  id,
  items,
  align,
  anchor,
}: {
  id: string;
  items: ArchiveItem[];
  align: boolean;
  anchor?: LandPoint;
}) {
  const index = items.findIndex((item) => item.id === id);
  const cached: LandEvidence | undefined = items[index]?.evidence;
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
  if (align) {
    const year = items[index]?.year;
    return (
      <>
        <Back label="Map sheet" />
        <h4 className="records-item__title">Line up the {year ? `${year} ` : "old "}map</h4>
        <RecordsAlignment evidence={evidence} />
      </>
    );
  }
  const open = (next: number) => {
    const item = items[next];
    if (item) useRecords.getState().open({ kind: "archive", id: item.id });
  };
  return (
    <RecordsArchiveItem
      key={id}
      evidence={evidence}
      back={<Back label="Records" />}
      anchor={anchor}
      navigation={
        index >= 0
          ? {
              index,
              total: items.length,
              onPrevious: () => open(index - 1),
              onNext: () => open(index + 1),
            }
          : undefined
      }
      onAlign={() => useRecords.getState().open({ kind: "align", id })}
    />
  );
}

/**
 * Records: the paper trail of the land. Where the outline came from and what it means, the
 * documents people added (pages, machine reading, search, connections), and old photos and
 * map sheets found in open archives, placed on the map; a map sheet can be lined up with the
 * land and faded over today's map.
 */
export function RecordsLens({ land }: { land: LandArea }) {
  const storeLand = useRecords((s) => s.landId);
  const storeView = useRecords((s) => s.view);
  const view = storeLand === land.id ? storeView : SUMMARY;
  useEffect(() => useRecords.getState().forLand(land.id), [land.id]);
  useEffect(() => {
    useRecords.getState().restoreOverlay();
    return () => useRecords.getState().stowOverlay();
  }, []);

  const archive = useLandAnalysis(land, ARCHIVE_TITLE);
  const starter = useStartAnalysis(land);
  const documents = useLandDocuments(land.id, 0);
  const items = useMemo(() => archiveItems(archive.detail?.evidence), [archive.detail]);
  const selectedId = view.kind === "archive" || view.kind === "align" ? view.id : null;
  const anchors = useRecordsMap(land.boundary, items, selectedId);
  useLatestOverlay(land, items);

  const maps = items.filter((item) => item.kind === "historical-map").length;
  const headline =
    view.kind === "summary"
      ? recordsHeadline(land.source, documents.data?.length ?? null, maps, items.length - maps)
      : undefined;

  return (
    <LensCard
      lens="records"
      headline={headline}
      action={
        <button
          type="button"
          className="land-icon-button"
          aria-label="Ask about these records"
          title="Ask about these records"
          onClick={() =>
            askAgent(
              "What do this land's records, old maps and photos say about its history and any rights or conditions on it?",
            )
          }
        >
          <MessageCircleQuestion size={16} />
        </button>
      }
    >
      {view.kind === "summary" && (
        <RecordsSummary
          land={land}
          archive={archive}
          starter={starter}
          items={items}
          anchors={anchors}
          documents={documents}
        />
      )}
      {(view.kind === "archive" || view.kind === "align") && (
        <ArchiveView
          key={view.id}
          id={view.id}
          items={items}
          align={view.kind === "align"}
          anchor={anchors.get(view.id)}
        />
      )}
      {view.kind === "document" && (
        <>
          <Back label="Records" />
          <RecordsDocumentView land={land} view={view} />
        </>
      )}
      {view.kind === "add" && (
        <>
          <Back label="Records" />
          <h4 className="records-item__title">Add a document</h4>
          <RecordsAddDocument
            landId={land.id}
            onSaved={(document) =>
              useRecords.getState().open({ kind: "document", id: document.id, page: 1, jump: 0 })
            }
            onCancel={() => useRecords.getState().back()}
          />
        </>
      )}
    </LensCard>
  );
}

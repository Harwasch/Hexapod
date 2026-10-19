import { useState, type ReactNode } from "react";
import { useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { FilePlus2 } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { describeError } from "@/lib/log";
import type { LandPoint } from "@/state/land";
import { useLandScope } from "@/state/landIdentity";

import { cancelResearch } from "../../../researchApi";
import { LensDetails, LensEmpty } from "../../LensCard";
import { Fact, MapLink } from "../../mapLinks";
import { useFlyTo } from "../../mapHooks";
import { sourceMeaning } from "../../sourceLabels";
import type { useLandAnalysis, useStartAnalysis } from "../../useLandAnalysis";
import { DOCUMENT_PAGE } from "./recordsApi";
import {
  ARCHIVE_QUESTION,
  ARCHIVE_TITLE,
  documentYear,
  latestDocument,
  namesParcel,
  parcelIds,
  type ArchiveItem,
  type LandDocument,
} from "./recordsModel";
import { useRecords } from "./recordsStore";
import { RecordsArchiveList } from "./RecordsArchiveList";
import { RecordsBoundary } from "./RecordsBoundary";
import { RecordsDocuments } from "./RecordsDocuments";
import { RecordsThenNow } from "./RecordsThenNow";

type Analysis = ReturnType<typeof useLandAnalysis>;
type Starter = ReturnType<typeof useStartAnalysis>;

const openItem = (id: string) => useRecords.getState().open({ kind: "archive", id });
const openDocument = (id: string, page = 1, ocrId?: string) =>
  useRecords.getState().open({ kind: "document", id, page, ocrId, jump: 0 });

/** A fact about an old map or photo: hovering points at it, clicking opens it and flies there. */
function ItemFact({
  label,
  value,
  item,
  anchor,
}: {
  label: string;
  value: ReactNode;
  item: ArchiveItem;
  anchor?: LandPoint;
}) {
  const fly = useFlyTo();
  const target = item.geometry
    ? { key: `records-item-${item.id}`, geometry: item.geometry, anchor }
    : null;
  return (
    <MapLink as="li" target={target} fly={false} className="land-fact records-fact">
      <button
        type="button"
        className="records-fact__button"
        onClick={() => {
          openItem(item.id);
          if (item.geometry) fly(item.geometry);
        }}
      >
        <span className="land-fact__label">{label}</span>
        {target && (
          <span className="land-fact__go" aria-hidden="true">
            on map
          </span>
        )}
        <span className="land-fact__value">{value}</span>
      </button>
    </MapLink>
  );
}

function DocumentsValue({ query }: { query: UseQueryResult<LandDocument[]> }) {
  if (query.isPending) return <>Loading…</>;
  if (query.isError)
    return (
      <>
        Did not load.{" "}
        <button type="button" className="land-link" onClick={() => void query.refetch()}>
          Try again
        </button>
      </>
    );
  const documents = query.data;
  const latest = latestDocument(documents);
  if (!latest) return <>None yet</>;
  const year = documentYear(latest);
  const count = `${documents.length}${documents.length === DOCUMENT_PAGE ? "+" : ""}`;
  return (
    <>
      {count} · latest{" "}
      <button type="button" className="land-link" onClick={() => openDocument(latest.id)}>
        {latest.title}
      </button>
      {year ? ` (${year})` : ""}
    </>
  );
}

/** The archive search: not run, running, failed, empty, stale or partial, in one line each. */
function ArchiveState({
  archive,
  starter,
  count,
  onStart,
}: {
  archive: Analysis;
  starter: Starter;
  count: number;
  onStart: () => void;
}) {
  const scope = useLandScope();
  const cache = useQueryClient();
  const [stopping, setStopping] = useState(false);
  const [stopError, setStopError] = useState<string | null>(null);
  const canEdit = starter.canEdit;
  const stop = async () => {
    if (!archive.run) return;
    setStopping(true);
    setStopError(null);
    try {
      await cancelResearch(archive.run.id);
      await cache.invalidateQueries({ queryKey: ["land-research", scope] });
    } catch (cause) {
      setStopError(describeError(cause));
    } finally {
      setStopping(false);
    }
  };
  const again = canEdit && (
    <button type="button" className="land-link" disabled={starter.busy} onClick={onStart}>
      Search again
    </button>
  );
  if (archive.loading)
    return (
      <p className="land-muted" role="status">
        Loading old maps and photos…
      </p>
    );
  if (archive.error)
    return (
      <p className="land-alert" role="alert">
        Old maps and photos did not load.{" "}
        <button
          type="button"
          className="land-link"
          onClick={() => void cache.invalidateQueries({ queryKey: ["land-research", scope] })}
        >
          Try again
        </button>
      </p>
    );
  if (!archive.investigation)
    return (
      <LensEmpty
        action={
          canEdit && (
            <GlassButton variant="primary" loading={starter.busy} onClick={onStart}>
              Find old maps and photos
            </GlassButton>
          )
        }
      >
        {canEdit
          ? "Find openly licensed old photos and map sheets of this area, placed on the map."
          : "No one has looked for old maps and photos yet. An editor can start the search."}
      </LensEmpty>
    );
  if (archive.running)
    return (
      <div className="records-running" role="status" aria-live="polite">
        <span className="records-running__pulse" aria-hidden="true" />
        <span>Searching photo and map catalogs…</span>
        {canEdit && archive.run && (
          <button
            type="button"
            className="land-link"
            disabled={stopping}
            onClick={() => void stop()}
          >
            Stop
          </button>
        )}
        {stopError && <span className="land-muted">Could not stop: {stopError}</span>}
      </div>
    );
  if (archive.failed)
    return (
      <p className="land-alert" role="alert">
        {archive.run?.status === "cancelled"
          ? "The archive search was stopped."
          : `The archive search failed${archive.run?.error ? `: ${archive.run.error}` : "."}`}{" "}
        {canEdit && (
          <button type="button" className="land-link" disabled={starter.busy} onClick={onStart}>
            Try again
          </button>
        )}
      </p>
    );
  if (archive.stale) return <p className="land-note">Found for an earlier boundary. {again}</p>;
  if (!count && archive.run?.status === "partial")
    return (
      <p className="land-note">
        The photo and map catalogs did not answer.{" "}
        {canEdit && (
          <button type="button" className="land-link" disabled={starter.busy} onClick={onStart}>
            Try again
          </button>
        )}
      </p>
    );
  if (!count)
    return (
      <p className="land-muted">
        No openly licensed old maps or photos turned up near this land. {again}
      </p>
    );
  if (archive.run?.status === "partial")
    return (
      <p className="land-note">One catalog did not answer, so this may be incomplete. {again}</p>
    );
  return null;
}

/** The Records card's summary: facts that point at the map, one next step, then the details. */
export function RecordsSummary({
  land,
  archive,
  starter,
  items,
  anchors,
  documents,
}: {
  land: LandArea;
  archive: Analysis;
  starter: Starter;
  items: ArchiveItem[];
  /** Where each item's pin is, for the lines from its facts. */
  anchors: Map<string, LandPoint>;
  documents: UseQueryResult<LandDocument[]>;
}) {
  const ids = parcelIds(land.source);
  const maps = items.filter((item) => item.kind === "historical-map");
  const photos = items.filter((item) => item.kind === "photograph");
  const oldestMap = maps[0];
  const datedPhotos = photos.filter((item) => item.year !== null);
  const oldestPhoto =
    datedPhotos.reduce<ArchiveItem | undefined>(
      (oldest, item) => (!oldest || (item.year ?? 0) < (oldest.year ?? 0) ? item : oldest),
      undefined,
    ) ?? photos[0];
  const onLand = photos.filter((item) => item.onLand).length;
  const naming = (documents.data ?? []).filter((document) => namesParcel(document, ids)).length;
  const start = () =>
    void starter.start({ title: ARCHIVE_TITLE, kind: "archive", question: ARCHIVE_QUESTION });
  const archiveMissing = !archive.loading && !archive.error && !archive.investigation;
  const addDocument = () => useRecords.getState().open({ kind: "add" });

  return (
    <>
      <ul className="land-facts">
        <Fact
          label="Boundary source"
          value={land.source.label}
          note={sourceMeaning(land.source.meaning)}
          target={{ key: "records-boundary", geometry: land.boundary }}
        />
        <Fact
          label="Documents"
          value={<DocumentsValue query={documents} />}
          note={naming ? `${naming === 1 ? "One names" : `${naming} name`} this parcel` : undefined}
        />
        {oldestMap && (
          <ItemFact
            label={maps.length > 1 ? `Oldest of ${maps.length} maps` : "Old map"}
            value={`${oldestMap.year ?? "Undated"} · ${oldestMap.title}`}
            item={oldestMap}
            anchor={anchors.get(oldestMap.id)}
          />
        )}
        {oldestPhoto && (
          <ItemFact
            label={photos.length > 1 ? `${photos.length} old photos` : "Old photo"}
            value={[
              onLand ? `${onLand} on the land` : "all nearby",
              oldestPhoto.year ? `oldest ${oldestPhoto.year}` : "",
            ]
              .filter(Boolean)
              .join(" · ")}
            item={oldestPhoto}
            anchor={anchors.get(oldestPhoto.id)}
          />
        )}
      </ul>
      <ArchiveState archive={archive} starter={starter} count={items.length} onStart={start} />
      {starter.error && (
        <p className="land-alert" role="alert">
          The archive search could not start. {starter.error}
        </p>
      )}
      <RecordsThenNow />
      {items.length > 0 && (
        <p className="records-legend">
          {maps.length > 0 && (
            <span>
              <i className="records-legend__sheet" aria-hidden="true" /> Old map sheet
            </span>
          )}
          {photos.length > 0 && (
            <span>
              <i className="records-legend__photo" aria-hidden="true" /> Old photo
            </span>
          )}
        </p>
      )}
      {starter.canEdit && (
        <div className="land-row">
          <GlassButton
            variant={archiveMissing ? "ghost" : "primary"}
            size={archiveMissing ? "sm" : "md"}
            leadingIcon={<FilePlus2 size={15} />}
            onClick={addDocument}
          >
            Add a document
          </GlassButton>
        </div>
      )}
      <LensDetails>
        <RecordsBoundary land={land} />
        <RecordsDocuments landId={land.id} parcelIds={ids} onOpen={openDocument} />
        {items.length > 0 && (
          <RecordsArchiveList
            items={items}
            findings={archive.detail?.findings ?? []}
            onOpen={openItem}
          />
        )}
        {starter.canEdit && archive.investigation && !archive.running && (
          <div className="land-row">
            <button type="button" className="land-link" disabled={starter.busy} onClick={start}>
              Search the archives again
            </button>
          </div>
        )}
      </LensDetails>
    </>
  );
}

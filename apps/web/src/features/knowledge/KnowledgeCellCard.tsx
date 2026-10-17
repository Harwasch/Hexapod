import {
  Camera,
  ClipboardCheck,
  FileText,
  Image as ImageIcon,
  Satellite,
  Sprout,
  type LucideIcon,
} from "lucide-react";
import { useMemo } from "react";

import type { KnowledgeCellRecord, KnowledgeRecord } from "@twin/contracts";
import { GlassBadge, GlassButton } from "@twin/ui";

import { describeError } from "@/lib/log";
import { useKnowledge } from "@/state/knowledge";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useUi } from "@/state/ui";

import { FloatingPanel } from "../shell/FloatingPanel";
import { useKnowledgeCell, useKnowledgeGrid } from "./api";
import { agePhrase, classLabel } from "./classes";
import { decodeGrid, densityAt, NO_RECORD } from "./grid";

import "./knowledge.css";

const KIND_ICON: Record<KnowledgeRecord["kind"], LucideIcon> = {
  capture: Camera,
  inspection: ClipboardCheck,
  "raster-observation": Satellite,
  "archive-alignment": ImageIcon,
  "research-evidence": FileText,
  survey: Sprout,
};

/** Where a record lives in the Land tool, if it has a place there. */
function openRecord(record: KnowledgeRecord): (() => void) | null {
  const { kind, id, detail } = record.link;
  const context = useLandContext.getState();
  const investigation = typeof detail?.investigationId === "string" ? detail.investigationId : null;
  if (kind === "inventory")
    return () => {
      context.selectInventory(id);
      useUi.getState().setPanel("land");
    };
  if (kind === "survey")
    return () => {
      context.selectSurvey(id);
      context.setSection("ecology");
      useUi.getState().setPanel("land");
    };
  if (investigation && (kind === "raster" || kind === "evidence" || kind === "registration"))
    return () => {
      context.selectInvestigation(investigation);
      context.setSection("discover");
      useUi.getState().setPanel("land");
    };
  return null;
}

function RecordRow({ item }: { item: KnowledgeCellRecord }) {
  const { record } = item;
  const Icon = KIND_ICON[record.kind];
  const open = openRecord(record);
  const counted =
    record.countedClassId !== record.classId
      ? ` · counts as ${classLabel(record.countedClassId)}`
      : "";
  return (
    <li className="knowledge-record" data-testid="knowledge-record">
      <Icon size={14} className="glass-muted knowledge-record__icon" aria-hidden="true" />
      <div className="knowledge-record__body">
        <div className="knowledge-record__title">
          {record.title}
          {item.best && (
            <GlassBadge tone="accent" className="knowledge-record__best">
              Sets this cell
            </GlassBadge>
          )}
        </div>
        <div className="knowledge-record__meta">
          <span className={`knowledge-chip knowledge-class--${record.classId}`}>
            {classLabel(record.classId)}
            {counted}
          </span>
          <span>
            {record.observedLabel}
            {record.ageDays != null ? ` · ${agePhrase(record.ageDays)}` : ""}
          </span>
        </div>
        <div
          className="knowledge-record__math knowledge-num"
          title="Credit = Q × 0.5^(age ÷ half-life)"
        >
          {item.arithmetic}
        </div>
        {record.classNote && <div className="knowledge-record__note">{record.classNote}</div>}
        <div className="knowledge-record__source">
          {record.provenance.attribution} · {record.provenance.license}
          {record.provenance.url && (
            <>
              {" · "}
              <a href={record.provenance.url} target="_blank" rel="noreferrer">
                source
              </a>
            </>
          )}
        </div>
        {open && (
          <GlassButton size="sm" variant="ghost" onClick={open}>
            Open in Land
          </GlassButton>
        )}
      </div>
    </li>
  );
}

/**
 * The card for one cell of the Fog of Knowing: how foggy it is and every stored record that
 * covers it, with its date, class and the credit arithmetic, best first. Its header shows at
 * once from the grid already on the map; the ledger comes from the API. Records with no
 * published location count for the whole land and sit in their own group. All of it is
 * computed from stored records (policy fog-v1), never measured here.
 */
export function KnowledgeCellCard() {
  const selected = useKnowledge((s) => s.selected);
  const visible = useKnowledge((s) => s.visible);
  const clear = useKnowledge((s) => s.clear);
  const land = useLand((s) => s.active);
  const landId = land?.id ?? null;
  const open = visible && selected !== null && selected.landId === landId;
  const gridQuery = useKnowledgeGrid(land, visible);
  const decoded = useMemo(
    () => (gridQuery.data ? decodeGrid(gridQuery.data) : null),
    [gridQuery.data],
  );
  const cell = useKnowledgeCell(open ? selected : null, land?.revision ?? null);
  const density = decoded && selected ? densityAt(decoded, selected.col, selected.row) : null;
  const bestIndex =
    decoded && selected ? decoded.best[selected.row * decoded.columns + selected.col] : undefined;
  const best =
    bestIndex !== undefined && bestIndex !== NO_RECORD
      ? gridQuery.data?.records[bestIndex]
      : undefined;
  const records = cell.data?.records ?? [];
  const located = records.filter((item) => item.record.located);
  const landWide = records.filter((item) => !item.record.located);
  return (
    <FloatingPanel
      open={open}
      from="right"
      title="What we know here"
      onClose={clear}
      testId="knowledge-cell"
    >
      <div className="glass-stack knowledge-cell">
        <div className="glass-row" style={{ flexWrap: "wrap" }}>
          <GlassBadge tone="accent">Computed</GlassBadge>
          <GlassBadge>{gridQuery.data?.policy.version ?? "fog-v1"}</GlassBadge>
          <GlassBadge>{Math.round(decoded?.cellM ?? 20)} m cell</GlassBadge>
        </div>
        <p className="knowledge-cell__head" data-testid="knowledge-cell-head">
          {density === null ? (
            "Outside the land"
          ) : (
            <>
              Fog density <strong className="knowledge-num">{density.toFixed(2)}</strong>
              {" · "}
              {best
                ? `best: ${best.title}, ${agePhrase(best.ageDays)}`
                : "no stored record covers this cell"}
            </>
          )}
        </p>
        {cell.isLoading && <p className="glass-subtle">Reading the records…</p>}
        {cell.isError && (
          <p role="alert">The records could not be read: {describeError(cell.error)}</p>
        )}
        {cell.data && records.length === 0 && (
          <p className="glass-subtle">
            Nothing stored covers this cell. Inspect a feature here, store a satellite look, or fly
            a scout to lift the fog.
          </p>
        )}
        {located.length > 0 && (
          <ul className="knowledge-records" aria-label="Records here">
            {located.map((item) => (
              <RecordRow key={item.record.key} item={item} />
            ))}
          </ul>
        )}
        {landWide.length > 0 && (
          <details className="knowledge-group" open={located.length === 0}>
            <summary>
              No published location ({landWide.length}): counts for the whole land, capped at
              regional
            </summary>
            <ul className="knowledge-records">
              {landWide.map((item) => (
                <RecordRow key={item.record.key} item={item} />
              ))}
            </ul>
          </details>
        )}
        {cell.data && (
          <details className="knowledge-group">
            <summary>Not counted ({cell.data.notCounted.length})</summary>
            <ul className="knowledge-not-counted__list">
              {cell.data.notCounted.map((item) => (
                <li key={item.title}>
                  <strong>{item.title}</strong> {item.reason}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </FloatingPanel>
  );
}

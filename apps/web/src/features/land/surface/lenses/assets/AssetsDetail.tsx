import { useState, type ReactNode } from "react";
import { ClipboardCheck, MessageCircleQuestion, Pencil } from "lucide-react";
import { useQueries } from "@tanstack/react-query";

import type { LandArea } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";
import { useSettings } from "@/state/settings";

import { askAgent } from "../../agentBus";
import { Fact } from "../../mapLinks";
import { useAssetHistory, useAssetInspections, useRefreshAssets } from "./assetsApi";
import { AssetsInspectionForm } from "./AssetsInspectionForm";
import {
  CATEGORIES,
  CONDITIONS,
  needsAttention,
  sizeLabel,
  STATUSES,
  whenLabel,
  whereLabel,
  type Asset,
  type Inspection,
} from "./assetsModel";

/** A folded section whose content (and the requests behind it) loads only once opened. */
export function LazyDetails({
  summary,
  children,
}: {
  summary: ReactNode;
  children: (open: boolean) => ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details className="land-details" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>{summary}</summary>
      {open && <div className="land-details__body">{children(open)}</div>}
    </details>
  );
}

function InspectionItem({ inspection }: { inspection: Inspection }) {
  const measurements = Object.entries(inspection.measurements ?? {});
  return (
    <li className="assets-history__item">
      <span className="assets-history__head">
        <i
          className="assets-dot"
          style={{ background: CONDITIONS[inspection.condition].color }}
          aria-hidden="true"
        />
        <b>{CONDITIONS[inspection.condition].label}</b>
        <span className="land-muted">{new Date(inspection.observedAt).toLocaleString()}</span>
      </span>
      <span>{inspection.notes}</span>
      {measurements.length > 0 && (
        <span className="land-muted">
          {measurements
            .map(([name, value]) => `${name} ${value} ${inspection.measurementUnits?.[name] ?? ""}`)
            .join(" · ")}
        </span>
      )}
      <span className="land-muted">
        Version {inspection.featureRevision}
        {inspection.evidenceIds?.length ? ` · ${inspection.evidenceIds.length} sources` : ""}
      </span>
    </li>
  );
}

function Sources({ ids }: { ids: string[] }) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const results = useQueries({
    queries: ids.slice(0, 20).map((id) => ({
      queryKey: ["land-source", scope, id],
      enabled: ready,
      retry: false,
      queryFn: () =>
        unwrap(
          api.GET("/api/v1/research/evidence/{evidence_id}", {
            params: { path: { evidence_id: id } },
          }),
        ),
    })),
  });
  return (
    <ul className="assets-history">
      {results.map((result, index) => {
        const evidence = result.data;
        return (
          <li key={ids[index]} className="assets-history__item">
            {evidence ? (
              <>
                {evidence.url ? (
                  <a href={evidence.url} target="_blank" rel="noreferrer">
                    {evidence.title}
                  </a>
                ) : (
                  <b>{evidence.title}</b>
                )}
                <span className="land-muted">
                  {evidence.provider} · {evidence.license}
                </span>
              </>
            ) : (
              <span className="land-muted">
                {result.isError ? "This source could not be read." : "Loading…"}
              </span>
            )}
          </li>
        );
      })}
      {ids.length > 20 && <li className="land-muted">And {ids.length - 20} more.</li>}
    </ul>
  );
}

function Revisions({ land, asset }: { land: LandArea; asset: Asset }) {
  const history = useAssetHistory(land, asset, true);
  if (history.isError) return <p className="land-alert">Changes could not be loaded.</p>;
  if (!history.data) return <p className="land-muted">Loading changes…</p>;
  return (
    <ul className="assets-history">
      {history.data.map((revision) => (
        <li key={revision.revision} className="assets-history__item">
          <span className="assets-history__head">
            <b>Version {revision.revision}</b>
            <span className="land-muted">{new Date(revision.createdAt).toLocaleDateString()}</span>
          </span>
          {revision.note && <span>{revision.note}</span>}
        </li>
      ))}
    </ul>
  );
}

/**
 * One asset: what and where it is, its condition from the newest inspection, and one next
 * step (record an inspection). Its inspections, changes, attributes and evidence are folded
 * below.
 */
export function AssetsDetail({
  land,
  asset,
  canEdit,
  onEdit,
}: {
  land: LandArea;
  asset: Asset;
  canEdit: boolean;
  onEdit: () => void;
}) {
  const units = useSettings((s) => s.units);
  const inspections = useAssetInspections(land, asset.id);
  const refresh = useRefreshAssets(land);
  const [inspecting, setInspecting] = useState(false);
  const [saved, setSaved] = useState(false);
  const latest = inspections.data?.[0] ?? null;
  const attention = needsAttention(latest?.condition);
  const size = sizeLabel(asset.geometry, units);
  const attributes = Object.entries(asset.attributes ?? {});
  const evidence = asset.evidenceIds ?? [];
  const target = { key: `asset:${asset.id}`, geometry: asset.geometry };

  return (
    <>
      <p className="assets-meta">
        <i
          className="assets-dot"
          style={{ background: CATEGORIES[asset.category].color }}
          aria-hidden="true"
        />
        {[CATEGORIES[asset.category].one, STATUSES[asset.status ?? "candidate"], size]
          .filter(Boolean)
          .join(" · ")}
      </p>
      <ul className="land-facts">
        <Fact
          label="Condition"
          tone={attention ? "warn" : "accent"}
          value={
            inspections.isPending
              ? "Checking…"
              : latest
                ? `${CONDITIONS[latest.condition].label}, ${whenLabel(latest.observedAt)}`
                : "Not inspected yet"
          }
          note={latest?.notes ? latest.notes.slice(0, 140) : undefined}
        />
        <Fact label="Where" value={whereLabel(asset, units)} target={target} />
      </ul>
      {inspections.isError && (
        <p className="land-alert" role="alert">
          Inspections could not be loaded.{" "}
          <button type="button" className="land-link" onClick={() => void inspections.refetch()}>
            Retry
          </button>
        </p>
      )}
      {saved && !inspecting && (
        <p className="land-muted" role="status">
          Inspection saved.
        </p>
      )}
      {inspecting ? (
        <AssetsInspectionForm
          landId={land.id}
          featureId={asset.id}
          onCancel={() => setInspecting(false)}
          onSaved={async () => {
            setInspecting(false);
            setSaved(true);
            await refresh(asset.id);
          }}
        />
      ) : (
        <div className="land-row">
          {canEdit && (
            <GlassButton
              variant="primary"
              leadingIcon={<ClipboardCheck size={16} aria-hidden="true" />}
              onClick={() => {
                setSaved(false);
                setInspecting(true);
              }}
            >
              Record an inspection
            </GlassButton>
          )}
          {canEdit && (
            <GlassButton leadingIcon={<Pencil size={15} aria-hidden="true" />} onClick={onEdit}>
              Edit
            </GlassButton>
          )}
          <GlassButton
            variant="ghost"
            size="sm"
            leadingIcon={<MessageCircleQuestion size={15} aria-hidden="true" />}
            onClick={() =>
              askAgent(`What should I know about the ${asset.name.trim()} on this land?`)
            }
          >
            Ask about this
          </GlassButton>
        </div>
      )}
      <details className="land-details">
        <summary>
          Inspections
          {inspections.data?.length ? ` · ${inspections.data.length}` : ""}
        </summary>
        <div className="land-details__body">
          {inspections.data?.length ? (
            <ul className="assets-history">
              {inspections.data.map((inspection) => (
                <InspectionItem key={inspection.id} inspection={inspection} />
              ))}
            </ul>
          ) : (
            <p className="land-muted">No inspections yet.</p>
          )}
        </div>
      </details>
      <LazyDetails summary="Details and history">
        {() => (
          <>
            {asset.description && <p className="assets-notes">{asset.description}</p>}
            {attributes.length > 0 && (
              <dl className="assets-attributes-table">
                {attributes.map(([key, value]) => (
                  <div key={key}>
                    <dt>{key}</dt>
                    <dd>{value === null ? "Unknown" : String(value)}</dd>
                  </div>
                ))}
              </dl>
            )}
            {evidence.length > 0 && (
              <>
                <h4 className="assets-subhead">Linked evidence</h4>
                <Sources ids={evidence} />
              </>
            )}
            <h4 className="assets-subhead">Changes</h4>
            <Revisions land={land} asset={asset} />
            <p className="land-muted">
              From: {asset.source.label}
              {asset.externalRef &&
                ` · dataset ${asset.externalRef.namespace}, record ${asset.externalRef.recordId}`}
              {` · version ${asset.revision} · added ${new Date(asset.createdAt).toLocaleDateString()}`}
            </p>
          </>
        )}
      </LazyDetails>
    </>
  );
}

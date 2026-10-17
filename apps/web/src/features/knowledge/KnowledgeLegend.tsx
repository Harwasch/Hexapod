import { useQueryClient } from "@tanstack/react-query";
import { CloudFog, Info } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";

import type { KnowledgeGrid, LandArea } from "@twin/contracts";
import { GlassButton, GlassPopover, GlassSwitch } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useKnowledge } from "@/state/knowledge";
import { useLand } from "@/state/land";
import { useLandCanEdit } from "@/state/landIdentity";

import { invalidateLandDerived } from "../land/landDerived";
import { beginInvestigation, useInvestigation, useResearchStatus } from "../land/researchApi";
import { workerIdleNote } from "../land/researchAvailability";
import { useKnowledgeGrid } from "./api";
import { CLASS_ORDER, classLabel } from "./classes";
import { liftFog } from "./liftFog";

import "./knowledge.css";

/** The last 30 days, as one Sentinel-2 window (the API allows up to 31 days). */
function latestWindow(today = new Date()): { startDate: string; endDate: string } {
  const end = today.toISOString().slice(0, 10);
  const start = new Date(today.getTime() - 30 * 86_400_000).toISOString().slice(0, 10);
  return { startDate: start, endDate: end };
}

function percent(fraction: number): string {
  if (fraction > 0 && fraction < 0.01) return "<1%";
  return `${Math.round(fraction * 100)}%`;
}

/**
 * What sets each cell, as one stacked bar: the best record's class, finest first, then the
 * cells nothing covers. An ordinal ramp (one hue, light to dark) plus a hatched "no record";
 * every segment is named with its share below, so nothing is read from colour alone.
 */
function ClassBar({ grid }: { grid: KnowledgeGrid }) {
  const shares = grid.summary.byBestClass;
  const order = [...CLASS_ORDER, "none"] as const;
  const shown = order.filter((id) => (shares[id] ?? 0) > 0);
  return (
    <figure className="knowledge-bar" aria-label="What sets each cell">
      <div
        className="knowledge-bar__track"
        role="img"
        aria-label={shown.map((id) => `${classLabel(id)} ${percent(shares[id] ?? 0)}`).join(", ")}
      >
        {shown.map((id) => (
          <span
            key={id}
            className={`knowledge-bar__segment knowledge-class--${id}`}
            style={{ flexGrow: shares[id] ?? 0 }}
            title={`${classLabel(id)}: ${percent(shares[id] ?? 0)} of the land`}
          />
        ))}
      </div>
      <figcaption className="knowledge-bar__legend">
        {shown.map((id) => (
          <span key={id} className="knowledge-bar__key">
            <span className={`knowledge-swatch knowledge-class--${id}`} aria-hidden="true" />
            {classLabel(id)} <span className="knowledge-num">{percent(shares[id] ?? 0)}</span>
          </span>
        ))}
      </figcaption>
    </figure>
  );
}

/**
 * Starts a Sentinel-2 vegetation run over the last 30 days (the existing raster analysis, at
 * 20 m) and watches it: when it finishes, everything computed from the land's records is asked
 * again, so the fog lifts where the satellite saw the ground.
 */
function useSatelliteLook(land: LandArea) {
  const client = useQueryClient();
  const [pending, setPending] = useState<{ investigationId: string; runId: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const investigation = useInvestigation(pending?.investigationId ?? null);
  const run = investigation.data?.runs.find((item) => item.id === pending?.runId);
  const settled = useRef<string | null>(null);
  useEffect(() => {
    if (!run || !pending || settled.current === run.id) return;
    if (["succeeded", "partial", "failed", "cancelled"].includes(run.status)) {
      settled.current = run.id;
      void invalidateLandDerived(client, land.id);
    }
  }, [run, pending, client, land.id]);
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const created = await beginInvestigation(land, "Latest satellite look");
      const window = latestWindow();
      const started = await unwrap(
        api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
          params: { path: { investigation_id: created.id } },
          body: {
            kind: "raster",
            question: "Store the latest clear Sentinel-2 look at this land as dated evidence.",
            requestKey: crypto.randomUUID(),
            analysis: {
              dataset: "sentinel-2-ndvi",
              resolutionM: 20,
              maxDimension: 512,
              periods: [window],
            },
          },
        }),
      );
      setPending({ investigationId: created.id, runId: started.id });
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };
  return { start, busy, error, run };
}

/**
 * The Fog of Knowing's control in the strip: a pill to show or hide it over the selected Land,
 * and the details: the formula and class table from the API (policy fog-v1, verbatim), what
 * sets the cells, what is not counted, and the two ways forward: fetch a fresh satellite look
 * (stored evidence; the fog lifts when it lands) or draft a scout flight over the foggiest
 * ground. Everything here is computed from stored records, never measured by it.
 */
export function KnowledgeLegend() {
  const land = useLand((s) => s.active);
  if (!land) return null;
  return <Legend land={land} />;
}

function Legend({ land }: { land: LandArea }) {
  const scene = useScene();
  const visible = useKnowledge((s) => s.visible);
  const setVisible = useKnowledge((s) => s.setVisible);
  const canEdit = useLandCanEdit();
  const status = useResearchStatus();
  const query = useKnowledgeGrid(land.id, visible);
  const grid = query.data?.landId === land.id ? query.data : undefined;
  const [open, setOpen] = useState(false);
  const [lifting, setLifting] = useState(false);
  const look = useSatelliteLook(land);
  const switchId = useId();
  const idle = workerIdleNote(status.data);
  const records = grid?.records.length ?? 0;
  const lookState =
    look.run?.status === "queued" || look.run?.status === "running"
      ? "Fetching the latest satellite look…"
      : look.run?.status === "succeeded" || look.run?.status === "partial"
        ? "Satellite look stored. The fog now counts it."
        : look.run?.status === "failed"
          ? `The satellite look failed: ${look.run.error ?? "no reason given"}.`
          : null;
  return (
    <div className="knowledge-pill-wrap">
      <div
        className="glass glass--strong glass--pill knowledge-pill"
        role="group"
        aria-label="Fog of knowing"
        data-testid="knowledge-legend"
      >
        <CloudFog size={14} className="glass-muted" aria-hidden="true" />
        <label htmlFor={switchId} className="knowledge-pill__label">
          Fog of knowing
        </label>
        <GlassSwitch
          id={switchId}
          checked={visible}
          onCheckedChange={setVisible}
          aria-label="Show the fog of knowing"
        />
        {visible && grid && (
          <span className="knowledge-pill__stat knowledge-num" data-testid="knowledge-foggy">
            {percent(grid.summary.foggyFraction)} foggy
          </span>
        )}
        {visible && query.isLoading && <span className="glass-subtle">Computing…</span>}
        <GlassPopover
          open={open}
          onOpenChange={setOpen}
          side="top"
          align="center"
          className="knowledge-popover"
          aria-label="About the fog of knowing"
          data-hud-popover=""
          trigger={
            <GlassButton
              iconOnly
              size="sm"
              variant="ghost"
              aria-label="About the fog of knowing"
              data-testid="knowledge-details"
            >
              <Info size={14} aria-hidden="true" />
            </GlassButton>
          }
        >
          <div className="knowledge-legend" data-testid="knowledge-legend-details">
            <p className="glass-eyebrow">
              Computed from stored records · {grid?.policy.version ?? "fog-v1"}
            </p>
            <p className="knowledge-lead">
              How fine and how fresh what this workspace has stored about each{" "}
              {Math.round(grid?.density.cellM ?? 20)} m cell is. Thick fog: nothing stored there, or
              only old or coarse records. It lifts only when new evidence is stored.
            </p>
            {query.isError && (
              <p role="alert">The fog could not be computed: {describeError(query.error)}</p>
            )}
            {grid && (
              <>
                <ClassBar grid={grid} />
                <p className="knowledge-summary" data-testid="knowledge-summary">
                  Foggy: {percent(grid.summary.foggyFraction)} (density ≥{" "}
                  {grid.policy.foggyThreshold}) · computed from {records} cited record
                  {records === 1 ? "" : "s"} · {grid.policy.version}
                </p>
                <p className="knowledge-formula">{grid.policy.formula}</p>
                <table className="knowledge-table">
                  <caption className="glass-eyebrow">Classes (policy, not measurement)</caption>
                  <thead>
                    <tr>
                      <th scope="col">Class</th>
                      <th scope="col">Q</th>
                      <th scope="col">Half-life</th>
                    </tr>
                  </thead>
                  <tbody>
                    {grid.policy.classes.map((item) => (
                      <tr key={item.id} title={item.example}>
                        <th scope="row">
                          <span
                            className={`knowledge-swatch knowledge-class--${item.id}`}
                            aria-hidden="true"
                          />
                          {item.label}
                          <span className="knowledge-example">{item.example}</span>
                        </th>
                        <td className="knowledge-num">{item.quality.toFixed(2)}</td>
                        <td className="knowledge-num">
                          {item.halfLifeDays >= 365
                            ? `${Math.round(item.halfLifeDays / 365)} y`
                            : `${item.halfLifeDays} d`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <details className="knowledge-not-counted">
                  <summary>Not counted ({grid.notCounted.length})</summary>
                  <ul>
                    {grid.notCounted.map((item) => (
                      <li key={item.title}>
                        <strong>{item.title}</strong> {item.reason}
                      </li>
                    ))}
                  </ul>
                </details>
              </>
            )}
            <div className="knowledge-actions">
              <GlassButton
                size="sm"
                disabled={
                  !canEdit || look.busy || lookState === "Fetching the latest satellite look…"
                }
                onClick={() => void look.start()}
                data-testid="knowledge-satellite"
              >
                Fetch latest satellite look
              </GlassButton>
              <GlassButton
                size="sm"
                variant="primary"
                disabled={lifting || !grid}
                onClick={() => {
                  setLifting(true);
                  setVisible(true);
                  void liftFog(scene, land).finally(() => {
                    setLifting(false);
                    setOpen(false);
                  });
                }}
                data-testid="knowledge-lift"
              >
                {lifting ? "Finding the foggiest ground…" : "Lift the fog"}
              </GlassButton>
            </div>
            {!canEdit && (
              <p className="glass-subtle">
                Your workspace role can view the fog but not start research.
              </p>
            )}
            {canEdit && idle && <p className="glass-subtle">{idle}</p>}
            {lookState && <p role="status">{lookState}</p>}
            {look.error && <p role="alert">{look.error}</p>}
            <p className="glass-subtle">
              Lift the fog drafts a scout flight over the foggiest ground within one battery,
              nearest your view. Its passes are a forecast, not evidence.
            </p>
          </div>
        </GlassPopover>
      </div>
    </div>
  );
}

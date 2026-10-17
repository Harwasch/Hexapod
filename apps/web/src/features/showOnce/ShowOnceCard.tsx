import { useQueryClient } from "@tanstack/react-query";
import { Crosshair, Eraser, Shield, Sparkles, X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";

import type { LandArea, SignatureScene, SignatureSeed } from "@twin/contracts";
import { GlassSlider } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { TOUCH_MEDIA, useMediaQuery } from "@/lib/media";
import { useLandContext } from "@/state/landContext";
import { useLandAccessReady, useLandCanEdit } from "@/state/landIdentity";
import { useShowOnce } from "@/state/showOnce";
import { useUi } from "@/state/ui";

import { invalidateLandDerived } from "../land/landDerived";
import { RASTER_RUNS, rasterRequest, signaturePeriods } from "../land/rasterRuns";
import { beginInvestigation, useInvestigation, useResearchStatus } from "../land/researchApi";
import { workerIdleNote, workerLine } from "../land/researchAvailability";
import { freshSignature, signatureGap, useLandRasters, type LandRaster } from "./api";
import {
  ALIKE_MAX,
  ALIKE_MIN,
  expectedSpacingM,
  formatAcres,
  litStats,
  spacingLabel,
} from "./grid";
import { makeJob, type JobSummary } from "./makeJob";
import { dropPendingTaps } from "./session";

import "./showOnce.css";

const SETTLED = ["succeeded", "partial", "failed", "cancelled"];
/** The grid a signature run asks for: 10 m, at most 512 cells across (the server's default). */
const SIGNATURE_REQUEST = {
  resolutionM: RASTER_RUNS["sentinel-2-signature"].resolutionM,
  maxDimension: rasterRequest("sentinel-2-signature").maxDimension ?? 512,
};
type Period = ReturnType<typeof signaturePeriods>[number];

function sceneDay(iso: string | null | undefined): string | null {
  if (!iso) return null;
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

/**
 * Starts the land's seasonal signatures through the research worker (the raster run Explore
 * this land's "Prepare 10 m signatures" starts: `rasterRuns.ts`), and watches it: when it
 * settles, everything computed from the land's records is asked again, the raster list too,
 * and `looked` says that answer is in.
 */
function usePrepareSignatures(land: LandArea) {
  const client = useQueryClient();
  const [pending, setPending] = useState<{ investigationId: string; runId: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [looked, setLooked] = useState<string | null>(null);
  const investigation = useInvestigation(pending?.investigationId ?? null);
  const run = investigation.data?.runs.find((item) => item.id === pending?.runId);
  const settled = useRef<string | null>(null);
  useEffect(() => {
    if (!run || settled.current === run.id || !SETTLED.includes(run.status)) return;
    settled.current = run.id;
    const id = run.id;
    void invalidateLandDerived(client, land.id).finally(() => setLooked(id));
  }, [run, client, land.id]);
  const start = async (periods: Period[]) => {
    const { title, question, maxSeconds } = RASTER_RUNS["sentinel-2-signature"];
    setBusy(true);
    setError(null);
    try {
      const created = await beginInvestigation(land, title);
      const started = await unwrap(
        api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
          params: { path: { investigation_id: created.id } },
          body: {
            kind: "raster",
            question,
            requestKey: crypto.randomUUID(),
            analysis: rasterRequest("sentinel-2-signature", periods),
            budget: { maxSeconds },
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
  return { start, busy, error, run, looked: looked !== null && looked === run?.id };
}

/** "2025-01 · 2025-04 · 2025-07 · 2025-10": the months a signature's windows start in. */
function monthLine(periods: readonly Period[]): string {
  return periods.map((period) => period.startDate.slice(0, 7)).join(" · ");
}

/**
 * The seasons to prepare: the latest complete year's, or after signatures for this boundary in
 * which no cell was clear in every season, the year before the earliest year tried (the same
 * dates again would read the same cloudy scenes).
 */
function nextPeriods(rasters: readonly LandRaster[] | undefined, land: LandArea): Period[] {
  const tried = (rasters ?? [])
    .filter(
      (raster) =>
        raster.request.dataset === "sentinel-2-signature" &&
        !raster.stale &&
        raster.boundaryRevision === land.revision &&
        (raster.metadata.signature?.commonCells ?? 0) === 0,
    )
    .map((raster) => Number(raster.request.periods?.[0]?.startDate.slice(0, 4)))
    .filter((year) => Number.isFinite(year));
  if (tried.length === 0) return signaturePeriods();
  // signaturePeriods(day) reads the year before `day`'s.
  return signaturePeriods(`${Math.min(...tried)}-06-01`);
}

/** What the raster says about the windows that left no cell clear in every season. */
function unclearReasons(raster: LandRaster): string[] {
  return raster.metadata.warnings.filter(
    (warning) =>
      warning.includes("no clear land samples") || warning.startsWith("No cell is clear"),
  );
}

/** The tapped cell's NDVI through the seasons: words for every reader, a line for the eye. */
function SeedSeasons({ seed, scenes }: { seed: SignatureSeed; scenes: readonly SignatureScene[] }) {
  const points = seed.windows.map((window, index) => ({
    label:
      sceneDay(window.acquiredAt ?? scenes[window.window]?.acquiredAt) ?? `Window ${index + 1}`,
    ndvi: window.ndvi,
  }));
  const width = 96;
  const height = 28;
  const step = points.length > 1 ? width / (points.length - 1) : 0;
  const y = (ndvi: number) => height - ((Math.max(-0.2, Math.min(1, ndvi)) + 0.2) / 1.2) * height;
  return (
    <div className="show-once__seed" data-testid="show-once-seed">
      <svg
        className="show-once__spark"
        viewBox={`-3 -3 ${width + 6} ${height + 6}`}
        width={width + 6}
        height={height + 6}
        aria-hidden="true"
      >
        <polyline points={points.map((p, i) => `${i * step},${y(p.ndvi)}`).join(" ")} fill="none" />
        {points.map((p, i) => (
          <circle key={p.label} cx={i * step} cy={y(p.ndvi)} r="2.4" />
        ))}
      </svg>
      <p className="show-once__seasons">
        NDVI{" "}
        {points.map((p, i) => (
          <span key={p.label}>
            {i > 0 && " · "}
            {p.label} <span className="show-once__num">{p.ndvi.toFixed(2)}</span>
          </span>
        ))}
      </p>
    </div>
  );
}

/**
 * Show once, everywhere, for the selected Land (in its Ecology section): tap one place, and
 * every 10 m cell whose seasonal Sentinel-2 signature is alike lights up on the map
 * (`cesium/ShowOnceBridge.tsx`). "How alike" moves the bar a cell must clear as a uniform of
 * the drape, so nothing is fetched or uploaded while it slides; the tally counts the lit cells
 * of the display grid (approximate, and says so). Protect taps leave lit patches out, and
 * "Make this the job" turns the rest into areas of a new plan on the Land (`makeJob.ts`).
 *
 * The answer is a similar seasonal reflectance, measured by a satellite and compared here: an
 * inference, never a species identification, and the card says so whatever the slider.
 * Without signatures for the current boundary it offers to prepare them.
 */
export function ShowOnceCard({ land }: { land: LandArea }) {
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const scene = useScene();
  const section = useLandContext((s) => s.section);
  const panel = useUi((s) => s.activePanel);
  const visible = ready && section === "ecology" && panel === "land";
  const rasters = useLandRasters(land, visible);
  const raster = freshSignature(rasters.data, land);
  const gap = signatureGap(rasters.data, land);
  const prepare = usePrepareSignatures(land);
  const status = useResearchStatus(prepare.run?.status === "queued");
  const touch = useMediaQuery(TOUCH_MEDIA);
  const tap = touch ? "Tap" : "Click";
  const headingId = useId();

  const session = useShowOnce((s) => s.session);
  const mode = useShowOnce((s) => s.mode);
  const result = useShowOnce((s) => s.result);
  const pending = useShowOnce((s) => s.pending);
  const message = useShowOnce((s) => s.message);
  const threshold = useShowOnce((s) => s.threshold);
  const guarded = useShowOnce((s) => s.guarded);
  const mine = session?.landId === land.id && session.rasterId === raster?.id;
  const [making, setMaking] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);
  const [done, setDone] = useState<JobSummary | null>(null);

  // The map answers taps (and drapes) only while this card is on screen.
  useEffect(() => {
    useShowOnce.getState().setShown(visible);
  }, [visible]);
  useEffect(() => () => useShowOnce.getState().setShown(false), []);
  // A session on another Land or on signatures that are no longer the freshest ends.
  useEffect(() => {
    const current = useShowOnce.getState().session;
    if (current?.landId === land.id && raster && current.rasterId !== raster.id) {
      dropPendingTaps();
      useShowOnce.getState().stop();
    }
  }, [land.id, raster]);
  useEffect(
    () => () => {
      if (useShowOnce.getState().session?.landId === land.id) {
        dropPendingTaps();
        useShowOnce.getState().stop();
      }
    },
    [land.id],
  );

  const start = () => {
    if (!raster) return;
    setDone(null);
    setJobError(null);
    useShowOnce.getState().start({ landId: land.id, revision: land.revision, rasterId: raster.id });
  };
  const stop = () => {
    dropPendingTaps();
    setJobError(null);
    useShowOnce.getState().stop();
  };
  const make = async () => {
    setMaking(true);
    setJobError(null);
    try {
      const summary = await makeJob(land, scene);
      if (summary.areas === 0) {
        setJobError(
          "Nothing at this setting makes an area of half an acre or more. Slide How alike lower, or protect less.",
        );
        return;
      }
      dropPendingTaps();
      useShowOnce.getState().stop();
      setDone(summary);
    } catch (cause) {
      setJobError(`Could not make the job: ${describeError(cause)}`);
    } finally {
      setMaking(false);
    }
  };

  const signature = raster?.metadata.signature;
  // The grid's spacing as made: 10 m, or coarser on a land too wide for 506 cells across.
  const spacing = raster ? spacingLabel(raster.metadata.resolutionM) : null;
  const coarsened =
    raster && raster.metadata.resolutionM > SIGNATURE_REQUEST.resolutionM * 1.001
      ? (raster.metadata.warnings.find((warning) => warning.includes("coarsened")) ??
        `The grid was coarsened to ${spacing} to fit the analysis dimensions.`)
      : null;
  // Before any are made: the spacing this land's extent will give.
  const expected = expectedSpacingM(land.boundary, SIGNATURE_REQUEST);
  const wide = expected > SIGNATURE_REQUEST.resolutionM * 1.001;
  const periods = nextPeriods(rasters.data, land);
  const made = prepare.run && rasters.data?.some((item) => item.runId === prepare.run?.id);
  const scenes = raster?.metadata.vegetation?.observations ?? [];
  const sceneLine = scenes
    .map((o) => sceneDay(o.acquiredAt))
    .filter(Boolean)
    .join(", ");
  const year = scenes[0]?.acquiredAt?.slice(0, 4);
  const stats = mine && result ? litStats(result.grid, threshold, guarded) : null;
  const running = prepare.run?.status === "queued" || prepare.run?.status === "running";

  return (
    <section className="show-once" aria-labelledby={headingId} data-testid="show-once">
      <span className="land-eyebrow" id={headingId}>
        Show once, everywhere
      </span>
      <p className="show-once__lead">
        {tap} one place on the map. Every {spacing ? `${spacing} ` : ""}cell of {land.name} with a
        similar seasonal satellite signature lights up, and the lit ground can become the job.
      </p>
      <p className="show-once__chips">
        <span className="show-once__chip show-once__chip--measured">
          Measured: satellite reflectance
        </span>
        <span className="show-once__chip show-once__chip--inferred">Inferred: similarity</span>
      </p>

      {!raster && rasters.isLoading && (
        <p className="land-footnote" role="status">
          Looking for this land&apos;s seasonal signatures…
        </p>
      )}
      {rasters.isError && (
        <p className="land-error" role="alert">
          Could not list this land&apos;s rasters: {describeError(rasters.error)}
        </p>
      )}
      {!raster && gap && !rasters.isLoading && !rasters.isError && (
        <div className="show-once__prepare" data-testid="show-once-prepare">
          {gap.kind === "unclear" ? (
            <div data-testid="show-once-unclear">
              <p className="land-footnote">
                No cell of {land.name} was clear in every season on these dates (
                {monthLine(gap.raster.request.periods ?? [])}), so there is nothing to compare.
              </p>
              {unclearReasons(gap.raster).map((reason) => (
                <p key={reason} className="land-footnote">
                  {reason}
                </p>
              ))}
            </div>
          ) : (
            <p className="land-footnote">
              {gap.kind === "earlier"
                ? "The seasonal signatures were made for an earlier boundary. Prepare them again for this one."
                : "This land has no seasonal signatures yet. Preparing them reads four Sentinel-2 scenes, one per season, and takes a minute or two."}
            </p>
          )}
          {wide && (
            <p className="land-footnote" data-testid="show-once-wide">
              {land.name} is too wide for {spacingLabel(SIGNATURE_REQUEST.resolutionM)} cells in one
              analysis, so the grid will be about {spacingLabel(expected)}.
            </p>
          )}
          <p className="land-footnote">
            {monthLine(periods)} · {wide ? "about " : ""}
            {spacingLabel(expected)} grid
          </p>
          <div className="land-actions">
            <button
              type="button"
              className="land-primary"
              data-testid="show-once-prepare-start"
              disabled={!ready || !canEdit || prepare.busy || running}
              onClick={() => void prepare.start(periods)}
            >
              <Sparkles size={14} aria-hidden="true" />
              {prepare.busy
                ? "Starting…"
                : running
                  ? "Preparing…"
                  : gap.kind === "unclear"
                    ? `Try ${periods[0]?.startDate.slice(0, 4) ?? "earlier"}'s seasons`
                    : wide
                      ? "Prepare seasonal signatures"
                      : `Prepare ${spacingLabel(expected)} signatures`}
            </button>
          </div>
          {!canEdit && ready && (
            <p className="land-footnote">
              Your workspace role can view this land but not start analyses.
            </p>
          )}
          {prepare.run && !made && (
            <p className="land-footnote" role="status" data-testid="show-once-prepare-status">
              {prepare.run.status === "queued"
                ? (workerLine(status.data, prepare.run) ?? "Waiting for the research worker…")
                : prepare.run.status === "running"
                  ? "Reading four seasonal Sentinel-2 scenes for the signatures…"
                  : prepare.run.status === "failed"
                    ? `Preparing the signatures failed: ${prepare.run.error ?? "no reason given"}. Try again.`
                    : prepare.run.status === "cancelled"
                      ? "Preparing the signatures was stopped."
                      : prepare.looked
                        ? "The run finished without saving signatures. Try again."
                        : "Signatures ready. Looking them up…"}
            </p>
          )}
          {!prepare.run && workerIdleNote(status.data) && (
            <p className="land-footnote">{workerIdleNote(status.data)}</p>
          )}
          {prepare.error && (
            <p className="land-error" role="alert">
              {prepare.error}
            </p>
          )}
        </div>
      )}

      {raster && signature && (
        <p className="land-footnote" data-testid="show-once-scenes">
          Scenes {sceneLine}
          {year ? ` ${year}` : ""} · Sentinel-2 L2A · {spacing} grid ·{" "}
          {signature.commonCells.toLocaleString()} of{" "}
          {raster.metadata.boundaryCells.toLocaleString()} cells clear in every season
        </p>
      )}
      {coarsened && (
        <p className="land-footnote" data-testid="show-once-coarsened">
          {coarsened} {land.name} is too wide for {spacingLabel(SIGNATURE_REQUEST.resolutionM)}{" "}
          cells in one analysis.
        </p>
      )}

      {raster && !mine && (
        <>
          {done && (
            <p className="land-notice" role="status" data-testid="show-once-done">
              Drafted as a job: {done.areas} area{done.areas === 1 ? "" : "s"} ·{" "}
              {formatAcres(done.acres)} ac on {land.name}. Say what should happen there in the bar
              above the map.
            </p>
          )}
          <div className="land-actions">
            <button
              type="button"
              className="land-primary"
              data-testid="show-once-start"
              disabled={!visible}
              onClick={start}
            >
              <Crosshair size={14} aria-hidden="true" />
              {done ? `${tap} another place` : `${tap} a place`}
            </button>
          </div>
        </>
      )}

      {raster && mine && (
        <div className="show-once__session" data-testid="show-once-session">
          <div
            className="show-once__modes"
            role="group"
            aria-label={`What a ${tap.toLowerCase()} does`}
          >
            <button
              type="button"
              aria-pressed={mode === "compare"}
              data-testid="show-once-mode-compare"
              onClick={() => useShowOnce.getState().setMode("compare")}
            >
              <Crosshair size={14} aria-hidden="true" /> Compare
            </button>
            <button
              type="button"
              aria-pressed={mode === "protect"}
              data-testid="show-once-mode-protect"
              disabled={!result}
              onClick={() => useShowOnce.getState().setMode("protect")}
            >
              <Shield size={14} aria-hidden="true" /> Protect
            </button>
          </div>
          <p className="land-footnote" role="status" data-testid="show-once-prompt">
            {pending
              ? "Comparing that place with every other cell…"
              : mode === "compare"
                ? `${tap} a place inside ${land.name} to compare it with every other cell.`
                : `${tap} a lit patch to protect it from the job. ${tap} it again to bring it back.`}
          </p>
          {message && (
            <p className="land-error" role="alert" data-testid="show-once-message">
              {message}
            </p>
          )}
          {result && stats && (
            <>
              <p className="show-once__label">
                Like the cell at {result.read.seed.cellLatitude.toFixed(5)},{" "}
                {result.read.seed.cellLongitude.toFixed(5)}
              </p>
              <SeedSeasons seed={result.read.seed} scenes={result.read.scenes} />
              <div className="show-once__slider">
                <span aria-hidden="true">How alike</span>
                <GlassSlider
                  aria-label="How alike"
                  value={threshold}
                  min={ALIKE_MIN}
                  max={ALIKE_MAX}
                  step={0.01}
                  onValueChange={(value) => useShowOnce.getState().setThreshold(value)}
                />
                <span className="show-once__num" data-testid="show-once-threshold">
                  {Math.round(threshold * 100)}%
                </span>
              </div>
              <p className="show-once__tally" role="status" data-testid="show-once-tally">
                <strong>
                  {stats.cells.toLocaleString()} cell{stats.cells === 1 ? "" : "s"} · ≈
                  {formatAcres(stats.acres)} ac
                </strong>{" "}
                lit, of {result.read.comparedCells.toLocaleString()} compared
              </p>
              <p className="show-once__legend">
                <span className="show-once__ramp" aria-hidden="true" /> Lit cells, deeper where more
                alike
              </p>
              {stats.protectedCells > 0 && (
                <p className="show-once__protected" data-testid="show-once-protected">
                  <span className="show-once__hatch" aria-hidden="true" /> Protected, not part of
                  this job: {stats.protectedCells.toLocaleString()} cell
                  {stats.protectedCells === 1 ? "" : "s"} · ≈{formatAcres(stats.protectedAcres)} ac
                  <button
                    type="button"
                    className="show-once__link"
                    onClick={() => useShowOnce.getState().setGuarded(null)}
                  >
                    <Eraser size={12} aria-hidden="true" /> Free all
                  </button>
                </p>
              )}
              <p className="land-footnote">
                Counted on the map&apos;s grid, so ≈. The job is measured on the {spacing} analysis
                grid, and patches under half an acre are left out of it.
              </p>
            </>
          )}
          <div className="land-actions">
            {result && (
              <button
                type="button"
                className="land-primary"
                data-testid="show-once-make-job"
                disabled={making || !stats || stats.cells === 0}
                onClick={() => void make()}
              >
                {making ? "Outlining…" : "Make this the job"}
              </button>
            )}
            <button type="button" data-testid="show-once-clear" onClick={stop}>
              <X size={14} aria-hidden="true" /> {result ? "Clear" : "Cancel"}
            </button>
          </div>
          {jobError && (
            <p className="land-error" role="alert" data-testid="show-once-job-error">
              {jobError}
            </p>
          )}
        </div>
      )}

      <p className="land-footnote show-once__caveat" data-testid="show-once-caveat">
        <strong>Inferred</strong> · Similar seasonal satellite signature
        {spacing ? ` at ${spacing}` : ""}. Not a species ID. Alike is relative to the variety on
        this land.
      </p>
    </section>
  );
}

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import type { Footprint, LandArea } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { useLandContext } from "@/state/landContext";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { beginInvestigation } from "../../../researchApi";
import { landError } from "../../landFlow";
import {
  parsePendingRun,
  parseStoredDraft,
  requestSignature,
  settingsProblem,
  solarDraftKey,
  solarRequest,
  withZone,
  type PendingRun,
  type SolarPreview,
  type SolarRequest,
} from "./solarDraft";

/** An array being planned: its request, and the corners of an outline being drawn. */
export interface SolarDraftState {
  request: SolarRequest;
  points: number[][];
  /** The request holds a finished array outline (not the placeholder of a first sketch). */
  hasZone: boolean;
  /** Corners are being placed on the map. */
  sketching: boolean;
}

/** The investigation title of every hourly run (`SolarStudy` used the same). */
export const solarRunTitle = (year: number) => `Hourly solar assessment · ${year}`;
export const SOLAR_RUN_TITLE = /^Hourly solar assessment/;

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* Recovery is a convenience; the server keeps every accepted run. */
  }
}

function restore(key: string, land: LandArea) {
  const raw = readStorage(key);
  if (!raw) return { draft: null, notice: null, unreadable: false };
  try {
    const stored = parseStoredDraft(raw);
    const draft: SolarDraftState = {
      request: stored.request,
      points: stored.points,
      hasZone: stored.hasZone,
      sketching: stored.points.length > 0 || !stored.hasZone,
    };
    return {
      draft,
      notice:
        stored.revision && stored.revision !== land.revision
          ? "The land's outline changed since you drew this array. Check it still fits."
          : null,
      unreadable: false,
    };
  } catch {
    return { draft: null, notice: null, unreadable: true };
  }
}

/**
 * The array being planned in this lens: drawing its outline on the map, its settings, the
 * instant size check (the preview endpoint) and starting the hourly model. The draft lives in
 * this browser (`…:solar:<landId>`), so switching lenses or reloading picks it up again.
 */
export function useSolarDraft(land: LandArea) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const scene = useScene();
  const cache = useQueryClient();
  const key = solarDraftKey(scope, land.id);
  const owner = `solar-draft:${land.id}`;
  const [initial] = useState(() => restore(key, land));
  const [draft, setDraft] = useState<SolarDraftState | null>(initial.draft);
  const [notice, setNotice] = useState<string | null>(initial.notice);
  const [unreadable, setUnreadable] = useState(initial.unreadable);
  const [picking, setPicking] = useState(false);
  const [review, setReview] = useState<{ signature: string; value: SolarPreview } | null>(null);
  const [reviewError, setReviewError] = useState<{ signature: string; message: string } | null>(
    null,
  );
  const [storageError, setStorageError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pickTicket = useRef(0);
  const runTicket = useRef(0);
  const operation = useRef<PendingRun | null>(null);

  const request = draft?.request ?? null;
  const signature = request ? requestSignature(request, land.revision) : "";
  const zoneReady = Boolean(draft && draft.hasZone && !draft.sketching);
  const problem = request && zoneReady ? settingsProblem(request) : null;
  const previewable = Boolean(request && zoneReady && !problem && canEdit && ready);
  const preview = review?.signature === signature ? review.value : null;
  const previewError = reviewError?.signature === signature ? reviewError.message : null;
  const checking = previewable && !preview && !previewError;

  // Keep the draft for recovery. An unreadable one stays until the person discards it.
  const persisted =
    draft && (draft.hasZone || draft.points.length)
      ? JSON.stringify({
          revision: land.revision,
          request: draft.request,
          points: draft.points,
          hasZone: draft.hasZone,
        })
      : null;
  useEffect(() => {
    if (!canEdit || unreadable) return;
    let message: string | null = null;
    try {
      if (persisted) localStorage.setItem(key, persisted);
      else localStorage.removeItem(key);
    } catch {
      message = "This browser could not keep the sketch. Keep this page open until the model runs.";
    }
    let active = true;
    queueMicrotask(() => {
      if (active) setStorageError(message);
    });
    return () => {
      active = false;
    };
  }, [persisted, key, canEdit, unreadable]);

  // The instant size check: the server confirms the outline fits the land and the panels fit
  // the outline, and reports the capacity. It is a calculation, not an analysis run.
  useEffect(() => {
    if (!previewable || !request) return;
    if (review?.signature === signature || reviewError?.signature === signature) return;
    let live = true;
    const timer = setTimeout(() => {
      void unwrap(
        api.POST("/api/v1/land/{land_id}/solar-assessments/preview", {
          params: { path: { land_id: land.id } },
          body: request,
        }),
      )
        .then((value) => {
          if (live) setReview({ signature, value });
        })
        .catch((cause: unknown) => {
          if (live) setReviewError({ signature, message: landError(cause) });
        });
    }, 350);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [previewable, request, signature, review, reviewError, land.id]);

  // Leaving the lens leaves the map alone: a pending corner pick is cancelled.
  useEffect(
    () => () => {
      pickTicket.current += 1;
      runTicket.current += 1;
      if (useLandContext.getState().pointPicker === owner) {
        scene?.areas.cancelPick();
        useLandContext.getState().setPointPicker(null);
      }
    },
    [owner, scene],
  );

  function stopPicking() {
    pickTicket.current += 1;
    setPicking(false);
    if (useLandContext.getState().pointPicker === owner) {
      scene?.areas.cancelPick();
      useLandContext.getState().setPointPicker(null);
    }
  }

  /** Each click on the map adds a corner until Done, Cancel or Escape. */
  async function pickCorners() {
    if (!scene) return;
    const ticket = ++pickTicket.current;
    setPicking(true);
    useLandContext.getState().setPointPicker(owner);
    for (;;) {
      const point = await scene.areas.pickGround();
      if (ticket !== pickTicket.current) return;
      if (!point) break;
      setDraft((current) =>
        current && current.points.length < 1000
          ? { ...current, points: [...current.points, [point.longitude, point.latitude]] }
          : current,
      );
    }
    setPicking(false);
    if (useLandContext.getState().pointPicker === owner)
      useLandContext.getState().setPointPicker(null);
  }

  const reset = () => {
    setError(null);
    setNotice(null);
    setUnreadable(false);
  };

  /** "Sketch an array": a fresh outline, drawn corner by corner on the map. */
  const begin = () => {
    reset();
    operation.current = null;
    writeStorage(`${key}:operation`, null);
    setDraft({ request: solarRequest(land.boundary), points: [], hasZone: false, sketching: true });
    void pickCorners();
  };

  /** Draws the outline again, keeping the settings. */
  const redraw = () => {
    if (!draft) return;
    setError(null);
    setDraft({ ...draft, points: [], sketching: true });
    void pickCorners();
  };

  const keepDrawing = () => void pickCorners();

  const undoCorner = () =>
    setDraft((current) => current && { ...current, points: current.points.slice(0, -1) });

  /** Uses an outline from anywhere: the drawn corners, the whole land, a mapped shape, a file. */
  const setZone = (zone: Footprint, zoneBasis: string) => {
    stopPicking();
    setError(null);
    setDraft((current) => ({
      request: withZone(current?.request ?? solarRequest(zone), zone, zoneBasis),
      points: [],
      hasZone: true,
      sketching: false,
    }));
  };

  const finishSketch = () => {
    const corners = draft?.points ?? [];
    const first = corners[0];
    if (corners.length < 3 || !first) return;
    setZone(
      { type: "Polygon", coordinates: [[...corners, first]] },
      "Drawn on the map; check it on site.",
    );
  };

  const cancelSketch = () => {
    stopPicking();
    setDraft((current) => (current?.hasZone ? { ...current, points: [], sketching: false } : null));
  };

  const updateRequest = (next: SolarRequest) =>
    setDraft((current) => current && { ...current, request: next });

  const discard = () => {
    stopPicking();
    runTicket.current += 1;
    setBusy(false);
    reset();
    setDraft(null);
  };

  /** Starts a new plan from an earlier request (a saved assessment, a failed run). */
  const load = (next: SolarRequest, message: string | null = null) => {
    stopPicking();
    reset();
    operation.current = null;
    writeStorage(`${key}:operation`, null);
    setNotice(message);
    setDraft({ request: next, points: [], hasZone: true, sketching: false });
  };

  /**
   * Runs the hourly model. The investigation and request key are remembered before the run is
   * queued, so a retry after a lost response reuses them and the worker runs it once.
   */
  const run = async () => {
    if (!request || !preview || !canEdit || !ready || busy) return;
    const ticket = ++runTicket.current;
    setBusy(true);
    setError(null);
    try {
      let pending = operation.current ?? parsePendingRun(readStorage(`${key}:operation`));
      if (pending?.signature !== signature) {
        const investigation = await beginInvestigation(land, solarRunTitle(request.year));
        pending = { signature, id: investigation.id, key: crypto.randomUUID() };
        operation.current = pending;
      }
      writeStorage(`${key}:run`, pending.id);
      writeStorage(`${key}:operation`, JSON.stringify(pending));
      useLandContext.getState().selectSolar(null);
      await unwrap(
        api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
          params: { path: { investigation_id: pending.id } },
          body: {
            requestKey: pending.key,
            kind: "solar",
            question: `Calculate hourly generation for the mapped array in ${request.year}.`,
            analysis: request,
          },
        }),
      );
      operation.current = null;
      writeStorage(`${key}:operation`, null);
      if (ticket === runTicket.current) {
        setDraft(null);
        setNotice(null);
      }
      await cache.invalidateQueries({ queryKey: ["land-research", scope] });
    } catch (cause) {
      if (ticket === runTicket.current) setError(landError(cause));
    } finally {
      if (ticket === runTicket.current) setBusy(false);
    }
  };

  return {
    draft,
    request,
    picking,
    zoneReady,
    problem,
    preview,
    previewError,
    checking,
    /** Panel capacity from the settings, in kW DC (the preview's own formula). */
    capacityKw: request ? request.moduleAreaM2 * request.moduleEfficiency : Number.NaN,
    busy,
    error,
    notice,
    unreadable,
    storageError,
    canDraw: Boolean(scene),
    owner,
    begin,
    redraw,
    keepDrawing,
    undoCorner,
    finishSketch,
    cancelSketch,
    setZone,
    updateRequest,
    discard,
    load,
    run,
    dismissNotice: () => setNotice(null),
  };
}

export type SolarDraft = ReturnType<typeof useSolarDraft>;

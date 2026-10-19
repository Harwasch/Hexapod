import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import type { components, Footprint, LandArea } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { useLandContext } from "@/state/landContext";
import { useLandAccessReady, useLandCanEdit } from "@/state/landIdentity";

import { invalidateLandDerived } from "../../../landDerived";
import { parseSurveyCorners, parseSurveyDraft } from "../../../surveyDraft";
import { landError } from "../../landFlow";

export type SurveyDraft = components["schemas"]["SurveyCreate"];
export type SurveyPlot = components["schemas"]["SurveyPlot"];
export type SurveyRead = components["schemas"]["SurveyRead"];
export type SurveySummary = components["schemas"]["SurveySummary"];
export type Stratum = SurveyDraft["assessedStrata"][number];

export const STRATA: readonly Stratum[] = ["canopy", "shrub", "herb", "ground", "aquatic"];
export const PAGE = 20;
/** Survey files people import are limited to this size, as the server's request is. */
export const MAX_SURVEY_BYTES = 2_000_000;

export function freshSurvey(land: LandArea): SurveyDraft {
  return {
    requestKey: crypto.randomUUID(),
    name: "Field survey",
    boundaryRevision: land.revision,
    observedOn: new Date().toISOString().slice(0, 10),
    observer: "",
    method: "visual-cover",
    design: "purposive",
    assessedStrata: ["herb"],
    methodNotes: "",
    plots: [],
  };
}

export function downloadJson(value: unknown, filename: string): void {
  const url = URL.createObjectURL(
    new Blob([typeof value === "string" ? value : JSON.stringify(value, null, 2)], {
      type: "application/json",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** All of a survey's plots as one shape, for outlining and framing them together. */
export function plotsShape(plots: { boundary: Footprint }[]): Footprint | null {
  const coordinates = plots.flatMap((plot) =>
    plot.boundary.type === "Polygon" ? [plot.boundary.coordinates] : plot.boundary.coordinates,
  );
  return coordinates.length ? { type: "MultiPolygon", coordinates } : null;
}

/** The saved field surveys (paged), and the one selected anywhere in Land. */
export function useSurveyCatalog(land: LandArea, scope: string) {
  const ready = useLandAccessReady();
  const [offset, setOffset] = useState(0);
  const selectedId = useLandContext((s) => s.selectedSurveyId);
  const catalog = useQuery({
    queryKey: ["land-surveys", scope, land.id, offset, land.revision],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/surveys", {
          params: { path: { land_id: land.id }, query: { limit: PAGE, offset } },
        }),
      ),
    retry: false,
  });
  const listed = catalog.data?.find((survey) => survey.id === selectedId) ?? null;
  // A survey cited elsewhere (evidence, a restoration target) may be on another page.
  const single = useQuery({
    queryKey: ["land-survey", scope, land.id, selectedId, land.revision],
    enabled: ready && Boolean(selectedId) && !listed && !catalog.isPending,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/surveys/{survey_id}", {
          params: { path: { land_id: land.id, survey_id: selectedId ?? "" } },
        }),
      ),
    retry: false,
  });
  const fetched = single.data?.landId === land.id ? single.data : null;
  return {
    surveys: catalog.data ?? [],
    loading: catalog.isPending,
    error: catalog.isError,
    retry: () => void catalog.refetch(),
    offset,
    setOffset,
    selectedId,
    selected: selectedId ? (listed ?? fetched) : null,
    selectedError: single.isError,
    select: (id: string | null) => useLandContext.getState().selectSurvey(id),
  };
}

export type SurveyCatalog = ReturnType<typeof useSurveyCatalog>;

/**
 * An unfinished field survey: its fields, the plots drawn so far and the corners of the plot
 * being drawn, kept in this browser (same keys as before) until it is saved or discarded.
 * Corners are placed by clicking the map: drawing keeps asking for the next click until the
 * plot is finished, Escape is pressed or the lens closes.
 */
export function useSurveyDraft(land: LandArea, scope: string) {
  const scene = useScene();
  const cache = useQueryClient();
  const canEdit = useLandCanEdit();
  const owner = `survey:${land.id}`;
  const key = `living-world-land-draft:${encodeURIComponent(scope)}:survey:${land.id}`;
  const [draft, setDraft] = useState<SurveyDraft | null>(null);
  const [recovery, setRecovery] = useState<string | null>(() => {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  });
  const [corners, storeCorners] = useState<number[][]>(() => {
    try {
      return parseSurveyCorners(localStorage.getItem(`${key}:corners`));
    } catch {
      return [];
    }
  });
  const [review, setReview] = useState<{ signature: string; summary: SurveySummary } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [storageError, setStorageError] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const ticket = useRef(0);
  const pickTicket = useRef(0);
  const cornersRef = useRef(corners);
  const signature = JSON.stringify(draft);

  const setCorners = (change: number[][] | ((previous: number[][]) => number[][])) => {
    const next = typeof change === "function" ? change(cornersRef.current) : change;
    cornersRef.current = next;
    storeCorners(next);
    try {
      if (next.length) localStorage.setItem(`${key}:corners`, JSON.stringify(next));
      else localStorage.removeItem(`${key}:corners`);
    } catch {
      setStorageError(true);
    }
  };

  const releasePick = () => {
    pickTicket.current += 1;
    if (useLandContext.getState().pointPicker === owner) {
      scene?.areas.cancelPick();
      useLandContext.getState().setPointPicker(null);
    }
  };

  // Closing the lens (or switching land) stops any click-to-place in progress.
  useEffect(
    () => () => {
      ticket.current += 1;
      pickTicket.current += 1;
      if (useLandContext.getState().pointPicker === owner) {
        scene?.areas.cancelPick();
        useLandContext.getState().setPointPicker(null);
      }
    },
    [owner, scene],
  );

  const update = (value: SurveyDraft | null) => {
    ticket.current += 1;
    setBusy(false);
    setError(null);
    setReview(null);
    setDraft(value);
    try {
      if (value) localStorage.setItem(key, JSON.stringify(value));
      else {
        localStorage.removeItem(key);
        localStorage.removeItem(`${key}:corners`);
      }
      setStorageError(false);
      setRecovery(null);
    } catch {
      setStorageError(true);
    }
  };

  const change = (value: Partial<SurveyDraft>) => {
    if (draft) update({ ...draft, ...value });
  };

  const updatePlot = (index: number, value: Partial<SurveyPlot>) => {
    if (draft)
      update({
        ...draft,
        plots: draft.plots.map((plot, i) => (i === index ? { ...plot, ...value } : plot)),
      });
  };

  const addPlot = (boundary: Footprint) => {
    if (!draft) return;
    const taken = new Set(draft.plots.map((plot) => plot.label.trim().toLowerCase()));
    let number = draft.plots.length + 1;
    while (taken.has(`plot ${number}`)) number += 1;
    update({
      ...draft,
      plots: [
        ...draft.plots,
        {
          label: `Plot ${number}`,
          boundary,
          completeInventory: false,
          samplePoints: null,
          observations: [],
        },
      ],
    });
  };

  const removePlot = (index: number) => {
    if (draft) update({ ...draft, plots: draft.plots.filter((_, i) => i !== index) });
  };

  /** Changing the method clears every number, so no value is read in the wrong unit. */
  const setMethod = (method: SurveyDraft["method"]) => {
    if (!draft) return;
    update({
      ...draft,
      method,
      plots: draft.plots.map((plot) => ({
        ...plot,
        samplePoints: null,
        observations: plot.observations.map((observation) => ({
          ...observation,
          percentCover: null,
          hits: null,
        })),
      })),
    });
  };

  const draw = async () => {
    if (!scene || !draft || !canEdit) return;
    const current = ++pickTicket.current;
    setDrawing(true);
    useLandContext.getState().setPointPicker(owner);
    try {
      for (;;) {
        const point = await scene.areas.pickGround();
        if (current !== pickTicket.current) return;
        if (!point) break;
        setCorners((previous) =>
          previous.length >= 1000 ? previous : [...previous, [point.longitude, point.latitude]],
        );
      }
    } finally {
      if (current === pickTicket.current) {
        setDrawing(false);
        if (useLandContext.getState().pointPicker === owner)
          useLandContext.getState().setPointPicker(null);
      }
    }
  };

  const stopDrawing = () => {
    releasePick();
    setDrawing(false);
  };

  /** Closes the corners into a plot; returns its index, or null with fewer than 3 corners. */
  const finishPlot = (): number | null => {
    const points = cornersRef.current;
    const first = points[0];
    if (!draft || points.length < 3 || !first) return null;
    stopDrawing();
    addPlot({ type: "Polygon", coordinates: [[...points, first]] });
    setCorners([]);
    return draft.plots.length;
  };

  const preview = async (): Promise<boolean> => {
    if (!draft) return false;
    const mine = ++ticket.current;
    const body = draft;
    const signed = JSON.stringify(body);
    setBusy(true);
    setError(null);
    try {
      const summary = await unwrap(
        api.POST("/api/v1/land/{land_id}/surveys/preview", {
          params: { path: { land_id: land.id } },
          body,
        }),
      );
      if (mine !== ticket.current) return false;
      setReview({ signature: signed, summary });
      return true;
    } catch (cause) {
      if (mine === ticket.current) setError(landError(cause));
      return false;
    } finally {
      if (mine === ticket.current) setBusy(false);
    }
  };

  const save = async () => {
    if (!draft || !canEdit) return;
    const mine = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      const result = await unwrap(
        api.POST("/api/v1/land/{land_id}/surveys", {
          params: { path: { land_id: land.id } },
          body: draft,
        }),
      );
      if (mine === ticket.current) {
        stopDrawing();
        update(null);
        setCorners([]);
        useLandContext.getState().selectSurvey(result.id);
      }
      // A dated field survey counts for the whole land: the fog over it is asked again.
      void invalidateLandDerived(cache, land.id);
      await cache.invalidateQueries({ queryKey: ["land-surveys", scope, land.id] });
    } catch (cause) {
      if (mine === ticket.current) setError(landError(cause));
    } finally {
      if (mine === ticket.current) setBusy(false);
    }
  };

  /** Checks an exported survey against this land before it becomes the draft. */
  const restore = async (raw: string) => {
    const mine = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      if (raw.length > MAX_SURVEY_BYTES) throw new Error("Survey files are limited to 2 MB.");
      const value = parseSurveyDraft(raw);
      const normalized: SurveyDraft = {
        ...value,
        requestKey: crypto.randomUUID(),
        boundaryRevision: land.revision,
        supersedesId: null,
      };
      await unwrap(
        api.POST("/api/v1/land/{land_id}/surveys/preview", {
          params: { path: { land_id: land.id } },
          body: normalized,
        }),
      );
      if (mine === ticket.current) update(normalized);
    } catch (cause) {
      if (mine === ticket.current) setError(landError(cause));
    } finally {
      if (mine === ticket.current) setBusy(false);
    }
  };

  const importFile = (file: File) => {
    if (file.size > MAX_SURVEY_BYTES) {
      setError("Survey files are limited to 2 MB.");
      return;
    }
    void file
      .text()
      .then(restore)
      .catch((cause: unknown) => setError(landError(cause)));
  };

  const resume = () => {
    if (!recovery) return;
    try {
      setDraft(parseSurveyDraft(recovery));
      setRecovery(null);
    } catch {
      setError("The saved draft could not be read. Export it before discarding.");
    }
  };

  const discard = () => {
    stopDrawing();
    setCorners([]);
    update(null);
  };

  return {
    draft,
    recovery,
    corners,
    drawing,
    review: review?.signature === signature ? review.summary : null,
    busy,
    error,
    storageError,
    canDraw: Boolean(scene) && canEdit,
    begin: () => {
      useLandContext.getState().selectSurvey(null);
      update(freshSurvey(land));
    },
    correct: (survey: SurveyRead) =>
      update({
        ...parseSurveyDraft(JSON.stringify({ ...survey, requestKey: crypto.randomUUID() })),
        requestKey: crypto.randomUUID(),
        supersedesId: survey.id,
      }),
    change,
    setMethod,
    updatePlot,
    addPlot,
    removePlot,
    draw: () => void draw(),
    stopDrawing,
    undoCorner: () => setCorners((previous) => previous.slice(0, -1)),
    finishPlot,
    preview,
    save: () => void save(),
    importFile,
    resume,
    discard,
    exportDraft: () => {
      if (draft) downloadJson(draft, "survey-draft.json");
      else if (recovery) downloadJson(recovery, "survey-draft-recovery.json");
    },
  };
}

export type SurveyDraftState = ReturnType<typeof useSurveyDraft>;

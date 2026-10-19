import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import type { LandArea } from "@twin/contracts";

import { ApiError, api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { landError } from "../../landFlow";
import type { SolarAssessment } from "./solarDraft";
import {
  economicsDraftKey,
  economicsProblem,
  parseEconomicsDraft,
  sameScenario,
  serializeEconomics,
  solarDefaults,
  solarFinance,
  type EconomicsDraft,
  type Scenario,
  type ScenarioCreate,
  type ScenarioEdit,
  type ScenarioResult,
  type SolarInputs,
} from "./solarEconomics";
import { powerLabel } from "./solarWords";

function restore(key: string, landId: string) {
  let raw: string | null;
  try {
    raw = localStorage.getItem(key);
  } catch {
    return { draft: null, unreadable: null };
  }
  if (!raw) return { draft: null, unreadable: null };
  try {
    return { draft: parseEconomicsDraft(raw, landId), unreadable: null };
  } catch {
    return { draft: null, unreadable: raw };
  }
}

/**
 * Costs and payback for a solar array: the solar scenario. The draft is kept in this browser
 * (`…:solar-scenario:<landId>`); the result is recalculated as values change; saving keeps the
 * request key, so a save whose answer was lost never makes a second scenario.
 */
export function useSolarEconomics(land: LandArea) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const key = economicsDraftKey(scope, land.id);
  const listKey = ["land-scenarios", scope, land.id];
  const [initial] = useState(() => restore(key, land.id));
  const [draft, setDraft] = useState<EconomicsDraft | null>(initial.draft);
  const [unreadable, setUnreadable] = useState<string | null>(initial.unreadable);
  const [result, setResult] = useState<{ signature: string; value: ScenarioResult } | null>(null);
  const [calcError, setCalcError] = useState<{ signature: string; message: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [conflict, setConflict] = useState<Scenario | null>(null);
  const [storageError, setStorageError] = useState<string | null>(null);
  const ticket = useRef(0);

  const payload = draft?.payload ?? null;
  const signature = payload ? serializeEconomics(payload) : "";
  const problem = payload ? economicsProblem(payload) : null;
  const calculable = Boolean(payload && !problem && ready && canEdit);
  const shown = result?.signature === signature ? result.value : null;
  const shownError = calcError?.signature === signature ? calcError.message : null;

  const serialized = draft ? serializeEconomics(draft) : null;
  useEffect(() => {
    if (!canEdit || unreadable) return;
    let message: string | null = null;
    try {
      if (serialized && serialized.length > 2_000_000) throw new Error("Too large to keep");
      if (serialized) localStorage.setItem(key, serialized);
      else localStorage.removeItem(key);
    } catch {
      message = "This browser could not keep these costs. Download a copy before leaving.";
    }
    let active = true;
    queueMicrotask(() => {
      if (active) setStorageError(message);
    });
    return () => {
      active = false;
    };
  }, [serialized, key, canEdit, unreadable]);

  // A picked-up draft may already be saved (the answer to a save was lost): look it up once.
  const restored = initial.draft;
  useEffect(() => {
    if (!restored || !ready) return;
    let live = true;
    void (async () => {
      try {
        const response = await api.GET("/api/v1/land/{land_id}/scenarios/requests/{request_key}", {
          params: { path: { land_id: land.id, request_key: restored.requestKey } },
        });
        if (!live || response.response.status === 404) return;
        const found = await unwrap(Promise.resolve(response));
        if (!live) return;
        if (sameScenario(restored.payload, found.saved)) {
          setDraft((current) => (current?.requestKey === restored.requestKey ? null : current));
          setNotice("These costs were already saved.");
          await cache.invalidateQueries({ queryKey: ["land-scenarios", scope, land.id] });
          return;
        }
        setDraft((current) =>
          current?.requestKey === restored.requestKey
            ? {
                ...current,
                requestKey: crypto.randomUUID(),
                editing: {
                  id: found.saved.id,
                  revision: found.saved.revision,
                  landId: found.saved.landId,
                },
              }
            : current,
        );
        if (found.current.revision !== found.saved.revision) setConflict(found.current);
      } catch {
        /* The draft stays; saving reports any problem. */
      }
    })();
    return () => {
      live = false;
    };
  }, [restored, ready, land.id, scope, cache]);

  // Results follow the values: a short pause after typing, then the server's calculation.
  useEffect(() => {
    if (!calculable || !payload || !draft) return;
    if (result?.signature === signature || calcError?.signature === signature) return;
    let live = true;
    const body = { ...payload, requestKey: draft.requestKey };
    const timer = setTimeout(() => {
      void unwrap(
        api.POST("/api/v1/land/{land_id}/scenarios/preview", {
          params: { path: { land_id: land.id } },
          body,
        }),
      )
        .then((value) => {
          if (live) setResult({ signature, value });
        })
        .catch((cause: unknown) => {
          if (live) setCalcError({ signature, message: landError(cause) });
        });
    }, 450);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [calculable, payload, draft, signature, result, calcError, land.id]);

  const start = (next: ScenarioCreate, editing: ScenarioEdit | null) => {
    ticket.current += 1;
    setSaving(false);
    setUnreadable(null);
    setConflict(null);
    setError(null);
    setNotice(null);
    setDraft({
      version: 1,
      landId: land.id,
      requestKey: crypto.randomUUID(),
      payload: next,
      editing,
    });
  };

  /** Costs for a complete hourly assessment: its energy, editable money values. */
  const fromAssessment = (assessment: SolarAssessment) =>
    start(
      {
        name: `${powerLabel(assessment.metadata.capacityKwDc)} array · ${assessment.request.year} weather`,
        boundaryRevision: assessment.boundaryRevision,
        inputs: solarFinance(assessment),
        evidenceIds: [],
        fieldSurveyIds: [],
        solarAssessmentId: assessment.id,
      },
      null,
    );

  /** Costs from entered sunlight and panel area, without an hourly model. */
  const manual = () =>
    start(
      {
        name: "Solar estimate",
        boundaryRevision: land.revision,
        inputs: solarDefaults(),
        evidenceIds: [],
        fieldSurveyIds: [],
        solarAssessmentId: null,
      },
      null,
    );

  const edit = (scenario: Scenario) => {
    if (scenario.inputs.kind !== "solar") return;
    start(
      {
        name: scenario.name,
        boundaryRevision: scenario.boundaryRevision,
        inputs: scenario.inputs,
        evidenceIds: scenario.evidenceIds ?? [],
        fieldSurveyIds: scenario.fieldSurveyIds ?? [],
        solarAssessmentId: scenario.solarAssessmentId ?? null,
      },
      { id: scenario.id, revision: scenario.revision, landId: scenario.landId },
    );
  };

  const update = (changes: Partial<ScenarioCreate>) =>
    setDraft((current) => current && { ...current, payload: { ...current.payload, ...changes } });

  const setInputs = (inputs: SolarInputs) => update({ inputs });

  const cancel = () => {
    ticket.current += 1;
    setSaving(false);
    setDraft(null);
    setConflict(null);
    setError(null);
  };

  const save = async (): Promise<Scenario | null> => {
    if (!draft || !payload || conflict || !shown || saving) return null;
    const current = ++ticket.current;
    setSaving(true);
    setError(null);
    try {
      const saved = draft.editing
        ? await unwrap(
            api.PUT("/api/v1/land/{land_id}/scenarios/{scenario_id}", {
              params: { path: { land_id: land.id, scenario_id: draft.editing.id } },
              body: {
                ...payload,
                requestKey: draft.requestKey,
                expectedRevision: draft.editing.revision,
              },
            }),
          )
        : await unwrap(
            api.POST("/api/v1/land/{land_id}/scenarios", {
              params: { path: { land_id: land.id } },
              body: { ...payload, requestKey: draft.requestKey },
            }),
          );
      if (current === ticket.current) {
        setDraft(null);
        setNotice(null);
      }
      await cache.invalidateQueries({ queryKey: listKey });
      return saved;
    } catch (cause) {
      if (current !== ticket.current) return null;
      setError(landError(cause));
      if (cause instanceof ApiError && cause.status === 409) {
        try {
          if (draft.editing) {
            const latest = await unwrap(
              api.GET("/api/v1/land/{land_id}/scenarios/{scenario_id}", {
                params: { path: { land_id: land.id, scenario_id: draft.editing.id } },
              }),
            );
            if (current === ticket.current) setConflict(latest);
          } else {
            const found = await unwrap(
              api.GET("/api/v1/land/{land_id}/scenarios/requests/{request_key}", {
                params: { path: { land_id: land.id, request_key: draft.requestKey } },
              }),
            );
            if (current !== ticket.current) return null;
            if (sameScenario(payload, found.saved)) {
              setDraft(null);
              setError(null);
              setNotice("These costs were already saved.");
              await cache.invalidateQueries({ queryKey: listKey });
              return found.current;
            }
            setDraft((value) => value && { ...value, requestKey: crypto.randomUUID() });
            setError("An earlier save used this request. Save again to keep these as new costs.");
          }
        } catch {
          /* The original error and the local draft remain. */
        }
      }
      return null;
    } finally {
      if (current === ticket.current) setSaving(false);
    }
  };

  return {
    draft,
    payload,
    problem,
    result: shown,
    calcError: shownError,
    calculating: calculable && !shown && !shownError,
    saving,
    error,
    notice,
    conflict,
    unreadable,
    storageError,
    canEdit: canEdit && ready,
    fromAssessment,
    manual,
    edit,
    update,
    setInputs,
    cancel,
    save,
    /** Leaves the linked assessment: the physical values become editable. */
    unlink: () => update({ solarAssessmentId: null }),
    pinCurrentBoundary: () => update({ boundaryRevision: land.revision }),
    keepAsNew: () => {
      setDraft((value) => value && { ...value, editing: null, requestKey: crypto.randomUUID() });
      setConflict(null);
      setError(null);
    },
    loadLatest: () => {
      if (conflict) edit(conflict);
    },
    discardUnreadable: () => setUnreadable(null),
    dismissNotice: () => setNotice(null),
  };
}

export type SolarEconomicsState = ReturnType<typeof useSolarEconomics>;

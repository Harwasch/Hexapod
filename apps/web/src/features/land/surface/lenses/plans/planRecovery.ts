import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import type { LandArea } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandCanEdit } from "@/state/landIdentity";

import {
  actionDraftKey,
  actionEdit,
  downloadActionDraft,
  parseActionDraft,
  sameAction,
  serializeActionDraft,
  type ActionEdit,
} from "../../../actionDraft";
import {
  downloadScenarioDraft,
  parseScenarioDraft,
  sameScenario,
  scenarioDraftKey,
  serializeScenarioDraft,
  type ScenarioEdit,
} from "../../../scenarioDraft";
import type { ScenarioInputs } from "../../../scenarioDefaults";
import type { Action, ActionDraft, Scenario } from "./planModel";

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/** A plan opened in the editor: what to edit, which revision it revises, and any clash. */
export interface PlanSeed {
  draft: ActionDraft;
  editing: ActionEdit | null;
  conflict: Action | null;
  notice: string | null;
  /** Opened from a copy kept in this browser (so leaving keeps it, changed or not). */
  recovered: boolean;
}

/** A restoration scenario opened in the editor. */
export interface ScenarioSeed {
  name: string;
  inputs: ScenarioInputs;
  boundaryRevision: number;
  evidenceIds: string[];
  fieldSurveyIds: string[];
  solarAssessmentId: string | null;
  monitoringText: string;
  requestKey: string;
  editing: ScenarioEdit | null;
  conflict: Scenario | null;
  notice: string | null;
  recovered: boolean;
}

/**
 * Keeps an open draft in this browser while it is edited, so closing the lens or the tab never
 * loses it. Returns a message when the browser refused, and a way to clear the copy.
 */
export function useDraftAutosave(key: string, serialized: string | null, limit: number) {
  const canEdit = useLandCanEdit();
  const [storageError, setStorageError] = useState<string | null>(null);
  useEffect(() => {
    if (!serialized || !canEdit) return;
    let message: string | null = null;
    try {
      if (serialized.length > limit) throw new Error("Draft exceeds the recovery limit");
      localStorage.setItem(key, serialized);
    } catch {
      message = "This browser could not keep a copy of this draft. Download one before leaving.";
    }
    let active = true;
    queueMicrotask(() => {
      if (active) setStorageError(message);
    });
    return () => {
      active = false;
    };
  }, [serialized, canEdit, key, limit]);
  const clear = () => {
    try {
      localStorage.removeItem(key);
      setStorageError(null);
    } catch {
      setStorageError("The browser could not remove the saved copy. It may reappear after reload.");
    }
  };
  return { storageError, clear };
}

/**
 * An unfinished plan kept in this browser: recover it (checking whether it was in fact saved,
 * and whether someone changed the plan since), recover it as a new plan, download or discard.
 * Recovery never approves or schedules anything.
 */
export function usePlanRecovery(
  land: LandArea,
  scope: string,
  open: (seed: PlanSeed) => void,
  showSaved: (id: string) => void,
) {
  const cache = useQueryClient();
  const key = actionDraftKey(scope, land.id);
  const [recovery, setRecovery] = useState<string | null>(() => read(key));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const ticket = useRef(0);
  useEffect(
    () => () => {
      ticket.current += 1;
    },
    [],
  );
  const discard = () => {
    try {
      localStorage.removeItem(key);
      setRecovery(null);
      setError(null);
    } catch {
      setError("The browser could not remove the saved draft. It may reappear after reload.");
    }
  };
  const recover = async (asNew = false) => {
    if (!recovery) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const saved = parseActionDraft(recovery, land.id);
      let baseline = asNew ? null : saved.editing;
      let latest: Action | null = null;
      let value = asNew ? { ...saved.draft, requestKey: crypto.randomUUID() } : saved.draft;
      if (!asNew) {
        const response = await api.GET("/api/v1/land/{land_id}/actions/requests/{request_key}", {
          params: { path: { land_id: land.id, request_key: saved.draft.requestKey ?? "" } },
        });
        if (response.response.status !== 404) {
          const found = await unwrap(Promise.resolve(response));
          if (ticket.current !== current) return;
          if (sameAction(saved.draft, found.saved)) {
            cache.setQueryData(
              ["land-action-record", scope, land.id, found.current.id],
              found.current,
            );
            discard();
            showSaved(found.current.id);
            setNotice(
              "This draft was already saved. Opened the plan without approving or scheduling it.",
            );
            await cache.invalidateQueries({ queryKey: ["land-actions", scope, land.id] });
            return;
          }
          baseline = actionEdit(found.saved);
          latest = found.current;
          value = { ...saved.draft, requestKey: crypto.randomUUID() };
        }
        if (baseline && !latest)
          latest = await unwrap(
            api.GET("/api/v1/land/{land_id}/actions/{action_id}", {
              params: { path: { land_id: land.id, action_id: baseline.id } },
            }),
          );
      }
      if (ticket.current !== current) return;
      setRecovery(null);
      open({
        draft: value,
        editing: baseline,
        conflict:
          latest &&
          baseline &&
          (latest.revision !== baseline.revision ||
            latest.status !== baseline.status ||
            latest.missionId !== baseline.missionId)
            ? latest
            : null,
        notice: "Recovered your draft. Check the steps and areas before saving.",
        recovered: true,
      });
    } catch (cause) {
      if (ticket.current === current) setError(describeError(cause));
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  };
  return {
    recovery,
    busy,
    error,
    notice,
    clearNotice: () => setNotice(null),
    recover,
    discard,
    download: () => recovery && downloadActionDraft(recovery),
    /** Reads the browser copy again (after an editor closes and keeps its draft). */
    reload: () => setRecovery(read(key)),
  };
}

/** Whether a scenario draft in this browser belongs to the Solar lens. */
function draftKind(text: string, landId: string): "restoration" | "solar" | "unreadable" {
  try {
    return parseScenarioDraft(text, landId).payload.inputs.kind;
  } catch {
    return "unreadable";
  }
}

/**
 * An unfinished scenario kept in this browser. Restoration drafts (and unreadable ones, so
 * they can still be downloaded) are recovered here; a solar draft belongs to the Solar lens.
 */
export function useScenarioRecovery(
  land: LandArea,
  scope: string,
  open: (seed: ScenarioSeed) => void,
  showSaved: (id: string) => void,
) {
  const cache = useQueryClient();
  const key = scenarioDraftKey(scope, land.id);
  const [recovery, setRecovery] = useState<string | null>(() => read(key));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const ticket = useRef(0);
  useEffect(
    () => () => {
      ticket.current += 1;
    },
    [],
  );
  const kind = recovery ? draftKind(recovery, land.id) : null;
  const discard = () => {
    try {
      localStorage.removeItem(key);
      setRecovery(null);
      setError(null);
    } catch {
      setError("The browser could not remove the saved draft. It may reappear after reload.");
    }
  };
  const recover = async (asNew = false) => {
    if (!recovery) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const saved = parseScenarioDraft(recovery, land.id);
      let baseline: ScenarioEdit | null = asNew ? null : saved.editing;
      let latest: Scenario | null = null;
      let nextKey = asNew ? crypto.randomUUID() : saved.requestKey;
      if (!asNew) {
        const response = await api.GET("/api/v1/land/{land_id}/scenarios/requests/{request_key}", {
          params: { path: { land_id: land.id, request_key: saved.requestKey } },
        });
        if (response.response.status !== 404) {
          const found = await unwrap(Promise.resolve(response));
          if (ticket.current !== current) return;
          if (sameScenario(saved.payload, found.saved)) {
            discard();
            showSaved(found.current.id);
            setNotice("This draft was already saved. Opened the scenario without saving it again.");
            await cache.invalidateQueries({ queryKey: ["land-scenarios", scope, land.id] });
            return;
          }
          baseline = { id: found.saved.id, revision: found.saved.revision, landId: land.id };
          latest = found.current;
          nextKey = crypto.randomUUID();
        }
        if (baseline && !latest)
          latest = await unwrap(
            api.GET("/api/v1/land/{land_id}/scenarios/{scenario_id}", {
              params: { path: { land_id: land.id, scenario_id: baseline.id } },
            }),
          );
      }
      if (ticket.current !== current) return;
      setRecovery(null);
      open({
        name: saved.payload.name,
        inputs: saved.payload.inputs,
        boundaryRevision: saved.payload.boundaryRevision,
        evidenceIds: saved.payload.evidenceIds ?? [],
        fieldSurveyIds: saved.payload.fieldSurveyIds ?? [],
        solarAssessmentId: saved.payload.solarAssessmentId ?? null,
        monitoringText: saved.monitoringText,
        requestKey: nextKey,
        editing: baseline,
        conflict: latest && latest.revision !== baseline?.revision ? latest : null,
        notice: "Recovered your assumptions. Calculate again to see the results before saving.",
        recovered: true,
      });
    } catch (cause) {
      if (ticket.current === current) setError(describeError(cause));
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  };
  return {
    recovery,
    kind,
    busy,
    error,
    notice,
    clearNotice: () => setNotice(null),
    recover,
    discard,
    download: () => recovery && downloadScenarioDraft(recovery),
    reload: () => setRecovery(read(key)),
  };
}

export { serializeActionDraft, serializeScenarioDraft };

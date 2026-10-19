import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import type { LandArea } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { landScope, useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { beginInvestigation, startResearch } from "../../researchApi";
import {
  parseResearchDraft,
  researchDraftKey,
  type CapturedResearchRequest,
  type QuestionBudget,
  type ResearchDraft,
} from "../../researchDraft";

export const DEFAULT_BUDGET: QuestionBudget = {
  maxSteps: 16,
  maxSeconds: 180,
  maxOutputTokens: 12000,
  maxWebSearches: 6,
};

/**
 * Asking the agent: the question and its map-feature focus (shared Land context, so "Ask
 * about this" from anywhere fills them), the research limits, and the draft kept in this
 * browser so a reload neither loses the question nor sends it twice.
 *
 * A request is written down before it is sent. If the answer to the send is lost, the card
 * offers to retry that exact request (same investigation, same key), which the server runs
 * at most once. `followUp` is the investigation a question continues; null starts a new one.
 */
export function useAgentAsk(land: LandArea, followUp: string | null) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const question = useLandContext((state) => state.researchQuestion);
  const focus = useLandContext((state) => state.researchFocus);
  const storageKey = researchDraftKey(scope, land.id);
  const lifetime = useRef(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    lifetime.current += 1;
    return () => {
      lifetime.current += 1;
    };
  }, [land.id, scope]);

  const [initial] = useState(() => {
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(storageKey);
      const draft = raw ? parseResearchDraft(raw, land.id) : null;
      const context = useLandContext.getState();
      const apply =
        !context.researchQuestion ||
        (draft?.question === context.researchQuestion &&
          JSON.stringify(draft.focus) === JSON.stringify(context.researchFocus));
      return { draft, raw, apply, error: null, storageFailed: false };
    } catch (cause) {
      return {
        draft: null,
        raw,
        apply: false,
        error: raw ? describeError(cause) : null,
        storageFailed: !raw,
      };
    }
  });
  const [storageError, setStorageError] = useState(initial.storageFailed);
  const [recoveryError, setRecoveryError] = useState(initial.error);
  const [recovered, setRecovered] = useState<number | null>(
    initial.draft?.question && !useLandContext.getState().researchQuestion
      ? initial.draft.boundaryRevision
      : null,
  );
  const [budget, setBudget] = useState<QuestionBudget>(
    (initial.apply ? initial.draft?.budget : null) ?? DEFAULT_BUDGET,
  );
  const [pending, setPending] = useState<CapturedResearchRequest | null>(
    initial.draft?.pending ?? null,
  );
  const pendingRef = useRef(pending);
  const capture = (value: CapturedResearchRequest | null) => {
    pendingRef.current = value;
    setPending(value);
  };

  // Restore a saved question once, unless the person already has one in hand.
  useEffect(() => {
    const context = useLandContext.getState();
    if (initial.draft && !context.researchQuestion && !context.researchFocus) {
      context.setResearchQuestion(initial.draft.question);
      context.setResearchFocus(initial.draft.focus);
      if (initial.draft.investigationId) context.selectInvestigation(initial.draft.investigationId);
    }
  }, [initial]);

  const snapshot = (pendingValue: CapturedResearchRequest | null): ResearchDraft => {
    const context = useLandContext.getState();
    return {
      version: 1,
      landId: land.id,
      boundaryRevision: land.revision,
      question: context.researchQuestion,
      focus: context.researchFocus,
      budget,
      newTopic: followUp === null,
      investigationId: followUp,
      pending: pendingValue,
    };
  };

  // Keep the draft in step with what is on screen. An unreadable record is never overwritten
  // until the person discards it.
  useEffect(() => {
    if (recoveryError || landScope() !== scope || useLand.getState().active?.id !== land.id) return;
    const draft = snapshot(pendingRef.current);
    let saved = true;
    try {
      if (!draft.question && !draft.focus && !draft.pending) localStorage.removeItem(storageKey);
      else localStorage.setItem(storageKey, JSON.stringify(draft));
    } catch {
      saved = false;
    }
    let active = true;
    queueMicrotask(() => {
      if (active) setStorageError(!saved);
    });
    return () => {
      active = false;
    };
    // `snapshot` reads the question and focus through the store; these are what it depends on.
  }, [
    question,
    focus,
    followUp,
    budget,
    pending,
    land.id,
    land.revision,
    storageKey,
    scope,
    recoveryError,
  ]);

  const refresh = () => cache.invalidateQueries({ queryKey: ["land-research", scope] });

  /** Sends the question (or retries a captured request). Resolves to the investigation id. */
  const ask = async (captured?: CapturedResearchRequest): Promise<string | null> => {
    if (!canEdit || !ready || recoveryError || busy || (!captured && (!question.trim() || pending)))
      return null;
    const session = useLand.getState().session;
    const generation = lifetime.current;
    const current = () =>
      lifetime.current === generation &&
      landScope() === scope &&
      useLand.getState().session === session;
    setBusy(true);
    setError(null);
    try {
      let operation = captured;
      const targetId = operation?.investigationId ?? followUp;
      const target = targetId
        ? (
            await unwrap(
              api.GET("/api/v1/research/investigations/{investigation_id}", {
                params: { path: { investigation_id: targetId }, query: { limit: 1 } },
              }),
            )
          ).investigation
        : null;
      if (!current()) return null;
      if (
        target &&
        (target.landId !== land.id ||
          (operation && operation.boundaryRevision !== target.boundaryRevision))
      )
        throw new Error(
          "This captured investigation does not match the selected land and boundary revision.",
        );
      if (!operation) {
        const investigation = target ?? (await beginInvestigation(land, question));
        if (!current()) return null;
        operation = {
          investigationId: investigation.id,
          boundaryRevision: investigation.boundaryRevision,
          question,
          key: crypto.randomUUID(),
          budget,
          focus,
        };
      }
      capture(operation);
      // Write the exact request down before sending it: a retry never spends a new key.
      try {
        localStorage.setItem(storageKey, JSON.stringify(snapshot(operation)));
        setStorageError(false);
      } catch {
        setStorageError(true);
      }
      const reference = operation.focus
        ? { artifactId: operation.focus.artifactId, featureIndex: operation.focus.featureIndex }
        : null;
      await startResearch(
        operation.investigationId,
        operation.question,
        "investigation",
        operation.key,
        operation.budget,
        reference,
      );
      if (!current()) return null;
      const context = useLandContext.getState();
      context.selectInvestigation(operation.investigationId);
      // Clear the box only if it still holds what was sent; a newer question stays.
      if (
        context.researchQuestion === operation.question &&
        JSON.stringify(context.researchFocus) === JSON.stringify(operation.focus)
      ) {
        context.setResearchQuestion("");
        context.setResearchFocus(null);
      }
      capture(null);
      setRecovered(null);
      await refresh();
      return operation.investigationId;
    } catch (cause) {
      if (current()) setError(describeError(cause));
      return null;
    } finally {
      if (lifetime.current === generation) setBusy(false);
    }
  };

  const discardUnreadable = () => {
    try {
      localStorage.removeItem(storageKey);
      setRecoveryError(null);
    } catch {
      setStorageError(true);
    }
  };

  const downloadUnreadable = () => {
    const url = URL.createObjectURL(new Blob([initial.raw ?? ""], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "land-question-recovery.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return {
    question,
    setQuestion: (value: string) => useLandContext.getState().setResearchQuestion(value),
    focus,
    clearFocus: () => useLandContext.getState().setResearchFocus(null),
    budget,
    setBudget,
    pending,
    discardPending: () => capture(null),
    ask,
    busy,
    error,
    clearError: () => setError(null),
    canAsk: canEdit && ready,
    canEdit,
    ready,
    storageError,
    recoveryError,
    discardUnreadable,
    downloadUnreadable,
    recovered,
    dismissRecovered: () => setRecovered(null),
  };
}

export type AgentAsk = ReturnType<typeof useAgentAsk>;

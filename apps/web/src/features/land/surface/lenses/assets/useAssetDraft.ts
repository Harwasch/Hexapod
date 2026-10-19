import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import type { components, LandArea } from "@twin/contracts";

import { api, ApiError, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandCanEdit } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";

import { assetKeys, useRefreshAssets } from "./assetsApi";
import {
  assetDraftKey,
  parseInventoryDraft,
  sameAssetContent,
  type InventoryDraftSnapshot,
} from "./assetsDraft";
import type { InventoryShape } from "./assetsGeometry";
import type { Asset } from "./assetsModel";

export type FeatureDraft = components["schemas"]["LandFeatureCreate"];

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/**
 * An asset being added or changed: the draft, the saved revision it started from, a newer
 * saved revision to reconcile, and a recovery copy in this browser so nothing typed is lost.
 * Saving reuses the draft's request key, so a retried save never creates a second asset.
 */
export function useAssetDraft(
  land: LandArea,
  scope: string,
  onFrame?: (draft: FeatureDraft) => void,
) {
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const refreshAssets = useRefreshAssets(land);
  const storage = assetDraftKey(scope, land.id);
  const [draft, setDraftState] = useState<FeatureDraft | null>(null);
  const [editing, setEditing] = useState<Asset | null>(null);
  const [revisionNote, setRevisionNote] = useState("");
  const [working, setWorking] = useState<InventoryShape | null>(null);
  const [geometryEditing, setGeometryEditing] = useState(false);
  const [concurrent, setConcurrent] = useState<Asset | null>(null);
  const [recovery, setRecovery] = useState<string | null>(() => readStorage(storage));
  const [storageError, setStorageError] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  /** Changes whenever the draft is replaced from outside the form (new, recovered, merged). */
  const [version, setVersion] = useState(0);
  const ticket = useRef(0);
  useEffect(
    () => () => {
      ticket.current++;
    },
    [],
  );

  useEffect(() => {
    if (!draft) return;
    const value: InventoryDraftSnapshot = {
      version: 1,
      landId: land.id,
      boundaryRevision: land.revision,
      draft,
      editing,
      revisionNote,
      working: geometryEditing ? working : null,
    };
    let failed = false;
    let cancelled = false;
    try {
      localStorage.setItem(storage, JSON.stringify(value));
    } catch {
      failed = true;
    }
    queueMicrotask(() => {
      if (!cancelled) setStorageError(failed);
    });
    return () => {
      cancelled = true;
    };
  }, [draft, editing, revisionNote, working, geometryEditing, storage, land.id, land.revision]);

  function clearRecovery() {
    setRecovery(null);
    setNotice(null);
    try {
      localStorage.removeItem(storage);
      setStorageError(false);
    } catch {
      setStorageError(true);
    }
  }

  function reset() {
    setWorking(null);
    setConcurrent(null);
    setDraftState(null);
    setEditing(null);
    setGeometryEditing(false);
    setRevisionNote("");
  }

  /** Starts a new asset from a drawn, found or picked shape. */
  function startNew(next: FeatureDraft) {
    setEditing(null);
    setConcurrent(null);
    setGeometryEditing(false);
    setWorking(null);
    setRevisionNote("");
    setError(null);
    setNotice(null);
    setDraftState({ requestKey: crypto.randomUUID(), ...next });
    setVersion((value) => value + 1);
  }

  /** Opens a saved asset for changes; saving makes a new revision. */
  function startEdit(asset: Asset) {
    setEditing(asset);
    setConcurrent(null);
    setRevisionNote("");
    setGeometryEditing(false);
    setWorking(null);
    setError(null);
    setNotice(null);
    setDraftState({
      requestKey: crypto.randomUUID(),
      name: asset.name,
      category: asset.category,
      geometry: asset.geometry,
      source: asset.source,
      status: asset.status,
      description: asset.description,
      attributes: asset.attributes,
      evidenceIds: asset.evidenceIds,
      externalRef: asset.externalRef,
    });
    setVersion((value) => value + 1);
  }

  function update(patch: Partial<FeatureDraft>) {
    setDraftState((current) => (current ? { ...current, ...patch } : current));
  }

  function discard() {
    ticket.current++;
    setBusy(false);
    setError(null);
    clearRecovery();
    reset();
  }

  async function recover() {
    if (!recovery || !canEdit) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      const value = parseInventoryDraft(recovery, land.id);
      let baseline = value.editing;
      let latest: Asset | null = null;
      let creationRecorded = false;
      if (baseline)
        latest = await unwrap(
          api.GET("/api/v1/land/{land_id}/features/{feature_id}", {
            params: { path: { land_id: land.id, feature_id: baseline.id } },
          }),
        );
      else {
        try {
          const saved = await unwrap(
            api.GET("/api/v1/land/{land_id}/features/requests/{request_key}", {
              params: { path: { land_id: land.id, request_key: value.draft.requestKey ?? "" } },
            }),
          );
          latest = saved.current;
          creationRecorded = sameAssetContent(saved.original, value.draft);
          baseline = { ...saved.current, ...saved.original, revision: 1 };
        } catch (cause) {
          if (!(cause instanceof ApiError && cause.status === 404)) throw cause;
        }
      }
      if (current !== ticket.current) return;
      if (latest)
        cache.setQueryData(assetKeys.record(scope, land.id, land.revision, latest.id), latest);
      if (
        latest &&
        !value.working &&
        (creationRecorded ||
          (latest.requestKey === value.draft.requestKey && sameAssetContent(latest, value.draft)))
      ) {
        clearRecovery();
        useLandContext.getState().selectInventory(latest.id);
        setNotice("This asset was already saved. It is open below; nothing was saved twice.");
        return;
      }
      setDraftState(value.draft);
      setEditing(baseline);
      setRevisionNote(value.revisionNote);
      setWorking(value.working);
      setGeometryEditing(Boolean(value.working));
      setConcurrent(latest && latest.revision !== baseline?.revision ? latest : null);
      setRecovery(null);
      setVersion((count) => count + 1);
      if (latest) useLandContext.getState().selectInventory(latest.id);
      if (value.boundaryRevision !== land.revision)
        setError("The land's outline changed since this draft. Check where the asset sits.");
      onFrame?.(value.draft);
    } catch (cause) {
      if (current === ticket.current) setError(describeError(cause));
    } finally {
      if (current === ticket.current) setBusy(false);
    }
  }

  async function checkLatest() {
    if (!editing) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      const latest = await unwrap(
        api.GET("/api/v1/land/{land_id}/features/{feature_id}", {
          params: { path: { land_id: land.id, feature_id: editing.id } },
        }),
      );
      if (current !== ticket.current) return;
      cache.setQueryData(assetKeys.record(scope, land.id, land.revision, latest.id), latest);
      void refreshAssets();
      setConcurrent(latest.revision !== editing.revision ? latest : null);
      if (latest.revision === editing.revision) setNotice("You are editing the latest version.");
    } catch (cause) {
      if (current === ticket.current) setError(describeError(cause));
    } finally {
      if (current === ticket.current) setBusy(false);
    }
  }

  function resolveConflict(merged: FeatureDraft, keepWorking: boolean) {
    if (!concurrent) return;
    setDraftState(merged);
    setEditing(concurrent);
    setConcurrent(null);
    setError(null);
    setVersion((value) => value + 1);
    if (!keepWorking) {
      setWorking(null);
      setGeometryEditing(false);
    }
  }

  /** Saves the draft; resolves with the saved asset's id, or null when nothing was saved. */
  async function save(): Promise<string | null> {
    if (!draft || geometryEditing || concurrent || !canEdit || busy) return null;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      const feature = editing
        ? await unwrap(
            api.PUT("/api/v1/land/{land_id}/features/{feature_id}", {
              params: { path: { land_id: land.id, feature_id: editing.id } },
              body: { ...draft, expectedRevision: editing.revision, note: revisionNote.trim() },
            }),
          )
        : await unwrap(
            api.POST("/api/v1/land/{land_id}/features", {
              params: { path: { land_id: land.id } },
              body: draft,
            }),
          );
      if (ticket.current === current) {
        clearRecovery();
        reset();
        useLandContext.getState().selectInventory(feature.id);
      }
      await refreshAssets();
      return feature.id;
    } catch (cause) {
      if (ticket.current === current)
        setError(
          cause instanceof ApiError && cause.fieldErrors.length
            ? cause.fieldErrors.join(". ")
            : describeError(cause),
        );
      return null;
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  }

  return {
    draft,
    editing,
    revisionNote,
    working,
    geometryEditing,
    concurrent,
    recovery,
    storageError,
    error,
    notice,
    busy,
    version,
    startNew,
    startEdit,
    update,
    setRevisionNote,
    setWorking,
    setGeometryEditing,
    setError,
    setNotice,
    discard,
    recover,
    clearRecovery,
    checkLatest,
    resolveConflict,
    save,
  };
}

export type AssetDraftState = ReturnType<typeof useAssetDraft>;

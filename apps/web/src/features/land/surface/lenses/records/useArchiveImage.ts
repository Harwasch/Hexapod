import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import type { components } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";

import { saveBlob } from "./recordsModel";

type Snapshot = components["schemas"]["ArchiveImageRead"];

/**
 * A copy of an archive image saved in this workspace: whether there is one, its private
 * preview (an object URL, revoked on change), and saving or downloading it.
 */
export function useArchiveImage(id: string, hasMedia: boolean) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const key = ["land-archive-images", scope, id];
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const mounted = useRef(true);
  const request = useRef<AbortController | null>(null);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      request.current?.abort();
    };
  }, []);
  const query = useQuery({
    queryKey: key,
    enabled: ready && hasMedia,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}/image", {
          params: { path: { evidence_id: id } },
        }),
      ),
    retry: false,
  });
  const sha = query.data?.sha256;
  useEffect(() => {
    if (!sha || !ready) return;
    const controller = new AbortController();
    let objectUrl: string | null = null;
    void unwrap(
      api.GET("/api/v1/research/evidence/{evidence_id}/image/preview", {
        params: { path: { evidence_id: id } },
        parseAs: "blob",
        signal: controller.signal,
      }),
    )
      .then((blob) => {
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setError(describeError(cause));
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [id, scope, sha, ready, attempt]);

  /** Saves a copy; resolves to it, or null when saving failed (the error is shown). */
  const capture = async (): Promise<Snapshot | null> => {
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError(null);
    try {
      const image = await unwrap(
        api.POST("/api/v1/research/evidence/{evidence_id}/image", {
          params: { path: { evidence_id: id } },
          signal: controller.signal,
        }),
      );
      if (controller.signal.aborted) return null;
      cache.setQueryData(key, image);
      return image;
    } catch (cause) {
      if (!controller.signal.aborted) setError(describeError(cause));
      return null;
    } finally {
      if (mounted.current) setBusy(false);
    }
  };

  const download = async () => {
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError(null);
    try {
      const blob = await unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}/image/original", {
          params: { path: { evidence_id: id } },
          parseAs: "blob",
          signal: controller.signal,
        }),
      );
      if (controller.signal.aborted) return;
      const type = query.data?.sourceMediaType;
      saveBlob(
        blob,
        `archive-${id}.${type === "image/jpeg" ? "jpg" : type === "image/webp" ? "webp" : "png"}`,
      );
    } catch (cause) {
      if (!controller.signal.aborted) setError(describeError(cause));
    } finally {
      if (mounted.current) setBusy(false);
    }
  };

  return {
    url: sha && ready ? url : null,
    metadata: query.data ?? null,
    busy,
    error,
    canEdit: ready && canEdit,
    statusError: query.isError,
    statusPending: query.isPending && ready && hasMedia,
    retry: () => {
      setError(null);
      setAttempt((value) => value + 1);
      void query.refetch();
    },
    capture,
    download,
  };
}

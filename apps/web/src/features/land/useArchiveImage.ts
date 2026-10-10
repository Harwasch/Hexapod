import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";

export function useArchiveImage(id: string, hasMedia: boolean) {
  const scope = useLandScope(),
    ready = useLandAccessReady(),
    canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const key = ["land-archive-images", scope, id];
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [retry, setRetry] = useState(0);
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
  }, [id, scope, sha, ready, retry]);
  const capture = async () => {
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
      if (!controller.signal.aborted) cache.setQueryData(key, image);
    } catch (cause) {
      if (!controller.signal.aborted) setError(describeError(cause));
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
      const href = URL.createObjectURL(blob),
        anchor = document.createElement("a");
      const extension =
        query.data?.sourceMediaType === "image/jpeg"
          ? "jpg"
          : query.data?.sourceMediaType === "image/webp"
            ? "webp"
            : "png";
      anchor.href = href;
      anchor.download = `archive-${id}.${extension}`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(href), 1000);
    } catch (cause) {
      if (!controller.signal.aborted) setError(describeError(cause));
    } finally {
      if (mounted.current) setBusy(false);
    }
  };
  return {
    url: sha && ready ? url : null,
    metadata: query.data,
    busy,
    error,
    canEdit: ready && canEdit,
    statusError: query.isError,
    statusPending: query.isPending,
    retry: () => {
      setError(null);
      setRetry((value) => value + 1);
      void query.refetch();
    },
    capture,
    download,
  };
}

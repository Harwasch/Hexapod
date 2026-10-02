import { useState } from "react";

import type { CameraBookmark } from "@twin/contracts";

import { useCreateBookmark, useDeleteBookmark, useSite } from "@/api/queries";
import { describeError } from "@/lib/log";
import { readJson, writeJson } from "@/lib/storage";
import { useMission } from "@/state/mission";
import { useSites } from "@/state/sites";
import { useViewer } from "@/state/viewer";

/** A view saved in this browser: when there is no site to save it to, or the API is down. */
export interface LocalView {
  id: string;
  name: string;
  longitude: number;
  latitude: number;
  height: number;
  heading: number;
  pitch: number;
  roll: number;
  isDefault: boolean;
  siteId: string;
  createdAt: string;
}

export type SavedView = CameraBookmark | LocalView;

export const LOCAL_VIEWS_KEY = "twin.views.v1";

/** Local views for one site ("" is the views saved with no site), oldest first. */
export function localViewsFor(views: readonly LocalView[], siteId: string | null): LocalView[] {
  return views.filter((view) => view.siteId === (siteId ?? ""));
}

/**
 * The current site's saved camera views — the site switcher's second list.
 *
 * "Current" is the site the camera is at, or else the site the project was last on (the
 * project follows the last site visited, and so does the switcher's name). Views are saved to
 * the site through the API when it answers, and in this browser otherwise.
 */
export function useSavedViews() {
  const activeSiteId = useSites((s) => s.activeSiteId);
  const projectSiteId = useMission((s) => s.project?.siteId ?? null);
  const siteId = activeSiteId ?? projectSiteId;
  const site = useSite(siteId);
  const pose = useViewer((s) => s.camera);
  const create = useCreateBookmark(siteId);
  const remove = useDeleteBookmark(siteId);
  const [local, setLocal] = useState<LocalView[]>(
    () => (readJson(LOCAL_VIEWS_KEY) as LocalView[] | undefined) ?? [],
  );
  const [error, setError] = useState<string | null>(null);
  const remote = Boolean(siteId) && !site.builtin && Boolean(site.data);

  const writeLocal = (next: LocalView[]) => {
    setLocal(next);
    writeJson(LOCAL_VIEWS_KEY, next);
  };

  const save = async (name: string): Promise<boolean> => {
    const label = name.trim() || `View ${new Date().toLocaleTimeString()}`;
    const body = {
      name: label,
      longitude: pose.longitude,
      latitude: pose.latitude,
      height: pose.height,
      heading: pose.heading,
      pitch: pose.pitch,
      roll: pose.roll,
      isDefault: false,
    };
    setError(null);
    if (remote) {
      try {
        await create.mutateAsync(body);
      } catch (err) {
        setError(describeError(err));
        return false;
      }
      return true;
    }
    writeLocal([
      ...local,
      {
        ...body,
        id: `local-${Date.now()}`,
        siteId: siteId ?? "",
        createdAt: new Date().toISOString(),
      },
    ]);
    return true;
  };

  const removeView = (view: SavedView) => {
    if (view.id.startsWith("local-")) writeLocal(local.filter((v) => v.id !== view.id));
    else remove.mutate(view.id);
  };

  const views: SavedView[] = [
    ...(site.data?.id === siteId ? (site.data?.cameraBookmarks ?? []) : []),
    ...localViewsFor(local, siteId),
  ];

  return {
    siteId,
    views,
    save,
    remove: removeView,
    saving: create.isPending,
    error,
    /** Where a new view goes, in words for the form's hint. */
    destination: remote ? "site" : "browser",
  } as const;
}

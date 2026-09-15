import { Bookmark, Camera, Trash2 } from "lucide-react";
import { useState } from "react";

import { EmptyState, GlassButton, GlassField, GlassInput, useFieldId } from "@twin/ui";

import { useCreateBookmark, useDeleteBookmark, useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { readJson, writeJson } from "@/lib/storage";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";
import { useViewer } from "@/state/viewer";

import { FloatingPanel } from "../shell/FloatingPanel";

interface LocalView {
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

const LOCAL_KEY = "twin.views.v1";

/** Saved camera views: persisted to the active site when the API is available, locally otherwise. */
export function BookmarksPanel() {
  const scene = useScene();
  const open = useUi((s) => s.activePanel === "bookmarks");
  const setPanel = useUi((s) => s.setPanel);
  const activeSiteId = useSites((s) => s.activeSiteId);
  const site = useSite(activeSiteId);
  const pose = useViewer((s) => s.camera);
  const create = useCreateBookmark(activeSiteId);
  const remove = useDeleteBookmark(activeSiteId);
  const [name, setName] = useState("");
  const [local, setLocal] = useState<LocalView[]>(
    () => (readJson(LOCAL_KEY) as LocalView[] | undefined) ?? [],
  );
  const [error, setError] = useState<string | null>(null);
  const nameId = useFieldId("view-name");
  const canPersistRemotely = Boolean(activeSiteId) && !site.builtin;

  const save = async () => {
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
    if (canPersistRemotely) {
      try {
        await create.mutateAsync(body);
      } catch (err) {
        setError(describeError(err));
        return;
      }
    } else {
      const next = [
        ...local,
        {
          ...body,
          id: `local-${Date.now()}`,
          siteId: activeSiteId ?? "",
          createdAt: new Date().toISOString(),
        },
      ];
      setLocal(next);
      writeJson(LOCAL_KEY, next);
    }
    setName("");
  };

  const views = [...(site.data?.cameraBookmarks ?? []), ...local];

  return (
    <FloatingPanel
      open={open}
      title="Bookmarks"
      onClose={() => setPanel(null)}
      testId="bookmarks-panel"
    >
      <div className="glass-stack">
        <form
          className="form"
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          <GlassField
            label="Save current view"
            htmlFor={nameId}
            hint={
              canPersistRemotely
                ? `Saved to ${site.data?.name ?? "the active site"}`
                : "Saved in this browser (no active site or API offline)"
            }
            error={error}
          >
            <div className="glass-row">
              <GlassInput
                id={nameId}
                placeholder="View name"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
              <GlassButton
                type="submit"
                variant="primary"
                loading={create.isPending}
                leadingIcon={<Camera size={14} aria-hidden="true" />}
                data-testid="save-view"
              >
                Save
              </GlassButton>
            </div>
          </GlassField>
        </form>
        {views.length === 0 ? (
          <EmptyState
            icon={<Bookmark size={26} />}
            title="No saved views"
            body="Save the current camera to return to it later."
          />
        ) : (
          <ul className="glass-list" aria-label="Saved views">
            {views.map((view) => (
              <li key={view.id} className="card">
                <div className="card__row card__row--between">
                  <button
                    type="button"
                    className="card__title"
                    style={{
                      background: "none",
                      border: 0,
                      padding: 0,
                      cursor: "pointer",
                      color: "inherit",
                      textAlign: "left",
                    }}
                    onClick={() => scene?.camera.flyToBookmark(view)}
                  >
                    {view.name}
                    {view.isDefault && <span className="glass-subtle"> · default</span>}
                  </button>
                  <GlassButton
                    iconOnly
                    size="sm"
                    variant="ghost"
                    aria-label={`Delete ${view.name}`}
                    onClick={() => {
                      if (view.id.startsWith("local-")) {
                        const next = local.filter((v) => v.id !== view.id);
                        setLocal(next);
                        writeJson(LOCAL_KEY, next);
                      } else {
                        remove.mutate(view.id);
                      }
                    }}
                  >
                    <Trash2 size={14} aria-hidden="true" />
                  </GlassButton>
                </div>
                <div className="card__meta glass-mono">
                  <span>
                    {view.latitude.toFixed(5)}, {view.longitude.toFixed(5)}
                  </span>
                  <span>{Math.round(view.height)} m</span>
                  <span>
                    {Math.round(view.heading)}° / {Math.round(view.pitch)}°
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </FloatingPanel>
  );
}

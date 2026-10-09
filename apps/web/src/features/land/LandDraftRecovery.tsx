import { useEffect, useRef, useState } from "react";
import type { LandCreate } from "@twin/contracts";
import { ApiError, api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLand, type LandSketch } from "@/state/land";
import { landScope } from "@/state/landIdentity";
import { useUi } from "@/state/ui";
import { importBoundary } from "./geometry";

const PREFIX = "living-world-land-draft:";
interface SavedDraft {
  version: 3;
  draft: LandCreate | null;
  sketch: LandSketch | null;
  activeId: string | null;
  revision: number | null;
  savedAt: string;
}

function readDraft(key: string): SavedDraft | null {
  try {
    const text = localStorage.getItem(key);
    if (!text || text.length > 4_000_000) return null;
    const value = JSON.parse(text) as Omit<SavedDraft, "version"> & { version: number };
    if (![1, 2, 3].includes(value.version)) return null;
    if (value.draft) {
      if (typeof value.draft.name !== "string" || !value.draft.source) return null;
      importBoundary(JSON.stringify(value.draft.boundary));
    }
    const sketch = value.version >= 2 ? value.sketch : null;
    if (sketch) {
      if (
        !["draw", "corridor", "split"].includes(sketch.mode) ||
        (sketch.mode === "split" && !value.draft) ||
        (sketch.operation != null &&
          (sketch.mode !== "draw" ||
            !value.draft ||
            !["union", "difference", "intersection"].includes(sketch.operation))) ||
        !Array.isArray(sketch.points) ||
        !sketch.points.length ||
        sketch.points.length > 2000 ||
        sketch.points.some(
          (point) =>
            !Array.isArray(point) ||
            point.length !== 2 ||
            !Number.isFinite(point[0]) ||
            !Number.isFinite(point[1]) ||
            Math.abs(point[0]) > 180 ||
            Math.abs(point[1]) > 90,
        ) ||
        !["ft", "m"].includes(sketch.unit)
      )
        return null;
      // An unfinished numeric edit must not discard the drawn points.
      if (!Number.isFinite(sketch.width)) sketch.width = 0;
    }
    if (!value.draft && !sketch) return null;
    if (value.activeId != null && typeof value.activeId !== "string") return null;
    if (value.activeId && (!Number.isInteger(value.revision) || (value.revision ?? 0) < 1))
      return null;
    return { ...value, version: 3, draft: value.draft ?? null, sketch: sketch ?? null };
  } catch {
    return null;
  }
}

/** One recoverable draft per identity/workspace. Tokens are never persisted here. */
export function LandDraftRecovery({ scope }: { scope: string }) {
  const key = `${PREFIX}${encodeURIComponent(scope)}`;
  const scene = useScene();
  const [saved, setSaved] = useState(() => readDraft(key));
  const pending = useRef(Boolean(saved));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [conflict, setConflict] = useState(false);
  useEffect(() => {
    const store = () => {
      if (pending.current || landScope() !== scope) return;
      const { draft, active, mode, points, corridorWidth, corridorUnit, boundaryOperation } =
        useLand.getState();
      const sketch: LandSketch | null =
        (mode === "draw" || mode === "corridor" || mode === "split") && points.length
          ? { mode, points, width: corridorWidth, unit: corridorUnit, operation: boundaryOperation }
          : null;
      try {
        if (!draft && !sketch) localStorage.removeItem(key);
        else
          localStorage.setItem(
            key,
            JSON.stringify({
              version: 3,
              draft,
              sketch,
              activeId: active?.id ?? null,
              revision: active?.revision ?? null,
              savedAt: new Date().toISOString(),
            } satisfies SavedDraft),
          );
      } catch {
        /* A denied/quota-limited browser store must not break editing. */
      }
    };
    const off = useLand.subscribe((next, previous) => {
      if (
        next.draft !== previous.draft ||
        next.points !== previous.points ||
        next.mode !== previous.mode ||
        next.corridorWidth !== previous.corridorWidth ||
        next.boundaryOperation !== previous.boundaryOperation ||
        next.corridorUnit !== previous.corridorUnit
      ) {
        // A deliberate new edit replaces the previous recoverable draft.
        if (
          (next.draft && next.draft !== previous.draft) ||
          ((next.mode === "draw" || next.mode === "corridor" || next.mode === "split") &&
            next.points.length &&
            next.points !== previous.points)
        ) {
          pending.current = false;
          setSaved(null);
        }
        store();
      }
    });
    window.addEventListener("pagehide", store);
    return () => {
      store();
      off();
      window.removeEventListener("pagehide", store);
    };
  }, [key, scope]);
  if (!saved) return null;
  const restore = async (asNew = false) => {
    setBusy(true);
    setError(null);
    const session = useLand.getState().session;
    try {
      const active =
        saved.activeId && !asNew
          ? await unwrap(
              api.GET("/api/v1/land/{land_id}", { params: { path: { land_id: saved.activeId } } }),
            )
          : null;
      if (landScope() !== scope || session !== useLand.getState().session) return;
      if (active && active.revision !== saved.revision) {
        setConflict(true);
        setError(
          "The saved land has changed since this draft. Restore as a separate area to keep both boundaries.",
        );
        return;
      }
      // Keep the stored copy until the complete restored state is in place.
      if (active) useLand.getState().select(active);
      else useLand.getState().clear();
      if (saved.sketch) {
        useUi.getState().setMeasureMode(null);
        useUi.getState().setExploreMode(false);
        scene?.areas.cancelPick();
        scene?.areas.edit(null);
        useLand.getState().restoreSketch(saved.sketch, saved.draft);
        const xs = saved.sketch.points.map(([x]) => x),
          ys = saved.sketch.points.map(([, y]) => y);
        const west = Math.min(...xs),
          east = Math.max(...xs),
          south = Math.min(...ys),
          north = Math.max(...ys);
        const dx = Math.max(0.001, (east - west) * 0.4),
          dy = Math.max(0.001, (north - south) * 0.4);
        scene?.camera.flyToRectangle(
          Math.max(-180, west - dx),
          Math.max(-90, south - dy),
          Math.min(180, east + dx),
          Math.min(90, north + dy),
        );
      } else if (saved.draft) useLand.getState().propose(saved.draft);
      pending.current = false;
      setSaved(null);
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 404) {
        setConflict(true);
        setError(
          "The original land is no longer available. You can restore this draft as a separate area.",
        );
      } else setError(describeError(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="land-import-review" aria-label="Recover boundary draft">
      <strong>
        Resume{" "}
        {saved.sketch
          ? saved.sketch.mode === "split"
            ? "your boundary split"
            : saved.sketch.operation
              ? "your boundary composition"
              : saved.sketch.mode === "corridor"
                ? "your corridor"
                : "your drawing"
          : saved.draft?.name}
      </strong>
      <p>
        {saved.sketch
          ? `${saved.sketch.points.length} drawn points are stored in this browser. They have not been saved to your workspace.`
          : "A boundary draft is stored in this browser. It has not been saved to your workspace."}
      </p>
      {error && <p role="alert">{error}</p>}
      <div className="land-actions">
        <button type="button" disabled={busy} onClick={() => void restore(conflict)}>
          {busy
            ? "Loading…"
            : conflict
              ? "Restore as a separate area"
              : saved.sketch
                ? "Resume drawing"
                : "Resume boundary draft"}
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => {
            pending.current = false;
            try {
              localStorage.removeItem(key);
            } catch {
              /* Browser storage may be unavailable. */
            }
            setSaved(null);
          }}
        >
          Discard stored draft
        </button>
      </div>
    </section>
  );
}

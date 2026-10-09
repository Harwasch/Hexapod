import { useEffect, useRef, useState } from "react";
import type { LandCreate } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLand } from "@/state/land";
import { landScope } from "@/state/landIdentity";
import { importBoundary } from "./geometry";

const PREFIX = "living-world-land-draft:";
interface SavedDraft {
  version: 1;
  draft: LandCreate;
  activeId: string | null;
  revision: number | null;
  savedAt: string;
}

function readDraft(key: string): SavedDraft | null {
  try {
    const text = localStorage.getItem(key);
    if (!text || text.length > 4_000_000) return null;
    const value = JSON.parse(text) as SavedDraft;
    if (
      value.version !== 1 ||
      !value.draft ||
      typeof value.draft.name !== "string" ||
      !value.draft.source
    )
      return null;
    importBoundary(JSON.stringify(value.draft.boundary));
    return value;
  } catch {
    return null;
  }
}

/** One recoverable draft per identity/workspace. Tokens are never persisted here. */
export function LandDraftRecovery({ scope }: { scope: string }) {
  const key = `${PREFIX}${encodeURIComponent(scope)}`;
  const [saved, setSaved] = useState(() => readDraft(key));
  const pending = useRef(Boolean(saved));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [conflict, setConflict] = useState(false);
  useEffect(() => {
    const store = () => {
      if (pending.current || landScope() !== scope) return;
      const { draft, active } = useLand.getState();
      try {
        if (!draft) localStorage.removeItem(key);
        else
          localStorage.setItem(
            key,
            JSON.stringify({
              version: 1,
              draft,
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
      if (next.draft !== previous.draft) {
        // A new edit replaces an old recoverable draft deliberately.
        if (next.draft && !previous.draft) {
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
      pending.current = false;
      if (active) useLand.getState().select(active);
      else useLand.getState().clear();
      useLand.getState().propose(saved.draft);
      setSaved(null);
    } catch (failure) {
      setError(describeError(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="land-import-review" aria-label="Recover boundary draft">
      <strong>Resume {saved.draft.name}</strong>
      <p>A boundary draft is stored in this browser. It has not been saved to your workspace.</p>
      {error && <p role="alert">{error}</p>}
      <div className="land-actions">
        <button type="button" disabled={busy} onClick={() => void restore(conflict)}>
          {busy ? "Loading…" : conflict ? "Restore as a separate area" : "Resume boundary draft"}
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => {
            pending.current = false;
            localStorage.removeItem(key);
            setSaved(null);
          }}
        >
          Discard stored draft
        </button>
      </div>
    </section>
  );
}

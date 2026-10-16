import { Focus, Sparkles, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { GlassProgress, GlassSlider } from "@twin/ui";

import { paintedDocOf } from "@/cesium/scanView/scanInstances";
import { embeddingOf, instancesDocOf } from "@/cesium/splatInstances";
import {
  describedMatrix,
  embeddingModelName,
  formatBytes,
  loadEmbeddings,
  rankSimilar,
  withDescendants,
} from "@/lib/instances";
import { hiddenForShowOnly, selectionLabel } from "@/lib/sceneSelect";
import { similarSummary, similarText, useInstances } from "@/state/instances";

/** The slider's range, as cosines: below 0.3 most of a scan matches, above 0.95 nearly nothing. */
const SIMILAR_MIN = 0.3;
const SIMILAR_MAX = 0.95;

type Load =
  { kind: "loading"; received: number; total: number } | { kind: "error"; message: string };

/**
 * "More like this" on the selection card (features/sites/ObjectCard): the objects of the scan
 * whose views look like the chosen one's. Offered only where the scan's objects carry image
 * embeddings (`embeddingOf`: `instances.emb`, docs/SCENE_OBJECTS.md §4).
 *
 * The first press reads the scan's embeddings (43.6 MB for the camp, kept in memory after:
 * lib/payloadCache.ts), saying how much and how far along; then every described object is
 * compared with the chosen one, centred on what the scan's objects share
 * (lib/instances.ts `describedMatrix`), and the matches are lit through the objects store
 * (state/instances.ts `showSimilar`), so every renderer shows them as it shows any highlight.
 * "How alike" moves the bar a match must clear; the count says how many objects that lights
 * and what they mostly are. An object too small in every view to describe is compared by the
 * object it is part of, and the card says so. What it shows is inferred, and the card says
 * that too, whatever the slider: a similar look in this scan's embeddings, never a species or
 * an identification.
 */
export function MoreLikeThis({ assetId, seedId }: { assetId: string; seedId: number }) {
  const source = embeddingOf(assetId);
  const entry = useInstances((s) => s.assets[assetId]);
  const showSimilar = useInstances((s) => s.showSimilar);
  const setThreshold = useInstances((s) => s.setSimilarThreshold);
  const similar = entry?.similar?.seedId === seedId ? entry.similar : null;
  const summary = useMemo(() => (similar ? similarSummary(entry) : null), [entry, similar]);
  const [load, setLoad] = useState<Load | null>(null);
  // Which press is current: a card closed or another object chosen mid-read drops the result.
  const run = useRef(0);
  useEffect(
    () => () => {
      run.current += 1;
    },
    [],
  );

  // The seed's own instance must be in the file: an object painted in this browser is not.
  const doc = instancesDocOf(assetId);
  if (!source || !doc?.byId.has(seedId)) return null;
  const bytes = source.rows * source.dim * 2;
  const model = embeddingModelName(source.model);

  const start = async (): Promise<void> => {
    const mine = ++run.current;
    let shown = -1;
    setLoad({ kind: "loading", received: 0, total: bytes });
    try {
      const raw = await loadEmbeddings(source.url, source.rows, source.dim, (received, total) => {
        // A render per whole percent, not per network chunk.
        const pct = Math.floor((received / Math.max(1, total)) * 100);
        if (run.current !== mine || pct === shown) return;
        shown = pct;
        setLoad({ kind: "loading", received, total });
      });
      if (run.current !== mine) return;
      const ranked = rankSimilar(doc, describedMatrix(raw, source.dim), seedId);
      if (!ranked) {
        setLoad({
          kind: "error",
          message:
            "This object, and what it is part of, were too small in every photo to describe, so there is nothing to compare.",
        });
        return;
      }
      setLoad(null);
      showSimilar(assetId, seedId, ranked.seedUsed, ranked.scores);
    } catch (error) {
      if (run.current !== mine) return;
      const message = error instanceof Error ? error.message : String(error);
      setLoad({
        kind: "error",
        message: `Could not read this scan's object signatures (${message}). Try again.`,
      });
    }
  };

  /** Hides everything but the matches and what was compared, as the card's Show only does. */
  const showOnlyThese = (): void => {
    const store = useInstances.getState();
    const lit = store.assets[assetId]?.highlighted;
    const drawn = paintedDocOf(assetId);
    if (!lit || lit.size === 0 || !drawn) return;
    store.showAll(assetId);
    store.setHidden(assetId, hiddenForShowOnly(drawn, lit), true);
  };

  /** Gives the highlight back to the selection. */
  const clear = (): void => {
    useInstances.getState().highlight(assetId, [...withDescendants(doc, [seedId])]);
  };

  if (!similar) {
    return (
      <div className="mc-similar" data-testid="more-like-this">
        {load?.kind !== "loading" && (
          <div className="mc-actions mc-actions--tools">
            <button
              type="button"
              className="mc-btn"
              data-testid="more-like-this-start"
              onClick={() => void start()}
            >
              <Sparkles size={13} aria-hidden="true" /> More like this
            </button>
          </div>
        )}
        {load?.kind === "loading" && (
          <>
            <p className="mc-card__note" role="status" data-testid="more-like-this-status">
              Reading object signatures · {formatBytes(load.total)}
              {load.received > 0
                ? ` · ${String(Math.floor((load.received / load.total) * 100))}%`
                : ""}
            </p>
            <GlassProgress
              className="mc-progress"
              label="Reading object signatures"
              value={load.received}
              max={load.total}
              valueText={`${formatBytes(load.received)} of ${formatBytes(load.total)}`}
            />
          </>
        )}
        {load?.kind === "error" && (
          <p className="mc-card__note mc-similar__error" role="alert">
            {load.message}
          </p>
        )}
      </div>
    );
  }

  const pct = Math.round(similar.threshold * 100);
  const comparedBy =
    similar.seedUsed !== seedId
      ? selectionLabel(
          paintedDocOf(assetId)?.byId.get(similar.seedUsed) ?? doc.byId.get(similar.seedUsed),
          similar.seedUsed,
          entry?.index.categoryOf.get(similar.seedUsed),
        )
      : null;

  return (
    <div className="mc-similar" data-testid="more-like-this" data-active="">
      <div className="mc-similar__slider">
        <span className="mc-similar__label" aria-hidden="true">
          How alike
        </span>
        <GlassSlider
          aria-label="How alike"
          value={similar.threshold}
          min={SIMILAR_MIN}
          max={SIMILAR_MAX}
          step={0.01}
          onValueChange={(value) => setThreshold(assetId, value)}
        />
        <span className="mc-similar__pct mc-mono" data-testid="more-like-this-threshold">
          {pct}%
        </span>
      </div>
      <p className="mc-similar__count" role="status" data-testid="more-like-this-count">
        {summary ? similarText(summary) : ""}
      </p>
      {comparedBy !== null && (
        <p className="mc-card__note" data-testid="more-like-this-compared-by">
          Compared by {comparedBy}, the object it is part of.
        </p>
      )}
      <div className="mc-actions mc-actions--tools">
        <button type="button" className="mc-btn" onClick={showOnlyThese}>
          <Focus size={13} aria-hidden="true" /> Show only these
        </button>
        <button type="button" className="mc-btn" onClick={clear}>
          <X size={13} aria-hidden="true" /> Clear
        </button>
      </div>
      <p className="mc-card__note mc-similar__inferred" data-testid="more-like-this-inferred">
        <strong>Inferred</strong> · similar look in this scan&apos;s image embeddings ({model}, from
        each object&apos;s views), not a species ID or a verified identification. How alike is
        relative to the other objects in this scan.
      </p>
    </div>
  );
}

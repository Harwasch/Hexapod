import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Undo2, X } from "lucide-react";

import type { components } from "@twin/contracts";
import { GlassButton, GlassPanel } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { landScope } from "@/state/landIdentity";
import { useLandMap } from "@/state/landMap";

import { anchorOf } from "./geometryAnchor";
import { areaLabel } from "./landFlow";

type SplitResult = components["schemas"]["BoundarySplitResult"];
const LAYER = "boundary-split";
const PINS = "split";

/**
 * Split the draft with a line: draw the cut across it, then click the pieces on the map to
 * keep. The original land stays saved until a revision is saved.
 */
export function LandSplitBar() {
  const points = useLand((s) => s.points);
  const draft = useLand((s) => s.draft);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<{
    points: typeof points;
    boundary: NonNullable<typeof draft>["boundary"];
    result: SplitResult;
  } | null>(null);
  const [keep, setKeep] = useState<number[]>([]);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      useLandContext.getState().removeLayer(LAYER);
      useLandMap.getState().clearPins(PINS);
    };
  }, []);
  const current =
    preview?.points === points && preview.boundary === draft?.boundary ? preview.result : null;
  const features = useMemo(
    () =>
      current?.parts.map((part, index) => ({
        id: String(index),
        label: `Part ${index + 1}`,
        geometry: part.boundary,
      })),
    [current],
  );
  useEffect(() => {
    if (!features || !current) {
      useLandContext.getState().removeLayer(LAYER);
      useLandMap.getState().clearPins(PINS);
      return;
    }
    useLandContext.getState().setLayer({
      id: LAYER,
      title: "Pieces after the split",
      selectedIds: keep.map(String),
      features,
    });
    useLandMap.getState().setPins(
      PINS,
      current.parts.map((part, index) => ({
        id: `split-${index}`,
        position: anchorOf(part.boundary),
        label: `${keep.includes(index) ? "Keep · " : ""}Part ${index + 1} · ${areaLabel(part.areaM2)}`,
        tone: keep.includes(index) ? "warn" : "neutral",
        onSelect: () =>
          setKeep((chosen) =>
            chosen.includes(index) ? chosen.filter((id) => id !== index) : [...chosen, index],
          ),
      })),
    );
  }, [features, current, keep]);

  const request = async (apply: boolean) => {
    const state = useLand.getState();
    if (!state.draft || state.points.length < 2) return;
    const { points: cut, draft: base, session } = state;
    const scope = landScope();
    const valid = () =>
      mounted.current &&
      landScope() === scope &&
      useLand.getState().session === session &&
      useLand.getState().points === cut &&
      useLand.getState().draft === base;
    setBusy(true);
    state.setError(null);
    try {
      const result = await unwrap(
        api.POST("/api/v1/land/split", {
          body: {
            boundary: base.boundary,
            coordinates: cut,
            ...(apply ? { keepParts: keep } : {}),
          },
        }),
      );
      if (!valid()) return;
      if (apply && result.selection) {
        state.updateBoundary(result.selection.boundary);
        state.begin("browse");
      } else {
        setPreview({ points: cut, boundary: base.boundary, result });
        setKeep([]);
      }
    } catch (error) {
      if (valid()) state.setError(describeError(error));
    } finally {
      if (mounted.current) setBusy(false);
    }
  };

  return (
    <GlassPanel
      strong
      pill
      className="land-drawbar"
      data-testid="land-drawbar"
      aria-label="Split boundary"
    >
      <span className="land-drawbar__status" aria-live="polite">
        {current ? (
          <>
            Click pieces to keep · <b>{keep.length}</b> of {current.parts.length}
          </>
        ) : points.length ? (
          <>
            <b>{points.length}</b> cut point{points.length === 1 ? "" : "s"}
          </>
        ) : (
          "Draw a line across the land"
        )}
      </span>
      {!current && (
        <GlassButton
          size="sm"
          variant="ghost"
          disabled={busy || !points.length}
          leadingIcon={<Undo2 size={15} aria-hidden="true" />}
          onClick={() => useLand.getState().removePoint()}
        >
          Undo
        </GlassButton>
      )}
      <GlassButton
        size="sm"
        variant="ghost"
        leadingIcon={<X size={15} aria-hidden="true" />}
        onClick={() => useLand.getState().begin("browse")}
      >
        Cancel
      </GlassButton>
      {current ? (
        <GlassButton
          size="sm"
          variant="primary"
          disabled={busy || !keep.length}
          leadingIcon={<Check size={15} aria-hidden="true" />}
          onClick={() => void request(true)}
        >
          Keep {keep.length || ""} selected
        </GlassButton>
      ) : (
        <GlassButton
          size="sm"
          variant="primary"
          disabled={busy || points.length < 2}
          onClick={() => void request(false)}
        >
          {busy ? "Splitting…" : "Show pieces"}
        </GlassButton>
      )}
    </GlassPanel>
  );
}

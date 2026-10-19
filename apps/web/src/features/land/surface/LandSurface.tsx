import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";

import type { LandArea } from "@twin/contracts";
import { GlassPanel } from "@twin/ui";

import { landUsesOidc, useLandScope } from "@/state/landIdentity";
import { useLand } from "@/state/land";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

import { WriteTokenField } from "../../captures/WriteTokenField";
import { useLandAreas } from "../api";
import { LandBoundaryImport } from "../LandBoundaryImport";
import { LandDraftRecovery } from "../LandDraftRecovery";
import { revealLand } from "../landFraming";
import { WorkspaceIdentity } from "../WorkspaceIdentity";
import { LandAgentInline } from "./LandAgent";
import { LandChip } from "./LandChip";
import { frameBoundary, saveDraft, useMappedFeaturePick } from "./landFlow";
import { LandParcelPicker } from "./LandParcelPicker";
import { LandReview } from "./LandReview";
import { LandStart } from "./LandStart";
import "../land.css";
import "./surface.css";
import { LENS_CARDS } from "./lensCards";
import { useLens } from "./lenses";
import { useScene } from "@/cesium/SceneContext";
import { useQueryClient } from "@tanstack/react-query";

/** Modes in which the map is being drawn on; their controls are the draw bar in the strip. */
const DRAWING = new Set(["draw", "corridor", "pick", "split"]);

function LensHost({ land }: { land: LandArea }) {
  const lens = useLens((s) => s.lens);
  const Card = LENS_CARDS[lens];
  return <Card key={`${land.id}:${lens}`} land={land} />;
}

/**
 * Land in the left dock, as small as the moment allows: a start card with no land, the
 * parcel picker while picking, a review card for a draft, and for saved land the land chip
 * over the current lens's card. Drawing has no card at all; its controls are on the map.
 */
export function LandSurface() {
  const open = useUi((s) => s.activePanel === "land");
  const tokenPrompt = useUi((s) => s.writeTokenPrompt);
  const reducedMotion = useSettings((s) => s.reducedMotion);
  const scope = useLandScope();
  const scene = useScene();
  const queryClient = useQueryClient();
  const mode = useLand((s) => s.mode);
  const active = useLand((s) => s.active);
  const draft = useLand((s) => s.draft);
  const error = useLand((s) => s.error);
  const catalog = useLandAreas(open);
  const fileInput = useRef<HTMLInputElement>(null);
  const surface = useRef<HTMLDivElement>(null);
  const lens = useLens((s) => s.lens);
  useMappedFeaturePick(open);

  // A boundary under review stays in sight: on a phone the review card is a sheet over the
  // lower map, where the outline just drawn usually is. Checked whenever the reviewed boundary
  // changes (a new draft, a combined or split shape, corners adjusted), and framed only when
  // something hides it. Keyed on the boundary itself: naming the draft never moves the camera.
  const reviewBoundary = open && draft && mode === "browse" ? draft.boundary : null;
  useEffect(() => {
    if (!reviewBoundary) return;
    return revealLand(scene, reviewBoundary);
  }, [scene, reviewBoundary]);

  // Opening a land, or another lens, starts at its chip: never wherever the cards before it
  // were scrolled to (a long saved-land list, a lens's details).
  const shownId = open && !draft ? (active?.id ?? null) : null;
  useEffect(() => {
    if (shownId && surface.current) surface.current.scrollTop = 0;
  }, [shownId, lens]);

  const content =
    mode === "candidates" ? (
      <LandParcelPicker />
    ) : DRAWING.has(mode) ? null : draft ? (
      <LandReview />
    ) : active ? (
      <>
        <LandChip land={active} />
        <LandAgentInline />
        <LensHost land={active} />
      </>
    ) : (
      <LandStart fileInput={fileInput} />
    );

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          ref={surface}
          key="land"
          className="hud-item land-surface"
          data-testid="land-panel"
          initial={reducedMotion ? false : { opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          exit={reducedMotion ? { opacity: 0 } : { opacity: 0, x: -12 }}
          transition={{ type: "spring", stiffness: 420, damping: 34, mass: 0.8 }}
          role="region"
          aria-label="Your land"
        >
          {(landUsesOidc || tokenPrompt || error) && (
            <GlassPanel strong className="land-card land-card--notices">
              <WorkspaceIdentity />
              {tokenPrompt && !landUsesOidc && (
                <WriteTokenField
                  hint="This server needs its access token to load and save land."
                  onSaved={() => {
                    useLand.getState().setError(null);
                    void catalog.refetch();
                    if (useLand.getState().draft) void saveDraft(queryClient);
                  }}
                />
              )}
              {error && (
                <p className="land-alert" role="alert">
                  {error}
                </p>
              )}
            </GlassPanel>
          )}
          <LandDraftRecovery key={`recovery:${scope}`} scope={scope} />
          {content}
          <LandBoundaryImport
            key={`import:${scope}`}
            inputRef={fileInput}
            frame={(boundary) => frameBoundary(scene, boundary)}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}

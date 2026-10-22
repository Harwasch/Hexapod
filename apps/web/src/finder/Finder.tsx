import { X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import type { Layer, SourceNotice } from "@twin/contracts";
import { GlassButton, GlassPanel } from "@twin/ui";

import { useLayers } from "@/api/queries";
import { CesiumViewport } from "@/cesium/CesiumViewport";
import { LandBridge } from "@/cesium/LandBridge";
import { LandContextBridge } from "@/cesium/LandContextBridge";
import { useScene } from "@/cesium/SceneContext";
import { DescentForm } from "@/features/descent/DescentForm";
import { LandDrawBar } from "@/features/land/surface/LandDrawBar";
import { LandParcelPicker, type PublicParcelPick } from "@/features/land/surface/LandParcelPicker";
import { beginLand } from "@/features/land/surface/landFlow";
import { CreditSlot } from "@/features/shell/CreditSlot";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useUi } from "@/state/ui";
import { footprintAreaM2 } from "@twin/geo";

import { FINDER_SOURCE, type FinderMessage, type FoundLand } from "./handoff";
import { findPublicParcels, foundParcels } from "./publicParcels";

/**
 * The ground under the search: the catalog's basemap and terrain, as the app shows them
 * (cesium/SceneBridge.tsx registers the whole catalog; the finder needs only these), or the
 * Natural Earth fallback when the catalog has no imagery this browser can load.
 */
function FinderGround() {
  const scene = useScene();
  const catalog = useLayers();
  useEffect(() => {
    if (!scene || catalog.isLoading) return;
    // An API that answers with an error (rather than not at all) leaves no catalog, whatever
    // the hook's type says: the fallback basemap is then the ground.
    const catalogued: Layer[] | undefined = catalog.data;
    const layers = catalogued ?? [];
    const ground = layers.filter(
      (layer) =>
        layer.render.exclusiveGroup === "basemap" || layer.render.exclusiveGroup === "terrain",
    );
    scene.layers.register(ground);
    for (const layer of ground)
      if (layer.defaultVisible) void scene.layers.setVisible(layer.id, true);
    void scene.layers.ensureFallbackBasemap();
  }, [scene, catalog.data, catalog.isLoading]);
  return null;
}

/** Hands a message to the page that opened the finder; standalone, there is no one to tell. */
function tell(message: FinderMessage): void {
  if (window.parent !== window) window.parent.postMessage(message, window.location.origin);
}

/**
 * "Find my land", for the landing page's request form: the app's own Descent search and
 * parcel picker on the app's own globe, with the parcel lookup going to the public route.
 * In California the visitor picks parcels from county records; anywhere, they can draw the
 * outline instead. The choice goes back to the opener (handoff.ts); nothing is saved.
 */
export function Finder({ reducedMotion }: { reducedMotion: boolean }) {
  const scene = useScene();
  const mode = useLand((s) => s.mode);
  const candidates = useLandContext((s) => s.candidates);
  const [found, setFound] = useState<FoundLand | null>(null);
  // The notices each candidate's answer came with: what its source's terms require be shown
  // with it (Kern County's disclaimer), here and wherever the chosen land goes.
  const noticesById = useRef(new Map<string, SourceNotice[]>());
  const [notices, setNotices] = useState<SourceNotice[]>([]);

  // The land panel is what hands the map's taps to the land controller (cesium/LandBridge).
  useEffect(() => {
    useUi.getState().setPanel("land");
  }, []);

  // An outline drawn and closed becomes a draft (landFlow.finishSketch): that is the answer.
  useEffect(
    () =>
      useLand.subscribe((state) => {
        const drawn = state.draft;
        if (drawn?.source.method !== "drawn") return;
        state.cancel();
        const land: FoundLand = {
          method: "drawn",
          boundary: drawn.boundary,
          parcelIds: [],
          label: "Drawn outline",
          areaM2: footprintAreaM2(drawn.boundary),
          attributions: [],
          notices: [],
        };
        setFound(land);
        tell({ source: FINDER_SOURCE, type: "found", land });
      }),
    [],
  );

  const pick = useMemo<PublicParcelPick>(
    () => ({
      find: async (point, signal) => {
        const answer = await findPublicParcels(point, signal);
        const shown = answer.notices ?? [];
        for (const candidate of answer.candidates) noticesById.current.set(candidate.id, shown);
        setNotices(shown);
        return answer;
      },
      onUse: (chosen) => {
        const land = foundParcels(
          chosen,
          (candidate) => noticesById.current.get(candidate.id) ?? [],
        );
        setFound(land);
        tell({ source: FINDER_SOURCE, type: "found", land });
      },
      onCancel: () => beginLand(scene, "browse"),
    }),
    [scene],
  );

  const attributions = [
    ...new Set(candidates.map((candidate) => candidate.source.attribution).filter(Boolean)),
  ];

  return (
    <div className="app finder" data-testid="finder">
      <CesiumViewport />
      <FinderGround />
      <LandBridge />
      <LandContextBridge />
      <div className="finder-dock">
        <GlassPanel strong className="land-card finder-head">
          <div className="land-eyebrow-row">
            <span className="land-lens-label">Find your land</span>
            <GlassButton
              iconOnly
              size="sm"
              variant="ghost"
              aria-label="Close the finder"
              onClick={() => tell({ source: FINDER_SOURCE, type: "close" })}
            >
              <X size={16} aria-hidden="true" />
            </GlassButton>
          </div>
          <DescentForm
            label="Where is your land?"
            reducedMotion={reducedMotion}
            onDescend={() => {
              setFound(null);
              beginLand(scene, "candidates");
            }}
          />
        </GlassPanel>
        {mode === "candidates" && (
          <>
            <LandParcelPicker publicPick={pick} />
            {attributions.length > 0 && (
              <p className="finder-credit" data-testid="finder-attribution">
                Parcel records: {attributions.join("; ")}. County assessor mapping, not a survey.
              </p>
            )}
            {notices.map((notice) => (
              <details
                key={notice.source}
                className="finder-credit finder-notice"
                data-testid="finder-notice"
              >
                <summary>{notice.source}: the disclaimer that comes with these records</summary>
                <p>{notice.text}</p>
              </details>
            ))}
          </>
        )}
        {mode !== "draw" && (
          <GlassPanel className="land-card finder-alt">
            <p className="land-muted">
              No parcel records where your land is? Parcels are available in California.
            </p>
            <div className="land-row">
              <GlassButton size="sm" disabled={!scene} onClick={() => beginLand(scene, "draw")}>
                Draw the outline instead
              </GlassButton>
              {mode !== "candidates" && (
                <GlassButton
                  size="sm"
                  variant="ghost"
                  disabled={!scene}
                  onClick={() => beginLand(scene, "candidates")}
                >
                  Pick parcels
                </GlassButton>
              )}
            </div>
          </GlassPanel>
        )}
        {found && (
          <p className="finder-found" role="status">
            Chosen: {found.label}. You can close the finder.
          </p>
        )}
      </div>
      <div className="finder-strip">
        <LandDrawBar />
      </div>
      <CreditSlot className="finder-credits" />
    </div>
  );
}

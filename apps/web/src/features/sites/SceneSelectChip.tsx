import {
  Brush,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  EyeOff,
  Focus,
  Trash2,
  X,
} from "lucide-react";
import type { CSSProperties, ReactNode, WheelEvent } from "react";

import { GlassButton, GlassTooltip, GlassTooltipProvider } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import type { SceneSelectController } from "@/cesium/sceneSelect/SceneSelectController";
import { paintedDocOf } from "@/cesium/scanView/scanInstances";
import { chipText, PAINT_MIN_IOU, selectionLabel } from "@/lib/sceneSelect";
import { useInstances } from "@/state/instances";
import { useSceneSelect } from "@/state/sceneSelect";

import "./sceneSelectChip.css";

/** Where the chip stands: beside the click, kept on screen; top centre without one. */
function placement(anchor: { x: number; y: number } | null): {
  style: CSSProperties;
  centred: boolean;
} {
  if (!anchor) return { style: { left: "50%", top: 16 }, centred: true };
  const width = typeof window === "undefined" ? 1024 : window.innerWidth;
  const height = typeof window === "undefined" ? 768 : window.innerHeight;
  return {
    style: {
      left: Math.max(8, Math.min(anchor.x + 14, width - 320)),
      top: Math.max(8, Math.min(anchor.y + 14, height - 120)),
    },
    centred: false,
  };
}

/** One of the chip's actions: an icon with its name (and key) in a tooltip, as the panel's. */
function Action({
  label,
  shortcut,
  icon,
  active,
  testId,
  onClick,
}: {
  label: string;
  shortcut?: string;
  icon: ReactNode;
  active?: boolean;
  testId?: string;
  onClick: () => void;
}) {
  return (
    <GlassTooltip content={label} {...(shortcut ? { shortcut } : {})}>
      <GlassButton
        size="sm"
        variant="ghost"
        iconOnly
        active={active ?? false}
        aria-label={shortcut ? `${label} (${shortcut})` : label}
        data-testid={testId}
        onClick={onClick}
      >
        {icon}
      </GlassButton>
    </GlassTooltip>
  );
}

/**
 * The chip by the cursor for what a click or the brush selected in the scene
 * (cesium/sceneSelect): "Tree · 2 of 4", the candidates cycled with its arrows or the wheel
 * over it, and Hide, Show only, Fly to, the brush and Clear. While painting it says how the
 * brush works, what the painted area matched, and offers to keep the area as an object when
 * nothing matched it well.
 */
export function SceneSelectChip({ controller }: { controller: SceneSelectController | null }) {
  const assetId = useSceneSelect((s) => s.assetId);
  const candidates = useSceneSelect((s) => s.candidates);
  const index = useSceneSelect((s) => s.index);
  const anchor = useSceneSelect((s) => s.anchor);
  const mode = useSceneSelect((s) => s.mode);
  const paint = useSceneSelect((s) => s.paint);
  // Re-read the document when the scan's painted objects change.
  useSceneSelect((s) => (assetId ? s.custom[assetId] : undefined));
  // The broad category the objects panel files the selection under (lib/categories.ts).
  const chosen = candidates[index];
  const category = useInstances((s) =>
    assetId && chosen !== undefined ? s.assets[assetId]?.index.categoryOf.get(chosen) : undefined,
  );
  if (!controller) return null;
  const painting = mode === "paint";
  const id = candidates[index];
  if (!painting && id === undefined) return null;
  const doc = assetId ? paintedDocOf(assetId) : undefined;
  const label = id !== undefined ? selectionLabel(doc?.byId.get(id), id, category) : null;
  const { style, centred } = placement(painting && !anchor ? null : anchor);
  const onWheel = (event: WheelEvent<HTMLDivElement>): void => {
    if (event.deltaY !== 0 && candidates.length > 1) controller.cycle(event.deltaY > 0 ? 1 : -1);
  };
  return (
    <GlassTooltipProvider delayDuration={400}>
      <div
        className="scene-select-chip"
        data-testid="scene-select-chip"
        data-centred={centred || undefined}
        role="toolbar"
        aria-label={painting ? "Paint to select" : "Selected object"}
        style={style}
        onWheel={onWheel}
      >
        {label !== null && id !== undefined && (
          <div className="scene-select-chip__row">
            {candidates.length > 1 && (
              <GlassButton
                size="sm"
                variant="ghost"
                iconOnly
                aria-label="Previous candidate ([)"
                title="Previous candidate ([ or Shift+Tab)"
                onClick={() => controller.cycle(-1)}
              >
                <ChevronLeft size={14} aria-hidden />
              </GlassButton>
            )}
            <span
              className="scene-select-chip__label"
              data-testid="scene-select-label"
              data-id={id}
              role="status"
            >
              {chipText(label, index, candidates.length)}
            </span>
            {candidates.length > 1 && (
              <GlassButton
                size="sm"
                variant="ghost"
                iconOnly
                aria-label="Next candidate (])"
                title="Next candidate (] or Tab)"
                onClick={() => controller.cycle(1)}
              >
                <ChevronRight size={14} aria-hidden />
              </GlassButton>
            )}
          </div>
        )}
        {painting && (
          <p className="scene-select-chip__hint" data-testid="scene-select-paint-hint">
            {paint === null
              ? "Paint over an object. Shift adds, Alt removes; Alt+wheel sizes the brush."
              : paint.best === null
                ? `${String(paint.painted)} splats painted; no object under them.`
                : `Best match: ${(paint.iou * 100).toFixed(0)}% overlap · ${String(paint.painted)} splats.`}
          </p>
        )}
        <div className="scene-select-chip__row scene-select-chip__actions">
          {id !== undefined && (
            <>
              <Action
                label="Hide"
                icon={<EyeOff size={14} aria-hidden />}
                onClick={() => controller.hide()}
              />
              <Action
                label="Show only this"
                icon={<Focus size={14} aria-hidden />}
                onClick={() => controller.showOnly()}
              />
              <Action
                label="Fly to"
                icon={<Crosshair size={14} aria-hidden />}
                onClick={() => controller.flyTo()}
              />
            </>
          )}
          {!painting && controller.selectionIsPainted() && (
            <Action
              label="Delete the painted object"
              icon={<Trash2 size={14} aria-hidden />}
              onClick={() => controller.deletePainted()}
            />
          )}
          <span className="scene-select-chip__divider" aria-hidden="true" />
          <Action
            label={painting ? "Stop painting" : "Paint to select"}
            shortcut="B"
            active={painting}
            icon={<Brush size={14} aria-hidden />}
            onClick={() => controller.setPainting(!painting)}
          />
          <Action
            label="Clear the selection"
            shortcut="Esc"
            icon={<X size={14} aria-hidden />}
            onClick={() => {
              controller.setPainting(false);
              controller.clear();
            }}
          />
        </div>
        {painting && paint !== null && paint.painted > 0 && paint.iou < PAINT_MIN_IOU && (
          <GlassButton
            size="sm"
            variant="primary"
            data-testid="scene-select-use-painted"
            onClick={() => controller.usePaintedArea()}
          >
            Use painted area
          </GlassButton>
        )}
      </div>
    </GlassTooltipProvider>
  );
}

/** The chip for the app's scene (its controller is the scene manager's). */
export function SceneSelectOverlay() {
  const scene = useScene();
  return <SceneSelectChip controller={scene?.sceneSelect ?? null} />;
}

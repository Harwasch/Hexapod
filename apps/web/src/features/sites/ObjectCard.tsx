import {
  Brush,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  EyeOff,
  Focus,
  Minus,
  Plus,
  Trash2,
  X,
} from "lucide-react";
import type { WheelEvent } from "react";

import { GlassPanel, GlassSegmentedControl, Kbd } from "@twin/ui";

import { paintedDocOf } from "@/cesium/scanView/scanInstances";
import {
  SCENE_FOCUS_ATTRIBUTE,
  type SceneSelectController,
} from "@/cesium/sceneSelect/SceneSelectController";
import { categoryById } from "@/lib/categories";
import { TOUCH_MEDIA, useMediaQuery } from "@/lib/media";
import { chipText, PAINT_MIN_IOU, selectionLabel } from "@/lib/sceneSelect";
import { useInstances } from "@/state/instances";
import { useSceneSelect, type StrokeMode } from "@/state/sceneSelect";

const STROKE_MODES: readonly { value: StrokeMode; label: string }[] = [
  { value: "replace", label: "New" },
  { value: "add", label: "Add" },
  { value: "subtract", label: "Remove" },
];

/** Each tap of the touch screen's brush buttons sizes the brush by this factor. */
const BRUSH_STEP = 1.25;

/** What the brush says while it is out: how it works, then what the last stroke matched. */
function paintHint(
  paint: { best: number | null; iou: number; painted: number } | null,
  touch: boolean,
): string {
  if (paint === null)
    return touch
      ? "Paint over an object with a finger. New starts again; Add and Remove change the painted area."
      : "Paint over an object. Shift adds, Alt removes; Alt+wheel sizes the brush.";
  if (paint.best === null) return `${String(paint.painted)} splats painted; no object under them.`;
  return `Best match: ${(paint.iou * 100).toFixed(0)}% overlap · ${String(paint.painted)} splats.`;
}

/**
 * The selection card for an object of a scan selected in the scene (cesium/sceneSelect): what
 * a click, the brush or the objects panel chose. Its name and category, the candidates the
 * click offered (◀ 2 of 4 ▶, cycled with the arrows, `[` `]`, Alt and the wheel, the wheel over
 * the arrows, or Tab while the map or this card has focus), Hide, Show only, Fly to, the brush,
 * and for a painted object Delete. While painting it says how the brush works and what the
 * painted area matched, and offers to keep the area as an object when nothing matched it well.
 *
 * It is the HUD's one selection card (`features/mission/SelectionCard`) when the selection is
 * an object, so a machine, a zone and an object never show two cards. On a touch screen
 * (`TOUCH_MEDIA`) there is no Shift, Alt, wheel or Tab: the brush offers New / Add / Remove and
 * a size instead, and the keyboard's hints are left out.
 */
export function ObjectCard({ controller }: { controller: SceneSelectController }) {
  const assetId = useSceneSelect((s) => s.assetId);
  const candidates = useSceneSelect((s) => s.candidates);
  const index = useSceneSelect((s) => s.index);
  const mode = useSceneSelect((s) => s.mode);
  const paint = useSceneSelect((s) => s.paint);
  const brush = useSceneSelect((s) => s.brush);
  const strokeMode = useSceneSelect((s) => s.strokeMode);
  const setStrokeMode = useSceneSelect((s) => s.setStrokeMode);
  const setBrush = useSceneSelect((s) => s.setBrush);
  // Re-read the document when the scan's painted objects change.
  useSceneSelect((s) => (assetId ? s.custom[assetId] : undefined));
  const touch = useMediaQuery(TOUCH_MEDIA);
  const id = candidates[index];
  // The broad category the objects panel files the selection under (lib/categories.ts).
  const category = useInstances((s) =>
    assetId && id !== undefined ? s.assets[assetId]?.index.categoryOf.get(id) : undefined,
  );
  const listed = useInstances((s) => (assetId ? s.assets[assetId]?.instances : undefined));
  // The scan's document as drawn (painted objects included), else the store's table.
  const instance =
    assetId && id !== undefined
      ? (paintedDocOf(assetId)?.byId.get(id) ?? listed?.find((entry) => entry.id === id))
      : undefined;

  const painting = mode === "paint";
  const label = id !== undefined ? selectionLabel(instance, id, category) : "Paint to select";
  const categoryName = category ? categoryById(category).name : null;
  const painted = !painting && id !== undefined && controller.selectionIsPainted();
  const meta = [
    categoryName && categoryName !== label ? categoryName : null,
    painted ? "Painted in this browser" : null,
  ].filter((part): part is string => part !== null);
  const offerPainted = painting && paint !== null && paint.painted > 0 && paint.iou < PAINT_MIN_IOU;
  const onWheel = (event: WheelEvent<HTMLDivElement>): void => {
    if (event.deltaY !== 0 && candidates.length > 1) controller.cycle(event.deltaY > 0 ? 1 : -1);
  };
  const close = (): void => {
    controller.setPainting(false);
    controller.clear();
  };

  return (
    <GlassPanel
      className="mc-card mc-card--object"
      role="region"
      aria-label={
        id !== undefined
          ? `Selected object: ${chipText(label, index, candidates.length)}`
          : "Paint to select"
      }
      data-testid="selection-card"
      data-kind="object"
      data-painting={painting || undefined}
      // Focusable by a click on it (never by Tab): while it has focus Tab cycles candidates.
      tabIndex={-1}
      {...{ [SCENE_FOCUS_ATTRIBUTE]: "" }}
    >
      <div className="mc-card__head">
        <div
          className="mc-card__title"
          data-testid="object-label"
          data-id={id}
          role="status"
          aria-live="polite"
        >
          {label}
        </div>
        <button
          type="button"
          className="mc-close"
          onClick={close}
          aria-label={painting ? "Stop painting and clear the selection" : "Close"}
          aria-keyshortcuts="Escape"
        >
          <X size={13} aria-hidden="true" />
        </button>
      </div>
      {meta.length > 0 && <div className="mc-card__meta">{meta.join(" · ")}</div>}
      {id !== undefined && candidates.length > 1 && (
        <div className="mc-cycle" role="group" aria-label="Candidates" onWheel={onWheel}>
          <button
            type="button"
            className="mc-close"
            aria-label="Previous candidate"
            aria-keyshortcuts="[ Shift+Tab"
            title={touch ? undefined : "Previous candidate ([ or Shift+Tab)"}
            onClick={() => controller.cycle(-1)}
          >
            <ChevronLeft size={14} aria-hidden="true" />
          </button>
          <span className="mc-cycle__count mc-mono" data-testid="object-candidates">
            {index + 1} of {candidates.length}
          </span>
          <button
            type="button"
            className="mc-close"
            aria-label="Next candidate"
            aria-keyshortcuts="] Tab"
            title={touch ? undefined : "Next candidate (] or Tab)"}
            onClick={() => controller.cycle(1)}
          >
            <ChevronRight size={14} aria-hidden="true" />
          </button>
          {!touch && (
            <span className="mc-cycle__hint" aria-hidden="true">
              <Kbd>[</Kbd>
              <Kbd>]</Kbd> or <Kbd>Tab</Kbd>
            </span>
          )}
        </div>
      )}
      {painting && (
        <p className="mc-card__note" data-testid="object-paint-hint">
          {paintHint(paint, touch)}
        </p>
      )}
      {painting && touch && (
        <div className="mc-card__brush" data-testid="object-touch-brush">
          <GlassSegmentedControl
            aria-label="What a stroke does"
            value={strokeMode}
            onValueChange={setStrokeMode}
            options={STROKE_MODES}
          />
          <div className="mc-row" role="group" aria-label="Brush size">
            <button
              type="button"
              className="mc-close"
              aria-label="Smaller brush"
              onClick={() => setBrush(brush / BRUSH_STEP)}
            >
              <Minus size={13} aria-hidden="true" />
            </button>
            <span className="mc-mono mc-card__brush-size">{Math.round(brush)} px</span>
            <button
              type="button"
              className="mc-close"
              aria-label="Larger brush"
              onClick={() => setBrush(brush * BRUSH_STEP)}
            >
              <Plus size={13} aria-hidden="true" />
            </button>
          </div>
        </div>
      )}
      {id !== undefined && (
        <div className="mc-actions">
          <button type="button" className="mc-btn" onClick={() => controller.hide()}>
            <EyeOff size={13} aria-hidden="true" /> Hide
          </button>
          <button type="button" className="mc-btn" onClick={() => controller.showOnly()}>
            <Focus size={13} aria-hidden="true" /> Show only
          </button>
          <button type="button" className="mc-btn" onClick={() => controller.flyTo()}>
            <Crosshair size={13} aria-hidden="true" /> Fly to
          </button>
        </div>
      )}
      <div className="mc-actions mc-actions--tools">
        <button
          type="button"
          className={`mc-btn ${painting ? "is-on" : ""}`}
          aria-pressed={painting}
          aria-keyshortcuts="B"
          onClick={() => controller.setPainting(!painting)}
        >
          <Brush size={13} aria-hidden="true" /> {painting ? "Stop painting" : "Paint to select"}
          {!touch && <Kbd>B</Kbd>}
        </button>
        {painted && (
          <button type="button" className="mc-btn" onClick={() => controller.deletePainted()}>
            <Trash2 size={13} aria-hidden="true" /> Delete painted object
          </button>
        )}
      </div>
      {offerPainted && (
        <div className="mc-actions mc-actions--tools">
          <button
            type="button"
            className="mc-btn mc-btn--accent"
            data-testid="object-use-painted"
            onClick={() => controller.usePaintedArea()}
          >
            Use painted area
          </button>
        </div>
      )}
    </GlassPanel>
  );
}

import { Columns2 } from "lucide-react";
import { lazy, Suspense, useEffect, useRef, useState } from "react";

import { GlassPanel } from "@twin/ui";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { useScene } from "@/cesium/SceneContext";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";

const CompareControls = lazy(() =>
  import("./CompareControls").then((m) => ({ default: m.CompareControls })),
);

/**
 * Compare, as a mode of the Layers panel (CompareControls.tsx), loaded the first time the mode
 * is chosen: the globe's first load carries only this and the divider below.
 */
export function CompareMode() {
  return (
    <ErrorBoundary inline label="Compare">
      <Suspense fallback={null}>
        <CompareControls />
      </Suspense>
    </ErrorBoundary>
  );
}

/**
 * Applies the layer comparison to the scene and draws the divider, which two dates compared
 * (Compare › Dates; the scene's own) share; mounted once, in the shell.
 */
export function CompareSplit() {
  const scene = useScene();
  const active = useUi((s) => s.compareActive);
  const left = useUi((s) => s.compareLeft);
  const right = useUi((s) => s.compareRight);
  const dates = useSites((s) => s.compareDates !== null);

  useEffect(() => {
    if (!scene) return;
    if (active && left && right) {
      void scene.layers
        .setVisible(left, true)
        .then(() => scene.layers.setVisible(right, true))
        .then(() => scene.layers.setSplit(left, right));
    } else {
      scene.layers.setSplit(null, null);
    }
  }, [scene, active, left, right]);

  useEffect(() => () => scene?.layers.setSplit(null, null), [scene]);

  return (active && left && right) || dates ? <SplitSlider /> : null;
}

function SplitSlider() {
  const scene = useScene();
  const [fraction, setFraction] = useState(0.5);
  const dragging = useRef(false);
  useEffect(() => {
    scene?.layers.setSplitPosition(fraction);
  }, [scene, fraction]);
  useEffect(() => {
    const move = (e: PointerEvent) => {
      if (!dragging.current) return;
      setFraction(Math.min(0.98, Math.max(0.02, e.clientX / window.innerWidth)));
    };
    const up = () => {
      dragging.current = false;
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, []);
  return (
    <div
      className="split-slider"
      style={{ left: `${fraction * 100}%` }}
      role="slider"
      aria-label="Comparison split position"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(fraction * 100)}
      tabIndex={0}
      onPointerDown={() => {
        dragging.current = true;
      }}
      onKeyDown={(e) => {
        if (e.key === "ArrowLeft") setFraction((f) => Math.max(0.02, f - 0.02));
        if (e.key === "ArrowRight") setFraction((f) => Math.min(0.98, f + 0.02));
      }}
      data-testid="split-slider"
    >
      <GlassPanel strong pill className="split-slider__handle">
        <Columns2 size={16} aria-hidden="true" />
      </GlassPanel>
    </div>
  );
}

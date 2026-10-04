import { Columns2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { EmptyState, GlassField, GlassPanel, GlassSelect, GlassSwitch, useFieldId } from "@twin/ui";

import { useLayers as useLayerCatalog } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { useLayers } from "@/state/layers";
import { useUi } from "@/state/ui";

const COMPARABLE = new Set([
  "cesium-ion-imagery",
  "xyz",
  "wmts",
  "wms",
  "arcgis-mapserver",
  "cesium-ion-3d-tiles",
  "3d-tiles-url",
]);

/**
 * Compare, as a mode of the Layers panel: pick a layer for each side and swipe between them.
 *
 * The choice lives in the ui store, not here, because the swipe outlives the panel: closing
 * Layers leaves the divider on the map (`CompareSplit`) until the switch is turned off.
 */
export function CompareControls() {
  const active = useUi((s) => s.compareActive);
  const setActive = useUi((s) => s.setCompareActive);
  const left = useUi((s) => s.compareLeft);
  const right = useUi((s) => s.compareRight);
  const setLayers = useUi((s) => s.setCompareLayers);
  const catalog = useLayerCatalog();
  const runtime = useLayers((s) => s.runtime);
  const leftId = useFieldId("compare-left");
  const rightId = useFieldId("compare-right");
  const comparable = (catalog.data ?? []).filter((l) => COMPARABLE.has(l.sourceType));

  if (comparable.length < 2)
    return (
      <EmptyState
        icon={<Columns2 size={26} />}
        title="Nothing to compare yet"
        body="Add at least two imagery or 3D layers to swipe between them."
      />
    );
  return (
    <div className="glass-stack" data-testid="compare-controls">
      <GlassField label="Left" htmlFor={leftId}>
        <GlassSelect id={leftId} value={left} onChange={(e) => setLayers({ left: e.target.value })}>
          <option value="">Choose a layer…</option>
          {comparable.map((l) => (
            <option key={l.id} value={l.id} disabled={l.id === right}>
              {l.name}
              {runtime[l.id]?.visible ? " (on)" : ""}
            </option>
          ))}
        </GlassSelect>
      </GlassField>
      <GlassField label="Right" htmlFor={rightId}>
        <GlassSelect
          id={rightId}
          value={right}
          onChange={(e) => setLayers({ right: e.target.value })}
        >
          <option value="">Choose a layer…</option>
          {comparable.map((l) => (
            <option key={l.id} value={l.id} disabled={l.id === left}>
              {l.name}
            </option>
          ))}
        </GlassSelect>
      </GlassField>
      <div className="setting">
        <span className="setting__label" id="compare-toggle-label">
          Swipe comparison
        </span>
        <GlassSwitch
          checked={active}
          disabled={!left || !right}
          onCheckedChange={setActive}
          aria-labelledby="compare-toggle-label"
        />
      </div>
      <p className="glass-subtle" style={{ fontSize: "var(--text-xs)", margin: 0 }}>
        Drag the divider on the map to compare. It stays when this panel closes.
      </p>
    </div>
  );
}

/** Applies the comparison to the scene and draws its divider; mounted once, in the shell. */
export function CompareSplit() {
  const scene = useScene();
  const active = useUi((s) => s.compareActive);
  const left = useUi((s) => s.compareLeft);
  const right = useUi((s) => s.compareRight);

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

  return active && left && right ? <SplitSlider /> : null;
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

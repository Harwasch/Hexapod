import { Columns2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { EmptyState, GlassField, GlassPanel, GlassSelect, GlassSwitch, useFieldId } from "@twin/ui";

import { useLayers as useLayerCatalog } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { useLayers } from "@/state/layers";
import { useUi } from "@/state/ui";

import { FloatingPanel } from "../shell/FloatingPanel";

/** Swipe comparison between two visible imagery / 3D layers. */
export function ComparePanel() {
  const scene = useScene();
  const open = useUi((s) => s.activePanel === "compare");
  const setPanel = useUi((s) => s.setPanel);
  const active = useUi((s) => s.compareActive);
  const setActive = useUi((s) => s.setCompareActive);
  const catalog = useLayerCatalog();
  const runtime = useLayers((s) => s.runtime);
  const [left, setLeft] = useState<string>("");
  const [right, setRight] = useState<string>("");
  const leftId = useFieldId("compare-left");
  const rightId = useFieldId("compare-right");
  const comparable = (catalog.data ?? []).filter((l) =>
    [
      "cesium-ion-imagery",
      "xyz",
      "wmts",
      "wms",
      "arcgis-mapserver",
      "cesium-ion-3d-tiles",
      "3d-tiles-url",
    ].includes(l.sourceType),
  );

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

  return (
    <>
      <FloatingPanel
        open={open}
        title="Compare"
        onClose={() => setPanel(null)}
        testId="compare-panel"
      >
        <div className="glass-stack">
          {comparable.length < 2 ? (
            <EmptyState
              icon={<Columns2 size={26} />}
              title="Nothing to compare yet"
              body="Add at least two imagery or 3D layers to swipe between them."
            />
          ) : (
            <>
              <GlassField label="Left" htmlFor={leftId}>
                <GlassSelect id={leftId} value={left} onChange={(e) => setLeft(e.target.value)}>
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
                <GlassSelect id={rightId} value={right} onChange={(e) => setRight(e.target.value)}>
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
                Both layers are switched on; exclusive basemaps are compared side by side.
              </p>
            </>
          )}
        </div>
      </FloatingPanel>
      {active && left && right && <SplitSlider />}
    </>
  );
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

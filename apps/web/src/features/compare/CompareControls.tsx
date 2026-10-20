import { Columns2 } from "lucide-react";

import { EmptyState, GlassField, GlassSelect, GlassSwitch, useFieldId } from "@twin/ui";

import { useLayers as useLayerCatalog } from "@/api/queries";
import { recordAction } from "@/state/history";
import { useLayers } from "@/state/layers";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";

import { CompareDates } from "./CompareDates";

/** The active site's dates, side by side (ADR 0017), above the layers. */
function DatesSection() {
  const siteId = useSites((s) => s.activeSiteId);
  return siteId ? <CompareDates siteId={siteId} /> : null;
}

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
 * Compare, as a mode of the Layers panel: two of the active site's dates (Compare › Dates), and
 * a layer for each side, to swipe between. Loaded on demand (`CompareMode`).
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
      <>
        <DatesSection />
        <EmptyState
          icon={<Columns2 size={26} />}
          title="Nothing to compare yet"
          body="Add at least two imagery or 3D layers to swipe between them."
        />
      </>
    );
  return (
    <div className="glass-stack" data-testid="compare-controls">
      <DatesSection />
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
          onCheckedChange={(on) =>
            recordAction(
              `Turn swipe comparison ${on ? "on" : "off"}`,
              () => setActive(on),
              () => setActive(!on),
            )
          }
          aria-labelledby="compare-toggle-label"
        />
      </div>
      <p className="glass-subtle" style={{ fontSize: "var(--text-xs)", margin: 0 }}>
        Drag the divider on the map to compare. It stays when this panel closes.
      </p>
    </div>
  );
}

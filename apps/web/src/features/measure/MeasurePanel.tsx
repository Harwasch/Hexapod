import { ArrowUpDown, MapPin, Mountain, Ruler, Shapes, Trash2, X } from "lucide-react";

import { formatArea, formatLength } from "@twin/geo";
import { Divider, EmptyState, GlassBadge, GlassButton, GlassTooltip, Kbd } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { useMeasurements, type Measurement } from "@/state/measurements";
import { useSettings } from "@/state/settings";
import { useUi, type MeasureMode } from "@/state/ui";

import { FloatingPanel } from "../shell/FloatingPanel";

const MODES: { id: MeasureMode; label: string; hint: string; icon: typeof Ruler }[] = [
  { id: "point", label: "Point", hint: "Click to read coordinates", icon: MapPin },
  { id: "distance", label: "Distance", hint: "Two clicks · straight & ground", icon: Ruler },
  { id: "area", label: "Area", hint: "Click vertices · double-click to finish", icon: Shapes },
  { id: "height", label: "Height", hint: "Base then top", icon: ArrowUpDown },
  { id: "elevation", label: "Elevation", hint: "Terrain height at a point", icon: Mountain },
];

function describe(m: Measurement, units: "metric" | "imperial"): string {
  switch (m.mode) {
    case "distance":
      return `${formatLength(m.distance3dM ?? 0, units)} · ground ${formatLength(m.distance2dM ?? 0, units)}`;
    case "area":
      return formatArea(m.areaM2 ?? 0, units);
    case "height":
      return `Δh ${formatLength(m.heightDeltaM ?? 0, units)}`;
    case "elevation":
      return formatLength(m.elevationM ?? m.points[0]?.height ?? 0, units);
    case "point": {
      const p = m.points[0];
      return p ? `${p.latitude.toFixed(6)}, ${p.longitude.toFixed(6)}` : "";
    }
  }
}

export function MeasurePanel() {
  const scene = useScene();
  const open = useUi((s) => s.activePanel === "measure");
  const setPanel = useUi((s) => s.setPanel);
  const mode = useUi((s) => s.measureMode);
  const setMeasureMode = useUi((s) => s.setMeasureMode);
  const items = useMeasurements((s) => s.items);
  const remove = useMeasurements((s) => s.remove);
  const clear = useMeasurements((s) => s.clear);
  const units = useSettings((s) => s.units);

  const close = () => {
    setMeasureMode(null);
    setPanel(null);
  };

  return (
    <FloatingPanel open={open} title="Measure" onClose={close} testId="measure-panel">
      <div className="glass-stack">
        <div className="tabs" role="group" aria-label="Measurement tool">
          {MODES.map(({ id, label, icon: Icon, hint }) => (
            <GlassTooltip key={id} content={hint}>
              <GlassButton
                size="sm"
                active={mode === id}
                onClick={() => setMeasureMode(mode === id ? null : id)}
                leadingIcon={<Icon size={14} aria-hidden="true" />}
                data-testid={`measure-${id}`}
              >
                {label}
              </GlassButton>
            </GlassTooltip>
          ))}
        </div>
        {mode ? (
          <p
            className="glass-muted"
            style={{ fontSize: "var(--text-xs)", margin: 0 }}
            data-testid="measure-active"
          >
            {MODES.find((m) => m.id === mode)?.hint}. Press <Kbd>Esc</Kbd> to stop.
          </p>
        ) : (
          <p className="glass-subtle" style={{ fontSize: "var(--text-xs)", margin: 0 }}>
            Pick a tool, then click in the world. Results use{" "}
            {units === "metric" ? "metres" : "feet"}; change units in Settings.
          </p>
        )}
        <Divider />
        {items.length === 0 ? (
          <EmptyState
            icon={<Ruler size={26} />}
            title="No measurements"
            body="Measurements stay in the scene until you remove them."
          />
        ) : (
          <ul className="glass-list" aria-label="Measurements">
            {items.map((m) => (
              <li key={m.id} className="card">
                <div className="card__row card__row--between">
                  <GlassBadge tone="accent">{MODES.find((x) => x.id === m.mode)?.label}</GlassBadge>
                  <GlassButton
                    iconOnly
                    size="sm"
                    variant="ghost"
                    aria-label="Remove measurement"
                    onClick={() => {
                      scene?.measurement.remove(m.id);
                      remove(m.id);
                    }}
                  >
                    <Trash2 size={14} aria-hidden="true" />
                  </GlassButton>
                </div>
                <div className="glass-mono" style={{ fontSize: "var(--text-sm)" }}>
                  {describe(m, units)}
                </div>
              </li>
            ))}
          </ul>
        )}
        {items.length > 0 && (
          <GlassButton
            size="sm"
            variant="ghost"
            leadingIcon={<X size={14} aria-hidden="true" />}
            onClick={() => {
              scene?.measurement.clearAll();
              clear();
            }}
          >
            Clear all
          </GlassButton>
        )}
      </div>
    </FloatingPanel>
  );
}

import {
  Circle,
  CircleDot,
  DoorOpen,
  Droplet,
  Fence,
  Pentagon,
  Route,
  SolarPanel,
  Spline,
  Tractor,
  TreePine,
  Warehouse,
  Waves,
  Zap,
  type LucideIcon,
} from "lucide-react";

import { GlassButton } from "@twin/ui";

import { CATEGORIES, PRESETS, SHAPES, type AssetPreset } from "./assetsModel";

const ICONS: Record<string, LucideIcon> = {
  building: Warehouse,
  fence: Fence,
  gate: DoorOpen,
  track: Route,
  well: Droplet,
  pond: Waves,
  pipe: Spline,
  culvert: Circle,
  power: Zap,
  solar: SolarPanel,
  tree: TreePine,
  equipment: Tractor,
  point: CircleDot,
  line: Spline,
  area: Pentagon,
};

const HOW: Record<AssetPreset["kind"], string> = {
  Point: "a spot",
  LineString: "a line",
  Polygon: "an area",
};

/** What is being added decides how it is drawn: a spot, a line or an area. */
export function AssetsChooser({
  onChoose,
  onCancel,
}: {
  onChoose: (preset: AssetPreset) => void;
  onCancel: () => void;
}) {
  const option = (preset: AssetPreset) => {
    const Icon = ICONS[preset.id] ?? CircleDot;
    return (
      <li key={preset.id}>
        <button
          type="button"
          className="assets-kind"
          title={`Draw ${HOW[preset.kind]}`}
          onClick={() => onChoose(preset)}
        >
          <Icon size={18} aria-hidden="true" style={{ color: CATEGORIES[preset.category].color }} />
          <span>{preset.label}</span>
        </button>
      </li>
    );
  };
  return (
    <div className="assets-chooser" role="group" aria-label="What are you adding?">
      <p className="land-lede">What are you adding?</p>
      <ul className="assets-kinds">{PRESETS.map(option)}</ul>
      <p className="land-muted">Something else</p>
      <ul className="assets-kinds assets-kinds--shapes">{SHAPES.map(option)}</ul>
      <div className="land-row">
        <GlassButton variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </GlassButton>
      </div>
    </div>
  );
}

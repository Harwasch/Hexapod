import { Crosshair, Info, Loader2 } from "lucide-react";

import type { Layer } from "@twin/contracts";
import { GlassBadge, GlassButton, GlassSlider, GlassSwitch, GlassTooltip } from "@twin/ui";

import { env } from "@/app/env";
import { useScene } from "@/cesium/SceneContext";
import { formatDate, sourceLabel } from "@/lib/format";
import { useLayers, defaultRuntime } from "@/state/layers";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

export function LayerCard({ layer }: { layer: Layer }) {
  const scene = useScene();
  const runtime = useLayers((s) => s.runtime[layer.id] ?? defaultRuntime);
  const setAboutLayerId = useUi((s) => s.setAboutLayerId);
  const latest = layer.observedAt ?? layer.temporalExtent?.end ?? null;
  const switchId = `layer-switch-${layer.id}`;
  const isWorld = layer.source.type === "google-photorealistic";
  const worldLocked = isWorld && !env.photorealisticEnabled;
  const setSettings = useSettings((s) => s.set);
  const toggle = (checked: boolean) => {
    if (isWorld) setSettings({ world: checked ? "photorealistic" : "open" });
    else void scene?.layers.setVisible(layer.id, checked);
  };

  const flyToExtent = () => {
    const e = layer.spatialExtent;
    if (!e || !scene) return;
    const isWorld = e.west <= -179 && e.east >= 179;
    if (isWorld) scene.camera.flyHome();
    else scene.camera.flyToRectangle(e.west, e.south, e.east, e.north);
  };

  return (
    <li
      className={`card ${runtime.visible ? "card--selected" : ""}`}
      data-testid={`layer-card-${layer.slug}`}
    >
      <div className="card__row">
        <GlassSwitch
          id={switchId}
          checked={runtime.visible}
          disabled={runtime.loadState === "loading" || worldLocked}
          onCheckedChange={toggle}
          aria-label={`Show ${layer.name}`}
        />
        <label htmlFor={switchId} className="card__title">
          {layer.name}
        </label>
        {runtime.loadState === "loading" && (
          <Loader2
            size={14}
            className="glass-muted"
            aria-label="Loading"
            style={{ animation: "glass-spin 0.8s linear infinite" }}
          />
        )}
        {runtime.loadState === "error" && <GlassBadge tone="danger">Error</GlassBadge>}
        <GlassTooltip content="About this layer">
          <GlassButton
            iconOnly
            size="sm"
            variant="ghost"
            aria-label={`About ${layer.name}`}
            onClick={() => setAboutLayerId(layer.id)}
          >
            <Info size={14} aria-hidden="true" />
          </GlassButton>
        </GlassTooltip>
        <GlassTooltip content="Fly to extent">
          <GlassButton
            iconOnly
            size="sm"
            variant="ghost"
            aria-label={`Fly to ${layer.name} extent`}
            onClick={flyToExtent}
            disabled={!layer.spatialExtent}
          >
            <Crosshair size={14} aria-hidden="true" />
          </GlassButton>
        </GlassTooltip>
      </div>
      {layer.description && <p className="card__description">{layer.description}</p>}
      <div className="card__meta">
        <span title="Source">{sourceLabel(layer.source)}</span>
        {latest && <span title="Latest date">{formatDate(latest)}</span>}
        {layer.resolution && <span title="Resolution">{layer.resolution}</span>}
        {layer.coverage && <span title="Coverage">{layer.coverage}</span>}
        {layer.license && <span title="License">{layer.license.spdxId ?? layer.license.name}</span>}
      </div>
      {runtime.error && <p className="card__error">{runtime.error}</p>}
      {worldLocked && (
        <p className="glass-subtle" style={{ margin: 0, fontSize: "var(--text-xs)" }}>
          Switched off by VITE_ENABLE_PHOTOREALISTIC=false; needs an ion token with access.
        </p>
      )}
      {runtime.visible && layer.sourceType !== "cesium-ion-terrain" && (
        <div className="card__controls">
          <span className="glass-subtle" style={{ fontSize: "var(--text-xs)" }}>
            Opacity
          </span>
          <GlassSlider
            aria-label={`${layer.name} opacity`}
            value={runtime.opacity}
            min={0}
            max={1}
            step={0.05}
            onValueChange={(v) => scene?.layers.setOpacity(layer.id, v)}
          />
          <span
            className="glass-mono glass-subtle"
            style={{ fontSize: "var(--text-xs)", width: "2.2rem", textAlign: "right" }}
          >
            {Math.round(runtime.opacity * 100)}%
          </span>
        </div>
      )}
    </li>
  );
}

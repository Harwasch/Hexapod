import { Divider, GlassSegmentedControl, GlassSheet, GlassSlider, GlassSwitch } from "@twin/ui";

import { env } from "@/app/env";
import { QUALITY_SSE, useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

function Row({
  label,
  hint,
  control,
  id,
}: {
  label: string;
  hint?: string;
  control: React.ReactNode;
  id: string;
}) {
  return (
    <div className="setting">
      <div>
        <div className="setting__label" id={id}>
          {label}
        </div>
        {hint && <div className="setting__hint">{hint}</div>}
      </div>
      {control}
    </div>
  );
}

export function SettingsSheet() {
  const open = useUi((s) => s.settingsOpen);
  const setOpen = useUi((s) => s.setSettingsOpen);
  const s = useSettings();
  const bounds = QUALITY_SSE[s.quality];
  const manual = s.manualScreenSpaceError;
  return (
    <GlassSheet
      open={open}
      onOpenChange={setOpen}
      title="Settings"
      description="Appearance, units, rendering and world."
      side="right"
      testId="settings-sheet"
    >
      <section aria-labelledby="settings-appearance">
        <p className="glass-eyebrow" id="settings-appearance">
          Appearance
        </p>
        <Row
          id="theme-label"
          label="Theme"
          control={
            <GlassSegmentedControl
              aria-label="Theme"
              value={s.theme}
              onValueChange={(theme) => s.set({ theme })}
              options={[
                { value: "auto", label: "Mission dark" },
                { value: "light", label: "Light Glass" },
                { value: "dark", label: "Dark Glass" },
              ]}
            />
          }
        />
        <Row
          id="motion-label"
          label="Reduced motion"
          hint="Also follows your system preference"
          control={
            <GlassSwitch
              aria-labelledby="motion-label"
              checked={s.reducedMotion}
              onCheckedChange={(reducedMotion) => s.set({ reducedMotion })}
            />
          }
        />
        <Row
          id="contrast-label"
          label="High contrast"
          control={
            <GlassSwitch
              aria-labelledby="contrast-label"
              checked={s.highContrast}
              onCheckedChange={(highContrast) => s.set({ highContrast })}
            />
          }
        />
        <Row
          id="transparency-label"
          label="Reduce transparency"
          control={
            <GlassSwitch
              aria-labelledby="transparency-label"
              checked={s.reducedTransparency}
              onCheckedChange={(reducedTransparency) => s.set({ reducedTransparency })}
            />
          }
        />
      </section>
      <Divider />
      <section aria-labelledby="settings-units">
        <p className="glass-eyebrow" id="settings-units">
          Units
        </p>
        <Row
          id="units-label"
          label="Measurement units"
          control={
            <GlassSegmentedControl
              aria-label="Units"
              value={s.units}
              onValueChange={(units) => s.set({ units })}
              options={[
                { value: "metric", label: "Metric" },
                { value: "imperial", label: "Imperial" },
              ]}
            />
          }
        />
      </section>
      <Divider />
      <section aria-labelledby="settings-rendering">
        <p className="glass-eyebrow" id="settings-rendering">
          Rendering
        </p>
        <Row
          id="quality-label"
          label="Quality preset"
          hint={`Screen-space error ${bounds.base} px (adaptive ${bounds.min}–${bounds.max})`}
          control={
            <GlassSegmentedControl
              aria-label="Quality preset"
              value={s.quality}
              onValueChange={(quality) => s.set({ quality, manualScreenSpaceError: null })}
              options={[
                { value: "performance", label: "Performance" },
                { value: "balanced", label: "Balanced" },
                { value: "ultra", label: "Ultra" },
              ]}
            />
          }
        />
        <Row
          id="adaptive-label"
          label="Adaptive quality"
          hint="Responds to frame rate, motion and loading"
          control={
            <GlassSwitch
              aria-labelledby="adaptive-label"
              checked={s.adaptiveQuality}
              onCheckedChange={(adaptiveQuality) => s.set({ adaptiveQuality })}
            />
          }
        />
        <details>
          <summary className="setting__label" style={{ cursor: "pointer", padding: "0.5rem 0" }}>
            Advanced
          </summary>
          <Row
            id="manual-sse-label"
            label="Manual screen-space error"
            hint={
              manual === null
                ? "Off — preset controls quality"
                : `${manual} px (lower is sharper, heavier)`
            }
            control={
              <GlassSwitch
                aria-labelledby="manual-sse-label"
                checked={manual !== null}
                onCheckedChange={(on) => s.set({ manualScreenSpaceError: on ? bounds.base : null })}
              />
            }
          />
          {manual !== null && (
            <GlassSlider
              aria-label="Screen-space error"
              value={manual}
              min={1}
              max={64}
              step={1}
              onValueChange={(v) => s.set({ manualScreenSpaceError: v })}
            />
          )}
          <Row
            id="explore-speed-label"
            label="Explore speed"
            hint={`${s.exploreSpeed.toFixed(1)} m/s`}
            control={
              <div style={{ width: "9rem" }}>
                <GlassSlider
                  aria-label="Explore speed"
                  value={s.exploreSpeed}
                  min={0.5}
                  max={50}
                  step={0.5}
                  onValueChange={(exploreSpeed) => s.set({ exploreSpeed })}
                />
              </div>
            }
          />
        </details>
      </section>
      <Divider />
      <section aria-labelledby="settings-world">
        <p className="glass-eyebrow" id="settings-world">
          World
        </p>
        <Row
          id="world-label"
          label="Global representation"
          hint={
            env.photorealisticEnabled
              ? "Photorealistic tiles are visual context only, not analytical data"
              : "Set VITE_ENABLE_PHOTOREALISTIC=true to enable Google Photorealistic 3D Tiles"
          }
          control={
            <GlassSegmentedControl
              aria-label="Global representation"
              value={s.world}
              onValueChange={(world) => s.set({ world })}
              options={[
                { value: "open", label: "Open world" },
                {
                  value: "photorealistic",
                  label: "Photorealistic",
                  disabled: !env.photorealisticEnabled,
                },
              ]}
            />
          }
        />
      </section>
    </GlassSheet>
  );
}

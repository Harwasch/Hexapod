import {
  Divider,
  GlassButton,
  GlassInput,
  GlassSegmentedControl,
  GlassSheet,
  GlassSlider,
  GlassSwitch,
} from "@twin/ui";

import { env } from "@/app/env";
import { DEFAULT_WIND_STRENGTH, useLiving } from "@/state/living";
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

/**
 * The wind control.
 *
 * Three things it is careful never to say. **Strength is not a speed** — it is a dimensionless
 * 0..1 scale (`WindStrength` in `@twin/world`), and printing "m/s" would claim the sway
 * amplitude had been validated against a real tree at that speed, which it has not. And
 * **bearing is downwind**, the direction the wind blows *towards*, which is the opposite of the
 * meteorological convention; an inverted bearing is invisible in a screenshot, so the label says
 * which one it is. And **the geometry is not described as measured**: every rigged site today is
 * a procedural fixture, so this section says only what is true of all of them — that the motion
 * is modelled and the geometry under it is never written to. Which geometry is a measured
 * capture, and at what resolution, is the Inspector's job, because only the Inspector has that
 * site's catalog metadata in front of it.
 *
 * A fourth thing it used to say and no longer does: the sort-staleness figure in splat radii.
 * `sortStaleness` divides by `REFERENCE_GAUSSIAN_SCALE_M`, a *reference* median taken from a
 * real drone capture — not the median of whatever tileset is on screen, which nothing decodes
 * today. On the only site that can currently move, the synthetic fixture, the real median is
 * 10.7 cm and the printed figure was five times too alarming. "Radii" reads as a measurement of
 * the thing in front of you, so quoting it here made a modelled ratio look surveyed. The
 * displacement in metres is a property of this rig at this wind and is kept; the ratio moved to
 * the developer panel, where it is printed with the yardstick it actually used.
 */
function WindSection() {
  const wind = useLiving((s) => s.wind);
  const status = useLiving((s) => s.status);
  const setWind = useLiving((s) => s.setWind);
  const reducedMotion = useSettings((s) => s.reducedMotion);
  const site = status.sites.find((s) => s.phase === "ready");
  const on = wind.strength > 0;
  return (
    <section aria-labelledby="settings-living">
      <p className="glass-eyebrow" id="settings-living">
        Living survey
      </p>
      <Row
        id="wind-label"
        label="Simulated wind"
        hint={
          reducedMotion
            ? "Held calm by reduced motion"
            : "Modelled motion; the data underneath is never altered"
        }
        control={
          <GlassSwitch
            aria-labelledby="wind-label"
            checked={on}
            disabled={reducedMotion}
            onCheckedChange={(next) => setWind({ strength: next ? DEFAULT_WIND_STRENGTH : 0 })}
          />
        }
      />
      {on && !reducedMotion && (
        <>
          <Row
            id="wind-strength-label"
            label="Strength"
            hint={
              site
                ? `${wind.strength.toFixed(2)} of 1 — an arbitrary scale, not a wind speed. Moves ${site.siteSlug} by up to ${site.maxDisplacementM.toFixed(2)} m`
                : `${wind.strength.toFixed(2)} of 1 — an arbitrary scale, not a wind speed`
            }
            control={
              <div style={{ width: "9rem" }}>
                <GlassSlider
                  aria-label="Wind strength"
                  value={wind.strength}
                  min={0.01}
                  max={1}
                  step={0.01}
                  onValueChange={(strength) => setWind({ strength })}
                />
              </div>
            }
          />
          <Row
            id="wind-bearing-label"
            label="Blowing towards"
            hint={`${Math.round(wind.bearingDeg)}° from north (downwind, not the direction it comes from)`}
            control={
              <div style={{ width: "9rem" }}>
                <GlassSlider
                  aria-label="Wind bearing"
                  value={wind.bearingDeg}
                  min={0}
                  max={359}
                  step={1}
                  onValueChange={(bearingDeg) => setWind({ bearingDeg })}
                />
              </div>
            }
          />
        </>
      )}
    </section>
  );
}

/**
 * Shown only once a write token is in play — either stored, or asked for by a 401.
 *
 * Most deployments configure no `API_WRITE_TOKEN` at all, and writes are then open; a
 * token field on screen in that case would describe a step that does not exist.
 */
function WriteTokenSection() {
  const writeToken = useSettings((s) => s.writeToken);
  const setSettings = useSettings((s) => s.set);
  const prompted = useUi((s) => s.writeTokenPrompt);
  if (!writeToken && !prompted) return null;
  return (
    <>
      <section aria-labelledby="settings-api">
        <p className="glass-eyebrow" id="settings-api">
          API access
        </p>
        <Row
          id="write-token-label"
          label="Write token"
          hint="Sent as a bearer token on writes, and kept in this browser — which any script on this origin can read. Fine for a single-user prototype, not for real accounts."
          control={
            <div className="glass-row">
              <GlassInput
                type="password"
                aria-labelledby="write-token-label"
                value={writeToken}
                autoComplete="off"
                onChange={(event) => setSettings({ writeToken: event.target.value })}
                data-testid="settings-write-token"
              />
              <GlassButton
                size="sm"
                variant="ghost"
                onClick={() => setSettings({ writeToken: "" })}
                disabled={!writeToken}
              >
                Forget
              </GlassButton>
            </div>
          }
        />
      </section>
      <Divider />
    </>
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
              : "Google Photorealistic 3D Tiles are switched off by VITE_ENABLE_PHOTOREALISTIC=false"
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
      <Divider />
      <WriteTokenSection />
      <WindSection />
    </GlassSheet>
  );
}

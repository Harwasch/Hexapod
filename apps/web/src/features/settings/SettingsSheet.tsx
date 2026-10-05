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
import { MotionRendererNote } from "@/features/sites/MotionRendererNote";
import {
  CPU_REASON_TEXT,
  DEFAULT_WIND_STRENGTH,
  useLiving,
  type LivingSiteStatus,
} from "@/state/living";
import { rendererReadout, useScanRendererStatus } from "@/lib/rendererReadout";
import {
  QUALITY_SSE,
  useRendererOverride,
  useSettings,
  useSplatRenderer,
  type SplatRenderer,
} from "@/state/settings";
import { useSkinPoke } from "@/state/skinPoke";
import { useUi } from "@/state/ui";
import { useViewer } from "@/state/viewer";

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

/** One site's path, and why when it is the CPU. */
function pathLabel(site: LivingSiteStatus): string {
  if (site.motionPath === "gpu") return "GPU";
  return site.cpuReason === null ? "CPU" : `CPU (${CPU_REASON_TEXT[site.cpuReason]})`;
}

/**
 * Which path each rigged site animates on, and what that costs the main thread per frame —
 * the numbers to compare when "Motion on GPU" is flipped. Beside them, the scene's own frame
 * time: a path that costs 70 ms a frame is a frame rate of 14 whatever the GPU does.
 *
 * Rolling means over animated frames only (`MOTION_COST_WINDOW`), so the figure appears once
 * the wind has blown for a moment and starts over when the path is switched. The shader's own
 * time is spent on the GPU and is in the scene frame time, not in the motion figure.
 */
function MotionReadout({ sites }: { sites: readonly LivingSiteStatus[] }) {
  const perf = useViewer((s) => s.performance);
  if (sites.length === 0) return null;
  return (
    <div className="setting__hint" data-testid="living-motion-readout" style={{ paddingBottom: 8 }}>
      {sites.map((site) => (
        <div key={site.assetId} data-testid="living-motion-site">
          {`${site.siteSlug}: ${pathLabel(site)}`}
          {site.phase !== "ready"
            ? ` · ${site.phase}`
            : site.motionMs === null
              ? " · no animated frames yet"
              : ` · motion ${site.motionMs.toFixed(2)} ms/frame (write ${(site.applyMs ?? 0).toFixed(2)})`}
        </div>
      ))}
      <div data-testid="living-frame-time">
        {perf.rendering
          ? `Scene frame ${perf.frameTimeMs.toFixed(1)} ms (${Math.round(perf.fps)} fps)`
          : "Scene idle"}
      </div>
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
  const gpuMotion = useSettings((s) => s.livingGpuMotion);
  const setSettings = useSettings((s) => s.set);
  const poke = useSkinPoke((s) => s.active);
  const setPoke = useSkinPoke((s) => s.setActive);
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
      <Row
        id="gpu-motion-label"
        label="Motion on GPU"
        hint={
          env.splatGpuMotion
            ? "Moves the splats in the vertex shader; off moves them on the CPU, to compare"
            : "This build keeps motion on the CPU (VITE_SPLAT_GPU_MOTION)"
        }
        control={
          <GlassSwitch
            aria-labelledby="gpu-motion-label"
            checked={env.splatGpuMotion && gpuMotion}
            disabled={!env.splatGpuMotion}
            onCheckedChange={(livingGpuMotion) => setSettings({ livingGpuMotion })}
          />
        }
      />
      <Row
        id="poke-label"
        label="Poke objects"
        hint="Drag a plant or object that moves by its skin; let go to see it ring (K)"
        control={
          <GlassSwitch aria-labelledby="poke-label" checked={poke} onCheckedChange={setPoke} />
        }
      />
      <MotionReadout sites={status.sites} />
      {on && <MotionRendererNote />}
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
          hint="Needed to save uploads when the server asks for it. Stored in this browser only."
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
  return (
    <GlassSheet
      open={open}
      onOpenChange={setOpen}
      title="Settings"
      description="Appearance, units, rendering and world; developer options under Advanced."
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
          hint="Sharper detail costs frame rate. Fine-tune under Advanced."
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
      <Divider />
      <AdvancedSection />
    </GlassSheet>
  );
}

/** Who draws a splat, side by side for comparison (settings `splatRenderer`). */
const RENDERERS: { value: SplatRenderer; label: string; ariaLabel: string }[] = [
  { value: "playcanvas", label: "PlayCanvas", ariaLabel: "Draw splats with PlayCanvas" },
  {
    value: "playcanvas-webgpu",
    label: "WebGPU",
    ariaLabel: "Draw splats with PlayCanvas on WebGPU (beta)",
  },
  { value: "spark", label: "Spark", ariaLabel: "Draw splats with Spark" },
  { value: "cesium", label: "Cesium", ariaLabel: "Draw splats with CesiumJS" },
];

/**
 * Under the renderer switch: what draws now and how fast it drew while the camera last moved
 * (lib/rendererReadout.ts) -- the developer readouts' line, here too because a phone hides the
 * bottom bar and a phone is where WebGPU is to be judged (docs/WEBGPU_TRIAL.md) -- and a word
 * when the page address chose the renderer for this visit (`?renderer=`).
 */
function RendererNow({ chosen, fromAddress }: { chosen: SplatRenderer; fromAddress: boolean }) {
  const scan = useScanRendererStatus(chosen !== "cesium");
  const now = rendererReadout(chosen, scan);
  const parts = [
    fromAddress ? "Chosen by the page address for this visit" : null,
    scan?.active ? `Now: ${now.label}` : null,
    now.meter,
    now.notice,
  ].filter((part): part is string => part !== null);
  if (parts.length === 0) return null;
  return (
    <div className="setting__hint" data-testid="splat-renderer-now">
      {parts.join(" · ")}
    </div>
  );
}

/**
 * What a developer or deployer tunes and an operator never needs: which engine draws splats,
 * a fixed screen-space error, the explore speed, and the camera and renderer readouts.
 * Collapsed until asked for.
 */
function AdvancedSection() {
  const s = useSettings();
  const splatRenderer = useSplatRenderer();
  const fromAddress = useRendererOverride((o) => o.renderer !== null);
  const bounds = QUALITY_SSE[s.quality];
  const manual = s.manualScreenSpaceError;
  return (
    <section aria-labelledby="settings-advanced">
      <details className="disclosure settings-advanced" data-testid="settings-advanced">
        <summary className="disclosure__summary glass-eyebrow" id="settings-advanced">
          Advanced
        </summary>
        <Row
          id="dev-readouts-label"
          label="Show developer readouts"
          hint="Altitude, scale, metres per pixel and the renderer in the bottom bar, and setup notes for whoever deployed this"
          control={
            <GlassSwitch
              aria-labelledby="dev-readouts-label"
              checked={s.devReadouts}
              onCheckedChange={(devReadouts) => s.set({ devReadouts })}
            />
          }
        />
        <Row
          id="splat-renderer-label"
          label="Splat renderer"
          hint="Who draws Gaussian splats, for comparison; the globe, navigation and tools stay CesiumJS. WebGPU is PlayCanvas on WebGPU, in beta: WebGL2 where a device has none, and WebGL2 for scans with objects or motion"
          control={
            <GlassSegmentedControl
              aria-label="Splat renderer"
              data-testid="splat-renderer"
              block
              value={splatRenderer}
              onValueChange={(next) => s.set({ splatRenderer: next })}
              options={RENDERERS}
            />
          }
        />
        <RendererNow chosen={splatRenderer} fromAddress={fromAddress} />
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
        {/* The developer console: off the rail, here and on `D` (builds with dev tools only). */}
        {env.devToolsEnabled && (
          <Row
            id="dev-tools-label"
            label="Developer tools"
            hint="The performance, tiles and Living Survey console, in the right dock. Also D."
            control={
              <GlassSwitch
                aria-labelledby="dev-tools-label"
                checked={s.devToolsOpen}
                onCheckedChange={(devToolsOpen) => s.set({ devToolsOpen })}
              />
            }
          />
        )}
      </details>
    </section>
  );
}
